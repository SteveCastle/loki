// Vision-tower helpers (Qwen3-VL ViT): hi/lo bf16 split, 2D rope, ~f32-accurate attention, position embeddings.
// Ported from loki-reshoot/kernels/nvfp4.cu.
#include "common.cuh"

extern "C" {
// Split an f32 matrix into bf16 hi + bf16 lo parts (x ~= hi + lo to ~16 significant bits).
__global__ void k_split_hilo_bf16(const float* __restrict__ x, bf16* __restrict__ hi, bf16* __restrict__ lo, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    bf16 h = __float2bfloat16(v);
    hi[i] = h;
    lo[i] = __float2bfloat16(v - __bfloat162float(h));
}

// Vision rope (split-half over the full head dim D) in place on f32 qkv [M, 3, H, D]; rope [M, D/2, 2].
__global__ void k_rope_vision_f32(float* __restrict__ qkv, int M, int H, int D, const float* __restrict__ rope) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int64_t per_tok = (int64_t)2 * H * half;
    if (idx >= (int64_t)M * per_tok) return;
    int64_t m = idx / per_tok;
    int rem = idx % per_tok;
    int which = rem / (H * half);
    rem = rem % (H * half);
    int h = rem / half, i = rem % half;
    float* p = qkv + ((m * 3 + which) * H + h) * D;
    float c = rope[(m * half + i) * 2], s = rope[(m * half + i) * 2 + 1];
    float a = p[i], b = p[i + half];
    p[i] = a * c - b * s;
    p[i + half] = b * c + a * s;
}
}  // extern "C"

// ---------------------------------------------------------------------------------------------
// f32 SIMT flash attention (non-causal) for the vision tower: qkv f32 [n, 3, H, D] packed, out f32 [n, H, D].
// Block = 64 queries x 1 head, 4 threads per query (each owns D/4 dims), K/V tiles of 64 keys in smem.
namespace {
template <int D>
__device__ void attn_f32_kernel(const float* __restrict__ qkv, float* __restrict__ out, int n, int H, float scale_log2) {
    constexpr int DP = D / 4;  // dims per thread
    constexpr int TK = 64;
    __shared__ float Ks[TK * D];
    __shared__ float Vs[TK * D];
    const int tid = threadIdx.x;
    const int qi = blockIdx.x * 64 + (tid >> 2);
    const int part = tid & 3;
    const int h = blockIdx.y;
    const int64_t ts = (int64_t)3 * H * D;
    float q[DP], o[DP];
    const bool qok = qi < n;
#pragma unroll
    for (int d = 0; d < DP; ++d) {
        q[d] = qok ? qkv[(int64_t)qi * ts + h * D + part * DP + d] * scale_log2 : 0.f;
        o[d] = 0.f;
    }
    float mrun = -INFINITY, lrun = 0.f;
    for (int k0 = 0; k0 < n; k0 += TK) {
        __syncthreads();
        for (int i = tid; i < TK * D; i += blockDim.x) {
            int j = i / D, d = i % D;
            int key = k0 + j;
            bool ok = key < n;
            Ks[i] = ok ? qkv[(int64_t)key * ts + (H + h) * D + d] : 0.f;
            Vs[i] = ok ? qkv[(int64_t)key * ts + (2 * H + h) * D + d] : 0.f;
        }
        __syncthreads();
        const int kn = min(TK, n - k0);
        for (int j0 = 0; j0 < kn; j0 += 8) {
            float s[8];
            float cmax = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* kr = Ks + (j0 + jj) * D + part * DP;
                float acc = 0.f;
#pragma unroll
                for (int d = 0; d < DP; ++d) acc = fmaf(q[d], kr[d], acc);
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                acc += __shfl_xor_sync(0xffffffff, acc, 2);
                s[jj] = (j0 + jj < kn) ? acc : -INFINITY;
                cmax = fmaxf(cmax, s[jj]);
            }
            float mnew = fmaxf(mrun, cmax);
            float corr = exp2f(mrun - mnew);
            lrun *= corr;
#pragma unroll
            for (int d = 0; d < DP; ++d) o[d] *= corr;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                float p = exp2f(s[jj] - mnew);
                lrun += p;
                const float* vr = Vs + (j0 + jj) * D + part * DP;
#pragma unroll
                for (int d = 0; d < DP; ++d) o[d] = fmaf(p, vr[d], o[d]);
            }
            mrun = mnew;
        }
    }
    if (qok) {
        float inv = 1.f / lrun;
#pragma unroll
        for (int d = 0; d < DP; ++d) out[((int64_t)qi * H + h) * D + part * DP + d] = o[d] * inv;
    }
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) k_attn_f32_d72(const float* qkv, float* out, int n, int H, float scale_log2) {
    attn_f32_kernel<72>(qkv, out, n, H, scale_log2);
}

// ---------------------------------------------------------------------------------------------
// Vision attention with ~f32 accuracy on tensor cores (non-causal, head dim 72 padded to 80).
// Q/K/V come as bf16 hi + lo parts of the f32 values (qkv packed [n, 3, H, 72] each); both products use
// three bf16 MMAs:  S = Qh.Kh + Qh.Kl + Ql.Kh  and  O += Ph.Vh + Ph.Vl + Pl.Vh  (dropped terms ~2^-16).
// Block: 128 queries (8 warps x 16 rows) x 1 head; K/V tiles of 64 keys double-buffered; f32 out [n, H, 72].
namespace {
constexpr int VQ = 128, VKV = 64, VROWB = 176, VCH = 10, VKS = 5, VNT = 10, VD = 72;
constexpr int VSTG = 4 * VKV * VROWB;  // Kh, Kl, Vh, Vl
DEVI int voff(int row, int ch) { return row * VROWB + (ch << 4); }
DEVI uint32_t pack_lo_bf16x2(float a, float b) {
    float ra = a - __bfloat162float(__float2bfloat16(a));
    float rb = b - __bfloat162float(__float2bfloat16(b));
    return pack_bf16x2(ra, rb);
}
}  // namespace

extern "C" __global__ void __launch_bounds__(256) k_attn_split_d72(const bf16* __restrict__ qh_g, const bf16* __restrict__ ql_g, float* __restrict__ out,
                                                                 int n, int H, float scale_log2) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* st[2] = {smem, smem + VSTG};
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * VQ, h = blockIdx.y;
    const int64_t ts = (int64_t)3 * H * VD;
    constexpr int DCH = VD * 2 / 16;  // 9 real chunks

    // ---- Q (hi, lo) into the stage-1 area
    for (int c = tid; c < VQ * VCH; c += 256) {
        int row = c / VCH, ch = c % VCH;
        int i = q0 + row;
        bool pred = i < n;
        uint8_t* dh = st[1] + voff(row, ch);
        uint8_t* dl = st[1] + VQ * VROWB + voff(row, ch);
        if (ch < DCH) {
            int64_t o = (int64_t)(pred ? i : 0) * ts + h * VD + ch * 8;
            cp_async_16(smem_u32(dh), qh_g + o, pred);
            cp_async_16(smem_u32(dl), ql_g + o, pred);
        } else {
            *reinterpret_cast<uint4*>(dh) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(dl) = make_uint4(0, 0, 0, 0);
        }
    }
    auto load_kv = [&](int tile, int s) {
        const int j0 = tile * VKV;
        uint8_t* b = st[s];
        for (int c = tid; c < VKV * VCH; c += 256) {
            int row = c / VCH, ch = c % VCH;
            int j = j0 + row;
            bool pred = j < n;
            uint8_t* d0 = b + voff(row, ch);
            if (ch < DCH) {
                int64_t ko = (int64_t)(pred ? j : 0) * ts + (H + h) * VD + ch * 8;
                int64_t vo = ko + (int64_t)H * VD;
                cp_async_16(smem_u32(d0), qh_g + ko, pred);
                cp_async_16(smem_u32(d0 + VKV * VROWB), ql_g + ko, pred);
                cp_async_16(smem_u32(d0 + 2 * VKV * VROWB), qh_g + vo, pred);
                cp_async_16(smem_u32(d0 + 3 * VKV * VROWB), ql_g + vo, pred);
            } else {
#pragma unroll
                for (int q = 0; q < 4; ++q) *reinterpret_cast<uint4*>(d0 + q * VKV * VROWB) = make_uint4(0, 0, 0, 0);
            }
        }
    };
    load_kv(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int lidx = lane >> 3, lrow = lane & 7;
    uint32_t qfh[VKS][4], qfl[VKS][4];
#pragma unroll
    for (int s = 0; s < VKS; ++s) {
        int row = warp * 16 + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qfh[s][0], qfh[s][1], qfh[s][2], qfh[s][3], smem_u32(st[1] + voff(row, ch)));
        ldmatrix_x4(qfl[s][0], qfl[s][1], qfl[s][2], qfl[s][3], smem_u32(st[1] + VQ * VROWB + voff(row, ch)));
    }
    __syncthreads();

    const int g = lane >> 2, t4 = lane & 3;
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    float o_acc[VNT][4];
#pragma unroll
    for (int j = 0; j < VNT; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    const int ntiles = (n + VKV - 1) / VKV;

    for (int t = 0; t < ntiles; ++t) {
        const int s_ = t & 1;
        if (t + 1 < ntiles) load_kv(t + 1, s_ ^ 1);
        cp_async_commit();
        const uint8_t* kh = st[s_];
        const uint8_t* kl = kh + VKV * VROWB;
        const uint8_t* vh = kh + 2 * VKV * VROWB;
        const uint8_t* vl = kh + 3 * VKV * VROWB;
        const int j0 = t * VKV;

        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < VKS; ++ks) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                int row = (j + (lidx >> 1)) * 8 + lrow;
                int ch = 2 * ks + (lidx & 1);
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                ldmatrix_x4(a0, a1, a2, a3, smem_u32(kh + voff(row, ch)));
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(kl + voff(row, ch)));
                uint32_t h0[2] = {a0, a1}, h1[2] = {a2, a3}, l0[2] = {b0, b1}, l1[2] = {b2, b3};
                mma_bf16_16816(s[j], qfl[ks], h0);
                mma_bf16_16816(s[j], qfh[ks], l0);
                mma_bf16_16816(s[j], qfh[ks], h0);
                mma_bf16_16816(s[j + 1], qfl[ks], h1);
                mma_bf16_16816(s[j + 1], qfh[ks], l1);
                mma_bf16_16816(s[j + 1], qfh[ks], h1);
            }
        }
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int key = j0 + j * 8 + t4 * 2;
            s[j][0] = (key < n) ? s[j][0] * scale_log2 : -INFINITY;
            s[j][1] = (key + 1 < n) ? s[j][1] * scale_log2 : -INFINITY;
            s[j][2] = (key < n) ? s[j][2] * scale_log2 : -INFINITY;
            s[j][3] = (key + 1 < n) ? s[j][3] * scale_log2 : -INFINITY;
            mx_a = fmaxf(mx_a, fmaxf(s[j][0], s[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(s[j][2], s[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        float alpha_a = (m_a == -INFINITY) ? 0.f : exp2f(m_a - mn_a);
        float alpha_b = (m_b == -INFINITY) ? 0.f : exp2f(m_b - mn_b);
        float rs_a = 0.f, rs_b = 0.f;
        uint32_t pfh[4][4], pfl[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float p0 = exp2f(s[j][0] - mn_a);
            float p1 = exp2f(s[j][1] - mn_a);
            float p2 = exp2f(s[j][2] - mn_b);
            float p3 = exp2f(s[j][3] - mn_b);
            rs_a += p0 + p1;
            rs_b += p2 + p3;
            int ks = j >> 1, hi = j & 1;
            pfh[ks][hi * 2 + 0] = pack_bf16x2(p0, p1);
            pfh[ks][hi * 2 + 1] = pack_bf16x2(p2, p3);
            pfl[ks][hi * 2 + 0] = pack_lo_bf16x2(p0, p1);
            pfl[ks][hi * 2 + 1] = pack_lo_bf16x2(p2, p3);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a;
        l_b = l_b * alpha_b + rs_b;
        m_a = mn_a;
        m_b = mn_b;
#pragma unroll
        for (int j = 0; j < VNT; ++j) {
            o_acc[j][0] *= alpha_a; o_acc[j][1] *= alpha_a;
            o_acc[j][2] *= alpha_b; o_acc[j][3] *= alpha_b;
        }
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
#pragma unroll
            for (int j = 0; j < VNT; j += 2) {
                int key = ks * 16 + (lidx & 1) * 8 + lrow;
                int ch = j + (lidx >> 1);
                uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
                ldmatrix_x4_trans(a0, a1, a2, a3, smem_u32(vh + voff(key, ch)));
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vl + voff(key, ch)));
                uint32_t h0[2] = {a0, a1}, h1[2] = {a2, a3}, l0[2] = {b0, b1}, l1[2] = {b2, b3};
                mma_bf16_16816(o_acc[j], pfl[ks], h0);
                mma_bf16_16816(o_acc[j], pfh[ks], l0);
                mma_bf16_16816(o_acc[j], pfh[ks], h0);
                mma_bf16_16816(o_acc[j + 1], pfl[ks], h1);
                mma_bf16_16816(o_acc[j + 1], pfh[ks], l1);
                mma_bf16_16816(o_acc[j + 1], pfh[ks], h1);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
    }
    float inv_a = (l_a > 0.f) ? 1.f / l_a : 0.f, inv_b = (l_b > 0.f) ? 1.f / l_b : 0.f;
#pragma unroll
    for (int j = 0; j < VNT; ++j) {
        int d = j * 8 + t4 * 2;
        if (d >= VD) continue;
        if (row_a < n) *reinterpret_cast<float2*>(out + ((int64_t)row_a * H + h) * VD + d) = make_float2(o_acc[j][0] * inv_a, o_acc[j][1] * inv_a);
        if (row_b < n) *reinterpret_cast<float2*>(out + ((int64_t)row_b * H + h) * VD + d) = make_float2(o_acc[j][2] * inv_b, o_acc[j][3] * inv_b);
    }
}

// Position embedding with ComfyUI's bf16 arithmetic: x[t, c] += bf(bf(bf(bf(T0 w0) + bf(T1 w1)) + bf(T2 w2)) + bf(T3 w3))
// (table bf16 [*, C], idx i32 [n, 4], w f32 [n, 4] already bf16-rounded).
extern "C" __global__ void k_pos_embed_bf16emu(float* __restrict__ x, const bf16* __restrict__ table, const int* __restrict__ idx,
                                              const float* __restrict__ w, int n, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * C) return;
    int64_t t = i / C;
    int c = i % C;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float p = round_bf16(bf2f(table[(int64_t)idx[t * 4 + j] * C + c]) * w[t * 4 + j]);
        acc = (j == 0) ? p : round_bf16(acc + p);
    }
    x[i] += acc;
}
