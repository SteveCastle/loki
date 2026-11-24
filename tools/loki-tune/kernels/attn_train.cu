// Attention for training (head dim 128, bf16 in, f32 accumulate): a flash forward that also writes each query row's
// log-sum-exp, and the FlashAttention-2 backward split in two deterministic kernels (no atomics):
//   k_attn_bwd_dkdv  one CTA per 128 keys, loops over query tiles:  dV += P^T dO,  dK += dS^T Q
//   k_attn_bwd_dq    one CTA per 128 queries, loops over key tiles: dQ += dS K
// with P = exp2(S * scale_log2 - lse2), dS = P (dP - D), dP = dO V^T, D = rowsum(dO o O) (k_attn_bwd_prep).
// Full (non-causal) attention over n tokens: every query sees every key, as in MOSS-Image's joint stream.
//
// Layout: token-major with strides (elements): x[tok * ts + head * 128 + d]. lse2 / D are f32 [H, n].
// mma.m16n8k16 fragments as in loki-dream's attention.cu: A row-major 16x16, B "col" 16x8, C 16x8 (rows g, g+8).
#include "common.cuh"

namespace {

constexpr int D = 128;
constexpr int ROWB = 256;  // bytes per 128-dim bf16 row
constexpr int CHUNKS = 16; // 16-byte chunks per row
DEVI int off(int row, int chunk) { return row * ROWB + ((chunk ^ (row & 7)) << 4); }

struct View {
    const bf16* p;
    int64_t ts;
};

// rows [r0, r0 + rows) of a strided view into a swizzled smem tile (zero rows past n)
DEVI void load_rows(uint8_t* dst, const bf16* src, int64_t ts, int head, int r0, int rows, int n, int tid, int threads) {
    for (int c = tid; c < rows * CHUNKS; c += threads) {
        int row = c / CHUNKS, ch = c % CHUNKS;
        int i = r0 + row;
        bool pred = i < n;
        const bf16* s = src + (int64_t)(pred ? i : 0) * ts + head * D + ch * 8;
        cp_async_16(smem_u32(dst + off(row, ch)), s, pred);
    }
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------------------------------------- forward
// grid (ceil(n/128), H), 256 threads. o / lse2 written for every row < n.
__global__ void __launch_bounds__(256) k_attn_fwd_lse(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                      int64_t ts, bf16* __restrict__ o, int64_t o_ts, float* __restrict__ lse2,
                                                      int n, float scale_log2) {
    constexpr int BQ = 128, BKV = 64, THREADS = 256;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* Ks = smem + BQ * ROWB;
    uint8_t* Vs = Ks + 2 * BKV * ROWB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ, h = blockIdx.y;
    const int ntiles = (n + BKV - 1) / BKV;
    load_rows(Qs, q, ts, h, q0, BQ, n, tid, THREADS);
    auto load_kv = [&](int t, int stage) {
        load_rows(Ks + stage * BKV * ROWB, k, ts, h, t * BKV, BKV, n, tid, THREADS);
        load_rows(Vs + stage * BKV * ROWB, v, ts, h, t * BKV, BKV, n, tid, THREADS);
    };
    load_kv(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    const int lidx = lane >> 3, lrow = lane & 7;
    uint32_t qf[8][4];
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        int row = warp * 16 + (lidx & 1) * 8 + lrow;
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + off(row, 2 * s + (lidx >> 1))));
    }
    const int g = lane >> 2, t4 = lane & 3;
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    for (int t = 0; t < ntiles; ++t) {
        const int stage = t & 1;
        if (t + 1 < ntiles) load_kv(t + 1, stage ^ 1);
        cp_async_commit();
        const uint8_t* ks = Ks + stage * BKV * ROWB;
        const uint8_t* vs = Vs + stage * BKV * ROWB;
        const int j0 = t * BKV;
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + off((j + (lidx >> 1)) * 8 + lrow, 2 * kk + (lidx & 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(s[j], qf[kk], bb0);
                mma_bf16_16816(s[j + 1], qf[kk], bb1);
            }
        }
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int key = j0 + j * 8 + t4 * 2;
            s[j][0] = key < n ? s[j][0] * scale_log2 : -INFINITY;
            s[j][1] = key + 1 < n ? s[j][1] * scale_log2 : -INFINITY;
            s[j][2] = key < n ? s[j][2] * scale_log2 : -INFINITY;
            s[j][3] = key + 1 < n ? s[j][3] * scale_log2 : -INFINITY;
            mx_a = fmaxf(mx_a, fmaxf(s[j][0], s[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(s[j][2], s[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        float al_a = (m_a == -INFINITY) ? 0.f : fast_exp2(m_a - mn_a);
        float al_b = (m_b == -INFINITY) ? 0.f : fast_exp2(m_b - mn_b);
        float rs_a = 0.f, rs_b = 0.f;
        uint32_t pf[4][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float p0 = fast_exp2(s[j][0] - mn_a), p1 = fast_exp2(s[j][1] - mn_a);
            float p2 = fast_exp2(s[j][2] - mn_b), p3 = fast_exp2(s[j][3] - mn_b);
            rs_a += p0 + p1;
            rs_b += p2 + p3;
            pf[j >> 1][(j & 1) * 2 + 0] = pack_bf16x2(p0, p1);
            pf[j >> 1][(j & 1) * 2 + 1] = pack_bf16x2(p2, p3);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * al_a + rs_a;
        l_b = l_b * al_b + rs_b;
        m_a = mn_a;
        m_b = mn_b;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            o_acc[j][0] *= al_a; o_acc[j][1] *= al_a;
            o_acc[j][2] *= al_b; o_acc[j][3] *= al_b;
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vs + off(kk * 16 + (lidx & 1) * 8 + lrow, j + (lidx >> 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(o_acc[j], pf[kk], bb0);
                mma_bf16_16816(o_acc[j + 1], pf[kk], bb1);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
    }
    float inv_a = l_a > 0.f ? 1.f / l_a : 0.f, inv_b = l_b > 0.f ? 1.f / l_b : 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        if (row_a < n) *reinterpret_cast<bf162*>(o + (int64_t)row_a * o_ts + h * D + d) = __floats2bfloat162_rn(o_acc[j][0] * inv_a, o_acc[j][1] * inv_a);
        if (row_b < n) *reinterpret_cast<bf162*>(o + (int64_t)row_b * o_ts + h * D + d) = __floats2bfloat162_rn(o_acc[j][2] * inv_b, o_acc[j][3] * inv_b);
    }
    if (t4 == 0) {
        if (row_a < n) lse2[(int64_t)h * n + row_a] = m_a + __log2f(l_a);
        if (row_b < n) lse2[(int64_t)h * n + row_b] = m_b + __log2f(l_b);
    }
}

// ---------------------------------------------------------------------------------------------------------- backward
// D[h, i] = sum_d dO[i, h, d] * O[i, h, d]   (one warp per (row, head))
__global__ void k_attn_bwd_prep(const bf16* __restrict__ dout, int64_t do_ts, const bf16* __restrict__ o, int64_t o_ts,
                                float* __restrict__ dsum, int n, int H) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)n * H) return;
    int i = gw / H, h = gw % H;
    const bf16* a = dout + (int64_t)i * do_ts + h * D + lane * 4;
    const bf16* b = o + (int64_t)i * o_ts + h * D + lane * 4;
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) s += bf2f(a[j]) * bf2f(b[j]);
    s = warp_sum(s);
    if (lane == 0) dsum[(int64_t)h * n + i] = s;
}

// dK, dV for 128 keys per CTA (8 warps x 16 keys), looping over query tiles of 32 (double-buffered).
// grid (ceil(n/128), H), 256 threads, smem 2*128*ROWB + 2*2*32*ROWB + 2*2*32*4 bytes.
__global__ void __launch_bounds__(256) k_attn_bwd_dkdv(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                       int64_t ts, const bf16* __restrict__ dout, int64_t do_ts,
                                                       const float* __restrict__ lse2, const float* __restrict__ dsum,
                                                       bf16* __restrict__ dk, bf16* __restrict__ dv, int64_t dkv_ts,
                                                       int n, float scale_log2, float scale) {
    constexpr int BKV = 128, BQ = 32, THREADS = 256;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Ks = smem;
    uint8_t* Vs = Ks + BKV * ROWB;
    uint8_t* Qs = Vs + BKV * ROWB;     // 2 stages
    uint8_t* Os = Qs + 2 * BQ * ROWB;  // dO, 2 stages
    float* Ls = reinterpret_cast<float*>(Os + 2 * BQ * ROWB);  // lse2, 2 stages
    float* Ds = Ls + 2 * BQ;                                   // D, 2 stages
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int k0 = blockIdx.x * BKV, h = blockIdx.y;
    const int ntiles = (n + BQ - 1) / BQ;
    load_rows(Ks, k, ts, h, k0, BKV, n, tid, THREADS);
    load_rows(Vs, v, ts, h, k0, BKV, n, tid, THREADS);
    auto load_q = [&](int t, int stage) {
        load_rows(Qs + stage * BQ * ROWB, q, ts, h, t * BQ, BQ, n, tid, THREADS);
        load_rows(Os + stage * BQ * ROWB, dout, do_ts, h, t * BQ, BQ, n, tid, THREADS);
        if (tid < BQ) {
            int i = t * BQ + tid;
            Ls[stage * BQ + tid] = i < n ? lse2[(int64_t)h * n + i] : 0.f;
            Ds[stage * BQ + tid] = i < n ? dsum[(int64_t)h * n + i] : 0.f;
        }
    };
    load_q(0, 0);
    cp_async_commit();
    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    float dk_acc[16][4], dv_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j)
#pragma unroll
        for (int r = 0; r < 4; ++r) dk_acc[j][r] = dv_acc[j][r] = 0.f;
    const int arow = warp * 16 + (lidx & 1) * 8 + lrow;  // A-fragment row (key) for ldmatrix
    for (int t = 0; t < ntiles; ++t) {
        const int stage = t & 1;
        cp_async_wait<0>();
        __syncthreads();
        if (t + 1 < ntiles) load_q(t + 1, stage ^ 1);
        cp_async_commit();
        const uint8_t* qs = Qs + stage * BQ * ROWB;
        const uint8_t* os = Os + stage * BQ * ROWB;
        const float* ls = Ls + stage * BQ;
        const float* dsm = Ds + stage * BQ;
        // S^T = K Q^T and dP^T = V dO^T : 16 keys x 32 queries per warp (4 n-tiles)
        float s[4][4], dp[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) s[j][r] = dp[j][r] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t ka[4], va[4];
            ldmatrix_x4(ka[0], ka[1], ka[2], ka[3], smem_u32(Ks + off(arow, 2 * kk + (lidx >> 1))));
            ldmatrix_x4(va[0], va[1], va[2], va[3], smem_u32(Vs + off(arow, 2 * kk + (lidx >> 1))));
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(qs + off((j + (lidx >> 1)) * 8 + lrow, 2 * kk + (lidx & 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(s[j], ka, bb0);
                mma_bf16_16816(s[j + 1], ka, bb1);
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(os + off((j + (lidx >> 1)) * 8 + lrow, 2 * kk + (lidx & 1))));
                uint32_t cc0[2] = {b0, b1}, cc1[2] = {b2, b3};
                mma_bf16_16816(dp[j], va, cc0);
                mma_bf16_16816(dp[j + 1], va, cc1);
            }
        }
        // P^T and dS^T as A fragments (k = queries): 2 k-steps of 16 queries
        uint32_t pf[2][4], sf[2][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int c = j * 8 + t4 * 2;  // query column within the tile
            bool v0 = t * BQ + c < n, v1 = t * BQ + c + 1 < n;
            float l0 = ls[c], l1 = ls[c + 1], d0 = dsm[c], d1 = dsm[c + 1];
            float p0 = v0 ? fast_exp2(s[j][0] * scale_log2 - l0) : 0.f;
            float p1 = v1 ? fast_exp2(s[j][1] * scale_log2 - l1) : 0.f;
            float p2 = v0 ? fast_exp2(s[j][2] * scale_log2 - l0) : 0.f;
            float p3 = v1 ? fast_exp2(s[j][3] * scale_log2 - l1) : 0.f;
            pf[j >> 1][(j & 1) * 2 + 0] = pack_bf16x2(p0, p1);
            pf[j >> 1][(j & 1) * 2 + 1] = pack_bf16x2(p2, p3);
            sf[j >> 1][(j & 1) * 2 + 0] = pack_bf16x2(p0 * (dp[j][0] - d0), p1 * (dp[j][1] - d1));
            sf[j >> 1][(j & 1) * 2 + 1] = pack_bf16x2(p2 * (dp[j][2] - d0), p3 * (dp[j][3] - d1));
        }
        // dV += P^T dO, dK += dS^T Q : 16 keys x 128 dims, k = 32 queries
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                int qrow = kk * 16 + (lidx & 1) * 8 + lrow;
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(os + off(qrow, j + (lidx >> 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(dv_acc[j], pf[kk], bb0);
                mma_bf16_16816(dv_acc[j + 1], pf[kk], bb1);
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(qs + off(qrow, j + (lidx >> 1))));
                uint32_t cc0[2] = {b0, b1}, cc1[2] = {b2, b3};
                mma_bf16_16816(dk_acc[j], sf[kk], cc0);
                mma_bf16_16816(dk_acc[j + 1], sf[kk], cc1);
            }
        }
    }
    cp_async_wait<0>();
    const int key_a = k0 + warp * 16 + g, key_b = key_a + 8;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        if (key_a < n) {
            *reinterpret_cast<bf162*>(dk + (int64_t)key_a * dkv_ts + h * D + d) = __floats2bfloat162_rn(dk_acc[j][0] * scale, dk_acc[j][1] * scale);
            *reinterpret_cast<bf162*>(dv + (int64_t)key_a * dkv_ts + h * D + d) = __floats2bfloat162_rn(dv_acc[j][0], dv_acc[j][1]);
        }
        if (key_b < n) {
            *reinterpret_cast<bf162*>(dk + (int64_t)key_b * dkv_ts + h * D + d) = __floats2bfloat162_rn(dk_acc[j][2] * scale, dk_acc[j][3] * scale);
            *reinterpret_cast<bf162*>(dv + (int64_t)key_b * dkv_ts + h * D + d) = __floats2bfloat162_rn(dv_acc[j][2], dv_acc[j][3]);
        }
    }
}

// dQ for 128 queries per CTA (8 warps x 16 rows), looping over key tiles of 32 (double-buffered).
// grid (ceil(n/128), H), 256 threads, smem 2*128*ROWB + 2*2*32*ROWB bytes.
__global__ void __launch_bounds__(256) k_attn_bwd_dq(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                     int64_t ts, const bf16* __restrict__ dout, int64_t do_ts,
                                                     const float* __restrict__ lse2, const float* __restrict__ dsum,
                                                     bf16* __restrict__ dq, int64_t dq_ts, int n, float scale_log2, float scale) {
    constexpr int BQ = 128, BKV = 32, THREADS = 256;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* Os = Qs + BQ * ROWB;
    uint8_t* Ks = Os + BQ * ROWB;      // 2 stages
    uint8_t* Vs = Ks + 2 * BKV * ROWB; // 2 stages
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ, h = blockIdx.y;
    const int ntiles = (n + BKV - 1) / BKV;
    load_rows(Qs, q, ts, h, q0, BQ, n, tid, THREADS);
    load_rows(Os, dout, do_ts, h, q0, BQ, n, tid, THREADS);
    auto load_kv = [&](int t, int stage) {
        load_rows(Ks + stage * BKV * ROWB, k, ts, h, t * BKV, BKV, n, tid, THREADS);
        load_rows(Vs + stage * BKV * ROWB, v, ts, h, t * BKV, BKV, n, tid, THREADS);
    };
    load_kv(0, 0);
    cp_async_commit();
    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    const float l_a = row_a < n ? lse2[(int64_t)h * n + row_a] : 0.f, l_b = row_b < n ? lse2[(int64_t)h * n + row_b] : 0.f;
    const float d_a = row_a < n ? dsum[(int64_t)h * n + row_a] : 0.f, d_b = row_b < n ? dsum[(int64_t)h * n + row_b] : 0.f;
    const int arow = warp * 16 + (lidx & 1) * 8 + lrow;
    float dq_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) dq_acc[j][0] = dq_acc[j][1] = dq_acc[j][2] = dq_acc[j][3] = 0.f;
    for (int t = 0; t < ntiles; ++t) {
        const int stage = t & 1;
        cp_async_wait<0>();
        __syncthreads();
        if (t + 1 < ntiles) load_kv(t + 1, stage ^ 1);
        cp_async_commit();
        const uint8_t* ks = Ks + stage * BKV * ROWB;
        const uint8_t* vs = Vs + stage * BKV * ROWB;
        float s[4][4], dp[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) s[j][r] = dp[j][r] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t qa[4], oa[4];
            ldmatrix_x4(qa[0], qa[1], qa[2], qa[3], smem_u32(Qs + off(arow, 2 * kk + (lidx >> 1))));
            ldmatrix_x4(oa[0], oa[1], oa[2], oa[3], smem_u32(Os + off(arow, 2 * kk + (lidx >> 1))));
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + off((j + (lidx >> 1)) * 8 + lrow, 2 * kk + (lidx & 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(s[j], qa, bb0);
                mma_bf16_16816(s[j + 1], qa, bb1);
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(vs + off((j + (lidx >> 1)) * 8 + lrow, 2 * kk + (lidx & 1))));
                uint32_t cc0[2] = {b0, b1}, cc1[2] = {b2, b3};
                mma_bf16_16816(dp[j], oa, cc0);
                mma_bf16_16816(dp[j + 1], oa, cc1);
            }
        }
        uint32_t sf[2][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int key = t * BKV + j * 8 + t4 * 2;
            bool v0 = key < n, v1 = key + 1 < n;
            float p0 = v0 ? fast_exp2(s[j][0] * scale_log2 - l_a) : 0.f;
            float p1 = v1 ? fast_exp2(s[j][1] * scale_log2 - l_a) : 0.f;
            float p2 = v0 ? fast_exp2(s[j][2] * scale_log2 - l_b) : 0.f;
            float p3 = v1 ? fast_exp2(s[j][3] * scale_log2 - l_b) : 0.f;
            sf[j >> 1][(j & 1) * 2 + 0] = pack_bf16x2(p0 * (dp[j][0] - d_a), p1 * (dp[j][1] - d_a));
            sf[j >> 1][(j & 1) * 2 + 1] = pack_bf16x2(p2 * (dp[j][2] - d_b), p3 * (dp[j][3] - d_b));
        }
        // dQ += dS K : k = 32 keys, n = 128 dims
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(ks + off(kk * 16 + (lidx & 1) * 8 + lrow, j + (lidx >> 1))));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(dq_acc[j], sf[kk], bb0);
                mma_bf16_16816(dq_acc[j + 1], sf[kk], bb1);
            }
        }
    }
    cp_async_wait<0>();
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        if (row_a < n) *reinterpret_cast<bf162*>(dq + (int64_t)row_a * dq_ts + h * D + d) = __floats2bfloat162_rn(dq_acc[j][0] * scale, dq_acc[j][1] * scale);
        if (row_b < n) *reinterpret_cast<bf162*>(dq + (int64_t)row_b * dq_ts + h * D + d) = __floats2bfloat162_rn(dq_acc[j][2] * scale, dq_acc[j][3] * scale);
    }
}

}  // extern "C"
