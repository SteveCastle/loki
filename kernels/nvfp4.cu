// NVFP4 (E2M1 + fp8-e4m3 block scale per 16 along K, cuBLAS-swizzled scales, f32 per-row tensor scale)
// weight-only GEMM for the Qwen3-VL-32B conditioning encoder, plus small helpers.
//
//   C[M,N] (+)= (A[M,K] . Wdq[N,K]^T) * ts[n]          Wdq = e2m1(nibble) * e4m3(block scale)
//
// The 4-bit weights stay packed in global memory and are expanded straight into mma B fragments:
// e2m1 * e4m3 has at most 2+4 significant bits, so the per-block product is EXACT in bf16 and the
// per-tensor scale (weight_scale_2) is applied in the f32 epilogue.  Weights are therefore streamed
// at 4.5 bits/element instead of materializing a 16-bit copy (4x less traffic for short sequences).
//
// Layouts:  A bf16 row-major [M, lda];  W u8 [N, K/2], element 2j in the high nibble of byte j;
//           S u8 (e4m3) cuBLAS tiled: offset(n, kb) = ((n/128)*(K/64) + kb/4)*512 + (n%32)*16 + ((n%128)/32)*4 + kb%4
//           (for one 128-row x 64-col tile the 512 scale bytes are contiguous).
// Requirements: N % 128 == 0, K % 64 == 0, lda/ldc % 8 == 0.
//
// Tile BM x 128 x 64, 8 warps (2 x 4), warp tile (BM/2) x 32, mma.m16n8k16 bf16 -> f32, 4-stage cp.async.
#include "common.cuh"

namespace {

constexpr int BN = 128, BK = 64, STAGES = 4, THREADS = 256;
constexpr int B_STAGE = BN * BK / 2;  // 4096 bytes of packed fp4
constexpr int S_STAGE = 512;          // scale bytes

template <int BM>
struct Cfg {
    static constexpr int A_STAGE = BM * BK * 2;  // bytes
    static constexpr int STAGE = A_STAGE + B_STAGE + S_STAGE;
    static constexpr int SMEM = STAGES * STAGE;
    static constexpr int MI = BM / 32;  // m16 tiles per warp
};

// A tile rows are 128 bytes (8 chunks of 16B); xor swizzle on the chunk index.
DEVI int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

// e4m3 byte -> f32 (exact; handles subnormals): place eeeemmm in the f32 exponent/mantissa and rescale by 2^120.
DEVI float e4m3_to_f32(uint32_t b) {
    float f = __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20));
    return f * 1.329227995784916e36f;  // 2^120
}

// Two packed bytes x = byte_lo | byte_hi << 8 (each byte = {hi nibble: even element, lo nibble: odd element})
// -> two bf16x2 registers {e_even, e_odd} of byte_lo (r0) and of byte_hi (r1), unscaled e2m1 values.
// The magnitude (3 bits) is looked up with PRMT in byte tables of the bf16 encodings, the sign is or'ed in.
DEVI void e2m1x4_to_bf16x4(uint32_t x, uint32_t& r0, uint32_t& r1) {
    // bf16 of {0, .5, 1, 1.5, 2, 3, 4, 6}: 0000 3F00 3F80 3FC0 4000 4040 4080 40C0
    const uint32_t LO0 = 0xC0800000u, LO1 = 0xC0804000u;  // low bytes  [00 00 80 C0 | 00 40 80 C0]
    const uint32_t HI0 = 0x3F3F3F00u, HI1 = 0x40404040u;  // high bytes [00 3F 3F 3F | 40 40 40 40]
    uint32_t sel = x & 0x7777u;  // nibbles n0..n3 (n0 = odd elem of byte_lo, n1 = even elem of byte_lo, ...)
    uint32_t plo = __byte_perm(LO0, LO1, sel);
    uint32_t phi = __byte_perm(HI0, HI1, sel);
    // r0 = {val(n1), val(n0)}  (low half = even element), r1 = {val(n3), val(n2)}
    r0 = __byte_perm(plo, phi, 0x4051);
    r1 = __byte_perm(plo, phi, 0x6273);
    r0 |= ((x << 8) & 0x8000u) | ((x << 28) & 0x80000000u);
    r1 |= (x & 0x8000u) | ((x << 20) & 0x80000000u);
}

DEVI uint32_t bmul2(uint32_t a, uint32_t s) {
    bf162 r = __hmul2(*reinterpret_cast<bf162*>(&a), *reinterpret_cast<bf162*>(&s));
    return *reinterpret_cast<uint32_t*>(&r);
}

// MODE 0: C bf16 = v ; MODE 1: C f32 += v ; MODE 2: atomicAdd(C f32, v) (split-K)
template <int BM, int MODE>
__device__ void fp4_gemm_kernel(const bf16* __restrict__ A, int lda, const uint8_t* __restrict__ W, const uint8_t* __restrict__ S,
                                const float* __restrict__ ts, void* __restrict__ Cv, int ldc, int M, int N, int K, int k_split) {
    using C_ = Cfg<BM>;
    constexpr int MI = C_::MI;
    extern __shared__ __align__(128) uint8_t smem[];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int m_tiles = (M + BM - 1) / BM;
    const int mt = blockIdx.x % m_tiles, nt = blockIdx.x / m_tiles;
    const int m0 = mt * BM, n0 = nt * BN;
    const int kb0 = blockIdx.z * k_split;  // first K element of this split
    const int KT = min(k_split, K - kb0) / BK;
    const int ncb = K / BK;

    // ---- per-thread copy assignments
    constexpr int A_CH = BM * 8 / THREADS;  // 16B chunks per thread per stage (4 or 2)
    const bf16* a_src[A_CH];
    bool a_ok[A_CH];
    int a_off[A_CH];
#pragma unroll
    for (int i = 0; i < A_CH; ++i) {
        int id = tid + i * THREADS;
        int row = id >> 3, ch = id & 7;
        int m = m0 + row;
        a_ok[i] = m < M;
        a_src[i] = A + (int64_t)(a_ok[i] ? m : 0) * lda + kb0 + ch * 8;
        a_off[i] = swz128(row, ch);
    }
    const int b_row = tid >> 1, b_ch = tid & 1;
    const uint8_t* b_src = W + (int64_t)(n0 + b_row) * (K / 2) + kb0 / 2 + b_ch * 16;
    const uint8_t* s_src = S + ((int64_t)(n0 / 128) * ncb + kb0 / BK) * 512 + tid * 16;

    auto load_stage = [&](int kt, int st) {
        uint8_t* base = smem + st * C_::STAGE;
#pragma unroll
        for (int i = 0; i < A_CH; ++i) cp_async_16(smem_u32(base + a_off[i]), a_src[i] + kt * BK, a_ok[i]);
        cp_async_16(smem_u32(base + C_::A_STAGE + b_row * 32 + b_ch * 16), b_src + kt * (BK / 2));
        if (tid < 32) cp_async_16(smem_u32(base + C_::A_STAGE + B_STAGE + tid * 16), s_src + (int64_t)kt * 512);
    };

    float acc[MI][4][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_stage(s, s);
        cp_async_commit();
    }

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const uint32_t bsel = (uint32_t)t4 | ((uint32_t)(t4 + 4) << 4);  // bytes t and t+4 of an 8-byte block

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_stage(nk, nk % STAGES);
            cp_async_commit();
        }
        const uint8_t* base = smem + (kt % STAGES) * C_::STAGE;
        const uint8_t* As = base;
        const uint8_t* Bs = base + C_::A_STAGE;
        const uint8_t* Ss = Bs + B_STAGE;

        // B rows of this thread: packed bytes (8 words = 4 blocks of 16 elements) and 4 block scales
        uint32_t bw[4][8];
        uint32_t sc[4];
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            int row = wn * 32 + ni * 8 + g;
            uint4 q0 = *reinterpret_cast<const uint4*>(Bs + row * 32);
            uint4 q1 = *reinterpret_cast<const uint4*>(Bs + row * 32 + 16);
            bw[ni][0] = q0.x; bw[ni][1] = q0.y; bw[ni][2] = q0.z; bw[ni][3] = q0.w;
            bw[ni][4] = q1.x; bw[ni][5] = q1.y; bw[ni][6] = q1.z; bw[ni][7] = q1.w;
            sc[ni] = *reinterpret_cast<const uint32_t*>(Ss + (row & 31) * 16 + (row >> 5) * 4);
        }

#pragma unroll
        for (int s = 0; s < 4; ++s) {  // k16 steps
            uint32_t afrag[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                int row = wm * (BM / 2) + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = s * 2 + (lidx >> 1);
                ldmatrix_x4(afrag[mi][0], afrag[mi][1], afrag[mi][2], afrag[mi][3], smem_u32(As + swz128(row, chunk)));
            }
            uint32_t bfrag[4][2];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                uint32_t x = __byte_perm(bw[ni][2 * s], bw[ni][2 * s + 1], bsel);
                uint32_t r0, r1;
                e2m1x4_to_bf16x4(x, r0, r1);
                float bs = e4m3_to_f32((sc[ni] >> (8 * s)) & 0xFFu);
                bf162 b2 = __float2bfloat162_rn(bs);
                uint32_t s2 = *reinterpret_cast<uint32_t*>(&b2);
                bfrag[ni][0] = bmul2(r0, s2);
                bfrag[ni][1] = bmul2(r1, s2);
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_bf16_16816(acc[mi][ni], afrag[mi], bfrag[ni]);
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
        int n = n0 + wn * 32 + ni * 8 + t4 * 2;
        float s0 = ts[n], s1 = ts[n + 1];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                int m = m0 + wm * (BM / 2) + mi * 16 + g + half * 8;
                if (m >= M) continue;
                float v0 = acc[mi][ni][half * 2] * s0, v1 = acc[mi][ni][half * 2 + 1] * s1;
                int64_t off = (int64_t)m * ldc + n;
                if (MODE == 0) {
                    *reinterpret_cast<bf162*>(reinterpret_cast<bf16*>(Cv) + off) = __floats2bfloat162_rn(v0, v1);
                } else if (MODE == 1) {
                    float2* p = reinterpret_cast<float2*>(reinterpret_cast<float*>(Cv) + off);
                    float2 r = *p;
                    r.x += v0; r.y += v1;
                    *p = r;
                } else {
                    float* p = reinterpret_cast<float*>(Cv) + off;
                    atomicAdd(p, v0);
                    atomicAdd(p + 1, v1);
                }
            }
        }
    }
}

}  // namespace

#define FP4_GEMM(BM, MODE, NAME)                                                                                          \
    extern "C" __global__ void __launch_bounds__(256) NAME(const bf16* A, int lda, const uint8_t* W, const uint8_t* S,   \
                                                         const float* ts, void* C, int ldc, int M, int N, int K, int ks) { \
        fp4_gemm_kernel<BM, MODE>(A, lda, W, S, ts, C, ldc, M, N, K, ks);                                                 \
    }
FP4_GEMM(128, 0, k_fp4_gemm_128_bf16)
FP4_GEMM(128, 1, k_fp4_gemm_128_addf32)
FP4_GEMM(128, 2, k_fp4_gemm_128_atomf32)
FP4_GEMM(64, 0, k_fp4_gemm_64_bf16)
FP4_GEMM(64, 1, k_fp4_gemm_64_addf32)
FP4_GEMM(64, 2, k_fp4_gemm_64_atomf32)

extern "C" {

// Standalone dequantization (testing / fallback): out[n, k] = bf16(e2m1 * e4m3 * ts[n]).  One thread per 16-element block.
__global__ void k_nvfp4_dequant(const uint8_t* __restrict__ W, const uint8_t* __restrict__ S, const float* __restrict__ ts,
                                bf16* __restrict__ out, int N, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int kblocks = K / 16;
    if (i >= (int64_t)N * kblocks) return;
    int n = i / kblocks, kb = i % kblocks;
    uint2 q = *reinterpret_cast<const uint2*>(W + (int64_t)n * (K / 2) + kb * 8);
    int64_t so = ((int64_t)(n / 128) * (K / 64) + kb / 4) * 512 + (n % 32) * 16 + ((n % 128) / 32) * 4 + kb % 4;
    float sc = e4m3_to_f32(S[so]) * ts[n];
    uint32_t words[2] = {q.x, q.y};
    uint32_t o[8];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t x = (words[w] >> (16 * h)) & 0xFFFFu;
            uint32_t r0, r1;
            e2m1x4_to_bf16x4(x, r0, r1);
            bf162 a = *reinterpret_cast<bf162*>(&r0), b = *reinterpret_cast<bf162*>(&r1);
            bf162 ra = __floats2bfloat162_rn(__low2float(a) * sc, __high2float(a) * sc);
            bf162 rb = __floats2bfloat162_rn(__low2float(b) * sc, __high2float(b) * sc);
            o[w * 4 + h * 2] = *reinterpret_cast<uint32_t*>(&ra);
            o[w * 4 + h * 2 + 1] = *reinterpret_cast<uint32_t*>(&rb);
        }
    }
    uint4* dst = reinterpret_cast<uint4*>(out + (int64_t)n * K + kb * 16);
    dst[0] = make_uint4(o[0], o[1], o[2], o[3]);
    dst[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// h[m, j] = bf16(silu(gu[m, j]) * gu[m, H + j] * pqs[j])     (gate | up concatenated, AWQ scale of down_proj folded in)
__global__ void k_swiglu_pqs_bf16(const bf16* __restrict__ gu, const bf16* __restrict__ pqs, bf16* __restrict__ h, int64_t M, int H) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= M * H) return;
    int64_t m = i / H;
    int j = i % H;
    bf162 g = *reinterpret_cast<const bf162*>(gu + m * 2 * H + j);
    bf162 u = *reinterpret_cast<const bf162*>(gu + m * 2 * H + H + j);
    bf162 p = *reinterpret_cast<const bf162*>(pqs + j);
    float a = silu_f(__low2float(g)) * __low2float(u) * __low2float(p);
    float b = silu_f(__high2float(g)) * __high2float(u) * __high2float(p);
    *reinterpret_cast<bf162*>(h + i) = __floats2bfloat162_rn(a, b);
}

// x[m, k] *= s[k]   (bf16, in place; K even)
__global__ void k_mul_cols_bf16(bf16* __restrict__ x, const bf16* __restrict__ s, int64_t M, int K) {
    int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= M * K) return;
    int k = i % K;
    bf162 v = *reinterpret_cast<bf162*>(x + i);
    bf162 p = *reinterpret_cast<const bf162*>(s + k);
    *reinterpret_cast<bf162*>(x + i) = __hmul2(v, p);
}

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
