// MiniMax H3 video VAE kernels (fp16 activations, f32 accumulation).
//
// Encoder: channels-last (NDHWC) fp16 volumes. Per-frame GroupNorm(32)+SiLU fused with the causal / reflect
// padding pass (k_vv_gn_*), then a "valid" implicit-GEMM conv3d on the padded volume (k_vv_conv3d) with a
// bias (+ residual) epilogue.
// Decoder (ViT3D): fp16 tensor-core GEMM with fused epilogues (k_vv_gemm), RMSNorm / LayerNorm, fused per-head
// q/k RMSNorm + partial split-half 3D rope, fp16 flash attention (head dim 64, batched tiles), unpatchify +
// spatial tile blending into an f32 canvas, temporal blend + pixel de-normalization -> u8.
#include "common.cuh"

typedef __half f16;
typedef __half2 f162;

DEVI float h2f(f16 x) { return __half2float(x); }
DEVI f16 f2h(float x) { return __float2half_rn(x); }
DEVI float rh(float x) { return __half2float(__float2half_rn(x)); }

// D(16x8,f32) += A(16x16,f16,row) * B(16x8,f16,col)
DEVI void mma_f16_16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D(16x8,f16) += A(16x16,f16,row) * B(16x8,f16,col)   (fp16 accumulate: 2x tensor rate on GeForce Ada)
DEVI void mma_f16acc_16816(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI uint32_t pack_h2(float lo, float hi) {
    f162 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

namespace {
// block tile 256 x 128 x 32, 16 warps (4 x 4) of 64 x 32, 4-stage cp.async pipeline (96 KB smem)
// BM = 256: 16 warps; BM = 128: 8 warps (for small M)
constexpr int BN = 128, BK = 32, STAGES = 4;
constexpr int B_TILE = BN * BK * 2;
#define TILE_CONSTS                                      constexpr int THREADS = BM * 2;                      constexpr int A_TILE = BM * BK * 2;                  constexpr int A_IT = BM * 4 / THREADS;               constexpr int B_IT = BN * 4 / THREADS;

// One BK=32 slab of the 128x128 block tile: 8 warps (2x4), warp tile 64x32.
// HACC: the slab's 32-term partial sums accumulate in fp16 and are promoted to the f32 accumulators per slab.
// One BK=32 slab of the block tile; warp tile 64x32 (4 x m16, 4 x n8).
// HACC: per 16-row strip the slab's 32-term partial sums accumulate in fp16 (2x tensor rate on GeForce Ada) and are
// promoted to the f32 accumulators at the end of the slab.
template <bool HACC>
DEVI void mma_slab(const uint8_t* as, const uint8_t* bs, float (&acc)[4][4][4], int wm, int wn, int lane) {
    const int lidx = lane >> 3, lrow = lane & 7;
    uint32_t bfrag[2][4][2];
#pragma unroll
    for (int ks = 0; ks < 2; ++ks)
#pragma unroll
        for (int nj = 0; nj < 4; nj += 2) {
            int row = wn * 32 + (nj + (lidx >> 1)) * 8 + lrow;
            int chunk = ks * 2 + (lidx & 1);
            ldmatrix_x4(bfrag[ks][nj][0], bfrag[ks][nj][1], bfrag[ks][nj + 1][0], bfrag[ks][nj + 1][1], smem_u32(bs + swz64(row, chunk)));
        }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        uint32_t afrag[2][4];
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
            int chunk = ks * 2 + (lidx >> 1);
            ldmatrix_x4(afrag[ks][0], afrag[ks][1], afrag[ks][2], afrag[ks][3], smem_u32(as + swz64(row, chunk)));
        }
        if (HACC) {
            uint32_t h[4][2];
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) h[ni][0] = h[ni][1] = 0u;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_f16acc_16816(h[ni], afrag[ks], bfrag[ks][ni]);
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                float2 lo = __half22float2(*reinterpret_cast<f162*>(&h[ni][0]));
                float2 hi = __half22float2(*reinterpret_cast<f162*>(&h[ni][1]));
                acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y;
                acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
            }
        } else {
#pragma unroll
            for (int ks = 0; ks < 2; ++ks)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_f16_16816(acc[mi][ni], afrag[ks], bfrag[ks][ni]);
        }
    }
}

template <int BM>
DEVI void tile_coords(int M, int N, int& m0, int& n0) {
    const int n_tiles = (N + BN - 1) / BN, m_tiles = (M + BM - 1) / BM;
    constexpr int GROUP_M = 8;
    const int bid = blockIdx.x;
    const int group = bid / (GROUP_M * n_tiles);
    const int first_m = group * GROUP_M;
    const int gsize = min(GROUP_M, m_tiles - first_m);
    const int in_group = bid - group * GROUP_M * n_tiles;
    m0 = (first_m + (in_group % gsize)) * BM;
    n0 = (in_group / gsize) * BN;
}

struct ConvParams {
    const f16* in; int Tp, Hp, Wp, Ci;           // padded input volume [Tp, Hp, Wp, Ci]
    const f16* w; int ldw; int K;                // weights [Co][ldw], taps ordered (kt, kh, kw, ci); K = taps*Ci used
    const float* bias; int Co;
    f16* out; int To, Ho, Wo;                    // output [To, Ho, Wo, Co]
    int KT, KH, KW, st, sh, sw;
    const f16* res;                              // optional residual [To*Ho*Wo, Co]
};
}  // namespace

// =============================================================================================
// Implicit-GEMM conv3d (valid, strided) on a pre-padded NDHWC volume.
//   out[m, co] = bias[co] + sum_{tap, ci} in[(to*st+dt, ho*sh+dy, wo*sw+dx), ci] * w[co][tap*Ci + ci] (+ res[m, co])
// Ci % 8 == 0; ldw % 8 == 0.
template <bool HACC, int BM>
__device__ __forceinline__ void conv3d_body(const ConvParams& P) {
    TILE_CONSTS
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ int tap_off[32];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * A_TILE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int M = P.To * P.Ho * P.Wo, N = P.Co, K = P.K;
    int m0, n0;
    tile_coords<BM>(M, N, m0, n0);
    const int ntaps = P.KT * P.KH * P.KW;
    if (tid < ntaps) {
        int dt = tid / (P.KH * P.KW), r = tid - dt * P.KH * P.KW;
        int dy = r / P.KW, dx = r - dy * P.KW;
        tap_off[tid] = ((dt * P.Hp + dy) * P.Wp + dx) * P.Ci;
    }
    __syncthreads();

    const int c_chunk = tid & 3;
    int64_t a_base[A_IT];
    bool a_pred[A_IT];
    const f16* b_src[B_IT];
    bool b_pred[B_IT];
    const int HWo = P.Ho * P.Wo;
#pragma unroll
    for (int i = 0; i < A_IT; ++i) {
        int row = (tid + i * THREADS) >> 2;
        int m = m0 + row;
        a_pred[i] = m < M;
        int mm = a_pred[i] ? m : 0;
        int to = mm / HWo, r = mm - to * HWo;
        int ho = r / P.Wo, wo = r - ho * P.Wo;
        a_base[i] = (((int64_t)to * P.st * P.Hp + (int64_t)ho * P.sh) * P.Wp + (int64_t)wo * P.sw) * P.Ci;
    }
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
        int row = (tid + i * THREADS) >> 2;
        int n = n0 + row;
        b_pred[i] = n < N;
        b_src[i] = P.w + (int64_t)(b_pred[i] ? n : 0) * P.ldw + c_chunk * 8;
    }
    const bool fast = (P.Ci % BK) == 0;
    // incremental (tap, ci) of this thread's chunk for the next tile to load (fast path)
    int ld_tap = 0, ld_ci = c_chunk * 8;
    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * A_TILE;
        uint8_t* bs = Bs + stage * B_TILE;
        int k = k0 + c_chunk * 8;
        bool kp = k < K;
        int off;
        if (fast) {
            off = tap_off[min(ld_tap, ntaps - 1)] + ld_ci;
            ld_ci += BK;
            if (ld_ci >= P.Ci) { ld_ci -= P.Ci; ++ld_tap; }
        } else {
            int tap = kp ? k / P.Ci : 0;
            off = tap_off[tap] + (k - tap * P.Ci);
        }
#pragma unroll
        for (int i = 0; i < A_IT; ++i) {
            int row = (tid + i * THREADS) >> 2;
            bool pa = kp && a_pred[i];
            const f16* src = pa ? (P.in + a_base[i] + off) : P.in;
            cp_async_16(smem_u32(as + swz64(row, c_chunk)), src, pa);
        }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) {
            int row = (tid + i * THREADS) >> 2;
            cp_async_16(smem_u32(bs + swz64(row, c_chunk)), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
    const int KTl = (K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KTl) load_tile(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KTl; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KTl) load_tile(nk, nk % STAGES);
            cp_async_commit();
        }
        const int stage = kt % STAGES;
        mma_slab<HACC>(As + stage * A_TILE, Bs + stage * B_TILE, acc, wm, wn, lane);
    }
    cp_async_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = m0 + wm * 64 + mi * 16 + g + half * 8;
            if (m >= M) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n = n0 + wn * 32 + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = acc[mi][ni][half * 2 + 0], v1 = acc[mi][ni][half * 2 + 1];
                if (P.bias) { v0 += P.bias[n]; v1 += P.bias[n + 1]; }
                int64_t off = (int64_t)m * N + n;
                if (P.res) {
                    // the reference rounds the conv output to fp16 before the residual add
                    f162 r = *reinterpret_cast<const f162*>(P.res + off);
                    v0 = rh(v0) + __low2float(r);
                    v1 = rh(v1) + __high2float(r);
                }
                *reinterpret_cast<f162*>(P.out + off) = __floats2half2_rn(v0, v1);
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(512) k_vv_conv3d(ConvParams P) { conv3d_body<false, 256>(P); }
extern "C" __global__ void __launch_bounds__(512) k_vv_conv3d_h(ConvParams P) { conv3d_body<true, 256>(P); }
extern "C" __global__ void __launch_bounds__(256) k_vv_conv3d_s(ConvParams P) { conv3d_body<false, 128>(P); }
extern "C" __global__ void __launch_bounds__(256) k_vv_conv3d_h_s(ConvParams P) { conv3d_body<true, 128>(P); }

// =============================================================================================
// fp16 GEMM: C[M,N] = A[M,K] * B[N,K]^T (+ bias f32[N]) with epilogue
//   mode 0: out fp16 [M,N] = v
//   mode 1: out f32  [M,N] = v
//   mode 2: out f32  [M,N] = res f32[M,N] + v * gate f32[N]   (res may alias out)
//   mode 3: out fp16 [M,N/2]: out[m][n/2] = silu(v[n]) * v[n+1]   (weights interleaved gate/up)
// K % 8 == 0, N % 8 == 0.
template <bool HACC, int BM>
__device__ __forceinline__ void gemm_body(const f16* __restrict__ A, const f16* __restrict__ B, void* C, int M, int N, int K,
                                          const float* __restrict__ bias, int mode, const float* res, const float* __restrict__ gate) {
    TILE_CONSTS
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * A_TILE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    int m0, n0;
    tile_coords<BM>(M, N, m0, n0);
    const int c_chunk = tid & 3;
    const f16* a_src[A_IT];
    const f16* b_src[B_IT];
    bool a_pred[A_IT], b_pred[B_IT];
#pragma unroll
    for (int i = 0; i < A_IT; ++i) {
        int row = (tid + i * THREADS) >> 2;
        int am = m0 + row;
        a_pred[i] = am < M;
        a_src[i] = A + (int64_t)(a_pred[i] ? am : 0) * K + c_chunk * 8;
    }
#pragma unroll
    for (int i = 0; i < B_IT; ++i) {
        int row = (tid + i * THREADS) >> 2;
        int bn = n0 + row;
        b_pred[i] = bn < N;
        b_src[i] = B + (int64_t)(b_pred[i] ? bn : 0) * K + c_chunk * 8;
    }
    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * A_TILE;
        uint8_t* bs = Bs + stage * B_TILE;
        bool kp = (k0 + c_chunk * 8) < K;
#pragma unroll
        for (int i = 0; i < A_IT; ++i) {
            int row = (tid + i * THREADS) >> 2;
            cp_async_16(smem_u32(as + swz64(row, c_chunk)), a_src[i] + (kp ? k0 : 0), a_pred[i] && kp);
        }
#pragma unroll
        for (int i = 0; i < B_IT; ++i) {
            int row = (tid + i * THREADS) >> 2;
            cp_async_16(smem_u32(bs + swz64(row, c_chunk)), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;
    const int KTl = (K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KTl) load_tile(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < KTl; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KTl) load_tile(nk, nk % STAGES);
            cp_async_commit();
        }
        const int stage = kt % STAGES;
        mma_slab<HACC>(As + stage * A_TILE, Bs + stage * B_TILE, acc, wm, wn, lane);
    }
    cp_async_wait<0>();
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = m0 + wm * 64 + mi * 16 + g + half * 8;
            if (m >= M) continue;
#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n = n0 + wn * 32 + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = acc[mi][ni][half * 2 + 0], v1 = acc[mi][ni][half * 2 + 1];
                if (bias) { v0 += bias[n]; v1 += bias[n + 1]; }
                if (mode == 0) {
                    *reinterpret_cast<f162*>((f16*)C + (int64_t)m * N + n) = __floats2half2_rn(v0, v1);
                } else if (mode == 1) {
                    *reinterpret_cast<float2*>((float*)C + (int64_t)m * N + n) = make_float2(v0, v1);
                } else if (mode == 2) {
                    int64_t off = (int64_t)m * N + n;
                    float2 r = *reinterpret_cast<const float2*>(res + off);
                    r.x += rh(v0) * gate[n];
                    r.y += rh(v1) * gate[n + 1];
                    *reinterpret_cast<float2*>((float*)C + off) = r;
                } else {
                    float gt = rh(v0), up = rh(v1);
                    float s = rh(gt / (1.0f + __expf(-gt)));
                    ((f16*)C)[(int64_t)m * (N / 2) + (n >> 1)] = f2h(s * up);
                }
            }
        }
    }
}
extern "C" __global__ void __launch_bounds__(512) k_vv_gemm(const f16* A, const f16* B, void* C, int M, int N, int K, const float* bias,
                                                            int mode, const float* res, const float* gate) {
    gemm_body<false, 256>(A, B, C, M, N, K, bias, mode, res, gate);
}
extern "C" __global__ void __launch_bounds__(256) k_vv_gemm_s(const f16* A, const f16* B, void* C, int M, int N, int K, const float* bias,
                                                              int mode, const float* res, const float* gate) {
    gemm_body<false, 128>(A, B, C, M, N, K, bias, mode, res, gate);
}
extern "C" __global__ void __launch_bounds__(256) k_vv_gemm_h_s(const f16* A, const f16* B, void* C, int M, int N, int K, const float* bias,
                                                                int mode, const float* res, const float* gate) {
    gemm_body<true, 128>(A, B, C, M, N, K, bias, mode, res, gate);
}
extern "C" __global__ void __launch_bounds__(512) k_vv_gemm_h(const f16* A, const f16* B, void* C, int M, int N, int K, const float* bias,
                                                              int mode, const float* res, const float* gate) {
    gemm_body<true, 256>(A, B, C, M, N, K, bias, mode, res, gate);
}

// =============================================================================================
// Encoder elementwise kernels

// Pixels u8 [Tsrc, Hs, Ws, 3] -> normalized fp16 padded conv_in input [Tp, th+2, tw+2, 8]
// (reflect spatial pad 1, `front` zero frames, channels 3..7 zero). Frame f (0..T-1) reads source frame
// min(t0 + f, Tsrc - 1) (the reference repeats the last frame to fill a short clip).
// Reference rounding: x = fp16(u/255*2-1); x = ((x + 1) * 0.5 - mean) / std, each step rounded to fp16.
extern "C" __global__ void k_vv_pix_in(const uint8_t* __restrict__ pix, int Tsrc, int Hs, int Ws, int t0, int y0, int x0,
                                       int T, int th, int tw, int front, f16* __restrict__ out, const float* __restrict__ mean,
                                       const float* __restrict__ stdv) {
    const int Hp = th + 2, Wp = tw + 2;
    int64_t n = (int64_t)(T + front) * Hp * Wp;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int xp = (int)(i % Wp);
    int64_t r = i / Wp;
    int yp = (int)(r % Hp);
    int tp = (int)(r / Hp);
    float v[3] = {0.f, 0.f, 0.f};
    if (tp >= front) {
        int f = min(t0 + tp - front, Tsrc - 1);
        int y = yp - 1, x = xp - 1;
        if (y < 0) y = -y;
        if (y >= th) y = 2 * th - 2 - y;
        if (x < 0) x = -x;
        if (x >= tw) x = 2 * tw - 2 - x;
        const uint8_t* p = pix + (((int64_t)f * Hs + (y0 + y)) * Ws + (x0 + x)) * 3;
#pragma unroll
        for (int c = 0; c < 3; ++c) {
            float a = __fdiv_rn((float)p[c], 255.0f);
            a = __fsub_rn(__fmul_rn(a, 2.0f), 1.0f);
            a = rh(a);
            a = rh(a + 1.0f);
            a = rh(a * 0.5f);
            a = rh(a - mean[c]);
            a = rh(__fdiv_rn(a, stdv[c]));
            v[c] = a;
        }
    }
    uint4 o;
    o.x = pack_h2(v[0], v[1]);
    o.y = pack_h2(v[2], 0.f);
    o.z = 0;
    o.w = 0;
    *reinterpret_cast<uint4*>(out + i * 8) = o;
}

// Per-frame GroupNorm(32) statistics: x fp16 [T, P, C] -> stats double [T, 32, 2] (sum, sumsq), accumulated
// atomically (zero first). grid (blocks_per_frame, T), 256 threads; (C/8) divides 256.
extern "C" __global__ void k_vv_gn_stats(const f16* __restrict__ x, int P, int C, int pix_per_block, double* __restrict__ stats) {
    __shared__ float ssum[32], ssq[32];
    const int t = blockIdx.y;
    const int nch = C >> 3;
    const int lanes = 256 / nch;
    const int c8 = threadIdx.x % nch, pl = threadIdx.x / nch;
    if (threadIdx.x < 32) { ssum[threadIdx.x] = 0.f; ssq[threadIdx.x] = 0.f; }
    __syncthreads();
    float s[8], q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) { s[j] = 0.f; q[j] = 0.f; }
    const int p0 = blockIdx.x * pix_per_block, p1 = min(P, p0 + pix_per_block);
    const f16* base = x + (int64_t)t * P * C + c8 * 8;
    for (int p = p0 + pl; p < p1; p += lanes) {
        uint4 u = *reinterpret_cast<const uint4*>(base + (int64_t)p * C);
        const f162* h = reinterpret_cast<const f162*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __half22float2(h[j]);
            s[2 * j] += f.x; q[2 * j] += f.x * f.x;
            s[2 * j + 1] += f.y; q[2 * j + 1] += f.y * f.y;
        }
    }
    const int cg = C / 32;  // channels per group
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int grp = (c8 * 8 + j) / cg;
        atomicAdd(&ssum[grp], s[j]);
        atomicAdd(&ssq[grp], q[j]);
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        atomicAdd(&stats[((int64_t)t * 32 + threadIdx.x) * 2 + 0], (double)ssum[threadIdx.x]);
        atomicAdd(&stats[((int64_t)t * 32 + threadIdx.x) * 2 + 1], (double)ssq[threadIdx.x]);
    }
}

// stats (sum, sumsq) -> (mean, rstd) f32 [T*32, 2]
extern "C" __global__ void k_vv_gn_finalize(const double* __restrict__ stats, float* __restrict__ mr, int n, double count, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double m = stats[2 * i] / count;
    double v = stats[2 * i + 1] / count - m * m;
    if (v < 0) v = 0;
    mr[2 * i] = (float)m;
    mr[2 * i + 1] = (float)(1.0 / sqrt(v + (double)eps));
}

// Normalize (+SiLU) and pad: x fp16 [T, H, W, C] -> out [T+front, H+pt+pb, W+pl+pr, C].
// mr == nullptr: pad only. Spatial padding reflects, temporal front padding is zeros.
extern "C" __global__ void k_vv_gn_apply_pad(const f16* __restrict__ x, int T, int H, int W, int C, const float* __restrict__ mr,
                                             const f16* __restrict__ gamma, const f16* __restrict__ beta, int front, int pt, int pb,
                                             int pl, int pr, f16* __restrict__ out) {
    const int Hp = H + pt + pb, Wp = W + pl + pr, nch = C >> 3;
    int64_t n = (int64_t)(T + front) * Hp * Wp * nch;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c8 = (int)(i % nch);
    int64_t r = i / nch;
    int xp = (int)(r % Wp);
    r /= Wp;
    int yp = (int)(r % Hp);
    int tp = (int)(r / Hp);
    uint4 o = make_uint4(0, 0, 0, 0);
    if (tp >= front) {
        int t = tp - front;
        int y = yp - pt, xx = xp - pl;
        if (y < 0) y = -y;
        if (y >= H) y = 2 * H - 2 - y;
        if (xx < 0) xx = -xx;
        if (xx >= W) xx = 2 * W - 2 - xx;
        o = *reinterpret_cast<const uint4*>(x + (((int64_t)t * H + y) * W + xx) * C + c8 * 8);
        if (mr) {
            f162* h = reinterpret_cast<f162*>(&o);
            const int cg = C / 32;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h[j]);
                int c = c8 * 8 + 2 * j;
                int g0 = c / cg, g1 = (c + 1) / cg;
                float a = (f.x - mr[(t * 32 + g0) * 2]) * mr[(t * 32 + g0) * 2 + 1] * h2f(gamma[c]) + h2f(beta[c]);
                float b = (f.y - mr[(t * 32 + g1) * 2]) * mr[(t * 32 + g1) * 2 + 1] * h2f(gamma[c + 1]) + h2f(beta[c + 1]);
                a = rh(a);
                b = rh(b);
                a = a / (1.0f + __expf(-a));
                b = b / (1.0f + __expf(-b));
                h[j] = __floats2half2_rn(a, b);
            }
        }
    }
    *reinterpret_cast<uint4*>(out + i * 8) = o;
}

// =============================================================================================
// Decoder kernels

// Token embedding for one tile: z f32 [24, Tz, Hz, Wz] (normalized latent; frames >= Tz repeat the last),
// tile = frames [t0, t0+tc), rows [y0, y0+th), cols [x0, x0+tw). Writes x f32 [S, 2048] with
// S = tc*th*tw + 5: patch tokens, 4 register tokens, 1 zero token.
//   zd = fp16(fp16(z) * std + mean); p = fp16(post_quant(zd)); x = x_embedder(p)
extern "C" __global__ void k_vv_embed(const float* __restrict__ z, int Tz, int Hz, int Wz, int t0, int y0, int x0, int tc, int th,
                                      int tw, const float* __restrict__ lmean, const float* __restrict__ lstd,
                                      const float* __restrict__ pq_w, const float* __restrict__ pq_b, const float* __restrict__ xe_w,
                                      const float* __restrict__ xe_b, const f16* __restrict__ regs, float* __restrict__ x) {
    __shared__ float zin[24], zp[24];
    const int tok = blockIdx.x;
    const int np = tc * th * tw;
    float* xo = x + (int64_t)tok * 2048;
    if (tok >= np) {
        int rI = tok - np;
        for (int c = threadIdx.x; c < 2048; c += blockDim.x) xo[c] = rI < 4 ? h2f(regs[rI * 2048 + c]) : 0.f;
        return;
    }
    int t = tok / (th * tw), r = tok - t * th * tw;
    int yy = r / tw, xx = r - yy * tw;
    int tz = min(t0 + t, Tz - 1);
    if (threadIdx.x < 24) {
        int c = threadIdx.x;
        float v = rh(z[(((int64_t)c * Tz + tz) * Hz + (y0 + yy)) * Wz + (x0 + xx)]);
        v = rh(v * lstd[c]);
        zin[c] = rh(v + lmean[c]);
    }
    __syncthreads();
    if (threadIdx.x < 24) {
        int o = threadIdx.x;
        float a = pq_b[o];
        for (int c = 0; c < 24; ++c) a += pq_w[o * 24 + c] * zin[c];
        zp[o] = rh(a);
    }
    __syncthreads();
    for (int o = threadIdx.x; o < 2048; o += blockDim.x) {
        float a = xe_b[o];
        const float* wr = xe_w + o * 24;
#pragma unroll
        for (int c = 0; c < 24; ++c) a += wr[c] * zp[c];
        xo[o] = rh(a);
    }
}

// RMSNorm (affine fp16 weight) or LayerNorm (affine weight+bias) over rows of 2048: x f32 -> out fp16.
// One block (256 threads) per row.
extern "C" __global__ void k_vv_norm_rows(const float* __restrict__ x, f16* __restrict__ out, const f16* __restrict__ w,
                                          const f16* __restrict__ b, int layernorm, float eps) {
    __shared__ float red[32];
    const int row = blockIdx.x;
    const float* xr = x + (int64_t)row * 2048;
    float v[8];
    float4 a = reinterpret_cast<const float4*>(xr)[threadIdx.x * 2];
    float4 c = reinterpret_cast<const float4*>(xr)[threadIdx.x * 2 + 1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
    float mean = 0.f;
    if (layernorm) {
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) s += v[j];
        mean = block_sum(s, red) / 2048.f;
    }
    float q = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) { float d = v[j] - mean; q += d * d; }
    float r = rsqrtf(block_sum(q, red) / 2048.f + eps);
    uint4 o;
    f162* oh = reinterpret_cast<f162*>(&o);
    const int c0 = threadIdx.x * 8;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float y0 = (v[2 * j] - mean) * r * h2f(w[c0 + 2 * j]);
        float y1 = (v[2 * j + 1] - mean) * r * h2f(w[c0 + 2 * j + 1]);
        if (b) { y0 += h2f(b[c0 + 2 * j]); y1 += h2f(b[c0 + 2 * j + 1]); }
        oh[j] = __floats2half2_rn(y0, y1);
    }
    *reinterpret_cast<uint4*>(out + (int64_t)row * 2048 + c0) = o;
}

// In place on qkv fp16 [M, 32, 192] (q | k | v per head): RMSNorm (no affine) of q and k over 64 dims, then
// split-half rope on the first 48 dims with rope[(m % S)] = (cos, sin) x 24 pairs (f32). One warp per (token, head).
extern "C" __global__ void k_vv_qk_rope(f16* __restrict__ qkv, int M, int S, const float* __restrict__ rope, float eps) {
    __shared__ float buf[8][64];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t gw = (int64_t)blockIdx.x * 8 + warp;
    if (gw >= (int64_t)M * 32) return;
    const int m = (int)(gw / 32), h = (int)(gw % 32);
    const float* rp = rope + (int64_t)(m % S) * 48;
    for (int which = 0; which < 2; ++which) {
        f16* p = qkv + (int64_t)m * 6144 + h * 192 + which * 64;
        float a = h2f(p[lane]), b = h2f(p[lane + 32]);
        float ss = warp_sum(a * a + b * b);
        float r = rsqrtf(ss / 64.f + eps);
        a = rh(a * r);
        b = rh(b * r);
        buf[warp][lane] = a;
        buf[warp][lane + 32] = b;
        __syncwarp();
        // dims d < 24 pair with d + 24 (pair index d); dims 48..63 pass through
        float o0, o1;
        {
            int d = lane;  // 0..31
            if (d < 24) {
                float c = rp[d * 2], s = rp[d * 2 + 1];
                o0 = buf[warp][d] * c - buf[warp][d + 24] * s;
            } else {
                int pi = d - 24;  // 24..31 -> pair 0..7, second element
                float c = rp[pi * 2], s = rp[pi * 2 + 1];
                o0 = buf[warp][d] * c + buf[warp][pi] * s;
            }
            int d2 = lane + 32;  // 32..63
            if (d2 < 48) {
                int pi = d2 - 24;  // 8..23
                float c = rp[pi * 2], s = rp[pi * 2 + 1];
                o1 = buf[warp][d2] * c + buf[warp][pi] * s;
            } else {
                o1 = buf[warp][d2];
            }
        }
        __syncwarp();
        p[lane] = f2h(o0);
        p[lane + 32] = f2h(o1);
    }
}

// ---------------------------------------------------------------------------------------------
// Flash attention, fp16, head dim 64, non-causal, batched independent sequences.
//   qkv fp16 [B*S, 32, 192] (q | k | v per head); out fp16 [B*S, 2048] (head h at cols h*64).
// grid (ceil(S/64), 32 heads, B), 128 threads (4 warps x 16 query rows).
namespace {
DEVI int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
}
extern "C" __global__ void __launch_bounds__(128) k_vv_flash_d64(const f16* __restrict__ qkv, f16* __restrict__ out, int S, float scale_log2) {
    __shared__ __align__(128) uint8_t sQ[64 * 128];
    __shared__ __align__(128) uint8_t sK[2][64 * 128];
    __shared__ __align__(128) uint8_t sV[2][64 * 128];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * 64, h = blockIdx.y, b = blockIdx.z;
    const f16* base = qkv + (int64_t)b * S * 6144 + h * 192;
    // load Q tile (64 rows x 8 chunks): 512 chunks, 4 per thread
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int id = tid + i * 128, row = id >> 3, ch = id & 7;
        int q = q0 + row;
        bool p = q < S;
        cp_async_16(smem_u32(sQ + swz128(row, ch)), base + (int64_t)(p ? q : 0) * 6144 + ch * 8, p);
    }
    auto load_kv = [&](int kv0, int st) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int id = tid + i * 128, row = id >> 3, ch = id & 7;
            int kk = kv0 + row;
            bool p = kk < S;
            const f16* src = base + (int64_t)(p ? kk : 0) * 6144 + ch * 8;
            cp_async_16(smem_u32(sK[st] + swz128(row, ch)), src + 64, p);
            cp_async_16(smem_u32(sV[st] + swz128(row, ch)), src + 128, p);
        }
    };
    const int nkv = (S + 63) / 64;
    load_kv(0, 0);
    cp_async_commit();
    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    uint32_t qf[4][4];
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[i][j] = 0.f;
    float mrow[2] = {-INFINITY, -INFINITY}, lrow_[2] = {0.f, 0.f};

    for (int it = 0; it < nkv; ++it) {
        const int st = it & 1;
        if (it + 1 < nkv) load_kv((it + 1) * 64, st ^ 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        if (it == 0) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                int row = warp * 16 + (lidx & 1) * 8 + lrow;
                int ch = kk * 2 + (lidx >> 1);
                ldmatrix_x4(qf[kk][0], qf[kk][1], qf[kk][2], qf[kk][3], smem_u32(sQ + swz128(row, ch)));
            }
        }
        // S = Q K^T  (16 x 64 per warp)
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) s[i][j] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int nj = 0; nj < 8; nj += 2) {
                uint32_t b0[2], b1[2];
                int row = (nj + (lidx >> 1)) * 8 + lrow;
                int ch = kk * 2 + (lidx & 1);
                ldmatrix_x4(b0[0], b0[1], b1[0], b1[1], smem_u32(sK[st] + swz128(row, ch)));
                mma_f16_16816(s[nj], qf[kk], b0);
                mma_f16_16816(s[nj + 1], qf[kk], b1);
            }
        }
        const int kv0 = it * 64;
        // mask + online softmax (rows g and g+8)
        float mx[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                int key = kv0 + nj * 8 + t4 * 2 + (e & 1);
                float v = key < S ? s[nj][e] * scale_log2 : -INFINITY;
                s[nj][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffff, mx[r], 2));
            float mnew = fmaxf(mrow[r], mx[r]);
            alpha[r] = exp2f(mrow[r] - mnew);
            mrow[r] = mnew;
        }
        float ls[2] = {0.f, 0.f};
        uint32_t pf[4][4];
#pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
            float p0 = exp2f(s[nj][0] - mrow[0]);
            float p1 = exp2f(s[nj][1] - mrow[0]);
            float p2 = exp2f(s[nj][2] - mrow[1]);
            float p3 = exp2f(s[nj][3] - mrow[1]);
            ls[0] += p0 + p1;
            ls[1] += p2 + p3;
            int kk = nj >> 1, hi = nj & 1;
            pf[kk][hi * 2 + 0] = pack_h2(p0, p1);
            pf[kk][hi * 2 + 1] = pack_h2(p2, p3);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) lrow_[r] = lrow_[r] * alpha[r] + ls[r];
#pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            o[nd][0] *= alpha[0]; o[nd][1] *= alpha[0];
            o[nd][2] *= alpha[1]; o[nd][3] *= alpha[1];
        }
        // O += P V
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int nd = 0; nd < 8; nd += 2) {
                uint32_t b0[2], b1[2];
                int row = kk * 16 + (lidx & 1) * 8 + lrow;
                int ch = nd + (lidx >> 1);
                ldmatrix_x4_trans(b0[0], b0[1], b1[0], b1[1], smem_u32(sV[st] + swz128(row, ch)));
                mma_f16_16816(o[nd], pf[kk], b0);
                mma_f16_16816(o[nd + 1], pf[kk], b1);
            }
        }
        __syncthreads();
    }
    // finalize
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float l = lrow_[r];
        l += __shfl_xor_sync(0xffffffff, l, 1);
        l += __shfl_xor_sync(0xffffffff, l, 2);
        lrow_[r] = 1.f / l;
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        int q = q0 + warp * 16 + g + r * 8;
        if (q >= S) continue;
        f16* op = out + ((int64_t)b * S + q) * 2048 + h * 64;
#pragma unroll
        for (int nd = 0; nd < 8; ++nd) {
            *reinterpret_cast<f162*>(op + nd * 8 + t4 * 2) = __floats2half2_rn(o[nd][r * 2] * lrow_[r], o[nd][r * 2 + 1] * lrow_[r]);
        }
    }
}

// Place one decoded tile into the f32 canvas [F, H, W, 3] with the reference's blending:
//   v = tile (unpatchified from proj f32 [S, 3072] rows of this tile; F = 4*tc frames)
//   rows r < oy (i > 0): v = strip[f, r, X] * (1 - r/oy) + v * (r/oy)       strip f32 [F, oy_max, W, 3]
//   cols c < ox (j > 0): v = canvas[f, Y, X] * (1 - c/ox) + v * (c/ox)
extern "C" __global__ void k_vv_place_tile(const float* __restrict__ proj, int th, int tw, int F, float* __restrict__ canvas, int H,
                                           int W, int Y0, int X0, const float* __restrict__ strip, int oy_max, int oy, int ox) {
    const int TH = th * 16, TW = tw * 16;
    int64_t n = (int64_t)F * TH * TW;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c = (int)(i % TW);
    int64_t rr = i / TW;
    int r = (int)(rr % TH);
    int f = (int)(rr / TH);
    int tok = ((f >> 2) * th + (r >> 4)) * tw + (c >> 4);
    int col = (f & 3) * 256 + (r & 15) * 16 + (c & 15);
    const float* pr = proj + (int64_t)tok * 3072 + col;
    float v[3] = {pr[0], pr[1024], pr[2048]};
    const int Y = Y0 + r, X = X0 + c;
    if (r < oy) {
        float wb = (float)r / (float)oy, wa = 1.f - wb;
        const float* sp = strip + (((int64_t)f * oy_max + r) * W + X) * 3;
#pragma unroll
        for (int k = 0; k < 3; ++k) v[k] = sp[k] * wa + v[k] * wb;
    }
    float* cp = canvas + (((int64_t)f * H + Y) * W + X) * 3;
    if (c < ox) {
        float wb = (float)c / (float)ox, wa = 1.f - wb;
#pragma unroll
        for (int k = 0; k < 3; ++k) v[k] = cp[k] * wa + v[k] * wb;
    }
#pragma unroll
    for (int k = 0; k < 3; ++k) cp[k] = v[k];
}

// Copy canvas rows [y0, y0+rows) of every frame into strip [F, oy_max, W, 3].
extern "C" __global__ void k_vv_copy_strip(const float* __restrict__ canvas, int F, int H, int W, int y0, int rows, float* __restrict__ strip,
                                           int oy_max) {
    int64_t n = (int64_t)F * rows * W * 3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t per = (int64_t)rows * W * 3;
    int f = (int)(i / per);
    int64_t rem = i - f * per;
    strip[(int64_t)f * oy_max * W * 3 + rem] = canvas[((int64_t)f * H + y0) * W * 3 + rem];
}

// Temporal blend + finalize: out u8 [n, HW, 3] from canvas frames [src0, src0+n). The first `e` frames blend with
// ovl frames [ovl_len - e, ovl_len): v = ovl*(1 - f/e) + v*(f/e). Pixel = clamp(v*std + mean, 0, 1) -> round(x*255).
extern "C" __global__ void k_vv_finalize(const float* __restrict__ canvas, int src0, int n, int64_t hw, const float* __restrict__ ovl, int ovl_len,
                                         int e, const float* __restrict__ pmean, const float* __restrict__ pstd, uint8_t* __restrict__ out) {
    int64_t total = (int64_t)n * hw * 3;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int f = (int)(i / (hw * 3));
    int64_t rem = i - (int64_t)f * hw * 3;
    int ch = (int)(rem % 3);
    float v = canvas[(int64_t)(src0 + f) * hw * 3 + rem];
    if (f < e) {
        float wb = (float)f / (float)e;
        float a = ovl[(int64_t)(ovl_len - e + f) * hw * 3 + rem];
        v = a * (1.f - wb) + v * wb;
    }
    v = v * pstd[ch] + pmean[ch];
    v = fminf(fmaxf(v, 0.f), 1.f);
    out[i] = (uint8_t)__float2int_rn(v * 255.f);
}
