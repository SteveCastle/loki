// W8A8 int8 tensor-core GEMM with ConvRot-friendly epilogue.
//
//   C[M,N] = sum_k A[M,K] * B[N,K]        (int8 x int8 -> int32)
//   v      = acc * sa[m] * sb[n] (+ bias[n])
//   mode 0: out = v
//   mode 1: out = res + v
//   mode 2: out = res + v * gate[n]
//   mode 3: out[m][n/2] = silu(v[n]) * v[n+1] for even n (interleaved gate/up rows; output width N/2)
//
// A row-major [M,K] int8, B row-major [N,K] int8 (= weight as stored), K % 64 == 0, N % 8 == 0.
// Tile 128x128x64, 4-stage cp.async pipeline, 8 warps (2 x 4), warp tile 64x32, mma.m16n8k32.s8.
#include "common.cuh"

namespace {

constexpr int BM = 128, BK = 64;
constexpr int THREADS = 256;

template <typename OutT>
__device__ __forceinline__ void store2(OutT* p, float a, float b);
template <>
__device__ __forceinline__ void store2<bf16>(bf16* p, float a, float b) {
    *reinterpret_cast<bf162*>(p) = __floats2bfloat162_rn(a, b);
}
template <>
__device__ __forceinline__ void store2<float>(float* p, float a, float b) {
    *reinterpret_cast<float2*>(p) = make_float2(a, b);
}
template <typename T>
__device__ __forceinline__ void load2(const T* p, float& a, float& b);
template <>
__device__ __forceinline__ void load2<bf16>(const bf16* p, float& a, float& b) {
    bf162 v = *reinterpret_cast<const bf162*>(p);
    a = __low2float(v); b = __high2float(v);
}
template <>
__device__ __forceinline__ void load2<float>(const float* p, float& a, float& b) {
    float2 v = *reinterpret_cast<const float2*>(p);
    a = v.x; b = v.y;
}

template <typename OutT, int BN, int STAGES>
__device__ void gemm_i8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
                               int M, int N, int K, const float* __restrict__ sa, const float* __restrict__ sb,
                               const float* __restrict__ bias, int mode, const OutT* __restrict__ res,
                               const float* __restrict__ gate) {
    constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
    constexpr int WN = BN / 4;           // warp tile width (32 or 64)
    constexpr int NI = WN / 8;           // n-frags per warp (4 or 8)
    constexpr int ACH = A_BYTES / 16 / THREADS;  // A chunks per thread (2)
    constexpr int BCH = B_BYTES / 16 / THREADS;  // B chunks per thread (2 or 4)
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * A_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2;  // 0..1
    const int wn = warp & 3;   // 0..3
    // grouped rasterization: within a group of GROUP_M m-tiles, n varies fastest, so the
    // concurrently resident blocks share both their A rows and B rows through L2.
    const int n_tiles = (N + BN - 1) / BN;
    const int m_tiles = (M + BM - 1) / BM;
    constexpr int GROUP_M = 16;
    const int bid = blockIdx.x;
    const int group = bid / (GROUP_M * n_tiles);
    const int first_m = group * GROUP_M;
    const int gsize = min(GROUP_M, m_tiles - first_m);
    const int in_group = bid - group * GROUP_M * n_tiles;
    const int mt = first_m + (in_group % gsize);
    const int nt = in_group / gsize;
    const int m0 = mt * BM;
    const int n0 = nt * BN;

    // global -> shared copy assignment
    int a_row[ACH], a_chunk[ACH], b_row[BCH], b_chunk[BCH];
    const int8_t* a_src[ACH];
    const int8_t* b_src[BCH];
    bool a_pred[ACH], b_pred[BCH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
        int id = tid + i * THREADS;
        a_row[i] = id >> 2;
        a_chunk[i] = id & 3;
        int am = m0 + a_row[i];
        a_pred[i] = am < M;
        a_src[i] = A + (int64_t)(a_pred[i] ? am : 0) * K + a_chunk[i] * 16;
    }
#pragma unroll
    for (int i = 0; i < BCH; ++i) {
        int id = tid + i * THREADS;
        b_row[i] = id >> 2;
        b_chunk[i] = id & 3;
        int bn = n0 + b_row[i];
        b_pred[i] = bn < N;
        b_src[i] = B + (int64_t)(b_pred[i] ? bn : 0) * K + b_chunk[i] * 16;
    }

    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * A_BYTES;
        uint8_t* bs = Bs + stage * B_BYTES;
#pragma unroll
        for (int i = 0; i < ACH; ++i) {
            bool kp = (k0 + a_chunk[i] * 16) < K;
            cp_async_16(smem_u32(as + swz64(a_row[i], a_chunk[i])), a_src[i] + (kp ? k0 : 0), a_pred[i] && kp);
        }
#pragma unroll
        for (int i = 0; i < BCH; ++i) {
            bool kp = (k0 + b_chunk[i] * 16) < K;
            cp_async_16(smem_u32(bs + swz64(b_row[i], b_chunk[i])), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };

    int32_t acc[4][NI][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;

    const int KT = (K + BK - 1) / BK;
    // prologue
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    // ldmatrix lane geometry
    const int lidx = lane >> 3;  // matrix index 0..3
    const int lrow = lane & 7;

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk, nk % STAGES);
            cp_async_commit();
        }
        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * A_BYTES;
        const uint8_t* bs = Bs + stage * B_BYTES;
        // software-pipelined fragment loads: fragments for k-step kk+32 are fetched while the
        // mma instructions of k-step kk issue.
        uint32_t afrag[2][4][4];
        uint32_t bfrag[2][NI][2];
        auto load_frags = [&](int kk, int buf) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx >> 1);
                ldmatrix_x4(afrag[buf][mi][0], afrag[buf][mi][1], afrag[buf][mi][2], afrag[buf][mi][3], smem_u32(as + swz64(row, chunk)));
            }
#pragma unroll
            for (int nj = 0; nj < NI; nj += 2) {
                int row = wn * WN + (nj + (lidx >> 1)) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx & 1);
                ldmatrix_x4(bfrag[buf][nj][0], bfrag[buf][nj][1], bfrag[buf][nj + 1][0], bfrag[buf][nj + 1][1], smem_u32(bs + swz64(row, chunk)));
            }
        };
        load_frags(0, 0);
        load_frags(32, 1);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) mma_s8_16832(acc[mi][ni], afrag[0][mi], bfrag[0][ni]);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) mma_s8_16832(acc[mi][ni], afrag[1][mi], bfrag[1][ni]);
    }
    cp_async_wait<0>();

    // epilogue
    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = m0 + wm * 64 + mi * 16 + g + half * 8;
            if (m >= M) continue;
            float rs = sa[m];
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                int n = n0 + wn * WN + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = (float)acc[mi][ni][half * 2 + 0] * rs * sb[n];
                float v1 = (float)acc[mi][ni][half * 2 + 1] * rs * sb[n + 1];
                if (bias) { v0 += bias[n]; v1 += bias[n + 1]; }
                if (mode == 3) {
                    // gather the quad's 4 outputs into lane t4 == 0 and store 8 bytes at once (bf16 out only)
                    float o = silu_f(v0) * v1;
                    float o1 = __shfl_xor_sync(0xffffffff, o, 1);
                    uint32_t pk = pack_bf16x2(o, o1);
                    uint32_t pk2 = __shfl_xor_sync(0xffffffff, pk, 2);
                    if (t4 == 0) {
                        *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(C) + (int64_t)m * (N >> 1) + (n >> 1)) = make_uint2(pk, pk2);
                    }
                    continue;
                }
                int64_t off = (int64_t)m * N + n;
                if (mode == 1) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 += r0; v1 += r1;
                } else if (mode == 2) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 = r0 + v0 * gate[n]; v1 = r1 + v1 * gate[n + 1];
                }
                store2<OutT>(C + off, v0, v1);
            }
        }
    }
}

}  // namespace

extern "C" {
__global__ void __launch_bounds__(256) k_gemm_i8_bf16(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                       const float* sa, const float* sb, const float* bias, int mode,
                                                       const bf16* res, const float* gate) {
    gemm_i8_kernel<bf16, 128, 4>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_i8_f32(const int8_t* A, const int8_t* B, float* C, int M, int N, int K,
                                                      const float* sa, const float* sb, const float* bias, int mode,
                                                      const float* res, const float* gate) {
    gemm_i8_kernel<float, 128, 4>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
// 128x256 tiles, 3 stages (72 KB smem): higher arithmetic intensity for large N
__global__ void __launch_bounds__(256) k_gemm_i8_bf16_w(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                         const float* sa, const float* sb, const float* bias, int mode,
                                                         const bf16* res, const float* gate) {
    gemm_i8_kernel<bf16, 256, 3>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_i8_f32_w(const int8_t* A, const int8_t* B, float* C, int M, int N, int K,
                                                        const float* sa, const float* sb, const float* bias, int mode,
                                                        const float* res, const float* gate) {
    gemm_i8_kernel<float, 256, 3>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
}
