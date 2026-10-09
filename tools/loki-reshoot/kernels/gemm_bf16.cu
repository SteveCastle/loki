// bf16 tensor-core GEMM (f32 accumulate) with fused epilogue.
//
//   C[M,N] = A[M,K] * B[N,K]^T   (A row-major bf16, B row-major bf16 = weight as stored)
//   v = acc (+ bias[n]) ; v = act(v) ; mode 1: v += res ; mode 2: v = res + v*gate[n]
//   act: 0 none, 1 gelu_tanh, 2 silu, 3 gelu_erf
// K % 8 == 0, N % 8 == 0.  Tile 128x128x32, 4-stage cp.async, 8 warps (2x4), warp tile 64x32, mma.m16n8k16.
#include "common.cuh"

namespace {

constexpr int BM = 128, BN = 128, BK = 32, STAGES = 4;
constexpr int THREADS = 256;
constexpr int TILE_BYTES = BM * BK * 2;  // 8192
constexpr int SMEM_BYTES = STAGES * 2 * TILE_BYTES;

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

__device__ __forceinline__ float apply_act(float v, int act) {
    if (act == 1) return gelu_tanh_f(v);
    if (act == 2) return silu_f(v);
    if (act == 3) return gelu_erf_f(v);
    return v;
}

template <typename OutT>
__device__ void gemm_bf16_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B, OutT* __restrict__ C,
                                 int M, int N, int K, const float* __restrict__ bias, int act, int mode,
                                 const OutT* __restrict__ res, const float* __restrict__ gate) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * TILE_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
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
    const int m0 = mt * BM, n0 = nt * BN;

    int c_row[2], c_chunk[2];
    const bf16* a_src[2];
    const bf16* b_src[2];
    bool a_pred[2], b_pred[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int id = tid + i * THREADS;
        c_row[i] = id >> 2;
        c_chunk[i] = id & 3;
        int am = m0 + c_row[i];
        a_pred[i] = am < M;
        a_src[i] = A + (int64_t)(a_pred[i] ? am : 0) * K + c_chunk[i] * 8;
        int bn = n0 + c_row[i];
        b_pred[i] = bn < N;
        b_src[i] = B + (int64_t)(b_pred[i] ? bn : 0) * K + c_chunk[i] * 8;
    }
    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * TILE_BYTES;
        uint8_t* bs = Bs + stage * TILE_BYTES;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            bool kp = (k0 + c_chunk[i] * 8) < K;
            cp_async_16(smem_u32(as + swz64(c_row[i], c_chunk[i])), a_src[i] + (kp ? k0 : 0), a_pred[i] && kp);
            cp_async_16(smem_u32(bs + swz64(c_row[i], c_chunk[i])), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };

    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

    const int KT = (K + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }
    const int lidx = lane >> 3, lrow = lane & 7;

    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk, nk % STAGES);
            cp_async_commit();
        }
        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * TILE_BYTES;
        const uint8_t* bs = Bs + stage * TILE_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t afrag[4][4];
            uint32_t bfrag[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = (kk >> 3) + (lidx >> 1);
                ldmatrix_x4(afrag[mi][0], afrag[mi][1], afrag[mi][2], afrag[mi][3], smem_u32(as + swz64(row, chunk)));
            }
#pragma unroll
            for (int nj = 0; nj < 4; nj += 2) {
                int row = wn * 32 + (nj + (lidx >> 1)) * 8 + lrow;
                int chunk = (kk >> 3) + (lidx & 1);
                ldmatrix_x4(bfrag[nj][0], bfrag[nj][1], bfrag[nj + 1][0], bfrag[nj + 1][1], smem_u32(bs + swz64(row, chunk)));
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_bf16_16816(acc[mi][ni], afrag[mi], bfrag[ni]);
        }
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
                float v0 = acc[mi][ni][half * 2 + 0];
                float v1 = acc[mi][ni][half * 2 + 1];
                if (bias) { v0 += bias[n]; v1 += bias[n + 1]; }
                if (act) { v0 = apply_act(v0, act); v1 = apply_act(v1, act); }
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
__global__ void __launch_bounds__(256) k_gemm_bf16_bf16(const bf16* A, const bf16* B, bf16* C, int M, int N, int K,
                                                         const float* bias, int act, int mode, const bf16* res, const float* gate) {
    gemm_bf16_kernel<bf16>(A, B, C, M, N, K, bias, act, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_bf16_f32(const bf16* A, const bf16* B, float* C, int M, int N, int K,
                                                        const float* bias, int act, int mode, const float* res, const float* gate) {
    gemm_bf16_kernel<float>(A, B, C, M, N, K, bias, act, mode, res, gate);
}
}
