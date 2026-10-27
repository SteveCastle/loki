// Implicit-GEMM 2D convolution on channels-last (NHWC) bf16 planes (from loki-retouch's vae.cu).

#include "common.cuh"

namespace {

// ---------------------------------------------------------------------------------------------
// Implicit GEMM conv:  out[p, co] = bias[co] + sum_{ky,kx,ci} in[iy, ix, ci] * w[co][(ky*KW+kx)*Ci + ci]
//   p = (oy, ox) over OH x OW; iy = oy*stride + ky - pad_t; ix = ox*stride + kx - pad_l (zero outside)
//   with `up2`: the input is read as its 2x nearest upsample: in[(iy>>1), (ix>>1)] over an (2IH x 2IW) grid.
// Weights [Co][Kpad] with Kpad a multiple of 32 (zero padded). Ci % 8 == 0.
// mode: 0 store, 1 out += res[p, co] (res has Co channels, same geometry as out)
// Tile 128(pixels) x 128(co) x 32(k), 4-stage cp.async, same warp layout as gemm_bf16.
constexpr int BM = 128, BN = 128, BK = 32, STAGES = 4, THREADS = 256;
constexpr int TILE_BYTES = BM * BK * 2;

struct ConvParams {
    const bf16* in; int IH; int IW; int Ci;       // input plane (logical size before upsampling)
    const bf16* w; const float* bias; int Co; int Kpad;
    bf16* out; int OH; int OW;
    int KH; int KW; int stride; int pad_t; int pad_l; int up2;
    int mode; const bf16* res;
};

}  // namespace

extern "C" __global__ void __launch_bounds__(256) k_conv2d_nhwc(ConvParams P) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * TILE_BYTES;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2, wn = warp & 3;
    const int M = P.OH * P.OW, N = P.Co, K = P.KH * P.KW * P.Ci;
    const int n_tiles = (N + BN - 1) / BN, m_tiles = (M + BM - 1) / BM;
    constexpr int GROUP_M = 8;
    const int bid = blockIdx.x;
    const int group = bid / (GROUP_M * n_tiles);
    const int first_m = group * GROUP_M;
    const int gsize = min(GROUP_M, m_tiles - first_m);
    const int in_group = bid - group * GROUP_M * n_tiles;
    const int mt = first_m + (in_group % gsize);
    const int nt = in_group / gsize;
    const int m0 = mt * BM, n0 = nt * BN;

    // per-thread copy assignment: 2 chunks per tile
    int c_row[2], c_chunk[2];
    int a_oy[2], a_ox[2];
    bool a_pred[2];
    const bf16* b_src[2];
    bool b_pred[2];
    const int eff_IH = P.up2 ? P.IH * 2 : P.IH;
    const int eff_IW = P.up2 ? P.IW * 2 : P.IW;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int id = tid + i * THREADS;
        c_row[i] = id >> 2;
        c_chunk[i] = id & 3;
        int m = m0 + c_row[i];
        a_pred[i] = m < M;
        int mm = a_pred[i] ? m : 0;
        a_oy[i] = mm / P.OW;
        a_ox[i] = mm % P.OW;
        int n = n0 + c_row[i];
        b_pred[i] = n < N;
        b_src[i] = P.w + (int64_t)(b_pred[i] ? n : 0) * P.Kpad + c_chunk[i] * 8;
    }
    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * TILE_BYTES;
        uint8_t* bs = Bs + stage * TILE_BYTES;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int k = k0 + c_chunk[i] * 8;
            bool kp = k < K;
            // A: decompose k -> (tap, ci)
            int tap = k / P.Ci, ci = k - tap * P.Ci;
            int ky = tap / P.KW, kx = tap - ky * P.KW;
            int iy = a_oy[i] * P.stride + ky - P.pad_t;
            int ix = a_ox[i] * P.stride + kx - P.pad_l;
            bool inb = kp && a_pred[i] && iy >= 0 && iy < eff_IH && ix >= 0 && ix < eff_IW;
            if (P.up2) { iy >>= 1; ix >>= 1; }
            const bf16* src = inb ? (P.in + ((int64_t)iy * P.IW + ix) * P.Ci + ci) : P.in;
            cp_async_16(smem_u32(as + swz64(c_row[i], c_chunk[i])), src, inb);
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
            uint32_t afrag[4][4], bfrag[4][2];
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
                float v0 = acc[mi][ni][half * 2 + 0], v1 = acc[mi][ni][half * 2 + 1];
                if (P.bias) { v0 += P.bias[n]; v1 += P.bias[n + 1]; }
                int64_t off = (int64_t)m * N + n;
                if (P.mode == 1) {
                    bf162 r = *reinterpret_cast<const bf162*>(P.res + off);
                    v0 += __low2float(r); v1 += __high2float(r);
                }
                *reinterpret_cast<bf162*>(P.out + off) = __floats2bfloat162_rn(v0, v1);
            }
        }
    }
}
