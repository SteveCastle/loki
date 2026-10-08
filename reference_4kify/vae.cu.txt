// VAE kernels: implicit-GEMM 2D convolution on channels-last (NHWC) bf16 planes, channel RMS norm,
// shortcut ops (DupUp3D / AvgDown3D), softmax rows, transpose, and pixel conversion.
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

extern "C" {

// Channel RMS norm (Wan VAE RMS_norm): y = x / max(||x||_2, 1e-12) * sqrt(C) * gamma[c], optional SiLU.
// One warp per pixel; C % 8 == 0 (C <= 8192).
__global__ void k_vae_rmsnorm(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t npix, int C,
                              const bf16* __restrict__ gamma, int silu) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= npix) return;
    const bf16* px = x + gw * C;
    bf16* py = y + gw * C;
    float ss = 0.f;
    for (int c = lane * 8; c < C; c += 256) {
        uint4 v = *reinterpret_cast<const uint4*>(px + c);
        const bf162* h = reinterpret_cast<const bf162*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) { float a = __low2float(h[j]), b = __high2float(h[j]); ss += a * a + b * b; }
    }
    ss = warp_sum(ss);
    float nrm = sqrtf(ss);
    float scale = sqrtf((float)C) / fmaxf(nrm, 1e-12f);
    for (int c = lane * 8; c < C; c += 256) {
        uint4 v = *reinterpret_cast<const uint4*>(px + c);
        uint4 gv = *reinterpret_cast<const uint4*>(gamma + c);
        const bf162* h = reinterpret_cast<const bf162*>(&v);
        const bf162* gh = reinterpret_cast<const bf162*>(&gv);
        uint4 o;
        bf162* oh = reinterpret_cast<bf162*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = __low2float(h[j]) * scale * __low2float(gh[j]);
            float b = __high2float(h[j]) * scale * __high2float(gh[j]);
            if (silu) { a = silu_f(a); b = silu_f(b); }
            oh[j] = __floats2bfloat162_rn(a, b);
        }
        *reinterpret_cast<uint4*>(py + c) = o;
    }
}

// DupUp3D shortcut add: out[(2h+sh, 2w+sw), o] += in[(h, w), src(o, sh)]
//   mode 0: src = o ; mode 1: src = 2o+1 ; mode 2: src = 2o+sh
// out plane: (OH=2*IH, OW=2*IW, Co); in plane: (IH, IW, Ci)
__global__ void k_dupup_add(bf16* __restrict__ out, const bf16* __restrict__ in, int IH, int IW, int Ci, int Co, int mode) {
    int64_t OH = 2 * IH, OW = 2 * IW;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= OH * OW * Co) return;
    int o = i % Co;
    int64_t p = i / Co;
    int ox = p % OW, oy = p / OW;
    int sh = oy & 1;
    int src = (mode == 0) ? o : (mode == 1) ? (2 * o + 1) : (2 * o + sh);
    float v = bf2f(in[((int64_t)(oy >> 1) * IW + (ox >> 1)) * Ci + src]);
    out[i] = f2bf(bf2f(out[i]) + v);
}

// AvgDown3D shortcut add: out[(h, w), o] += mean_{2x2}(in[(2h+dy, 2w+dx), c]) with
//   mode 0: c = o (always) ; mode 1: only odd o get c = o/2 (even o add nothing, zero frame)
//   mode 2: identity add (same resolution, same channels)
__global__ void k_avgdown_add(bf16* __restrict__ out, const bf16* __restrict__ in, int OH, int OW, int Co, int Ci, int mode) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)OH * OW * Co) return;
    int o = i % Co;
    int64_t p = i / Co;
    int ox = p % OW, oy = p / OW;
    float v = 0.f;
    if (mode == 2) {
        v = bf2f(in[p * Ci + o]);
    } else {
        int c = -1;
        if (mode == 0) c = o;
        else if (o & 1) c = o >> 1;
        if (c >= 0) {
            int IW = OW * 2;
            int64_t base = ((int64_t)(2 * oy) * IW + 2 * ox) * Ci + c;
            v = 0.25f * (bf2f(in[base]) + bf2f(in[base + Ci]) + bf2f(in[base + (int64_t)IW * Ci]) + bf2f(in[base + (int64_t)IW * Ci + Ci]));
        }
    }
    out[i] = f2bf(bf2f(out[i]) + v);
}

// Row softmax over the first n of ld columns: s f32 [rows, ld] * scale -> p bf16 [rows, ld] (padding -> 0).
__global__ void k_softmax_rows(const float* __restrict__ s, bf16* __restrict__ p, int n, int ld, float scale) {
    __shared__ float red[32];
    int64_t r = blockIdx.x;
    const float* row = s + r * ld;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < n; i += 256) mx = fmaxf(mx, row[i] * scale);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += 256) sum += __expf(row[i] * scale - mx);
    sum = block_sum(sum, red);
    float inv = 1.f / sum;
    for (int i = threadIdx.x; i < ld; i += 256) p[r * ld + i] = (i < n) ? f2bf(__expf(row[i] * scale - mx) * inv) : f2bf(0.f);
}

// Transpose bf16 [R, C] -> [C, R] via 32x32 tiles.
__global__ void k_transpose_bf16(const bf16* __restrict__ in, bf16* __restrict__ out, int R, int C) {
    __shared__ bf16 tile[32][33];
    int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    int tx = threadIdx.x, ty = threadIdx.y;  // 32 x 8
    for (int j = ty; j < 32; j += 8) {
        int r = r0 + j, c = c0 + tx;
        if (r < R && c < C) tile[j][tx] = in[(int64_t)r * C + c];
    }
    __syncthreads();
    for (int j = ty; j < 32; j += 8) {
        int c = c0 + j, r = r0 + tx;
        if (r < R && c < C) out[(int64_t)c * R + r] = tile[tx][j];
    }
}

// Slice channels: out[p, 0..Cout) = in[p, c0..c0+Cout)
__global__ void k_slice_channels(const bf16* __restrict__ in, bf16* __restrict__ out, int64_t npix, int Cin, int c0, int Cout) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix * Cout) return;
    int64_t p = i / Cout; int c = i % Cout;
    out[i] = in[p * Cin + c0 + c];
}

// RGB f32 [H,W,3] in [0,1] -> NHWC bf16 [H,W,8]: (2r-1, 2g-1, 2b-1, 1, 0, 0, 0, 0)
__global__ void k_rgb_to_vae_in(const float* __restrict__ rgb, bf16* __restrict__ out, int64_t npix) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    float r = rgb[i * 3], g = rgb[i * 3 + 1], b = rgb[i * 3 + 2];
    bf16* o = out + i * 8;
    o[0] = f2bf(2.f * r - 1.f); o[1] = f2bf(2.f * g - 1.f); o[2] = f2bf(2.f * b - 1.f); o[3] = f2bf(1.f);
    o[4] = f2bf(0.f); o[5] = f2bf(0.f); o[6] = f2bf(0.f); o[7] = f2bf(0.f);
}

// VAE output (NHWC bf16 [H,W,Cpad], RGBA in the first 4) -> RGB u8 [H,W,3]: (x+1)/2 clamped
__global__ void k_vae_out_to_rgb8(const bf16* __restrict__ x, uint8_t* __restrict__ out, int64_t npix, int Cpad) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
#pragma unroll
    for (int c = 0; c < 3; ++c) {
        float v = (bf2f(x[i * Cpad + c]) + 1.f) * 0.5f;
        v = fminf(fmaxf(v, 0.f), 1.f);
        out[i * 3 + c] = (uint8_t)(v * 255.f + 0.5f);
    }
}

// Latent normalization on token-major [N, 64]: y = (x - mean[c]) / std[c]  (f32 in, bf16 out)  [process_in]
__global__ void k_latent_norm_in(const float* __restrict__ x, bf16* __restrict__ y, int64_t n, const float* __restrict__ mean, const float* __restrict__ std) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 64) return;
    int c = i % 64;
    y[i] = f2bf((x[i] - mean[c]) / std[c]);
}
// y = x * std[c] + mean[c]  (f32 in, bf16 out)  [process_out]
__global__ void k_latent_norm_out(const float* __restrict__ x, bf16* __restrict__ y, int64_t n, const float* __restrict__ mean, const float* __restrict__ std) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 64) return;
    int c = i % 64;
    y[i] = f2bf(x[i] * std[c] + mean[c]);
}
// Euler step: x[i] += alpha * v[i]  (x f32, v bf16)
__global__ void k_axpy_f32_bf16(float* __restrict__ x, const bf16* __restrict__ v, float alpha, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += alpha * bf2f(v[i]);
}

// bf16 [C, H, W] -> token-major f32 [H*W, C]  (CHW to HWC) and back
__global__ void k_chw_to_hwc_f32(const float* __restrict__ in, float* __restrict__ out, int C, int64_t hw) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hw * C) return;
    int64_t p = i / C; int c = i % C;
    out[i] = in[(int64_t)c * hw + p];
}
__global__ void k_hwc_to_chw_f32(const float* __restrict__ in, float* __restrict__ out, int C, int64_t hw) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hw * C) return;
    int64_t c = i / hw; int64_t p = i % hw;
    out[i] = in[p * C + c];
}

}  // extern "C"
