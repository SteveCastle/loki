// MiniMax H3 latent upscaler (LBH-123-AI `LatentResizer3D`) helper kernels.
//
// The network runs on fp16 channels-last volumes [T, H, W, C] like the video VAE encoder and reuses its implicit-GEMM
// conv3d (k_vv_conv3d*) and GroupNorm statistics (k_vv_gn_stats / k_vv_gn_finalize over the whole volume as one
// "frame"). These kernels add what the VAE does not have: the f32 channels-first latent <-> fp16 channels-last
// conversions with the replicate-padded temporal chunking, GroupNorm(32)+SiLU with zero padding on every side
// (symmetric in time), the bilinear (trilinear with T unchanged) resize, and the chunk blend.
#include "common.cuh"

typedef __half f16;
typedef __half2 f162;

DEVI float rh(float x) { return __half2float(__float2half_rn(x)); }

// z f32 [C, T, H, W] -> out fp16 [n, H, W, C] = (z - mean[c]) / std[c]: frame i of the output is frame
// clamp(t0 + i, 0, T-1) of z (replicate padding when t0 < 0 or t0 + n > T).
extern "C" __global__ void k_lu_to_cl(const float* __restrict__ z, int C, int T, int H, int W, int t0, int n, const float* __restrict__ mean,
                                      const float* __restrict__ stdv, f16* __restrict__ out) {
    int64_t total = (int64_t)n * H * W * C;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)(i % C);
    int64_t r = i / C;
    int x = (int)(r % W);
    r /= W;
    int y = (int)(r % H);
    int f = (int)(r / H);
    int t = min(max(t0 + f, 0), T - 1);
    out[i] = __float2half_rn((z[(((int64_t)c * T + t) * H + y) * W + x] - mean[c]) / stdv[c]);
}

// GroupNorm(32) (+SiLU) with zero padding: x fp16 [T, H, W, C] -> out [T+2pt, H+2ps, W+2ps, C].
// mr: (mean, rstd) f32 [32, 2] over the whole volume; gamma/beta f32 [C] (the block's modulation already folded in).
// mr == nullptr: pad only.
extern "C" __global__ void k_lu_gn_pad(const f16* __restrict__ x, int T, int H, int W, int C, const float* __restrict__ mr,
                                       const float* __restrict__ gamma, const float* __restrict__ beta, int pt, int ps,
                                       f16* __restrict__ out) {
    const int Tp = T + 2 * pt, Hp = H + 2 * ps, Wp = W + 2 * ps, nch = C >> 3;
    int64_t n = (int64_t)Tp * Hp * Wp * nch;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c8 = (int)(i % nch);
    int64_t r = i / nch;
    int xp = (int)(r % Wp);
    r /= Wp;
    int yp = (int)(r % Hp);
    int tp = (int)(r / Hp);
    int t = tp - pt, y = yp - ps, xx = xp - ps;
    uint4 o = make_uint4(0, 0, 0, 0);
    if (t >= 0 && t < T && y >= 0 && y < H && xx >= 0 && xx < W) {
        o = *reinterpret_cast<const uint4*>(x + (((int64_t)t * H + y) * W + xx) * C + c8 * 8);
        if (mr) {
            f162* h = reinterpret_cast<f162*>(&o);
            const int cg = C / 32;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __half22float2(h[j]);
                int c = c8 * 8 + 2 * j;
                int g0 = c / cg, g1 = (c + 1) / cg;
                // torch fp16: group_norm output (affine) is fp16, then the modulation; folded into one affine here
                float a = (f.x - mr[g0 * 2]) * mr[g0 * 2 + 1] * gamma[c] + beta[c];
                float b = (f.y - mr[g1 * 2]) * mr[g1 * 2 + 1] * gamma[c + 1] + beta[c + 1];
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

// Bilinear resize of every frame (torch trilinear, align_corners=False, T unchanged): x fp16 [T, H, W, C] ->
// out [T, Ho, Wo, C]. C % 8 == 0.
extern "C" __global__ void k_lu_resize(const f16* __restrict__ x, int T, int H, int W, int C, int Ho, int Wo, f16* __restrict__ out) {
    const int nch = C >> 3;
    int64_t n = (int64_t)T * Ho * Wo * nch;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int c8 = (int)(i % nch);
    int64_t r = i / nch;
    int xo = (int)(r % Wo);
    r /= Wo;
    int yo = (int)(r % Ho);
    int t = (int)(r / Ho);
    // area_pixel_compute_source_index (align_corners=False): src = scale * (dst + 0.5) - 0.5, clamped at 0
    float sy = fmaxf((float)H / (float)Ho * (yo + 0.5f) - 0.5f, 0.f);
    float sx = fmaxf((float)W / (float)Wo * (xo + 0.5f) - 0.5f, 0.f);
    int y0 = min((int)sy, H - 1), x0 = min((int)sx, W - 1);
    int y1 = min(y0 + 1, H - 1), x1 = min(x0 + 1, W - 1);
    float ly = sy - y0, lx = sx - x0;
    float w00 = (1.f - ly) * (1.f - lx), w01 = (1.f - ly) * lx, w10 = ly * (1.f - lx), w11 = ly * lx;
    const f16* base = x + (int64_t)t * H * W * C + c8 * 8;
    uint4 u00 = *reinterpret_cast<const uint4*>(base + ((int64_t)y0 * W + x0) * C);
    uint4 u01 = *reinterpret_cast<const uint4*>(base + ((int64_t)y0 * W + x1) * C);
    uint4 u10 = *reinterpret_cast<const uint4*>(base + ((int64_t)y1 * W + x0) * C);
    uint4 u11 = *reinterpret_cast<const uint4*>(base + ((int64_t)y1 * W + x1) * C);
    const f162* a = reinterpret_cast<const f162*>(&u00);
    const f162* b = reinterpret_cast<const f162*>(&u01);
    const f162* c = reinterpret_cast<const f162*>(&u10);
    const f162* d = reinterpret_cast<const f162*>(&u11);
    uint4 o;
    f162* h = reinterpret_cast<f162*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 fa = __half22float2(a[j]), fb = __half22float2(b[j]), fc = __half22float2(c[j]), fd = __half22float2(d[j]);
        h[j] = __floats2half2_rn(w00 * fa.x + w01 * fb.x + w10 * fc.x + w11 * fd.x, w00 * fa.y + w01 * fb.y + w10 * fc.y + w11 * fd.y);
    }
    *reinterpret_cast<uint4*>(out + i * 8) = o;
}

// Blend one chunk into the full output: seg fp16 [n_seg, H, W, C] channels-last; frames s0 .. s0+n of it land on frames
// t0 .. t0+n of acc f32 [C, T, H, W] as (v * std[c] + mean[c]), weighted by w[0..n) (the weights of a frame sum to 1
// after k_lu_div_t, so de-normalizing before the blend is exact).
extern "C" __global__ void k_lu_blend(const f16* __restrict__ seg, int C, int H, int W, int s0, int n, const float* __restrict__ w,
                                      int t0, int T, const float* __restrict__ mean, const float* __restrict__ stdv, float* __restrict__ acc) {
    int64_t total = (int64_t)n * H * W * C;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int c = (int)(i % C);
    int64_t r = i / C;
    int x = (int)(r % W);
    r /= W;
    int y = (int)(r % H);
    int f = (int)(r / H);
    float v = __half2float(seg[(((int64_t)(s0 + f) * H + y) * W + x) * C + c]) * stdv[c] + mean[c];
    acc[(((int64_t)c * T + t0 + f) * H + y) * W + x] += v * w[f];
}

// acc f32 [C, T, H, W] /= wsum[t]
extern "C" __global__ void k_lu_div_t(float* __restrict__ acc, int C, int T, int HW, const float* __restrict__ wsum) {
    int64_t total = (int64_t)C * T * HW;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    int t = (int)((i / HW) % T);
    acc[i] = acc[i] / fmaxf(wsum[t], 1e-8f);
}
