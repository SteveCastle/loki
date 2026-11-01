// Kernels of the Krea 2, FLUX.2 klein and Anima paths (and their shared Qwen3 text encoder and Qwen-Image VAE):
// weight dequantization, norms with the references' rounding, fused q/k norm + rope, modulated norms, small grouped
// attention, latent (un)packing.
//
// Rounding conventions (the references run in bf16):
//   "hf"   (Qwen3RMSNorm / torch RMSNorm): y = bf16(bf16(x * r) * w)
//   "1p"   (Krea2RMSNorm, fp32 weight + 1):  y = bf16(x * r * (1 + w))
//   rope "half" (Qwen3, bf16 tensor ops):    y = bf16(bf16(x*cos) + bf16(rot*sin))
//   rope "half32" / "pairs32" (Cosmos / FLUX-style, computed in fp32): y = bf16(x*cos + rot*sin)
#include "common.cuh"

extern "C" {

// ------------------------------------------------------------------------------------------------ weights
__global__ void k_fp8_to_bf16(const uint8_t* __restrict__ src, float scale, bf16* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __nv_fp8_e4m3 v;
    *reinterpret_cast<uint8_t*>(&v) = src[i];
    dst[i] = f2bf((float)v * scale);
}
// ComfyUI NVFP4: e2m1 values packed two per byte (the first element in the high nibble when hi_first), fp8 e4m3 scales
// per 16 consecutive values of a row in cuBLAS's blocked layout (128-row x 4-block tiles, see comfy/float.py to_blocked),
// and a per-tensor scale g: w = fp4 * block_scale * g. kb_pad = the padded number of 16-wide blocks per row.
__device__ __forceinline__ float e2m1(uint8_t v) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float m = mag[v & 7];
    return (v & 8) ? -m : m;
}
__global__ void k_nvfp4_to_bf16(const uint8_t* __restrict__ src, const uint8_t* __restrict__ scales, int kb_pad, float g,
                                bf16* __restrict__ dst, int N, int K, int hi_first) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * K) return;
    int r = i / K, k = i % K;
    uint8_t b = src[((int64_t)r * K + k) >> 1];
    uint8_t v = ((k & 1) ^ hi_first) ? (b >> 4) : (b & 15);
    int kb = k >> 4;
    int rb = r >> 7, a = (r & 127) >> 5, bb = r & 31, cb = kb >> 2, c = kb & 3;
    int64_t tile = (int64_t)rb * (kb_pad >> 2) + cb;
    __nv_fp8_e4m3 s;
    *reinterpret_cast<uint8_t*>(&s) = scales[((tile * 32 + bb) * 4 + a) * 4 + c];
    dst[i] = f2bf(e2m1(v) * (float)s * g);
}
__global__ void k_f16_to_bf16(const __half* __restrict__ src, bf16* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f2bf(__half2float(src[i]));
}

// ------------------------------------------------------------------------------------------------ norms
// RMSNorm bf16 -> bf16, one block per row. mode 0 "hf": bf16(bf16(x*r)*w); 1 "1p": bf16(x*r*(1+w)); 2: bf16(x*r) (no w)
__global__ void k_rmsnorm_x_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int M, int K, float eps, const bf16* __restrict__ w, int mode) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    const bf16* px = x + m * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = bf2f(px[i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / K + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = bf2f(px[i]) * r;
        if (mode == 0) v = round_bf16(v) * bf2f(w[i]);
        else if (mode == 1) v = v * (1.f + bf2f(w[i]));
        y[m * K + i] = f2bf(v);
    }
}

// Modulated norm: n = norm(x) (bf16), y = bf16(bf16(n * bf16(1 + scale)) + shift)   (diffusers `(1 + scale) * n + shift`)
// norm 0: LayerNorm without affine (eps); 1: RMSNorm "1p" with w. Rows < split use (scale0, shift0), the rest (scale1,
// shift1); scale/shift f32 holding bf16 values. One block per row.
__global__ void k_norm_mod_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int M, int K, float eps, int norm, const bf16* __restrict__ w,
                                int split, const float* __restrict__ scale0, const float* __restrict__ shift0,
                                const float* __restrict__ scale1, const float* __restrict__ shift1) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    const bf16* px = x + m * K;
    const float* sc = m < split ? scale0 : scale1;
    const float* sh = m < split ? shift0 : shift1;
    float mean = 0.f;
    if (norm == 0) {
        float s = 0.f;
        for (int i = threadIdx.x; i < K; i += blockDim.x) s += bf2f(px[i]);
        mean = block_sum(s, red) / K;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = bf2f(px[i]) - mean; ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / K + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float n = (bf2f(px[i]) - mean) * r;
        if (norm == 1) n = n * (1.f + bf2f(w[i]));
        n = round_bf16(n);
        float v = round_bf16(n * round_bf16(1.f + sc[i]));
        y[m * K + i] = f2bf(v + sh[i]);
    }
}

// ------------------------------------------------------------------------------------------------ q/k norm + rope
}  // extern "C"

template <int D>
__device__ __forceinline__ void qk_norm_rope_head(bf16* __restrict__ p, const bf16* __restrict__ w, int lane, float eps, int norm, int rope,
                                                  const float* __restrict__ rp) {
    constexpr int E = D / 32, HALF = E / 2;
    int idx[E];
    float v[E];
#pragma unroll
    for (int j = 0; j < E; ++j) {
        idx[j] = rope == 3 ? lane * E + j : (j < HALF ? lane * HALF + j : D / 2 + lane * HALF + (j - HALF));
        v[j] = bf2f(p[idx[j]]);
    }
    if (norm != 3) {
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < E; ++j) ss += v[j] * v[j];
        ss = warp_sum(ss);
        float r = rsqrtf(ss / (float)D + eps);
#pragma unroll
        for (int j = 0; j < E; ++j) {
            float n = v[j] * r;
            v[j] = norm == 0 ? round_bf16(round_bf16(n) * bf2f(w[idx[j]])) : round_bf16(n * (1.f + bf2f(w[idx[j]])));
        }
    }
    if (rope == 3) {
#pragma unroll
        for (int j = 0; j < E; j += 2) {
            int f = idx[j] >> 1;
            float c = rp[f * 2], s = rp[f * 2 + 1];
            float a = v[j], b = v[j + 1];
            v[j] = a * c - b * s;
            v[j + 1] = b * c + a * s;
        }
    } else if (rope != 0) {
#pragma unroll
        for (int j = 0; j < HALF; ++j) {
            int f = idx[j];
            float c = rp[f * 2], s = rp[f * 2 + 1];
            float a = v[j], b = v[j + HALF];
            if (rope == 1) {
                v[j] = round_bf16(a * c) + round_bf16(-b * s);
                v[j + HALF] = round_bf16(b * c) + round_bf16(a * s);
            } else {
                v[j] = a * c - b * s;
                v[j + HALF] = b * c + a * s;
            }
        }
    }
#pragma unroll
    for (int j = 0; j < E; ++j) p[idx[j]] = f2bf(v[j]);
}

extern "C" {

// In place on fused rows (token stride ts elements): Hq query heads at column q_off, Hk key heads at k_off, head dim D
// (64 or 128). norm: 0 "hf", 1 "1p", 3 none.  rope: 0 none, 1 half (bf16 ops), 2 half32, 3 pairs32 (interleaved).
// rope table f32 [M, D/2, 2] = (cos, sin) of frequency f; split-half pairs (f, f + D/2), interleaved pairs (2f, 2f+1).
// One warp per (token, head); a lane owns D/32 dims: consecutive ones (pairs) or matching ones in both halves.
__global__ void k_qk_norm_rope(bf16* __restrict__ x, int64_t ts, int q_off, int k_off, int M, int Hq, int Hk, int D,
                               const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps, int norm, int rope,
                               const float* __restrict__ tab) {
    const int H = Hq + Hk;
    const int gw = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    if (gw >= M * H) return;
    const int m = gw / H, h = gw - m * H;
    bf16* p = x + (int64_t)m * ts + (h < Hq ? q_off + h * D : k_off + (h - Hq) * D);
    const bf16* w = h < Hq ? wq : wk;
    const float* rp = rope ? tab + (int64_t)m * D : nullptr;
    if (D == 128) qk_norm_rope_head<128>(p, w, lane, eps, norm, rope, rp);
    else qk_norm_rope_head<64>(p, w, lane, eps, norm, rope, rp);
}

// ------------------------------------------------------------------------------------------------ elementwise
// a[m, c] = bf16(a * bf16(sigmoid(g[m * gts + c])))   (Krea 2 attention output gate), a contiguous [M, C]
__global__ void k_sigmoid_gate(bf16* __restrict__ a, const bf16* __restrict__ g, int64_t gts, int64_t M, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * C) return;
    int64_t m = i / C;
    int c = i % C;
    float gv = round_bf16(1.f / (1.f + __expf(-bf2f(g[m * gts + c]))));
    a[i] = f2bf(bf2f(a[i]) * gv);
}
// SwiGLU over strided rows: out[m * ots + f] = bf16(bf16(silu(x[m*xts + off + f])) * x[m*xts + off + F + f])
__global__ void k_swiglu_strided(const bf16* __restrict__ x, int64_t xts, int off, bf16* __restrict__ out, int64_t ots, int64_t M, int F) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * F) return;
    int64_t m = i / F;
    int f = i % F;
    const bf16* r = x + m * xts + off;
    float g = round_bf16(silu_f(bf2f(r[f])));
    out[m * ots + f] = f2bf(g * bf2f(r[F + f]));
}
// x[m, c] = bf16(x + bf16(y[m * yts + c] * gate[c]))   (gated residual; y strided)
__global__ void k_gated_add_strided(bf16* __restrict__ x, const bf16* __restrict__ y, int64_t yts, const float* __restrict__ g, int64_t M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int64_t m = i / N;
    int c = i % N;
    x[i] = f2bf(bf2f(x[i]) + round_bf16(bf2f(y[m * yts + c]) * g[c]));
}
// x[i] = bf16(x[i] * m[i / C])   (zero rows by a per-row mask)
__global__ void k_row_scale(bf16* __restrict__ x, const float* __restrict__ s, int64_t M, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * C) x[i] = f2bf(bf2f(x[i]) * s[i / C]);
}
// Krea 2 layer projector: out[t, d] = bf16(sum_l w[l] * x[t, l, d])   (Linear(L -> 1) over the tapped-layer axis)
__global__ void k_layer_mix(const bf16* __restrict__ x, const float* __restrict__ w, int L, bf16* __restrict__ out, int64_t T, int D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * D) return;
    int64_t t = i / D;
    int d = i % D;
    float s = 0.f;
    for (int l = 0; l < L; ++l) s += w[l] * bf2f(x[(t * L + l) * D + d]);
    out[i] = f2bf(s);
}

// ------------------------------------------------------------------------------------------------ grouped attention
// Attention within groups of G consecutive tokens (Krea 2's layer-wise text fusion: every text token attends across its
// own tapped layers). q, k, v rows of stride ts at q_off / k_off / v_off, head dim 128, H heads (no GQA), out [N, H*128].
// grid (N, H), block 128; G <= 32.
__global__ void __launch_bounds__(128) k_attn_groups(const bf16* __restrict__ x, int64_t ts, int q_off, int k_off, int v_off,
                                                     bf16* __restrict__ out, int N, int H, int G, float scale) {
    __shared__ float qs[128];
    __shared__ float sc[32];
    const int i = blockIdx.x, h = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int g0 = (i / G) * G;
    qs[t] = bf2f(x[(int64_t)i * ts + q_off + h * 128 + t]);
    __syncthreads();
    for (int j = warp; j < G; j += 4) {
        const bf16* kr = x + (int64_t)(g0 + j) * ts + k_off + h * 128;
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) s += qs[lane * 4 + e] * bf2f(kr[lane * 4 + e]);
        s = warp_sum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = 0; j < G; ++j) mx = fmaxf(mx, sc[j]);
    float sum = 0.f, acc = 0.f;
    for (int j = 0; j < G; ++j) {
        float e = __expf(sc[j] - mx);
        sum += e;
        acc += e * bf2f(x[(int64_t)(g0 + j) * ts + v_off + h * 128 + t]);
    }
    out[(int64_t)i * H * 128 + h * 128 + t] = f2bf(acc / sum);
}

// ------------------------------------------------------------------------------------------------ Qwen-Image VAE latents
// DiT tokens f32 [h*w, 4C] (channel c*4 + dy*2 + dx, normalized) -> VAE latent plane bf16 [2h, 2w, Cp]: z = x * std + mean
__global__ void k_qlat_unpack(const float* __restrict__ tok, bf16* __restrict__ plane, int h, int w, int C, int Cp,
                              const float* __restrict__ mean, const float* __restrict__ std) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)h * w * 4 * C) return;
    int ch = i % (4 * C);
    int64_t t = i / (4 * C);
    int y = t / w, x = t % w;
    int c = ch >> 2, dy = (ch >> 1) & 1, dx = ch & 1;
    plane[(((int64_t)(2 * y + dy)) * (2 * w) + (2 * x + dx)) * Cp + c] = f2bf(tok[i] * std[c] + mean[c]);
}
// Encoder mean plane (bf16 [2h, 2w, Cp], the C mean channels first) -> normalized tokens f32 [h*w, 4C]
__global__ void k_qlat_pack(const bf16* __restrict__ plane, int Cp, float* __restrict__ tok, int h, int w, int C,
                            const float* __restrict__ mean, const float* __restrict__ std) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)h * w * 4 * C) return;
    int ch = i % (4 * C);
    int64_t t = i / (4 * C);
    int y = t / w, x = t % w;
    int c = ch >> 2, dy = (ch >> 1) & 1, dx = ch & 1;
    float z = bf2f(plane[(((int64_t)(2 * y + dy)) * (2 * w) + (2 * x + dx)) * Cp + c]);
    tok[i] = (z - mean[c]) / std[c];
}

}  // extern "C"
