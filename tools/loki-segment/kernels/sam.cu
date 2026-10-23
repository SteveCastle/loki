// SAM 3.1 model-specific kernels: ViT patchify / window order / RoPE, FPN pixel shuffle, detector box
// relative-position bias, sine embeddings, mask resampling.
#include "common.cuh"

// ViT token order: the 72x72 grid is kept in WINDOW-MAJOR order for the whole trunk (9 windows of 24x24,
// row-major inside each window), so windowed attention is a batch of 9 contiguous runs and global attention
// (permutation-equivariant) just uses per-token global RoPE angles. Token t -> (y, x):
DEVI void win_yx(int t, int& y, int& x) {
    int win = t / 576, l = t - win * 576;
    y = (win / 3) * 24 + l / 24;
    x = (win % 3) * 24 + l % 24;
}

extern "C" {

// img f32 [3, S, S] (normalized) -> patches bf16 [5184, Kp] in window-major token order; k = c*196 + ky*14 + kx.
__global__ void k_vit_patchify(const float* __restrict__ img, bf16* __restrict__ out, int S, int Kp) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)5184 * Kp) return;
    int t = (int)(i / Kp), k = (int)(i % Kp);
    float v = 0.f;
    if (k < 588) {
        int y, x;
        win_yx(t, y, x);
        int c = k / 196, r = k % 196, ky = r / 14, kx = r % 14;
        v = img[(int64_t)c * S * S + (int64_t)(y * 14 + ky) * S + (x * 14 + kx)];
    }
    out[i] = f2bf(v);
}
// x f32 [5184, C] += pos[1 + (t % 576)] (tiled 24x24 abs pos table f32 [577, C], cls row 0 skipped)
__global__ void k_vit_add_pos(float* __restrict__ x, const float* __restrict__ pos, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)5184 * C) return;
    int t = (int)(i / C), c = (int)(i % C);
    x[i] += pos[(int64_t)(1 + t % 576) * C + c];
}
// window-major f32 [5184, C] -> raster bf16 [5184, C]
__global__ void k_vit_unwindow(const float* __restrict__ x, bf16* __restrict__ out, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)5184 * C) return;
    int t = (int)(i / C), c = (int)(i % C);
    int y, xx;
    win_yx(t, y, xx);
    out[(int64_t)(y * 72 + xx) * C + c] = f2bf(x[i]);
}
// 2D axial RoPE on packed qkv bf16 [M, 3, H, D] (q and k), interleaved pairs (2p, 2p+1).
// cs f32 [period, D/2, 2] (cos, sin) indexed by token % period.
__global__ void k_rope_qkv(bf16* __restrict__ qkv, int M, int H, int D, const float* __restrict__ cs, int period) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int64_t total = (int64_t)M * 2 * H * half;
    if (i >= total) return;
    int p = (int)(i % half);
    int64_t r = i / half;
    int h = (int)(r % H); r /= H;
    int which = (int)(r % 2);
    int t = (int)(r / 2);
    bf16* v = qkv + (int64_t)t * 3 * H * D + which * H * D + h * D + 2 * p;
    const float* c = cs + ((int64_t)(t % period) * half + p) * 2;
    float a = bf2f(v[0]), b = bf2f(v[1]);
    float co = c[0], si = c[1];
    v[0] = f2bf(a * co - b * si);
    v[1] = f2bf(a * si + b * co);
}
// RoPE on a strided bf16 tensor x[t * ts + h * D + d] for t in [0, M): angles cs[(t % period)].
__global__ void k_rope_strided(bf16* __restrict__ x, int64_t ts, int M, int H, int D, const float* __restrict__ cs, int period) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    if (i >= (int64_t)M * H * half) return;
    int p = (int)(i % half);
    int64_t r = i / half;
    int h = (int)(r % H);
    int t = (int)(r / H);
    bf16* v = x + (int64_t)t * ts + h * D + 2 * p;
    const float* c = cs + ((int64_t)(t % period) * half + p) * 2;
    float a = bf2f(v[0]), b = bf2f(v[1]);
    v[0] = f2bf(a * c[0] - b * c[1]);
    v[1] = f2bf(a * c[1] + b * c[0]);
}

// ConvTranspose2d(k=2, s=2) output: GEMM result g [H*W, 4*Co] laid out (dy, dx, co) -> out [(2H)*(2W), Co].
__global__ void k_pixel_shuffle2(const bf16* __restrict__ g, bf16* __restrict__ out, int H, int W, int Co, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)H * W * 4 * Co;
    if (i >= total) return;
    int co = (int)(i % Co);
    int64_t r = i / Co;
    int d = (int)(r % 4);
    int64_t p = r / 4;
    int y = (int)(p / W), x = (int)(p % W);
    int dy = d >> 1, dx = d & 1;
    float v = bf2f(g[i]);
    if (act == 3) v = gelu_erf_f(v);
    out[((int64_t)(2 * y + dy) * (2 * W) + 2 * x + dx) * Co + co] = f2bf(v);
}
// out[y, x, c] = a[y, x, c] + b[y >> 1, x >> 1, c]   (nearest 2x upsample of b), bf16, out size H x W
__global__ void k_upadd2(const bf16* __restrict__ a, const bf16* __restrict__ b, bf16* __restrict__ out, int H, int W, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)H * W * C) return;
    int c = (int)(i % C);
    int64_t p = i / C;
    int y = (int)(p / W), x = (int)(p % W);
    out[i] = f2bf(bf2f(a[i]) + bf2f(b[((int64_t)(y >> 1) * (W >> 1) + (x >> 1)) * C + c]));
}

// ---------------------------------------------------------------- detector decoder helpers
DEVI float inv_sigmoid(float x) {
    x = fminf(fmaxf(x, 0.f), 1.f);
    float x1 = fmaxf(x, 1e-3f), x2 = fmaxf(1.f - x, 1e-3f);
    return logf(x1 / x2);
}
// gen_sineembed_for_position for cxcywh boxes [Q, 4] -> [Q, 512] in order (y, x, w, h), 128 feats each.
__global__ void k_sine_embed_boxes(const float* __restrict__ ref, float* __restrict__ out, int Q) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q * 512) return;
    int q = i / 512, f = i % 512;
    int part = f / 128, k = f % 128;
    const int src[4] = {1, 0, 2, 3};
    float v = ref[q * 4 + src[part]] * 6.283185307179586f;
    float dim_t = powf(10000.f, (float)(2 * (k / 2)) / 128.f);
    float a = v / dim_t;
    out[i] = (k & 1) ? cosf(a) : sinf(a);
}
// ref = sigmoid(inverse_sigmoid(ref) + delta)
__global__ void k_box_refine(float* __restrict__ ref, const float* __restrict__ delta, int n, float* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = inv_sigmoid(ref[i]) + delta[i];
    out[i] = 1.f / (1.f + expf(-v));
}
// Box relative-position bias MLP along one axis. ref cxcywh [Q, 4]; out [Q, L, NH].
//   delta_lo = i/L - box_lo, delta_hi = i/L - box_hi, log-scaled; h = relu(W1 d + b1) (256); out = W2 h + b2.
__global__ void k_rpb_axis(const float* __restrict__ ref, int Q, int L, int axis, const float* __restrict__ w1,
                           const float* __restrict__ b1, const float* __restrict__ w2, const float* __restrict__ b2,
                           int NH, float* __restrict__ out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= Q * L) return;
    int q = idx / L, i = idx % L;
    float c = ref[q * 4 + axis], s = ref[q * 4 + 2 + axis];
    float lo = c - 0.5f * s, hi = c + 0.5f * s;
    float coord = (float)i / (float)L;
    float d[2] = {coord - lo, coord - hi};
    for (int j = 0; j < 2; ++j) {
        float t = d[j] * 8.f;
        float sg = t > 0.f ? 1.f : (t < 0.f ? -1.f : 0.f);
        d[j] = sg * log2f(fabsf(t) + 1.f) / 3.f;
    }
    float acc[16];
    for (int h = 0; h < NH; ++h) acc[h] = b2[h];
    for (int u = 0; u < 256; ++u) {
        float hv = fmaxf(w1[u * 2] * d[0] + w1[u * 2 + 1] * d[1] + b1[u], 0.f);
        for (int h = 0; h < NH; ++h) acc[h] += w2[h * 256 + u] * hv;
    }
    for (int h = 0; h < NH; ++h) out[(int64_t)idx * NH + h] = acc[h];
}
// bias [NH, Q+1, H*W]: row 0 (presence token) zero; row q+1 = by[q, y, h] + bx[q, x, h]
__global__ void k_rpb_bias(const float* __restrict__ bx, const float* __restrict__ by, int Q, int H, int W, int NH,
                           float* __restrict__ bias) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t per_head = (int64_t)(Q + 1) * H * W;
    if (i >= per_head * NH) return;
    int h = (int)(i / per_head);
    int64_t r = i % per_head;
    int qq = (int)(r / (H * W));
    int p = (int)(r % (H * W));
    if (qq == 0) { bias[i] = 0.f; return; }
    int q = qq - 1, y = p / W, x = p % W;
    bias[i] = by[((int64_t)q * H + y) * NH + h] + bx[((int64_t)q * W + x) * NH + h];
}

// ---------------------------------------------------------------- mask output
// Bilinear resize (align_corners=False, no antialias, like F.interpolate) of logits f32 [N, h, w] to [N, H, W],
// then sigmoid -> u8 (round(255 * p)). Optional temporal blend handled by the caller.
__global__ void k_resize_logits_u8(const float* __restrict__ in, int N, int h, int w, uint8_t* __restrict__ out, int H, int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * H * W) return;
    int n = (int)(i / ((int64_t)H * W));
    int64_t r = i % ((int64_t)H * W);
    int y = (int)(r / W), x = (int)(r % W);
    float sy = fmaxf(((float)y + 0.5f) * (float)h / (float)H - 0.5f, 0.f);
    float sx = fmaxf(((float)x + 0.5f) * (float)w / (float)W - 0.5f, 0.f);
    int y0 = min((int)sy, h - 1), x0 = min((int)sx, w - 1);
    int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
    float fy = sy - y0, fx = sx - x0;
    const float* m = in + (int64_t)n * h * w;
    float v = (1.f - fy) * ((1.f - fx) * m[y0 * w + x0] + fx * m[y0 * w + x1]) + fy * ((1.f - fx) * m[y1 * w + x0] + fx * m[y1 * w + x1]);
    float p = 1.f / (1.f + expf(-v));
    out[i] = (uint8_t)fminf(fmaxf(rintf(p * 255.f), 0.f), 255.f);
}
// Same resize, f32 logits out (for video smoothing / memory).
__global__ void k_resize_logits_f32(const float* __restrict__ in, int N, int h, int w, float* __restrict__ out, int H, int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)N * H * W) return;
    int n = (int)(i / ((int64_t)H * W));
    int64_t r = i % ((int64_t)H * W);
    int y = (int)(r / W), x = (int)(r % W);
    float sy = fmaxf(((float)y + 0.5f) * (float)h / (float)H - 0.5f, 0.f);
    float sx = fmaxf(((float)x + 0.5f) * (float)w / (float)W - 0.5f, 0.f);
    int y0 = min((int)sy, h - 1), x0 = min((int)sx, w - 1);
    int y1 = min(y0 + 1, h - 1), x1 = min(x0 + 1, w - 1);
    float fy = sy - y0, fx = sx - x0;
    const float* m = in + (int64_t)n * h * w;
    out[i] = (1.f - fy) * ((1.f - fx) * m[y0 * w + x0] + fx * m[y0 * w + x1]) + fy * ((1.f - fx) * m[y1 * w + x0] + fx * m[y1 * w + x1]);
}

}  // extern "C"

extern "C" {
// Small linear layer for odd output widths: out[m, n] = act(sum_k x[m, k] * w[n, k] + b[n]); x f32 or bf16 (xt), w f32.
__global__ void k_linear_small(const void* __restrict__ x, int xt, const float* __restrict__ w, const float* __restrict__ b,
                               float* __restrict__ out, int M, int N, int K, int act) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int m = i / N, n = i % N;
    float acc = b ? b[n] : 0.f;
    if (xt) {
        const bf16* xr = (const bf16*)x + (int64_t)m * K;
        for (int k = 0; k < K; ++k) acc += bf2f(xr[k]) * w[(int64_t)n * K + k];
    } else {
        const float* xr = (const float*)x + (int64_t)m * K;
        for (int k = 0; k < K; ++k) acc += xr[k] * w[(int64_t)n * K + k];
    }
    if (act == 4) acc = fmaxf(acc, 0.f);
    out[i] = acc;
}
}

// ---------------------------------------------------------------- tracker (memory encoder) helpers
extern "C" {
// Depthwise 7x7 conv, padding 3, NHWC: in bf16 [H, W, C], w f32 [C, 49], b f32 [C] -> out f32 [H, W, C]
__global__ void k_dwconv7(const bf16* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                          float* __restrict__ out, int H, int W, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)H * W * C) return;
    int c = (int)(i % C);
    int64_t p = i / C;
    int y = (int)(p / W), x = (int)(p % W);
    float acc = b[c];
    for (int ky = 0; ky < 7; ++ky) {
        int iy = y + ky - 3;
        if (iy < 0 || iy >= H) continue;
        for (int kx = 0; kx < 7; ++kx) {
            int ix = x + kx - 3;
            if (ix < 0 || ix >= W) continue;
            acc += bf2f(in[((int64_t)iy * W + ix) * C + c]) * w[c * 49 + ky * 7 + kx];
        }
    }
    out[i] = acc;
}
// Separable resampling along one axis of an f32 NHWC plane [H, W, C] with a precomputed filter table:
// for output index o: sum_{k < cnt[o]} wt[o * kmax + k] * in[start[o] + k] (along `axis`: 0 = rows, 1 = columns).
__global__ void k_resample_axis(const float* __restrict__ in, float* __restrict__ out, int H, int W, int C, int axis,
                                int out_len, const int* __restrict__ start, const int* __restrict__ cnt,
                                const float* __restrict__ wt, int kmax) {
    int OH = axis == 0 ? out_len : H, OW = axis == 1 ? out_len : W;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)OH * OW * C) return;
    int c = (int)(i % C);
    int64_t p = i / C;
    int y = (int)(p / OW), x = (int)(p % OW);
    int o = axis == 0 ? y : x;
    float acc = 0.f;
    for (int k = 0; k < cnt[o]; ++k) {
        int s = start[o] + k;
        int iy = axis == 0 ? s : y, ix = axis == 1 ? s : x;
        acc += wt[o * kmax + k] * in[((int64_t)iy * W + ix) * C + c];
    }
    out[i] = acc;
}
// Multiplex the memory-encoder input: logits f32 [n, P] -> NHWC f32 [P, 2*M]:
//   channel s < n: sigmoid(logit)*2 - 1; channel M + s (s < n): cond[s]; padding slots 0.
__global__ void k_mux_masks(const float* __restrict__ logits, const float* __restrict__ cond, int n, int64_t P, int M,
                            float* __restrict__ out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * 2 * M) return;
    int ch = (int)(i % (2 * M));
    int64_t p = i / (2 * M);
    float v = 0.f;
    if (ch < M) {
        if (ch < n) v = 2.f / (1.f + expf(-logits[(int64_t)ch * P + p])) - 1.f;
    } else if (ch - M < n) {
        v = cond[ch - M];
    }
    out[i] = v;
}
// Non-overlap constraint in place over [n, P]: keep the per-pixel argmax object, clamp the others to <= -10
__global__ void k_nonoverlap(float* __restrict__ m, int n, int64_t P) {
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P || n <= 1) return;
    int best = 0;
    float bv = m[p];
    for (int k = 1; k < n; ++k) {
        float v = m[(int64_t)k * P + p];
        if (v > bv) { bv = v; best = k; }
    }
    for (int k = 0; k < n; ++k)
        if (k != best) m[(int64_t)k * P + p] = fminf(m[(int64_t)k * P + p], -10.f);
}
// LayerNorm over C (<= 2048) per row, one warp per row: in bf16/f32 (it), out bf16, optional exact GELU.
__global__ void k_ln_warp(const void* __restrict__ in, int it, bf16* __restrict__ out, int64_t M, int C, float eps,
                          const float* __restrict__ w, const float* __restrict__ b, int gelu) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= M) return;
    float v[64];
    int n = 0;
    float s = 0.f;
    for (int c = lane; c < C; c += 32, ++n) {
        float x = it ? bf2f(((const bf16*)in)[row * C + c]) : ((const float*)in)[row * C + c];
        v[n] = x;
        s += x;
    }
    float mean = warp_sum(s) / C;
    float ss = 0.f;
    for (int k = 0; k < n; ++k) { float d = v[k] - mean; ss += d * d; }
    float rstd = rsqrtf(warp_sum(ss) / C + eps);
    n = 0;
    for (int c = lane; c < C; c += 32, ++n) {
        float y = (v[n] - mean) * rstd * w[c] + b[c];
        if (gelu) y = gelu_erf_f(y);
        out[row * C + c] = f2bf(y);
    }
}
}
