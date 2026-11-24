// loki-tune: the backward passes of the MOSS-Image block ops (loki-dream's dream.cu / quant.cu forward), the weight
// transposes the input-gradient GEMMs need, the small-K GEMM for LoRA weight gradients, the flow-matching loss and AdamW.
//
// Gradients are f32 where they accumulate (the residual stream, LoRA weights) and bf16 where they feed a tensor-core
// GEMM. The forward rounds to bf16 the way the reference does; the backward does its arithmetic in f32 and treats each
// forward rounding as the identity (and the int8 activation quantizer as a straight-through estimator).
#include "common.cuh"
#include <mma.h>

extern "C" {

// ------------------------------------------------------------------------------------------------ norms
// Modulated RMSNorm backward: y = (x r) w (1 + scale) + shift, r = rsqrt(mean(x^2) + eps).
//   u = dy w (1 + scale) ;  dx += r u - x r^3 mean(u x)      (scale == nullptr: plain RMSNorm with weight w)
// x bf16 [M, K], dy f32 [M, K], dx f32 [M, K] accumulated. One block of 256 threads per row.
__global__ void k_modrms_bwd(const bf16* __restrict__ x, const bf16* __restrict__ w, const float* __restrict__ scale,
                             const float* __restrict__ dy, float* __restrict__ dx, int M, int K, float eps) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    const bf16* px = x + m * K;
    const float* pdy = dy + m * K;
    float* pdx = dx + m * K;
    float ss = 0.f, su = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = bf2f(px[i]);
        float u = pdy[i] * bf2f(w[i]) * (scale ? 1.f + scale[i] : 1.f);
        ss += v * v;
        su += u * v;
    }
    ss = block_sum(ss, red);
    su = block_sum(su, red);
    float r = rsqrtf(ss / K + eps);
    float c = r * r * r * su / K;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float u = pdy[i] * bf2f(w[i]) * (scale ? 1.f + scale[i] : 1.f);
        pdx[i] += r * u - bf2f(px[i]) * c;
    }
}

// q/k per-head RMSNorm + split-half rope backward (k_qk_norm_rope_moss forward), in place on the q and k parts of a
// fused [M, ts] gradient: on entry the gradient w.r.t. the rotated q/k, on exit the gradient w.r.t. the projection
// output. pre = the projection output before the norm (same layout). One warp per (token, head, q|k).
__global__ void k_qk_norm_rope_bwd(const bf16* __restrict__ pre, bf16* __restrict__ g, int64_t ts, int M, int H,
                                   const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps, const float* __restrict__ rope) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * H * 2) return;
    int64_t m = gw / (2 * H);
    int rem = gw % (2 * H);
    int which = rem / H, h = rem % H;
    int64_t base = m * ts + (int64_t)which * H * 128 + h * 128;
    const bf16* px = pre + base;
    bf16* pg = g + base;
    const bf16* w = which == 0 ? wq : wk;
    int i0 = lane * 2;
    float x[4] = {bf2f(px[i0]), bf2f(px[i0 + 1]), bf2f(px[i0 + 64]), bf2f(px[i0 + 65])};
    float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    float r = rsqrtf(ss / 128.f + eps);
    const float* rp = rope + m * 128;
    float c0 = rp[i0 * 2], s0 = rp[i0 * 2 + 1], c1 = rp[i0 * 2 + 2], s1 = rp[i0 * 2 + 3];
    float ga0 = bf2f(pg[i0]), ga1 = bf2f(pg[i0 + 1]), gb0 = bf2f(pg[i0 + 64]), gb1 = bf2f(pg[i0 + 65]);
    // rope^-1: out_a = a c - b s, out_b = b c + a s  ->  da = ga c + gb s, db = gb c - ga s
    float dn[4] = {ga0 * c0 + gb0 * s0, ga1 * c1 + gb1 * s1, gb0 * c0 - ga0 * s0, gb1 * c1 - ga1 * s1};
    int idx[4] = {i0, i0 + 1, i0 + 64, i0 + 65};
    float u[4], su = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u[j] = dn[j] * bf2f(w[idx[j]]);
        su += u[j] * x[j];
    }
    su = warp_sum(su);
    float c = r * r * r * su / 128.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) pg[idx[j]] = f2bf(r * u[j] - x[j] * c);
}

// ------------------------------------------------------------------------------------------------ activations
// situ(h, g) = left(g) right(h), left = 4 tanh(g/4) sigmoid(g), right = 25 tanh(h/25).
// ug bf16 [M, 2F] = [hidden | gate], ds f32 [M, F] -> dug bf16 [M, 2F].
__global__ void k_situ_bwd(const bf16* __restrict__ ug, const float* __restrict__ ds, bf16* __restrict__ dug, int64_t M, int F) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * F) return;
    int64_t m = i / F;
    int f = i % F;
    float h = bf2f(ug[m * 2 * F + f]), gt = bf2f(ug[m * 2 * F + F + f]);
    float th = tanhf(h / 25.f), tg = tanhf(gt * 0.25f);
    float sg = 1.f / (1.f + __expf(-gt));
    float right = 25.f * th, dright = 1.f - th * th;
    float left = 4.f * tg * sg;
    float dleft = (1.f - tg * tg) * sg + 4.f * tg * sg * (1.f - sg);
    float d = ds[i];
    dug[m * 2 * F + f] = f2bf(d * left * dright);
    dug[m * 2 * F + F + f] = f2bf(d * right * dleft);
}

// out[m, n] = bf16(x[m, n] * g[n])   (x f32, g f32 per column: the residual gates)
__global__ void k_scale_cols_bf16(const float* __restrict__ x, const float* __restrict__ g, bf16* __restrict__ out, int64_t M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) out[i] = f2bf(x[i] * g[i % N]);
}

// ------------------------------------------------------------------------------------------------ ConvRot
// Normalized Hadamard-256 (ConvRot's rotation, its own inverse) on every 256-group of f32 rows [M, K], in place.
// One block of 256 threads per row.
__global__ void k_fwht256_f32(float* __restrict__ x, int M, int K) {
    __shared__ float gsh[256];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    int t = threadIdx.x;
    for (int grp = 0; grp < K / 256; ++grp) {
        float* p = x + m * K + grp * 256;
        gsh[t] = p[t];
#pragma unroll
        for (int st = 1; st < 256; st <<= 2) {
            __syncthreads();
            if (t < 64) {
                int lo = t % st, hi = (t / st) * st * 4, b = hi + lo;
                float x0 = gsh[b], x1 = gsh[b + st], x2 = gsh[b + 2 * st], x3 = gsh[b + 3 * st];
                gsh[b] = x0 + x1 + x2 - x3;
                gsh[b + st] = x0 + x1 - x2 + x3;
                gsh[b + 2 * st] = x0 - x1 + x2 + x3;
                gsh[b + 3 * st] = -x0 + x1 + x2 + x3;
            }
        }
        __syncthreads();
        p[t] = gsh[t] * 0.0625f;
        __syncthreads();
    }
}

// ------------------------------------------------------------------------------------------------ transposes
// W^T for the input-gradient GEMM: out[k, n] = W[n, k] (bf16), or dequantized int8 W[n, k] * scale[n].
// 32x32 tiles, block (32, 8).
__global__ void k_transpose_w(const void* __restrict__ w, int is_i8, const float* __restrict__ scale, bf16* __restrict__ out, int N, int K) {
    __shared__ float tile[32][33];
    int n0 = blockIdx.y * 32, k0 = blockIdx.x * 32;
    for (int r = threadIdx.y; r < 32; r += 8) {
        int n = n0 + r, k = k0 + threadIdx.x;
        float v = 0.f;
        if (n < N && k < K) {
            int64_t o = (int64_t)n * K + k;
            v = is_i8 ? (float)reinterpret_cast<const int8_t*>(w)[o] * scale[n] : bf2f(reinterpret_cast<const bf16*>(w)[o]);
        }
        tile[r][threadIdx.x] = v;
    }
    __syncthreads();
    for (int r = threadIdx.y; r < 32; r += 8) {
        int k = k0 + r, n = n0 + threadIdx.x;
        if (k < K && n < N) out[(int64_t)k * N + n] = f2bf(tile[threadIdx.x][r]);
    }
}

// LoRA weights for a step: out = bf16(src * s) as [R, C] (transpose == 0) or as its transpose [C, R].
__global__ void k_lora_cast(const float* __restrict__ src, bf16* __restrict__ out, int R, int C, float s, int transpose) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)R * C) return;
    int r = i / C, c = i % C;
    float v = src[i] * s;
    if (transpose) out[(int64_t)c * R + r] = f2bf(v);
    else out[i] = f2bf(v);
}

// Zero the off-diagonal blocks of a fused q|k|v LoRA up-gradient [3*Dm, 3*r] (q, k and v each have their own LoRA).
__global__ void k_mask_blockdiag(float* __restrict__ g, int Dm, int r) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)3 * Dm * 3 * r) return;
    int row = i / (3 * r), col = i % (3 * r);
    if (row / Dm != col / r) g[i] = 0.f;
}

// ------------------------------------------------------------------------------------------------ weight gradients
// C[P, Q] += alpha * U^T V  with U bf16 [M, P] (row stride ldu), V bf16 [M, Q] (row stride ldv), C f32 [P, Q]:
// the LoRA gradients dB = d_out^T mid and dA = dmid^T x, contracted over the tokens. Tile 64x64, 4 warps of 32x32,
// M in chunks of 32 staged through shared memory (zero-padded at every edge). grid (ceil(Q/64), ceil(P/64)).
__global__ void __launch_bounds__(128) k_gemm_tn(const bf16* __restrict__ U, int64_t ldu, const bf16* __restrict__ V, int64_t ldv,
                                                 float* __restrict__ C, int M, int P, int Q, float alpha) {
    using namespace nvcuda;
    constexpr int T = 64, KC = 32, LD = T + 8;
    __shared__ __align__(32) bf16 Us[KC][LD];
    __shared__ __align__(32) bf16 Vs[KC][LD];
    __shared__ __align__(32) float Cs[T][T + 4];
    const int p0 = blockIdx.y * T, q0 = blockIdx.x * T;
    const int tid = threadIdx.x, warp = tid >> 5;
    const int wp = (warp >> 1) * 32, wq = (warp & 1) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.f);
    for (int m0 = 0; m0 < M; m0 += KC) {
        for (int e = tid; e < KC * T; e += 128) {
            int r = e / T, c = e % T;
            int m = m0 + r;
            Us[r][c] = (m < M && p0 + c < P) ? U[(int64_t)m * ldu + p0 + c] : f2bf(0.f);
            Vs[r][c] = (m < M && q0 + c < Q) ? V[(int64_t)m * ldv + q0 + c] : f2bf(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KC; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &Us[kk][wp + i * 16], LD);
#pragma unroll
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Vs[kk][wq + j * 16], LD);
#pragma unroll
            for (int i = 0; i < 2; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::store_matrix_sync(&Cs[wp + i * 16][wq + j * 16], acc[i][j], T + 4, wmma::mem_row_major);
    __syncthreads();
    for (int e = tid; e < T * T; e += 128) {
        int r = e / T, c = e % T;
        if (p0 + r < P && q0 + c < Q) C[(int64_t)(p0 + r) * Q + q0 + c] += alpha * Cs[r][c];
    }
}

// ------------------------------------------------------------------------------------------------ loss
// Flow matching: x_t = (1 - s) x0 + s eps (bf16 model input), target v = eps - x0.
__global__ void k_noise_mix_bf16(const float* __restrict__ x0, const float* __restrict__ eps, float s, bf16* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2bf((1.f - s) * x0[i] + s * eps[i]);
}
// MSE over n elements: grad = 2 (v - target) / n as bf16; per-block partial sums of the squared error in partial[].
__global__ void k_mse(const bf16* __restrict__ v, const float* __restrict__ x0, const float* __restrict__ eps,
                      bf16* __restrict__ grad, float* __restrict__ partial, int64_t n) {
    __shared__ float red[32];
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float e = 0.f;
    if (i < n) {
        float d = bf2f(v[i]) - (eps[i] - x0[i]);
        grad[i] = f2bf(2.f * d / (float)n);
        e = d * d;
    }
    e = block_sum(e, red);
    if (threadIdx.x == 0) partial[blockIdx.x] = e;
}

// ------------------------------------------------------------------------------------------------ optimizer
// Per-block partial sums of squares (gradient norm for clipping).
__global__ void k_sumsq(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
    __shared__ float red[32];
    float s = 0.f;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) s += x[i] * x[i];
    s = block_sum(s, red);
    if (threadIdx.x == 0) partial[blockIdx.x] = s;
}
// AdamW (decoupled weight decay) on f32 params; g is scaled by gscale (accumulation average x clipping) and cleared.
__global__ void k_adamw(float* __restrict__ p, float* __restrict__ g, float* __restrict__ m, float* __restrict__ v, int64_t n,
                        float lr, float b1, float b2, float eps, float wd, float bc1, float bc2, float gscale) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i] * gscale;
    g[i] = 0.f;
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi;
    v[i] = vi;
    float upd = (mi / bc1) / (sqrtf(vi / bc2) + eps);
    p[i] = p[i] * (1.f - lr * wd) - lr * upd;
}

}  // extern "C"
