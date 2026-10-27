// loki-dream kernels: MOSS-Image DiT norms/rope/activations, the Qwen3.5 text encoder's gated DeltaNet and gated
// attention, the FLUX.2 VAE's group norm and latent (un)packing, and the sampler's fp32 update steps.
//
// Rounding: the official pipeline (pipeline_moss_image.py) runs the DiT as chains of bf16 tensor ops; the kernels here
// round every intermediate to bf16 where it does, so the native engine follows the reference op for op.
#include "common.cuh"

extern "C" {

// ------------------------------------------------------------------------------------------------ DiT
// Modulated RMSNorm (MOSS _RMSNorm then x*(1+scale)+shift), bf16 in -> bf16 out. scale == nullptr: plain _RMSNorm.
__global__ void k_modrms_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int M, int K, float eps,
                              const bf16* __restrict__ w, const float* __restrict__ scale, const float* __restrict__ shift) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    const bf16* px = x + m * K;
    bf16* py = y + m * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = bf2f(px[i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / K + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x)
        py[i] = f2bf(mod_rms(bf2f(px[i]), r, w ? bf2f(w[i]) : 1.f, scale ? scale[i] : 0.f, shift ? shift[i] : 0.f, scale != nullptr));
}

// situ on [M, 2F] = [hidden | gate] rows -> [M, F]
__global__ void k_situ_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t M, int F) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * F) return;
    int64_t m = i / F;
    int f = i % F;
    y[i] = f2bf(situ_bf16(bf2f(x[m * 2 * F + f]), bf2f(x[m * 2 * F + F + f])));
}

// MOSS per-head q/k RMSNorm + split-half rope, in place on a fused [M, 3*H*128] qkv (q at column 0, k at H*128).
// rope [M, 64, 2] = (cos, sin) as f32 holding bf16 values (the reference casts its rope table to bf16).
//   n = (x * r).bf16 * w ; out[i] = n[i]*cos - n[i+64]*sin ; out[i+64] = n[i+64]*cos + n[i]*sin   (each product a bf16 op)
// One warp per (token, head, q|k); lane owns dims 2l, 2l+1 and their +64 partners.
__global__ void k_qk_norm_rope_moss(bf16* __restrict__ qkv, int64_t ts, int M, int H, const bf16* __restrict__ wq,
                                    const bf16* __restrict__ wk, float eps, const float* __restrict__ rope) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * H * 2) return;
    int64_t m = gw / (2 * H);
    int rem = gw % (2 * H);
    int which = rem / H, h = rem % H;
    bf16* p = qkv + m * ts + (int64_t)which * H * 128 + h * 128;
    const bf16* w = which == 0 ? wq : wk;
    int i0 = lane * 2;
    float a0 = bf2f(p[i0]), a1 = bf2f(p[i0 + 1]), b0 = bf2f(p[i0 + 64]), b1 = bf2f(p[i0 + 65]);
    float ss = warp_sum(a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1);
    float r = rsqrtf(ss / 128.f + eps);
    a0 = round_bf16(round_bf16(a0 * r) * bf2f(w[i0]));
    a1 = round_bf16(round_bf16(a1 * r) * bf2f(w[i0 + 1]));
    b0 = round_bf16(round_bf16(b0 * r) * bf2f(w[i0 + 64]));
    b1 = round_bf16(round_bf16(b1 * r) * bf2f(w[i0 + 65]));
    const float* rp = rope + m * 128;
    float c0 = rp[i0 * 2], s0 = rp[i0 * 2 + 1], c1 = rp[i0 * 2 + 2], s1 = rp[i0 * 2 + 3];
    p[i0] = f2bf(round_bf16(a0 * c0) + round_bf16(-b0 * s0));
    p[i0 + 1] = f2bf(round_bf16(a1 * c1) + round_bf16(-b1 * s1));
    p[i0 + 64] = f2bf(round_bf16(b0 * c0) + round_bf16(a0 * s0));
    p[i0 + 65] = f2bf(round_bf16(b1 * c1) + round_bf16(a1 * s1));
}

// Gated residual with the reference's rounding: x = x + (y * gate).bf16   (bf16 x, y; gate f32 holding bf16 values)
__global__ void k_gated_add_bf16(bf16* __restrict__ x, const bf16* __restrict__ y, const float* __restrict__ g, int64_t M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) x[i] = f2bf(bf2f(x[i]) + round_bf16(bf2f(y[i]) * g[i % N]));
}

// ------------------------------------------------------------------------------------------------ sampler
// CFG + Euler: pred = vu + cfg*(vc - vu) (vu == null: pred = vc); x += dt * pred.   x f32, v bf16, token-major.
__global__ void k_cfg_euler(float* __restrict__ x, const bf16* __restrict__ vc, const bf16* __restrict__ vu, float cfg, float dt, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = bf2f(vc[i]);
    float p = c;
    if (vu) { float u = bf2f(vu[i]); p = u + cfg * (c - u); }
    x[i] = x[i] + dt * p;
}
// Inpainting: outside the mask, re-impose the init latent noised to sigma:  x = m*x + (1-m)*((1-s)*x0 + s*noise)
// mask [n/C] per token (soft), x/x0/noise token-major [tokens, C].
__global__ void k_inpaint_blend(float* __restrict__ x, const float* __restrict__ x0, const float* __restrict__ noise,
                                const float* __restrict__ mask, float sigma, int64_t n, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float m = mask[i / C];
    float ref = (1.f - sigma) * x0[i] + sigma * noise[i];
    x[i] = m * x[i] + (1.f - m) * ref;
}
// SDEdit start: x = (1-s)*x0 + s*noise
__global__ void k_noise_mix(float* __restrict__ x, const float* __restrict__ x0, const float* __restrict__ noise, float sigma, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (1.f - sigma) * x0[i] + sigma * noise[i];
}

// ------------------------------------------------------------------------------------------------ LoRA
// x[m, col0 + j] += y[m, j] for a slice of columns (bf16 both; out row stride ts, y row stride ys)
__global__ void k_add_cols_bf16(bf16* __restrict__ x, int64_t ts, int col0, const bf16* __restrict__ y, int64_t ys, int64_t M, int W) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * W) return;
    int64_t m = i / W;
    int j = i % W;
    bf16* p = x + m * ts + col0 + j;
    *p = f2bf(bf2f(*p) + bf2f(y[m * ys + j]));
}

// ------------------------------------------------------------------------------------------------ Qwen3.5 text encoder
// Full-attention layers: q_proj rows hold, per head, [q (256) | gate (256)]; k rows [Hk, 256] at k_off of a fused row.
// RMSNorm over the 256 dims with (1 + w) (one rounding), then rope on the first 64 dims (split halves of 32) with the
// bf16 cos/sin table rope [M, 32, 2]; q goes to qo [M, Hq, 256], k to ko [M, Hk, 256].
// One warp per (token, head); lane owns 8 dims: 2l, 2l+1 in the rotary part handled separately.
__global__ void k_qk_norm_rope_q35(const bf16* __restrict__ qkv, int64_t ts, int k_off, int M, int Hq, int Hk,
                                   const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,
                                   const float* __restrict__ rope, bf16* __restrict__ qo, bf16* __restrict__ ko) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * (Hq + Hk)) return;
    int64_t m = gw / (Hq + Hk);
    int h = gw % (Hq + Hk);
    const bf16* src;
    const bf16* w;
    bf16* dst;
    if (h < Hq) { src = qkv + m * ts + (int64_t)h * 512; w = wq; dst = qo + (m * Hq + h) * 256; }
    else { src = qkv + m * ts + k_off + (int64_t)(h - Hq) * 256; w = wk; dst = ko + (m * Hk + (h - Hq)) * 256; }
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = bf2f(src[lane * 8 + j]);
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) ss += v[j] * v[j];
    ss = warp_sum(ss);
    float r = rsqrtf(ss / 256.f + eps);
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = round_bf16(v[j] * r * (1.0f + bf2f(w[lane * 8 + j])));
    // dims 0..63 are owned by lanes 0..7; partner of d (< 32) is d + 32, owned by lane + 4
    float partner[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) partner[j] = __shfl_sync(0xffffffff, v[j], (lane < 4) ? lane + 4 : (lane < 8 ? lane - 4 : lane));
    if (lane < 8) {
        const float* rp = rope + m * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int d = lane * 8 + j;
            int f = d & 31;
            float c = rp[f * 2], s = rp[f * 2 + 1];
            float rot = d < 32 ? -partner[j] : partner[j];
            v[j] = round_bf16(round_bf16(v[j] * c) + round_bf16(rot * s));
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) dst[lane * 8 + j] = f2bf(v[j]);
}

// Causal attention for short sequences, head dim 256, GQA, with the output gate: out = (softmax(qk^T/16) v).bf16 * sigmoid(gate).bf16
// grid (L, Hq), block 256 (one thread per output dim). Scores live in shared memory (L <= 8192).
__global__ void __launch_bounds__(256) k_attn_q35(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
                                                  int64_t v_ts, const bf16* __restrict__ gate, int64_t g_ts, bf16* __restrict__ out,
                                                  int L, int Hq, int Hk, float scale) {
    extern __shared__ float sc[];
    __shared__ float red[32];
    __shared__ float qs[256];
    const int i = blockIdx.x, h = blockIdx.y, hk = h / (Hq / Hk);
    const int t = threadIdx.x, warp = t >> 5, lane = t & 31;
    qs[t] = bf2f(q[((int64_t)i * Hq + h) * 256 + t]);
    __syncthreads();
    const int n = i + 1;
    for (int j = warp; j < n; j += 8) {
        const bf16* kr = k + ((int64_t)j * Hk + hk) * 256;
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) s += qs[lane * 8 + e] * bf2f(kr[lane * 8 + e]);
        s = warp_sum(s);
        if (lane == 0) sc[j] = s * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int j = t; j < n; j += 256) mx = fmaxf(mx, sc[j]);
    mx = block_max(mx, red);
    float sum = 0.f;
    for (int j = t; j < n; j += 256) { float e = __expf(sc[j] - mx); sc[j] = e; sum += e; }
    sum = block_sum(sum, red);
    __syncthreads();
    float acc = 0.f;
    for (int j = 0; j < n; ++j) acc += sc[j] * bf2f(v[(int64_t)j * v_ts + hk * 256 + t]);
    float o = round_bf16(acc / sum);
    float g = round_bf16(1.f / (1.f + __expf(-bf2f(gate[(int64_t)i * g_ts + h * 512 + 256 + t]))));
    out[((int64_t)i * Hq + h) * 256 + t] = f2bf(o * g);
}

// Depthwise causal conv1d (kernel 4) + SiLU: out[t, c] = silu(sum_j w[c, j] * x[t - 3 + j, c]) (bf16 conv result, bf16 silu)
__global__ void k_causal_conv_silu(const bf16* __restrict__ x, int64_t ts, const bf16* __restrict__ w, bf16* __restrict__ out, int L, int C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * C) return;
    int t = i / C, c = i % C;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int tt = t - 3 + j;
        if (tt >= 0) acc += bf2f(w[c * 4 + j]) * bf2f(x[(int64_t)tt * ts + c]);
    }
    out[i] = f2bf(silu_f(round_bf16(acc)));
}

// Gated DeltaNet inputs: conv output [L, 2*Hk*128 + Hv*128] (q | k | v) -> f32 q, k (L2-normalized, q / sqrt(128)) per key
// head, v per value head; beta = sigmoid(b) (bf16), g = -exp(A_log) * softplus(a + dt_bias) (f32). ab rows: [b(Hv) | a(Hv)].
// One warp per (token, head) over Hk q heads, Hk k heads, Hv v heads.
__global__ void k_gdn_prep(const bf16* __restrict__ conv, const bf16* __restrict__ ab, int64_t ab_ts, int L, int Hk, int Hv,
                           const float* __restrict__ a_log, const float* __restrict__ dt_bias,
                           float* __restrict__ qf, float* __restrict__ kf, float* __restrict__ vf, float* __restrict__ g, float* __restrict__ beta) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    const int per = 2 * Hk + Hv;
    if (gw >= (int64_t)L * per) return;
    int t = gw / per, hh = gw % per;
    const int C = (2 * Hk + Hv) * 128;
    const bf16* row = conv + (int64_t)t * C;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = bf2f(row[hh * 128 + lane * 4 + j]);
    if (hh < 2 * Hk) {
        float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
        float inv = rsqrtf(ss + 1e-6f);
        bool isq = hh < Hk;
        if (isq) inv *= 0.08838834764831845f;  // 1/sqrt(128)
        float* dst = (isq ? qf + ((int64_t)t * Hk + hh) * 128 : kf + ((int64_t)t * Hk + hh - Hk) * 128) + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j] = x[j] * inv;
    } else {
        int hv = hh - 2 * Hk;
        float* dst = vf + ((int64_t)t * Hv + hv) * 128 + lane * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) dst[j] = x[j];
        if (lane == 0) {
            float b = bf2f(ab[(int64_t)t * ab_ts + hv]);
            float a = bf2f(ab[(int64_t)t * ab_ts + Hv + hv]);
            beta[(int64_t)t * Hv + hv] = round_bf16(1.f / (1.f + expf(-b)));
            float z = a + dt_bias[hv];
            float sp = z > 20.f ? z : log1pf(expf(z));
            g[(int64_t)t * Hv + hv] = -expf(a_log[hv]) * sp;
        }
    }
}

// Gated delta rule, recurrent form (fp32). One block of 256 threads per value head; thread (half, j) owns state column j
// (value dim) for key rows [64*half, 64*half+64) in registers.
//   S *= exp(g_t); kv = S^T k; delta = (v - kv) * beta; S += k delta^T; o = S^T q
// Value head hv reads key/query head hv / (Hv/Hk). out f32 [L, Hv, 128].
__global__ void __launch_bounds__(256) k_gdn_recurrent(const float* __restrict__ qf, const float* __restrict__ kf, const float* __restrict__ vf,
                                                       const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ out,
                                                       int L, int Hk, int Hv) {
    __shared__ float ks[128], qs[128];
    __shared__ float part[2][128];
    const int hv = blockIdx.x, hk = hv / (Hv / Hk);
    const int t = threadIdx.x, j = t & 127, half = t >> 7, k0 = half * 64;
    float S[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) S[i] = 0.f;
    for (int tok = 0; tok < L; ++tok) {
        if (t < 128) ks[t] = kf[((int64_t)tok * Hk + hk) * 128 + t];
        else qs[t - 128] = qf[((int64_t)tok * Hk + hk) * 128 + t - 128];
        const float decay = expf(g[(int64_t)tok * Hv + hv]);
        const float b = beta[(int64_t)tok * Hv + hv];
        const float vj = vf[((int64_t)tok * Hv + hv) * 128 + j];
        __syncthreads();
        float p = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) { S[i] *= decay; p += S[i] * ks[k0 + i]; }
        part[half][j] = p;
        __syncthreads();
        const float delta = (vj - (part[0][j] + part[1][j])) * b;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) { S[i] += ks[k0 + i] * delta; o += S[i] * qs[k0 + i]; }
        __syncthreads();
        part[half][j] = o;
        __syncthreads();
        if (half == 0) out[((int64_t)tok * Hv + hv) * 128 + j] = part[0][j] + part[1][j];
        __syncthreads();
    }
}

// Gated RMSNorm of the DeltaNet output: core (rounded to bf16 first) -> rms -> (.bf16 * w).bf16 * silu(z) -> bf16.
// One warp per (token, head) of 128 dims; z at z_off of a fused row (stride z_ts).
__global__ void k_gdn_norm(const float* __restrict__ core, const bf16* __restrict__ zrow, int64_t z_ts, int z_off,
                           const bf16* __restrict__ w, float eps, bf16* __restrict__ out, int L, int Hv) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)L * Hv) return;
    int t = gw / Hv, h = gw % Hv;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) x[j] = round_bf16(core[gw * 128 + lane * 4 + j]);
    float ss = warp_sum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    float r = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int d = lane * 4 + j;
        float n = round_bf16(round_bf16(x[j] * r) * bf2f(w[d]));
        float z = bf2f(zrow[(int64_t)t * z_ts + z_off + h * 128 + d]);
        out[(int64_t)t * Hv * 128 + h * 128 + d] = f2bf(n * silu_f(z));
    }
}

// SwiGLU on separate gate/up columns of a fused row [gate (F) | up (F)] with bf16 rounding of silu(gate): out [M, F]
__global__ void k_swiglu_round_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t M, int F) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * F) return;
    int64_t m = i / F;
    int f = i % F;
    float g = round_bf16(silu_f(bf2f(x[m * 2 * F + f])));
    y[i] = f2bf(g * bf2f(x[m * 2 * F + F + f]));
}

// RMSNorm with (1 + w), bf16 -> bf16, one rounding (Qwen3_5RMSNorm)
__global__ void k_rmsnorm1p_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int M, int K, float eps, const bf16* __restrict__ w) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) { float v = bf2f(x[m * K + i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / K + eps);
    for (int i = threadIdx.x; i < K; i += blockDim.x) y[m * K + i] = f2bf(bf2f(x[m * K + i]) * r * (1.0f + bf2f(w[i])));
}

// Embedding gather from a bf16 table: out[i, :] = table[tok[i], :]
__global__ void k_embed_bf16(const bf16* __restrict__ table, const int* __restrict__ tok, bf16* __restrict__ out, int L, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)L * K) return;
    int64_t r = i / K;
    int c = i % K;
    out[i] = table[(int64_t)tok[r] * K + c];
}

// ------------------------------------------------------------------------------------------------ FLUX.2 VAE
// GroupNorm statistics over an NHWC bf16 plane: per-block partial (sum x, sum x^2) per group, reduced in order.
// grid (ceil(npix / 256), 1), block 256: each thread sums one pixel's channels per group.
__global__ void k_groupnorm_stats(const bf16* __restrict__ x, int64_t npix, int C, int G, float* __restrict__ partial) {
    __shared__ float s1[32 * 8], s2[32 * 8];
    const int cpg = C / G;
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bf16* px = x + p * C;
    for (int g = 0; g < G; ++g) {
        float a = 0.f, b = 0.f;
        if (p < npix)
            for (int c = 0; c < cpg; ++c) { float v = bf2f(px[g * cpg + c]); a += v; b += v * v; }
        a = warp_sum(a);
        b = warp_sum(b);
        if (lane == 0) { s1[g * 8 + warp] = a; s2[g * 8 + warp] = b; }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < G; i += blockDim.x) {
        float a = 0, b = 0;
        for (int w = 0; w < 8; ++w) { a += s1[i * 8 + w]; b += s2[i * 8 + w]; }
        partial[(int64_t)blockIdx.x * G * 2 + i * 2] = a;
        partial[(int64_t)blockIdx.x * G * 2 + i * 2 + 1] = b;
    }
}
// fixed-order reduction of the per-block partials into double sums (deterministic)
__global__ void k_groupnorm_reduce(const float* __restrict__ partial, int blocks, int G, double* __restrict__ sums) {
    int i = threadIdx.x;
    if (i >= 2 * G) return;
    double s = 0;
    for (int b = 0; b < blocks; ++b) s += partial[(int64_t)b * G * 2 + i];
    sums[i] = s;
}
// y = (x - mean_g) * rstd_g * gamma[c] + beta[c], optional SiLU (bf16 out)
__global__ void k_groupnorm_apply(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t npix, int C, int G, const double* __restrict__ sums,
                                  const float* __restrict__ gamma, const float* __restrict__ beta, float eps, int silu) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix * C) return;
    int c = i % C;
    int g = c / (C / G);
    double n = (double)npix * (C / G);
    double mean = sums[g * 2] / n;
    double var = sums[g * 2 + 1] / n - mean * mean;
    float rstd = (float)(1.0 / sqrt(fmax(var, 0.0) + (double)eps));
    float v = (bf2f(x[i]) - (float)mean) * rstd * gamma[c] + beta[c];
    if (silu) v = silu_f(v);
    y[i] = f2bf(v);
}

// RGB f32 [H, W, 3] in [0,1] -> NHWC bf16 [H, W, 8]: (2r-1, 2g-1, 2b-1, 0...)
__global__ void k_rgb_to_plane8(const float* __restrict__ rgb, bf16* __restrict__ out, int64_t npix) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    bf16* o = out + i * 8;
#pragma unroll
    for (int c = 0; c < 3; ++c) o[c] = f2bf(2.f * rgb[i * 3 + c] - 1.f);
#pragma unroll
    for (int c = 3; c < 8; ++c) o[c] = f2bf(0.f);
}

// DiT tokens (f32 [h*w, 128], normalized) -> VAE latent plane bf16 [2h, 2w, 32]: z = x * sqrt(var + eps) + mean,
// channel 4c + 2dy + dx of token (y, x) -> pixel (2y + dy, 2x + dx), channel c.
__global__ void k_latent_unpack(const float* __restrict__ tok, bf16* __restrict__ plane, int h, int w,
                                const float* __restrict__ mean, const float* __restrict__ var, float eps) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)h * w * 128) return;
    int ch = i % 128;
    int64_t t = i / 128;
    int y = t / w, x = t % w;
    int c = ch >> 2, dy = (ch >> 1) & 1, dx = ch & 1;
    float z = tok[i] * sqrtf(var[ch] + eps) + mean[ch];
    plane[(((int64_t)(2 * y + dy)) * (2 * w) + (2 * x + dx)) * 32 + c] = f2bf(z);
}
// Encoder mean plane (bf16 [2h, 2w, Cp] with the 32 mean channels first) -> normalized tokens f32 [h*w, 128]
__global__ void k_latent_pack(const bf16* __restrict__ plane, int Cp, float* __restrict__ tok, int h, int w,
                              const float* __restrict__ mean, const float* __restrict__ var, float eps) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)h * w * 128) return;
    int ch = i % 128;
    int64_t t = i / 128;
    int y = t / w, x = t % w;
    int c = ch >> 2, dy = (ch >> 1) & 1, dx = ch & 1;
    float z = bf2f(plane[(((int64_t)(2 * y + dy)) * (2 * w) + (2 * x + dx)) * Cp + c]);
    tok[i] = (z - mean[ch]) / sqrtf(var[ch] + eps);
}

}  // extern "C"
