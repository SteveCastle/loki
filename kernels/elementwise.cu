// Elementwise, normalization and embedding kernels.
#include "common.cuh"

extern "C" {

__global__ void k_smoke_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void k_f32_to_bf16(const float* __restrict__ x, bf16* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf(x[i]);
}
__global__ void k_bf16_to_f32(const bf16* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = bf2f(x[i]);
}
__global__ void k_fill_f32(float* y, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = v;
}
__global__ void k_fill_bf16(bf16* y, float v, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf(v);
}

// y = act(x) in place / out of place, bf16
__global__ void k_act_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t n, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf2f(x[i]);
    if (act == 1) v = gelu_tanh_f(v);
    else if (act == 2) v = silu_f(v);
    else if (act == 3) v = gelu_erf_f(v);
    y[i] = f2bf(v);
}
__global__ void k_act_f32(const float* __restrict__ x, float* __restrict__ y, int64_t n, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if (act == 1) v = gelu_tanh_f(v);
    else if (act == 2) v = silu_f(v);
    else if (act == 3) v = gelu_erf_f(v);
    y[i] = v;
}

// y[m, :] = x[m, :] + b[:]   (f32)
__global__ void k_add_bias_f32(float* __restrict__ x, const float* __restrict__ b, int M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)M * N) x[i] += b[i % N];
}
// x += y (f32)
__global__ void k_add_f32(float* __restrict__ x, const float* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
// x += y (bf16)
__global__ void k_add_bf16(bf16* __restrict__ x, const bf16* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = f2bf(bf2f(x[i]) + bf2f(y[i]));
}
// x[m,n] += y[m,n] * g[n]   (bf16 x,y; f32 gate)
__global__ void k_addcmul_bf16(bf16* __restrict__ x, const bf16* __restrict__ y, const float* __restrict__ g, int64_t M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) x[i] = f2bf(bf2f(x[i]) + bf2f(y[i]) * g[i % N]);
}
// swiglu: in [M, 2H] -> out [M, H]; silu(gate)*up (bf16)
__global__ void k_swiglu_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t M, int H) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * H) return;
    int64_t m = i / H; int h = i % H;
    float g = bf2f(x[m * 2 * H + h]);
    float u = bf2f(x[m * 2 * H + H + h]);
    y[i] = f2bf(silu_f(g) * u);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// LayerNorm family. One block (256 threads) per row. Row length N <= 8192 handled
// by looping. Stats in fp32.
//   mode 0: y = LN(x)*w + b                 (affine; w,b may be null for no affine)
//   mode 1: y = LN(x)*(1+scale[sel(m)])     (adaln, no shift). scale0 for rows < prefix_len else scale1
// in/out dtype selected by template.
template <typename TI, typename TO>
__device__ void layernorm_row(const TI* __restrict__ x, TO* __restrict__ y, int N, float eps,
                              const bf16* __restrict__ w, const bf16* __restrict__ b,
                              const float* __restrict__ scale, float* red) {
    const int tid = threadIdx.x;
    float s = 0.f, ss = 0.f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = (float)x[i];
        s += v;
    }
    float mean = block_sum(s, red) / N;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = (float)x[i] - mean;
        ss += v * v;
    }
    float var = block_sum(ss, red) / N;
    float rstd = rsqrtf(var + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float v = ((float)x[i] - mean) * rstd;
        if (w) v = v * bf2f(w[i]) + (b ? bf2f(b[i]) : 0.f);
        if (scale) v = v * (1.0f + scale[i]);
        y[i] = (TO)v;
    }
}

extern "C" {
__global__ void k_layernorm_bf16_bf16(const bf16* x, bf16* y, int M, int N, float eps, const bf16* w, const bf16* b) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    layernorm_row<bf16, bf16>(x + m * N, y + m * N, N, eps, w, b, nullptr, red);
}
__global__ void k_layernorm_f32_bf16(const float* x, bf16* y, int M, int N, float eps, const bf16* w, const bf16* b) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    layernorm_row<float, bf16>(x + m * N, y + m * N, N, eps, w, b, nullptr, red);
}
__global__ void k_layernorm_f32_f32(const float* x, float* y, int M, int N, float eps, const bf16* w, const bf16* b) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    layernorm_row<float, float>(x + m * N, y + m * N, N, eps, w, b, nullptr, red);
}
// adaln: scale0 applies to rows < prefix_len, scale1 to the rest (both [N] f32)
__global__ void k_adaln_bf16(const bf16* x, bf16* y, int M, int N, float eps, const float* scale0, const float* scale1, int prefix_len) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    const float* sc = (m < prefix_len) ? scale0 : scale1;
    layernorm_row<bf16, bf16>(x + m * N, y + m * N, N, eps, nullptr, nullptr, sc, red);
}

}  // extern "C"

// ---------------------------------------------------------------------------
// RMSNorm: y = x * rsqrt(mean(x^2)+eps) * (w (+1 if add_one))
template <typename TI, typename TO>
__device__ void rmsnorm_row(const TI* __restrict__ x, TO* __restrict__ y, int N, float eps, const bf16* __restrict__ w, int add_one, float* red) {
    const int tid = threadIdx.x;
    float ss = 0.f;
    for (int i = tid; i < N; i += blockDim.x) { float v = (float)x[i]; ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / N + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float v = (float)x[i] * r;
        if (w) v *= (bf2f(w[i]) + (add_one ? 1.0f : 0.f));
        y[i] = (TO)v;
    }
}
extern "C" {
__global__ void k_rmsnorm_f32_f32(const float* x, float* y, int M, int N, float eps, const bf16* w, int add_one) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    rmsnorm_row<float, float>(x + m * N, y + m * N, N, eps, w, add_one, red);
}
__global__ void k_rmsnorm_f32_bf16(const float* x, bf16* y, int M, int N, float eps, const bf16* w, int add_one) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    rmsnorm_row<float, bf16>(x + m * N, y + m * N, N, eps, w, add_one, red);
}
__global__ void k_rmsnorm_bf16_bf16(const bf16* x, bf16* y, int M, int N, float eps, const bf16* w, int add_one) {
    __shared__ float red[32];
    int64_t m = blockIdx.x; if (m >= M) return;
    rmsnorm_row<bf16, bf16>(x + m * N, y + m * N, N, eps, w, add_one, red);
}

// ---------------------------------------------------------------------------
// DiT fused rms_rope: q [M, H, 128] (token stride q_ts), k [M, H, 128] (token stride k_ts), bf16 in place.
// Per head: RMSNorm over 128 with weight (eps), then interleaved-pair rope using table rope[M, 64, 2] = (cos, sin)
//   out[2i]   = cos*x[2i] - sin*x[2i+1]
//   out[2i+1] = sin*x[2i] + cos*x[2i+1]
// One warp per (token, head, q-or-k): lane handles 4 consecutive dims (2 pairs).
__global__ void k_rms_rope_dit(bf16* __restrict__ q, int64_t q_ts, bf16* __restrict__ k, int64_t k_ts, int M, int H,
                               const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,
                               const float* __restrict__ rope) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;  // global warp
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * H * 2) return;
    int64_t m = gw / (2 * H);
    int rem = gw % (2 * H);
    int which = rem / H, h = rem % H;
    bf16* p = (which == 0 ? q + m * q_ts : k + m * k_ts) + h * 128 + lane * 4;
    const bf16* w = (which == 0 ? wq : wk) + lane * 4;
    float v0 = bf2f(p[0]), v1 = bf2f(p[1]), v2 = bf2f(p[2]), v3 = bf2f(p[3]);
    float ss = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    ss = warp_sum(ss);
    float r = rsqrtf(ss / 128.f + eps);
    float n0 = v0 * r * bf2f(w[0]);
    float n1 = v1 * r * bf2f(w[1]);
    float n2 = v2 * r * bf2f(w[2]);
    float n3 = v3 * r * bf2f(w[3]);
    const float* rp = rope + m * 128 + lane * 4;  // 64 pairs * 2 = 128 floats per token
    float c0 = rp[0], s0 = rp[1], c1 = rp[2], s1 = rp[3];
    p[0] = f2bf(c0 * n0 - s0 * n1);
    p[1] = f2bf(s0 * n0 + c0 * n1);
    p[2] = f2bf(c1 * n2 - s1 * n3);
    p[3] = f2bf(s1 * n2 + c1 * n3);
}

// LLM (Qwen3) fused qk norm + split-half rope.  q: [M, Hq, 128] with token stride q_ts, k: [M, Hk, 128]
// with token stride k_ts (elements), bf16, in place.
// RMSNorm over head dim with weights wq/wk (eps), then
//   out[i]      = x[i]*cos[i]   - x[i+64]*sin[i]       (i < 64)
//   out[i+64]   = x[i+64]*cos[i] + x[i]*sin[i]
// rope table: [M, 64, 2] (cos, sin) f32.  One warp per (token, head): lane handles dims lane*2, lane*2+1 and +64.
__global__ void k_qk_norm_rope_llm(bf16* __restrict__ q, int64_t q_ts, bf16* __restrict__ k, int64_t k_ts, int M, int Hq, int Hk,
                                   const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,
                                   const float* __restrict__ rope) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int64_t total = (int64_t)M * (Hq + Hk);
    if (gw >= total) return;
    int64_t m = gw / (Hq + Hk);
    int h = gw % (Hq + Hk);
    bf16* p; const bf16* w;
    if (h < Hq) { p = q + m * q_ts + h * 128; w = wq; }
    else { p = k + m * k_ts + (h - Hq) * 128; w = wk; }
    int i0 = lane * 2, i1 = lane * 2 + 1;
    float a0 = bf2f(p[i0]), a1 = bf2f(p[i1]), b0 = bf2f(p[i0 + 64]), b1 = bf2f(p[i1 + 64]);
    float ss = a0 * a0 + a1 * a1 + b0 * b0 + b1 * b1;
    ss = warp_sum(ss);
    float r = rsqrtf(ss / 128.f + eps);
    a0 *= r * bf2f(w[i0]); a1 *= r * bf2f(w[i1]); b0 *= r * bf2f(w[i0 + 64]); b1 *= r * bf2f(w[i1 + 64]);
    const float* rp = rope + m * 128;
    float c0 = rp[i0 * 2], s0 = rp[i0 * 2 + 1], c1 = rp[i1 * 2], s1 = rp[i1 * 2 + 1];
    p[i0] = f2bf(a0 * c0 - b0 * s0);
    p[i1] = f2bf(a1 * c1 - b1 * s1);
    p[i0 + 64] = f2bf(b0 * c0 + a0 * s0);
    p[i1 + 64] = f2bf(b1 * c1 + a1 * s1);
}

// Vision position embedding: x[tok, :] += sum_j w[tok][j] * table[idx[tok][j], :]   (x f32 [M,N], table bf16)
__global__ void k_vision_pos_embed(float* __restrict__ x, const bf16* __restrict__ table, const int* __restrict__ idx,
                                   const float* __restrict__ w, int M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * N) return;
    int64_t m = i / N; int c = i % N;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += w[m * 4 + j] * bf2f(table[(int64_t)idx[m * 4 + j] * N + c]);
    x[i] += acc;
}

// Vision rope (split-half, head dim D even, rotary over full D): qkv [M, 3, H, D] bf16 in place for q and k.
// rope table [M, D/2, 2] (cos, sin). One thread per (token, head, i<D/2).
__global__ void k_rope_vision(bf16* __restrict__ qkv, int M, int H, int D, const float* __restrict__ rope) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int half = D / 2;
    int64_t per_tok = (int64_t)2 * H * half;  // q and k
    if (idx >= (int64_t)M * per_tok) return;
    int64_t m = idx / per_tok;
    int rem = idx % per_tok;
    int which = rem / (H * half);  // 0 q, 1 k
    rem = rem % (H * half);
    int h = rem / half, i = rem % half;
    bf16* p = qkv + ((m * 3 + which) * H + h) * D;
    float c = rope[(m * half + i) * 2], s = rope[(m * half + i) * 2 + 1];
    float a = bf2f(p[i]), b = bf2f(p[i + half]);
    p[i] = f2bf(a * c - b * s);
    p[i + half] = f2bf(b * c + a * s);
}

// ---------------------------------------------------------------------------
// Hadamard-256 un-rotation of f32 rows in place (ConvRot inverse == forward, H symmetric).
// Used for embedding rows. One block of 256 threads per row of length K (K % 256 == 0).
}  // extern "C"
__device__ void hadamard256_inplace_smem(float* g /* 256 floats, shared */, int t /* 0..255 */) {
    // radix-4 butterflies over the 4 base-4 digits of the index.
    // h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]
#pragma unroll
    for (int st = 1; st < 256; st <<= 2) {
        __syncthreads();
        if (t < 64) {
            // each of 64 threads handles one group of 4 indices differing in digit `st`
            int lo = t % st;
            int hi = (t / st) * st * 4;
            int base = hi + lo;
            float x0 = g[base], x1 = g[base + st], x2 = g[base + 2 * st], x3 = g[base + 3 * st];
            g[base] = x0 + x1 + x2 - x3;
            g[base + st] = x0 + x1 - x2 + x3;
            g[base + 2 * st] = x0 - x1 + x2 + x3;
            g[base + 3 * st] = -x0 + x1 + x2 + x3;
        }
    }
    __syncthreads();
    g[t] *= 0.0625f;  // 1/sqrt(256)
    __syncthreads();
}

extern "C" {
// Embedding lookup for int8 convrot tables: out[i, :] = unrotate(q[tok[i], :] * scale[tok[i]])  -> f32
__global__ void k_embed_int8_convrot(const int8_t* __restrict__ table, const float* __restrict__ scale,
                                     const int* __restrict__ tokens, float* __restrict__ out, int K) {
    __shared__ float g[256];
    int row = blockIdx.x;
    int tok = tokens[row];
    const int8_t* src = table + (int64_t)tok * K;
    float sc = scale[tok];
    for (int grp = 0; grp < K / 256; ++grp) {
        g[threadIdx.x] = (float)src[grp * 256 + threadIdx.x] * sc;
        hadamard256_inplace_smem(g, threadIdx.x);
        out[(int64_t)row * K + grp * 256 + threadIdx.x] = g[threadIdx.x];
    }
}

// Scatter-add deepstack features into image positions: x[pos0 + i, :] += ds[i, :]  (f32 x, bf16/f32 ds)
__global__ void k_add_rows_f32(float* __restrict__ x, int N, int pos0, const float* __restrict__ ds, int rows) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * N) return;
    int64_t r = i / N; int c = i % N;
    x[(pos0 + r) * (int64_t)N + c] += ds[i];
}

// Strided row copy (bf16): dst[r*dst_ts + j] = src[r*src_ts + j], j < width
__global__ void k_copy_rows_bf16(bf16* __restrict__ dst, int64_t dst_ts, const bf16* __restrict__ src, int64_t src_ts, int width, int rows) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t w8 = width / 8;
    if (i >= (int64_t)rows * w8) return;
    int64_t r = i / w8; int j = (i % w8) * 8;
    *reinterpret_cast<uint4*>(dst + r * dst_ts + j) = *reinterpret_cast<const uint4*>(src + r * src_ts + j);
}

// Gather rows of int8 [rows, K]: out[i,:] = in[idx[i],:]  (16B vectors; K % 16 == 0)
__global__ void k_gather_rows_i8(const int8_t* __restrict__ in, const int* __restrict__ idx, int8_t* __restrict__ out, int rows, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t k16 = K / 16;
    if (i >= (int64_t)rows * k16) return;
    int64_t r = i / k16; int c = (i % k16) * 16;
    *reinterpret_cast<uint4*>(out + r * K + c) = *reinterpret_cast<const uint4*>(in + (int64_t)idx[r] * K + c);
}

// Gather rows: out[i,:] = in[idx[i],:]  (f32)
__global__ void k_gather_rows_f32(const float* __restrict__ in, const int* __restrict__ idx, float* __restrict__ out, int rows, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * N) return;
    int64_t r = i / N; int c = i % N;
    out[i] = in[(int64_t)idx[r] * N + c];
}

// Patchify image for the vision tower: img f32 [Hp, Wp, 3] (values already normalized)
// -> patches bf16 [Ntok, 1536] with token order (block_row, block_col, intra_row, intra_col)
// and feature order (c, t(2), py(16), px(16)); the temporal pair duplicates the frame.
__global__ void k_vision_patchify(const float* __restrict__ img, bf16* __restrict__ out, int Hp, int Wp) {
    int gh = Hp / 16, gw = Wp / 16;
    int64_t ntok = (int64_t)gh * gw;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ntok * 1536) return;
    int64_t tok = idx / 1536;
    int f = idx % 1536;
    int c = f / 512, rem = f % 512;
    int py = (rem % 256) / 16, px = rem % 16;  // t = rem / 256 ignored (duplicate)
    int mw = gw / 2;
    int br = tok / (mw * 4), r2 = tok % (mw * 4);
    int bc = r2 / 4, r3 = r2 % 4;
    int ir = r3 / 2, ic = r3 % 2;
    int prow = br * 2 + ir, pcol = bc * 2 + ic;
    int y = prow * 16 + py, x = pcol * 16 + px;
    out[idx] = f2bf(img[((int64_t)y * Wp + x) * 3 + c]);
}

}  // extern "C"
