// Fused activation -> ConvRot (Hadamard-256 per group) -> per-row int8 quantization.
//
//   in : rows of width Kin (f32 or bf16)
//   act: 0 none (K = Kin), 1 gelu_tanh (K = Kin), 2 swiglu (K = Kin/2, row = [gate|up]),
//        3 rmsnorm with weight w[K], eps (K = Kin)
//        4 adaln: LayerNorm(x, no affine, eps) * (1 + scale[K]) with scale f32 (passed as `w`)
//        5 situ (MOSS-Image): K = Kin/2, row = [hidden | gate] -> situ(hidden, gate)
//        6 modulated rmsnorm (MOSS-Image): rms(x) * w * (1 + scale) + shift, see mod_rms in common.cuh
//   out: int8 [M, K], scale f32 [M]   (scale = absmax/127, q = rint(x/scale))
//
// One block of 256 threads per row; the row lives in dynamic shared memory as f32 (K*4 bytes).
#include "common.cuh"

template <typename TI>
__device__ void quant_row(const TI* __restrict__ in, int Kin, int act, const bf16* __restrict__ w, float eps,
                          int8_t* __restrict__ out, float* __restrict__ scale_out, float* row,
                          const float* __restrict__ adascale = nullptr, const float* __restrict__ adashift = nullptr) {
    __shared__ float red[32];
    const int t = threadIdx.x;
    const int K = (act == 2 || act == 5) ? Kin / 2 : Kin;
    // 1. load + activation
    if (act == 5) {
        for (int i = t; i < K; i += 256) row[i] = situ_bf16((float)in[i], (float)in[K + i]);
    } else if (act == 6) {
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; ss += v * v; }
        float r = rsqrtf(block_sum(ss, red) / K + eps);
        for (int i = t; i < K; i += 256) row[i] = mod_rms(row[i], r, w ? bf2f(w[i]) : 1.f, adascale ? adascale[i] : 0.f, adashift ? adashift[i] : 0.f, adascale != nullptr);
    } else if (act == 2) {
        for (int i = t; i < K; i += 256) {
            float g = (float)in[i], u = (float)in[K + i];
            row[i] = silu_f(g) * u;
        }
    } else if (act == 1) {
        for (int i = t; i < K; i += 256) row[i] = gelu_tanh_f((float)in[i]);
    } else if (act == 3) {
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; ss += v * v; }
        float r = rsqrtf(block_sum(ss, red) / K + eps);
        for (int i = t; i < K; i += 256) row[i] = row[i] * r * bf2f(w[i]);
    } else if (act == 4) {
        float sum = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; sum += v; }
        float mean = block_sum(sum, red) / K;
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = row[i] - mean; ss += v * v; }
        float rstd = rsqrtf(block_sum(ss, red) / K + eps);
        // ComfyUI's adaln kernel hands the modulated activation to the int8 quantizer as bf16
        for (int i = t; i < K; i += 256) row[i] = round_bf16((row[i] - mean) * rstd * (1.0f + adascale[i]));
    } else {
        for (int i = t; i < K; i += 256) row[i] = (float)in[i];
    }
    __syncthreads();
    // 2. Hadamard-256 per group of 256: 4 radix-4 stages. Each thread handles index t within
    //    every group; a stage needs 64 butterflies per group -> thread t<64*ngroups works.
    const int ngroups = K / 256;
#pragma unroll
    for (int st = 1; st < 256; st <<= 2) {
        for (int job = t; job < 64 * ngroups; job += 256) {
            int grp = job / 64, j = job % 64;
            int lo = j % st;
            int hi = (j / st) * st * 4;
            float* g = row + grp * 256;
            int base = hi + lo;
            float x0 = g[base], x1 = g[base + st], x2 = g[base + 2 * st], x3 = g[base + 3 * st];
            g[base] = x0 + x1 + x2 - x3;
            g[base + st] = x0 + x1 - x2 + x3;
            g[base + 2 * st] = x0 - x1 + x2 + x3;
            g[base + 3 * st] = -x0 + x1 + x2 + x3;
        }
        __syncthreads();
    }
    // 3. absmax (with the 1/16 normalization folded in)
    float amax = 0.f;
    for (int i = t; i < K; i += 256) amax = fmaxf(amax, fabsf(row[i]));
    amax = block_max(amax, red) * 0.0625f;
    float scale = amax / 127.0f;
    float inv = (scale > 0.f) ? (1.0f / scale) : 0.f;
    if (t == 0) scale_out[0] = (scale > 0.f) ? scale : 1.17549435e-38f;
    // 4. quantize
    for (int i = t; i < K; i += 256) {
        float v = row[i] * 0.0625f * inv;
        int q = __float2int_rn(v);
        q = max(-128, min(127, q));
        out[i] = (int8_t)q;
    }
}

extern "C" {
__global__ void k_quant_rows_f32(const float* __restrict__ in, int M, int Kin, int act, const bf16* w, float eps,
                                 int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    const int K = (act == 2 || act == 5) ? Kin / 2 : Kin;
    quant_row<float>(in + m * Kin, Kin, act, w, eps, out + m * K, scale + m, row);
}
__global__ void k_quant_rows_bf16(const bf16* __restrict__ in, int M, int Kin, int act, const bf16* w, float eps,
                                  int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    const int K = (act == 2 || act == 5) ? Kin / 2 : Kin;
    quant_row<bf16>(in + m * Kin, Kin, act, w, eps, out + m * K, scale + m, row);
}
// adaln + quantize: scale f32 [K]
__global__ void k_adaln_quant_bf16(const bf16* __restrict__ in, int M, int K, float eps, const float* __restrict__ adascale,
                                   int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    quant_row<bf16>(in + m * K, K, 4, nullptr, eps, out + m * K, scale + m, row, adascale);
}
// MOSS-Image modulated rmsnorm + quantize: w bf16 [K] (norm weight), scale/shift f32 [K] (bf16 values)
__global__ void k_modrms_quant_bf16(const bf16* __restrict__ in, int M, int K, float eps, const bf16* __restrict__ w,
                                    const float* __restrict__ scale, const float* __restrict__ shift,
                                    int8_t* __restrict__ out, float* __restrict__ oscale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    quant_row<bf16>(in + m * K, K, 6, w, eps, out + m * K, oscale + m, row, scale, shift);
}
}
