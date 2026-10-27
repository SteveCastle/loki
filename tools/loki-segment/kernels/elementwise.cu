// Elementwise, conversion and normalization kernels.
#include "common.cuh"

extern "C" {

__global__ void k_f16_to_bf16(const __half* __restrict__ x, bf16* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf(__half2float(x[i]));
}
__global__ void k_f16_to_f32(const __half* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __half2float(x[i]);
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

DEVI float act_f(float v, int act) {
    if (act == 1) return gelu_tanh_f(v);
    if (act == 2) return silu_f(v);
    if (act == 3) return gelu_erf_f(v);
    if (act == 4) return fmaxf(v, 0.f);
    if (act == 5) return 1.f / (1.f + __expf(-v));
    return v;
}
__global__ void k_act_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t n, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf(act_f(bf2f(x[i]), act));
}
__global__ void k_act_f32(const float* __restrict__ x, float* __restrict__ y, int64_t n, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = act_f(x[i], act);
}
// out = act(a + b) (b may be null; b broadcast over rows when b_rows == 1). a,b f32 or bf16 by flags; out bf16/f32.
// ta/tb/to: 0 = f32, 1 = bf16.
__global__ void k_add_any(const void* __restrict__ a, int ta, const void* __restrict__ b, int tb, int64_t b_period,
                          void* __restrict__ out, int to, int64_t n, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = ta ? bf2f(((const bf16*)a)[i]) : ((const float*)a)[i];
    if (b) {
        int64_t j = b_period > 0 ? i % b_period : i;
        v += tb ? bf2f(((const bf16*)b)[j]) : ((const float*)b)[j];
    }
    v = act_f(v, act);
    if (to) ((bf16*)out)[i] = f2bf(v);
    else ((float*)out)[i] = v;
}

}  // extern "C"

// ---------------------------------------------------------------------------
// LayerNorm over the last dim. One block (256 threads) per row; two-pass stats in f32.
//   y = LN(x) * w + b          (w, b f32, nullable)
//   y2 = y + pos               (optional second output, pos f32 [M, N] or broadcast over rows if pos_rows == 1)
template <typename TI, typename TO>
__device__ void layernorm_row(const TI* __restrict__ x, TO* __restrict__ y, int N, float eps,
                              const float* __restrict__ w, const float* __restrict__ b,
                              const float* __restrict__ pos, TO* __restrict__ y2, float* red) {
    const int tid = threadIdx.x;
    float s = 0.f, ss = 0.f;
    for (int i = tid; i < N; i += blockDim.x) s += (float)x[i];
    float mean = block_sum(s, red) / N;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = (float)x[i] - mean;
        ss += v * v;
    }
    float var = block_sum(ss, red) / N;
    float rstd = rsqrtf(var + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        float v = ((float)x[i] - mean) * rstd;
        if (w) v = v * w[i];
        if (b) v += b[i];
        y[i] = (TO)v;
        if (y2) y2[i] = (TO)(v + pos[i]);
    }
}

#define LN_KERNEL(NAME, TI, TO)                                                                                  \
    extern "C" __global__ void NAME(const TI* x, TO* y, int M, int N, float eps, const float* w, const float* b, \
                                    const float* pos, int pos_rows, TO* y2) {                                     \
        __shared__ float red[32];                                                                                \
        int64_t m = blockIdx.x;                                                                                  \
        if (m >= M) return;                                                                                      \
        const float* pr = pos ? pos + (pos_rows == 1 ? 0 : m * N) : nullptr;                                     \
        layernorm_row<TI, TO>(x + m * N, y + m * N, N, eps, w, b, pr, y2 ? y2 + m * N : nullptr, red);           \
    }
LN_KERNEL(k_layernorm_f32_bf16, float, bf16)
LN_KERNEL(k_layernorm_f32_f32, float, float)
LN_KERNEL(k_layernorm_bf16_bf16, bf16, bf16)
LN_KERNEL(k_layernorm_bf16_f32, bf16, float)

extern "C" {

// GroupNorm on NHWC bf16 [P, C] with G groups: stats over all pixels x C/G channels (f64 accumulation).
// stats[g*2 + 0/1] = sum, sumsq (double), zeroed by the caller.
__global__ void k_groupnorm_stats(const bf16* __restrict__ x, int64_t P, int C, int G, double* __restrict__ stats) {
    __shared__ double sh[2 * 32];
    const int cpg = C / G;
    for (int i = threadIdx.x; i < 2 * G; i += blockDim.x) sh[i] = 0.0;
    __syncthreads();
    double ls[32], lss[32];
    for (int g = 0; g < G; ++g) { ls[g] = 0.0; lss[g] = 0.0; }
    int64_t total = P * C;
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (int64_t)gridDim.x * blockDim.x) {
        int c = (int)(i % C);
        float v = bf2f(x[i]);
        int g = c / cpg;
        ls[g] += v;
        lss[g] += (double)v * v;
    }
    for (int g = 0; g < G; ++g) {
        atomicAdd(&sh[2 * g], ls[g]);
        atomicAdd(&sh[2 * g + 1], lss[g]);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 2 * G; i += blockDim.x) atomicAdd(&stats[i], sh[i]);
}
// y = act((x - mean_g) * rstd_g * w[c] + b[c])
__global__ void k_groupnorm_apply(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t P, int C, int G,
                                  const double* __restrict__ stats, float eps, const float* __restrict__ w,
                                  const float* __restrict__ b, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= P * C) return;
    int c = (int)(i % C);
    int g = c / (C / G);
    double cnt = (double)P * (C / G);
    double mean = stats[2 * g] / cnt;
    double var = stats[2 * g + 1] / cnt - mean * mean;
    float rstd = rsqrtf((float)var + eps);
    float v = (bf2f(x[i]) - (float)mean) * rstd * w[c] + b[c];
    y[i] = f2bf(act_f(v, act));
}

}  // extern "C"
