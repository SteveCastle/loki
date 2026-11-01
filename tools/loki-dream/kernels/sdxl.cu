// SDXL (UNet + CLIP) kernels: GEGLU feed-forward, CLIP-L's quick GELU, per-channel adds of the timestep embedding.
#include "common.cuh"

extern "C" {

// GEGLU: in [M, 2H] (hidden | gate) -> out [M, H] = hidden * gelu_erf(gate)  (diffusers GEGLU, LDM GEGLU)
__global__ void k_geglu_bf16(const bf16* __restrict__ x, bf16* __restrict__ y, int64_t M, int H) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * H) return;
    int64_t m = i / H;
    int h = i % H;
    float v = bf2f(x[m * 2 * H + h]);
    float g = bf2f(x[m * 2 * H + H + h]);
    y[i] = f2bf(v * gelu_erf_f(g));
}

// quick GELU in place: x * sigmoid(1.702 x)  (OpenAI CLIP)
__global__ void k_quick_gelu_bf16(bf16* __restrict__ x, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bf2f(x[i]);
    x[i] = f2bf(v / (1.0f + __expf(-1.702f * v)));
}

// x[m, n] += v[n]  (bf16 plane, bf16 row vector)
__global__ void k_add_rowvec_bf16(bf16* __restrict__ x, const bf16* __restrict__ v, int64_t M, int N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M * N) x[i] = f2bf(bf2f(x[i]) + bf2f(v[i % N]));
}

}  // extern "C"
