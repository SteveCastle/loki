// Sampler update kernels (flow-matching / res_multistep) on f32 latents.
#include "common.cuh"

extern "C" {

// out = a*x + b*d + c*o   (o may be null; out may alias x)
__global__ void k_axpby3_f32(float* out, const float* x, const float* d, const float* o, int64_t n, float a, float b, float c) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a * x[i] + b * d[i];
    if (o) v += c * o[i];
    out[i] = v;
}

}  // extern "C"
