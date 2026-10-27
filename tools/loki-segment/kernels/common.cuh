// Shared device helpers for the 4kify inference engine.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;
typedef __nv_bfloat162 bf162;

#define DEVI __device__ __forceinline__

DEVI float bf2f(bf16 x) { return __bfloat162float(x); }
DEVI bf16 f2bf(float x) { return __float2bfloat16(x); }
DEVI float round_bf16(float x) { return __bfloat162float(__float2bfloat16(x)); }

DEVI float silu_f(float x) { return x / (1.0f + __expf(-x)); }
DEVI float gelu_tanh_f(float x) {
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float u = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + tanhf(u));
}
DEVI float gelu_erf_f(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }

DEVI float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
DEVI float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// Block-wide sum for blockDim.x threads (<= 1024). `red` must hold >= 32 floats.
DEVI float block_sum(float v, float* red) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float r = (lane < nw) ? red[lane] : 0.f;
    r = warp_sum(r);
    return r;
}
DEVI float block_max(float v, float* red) {
    v = warp_max(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int nw = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float r = (lane < nw) ? red[lane] : -INFINITY;
    r = warp_max(r);
    return r;
}

// ---------------------------------------------------------------- async copy
DEVI uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

DEVI void cp_async_16(uint32_t smem_addr, const void* gptr, bool pred = true) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr), "l"(gptr), "r"(sz));
}
DEVI void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
DEVI void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// ---------------------------------------------------------------- ldmatrix
DEVI void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
DEVI void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
DEVI void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}
DEVI void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// ---------------------------------------------------------------- mma
// D(16x8,f32) += A(16x16,bf16,row) * B(16x8,bf16,col)
DEVI void mma_bf16_16816(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D(16x8,s32) += A(16x32,s8,row) * B(32x8,s8,col)
DEVI void mma_s8_16832(int32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// D(16x8,f32) += A(16x32,e4m3,row) * B(32x8,e4m3,col)   (sm_89+)
DEVI void mma_e4m3_16832(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Hardware exp2 (MUFU.EX2), ~2 ulp; plenty for softmax weights.
DEVI float fast_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// exp2 on the FMA pipe (degree-4 polynomial on the fraction, exponent via integer add); x <= 0 expected.
// Relative error ~2e-5: more than enough for softmax weights that are quantized to fp8 anyway.
DEVI float poly_exp2(float x) {
    x = fmaxf(x, -126.f);
    float r = rintf(x);
    float f = x - r;  // [-0.5, 0.5]
    float p = fmaf(f, 1.3333558e-3f, 9.6181291e-3f);
    p = fmaf(p, f, 5.5504109e-2f);
    p = fmaf(p, f, 2.4022651e-1f);
    p = fmaf(p, f, 6.9314718e-1f);
    p = fmaf(p, f, 1.0f);
    return __int_as_float(__float_as_int(p) + ((int)r << 23));
}

DEVI uint32_t pack_bf16x2(float lo, float hi) {
    bf162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// 64-byte-row shared memory swizzle: 16B chunk index XORed with (row>>1)&3.
// Rows are 64 bytes; a ldmatrix phase touching 8 consecutive rows at the same
// logical chunk then hits 8 distinct 16B bank groups.
DEVI int swz64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

template <typename T>
DEVI T load_or_zero(const T* p, bool pred) {
    return pred ? *p : T(0);
}
