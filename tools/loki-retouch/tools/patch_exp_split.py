import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
p='kernels/common.cuh'; s=open(p).read()
s=s.replace('''DEVI uint32_t pack_bf16x2(float lo, float hi) {''','''// exp2 on the FMA pipe (degree-4 polynomial on the fraction, exponent via integer add); x <= 0 expected.
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

DEVI uint32_t pack_bf16x2(float lo, float hi) {''')
open(p,'w').write(s)

p='kernels/sage_attn.cu'; s=open(p).read()
s=s.replace('''                pa[jj * 2] = fast_exp2(sf[j][0] - mn_a);
                pa[jj * 2 + 1] = fast_exp2(sf[j][1] - mn_a);
                pb[jj * 2] = fast_exp2(sf[j][2] - mn_b);
                pb[jj * 2 + 1] = fast_exp2(sf[j][3] - mn_b);''','''                // half of the exponentials on MUFU, half on the FMA pipe
                pa[jj * 2] = fast_exp2(sf[j][0] - mn_a);
                pa[jj * 2 + 1] = poly_exp2(sf[j][1] - mn_a);
                pb[jj * 2] = fast_exp2(sf[j][2] - mn_b);
                pb[jj * 2 + 1] = poly_exp2(sf[j][3] - mn_b);''')
open(p,'w').write(s)

# packed SwiGLU epilogue stores
p='kernels/gemm_int8.cu'; s=open(p).read()
s=s.replace('''                if (mode == 3) {
                    C[(int64_t)m * (N >> 1) + (n >> 1)] = (OutT)(silu_f(v0) * v1);
                    continue;
                }''','''                if (mode == 3) {
                    // gather the quad's 4 outputs into lane t4 == 0 and store 8 bytes at once (bf16 out only)
                    float o = silu_f(v0) * v1;
                    float o1 = __shfl_xor_sync(0xffffffff, o, 1);
                    uint32_t pk = pack_bf16x2(o, o1);
                    uint32_t pk2 = __shfl_xor_sync(0xffffffff, pk, 2);
                    if (t4 == 0) {
                        *reinterpret_cast<uint2*>(reinterpret_cast<bf16*>(C) + (int64_t)m * (N >> 1) + (n >> 1)) = make_uint2(pk, pk2);
                    }
                    continue;
                }''')
open(p,'w').write(s)
print("patched")
