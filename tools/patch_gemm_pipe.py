import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
p='kernels/gemm_int8.cu'; s=open(p).read()
old_start = s.index('        const int stage = kt % STAGES;\n        const uint8_t* as = As + stage * A_BYTES;')
old_end = s.index('    cp_async_wait<0>();\n\n    // epilogue')
new = '''        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * A_BYTES;
        const uint8_t* bs = Bs + stage * B_BYTES;
        // software-pipelined fragment loads: fragments for k-step kk+32 are fetched while the
        // mma instructions of k-step kk issue.
        uint32_t afrag[2][4][4];
        uint32_t bfrag[2][NI][2];
        auto load_frags = [&](int kk, int buf) {
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx >> 1);
                ldmatrix_x4(afrag[buf][mi][0], afrag[buf][mi][1], afrag[buf][mi][2], afrag[buf][mi][3], smem_u32(as + swz64(row, chunk)));
            }
#pragma unroll
            for (int nj = 0; nj < NI; nj += 2) {
                int row = wn * WN + (nj + (lidx >> 1)) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx & 1);
                ldmatrix_x4(bfrag[buf][nj][0], bfrag[buf][nj][1], bfrag[buf][nj + 1][0], bfrag[buf][nj + 1][1], smem_u32(bs + swz64(row, chunk)));
            }
        };
        load_frags(0, 0);
        load_frags(32, 1);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) mma_s8_16832(acc[mi][ni], afrag[0][mi], bfrag[0][ni]);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) mma_s8_16832(acc[mi][ni], afrag[1][mi], bfrag[1][ni]);
    }
'''
s = s[:old_start] + new + s[old_end:]
open(p,'w').write(s)
print("patched")
