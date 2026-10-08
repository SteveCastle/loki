import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
p='kernels/sage_attn.cu'; s=open(p).read()
old_start = s.index('        // dequantize + mask + row max')
old_end = s.index('        // O += P V')
new = '''        // dequantize + mask + row max
        float sf[G::NT][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const int tile_end = j0 + BKV;
        const bool full_tile = (tile_end <= lim_a) && (tile_end <= lim_b) && (!seg1 || tile_end <= p.len1);
        if (full_tile) {
#pragma unroll
            for (int j = 0; j < G::NT; ++j) {
                int kl = j * 8 + t4 * 2;
                float sk0 = sks[kl], sk1 = sks[kl + 1];
                sf[j][0] = fmaf((float)s[j][0], sq_a * sk0, ca);
                sf[j][1] = fmaf((float)s[j][1], sq_a * sk1, ca);
                sf[j][2] = fmaf((float)s[j][2], sq_b * sk0, cb);
                sf[j][3] = fmaf((float)s[j][3], sq_b * sk1, cb);
                mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
                mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
            }
        } else {
#pragma unroll
            for (int j = 0; j < G::NT; ++j) {
                int kl = j * 8 + t4 * 2;
                int key = j0 + kl;
                float sk0 = sks[kl], sk1 = sks[kl + 1];
                bool v0 = key < lim_a, v1 = key + 1 < lim_a, v2 = key < lim_b, v3 = key + 1 < lim_b;
                if (seg1) { v0 &= key < p.len1; v1 &= key + 1 < p.len1; v2 &= key < p.len1; v3 &= key + 1 < p.len1; }
                sf[j][0] = v0 ? fmaf((float)s[j][0], sq_a * sk0, ca) : -INFINITY;
                sf[j][1] = v1 ? fmaf((float)s[j][1], sq_a * sk1, ca) : -INFINITY;
                sf[j][2] = v2 ? fmaf((float)s[j][2], sq_b * sk0, cb) : -INFINITY;
                sf[j][3] = v3 ? fmaf((float)s[j][3], sq_b * sk1, cb) : -INFINITY;
                mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
                mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
            }
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        float alpha_a = (m_a == -INFINITY) ? 0.f : fast_exp2(m_a - mn_a);
        float alpha_b = (m_b == -INFINITY) ? 0.f : fast_exp2(m_b - mn_b);
        float rs_a = 0.f, rs_b = 0.f;
        uint32_t pf[G::KS][4];  // P (x448, e4m3) as A fragments for KS k-steps of 32 keys
        const bool any_inf = (mn_a == -INFINITY) || (mn_b == -INFINITY);
#pragma unroll
        for (int ks_ = 0; ks_ < G::KS; ++ks_) {
            float pa[8], pb[8];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                int j = ks_ * 4 + jj;
                pa[jj * 2] = fast_exp2(sf[j][0] - mn_a);
                pa[jj * 2 + 1] = fast_exp2(sf[j][1] - mn_a);
                pb[jj * 2] = fast_exp2(sf[j][2] - mn_b);
                pb[jj * 2 + 1] = fast_exp2(sf[j][3] - mn_b);
            }
            if (any_inf) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (mn_a == -INFINITY) pa[e] = 0.f;
                    if (mn_b == -INFINITY) pb[e] = 0.f;
                }
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) { rs_a += pa[e]; rs_b += pb[e]; }
            // a0: row g, tiles (4ks, 4ks+1); a1: row g+8 same; a2: row g, tiles (4ks+2, 4ks+3); a3: row g+8
            pf[ks_][0] = pack_e4m3x4(pa[0] * 448.f, pa[1] * 448.f, pa[2] * 448.f, pa[3] * 448.f);
            pf[ks_][1] = pack_e4m3x4(pb[0] * 448.f, pb[1] * 448.f, pb[2] * 448.f, pb[3] * 448.f);
            pf[ks_][2] = pack_e4m3x4(pa[4] * 448.f, pa[5] * 448.f, pa[6] * 448.f, pa[7] * 448.f);
            pf[ks_][3] = pack_e4m3x4(pb[4] * 448.f, pb[5] * 448.f, pb[6] * 448.f, pb[7] * 448.f);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        if (seg1) { l1_a = l1_a * alpha_a + rs_a; l1_b = l1_b * alpha_b + rs_b; l2_a *= alpha_a; l2_b *= alpha_b; }
        else { l2_a = l2_a * alpha_a + rs_a; l2_b = l2_b * alpha_b + rs_b; l1_a *= alpha_a; l1_b *= alpha_b; }
        m_a = mn_a; m_b = mn_b;
        // rescale O: softmax alpha and V-scale change (O_acc held in units of sv_prev); skipped when the
        // whole warp's factors are exactly 1 (running max and V scale unchanged).
        float ra = alpha_a * (sv_prev / sv_cur), rb = alpha_b * (sv_prev / sv_cur);
        sv_prev = sv_cur;
        if (__any_sync(0xffffffff, (ra != 1.f) || (rb != 1.f))) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o_acc[j][0] *= ra; o_acc[j][1] *= ra;
                o_acc[j][2] *= rb; o_acc[j][3] *= rb;
            }
        }
'''
s = s[:old_start] + new + s[old_end:]
open(p,'w').write(s)
print("patched")
