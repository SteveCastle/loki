// MiniMax H3 audio VAE kernels (DAC encoder + AttnProjection head + BigVGAN decoder), all fp32.
//
//  avae_conv_{128,64,32,16,8}  implicit-GEMM 1-D (transposed) convolution, fp32 SIMT, register-tiled,
//                               double-buffered smem, fused epilogue (bias, residual, accumulate/div,
//                               optional Snake1d second output).
//  avae_act                     BigVGAN Activation1d fused per tile: replicate-pad x2 kaiser-sinc upsample ->
//                               SnakeBeta -> replicate-pad x2 kaiser-sinc downsample (the 2x signal lives in smem).
//  avae_act_post_out            activation_post + conv_post (C->1, k7, no bias) + clamp[-1,1] fused.
//  avae_ln2_cm                  channel-major LayerNorm (two affine outputs from one statistics pass).
//  avae_attn                    causal fp32 attention (head_dim 256), online softmax.
//  avae_head_tail               head-mean + adaptive pool + attn.proj + residual + LN + LN + GeGLU MLP +
//                               mean_proj + latent normalization, written as [32, B, T].
//  avae_denorm                  z * std + mean, [32, B, T] -> [B, 32, T].
#include "common.cuh"

// ------------------------------------------------------------------------------------------------
// Implicit GEMM convolution.
//   out[b, co, t] = sum_{ci, m} W[ph][m*Ci + ci][co] * X[b, ci, q*istride + m*dil - ipad]
//   with output position t = q*ostride + ph - opad (phase ph = blockIdx.z % nph).
// Regular conv: nph = 1, ostride = 1, opad = 0. Transposed conv (stride s, padding p, weights
// re-laid out per phase on the host): nph = s, ostride = s, opad = p, istride = 1, dil = -1, ipad = 0.
// Epilogue: v = acc + bias; v += R; v = Acc + v; v = v / div; Y = v; Y2 = sin(a v)^2 * inv + v.
// R / Acc may alias Y (each element is read and written by the same thread).
// ------------------------------------------------------------------------------------------------
template <int BM, int BN, int TM, int TN, int BK, int STAGES>
__device__ __forceinline__ void conv_igemm(const float* __restrict__ W, const float* __restrict__ bias, const float* __restrict__ X,
                                           long long xbs, int xcs, float* Y, float* Y2, const float* R, const float* Acc,
                                           const float* __restrict__ snake_a, const float* __restrict__ snake_inv, int Ci, int Co, int Lin,
                                           int Lout, int KT, int istride, int dil, int ipad, int nph, int ostride, int opad, float div) {
    constexpr int NT = (BM / TM) * (BN / TN);
    constexpr int A_CHUNKS = BM * BK / 4;  // 16-byte chunks per A stage
    constexpr int A_PER = (A_CHUNKS + NT - 1) / NT;
    constexpr int B_PER = BK * BN / NT;
    __shared__ __align__(16) float As[STAGES][BK][BM];
    __shared__ __align__(16) float Bs[STAGES][BK][BN];

    const int tid = threadIdx.x;
    const int z = blockIdx.z;
    const int ph = z % nph;
    const int b = z / nph;
    const int Kd = Ci * KT;
    int a0 = opad - ph;
    const int q0 = a0 > 0 ? (a0 + ostride - 1) / ostride : 0;
    const int qe = (Lout + opad - ph + ostride - 1) / ostride;
    const int nq = qe - q0;
    const int n0 = blockIdx.x * BN;
    if (n0 >= nq) return;
    const int m0 = blockIdx.y * BM;
    const float* Wp = W + (size_t)ph * Co * Kd;  // [Kd][Co]
    const float* Xb = X + (size_t)b * xbs;

    // K ordering is tap-major: kd = m * Ci + ci (weights re-laid out on the host as [kd][co]).
    // When Ci % BK == 0 every K tile has a single tap m and consecutive channels (no per-element division).
    const bool cimul = (Ci % BK) == 0;
    int colpos[B_PER];  // per-thread column base position (q * istride - ipad), very negative when out of range
#pragma unroll
    for (int r = 0; r < B_PER; ++r) {
        int bn = (tid + r * NT) % BN;
        int nn = n0 + bn;
        colpos[r] = nn < nq ? (q0 + nn) * istride - ipad : -(1 << 30);
    }
    auto issue_tile = [&](int k0, int st) {
#pragma unroll
        for (int r = 0; r < A_PER; ++r) {
            int e = tid + r * NT;
            if (e < A_CHUNKS) {
                int ak = e / (BM / 4), am = (e % (BM / 4)) * 4;
                int k = k0 + ak, m = m0 + am;
                bool ok = k < Kd && m < Co;
                const float* src = ok ? Wp + (size_t)k * Co + m : Wp;
                cp_async_16(smem_u32(&As[st][ak][am]), src, ok);
            }
        }
        int mt = 0, ci0 = 0;
        if (cimul) {
            mt = k0 / Ci;
            ci0 = k0 - mt * Ci;
        }
#pragma unroll
        for (int r = 0; r < B_PER; ++r) {
            int e = tid + r * NT;
            int bk = e / BN, bn = e % BN;
            int ci, mm;
            bool ok;
            if (cimul) {
                ci = ci0 + bk;
                mm = mt;
                ok = true;
            } else {
                int kd = k0 + bk;
                mm = kd / Ci;
                ci = kd - mm * Ci;
                ok = kd < Kd;
            }
            int pos = colpos[r] + mm * dil;
            ok = ok && pos >= 0 && pos < Lin;
            const float* src = ok ? Xb + (size_t)ci * xcs + pos : Xb;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(smem_u32(&Bs[st][bk][bn])), "l"(src), "r"(ok ? 4 : 0));
        }
    };

    // thread grid (BM/TM) x (BN/TN); each warp covers a WR x WC sub-grid (WC = 8 columns) so that
    // the per-k float4 smem reads are broadcast-friendly (8 lanes of a phase share one A address and read 128 B of B).
    constexpr int TCOLS = BN / TN;
    constexpr int WC = TCOLS < 8 ? TCOLS : 8;
    constexpr int WR = 32 / WC;
    constexpr int WXN = TCOLS / WC;
    const int lane = tid & 31, wid = tid >> 5;
    const int ty = (wid / WXN) * WR + lane / WC;
    const int tx = (wid % WXN) * WC + lane % WC;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    auto midx = [&](int i) -> int { return TM == 8 ? (i < 4 ? ty * 4 + i : BM / 2 + ty * 4 + i - 4) : ty * TM + i; };
    auto nidx = [&](int j) -> int { return TN == 8 ? (j < 4 ? tx * 4 + j : BN / 2 + tx * 4 + j - 4) : tx * TN + j; };

    const int nk = (Kd + BK - 1) / BK;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < nk) issue_tile(st * BK, st);
        cp_async_commit();
    }
    for (int kt = 0; kt < nk; ++kt) {
        const int cur = kt % STAGES;
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nt = kt + STAGES - 1;
            if (nt < nk) issue_tile(nt * BK, nt % STAGES);
            cp_async_commit();
        }
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float av[TM], bv[TN];
            if constexpr (TM == 8) {
                float4 x0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
                float4 x1 = *reinterpret_cast<const float4*>(&As[cur][k][BM / 2 + ty * 4]);
                av[0] = x0.x; av[1] = x0.y; av[2] = x0.z; av[3] = x0.w;
                av[4] = x1.x; av[5 ] = x1.y; av[6 ] = x1.z; av[7 ] = x1.w;
            } else if constexpr (TM == 4) {
                float4 x0 = *reinterpret_cast<const float4*>(&As[cur][k][ty * 4]);
                av[0] = x0.x; av[1 ] = x0.y; av[2 ] = x0.z; av[3 ] = x0.w;
            } else {
#pragma unroll
                for (int i = 0; i < TM; ++i) av[i] = As[cur][k][ty * TM + i];
            }
            if constexpr (TN == 8) {
                float4 y0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
                float4 y1 = *reinterpret_cast<const float4*>(&Bs[cur][k][BN / 2 + tx * 4]);
                bv[0] = y0.x; bv[1] = y0.y; bv[2] = y0.z; bv[3] = y0.w;
                bv[4 ] = y1.x; bv[5 ] = y1.y; bv[6 ] = y1.z; bv[7 ] = y1.w;
            } else {
                float4 y0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
                bv[0] = y0.x; bv[1 ] = y0.y; bv[2 ] = y0.z; bv[3 ] = y0.w;
            }
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
    }

    const size_t ybase = (size_t)b * Co * Lout;
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int co = m0 + midx(i);
        if (co >= Co) continue;
        const float bi = bias ? __ldg(bias + co) : 0.f;
        float sa = 0.f, si = 0.f;
        if (Y2) { sa = __ldg(snake_a + co); si = __ldg(snake_inv + co); }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int nn = n0 + nidx(j);
            if (nn >= nq) continue;
            const int t = (q0 + nn) * ostride + ph - opad;
            const size_t idx = ybase + (size_t)co * Lout + t;
            float v = acc[i][j] + bi;
            if (R) v += R[idx];
            if (Acc) v = Acc[idx] + v;
            if (div != 1.f) v = v / div;
            if (Y) Y[idx] = v;
            if (Y2) {
                float s = sinf(sa * v);
                Y2[idx] = s * s * si + v;
            }
        }
    }
}

#define CONV_ARGS                                                                                                                \
    const float *__restrict__ W, const float *__restrict__ bias, const float *__restrict__ X, long long xbs, int xcs, float *Y, \
        float *Y2, const float *R, const float *Acc, const float *__restrict__ snake_a, const float *__restrict__ snake_inv, int Ci,  \
        int Co, int Lin, int Lout, int KT, int istride, int dil, int ipad, int nph, int ostride, int opad, float div
#define CONV_PASS W, bias, X, xbs, xcs, Y, Y2, R, Acc, snake_a, snake_inv, Ci, Co, Lin, Lout, KT, istride, dil, ipad, nph, ostride, opad, div

extern "C" __global__ void __launch_bounds__(256, 1) avae_conv_128(CONV_ARGS) { conv_igemm<128, 128, 8, 8, 16, 3>(CONV_PASS); }
extern "C" __global__ void __launch_bounds__(256, 2) avae_conv_64(CONV_ARGS) { conv_igemm<64, 128, 8, 4, 8, 4>(CONV_PASS); }
extern "C" __global__ void __launch_bounds__(256, 2) avae_conv_64s(CONV_ARGS) { conv_igemm<64, 64, 4, 4, 8, 4>(CONV_PASS); }
extern "C" __global__ void __launch_bounds__(256, 2) avae_conv_32(CONV_ARGS) { conv_igemm<32, 256, 4, 8, 8, 4>(CONV_PASS); }
extern "C" __global__ void __launch_bounds__(256, 2) avae_conv_16(CONV_ARGS) { conv_igemm<16, 256, 2, 8, 8, 4>(CONV_PASS); }
extern "C" __global__ void __launch_bounds__(256, 2) avae_conv_8(CONV_ARGS) { conv_igemm<8, 256, 1, 8, 8, 4>(CONV_PASS); }

// ------------------------------------------------------------------------------------------------
// Activation1d (BigVGAN alias-free SnakeBeta), one row tile [ta, tb) of the output (0 <= ta < tb <= L).
//   up:   xp = replicate_pad(x, 5); u = conv_transpose1d(xp, f, stride 2) * 2, cropped [15, 15]  (len 2L)
//         u[j] = 2 * sum_{k = (J&1)+2m} f[k] * x[clamp((J-k)/2 - 5)],  J = j + 15
//   act:  s = u + inv_beta * sin(alpha * u)^2
//   down: out[t] = sum_k f[k] * s[clamp(2t + k - 5, 0, 2L-1)]
// xs needs (tb-ta)+16 floats, ss needs 2(tb-ta)+11 floats. Must be called by all threads of the block.
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ void act_tile(const float* __restrict__ x, int L, int ta, int tb, float* xs, float* ss, const float* f, float alpha,
                                         float inv, float* dst) {
    const int nx = tb - ta + 16;
    for (int i = threadIdx.x; i < nx; i += blockDim.x) {
        int idx = min(max(ta - 8 + i, 0), L - 1);
        xs[i] = __ldg(x + idx);
    }
    __syncthreads();
    const int ns = 2 * (tb - ta) + 11;
    const int j0 = 2 * ta - 5;
    for (int i = threadIdx.x; i < ns; i += blockDim.x) {
        int jj = min(max(j0 + i, 0), 2 * L - 1);
        int J = jj + 15;
        int par = J & 1;
        float u = 0.f;
#pragma unroll
        for (int m = 0; m < 6; ++m) {
            int k = par + 2 * m;
            int xi = ((J - k) >> 1) - 5;  // index into x (before clamp, already clamped in xs)
            u = fmaf(xs[xi - ta + 8], f[k], u);
        }
        u = u * 2.f;
        float sn = sinf(alpha * u);
        ss[i] = sn * sn * inv + u;
    }
    __syncthreads();
    for (int t = ta + threadIdx.x; t < tb; t += blockDim.x) {
        const float* sp = ss + 2 * (t - ta);
        float o = 0.f;
#pragma unroll
        for (int k = 0; k < 12; ++k) o = fmaf(f[k], sp[k], o);
        dst[t - ta] = o;
    }
}

#define ACT_TT 256
// X, Y: [B][C][L] (batch strides xbs/ybs). filt: 12 taps. alpha_e/inv_beta: [C] (already exp'd / reciprocal'd).
extern "C" __global__ void __launch_bounds__(256) avae_act(const float* __restrict__ X, float* __restrict__ Y, const float* __restrict__ filt,
                                                          const float* __restrict__ alpha_e, const float* __restrict__ inv_beta, int C, int L,
                                                          long long xbs, long long ybs) {
    __shared__ float xs[ACT_TT + 16];
    __shared__ float ss[2 * ACT_TT + 11];
    __shared__ float f[12];
    const int c = blockIdx.y, b = blockIdx.z;
    const int ta = blockIdx.x * ACT_TT;
    const int tb = min(ta + ACT_TT, L);
    if (threadIdx.x < 12) f[threadIdx.x] = filt[threadIdx.x];
    const float* x = X + (size_t)b * xbs + (size_t)c * L;
    float* y = Y + (size_t)b * ybs + (size_t)c * L;
    act_tile(x, L, ta, tb, xs, ss, f, __ldg(alpha_e + c), __ldg(inv_beta + c), y + ta);
}

// activation_post + conv_post (C -> 1, kernel 7, padding 3, no bias) + clamp. X: [B][C][L], out: [B][L]. C <= 16.
#define POST_TT 256
extern "C" __global__ void __launch_bounds__(256) avae_act_post_out(const float* __restrict__ X, float* __restrict__ out, const float* __restrict__ filt,
                                                                   const float* __restrict__ alpha_e, const float* __restrict__ inv_beta,
                                                                   const float* __restrict__ wpost, int C, int L) {
    __shared__ float xs[POST_TT + 6 + 16];
    __shared__ float ss[2 * (POST_TT + 6) + 11];
    __shared__ float as[16][POST_TT + 6];
    __shared__ float f[12];
    __shared__ float w[16 * 7];
    const int b = blockIdx.y;
    const int t0 = blockIdx.x * POST_TT;
    const int t1 = min(t0 + POST_TT, L);
    if (threadIdx.x < 12) f[threadIdx.x] = filt[threadIdx.x];
    for (int i = threadIdx.x; i < C * 7; i += blockDim.x) w[i] = wpost[i];
    const int ta = max(t0 - 3, 0), tb = min(t1 + 3, L);
    for (int c = 0; c < C; ++c) {
        // zero padding of conv_post outside [0, L)
        for (int i = threadIdx.x; i < POST_TT + 6; i += blockDim.x) {
            int t = t0 - 3 + i;
            if (t < 0 || t >= L) as[c][i] = 0.f;
        }
        __syncthreads();
        act_tile(X + (size_t)b * C * L + (size_t)c * L, L, ta, tb, xs, ss, f, __ldg(alpha_e + c), __ldg(inv_beta + c), &as[c][ta - (t0 - 3)]);
        __syncthreads();
    }
    for (int t = t0 + threadIdx.x; t < t1; t += blockDim.x) {
        float o = 0.f;
        const int li = t - (t0 - 3);
        for (int c = 0; c < C; ++c) {
#pragma unroll
            for (int k = 0; k < 7; ++k) o = fmaf(w[c * 7 + k], as[c][li - 3 + k], o);
        }
        out[(size_t)b * L + t] = fminf(fmaxf(o, -1.f), 1.f);
    }
}

// ------------------------------------------------------------------------------------------------
// Channel-major LayerNorm: X [B][C][T] -> Y1 = LN(X; w1, b1), Y3 = LN(X; w3, b3) (same layout). eps given.
// grid (ceil(T/32), B), block (32, 8).
// ------------------------------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) avae_ln2_cm(const float* __restrict__ X, float* __restrict__ Y1, float* __restrict__ Y3,
                                                             const float* __restrict__ w1, const float* __restrict__ b1, const float* __restrict__ w3,
                                                             const float* __restrict__ b3, int C, int T, float eps) {
    __shared__ float red[8][33];
    __shared__ float stat[2][32];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int t = blockIdx.x * 32 + tx;
    const int b = blockIdx.y;
    const float* x = X + (size_t)b * C * T;
    const bool ok = t < T;
    float s = 0.f;
    if (ok)
        for (int c = ty; c < C; c += 8) s += x[(size_t)c * T + t];
    red[ty][tx] = s;
    __syncthreads();
    if (ty == 0) {
        float a = 0.f;
        for (int i = 0; i < 8; ++i) a += red[i][tx];
        stat[0][tx] = a / (float)C;
    }
    __syncthreads();
    const float mean = stat[0][tx];
    float v = 0.f;
    if (ok)
        for (int c = ty; c < C; c += 8) {
            float d = x[(size_t)c * T + t] - mean;
            v = fmaf(d, d, v);
        }
    red[ty][tx] = v;
    __syncthreads();
    if (ty == 0) {
        float a = 0.f;
        for (int i = 0; i < 8; ++i) a += red[i][tx];
        stat[1][tx] = 1.f / sqrtf(a / (float)C + eps);
    }
    __syncthreads();
    const float rstd = stat[1][tx];
    if (!ok) return;
    for (int c = ty; c < C; c += 8) {
        size_t i = (size_t)b * C * T + (size_t)c * T + t;
        float n = (x[(size_t)c * T + t] - mean) * rstd;
        Y1[i] = n * w1[c] + b1[c];
        Y3[i] = n * w3[c] + b3[c];
    }
}

// ------------------------------------------------------------------------------------------------
// Causal attention, fp32. QKV: [B][3*H*D][T] channel-major (q | k | v, head h at h*D). D = 256.
// O: [B][H][T][D]. grid (ceil(T/16), H, B), block 256 (8 warps x 2 queries), dyn smem (16*D + 2*32*(D+1)) floats.
// ------------------------------------------------------------------------------------------------
#define ATT_D 256
#define ATT_QT 16
#define ATT_KC 32
extern "C" __global__ void __launch_bounds__(256) avae_attn(const float* __restrict__ QKV, float* __restrict__ O, int H, int T, float scale) {
    extern __shared__ float sm[];
    float* Qs = sm;                       // [16][256]
    float* Ks = Qs + ATT_QT * ATT_D;      // [32][257]
    float* Vs = Ks + ATT_KC * (ATT_D + 1);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int qt0 = blockIdx.x * ATT_QT, h = blockIdx.y, b = blockIdx.z;
    const int HD = H * ATT_D;
    const float* base = QKV + (size_t)b * 3 * HD * T;
    for (int e = tid; e < ATT_QT * ATT_D; e += 256) {
        int i = e % ATT_QT, d = e / ATT_QT;
        int t = qt0 + i;
        Qs[i * ATT_D + d] = t < T ? base[(size_t)(h * ATT_D + d) * T + t] : 0.f;
    }
    float m[2], l[2], o[2][8];
#pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        m[qq] = -INFINITY;
        l[qq] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) o[qq][e] = 0.f;
    }
    const int kend = min(T, qt0 + ATT_QT);
    for (int kc = 0; kc < kend; kc += ATT_KC) {
        __syncthreads();
        for (int e = tid; e < ATT_KC * ATT_D; e += 256) {
            int j = e % ATT_KC, d = e / ATT_KC;
            int kt = kc + j;
            float kv = 0.f, vv = 0.f;
            if (kt < T) {
                kv = base[(size_t)(HD + h * ATT_D + d) * T + kt];
                vv = base[(size_t)(2 * HD + h * ATT_D + d) * T + kt];
            }
            Ks[j * (ATT_D + 1) + d] = kv;
            Vs[j * (ATT_D + 1) + d] = vv;
        }
        __syncthreads();
#pragma unroll
        for (int qq = 0; qq < 2; ++qq) {
            const int qi = warp * 2 + qq;
            const int t = qt0 + qi;
            if (t >= T || kc > t) continue;  // warp-uniform
            const float* q = Qs + qi * ATT_D;
            const float* k = Ks + lane * (ATT_D + 1);
            float s = 0.f;
#pragma unroll 8
            for (int d = 0; d < ATT_D; ++d) s = fmaf(q[d], k[d], s);
            s *= scale;
            if (kc + lane > t) s = -INFINITY;
            const float mx = warp_max(s);
            const float mn = fmaxf(m[qq], mx);
            const float corr = expf(m[qq] - mn);
            const float p = expf(s - mn);
            l[qq] = l[qq] * corr + warp_sum(p);
#pragma unroll
            for (int e = 0; e < 8; ++e) o[qq][e] *= corr;
            for (int jj = 0; jj < ATT_KC; ++jj) {
                const float pj = __shfl_sync(0xffffffff, p, jj);
                const float* v = Vs + jj * (ATT_D + 1) + lane;
#pragma unroll
                for (int e = 0; e < 8; ++e) o[qq][e] = fmaf(pj, v[32 * e], o[qq][e]);
            }
            m[qq] = mn;
        }
    }
#pragma unroll
    for (int qq = 0; qq < 2; ++qq) {
        const int t = qt0 + warp * 2 + qq;
        if (t >= T) continue;
        float* op = O + (((size_t)b * H + h) * T + t) * ATT_D;
        const float il = 1.f / l[qq];
#pragma unroll
        for (int e = 0; e < 8; ++e) op[lane + 32 * e] = o[qq][e] * il;
    }
}

// ------------------------------------------------------------------------------------------------
// Posterior head tail, one block (256 threads) per (t, b):
//   pooled[c] = avg_8( mean_h O[b,h,t,:] )[c]          (torch.mean over heads, adaptive_avg_pool1d 256 -> 32)
//   x1 = P[b,:,t] + (Wa pooled + ba)                    (proj(norm3(x)) + attn)
//   x2 = x1 + W2( gelu_tanh(W0 n + b0) * (W1 n + b1) ) + b2,  n = LNm(LN2(x1))
//   z  = (Wm x2 + bm - latents_mean) / latents_std      -> Z[c][b][t]  ([32, B, T])
// ------------------------------------------------------------------------------------------------
__device__ __forceinline__ float ln32(float x, float w, float b, float eps) {
    float mean = warp_sum(x) / 32.f;
    float d = x - mean;
    float var = warp_sum(d * d) / 32.f;
    return d / sqrtf(var + eps) * w + b;
}
extern "C" __global__ void __launch_bounds__(256) avae_head_tail(const float* __restrict__ O, const float* __restrict__ P, float* __restrict__ Z,
                                                                const float* __restrict__ wa, const float* __restrict__ ba,
                                                                const float* __restrict__ n2w, const float* __restrict__ n2b,
                                                                const float* __restrict__ nmw, const float* __restrict__ nmb,
                                                                const float* __restrict__ w0, const float* __restrict__ b0,
                                                                const float* __restrict__ w1, const float* __restrict__ b1,
                                                                const float* __restrict__ w2, const float* __restrict__ b2,
                                                                const float* __restrict__ wm, const float* __restrict__ bm,
                                                                const float* __restrict__ lmean, const float* __restrict__ lstd, int H, int T,
                                                                int B, float eps) {
    __shared__ float hm[ATT_D];
    __shared__ float v32[32];
    __shared__ float g64[64];
    const int t = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    {
        float s = 0.f;
        for (int h = 0; h < H; ++h) s += O[(((size_t)b * H + h) * T + t) * ATT_D + tid];
        hm[tid] = s / (float)H;
    }
    __syncthreads();
    if (tid >= 32) return;
    const int c = tid;
    float pooled = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) pooled += hm[c * 8 + i];
    pooled = pooled / 8.f;
    v32[c] = pooled;
    __syncwarp();
    float att = 0.f;
    for (int k = 0; k < 32; ++k) att = fmaf(wa[c * 32 + k], v32[k], att);
    att += ba[c];
    const float x1 = P[((size_t)b * 32 + c) * T + t] + att;
    const float n2 = ln32(x1, n2w[c], n2b[c], eps);
    const float nm = ln32(n2, nmw[c], nmb[c], eps);
    __syncwarp();
    v32[c] = nm;
    __syncwarp();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int j = c + 32 * r;
        float h0 = 0.f, h1 = 0.f;
        for (int k = 0; k < 32; ++k) {
            h0 = fmaf(w0[j * 32 + k], v32[k], h0);
            h1 = fmaf(w1[j * 32 + k], v32[k], h1);
        }
        h0 += b0[j];
        h1 += b1[j];
        g64[j] = gelu_tanh_f(h0) * h1;
    }
    __syncwarp();
    float y = 0.f;
    for (int j = 0; j < 64; ++j) y = fmaf(w2[c * 64 + j], g64[j], y);
    y += b2[c];
    const float x2 = x1 + y;
    __syncwarp();
    v32[c] = x2;
    __syncwarp();
    float zz = 0.f;
    for (int k = 0; k < 32; ++k) zz = fmaf(wm[c * 32 + k], v32[k], zz);
    zz += bm[c];
    Z[((size_t)c * B + b) * T + t] = (zz - lmean[c]) / lstd[c];
}

// z [32][B][T] (normalized) -> out [B][32][T] = z * std + mean
extern "C" __global__ void avae_denorm(const float* __restrict__ z, float* __restrict__ out, const float* __restrict__ lmean,
                                       const float* __restrict__ lstd, int B, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = 32 * B * T;
    if (i >= n) return;
    int t = i % T;
    int bb = (i / T) % B;
    int c = i / (T * B);
    out[((size_t)bb * 32 + c) * T + t] = z[i] * lstd[c] + lmean[c];
}
