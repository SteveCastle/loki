// Qwen3.5 / Qwen3.8 hybrid decoder kernels (GGUF arch "qwen35"):
//   * RMSNorm / LayerNorm with f32 weights (GGUF keeps norms in f32; the stored RMSNorm weights already include the +1)
//   * Gated DeltaNet linear attention: causal depthwise conv (kernel 4) + SiLU with a rolling conv state, q/k L2 norm,
//     gates g = softplus(alpha + dt_bias) * A (A stored as -exp(A_log)), beta = sigmoid(b), the per-head delta-rule
//     recurrence (one thread per value column, the column of the 128x128 state in registers), gated RMSNorm
//   * gated full attention: per-head q/k RMSNorm + interleaved M-RoPE on the first 64 of 256 dims (NEOX pairs),
//     bf16 KV cache, causal prefill attention and split-K decode attention (GQA 24/4, head dim 256)
//
// Row layouts follow the GGUF tensors (V heads are already reordered by the converter so that v-head h reads q/k
// head h % n_k_heads).
#include "common.cuh"

namespace {
DEVI float sigmoid_f(float x) { return 1.f / (1.f + __expf(-x)); }
DEVI float softplus_f(float x) { return x > 20.f ? x : log1pf(expf(x)); }
}  // namespace

extern "C" {

// ------------------------------------------------------------------------------------------------ norms

// y = x * rsqrt(mean(x^2) + eps) * w    (x f32 [M, N], w f32 [N]); y f32 and/or bf16 (either may be null)
__global__ void k_rmsnorm_w32(const float* __restrict__ x, float* __restrict__ yf, bf16* __restrict__ yb, int M, int N,
                              float eps, const float* __restrict__ w) {
    __shared__ float red[32];
    const int64_t m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + m * N;
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss, red) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = xr[i] * r * (w ? w[i] : 1.f);
        if (yf) yf[m * N + i] = v;
        if (yb) yb[m * N + i] = f2bf(v);
    }
}

// LayerNorm with f32 affine: x f32 [M, N] -> y f32 [M, N]
__global__ void k_layernorm_w32(const float* __restrict__ x, float* __restrict__ y, int M, int N, float eps,
                                const float* __restrict__ w, const float* __restrict__ b) {
    __shared__ float red[32];
    const int64_t m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + m * N;
    float s = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s, red) / N;
    float ss = 0.f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float v = xr[i] - mean;
        ss += v * v;
    }
    const float rstd = rsqrtf(block_sum(ss, red) / N + eps);
    for (int i = threadIdx.x; i < N; i += blockDim.x) y[m * N + i] = (xr[i] - mean) * rstd * w[i] + b[i];
}

// ------------------------------------------------------------------------------------------------ gated DeltaNet

// Causal depthwise conv over the qkv channels, then SiLU. x f32 [S*T, C] (row stride ldx), w f32 [C, 4],
// states[s] -> f32 [3][C] (the last 3 inputs of sequence s, oldest first); out f32 [S*T, C].
// out[t, c] = silu(w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t]); the state is updated to the last 3 inputs.
__global__ void k_conv_silu(const float* __restrict__ x, int ldx, const float* __restrict__ w, float* const* __restrict__ states,
                            float* __restrict__ out, int S, int T, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)S * C) return;
    const int s = i / C, c = i % C;
    float* st = states[s];
    const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    float a = st[c], b = st[C + c], d = st[2 * C + c];
    for (int t = 0; t < T; ++t) {
        const int64_t r = (int64_t)s * T + t;
        const float v = x[r * ldx + c];
        const float o = w0 * a + w1 * b + w2 * d + w3 * v;
        out[r * C + c] = silu_f(o);
        a = b; b = d; d = v;
    }
    st[c] = a; st[C + c] = b; st[2 * C + c] = d;
}

// In-place L2 normalization of `heads` consecutive 128-wide head vectors per row: x / sqrt(sum(x^2) + eps).
// One warp per (row, head).
__global__ void k_l2norm_heads(float* __restrict__ x, int ld, int R, int heads, float eps) {
    const int64_t wid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t)R * heads) return;
    const int64_t r = wid / heads;
    const int h = wid % heads;
    float4* p = reinterpret_cast<float4*>(x + r * ld + h * 128) + lane;
    float4 v = *p;
    const float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float inv = rsqrtf(ss + eps);
    v.x *= inv; v.y *= inv; v.z *= inv; v.w *= inv;
    *p = v;
}

// Gates per (row, v-head): g = softplus(alpha + dt_bias) * A, beta = sigmoid(beta_raw). In place.
__global__ void k_gdn_gates(float* __restrict__ alpha, float* __restrict__ beta, const float* __restrict__ dt_bias,
                            const float* __restrict__ A, int R, int H) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)R * H) return;
    const int h = i % H;
    alpha[i] = softplus_f(alpha[i] + dt_bias[h]) * A[h];
    beta[i] = sigmoid_f(beta[i]);
}

// Gated delta rule for one v-head of one sequence over T tokens. Block (h, s), 128 threads: thread j owns
// column j of the state S[128 x 128] (S[i][j] at state[(i*128) + j]).
//   kv_j = sum_i S[i][j] k_i ; delta_j = (v_j - g*kv_j) * beta ; S[i][j] = g*S[i][j] + k_i*delta_j ;
//   o_j = scale * sum_i S[i][j] q_i       (g = exp(gate))
// qkv: conv output f32 [S*T, ldq] (q at 0, k at nk*128, v at 2*nk*128), gate/beta f32 [S*T, H],
// states[s] -> f32 [H][128][128], out f32 [S*T, H*128].
__global__ void __launch_bounds__(128) k_gdn(const float* __restrict__ qkv, int ldq, const float* __restrict__ gate,
                                             const float* __restrict__ beta, float* const* __restrict__ states,
                                             float* __restrict__ out, int T, int H, int nk, float scale) {
    const int h = blockIdx.x, s = blockIdx.y, j = threadIdx.x;
    const int kh = h % nk;
    __shared__ float4 ks[32], qs[32];
    float* st = states[s] + (int64_t)h * 128 * 128;
    float S[128];
#pragma unroll
    for (int i = 0; i < 128; ++i) S[i] = st[i * 128 + j];
    for (int t = 0; t < T; ++t) {
        const int64_t r = (int64_t)s * T + t;
        const float* row = qkv + r * ldq;
        __syncthreads();
        reinterpret_cast<float*>(qs)[j] = row[kh * 128 + j];
        reinterpret_cast<float*>(ks)[j] = row[nk * 128 + kh * 128 + j];
        __syncthreads();
        const float v = row[2 * nk * 128 + h * 128 + j];
        const float g = __expf(gate[r * H + h]);
        const float b = beta[r * H + h];
        float kv = 0.f;
#pragma unroll
        for (int i4 = 0; i4 < 32; ++i4) {
            const float4 k4 = ks[i4];
            kv = fmaf(S[i4 * 4 + 0], k4.x, kv);
            kv = fmaf(S[i4 * 4 + 1], k4.y, kv);
            kv = fmaf(S[i4 * 4 + 2], k4.z, kv);
            kv = fmaf(S[i4 * 4 + 3], k4.w, kv);
        }
        const float delta = (v - g * kv) * b;
        float o = 0.f;
#pragma unroll
        for (int i4 = 0; i4 < 32; ++i4) {
            const float4 k4 = ks[i4];
            const float4 q4 = qs[i4];
            S[i4 * 4 + 0] = fmaf(g, S[i4 * 4 + 0], k4.x * delta);
            S[i4 * 4 + 1] = fmaf(g, S[i4 * 4 + 1], k4.y * delta);
            S[i4 * 4 + 2] = fmaf(g, S[i4 * 4 + 2], k4.z * delta);
            S[i4 * 4 + 3] = fmaf(g, S[i4 * 4 + 3], k4.w * delta);
            o = fmaf(S[i4 * 4 + 0], q4.x, o);
            o = fmaf(S[i4 * 4 + 1], q4.y, o);
            o = fmaf(S[i4 * 4 + 2], q4.z, o);
            o = fmaf(S[i4 * 4 + 3], q4.w, o);
        }
        out[r * H * 128 + h * 128 + j] = o * scale;
    }
#pragma unroll
    for (int i = 0; i < 128; ++i) st[i * 128 + j] = S[i];
}

// y = rmsnorm(o) * w * silu(z) per 128-wide head. o, z f32 [R, H*128] (z row stride ldz); y f32/bf16 [R, H*128].
// One warp per (row, head).
__global__ void k_gated_rmsnorm(const float* __restrict__ o, const float* __restrict__ z, int ldz, const float* __restrict__ w,
                                float* __restrict__ yf, bf16* __restrict__ yb, int R, int H, float eps) {
    const int64_t wid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t)R * H) return;
    const int64_t r = wid / H;
    const int h = wid % H;
    const float4 v = reinterpret_cast<const float4*>(o + r * H * 128 + h * 128)[lane];
    const float4 g = reinterpret_cast<const float4*>(z + r * ldz + h * 128)[lane];
    const float4 ww = reinterpret_cast<const float4*>(w)[lane];
    const float ss = warp_sum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
    const float rr = rsqrtf(ss / 128.f + eps);
    float y[4] = {v.x * rr * ww.x * silu_f(g.x), v.y * rr * ww.y * silu_f(g.y), v.z * rr * ww.z * silu_f(g.z), v.w * rr * ww.w * silu_f(g.w)};
    const int64_t off = r * H * 128 + h * 128 + lane * 4;
    if (yf) *reinterpret_cast<float4*>(yf + off) = make_float4(y[0], y[1], y[2], y[3]);
    if (yb) {
        yb[off + 0] = f2bf(y[0]); yb[off + 1] = f2bf(y[1]); yb[off + 2] = f2bf(y[2]); yb[off + 3] = f2bf(y[3]);
    }
}

// ------------------------------------------------------------------------------------------------ full attention

// Per-head RMSNorm (256) + M-RoPE on dims [0, 64) as NEOX pairs (i, i+32), for q heads (from the joint q|gate
// projection: head h's query at h*512) and k heads; writes q f32 [R, HQ, 256] and k, v into the bf16 cache of
// the row's sequence (kcs/vcs[row_seq[r]] -> [pos][HK][256]) at position row_pos[r]. One warp per (row, head) over HQ + HK heads; lane owns dims lane*8 .. +8.
// rope f32 [R, 32, 2] (cos, sin).
__global__ void k_qk_norm_rope(const float* __restrict__ qg, const float* __restrict__ kin, const float* __restrict__ vin,
                               const float* __restrict__ wq, const float* __restrict__ wk, const float* __restrict__ rope,
                               float* __restrict__ qout, bf16* const* __restrict__ kcs, bf16* const* __restrict__ vcs,
                               const int* __restrict__ row_seq, const int* __restrict__ row_pos, int R, int HQ, int HK,
                               float eps) {
    const int64_t wid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wid >= (int64_t)R * (HQ + HK)) return;
    const int64_t r = wid / (HQ + HK);
    const int hh = wid % (HQ + HK);
    const bool isq = hh < HQ;
    const int h = isq ? hh : hh - HQ;
    const float* src = isq ? qg + r * HQ * 512 + h * 512 : kin + r * HK * 256 + h * 256;
    const float* w = isq ? wq : wk;
    float v[8];
    const float4 a = reinterpret_cast<const float4*>(src)[lane * 2];
    const float4 b = reinterpret_cast<const float4*>(src)[lane * 2 + 1];
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) ss += v[e] * v[e];
    const float rr = rsqrtf(warp_sum(ss) / 256.f + eps);
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] *= rr * w[lane * 8 + e];
    // rope: lanes 0..3 hold dims 0..31 (first of each pair), lanes 4..7 hold 32..63
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const float p = __shfl_xor_sync(0xffffffff, v[e], 4);
        if (lane < 8) {
            const int i = (lane & 3) * 8 + e;
            const float c = rope[(r * 32 + i) * 2], s = rope[(r * 32 + i) * 2 + 1];
            v[e] = lane < 4 ? v[e] * c - p * s : v[e] * c + p * s;
        }
    }
    if (isq) {
        float* o = qout + (r * HQ + h) * 256 + lane * 8;
        *reinterpret_cast<float4*>(o) = make_float4(v[0], v[1], v[2], v[3]);
        *reinterpret_cast<float4*>(o + 4) = make_float4(v[4], v[5], v[6], v[7]);
    } else {
        const int64_t base = ((int64_t)row_pos[r] * HK + h) * 256 + lane * 8;
        bf16* kc = kcs[row_seq[r]];
        bf16* vc = vcs[row_seq[r]];
        const float* vs = vin + r * HK * 256 + h * 256 + lane * 8;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            kc[base + e] = f2bf(v[e]);
            vc[base + e] = f2bf(vs[e]);
        }
    }
}

// out[r, h*256 + d] *= sigmoid(gate) with gate = qg[r, h*512 + 256 + d]
__global__ void k_attn_gate(float* __restrict__ out, const float* __restrict__ qg, int R, int HQ) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)R * HQ * 256) return;
    const int64_t r = i / (HQ * 256);
    const int rem = i % (HQ * 256);
    const int h = rem / 256, d = rem % 256;
    out[i] *= sigmoid_f(qg[r * HQ * 512 + h * 512 + 256 + d]);
}

// Causal prefill attention for one sequence: T queries at cache positions p0 .. p0+T-1 against keys 0 .. p0+t.
// q f32 [T, HQ, 256]; K/V bf16 cache rows [pos][HK][256] (this sequence's slot); out f32 [T, HQ, 256].
// Block: 32 queries x 1 head, 256 threads (8 per query, 32 dims each); 32-key f32 tiles of K and V in smem.
__global__ void __launch_bounds__(256) k_attn_prefill(const float* __restrict__ q, const bf16* __restrict__ kc,
                                                      const bf16* __restrict__ vc, float* __restrict__ out, int T, int p0,
                                                      int HQ, int HK, float scale_log2) {
    constexpr int D = 256, DP = 32, TK = 32;
    extern __shared__ float smf[];
    float* Ks = smf;
    float* Vs = smf + TK * D;
    const int tid = threadIdx.x;
    const int qi = blockIdx.x * 32 + (tid >> 3);
    const int part = tid & 7;
    const int h = blockIdx.y;
    const int kh = h / (HQ / HK);
    const bool qok = qi < T;
    float qv[DP], o[DP];
#pragma unroll
    for (int d = 0; d < DP; ++d) {
        qv[d] = qok ? q[((int64_t)qi * HQ + h) * D + part * DP + d] * scale_log2 : 0.f;
        o[d] = 0.f;
    }
    float mrun = -INFINITY, lrun = 0.f;
    const int qlast = min(T - 1, blockIdx.x * 32 + 31);
    const int kend = p0 + qlast + 1;  // keys needed by the block
    const int mylim = p0 + qi;        // last key this query may see
    for (int k0 = 0; k0 < kend; k0 += TK) {
        __syncthreads();
        for (int i = tid; i < TK * D / 2; i += blockDim.x) {
            const int j = (i * 2) / D, d = (i * 2) % D;
            const int key = k0 + j;
            float2 kk = make_float2(0.f, 0.f), vv = make_float2(0.f, 0.f);
            if (key < kend) {
                const int64_t off = ((int64_t)key * HK + kh) * D + d;
                const bf162 k2 = *reinterpret_cast<const bf162*>(kc + off);
                const bf162 v2 = *reinterpret_cast<const bf162*>(vc + off);
                kk = make_float2(__low2float(k2), __high2float(k2));
                vv = make_float2(__low2float(v2), __high2float(v2));
            }
            reinterpret_cast<float2*>(Ks)[i] = kk;
            reinterpret_cast<float2*>(Vs)[i] = vv;
        }
        __syncthreads();
        const int kn = min(TK, kend - k0);
        for (int j0 = 0; j0 < kn; j0 += 8) {
            float s[8];
            float cmax = -INFINITY;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* kr = Ks + (j0 + jj) * D + part * DP;
                float acc = 0.f;
#pragma unroll
                for (int d = 0; d < DP; ++d) acc = fmaf(qv[d], kr[d], acc);
                acc += __shfl_xor_sync(0xffffffff, acc, 1);
                acc += __shfl_xor_sync(0xffffffff, acc, 2);
                acc += __shfl_xor_sync(0xffffffff, acc, 4);
                const int key = k0 + j0 + jj;
                s[jj] = (j0 + jj < kn && key <= mylim) ? acc : -INFINITY;
                cmax = fmaxf(cmax, s[jj]);
            }
            const float mnew = fmaxf(mrun, cmax);
            if (mnew == -INFINITY) continue;
            const float corr = exp2f(mrun - mnew);
            lrun *= corr;
#pragma unroll
            for (int d = 0; d < DP; ++d) o[d] *= corr;
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float p = exp2f(s[jj] - mnew);
                lrun += p;
                const float* vr = Vs + (j0 + jj) * D + part * DP;
#pragma unroll
                for (int d = 0; d < DP; ++d) o[d] = fmaf(p, vr[d], o[d]);
            }
            mrun = mnew;
        }
    }
    if (qok) {
        const float inv = 1.f / lrun;
        float* op = out + ((int64_t)qi * HQ + h) * D + part * DP;
#pragma unroll
        for (int d = 0; d < DP; ++d) op[d] = o[d] * inv;
    }
}

// Split-K decode attention, one query per sequence. Block (split, kv head, seq), 128 threads = 4 warps; the block
// covers keys [split*256, split*256+256) of the sequence (len[s] keys in kcs/vcs[s]); each warp takes every
// 4th key and all G = HQ/HK query heads of the kv head; lane owns dims lane*8 .. +8.
// Partials: pm, pl [S][nsplit][HQ], po [S][nsplit][HQ][256] (unnormalized).
#define DEC_G 6
__global__ void __launch_bounds__(128) k_attn_decode(const float* __restrict__ q, const bf16* const* __restrict__ kcs,
                                                     const bf16* const* __restrict__ vcs,
                                                     const int* __restrict__ len, float* __restrict__ pm,
                                                     float* __restrict__ pl, float* __restrict__ po, int nsplit, int HQ,
                                                     int HK, float scale_log2) {
    constexpr int D = 256, SPLIT = 256;
    const int sp = blockIdx.x, kh = blockIdx.y, s = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int G = HQ / HK;
    const int L = len[s];
    const int k0 = sp * SPLIT, k1 = min(L, k0 + SPLIT);
    __shared__ float sm_m[4][DEC_G], sm_l[4][DEC_G];
    __shared__ float sm_o[4][DEC_G][D];
    float qv[DEC_G][8], o[DEC_G][8], m[DEC_G], l[DEC_G];
#pragma unroll
    for (int g = 0; g < DEC_G; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            qv[g][e] = g < G ? q[((int64_t)s * HQ + kh * G + g) * D + lane * 8 + e] * scale_log2 : 0.f;
            o[g][e] = 0.f;
        }
    }
    const bf16* kbase = kcs[s];
    const bf16* vbase = vcs[s];
    for (int key = k0 + warp; key < k1; key += 4) {
        const int64_t off = ((int64_t)key * HK + kh) * D + lane * 8;
        const uint4 kr = *reinterpret_cast<const uint4*>(kbase + off);
        const uint4 vr = *reinterpret_cast<const uint4*>(vbase + off);
        const bf162* k2 = reinterpret_cast<const bf162*>(&kr);
        const bf162* v2 = reinterpret_cast<const bf162*>(&vr);
        float kf[8], vf[8];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            kf[2 * e] = __low2float(k2[e]); kf[2 * e + 1] = __high2float(k2[e]);
            vf[2 * e] = __low2float(v2[e]); vf[2 * e + 1] = __high2float(v2[e]);
        }
#pragma unroll
        for (int g = 0; g < DEC_G; ++g) {
            if (g >= G) break;
            float sc = 0.f;
#pragma unroll
            for (int e = 0; e < 8; ++e) sc = fmaf(qv[g][e], kf[e], sc);
            sc = warp_sum(sc);
            const float mn = fmaxf(m[g], sc);
            const float corr = exp2f(m[g] - mn);
            const float p = exp2f(sc - mn);
            l[g] = l[g] * corr + p;
#pragma unroll
            for (int e = 0; e < 8; ++e) o[g][e] = fmaf(o[g][e], corr, p * vf[e]);
            m[g] = mn;
        }
    }
#pragma unroll
    for (int g = 0; g < DEC_G; ++g) {
        if (lane == 0) { sm_m[warp][g] = m[g]; sm_l[warp][g] = l[g]; }
#pragma unroll
        for (int e = 0; e < 8; ++e) sm_o[warp][g][lane * 8 + e] = o[g][e];
    }
    __syncthreads();
    // combine the 4 warps: thread t handles dims t and t+128 for every head
    for (int g = 0; g < G; ++g) {
        float mm = -INFINITY;
        for (int w = 0; w < 4; ++w) mm = fmaxf(mm, sm_m[w][g]);
        float ll = 0.f, o0 = 0.f, o1 = 0.f;
        for (int w = 0; w < 4; ++w) {
            const float c = sm_m[w][g] == -INFINITY ? 0.f : exp2f(sm_m[w][g] - mm);
            ll += sm_l[w][g] * c;
            o0 += sm_o[w][g][threadIdx.x] * c;
            o1 += sm_o[w][g][threadIdx.x + 128] * c;
        }
        const int h = kh * G + g;
        const int64_t pi = ((int64_t)s * nsplit + sp) * HQ + h;
        if (threadIdx.x == 0) { pm[pi] = mm; pl[pi] = ll; }
        po[pi * D + threadIdx.x] = o0;
        po[pi * D + threadIdx.x + 128] = o1;
    }
}

// Combine the splits: out f32 [S, HQ, 256]. Block (h, s), 256 threads (one per dim).
__global__ void k_attn_decode_combine(const float* __restrict__ pm, const float* __restrict__ pl, const float* __restrict__ po,
                                      const int* __restrict__ len, float* __restrict__ out, int nsplit, int HQ) {
    const int h = blockIdx.x, s = blockIdx.y, d = threadIdx.x;
    const int ns = min(nsplit, (len[s] + 255) / 256);
    float mm = -INFINITY;
    for (int i = 0; i < ns; ++i) mm = fmaxf(mm, pm[((int64_t)s * nsplit + i) * HQ + h]);
    float ll = 0.f, o = 0.f;
    for (int i = 0; i < ns; ++i) {
        const int64_t pi = ((int64_t)s * nsplit + i) * HQ + h;
        const float c = pm[pi] == -INFINITY ? 0.f : exp2f(pm[pi] - mm);
        ll += pl[pi] * c;
        o += po[pi * 256 + d] * c;
    }
    out[((int64_t)s * HQ + h) * 256 + d] = o / ll;
}

// ------------------------------------------------------------------------------------------------ FFN / misc

// h = silu(g) * u  (f32 or bf16 in/out pairs)
__global__ void k_silu_mul_f32(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu_f(g[i]) * u[i];
}
__global__ void k_silu_mul_bf16(const bf16* __restrict__ g, const bf16* __restrict__ u, bf16* __restrict__ h, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = f2bf(silu_f(bf2f(g[i])) * bf2f(u[i]));
}

// Gather rows: out[i, :] = x[idx[i], :]   (f32, N columns)
__global__ void k_gather_rows(const float* __restrict__ x, const int* __restrict__ idx, float* __restrict__ out, int rows, int N) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)rows * N) return;
    const int64_t r = i / N;
    out[i] = x[(int64_t)idx[r] * N + i % N];
}

// Vision: x[tok, :] += sum_j w[tok*4+j] * table[idx[tok*4+j], :]  (f32 table: bilinear, align_corners)
__global__ void k_vis_pos_embed(float* __restrict__ x, const float* __restrict__ table, const int* __restrict__ idx,
                                const float* __restrict__ w, int n, int C) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * C) return;
    const int64_t t = i / C;
    const int c = i % C;
    float acc = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) acc += w[t * 4 + j] * table[(int64_t)idx[t * 4 + j] * C + c];
    x[i] += acc;
}

}  // extern "C"
