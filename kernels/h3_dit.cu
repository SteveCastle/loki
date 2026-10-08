// MiniMax H3 DiT kernels.
//
//  * k_h3_mod_table / k_h3_mod_final : per-forward adaLN tables from the 8-d curve embedding (fp32 GEMV, f16 weights)
//  * k_h3_norm_mod_quant             : RMSNorm -> per-row modulation (scale/shift from a mod-row index) -> ConvRot
//                                      Hadamard-256 -> per-row int8 quantization (input of the next int8 GEMM)
//  * k_h3_qk_norm_rope               : per-head RMSNorm + partial split-half rope (rot_dim 96 of 128), in place
//  * k_h3_quant_v                    : V -> fp8 e4m3, transposed + permuted for the Sage PV mma, per-128-key scale
//  * k_h3_gemm_i8_*                  : int8 tensor-core GEMM (128x128 / 128x256 tiles) with fused epilogues,
//                                      incl. per-row gated residual (gate row chosen by the token's mod row)
//  * k_h3_attn                       : Sage-style int8 QK / fp8 PV flash attention, bidirectional, 1 segment
//  * k_h3_embed                      : patchify / audio packing fused with the fp32 patch projection
//  * k_h3_final_mod / k_h3_head      : final RMSNorm+modulation (fp32) and the fp32 output heads with
//                                      unpatchify / unpack, negation and the audio carry conversion
#include "common.cuh"

// =============================================================================================
// adaLN tables
// =============================================================================================
struct TEmb { float v[4][8]; };

// W f16 [L][NO][8], B f16 [L][NO] with NO = 3 * 6 * hidden. out f32 [L][12][6][hidden]:
// row r = class * 3 + tag; the scale slots (k = 1, 4) hold bf16(1 + bf16(v)), the others bf16(v).
extern "C" __global__ void k_h3_mod_table(const __half* __restrict__ W, const __half* __restrict__ B, int L, int hidden,
                                          const TEmb te, float* __restrict__ out) {
    const int NO = 18 * hidden;
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= (int64_t)L * NO) return;
    int l = (int)(gid / NO);
    int o = (int)(gid % NO);
    int tag = o / (6 * hidden);
    int k = (o / hidden) % 6;
    int j = o % hidden;
    const __half* w = W + gid * 8;
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wf[i] = __half2float(w[i]);
    float b = __half2float(B[gid]);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = fmaf(wf[i], te.v[c][i], acc);
        float v = round_bf16(acc + b);
        if (k == 1 || k == 4) v = round_bf16(1.0f + v);
        out[(((int64_t)l * 12 + c * 3 + tag) * 6 + k) * hidden + j] = v;
    }
}

// final layer: W f16 [2*hidden][8]. out f32 [4][2][hidden] (shift, scale) unrounded.
extern "C" __global__ void k_h3_mod_final(const __half* __restrict__ W, const __half* __restrict__ B, int hidden,
                                          const TEmb te, float* __restrict__ out) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= 2 * hidden) return;
    float wf[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) wf[i] = __half2float(W[(int64_t)o * 8 + i]);
    float b = __half2float(B[o]);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) acc = fmaf(wf[i], te.v[c][i], acc);
        out[(int64_t)c * 2 * hidden + o] = acc + b;
    }
}

// =============================================================================================
// Hadamard-256 (ConvRot) + per-row int8 quantization of a row held in shared memory (f32).
// =============================================================================================
__device__ void hadamard_quant_row(float* row, int K, int8_t* __restrict__ out, float* __restrict__ scale_out) {
    __shared__ float red[32];
    const int t = threadIdx.x;
    const int ngroups = K / 256;
#pragma unroll
    for (int st = 1; st < 256; st <<= 2) {
        for (int job = t; job < 64 * ngroups; job += 256) {
            int grp = job >> 6, j = job & 63;
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
    float amax = 0.f;
    for (int i = t; i < K; i += 256) amax = fmaxf(amax, fabsf(row[i]));
    amax = block_max(amax, red) * 0.0625f;
    float scale = amax / 127.0f;
    float inv = (scale > 0.f) ? (1.0f / scale) : 0.f;
    if (t == 0) scale_out[0] = (scale > 0.f) ? scale : 1.17549435e-38f;
    for (int i = t * 4; i < K; i += 256 * 4) {
        char4 q;
        q.x = (int8_t)max(-128, min(127, __float2int_rn(row[i] * 0.0625f * inv)));
        q.y = (int8_t)max(-128, min(127, __float2int_rn(row[i + 1] * 0.0625f * inv)));
        q.z = (int8_t)max(-128, min(127, __float2int_rn(row[i + 2] * 0.0625f * inv)));
        q.w = (int8_t)max(-128, min(127, __float2int_rn(row[i + 3] * 0.0625f * inv)));
        *reinterpret_cast<char4*>(out + i) = q;
    }
}

// x bf16 [M][K] -> RMSNorm(w, eps) -> h * modtab[r][k_scale] + modtab[r][k_shift] (bf16 rounding like torch) -> int8
// modtab: f32 [rows][6][K] for this layer; r = modrow[m].
extern "C" __global__ void __launch_bounds__(256) k_h3_norm_mod_quant(const bf16* __restrict__ x, int M, int K, const bf16* __restrict__ w, float eps,
                                                                     const int* __restrict__ modrow, const float* __restrict__ modtab,
                                                                     int k_shift, int k_scale, int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    __shared__ float red2[32];
    int64_t m = blockIdx.x;
    if (m >= M) return;
    const int t = threadIdx.x;
    const bf16* xr = x + m * K;
    float ss = 0.f;
    for (int i = t * 8; i < K; i += 256 * 8) {
        uint4 raw = *reinterpret_cast<const uint4*>(xr + i);
        const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float a = __low2float(h2[j]), b = __high2float(h2[j]);
            row[i + 2 * j] = a; row[i + 2 * j + 1] = b;
            ss += a * a + b * b;
        }
    }
    float r = rsqrtf(block_sum(ss, red2) / K + eps);
    const int mr = modrow[m];
    const float* sh = modtab + ((int64_t)mr * 6 + k_shift) * K;
    const float* sc = modtab + ((int64_t)mr * 6 + k_scale) * K;
    for (int i = t; i < K; i += 256) {
        float h = round_bf16(row[i] * r * bf2f(w[i]));
        h = round_bf16(h * sc[i]);
        row[i] = round_bf16(h + sh[i]);
    }
    __syncthreads();
    hadamard_quant_row(row, K, out + m * K, scale + m);
}

// =============================================================================================
// q/k per-head RMSNorm + partial split-half rope (in place, bf16).
// qkv rows with token stride ts; q at column 0, k at column kofs; H heads of 128.
// rope: bf16 [M][48][2] (cos, sin) or null (norm only). Rotates dims [0,48) with [48,96).
// One warp per (token, q|k, head); lane owns dims 4*lane .. 4*lane+3.
// =============================================================================================
extern "C" __global__ void k_h3_qk_norm_rope(bf16* __restrict__ qkv, int64_t ts, int M, int H, int kofs,
                                            const bf16* __restrict__ wq, const bf16* __restrict__ wk, float eps,
                                            const bf16* __restrict__ rope, int nw, int rot) {
    // nw = 2: q at column 0 and k at column kofs; nw = 1: one tensor (q only) at column 0
    // rot = 1: finally apply the orthonormal Sylvester Hadamard H128 per head (q.k invariant; flattens outliers
    //          before the int8 attention quantization)
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * H * nw) return;
    int64_t m = gw / (nw * H);
    int rem = (int)(gw % (nw * H));
    int which = rem / H, h = rem % H;
    bf16* p = qkv + m * ts + (which ? kofs : 0) + h * 128 + lane * 4;
    const bf16* w = (which ? wk : wq) + lane * 4;
    uint2 raw = *reinterpret_cast<const uint2*>(p);
    const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
    float v[4] = {__low2float(h2[0]), __high2float(h2[0]), __low2float(h2[1]), __high2float(h2[1])};
    float ss = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
    ss = warp_sum(ss);
    float r = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = round_bf16(v[e] * r * bf2f(w[e]));
    if (rope) {
        int src = lane < 12 ? lane + 12 : (lane < 24 ? lane - 12 : lane);
        float o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = __shfl_sync(0xffffffff, v[e], src);
        if (lane < 24) {
            const bf16* rp = rope + (m * 48 + (lane % 12) * 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float c = bf2f(rp[2 * e]), s = bf2f(rp[2 * e + 1]);
                if (lane < 12) v[e] = round_bf16(v[e] * c - o[e] * s);
                else v[e] = round_bf16(v[e] * c + o[e] * s);
            }
        }
    }
    if (rot) {
        // dims d = 4*lane + e: bits 0-1 in-lane, bits 2-6 across lanes
        float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
        v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
#pragma unroll
        for (int b = 1; b < 32; b <<= 1) {
            const bool up = lane & b;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float o = __shfl_xor_sync(0xffffffff, v[e], b);
                v[e] = up ? (o - v[e]) : (v[e] + o);
            }
        }
        const float s = 0.08838834764831845f;  // 1/sqrt(128)
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= s;
    }
    bf162 a = __floats2bfloat162_rn(v[0], v[1]), b = __floats2bfloat162_rn(v[2], v[3]);
    uint2 st;
    st.x = *reinterpret_cast<uint32_t*>(&a);
    st.y = *reinterpret_cast<uint32_t*>(&b);
    *reinterpret_cast<uint2*>(p) = st;
}

// =============================================================================================
// V -> fp8 (e4m3) transposed per head with the 32-key mma permutation, per-(128-key tile, head) scale.
// v bf16 rows (token stride ts), n valid rows starting at global token tok0 (multiple of 128).
// vt u8 [H][128][s_pad], sv f32 [s_pad/128][H]. grid (ceil(n/128), H), dyn smem 128*129*4.
// =============================================================================================
extern "C" __global__ void __launch_bounds__(256) k_h3_quant_v(const bf16* __restrict__ v, int64_t ts, int n, int H, const float* __restrict__ mean,
                                                              uint8_t* __restrict__ vt, int64_t s_pad, int tok0, float* __restrict__ sv) {
    extern __shared__ float tile[];  // [128][129]
    __shared__ float red[32];
    const int tb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    float amax = 0.f;
    for (int e = tid; e < 128 * 64; e += 256) {
        int key = e >> 6, d2 = (e & 63) * 2;
        int j = tb * 128 + key;
        float a = 0.f, b = 0.f;
        if (j < n) {
            bf162 x2 = *reinterpret_cast<const bf162*>(v + (int64_t)j * ts + h * 128 + d2);
            a = __low2float(x2) - mean[h * 128 + d2];
            b = __high2float(x2) - mean[h * 128 + d2 + 1];
        }
        tile[key * 129 + d2] = a;
        tile[key * 129 + d2 + 1] = b;
        amax = fmaxf(amax, fmaxf(fabsf(a), fabsf(b)));
    }
    amax = block_max(amax, red);
    float s = amax / 448.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    const int gt = tok0 / 128 + tb;
    if (tid == 0) sv[(int64_t)gt * H + h] = s > 0.f ? s : 1.f;
    __syncthreads();
    // each thread writes 4 consecutive positions (one u32) of a dim row
    for (int e = tid; e < 128 * 32; e += 256) {
        int d = e >> 5, pos0 = (e & 31) * 4;
        uint32_t packed = 0;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            int pos = pos0 + u;
            int grp = pos >> 5, pp = pos & 31;
            int half = pp >> 4, tt = (pp & 15) >> 2, i = pp & 3;
            int key = grp * 32 + half * 16 + (i >> 1) * 8 + 2 * tt + (i & 1);
            __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(tile[key * 129 + d] * inv, __NV_SATFINITE, __NV_E4M3);
            packed |= ((uint32_t)f) << (8 * u);
        }
        *reinterpret_cast<uint32_t*>(vt + ((int64_t)h * 128 + d) * s_pad + gt * 128 + pos0) = packed;
    }
}

// =============================================================================================
// int8 GEMM (copy of the shared engine's kernel with an extra epilogue):
//   v = acc * sa[m] * sb[n] (+ bias[n])
//   mode 0: out = v ; 1: out = res + v ; 3: SwiGLU pairs (bf16 out, width N/2)
//   mode 4: out = bf16(res + bf16(v) * gate[modrow[m] * gstride + n])   (torch addcmul on bf16)
// =============================================================================================
namespace h3gemm {

constexpr int BM = 128, BK = 64;
constexpr int THREADS = 256;

struct Epi {
    const float* bias; int mode; const bf16* res; const float* gate; const int* modrow; int64_t gstride;
};

template <int BN, int STAGES>
__device__ void gemm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, bf16* __restrict__ C,
                            int M, int N, int K, const float* __restrict__ sa, const float* __restrict__ sb, const Epi ep) {
    constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
    constexpr int WN = BN / 4;
    constexpr int NI = WN / 8;
    constexpr int ACH = A_BYTES / 16 / THREADS;
    constexpr int BCH = B_BYTES / 16 / THREADS;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * A_BYTES;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int wm = warp >> 2;
    const int wn = warp & 3;
    const int n_tiles = (N + BN - 1) / BN;
    const int m_tiles = (M + BM - 1) / BM;
    constexpr int GROUP_M = 16;
    const int bid = blockIdx.x;
    const int group = bid / (GROUP_M * n_tiles);
    const int first_m = group * GROUP_M;
    const int gsize = min(GROUP_M, m_tiles - first_m);
    const int in_group = bid - group * GROUP_M * n_tiles;
    const int mt = first_m + (in_group % gsize);
    const int nt = in_group / gsize;
    const int m0 = mt * BM;
    const int n0 = nt * BN;

    int a_row[ACH], a_chunk[ACH], b_row[BCH], b_chunk[BCH];
    const int8_t* a_src[ACH];
    const int8_t* b_src[BCH];
    bool a_pred[ACH], b_pred[BCH];
#pragma unroll
    for (int i = 0; i < ACH; ++i) {
        int id = tid + i * THREADS;
        a_row[i] = id >> 2;
        a_chunk[i] = id & 3;
        int am = m0 + a_row[i];
        a_pred[i] = am < M;
        a_src[i] = A + (int64_t)(a_pred[i] ? am : 0) * K + a_chunk[i] * 16;
    }
#pragma unroll
    for (int i = 0; i < BCH; ++i) {
        int id = tid + i * THREADS;
        b_row[i] = id >> 2;
        b_chunk[i] = id & 3;
        int bn = n0 + b_row[i];
        b_pred[i] = bn < N;
        b_src[i] = B + (int64_t)(b_pred[i] ? bn : 0) * K + b_chunk[i] * 16;
    }
    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * A_BYTES;
        uint8_t* bs = Bs + stage * B_BYTES;
#pragma unroll
        for (int i = 0; i < ACH; ++i) cp_async_16(smem_u32(as + swz64(a_row[i], a_chunk[i])), a_src[i] + k0, a_pred[i]);
#pragma unroll
        for (int i = 0; i < BCH; ++i) cp_async_16(smem_u32(bs + swz64(b_row[i], b_chunk[i])), b_src[i] + k0, b_pred[i]);
    };

    int32_t acc[4][NI][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;

    const int KT = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }
    const int lidx = lane >> 3;
    const int lrow = lane & 7;
    for (int kt = 0; kt < KT; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk, nk % STAGES);
            cp_async_commit();
        }
        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * A_BYTES;
        const uint8_t* bs = Bs + stage * B_BYTES;
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
    cp_async_wait<0>();

    const int g = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int m = m0 + wm * 64 + mi * 16 + g + half * 8;
            if (m >= M) continue;
            float rs = sa[m];
            const float* grow = (ep.mode == 4) ? ep.gate + (int64_t)ep.modrow[m] * ep.gstride : nullptr;
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                int n = n0 + wn * WN + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = (float)acc[mi][ni][half * 2 + 0] * rs * sb[n];
                float v1 = (float)acc[mi][ni][half * 2 + 1] * rs * sb[n + 1];
                if (ep.bias) { v0 += ep.bias[n]; v1 += ep.bias[n + 1]; }
                if (ep.mode == 3) {
                    float o = silu_f(v0) * v1;
                    float o1 = __shfl_xor_sync(0xffffffff, o, 1);
                    uint32_t pk = pack_bf16x2(o, o1);
                    uint32_t pk2 = __shfl_xor_sync(0xffffffff, pk, 2);
                    if (t4 == 0) *reinterpret_cast<uint2*>(C + (int64_t)m * (N >> 1) + (n >> 1)) = make_uint2(pk, pk2);
                    continue;
                }
                int64_t off = (int64_t)m * N + n;
                if (ep.mode == 1) {
                    bf162 r2 = *reinterpret_cast<const bf162*>(ep.res + off);
                    v0 += __low2float(r2); v1 += __high2float(r2);
                } else if (ep.mode == 4) {
                    bf162 r2 = *reinterpret_cast<const bf162*>(ep.res + off);
                    v0 = __low2float(r2) + round_bf16(v0) * grow[n];
                    v1 = __high2float(r2) + round_bf16(v1) * grow[n + 1];
                }
                *reinterpret_cast<bf162*>(C + off) = __floats2bfloat162_rn(v0, v1);
            }
        }
    }
}

}  // namespace h3gemm

extern "C" __global__ void __launch_bounds__(256) k_h3_gemm_i8(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                              const float* sa, const float* sb, const h3gemm::Epi ep) {
    h3gemm::gemm_kernel<128, 4>(A, B, C, M, N, K, sa, sb, ep);
}
extern "C" __global__ void __launch_bounds__(256) k_h3_gemm_i8_w(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                                const float* sa, const float* sb, const h3gemm::Epi ep) {
    h3gemm::gemm_kernel<256, 3>(A, B, C, M, N, K, sa, sb, ep);
}
extern "C" __global__ void __launch_bounds__(256) k_h3_gemm_i8_w4(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                                 const float* sa, const float* sb, const h3gemm::Epi ep) {
    h3gemm::gemm_kernel<256, 4>(A, B, C, M, N, K, sa, sb, ep);
}

// =============================================================================================
// Patch embedding: rows gathered from a channel-first latent, fp32 projection (weight transposed [Kf][N]) -> bf16.
// kind 0: video [24][T][H][W] -> row (t, hh, ww) of the 2x2 patch grid, feature c*4 + p*2 + q
// kind 1: audio [32][2][T]    -> row ch*T + t, feature c (times in_scale)
// 8 rows per block, 256 threads.
// =============================================================================================
extern "C" __global__ void __launch_bounds__(256) k_h3_embed(const float* __restrict__ src, int kind, int T, int Hl, int Wl, int nrows, float in_scale,
                                                            const float* __restrict__ wt, const float* __restrict__ bias, int Kf, int N,
                                                            bf16* __restrict__ out) {
    __shared__ float feat[8][96];
    const int r0 = blockIdx.x * 8;
    const int tid = threadIdx.x;
    for (int e = tid; e < 8 * Kf; e += 256) {
        int rr = e / Kf, f = e % Kf;
        int r = r0 + rr;
        float v = 0.f;
        if (r < nrows) {
            if (kind == 0) {
                int h2 = Hl / 2, w2 = Wl / 2;
                int t = r / (h2 * w2), rem = r % (h2 * w2);
                int hh = rem / w2, ww = rem % w2;
                int c = f >> 2, p = (f >> 1) & 1, q = f & 1;
                v = src[(((int64_t)c * T + t) * Hl + hh * 2 + p) * Wl + ww * 2 + q];
            } else {
                int ch = r / T, t = r % T;
                v = __fmul_rn(src[((int64_t)f * 2 + ch) * T + t], in_scale);
            }
        }
        feat[rr][f] = v;
    }
    __syncthreads();
    const int nr = min(8, nrows - r0);
    for (int n = tid; n < N; n += 256) {
        float acc[8];
        float b = bias[n];
#pragma unroll
        for (int rr = 0; rr < 8; ++rr) acc[rr] = 0.f;
        for (int f = 0; f < Kf; ++f) {
            float wv = wt[(int64_t)f * N + n];
#pragma unroll
            for (int rr = 0; rr < 8; ++rr) acc[rr] = fmaf(feat[rr][f], wv, acc[rr]);
        }
        for (int rr = 0; rr < nr; ++rr) out[(int64_t)(r0 + rr) * N + n] = f2bf(acc[rr] + b);
    }
}

// =============================================================================================
// Final layer
// =============================================================================================
// x bf16 [n][K] -> f32 [n][K]: bf16(RMSNorm(x) * w) * (1 + scale) + shift   (fp32 modulation)
extern "C" __global__ void __launch_bounds__(256) k_h3_final_mod(const bf16* __restrict__ x, int n, int K, const bf16* __restrict__ w, float eps,
                                                                const float* __restrict__ shift, const float* __restrict__ scale, float* __restrict__ out) {
    __shared__ float red[32];
    int64_t m = blockIdx.x;
    if (m >= n) return;
    const bf16* xr = x + m * K;
    float ss = 0.f;
    for (int i = threadIdx.x; i < K; i += 256) { float v = bf2f(xr[i]); ss += v * v; }
    float r = rsqrtf(block_sum(ss, red) / K + eps);
    for (int i = threadIdx.x; i < K; i += 256) {
        float h = round_bf16(bf2f(xr[i]) * r * bf2f(w[i]));
        out[m * K + i] = __fadd_rn(__fmul_rn(h, __fadd_rn(1.0f, scale[i])), shift[i]);
    }
}

// fp32 head: o[r][c] = A[r] . W[c] + bias[c]; out element = alpha * (y * ycarry) + beta * (-o)
// kind 0: video unpatchify into [24][T][H][W] (row (t,hh,ww), c = ch*4 + p*2 + q); kind 1: audio [32][2][T] (row ch*T+t).
// Block: 64 rows x No (<= 96) cols, 256 threads, K tiles of 32.
extern "C" __global__ void __launch_bounds__(256) k_h3_head(const float* __restrict__ A, int n, int K, const float* __restrict__ W,
                                                           const float* __restrict__ bias, int No, int kind, int T, int Hl, int Wl, int row0,
                                                           const float* __restrict__ y, float ycarry, float alpha, float beta,
                                                           float* __restrict__ out) {
    __shared__ float As[64][33];
    __shared__ float Ws[96][33];
    const int tid = threadIdx.x;
    const int r0 = blockIdx.x * 64;
    const int rg = tid >> 5;      // rows rg*8 .. +8
    const int cl = tid & 31;      // cols cl + 32*j
    float acc[8][3];
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = 0.f;
    for (int k0 = 0; k0 < K; k0 += 32) {
        for (int e = tid; e < 64 * 32; e += 256) {
            int rr = e >> 5, kk = e & 31;
            int r = r0 + rr;
            As[rr][kk] = r < n ? A[(int64_t)r * K + k0 + kk] : 0.f;
        }
        for (int e = tid; e < No * 32; e += 256) {
            int c = e >> 5, kk = e & 31;
            Ws[c][kk] = W[(int64_t)c * K + k0 + kk];
        }
        __syncthreads();
#pragma unroll 4
        for (int kk = 0; kk < 32; ++kk) {
            float wv[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) wv[j] = (cl + 32 * j < No) ? Ws[cl + 32 * j][kk] : 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float a = As[rg * 8 + i][kk];
#pragma unroll
                for (int j = 0; j < 3; ++j) acc[i][j] = fmaf(a, wv[j], acc[i][j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int rl = r0 + rg * 8 + i;
        if (rl >= n) continue;
        int r = rl + row0;  // global row (index math); A is chunk-local
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            int c = cl + 32 * j;
            if (c >= No) continue;
            float o = -(acc[i][j] + bias[c]);
            int64_t idx;
            if (kind == 0) {
                int h2 = Hl / 2, w2 = Wl / 2;
                int t = r / (h2 * w2), rem = r % (h2 * w2);
                int hh = rem / w2, ww = rem % w2;
                int ch = c >> 2, p = (c >> 1) & 1, q = c & 1;
                idx = (((int64_t)ch * T + t) * Hl + hh * 2 + p) * Wl + ww * 2 + q;
            } else {
                int ch = r / T, t = r % T;
                idx = ((int64_t)c * 2 + ch) * T + t;
            }
            float v = __fmul_rn(beta, o);
            if (y) v = __fadd_rn(__fmul_rn(alpha, __fmul_rn(y[idx], ycarry)), v);
            out[idx] = v;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// micro-benchmark of tensor-core instruction throughput (mode 0: s8 m16n8k32, 1: e4m3 f32-acc m16n8k32,
// 2: f16 f16-acc m16n8k16, 3: bf16 f32-acc m16n8k16, 4: f16 f32-acc m16n8k16)
DEVI void mma_f16_16816_f16acc(uint32_t* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
DEVI void mma_f16_16816_f32acc(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int MODE>
__device__ void mma_bench_t(int iters, float* out) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, 7u, 9u}, b[2] = {threadIdx.x * 5u, 11u};
    float fc[16][4] = {};
    int32_t ic[16][4] = {};
    uint32_t hc[16][2] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            if (MODE == 0) mma_s8_16832(ic[j], a, b);
            else if (MODE == 1) mma_e4m3_16832(fc[j], a, b);
            else if (MODE == 2) mma_f16_16816_f16acc(hc[j], a, b);
            else if (MODE == 3) mma_bf16_16816(fc[j], a, b);
            else if (MODE == 5) { if (j % 3 == 0) mma_s8_16832(ic[j], a, b); else mma_f16_16816_f16acc(hc[j], a, b); }
            else if (MODE == 6) { if (j < 5) mma_s8_16832(ic[j], a, b); else mma_f16_16816_f16acc(hc[j], a, b); }
            else if (MODE == 7) { if (j & 1) mma_s8_16832(ic[j], a, b); else mma_e4m3_16832(fc[j], a, b); }
            else mma_f16_16816_f32acc(fc[j], a, b);
        }
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 16; ++j) s += fc[j][0] + (float)ic[j][0] + (float)hc[j][0];
    if (s == 12345.f) out[0] = s;
}
extern "C" __global__ void __launch_bounds__(128) k_h3_mma_bench(int mode, int iters, float* out) {
    switch (mode) {
        case 0: mma_bench_t<0>(iters, out); break;
        case 1: mma_bench_t<1>(iters, out); break;
        case 2: mma_bench_t<2>(iters, out); break;
        case 3: mma_bench_t<3>(iters, out); break;
        case 5: mma_bench_t<5>(iters, out); break;
        case 6: mma_bench_t<6>(iters, out); break;
        case 7: mma_bench_t<7>(iters, out); break;
        default: mma_bench_t<4>(iters, out); break;
    }
}


// =============================================================================================
// V -> fp16, smoothed by the per-channel mean and scaled per (256-key group, head) to |v| <= 128, so the fp16
// accumulators of the PV mma (flushed every 256 keys) cannot overflow (256 * 1 * 128 = 32768 < 65504).
// v bf16 rows (token stride ts), n valid rows at global token tok0 (multiple of 256).
// out: v16 [s_pad][H][128] fp16 (rows >= n of the group zeroed), sv f32 [s_pad/256][H].
// grid (ceil(n/256), H), 256 threads (thread = one key).
// =============================================================================================
constexpr int H3_VGROUP = 256;
extern "C" __global__ void __launch_bounds__(256) k_h3_quant_v16(const bf16* __restrict__ v, int64_t ts, int n, int H, const float* __restrict__ mean,
                                                                __half* __restrict__ v16, int tok0, float* __restrict__ sv) {
    __shared__ float red[32];
    __shared__ float mn[128];
    const int tb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid < 128) mn[tid] = mean[h * 128 + tid];
    __syncthreads();
    const int j = tb * H3_VGROUP + tid;
    const bf16* src = v + (int64_t)j * ts + h * 128;
    float amax = 0.f;
    if (j < n) {
#pragma unroll 4
        for (int c = 0; c < 16; ++c) {
            uint4 raw = *reinterpret_cast<const uint4*>(src + c * 8);
            const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                amax = fmaxf(amax, fabsf(__low2float(h2[e]) - mn[c * 8 + 2 * e]));
                amax = fmaxf(amax, fabsf(__high2float(h2[e]) - mn[c * 8 + 2 * e + 1]));
            }
        }
    }
    amax = block_max(amax, red);
    const float s = amax > 0.f ? amax / 128.f : 1.f;
    const float inv = 1.f / s;
    const int gt = tok0 / H3_VGROUP + tb;
    if (tid == 0) sv[(int64_t)gt * H + h] = s;
    __half* dst = v16 + ((int64_t)(tok0 + tb * H3_VGROUP + tid) * H + h) * 128;
#pragma unroll 4
    for (int c = 0; c < 16; ++c) {
        uint4 o = make_uint4(0, 0, 0, 0);
        if (j < n) {
            uint4 raw = *reinterpret_cast<const uint4*>(src + c * 8);
            const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
            uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                __half2 hh = __floats2half2_rn((__low2float(h2[e]) - mn[c * 8 + 2 * e]) * inv, (__high2float(h2[e]) - mn[c * 8 + 2 * e + 1]) * inv);
                ow[e] = *reinterpret_cast<uint32_t*>(&hh);
            }
        }
        *reinterpret_cast<uint4*>(dst + c * 8) = o;
    }
}

// =============================================================================================
// Bidirectional attention, one segment, head dim 128:
//   S = Q8 K8^T (int8 mma, per-(token, head) scales; K smoothed by a per-channel mean, exact for softmax)
//   O = P V (fp16 mma with fp16 accumulators over a 256-key group, flushed into fp32 with the group's V scale)
// q8 [nq][H][128], sq [nq][H]; k8 [nk][H][128], sk [nk][H]; v16 [nk_pad][H][128] (zero padded), sv [nk_pad/256][H];
// mean_v [H*128]; out bf16 rows (token stride o_ts) at column h*128.
// Block: 128 queries (8 warps x 16 rows), 64-key tiles, STAGES-deep cp.async pipeline.
// =============================================================================================
namespace h3attn {
constexpr int BQ = 128, BKV = 64, THREADS = 256;
constexpr int GROUP_TILES = H3_VGROUP / BKV;
constexpr int SMEM_Q = BQ * 128;
constexpr int SMEM_K = BKV * 128;
constexpr int SMEM_V = BKV * 256;
constexpr int SMEM_SK = BKV * 4;
constexpr int STAGE = SMEM_K + SMEM_V + SMEM_SK;
struct Params {
    const int8_t* q; const float* sq;
    const int8_t* k; const float* sk;
    const __half* v; const float* sv; const float* mv;
    bf16* o; int64_t o_ts;
    int nq, nk, H;
    float scale_log2;
};
DEVI int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
DEVI int swz256(int row, int chunk) { return row * 256 + ((chunk ^ (row & 7)) << 4); }
DEVI void mma_f16acc(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(c[0]), "+r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
DEVI uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
// exact int32 -> float for |x| < 2^22 on the integer + FMA pipes (avoids I2F)
DEVI float i2f(int x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }
}  // namespace h3attn

// Software-pipelined variant: at iteration t the QK^T MMAs of tile t+1 are issued before the softmax of tile t,
// so the tensor pipe works on them while the ALU/MUFU pipes run the softmax. 3 smem stages:
// tile t (V for PV), tile t+1 (K for QK), tile t+2 (in flight).
template <bool PIPE, int ABL = 0>
__device__ void h3_attn_kernel(const h3attn::Params p) {
    using namespace h3attn;
    constexpr int STAGES = 3;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* St = smem + SMEM_Q;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int ntiles = (p.nk + BKV - 1) / BKV;

    for (int c = tid; c < BQ * 8; c += THREADS) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16, pred);
    }
    auto load_kv = [&](int tile, int stage) {
        if ((ABL & 8) && tile >= 2) return;
        const int j0 = tile * BKV;
        uint8_t* ks = St + stage * STAGE;
        uint8_t* vs = ks + SMEM_K;
        float* sks = reinterpret_cast<float*>(vs + SMEM_V);
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int c = tid + it * THREADS;
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            bool pred = j < p.nk;
            cp_async_16(smem_u32(ks + swz128(row, ch)), p.k + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16, pred);
        }
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = tid + it * THREADS;
            int row = c >> 4, ch = c & 15;
            cp_async_16(smem_u32(vs + swz256(row, ch)), p.v + ((int64_t)(j0 + row) * H + h) * 128 + ch * 8, true);
        }
        if (tid < BKV) {
            int j = j0 + tid;
            sks[tid] = (j < p.nk) ? p.sk[(int64_t)j * H + h] : 0.f;
        }
    };
    // prologue: tiles 0 and 1 in flight
    load_kv(0, 0);
    cp_async_commit();
    if (1 < ntiles) load_kv(1, 1);
    cp_async_commit();

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    const bool va = row_a < p.nq, vb = row_b < p.nq;
    const float sq_a = (va ? p.sq[(int64_t)row_a * H + h] : 0.f) * p.scale_log2;
    const float sq_b = (vb ? p.sq[(int64_t)row_b * H + h] : 0.f) * p.scale_log2;

    uint32_t qf[4][4];
    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    uint32_t oh[16][2];
#pragma unroll
    for (int j = 0; j < 16; ++j) oh[j][0] = oh[j][1] = 0u;
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    float ga_a = 1.f, ga_b = 1.f;

    auto qk = [&](int stage, int32_t (&s)[8][4]) {
        const uint8_t* ks = St + stage * STAGE;
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0;
        if (ABL & 1) { s[0][0] = stage; return; }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t b0, b1, b2, b3;
                int row = (j + (lidx >> 1)) * 8 + lrow;
                int ch = 2 * kk + (lidx & 1);
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + swz128(row, ch)));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_s8_16832(s[j], qf[kk], bb0);
                mma_s8_16832(s[j + 1], qf[kk], bb1);
            }
        }
    };

    int32_t s_cur[8][4];
    // tile 0 ready -> Q frags + QK(0)
    cp_async_wait<1>();
    __syncthreads();
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int row = warp * 16 + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
    }
    qk(0, s_cur);

    for (int t = 0; t < ntiles; ++t) {
        // tile t+1 landed and every warp finished PV(t-1) (its stage is reused for tile t+2)
        cp_async_wait<0>();
        __syncthreads();
        if (t + 2 < ntiles) load_kv(t + 2, (t + 2) % STAGES);
        cp_async_commit();
        int32_t s_next[8][4];
        if (PIPE && t + 1 < ntiles) qk((t + 1) % STAGES, s_next);

        const uint8_t* vs = St + (t % STAGES) * STAGE + SMEM_K;
        const float* sks = reinterpret_cast<const float*>(vs + SMEM_V);
        const int j0 = t * BKV;
        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const bool full = j0 + BKV <= p.nk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int kl = j * 8 + t4 * 2;
            float2 sk2 = *reinterpret_cast<const float2*>(sks + kl);
            sf[j][0] = i2f(s_cur[j][0]) * (sq_a * sk2.x);
            sf[j][1] = i2f(s_cur[j][1]) * (sq_a * sk2.y);
            sf[j][2] = i2f(s_cur[j][2]) * (sq_b * sk2.x);
            sf[j][3] = i2f(s_cur[j][3]) * (sq_b * sk2.y);
            if (!full) {
                if (j0 + kl >= p.nk) { sf[j][0] = -INFINITY; sf[j][2] = -INFINITY; }
                if (j0 + kl + 1 >= p.nk) { sf[j][1] = -INFINITY; sf[j][3] = -INFINITY; }
            }
            mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        const float alpha_a = fast_exp2(m_a - mn_a), alpha_b = fast_exp2(m_b - mn_b);  // exp2(-inf) = 0
        m_a = mn_a; m_b = mn_b;
        float rs_a = 0.f, rs_b = 0.f;
        uint32_t pf[4][4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            if (ABL & 4) {
                pf[kk][0] = __float_as_uint(sf[2 * kk][0]); pf[kk][1] = __float_as_uint(sf[2 * kk][1]);
                pf[kk][2] = __float_as_uint(sf[2 * kk + 1][2]); pf[kk][3] = __float_as_uint(sf[2 * kk + 1][3]);
                rs_a += sf[2 * kk][0]; rs_b += sf[2 * kk][3];
                continue;
            }
            float p0 = fast_exp2(sf[2 * kk][0] - mn_a), p1 = fast_exp2(sf[2 * kk][1] - mn_a);
            float p2 = fast_exp2(sf[2 * kk][2] - mn_b), p3 = fast_exp2(sf[2 * kk][3] - mn_b);
            float p4 = fast_exp2(sf[2 * kk + 1][0] - mn_a), p5 = fast_exp2(sf[2 * kk + 1][1] - mn_a);
            float p6 = fast_exp2(sf[2 * kk + 1][2] - mn_b), p7 = fast_exp2(sf[2 * kk + 1][3] - mn_b);
            rs_a += (p0 + p1) + (p4 + p5);
            rs_b += (p2 + p3) + (p6 + p7);
            pf[kk][0] = pack_h2(p0, p1);
            pf[kk][1] = pack_h2(p2, p3);
            pf[kk][2] = pack_h2(p4, p5);
            pf[kk][3] = pack_h2(p6, p7);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a;
        l_b = l_b * alpha_b + rs_b;
        if (__any_sync(0xffffffff, (alpha_a != 1.f) || (alpha_b != 1.f))) {
            const __half2 fa = __float2half2_rn(alpha_a), fb = __float2half2_rn(alpha_b);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                __half2 x0 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][0]), fa);
                __half2 x1 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][1]), fb);
                oh[j][0] = *reinterpret_cast<uint32_t*>(&x0);
                oh[j][1] = *reinterpret_cast<uint32_t*>(&x1);
            }
            ga_a *= alpha_a;
            ga_b *= alpha_b;
        }
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            if (ABL & 2) { oh[kk][0] ^= pf[kk][0] ^ pf[kk][1] ^ pf[kk][2] ^ pf[kk][3]; continue; }
#pragma unroll
            for (int dj = 0; dj < 16; dj += 2) {
                uint32_t b0, b1, b2, b3;
                int row = kk * 16 + (lidx & 1) * 8 + lrow;
                int ch = dj + (lidx >> 1);
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vs + swz256(row, ch)));
                mma_f16acc(oh[dj], pf[kk], b0, b1);
                mma_f16acc(oh[dj + 1], pf[kk], b2, b3);
            }
        }
        if ((t % GROUP_TILES) == GROUP_TILES - 1 || t == ntiles - 1) {
            const float svt = p.sv[(int64_t)(j0 / H3_VGROUP) * H + h];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                float2 lo = __half22float2(*reinterpret_cast<__half2*>(&oh[j][0]));
                float2 hi = __half22float2(*reinterpret_cast<__half2*>(&oh[j][1]));
                o_acc[j][0] = fmaf(o_acc[j][0], ga_a, lo.x * svt);
                o_acc[j][1] = fmaf(o_acc[j][1], ga_a, lo.y * svt);
                o_acc[j][2] = fmaf(o_acc[j][2], ga_b, hi.x * svt);
                o_acc[j][3] = fmaf(o_acc[j][3], ga_b, hi.y * svt);
                oh[j][0] = oh[j][1] = 0u;
            }
            ga_a = ga_b = 1.f;
        }
        if (!PIPE) {
            if (t + 1 < ntiles) qk((t + 1) % STAGES, s_next);
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            s_cur[j][0] = s_next[j][0]; s_cur[j][1] = s_next[j][1]; s_cur[j][2] = s_next[j][2]; s_cur[j][3] = s_next[j][3];
        }
    }
    cp_async_wait<0>();

    const float inv_a = l_a > 0.f ? 1.f / l_a : 0.f, inv_b = l_b > 0.f ? 1.f / l_b : 0.f;
    const float* mv = p.mv + h * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        float m0 = mv[d], m1 = mv[d + 1];
        if (va) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][0] * inv_a + m0, o_acc[j][1] * inv_a + m1);
        }
        if (vb) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][2] * inv_b + m0, o_acc[j][3] * inv_b + m1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) k_h3_attn2(const h3attn::Params p) { h3_attn_kernel<false>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn3(const h3attn::Params p) { h3_attn_kernel<true>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl1(const h3attn::Params p) { h3_attn_kernel<true, 1>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl2(const h3attn::Params p) { h3_attn_kernel<true, 2>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl4(const h3attn::Params p) { h3_attn_kernel<true, 4>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl3(const h3attn::Params p) { h3_attn_kernel<true, 3>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl6(const h3attn::Params p) { h3_attn_kernel<true, 6>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl7(const h3attn::Params p) { h3_attn_kernel<true, 7>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl8(const h3attn::Params p) { h3_attn_kernel<true, 8>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl15(const h3attn::Params p) { h3_attn_kernel<true, 15>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_abl11(const h3attn::Params p) { h3_attn_kernel<true, 11>(p); }

// =============================================================================================
// Ping-pong attention (FA3-style scheduling on sm_89): two warp groups (warps 0-3: query rows 0-63, warps 4-7:
// rows 64-127) alternate their MMA phases through named barriers, so one group's softmax (ALU/MUFU) overlaps the
// other group's tensor-core work. K/V stages are tracked with mbarriers (no per-tile __syncthreads):
//   full[s]  : completes when the cp.async copies of the stage landed (cp.async.mbarrier.arrive.noinc by the
//              128 threads of group 1, which owns all loads: it is the trailing group, so it never waits for the
//              leading one when it recycles a stage)
//   empty[s] : completes when all 256 threads finished reading the stage
// Per iteration t of a group: [turn] QK(t) + PV(t-1) [pass turn] release stage t-1, (group 1: refill it with
// tile t-1+STAGES) softmax(t).
// =============================================================================================
namespace h3pp {
DEVI void bar_sync(int id, int n) { asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEVI void bar_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory"); }
DEVI void mbar_init(uint64_t* b, int count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count) : "memory");
}
DEVI void mbar_arrive(uint64_t* b) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared.b64 st, [%0];\n}" ::"r"(smem_u32(b)) : "memory");
}
DEVI void mbar_cp_async_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
DEVI void mbar_wait(uint64_t* b, int parity) {
    uint32_t done = 0;
    while (true) {
        asm volatile("{\n .reg .pred p;\n mbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
                     : "=r"(done)
                     : "r"(smem_u32(b)), "r"(parity)
                     : "memory");
        if (done) break;
    }
}
}  // namespace h3pp

template <int STAGES, int ABL = 0>
__device__ void h3_attn_pp_kernel(const h3attn::Params p) {
    using namespace h3attn;
    using namespace h3pp;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* St = smem + SMEM_Q;
    __shared__ __align__(8) uint64_t full[STAGES], empty[STAGES];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int ntiles = (p.nk + BKV - 1) / BKV;

    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 128);
            mbar_init(&empty[s], 256);
        }
    }
    for (int c = tid; c < BQ * 8; c += THREADS) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16, pred);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // loads by group 1 only (128 threads): K 512 chunks, V 1024 chunks, sk 64 floats
    const int lt = tid - 128;
    auto load_kv = [&](int tile, int stage) {
        if ((ABL & 8) && tile >= STAGES) { mbar_cp_async_arrive(&full[stage]); return; }
        if ((ABL & 16) && tile >= STAGES && (tile & 1)) { mbar_cp_async_arrive(&full[stage]); return; }
        const int j0 = tile * BKV;
        uint8_t* ks = St + stage * STAGE;
        uint8_t* vs = ks + SMEM_K;
        float* sks = reinterpret_cast<float*>(vs + SMEM_V);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            bool pred = j < p.nk;
            cp_async_16(smem_u32(ks + swz128(row, ch)), p.k + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16, pred);
        }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            int c = lt + it * 128;
            int row = c >> 4, ch = c & 15;
            cp_async_16(smem_u32(vs + swz256(row, ch)), p.v + ((int64_t)(j0 + row) * H + h) * 128 + ch * 8, true);
        }
        if (lt < BKV) {
            int j = j0 + lt;
            bool pred = j < p.nk;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(sks + lt)),
                         "l"(p.sk + (int64_t)(pred ? j : 0) * H + h), "r"(pred ? 4 : 0));
        }
        mbar_cp_async_arrive(&full[stage]);
    };
    if (grp == 1) {
        for (int s = 0; s < STAGES; ++s) {
            if (s < ntiles) load_kv(s, s);
        }
    }

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 16;
    const int row_a = q0 + wrow + g, row_b = row_a + 8;
    const bool va = row_a < p.nq, vb = row_b < p.nq;
    const float sq_a = (va ? p.sq[(int64_t)row_a * H + h] : 0.f) * p.scale_log2;
    const float sq_b = (vb ? p.sq[(int64_t)row_b * H + h] : 0.f) * p.scale_log2;

    uint32_t qf[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int row = wrow + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
    }
    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    uint32_t oh[16][2];
#pragma unroll
    for (int j = 0; j < 16; ++j) oh[j][0] = oh[j][1] = 0u;
    uint32_t pf[4][4];
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    float ga_a = 1.f, ga_b = 1.f;

    auto pv = [&](int stage) {
        const uint8_t* vs = St + stage * STAGE + SMEM_K;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int dj = 0; dj < 16; dj += 2) {
                uint32_t b0, b1, b2, b3;
                int row = kk * 16 + (lidx & 1) * 8 + lrow;
                int ch = dj + (lidx >> 1);
                if (ABL & 4) { oh[dj][0] ^= pf[kk][dj & 3]; continue; }
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vs + swz256(row, ch)));
                if (ABL & 2) { oh[dj][0] ^= b0 ^ b1; oh[dj + 1][0] ^= b2 ^ b3; continue; }
                mma_f16acc(oh[dj], pf[kk], b0, b1);
                mma_f16acc(oh[dj + 1], pf[kk], b2, b3);
            }
        }
    };
    auto flush = [&](int tile) {
        const float svt = p.sv[(int64_t)(tile * BKV / H3_VGROUP) * H + h];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float2 lo = __half22float2(*reinterpret_cast<__half2*>(&oh[j][0]));
            float2 hi = __half22float2(*reinterpret_cast<__half2*>(&oh[j][1]));
            o_acc[j][0] = fmaf(o_acc[j][0], ga_a, lo.x * svt);
            o_acc[j][1] = fmaf(o_acc[j][1], ga_a, lo.y * svt);
            o_acc[j][2] = fmaf(o_acc[j][2], ga_b, hi.x * svt);
            o_acc[j][3] = fmaf(o_acc[j][3], ga_b, hi.y * svt);
            oh[j][0] = oh[j][1] = 0u;
        }
        ga_a = ga_b = 1.f;
    };

    if (grp == 1) bar_arrive(1, 256);  // group 0 takes the first MMA turn
    for (int t = 0; t < ntiles; ++t) {
        const int st = t % STAGES;
        mbar_wait(&full[st], (t / STAGES) & 1);
        bar_sync(1 + grp, 256);
        // ---- MMA phase: QK(t), PV(t-1)
        int32_t s[8][4];
        {
            const uint8_t* ks = St + st * STAGE;
#pragma unroll
            for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    int row = (j + (lidx >> 1)) * 8 + lrow;
                    int ch = 2 * kk + (lidx & 1);
                    ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + swz128(row, ch)));
                    if (ABL & 1) { s[j][0] ^= b0 ^ b1; s[j + 1][0] ^= b2 ^ b3; continue; }
                    uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                    mma_s8_16832(s[j], qf[kk], bb0);
                    mma_s8_16832(s[j + 1], qf[kk], bb1);
                }
            }
        }
        if (t > 0) pv((t - 1) % STAGES);
        bar_arrive(1 + (grp ^ 1), 256);
        if (t > 0) {
            const int ps = (t - 1) % STAGES;
            // stage ps (tile t-1) is free once group 0 is done with it (group 1, the trailing group, refills it).
            // Named barrier per stage: no memory fence needed (all smem reads of the stage were consumed by MMAs).
            if (t - 1 + STAGES < ntiles) {
                if (grp == 0) {
                    bar_arrive(3 + ps, 256);
                } else {
                    bar_sync(3 + ps, 256);
                    load_kv(t - 1 + STAGES, ps);
                }
            }
            if ((t - 1) % GROUP_TILES == GROUP_TILES - 1) flush(t - 1);
        }
        // ---- softmax(t)
        const float* sks = reinterpret_cast<const float*>(St + st * STAGE + SMEM_K + SMEM_V);
        const int j0 = t * BKV;
        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const bool full_tile = j0 + BKV <= p.nk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int kl = j * 8 + t4 * 2;
            float2 sk2 = *reinterpret_cast<const float2*>(sks + kl);
            sf[j][0] = i2f(s[j][0]) * (sq_a * sk2.x);
            sf[j][1] = i2f(s[j][1]) * (sq_a * sk2.y);
            sf[j][2] = i2f(s[j][2]) * (sq_b * sk2.x);
            sf[j][3] = i2f(s[j][3]) * (sq_b * sk2.y);
            if (!full_tile) {
                if (j0 + kl >= p.nk) { sf[j][0] = -INFINITY; sf[j][2] = -INFINITY; }
                if (j0 + kl + 1 >= p.nk) { sf[j][1] = -INFINITY; sf[j][3] = -INFINITY; }
            }
            mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        const float alpha_a = fast_exp2(m_a - mn_a), alpha_b = fast_exp2(m_b - mn_b);  // exp2(-inf) = 0
        m_a = mn_a; m_b = mn_b;
        float rs_a = 0.f, rs_b = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p0 = fast_exp2(sf[2 * kk][0] - mn_a), p1 = fast_exp2(sf[2 * kk][1] - mn_a);
            float p2 = fast_exp2(sf[2 * kk][2] - mn_b), p3 = fast_exp2(sf[2 * kk][3] - mn_b);
            float p4 = fast_exp2(sf[2 * kk + 1][0] - mn_a), p5 = fast_exp2(sf[2 * kk + 1][1] - mn_a);
            float p6 = fast_exp2(sf[2 * kk + 1][2] - mn_b), p7 = fast_exp2(sf[2 * kk + 1][3] - mn_b);
            rs_a += (p0 + p1) + (p4 + p5);
            rs_b += (p2 + p3) + (p6 + p7);
            pf[kk][0] = pack_h2(p0, p1);
            pf[kk][1] = pack_h2(p2, p3);
            pf[kk][2] = pack_h2(p4, p5);
            pf[kk][3] = pack_h2(p6, p7);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a;
        l_b = l_b * alpha_b + rs_b;
        if (__any_sync(0xffffffff, (alpha_a != 1.f) || (alpha_b != 1.f))) {
            const __half2 fa = __float2half2_rn(alpha_a), fb = __float2half2_rn(alpha_b);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                __half2 x0 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][0]), fa);
                __half2 x1 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][1]), fb);
                oh[j][0] = *reinterpret_cast<uint32_t*>(&x0);
                oh[j][1] = *reinterpret_cast<uint32_t*>(&x1);
            }
            ga_a *= alpha_a;
            ga_b *= alpha_b;
        }
    }
    // tail: PV of the last tile (outside the turn protocol: the other group no longer needs the tensor pipe badly)
    bar_sync(1 + grp, 256);
    pv((ntiles - 1) % STAGES);
    bar_arrive(1 + (grp ^ 1), 256);
    flush(ntiles - 1);
    if (grp == 0) bar_sync(1, 256);  // consume the final turn token passed by group 1 (keeps barrier counts balanced)

    const float inv_a = l_a > 0.f ? 1.f / l_a : 0.f, inv_b = l_b > 0.f ? 1.f / l_b : 0.f;
    const float* mv = p.mv + h * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        float m0 = mv[d], m1 = mv[d + 1];
        if (va) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][0] * inv_a + m0, o_acc[j][1] * inv_a + m1);
        }
        if (vb) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][2] * inv_b + m0, o_acc[j][3] * inv_b + m1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp(const h3attn::Params p) { h3_attn_pp_kernel<3>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a1(const h3attn::Params p) { h3_attn_pp_kernel<3, 1>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a2(const h3attn::Params p) { h3_attn_pp_kernel<3, 2>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a3(const h3attn::Params p) { h3_attn_pp_kernel<3, 3>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a4(const h3attn::Params p) { h3_attn_pp_kernel<3, 4>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a5(const h3attn::Params p) { h3_attn_pp_kernel<3, 5>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a8(const h3attn::Params p) { h3_attn_pp_kernel<3, 8>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a16(const h3attn::Params p) { h3_attn_pp_kernel<3, 16>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_pp_a13(const h3attn::Params p) { h3_attn_pp_kernel<3, 13>(p); }

// =============================================================================================
// v4: ping-pong scheduling (as above) + padded shared-memory rows (K 144 B, V 272 B: conflict-free ldmatrix with
// all fragment addresses = per-lane base + immediate) + folded score scaling:
//   t = int2float(s) * sk[key]  (one IADD + one FFMA via the magic-number conversion)
//   running max over t; p = exp2(t * sq' - m * sq')  (one FFMA + MUFU), sq' = sq * log2(e) / sqrt(d) > 0
// =============================================================================================
namespace h3v4 {
constexpr int BQ = 128, BKV = 64;
constexpr int KROW = 144, VROW = 272;
constexpr int SMEM_Q = BQ * 128;  // Q keeps the xor swizzle (loaded once)
constexpr int SMEM_K = BKV * KROW;
constexpr int SMEM_V = BKV * VROW;
constexpr int SMEM_SK = BKV * 4;
constexpr int STAGE = SMEM_K + SMEM_V + SMEM_SK;
constexpr int STAGES = 3;
constexpr int SMEM_TOTAL = SMEM_Q + STAGES * STAGE;
constexpr int GROUP_TILES = H3_VGROUP / BKV;
DEVI void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
DEVI void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
}  // namespace h3v4

template <int ABL = 0>
__device__ __forceinline__ void h3_attn_v4_kernel(const h3attn::Params p) {
    using namespace h3v4;
    using h3attn::mma_f16acc;
    using h3attn::pack_h2;
    using h3attn::swz128;
    using namespace h3pp;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* St = smem + SMEM_Q;
    __shared__ __align__(8) uint64_t full[STAGES];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int ntiles = (p.nk + BKV - 1) / BKV;

    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) mbar_init(&full[s], 128);
    }
    for (int c = tid; c < BQ * 8; c += 256) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16, pred);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int lt = tid - 128;
    auto load_kv = [&](int tile, int stage) {
        if ((ABL & 8) && tile >= STAGES) { mbar_cp_async_arrive(&full[stage]); return; }
        const int j0 = tile * BKV;
        uint8_t* ks = St + stage * STAGE;
        uint8_t* vs = ks + SMEM_K;
        float* sks = reinterpret_cast<float*>(vs + SMEM_V);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            bool pred = j < p.nk;
            cp_async_16(smem_u32(ks + row * KROW + ch * 16), p.k + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16, pred);
        }
#pragma unroll
        for (int it = 0; it < 8; ++it) {
            int c = lt + it * 128;
            int row = c >> 4, ch = c & 15;
            cp_async_16(smem_u32(vs + row * VROW + ch * 16), p.v + ((int64_t)(j0 + row) * H + h) * 128 + ch * 8, true);
        }
        if (lt < BKV) {
            int j = j0 + lt;
            bool pred = j < p.nk;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(sks + lt)),
                         "l"(p.sk + (int64_t)(pred ? j : 0) * H + h), "r"(pred ? 4 : 0));
        }
        mbar_cp_async_arrive(&full[stage]);
    };
    if (grp == 1) {
        for (int s = 0; s < STAGES; ++s)
            if (s < ntiles) load_kv(s, s);
    }

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 16;
    const int row_a = q0 + wrow + g, row_b = row_a + 8;
    const bool va = row_a < p.nq, vb = row_b < p.nq;
    const float sq_a = (va ? p.sq[(int64_t)row_a * H + h] : 1.f) * p.scale_log2;
    const float sq_b = (vb ? p.sq[(int64_t)row_b * H + h] : 1.f) * p.scale_log2;

    uint32_t qf[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int row = wrow + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldsm_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
    }
    // per-lane fragment offsets inside a stage (everything else is an immediate)
    const uint32_t st0 = smem_u32(St);
    const uint32_t k_lane = ((lidx >> 1) * 8 + lrow) * KROW + (lidx & 1) * 16;
    const uint32_t v_lane = SMEM_K + ((lidx & 1) * 8 + lrow) * VROW + (lidx >> 1) * 16;
    const uint32_t sk_lane = SMEM_K + SMEM_V + t4 * 8;

    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    uint32_t oh[16][2];
#pragma unroll
    for (int j = 0; j < 16; ++j) oh[j][0] = oh[j][1] = 0u;
    uint32_t pf[4][4];
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    float ga_a = 1.f, ga_b = 1.f;

    auto pv = [&](uint32_t sbase) {
        const uint32_t vb_ = sbase + v_lane;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
            for (int dj = 0; dj < 16; dj += 2) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, vb_ + kk * 16 * VROW + dj * 16);
                mma_f16acc(oh[dj], pf[kk], b0, b1);
                mma_f16acc(oh[dj + 1], pf[kk], b2, b3);
            }
        }
    };
    auto flush = [&](int tile) {
        const float svt = p.sv[(int64_t)(tile * BKV / H3_VGROUP) * H + h];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float2 lo = __half22float2(*reinterpret_cast<__half2*>(&oh[j][0]));
            float2 hi = __half22float2(*reinterpret_cast<__half2*>(&oh[j][1]));
            o_acc[j][0] = fmaf(o_acc[j][0], ga_a, lo.x * svt);
            o_acc[j][1] = fmaf(o_acc[j][1], ga_a, lo.y * svt);
            o_acc[j][2] = fmaf(o_acc[j][2], ga_b, hi.x * svt);
            o_acc[j][3] = fmaf(o_acc[j][3], ga_b, hi.y * svt);
            oh[j][0] = oh[j][1] = 0u;
        }
        ga_a = ga_b = 1.f;
    };

    if (grp == 1) bar_arrive(1, 256);
    int st = 0, par = 0;
    uint32_t prev_base = 0;
    for (int t = 0; t < ntiles; ++t) {
        const uint32_t sbase = st0 + st * STAGE;
        mbar_wait(&full[st], par);
        bar_sync(1 + grp, 256);
        int32_t s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0;
        {
            const uint32_t kb = sbase + k_lane;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, kb + j * 8 * KROW + kk * 32);
                    uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                    mma_s8_16832(s[j], qf[kk], bb0);
                    mma_s8_16832(s[j + 1], qf[kk], bb1);
                }
            }
        }
        if (t > 0 && !(ABL & 64)) pv(prev_base);
        bar_arrive(1 + (grp ^ 1), 256);
        if (t > 0) {
            const int ps = st == 0 ? STAGES - 1 : st - 1;
            if (t - 1 + STAGES < ntiles) {
                if (grp == 0) {
                    bar_arrive(3 + ps, 256);
                } else {
                    bar_sync(3 + ps, 256);
                    load_kv(t - 1 + STAGES, ps);
                }
            }
            if (((t - 1) & (GROUP_TILES - 1)) == GROUP_TILES - 1) flush(t - 1);
        }
        // ---- softmax(t)
        if (ABL & 32) {
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                pf[kk][0] = s[2 * kk][0] & 0x3c003c00u; pf[kk][1] = s[2 * kk][1] & 0x3c003c00u;
                pf[kk][2] = s[2 * kk + 1][2] & 0x3c003c00u; pf[kk][3] = s[2 * kk + 1][3] & 0x3c003c00u;
            }
            prev_base = sbase;
            if (++st == STAGES) { st = 0; par ^= 1; }
            continue;
        }
        const int j0 = t * BKV;
        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const bool full_tile = j0 + BKV <= p.nk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 sk2;
            asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(sk2.x), "=f"(sk2.y) : "r"(sbase + sk_lane + j * 32));
            const float c0 = -12582912.f * sk2.x, c1 = -12582912.f * sk2.y;
            sf[j][0] = fmaf(__int_as_float(s[j][0] + 0x4B400000), sk2.x, c0);
            sf[j][1] = fmaf(__int_as_float(s[j][1] + 0x4B400000), sk2.y, c1);
            sf[j][2] = fmaf(__int_as_float(s[j][2] + 0x4B400000), sk2.x, c0);
            sf[j][3] = fmaf(__int_as_float(s[j][3] + 0x4B400000), sk2.y, c1);
            if (!full_tile) {
                int kl = j * 8 + t4 * 2;
                if (j0 + kl >= p.nk) { sf[j][0] = -INFINITY; sf[j][2] = -INFINITY; }
                if (j0 + kl + 1 >= p.nk) { sf[j][1] = -INFINITY; sf[j][3] = -INFINITY; }
            }
            mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        const float alpha_a = fast_exp2((m_a - mn_a) * sq_a), alpha_b = fast_exp2((m_b - mn_b) * sq_b);
        m_a = mn_a; m_b = mn_b;
        const float nm_a = -mn_a * sq_a, nm_b = -mn_b * sq_b;
        float rs_a = 0.f, rs_b = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p0 = fast_exp2(fmaf(sf[2 * kk][0], sq_a, nm_a)), p1 = fast_exp2(fmaf(sf[2 * kk][1], sq_a, nm_a));
            float p2 = fast_exp2(fmaf(sf[2 * kk][2], sq_b, nm_b)), p3 = fast_exp2(fmaf(sf[2 * kk][3], sq_b, nm_b));
            float p4 = fast_exp2(fmaf(sf[2 * kk + 1][0], sq_a, nm_a)), p5 = fast_exp2(fmaf(sf[2 * kk + 1][1], sq_a, nm_a));
            float p6 = fast_exp2(fmaf(sf[2 * kk + 1][2], sq_b, nm_b)), p7 = fast_exp2(fmaf(sf[2 * kk + 1][3], sq_b, nm_b));
            rs_a += (p0 + p1) + (p4 + p5);
            rs_b += (p2 + p3) + (p6 + p7);
            pf[kk][0] = pack_h2(p0, p1);
            pf[kk][1] = pack_h2(p2, p3);
            pf[kk][2] = pack_h2(p4, p5);
            pf[kk][3] = pack_h2(p6, p7);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a;
        l_b = l_b * alpha_b + rs_b;
        if (__any_sync(0xffffffff, (alpha_a != 1.f) || (alpha_b != 1.f))) {
            const __half2 fa = __float2half2_rn(alpha_a), fb = __float2half2_rn(alpha_b);
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                __half2 x0 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][0]), fa);
                __half2 x1 = __hmul2(*reinterpret_cast<__half2*>(&oh[j][1]), fb);
                oh[j][0] = *reinterpret_cast<uint32_t*>(&x0);
                oh[j][1] = *reinterpret_cast<uint32_t*>(&x1);
            }
            ga_a *= alpha_a;
            ga_b *= alpha_b;
        }
        prev_base = sbase;
        if (++st == STAGES) { st = 0; par ^= 1; }
    }
    bar_sync(1 + grp, 256);
    pv(prev_base);
    bar_arrive(1 + (grp ^ 1), 256);
    flush(ntiles - 1);
    if (grp == 0) bar_sync(1, 256);

    const float inv_a = l_a > 0.f ? 1.f / l_a : 0.f, inv_b = l_b > 0.f ? 1.f / l_b : 0.f;
    const float* mv = p.mv + h * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        float m0 = mv[d], m1 = mv[d + 1];
        if (va) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][0] * inv_a + m0, o_acc[j][1] * inv_a + m1);
        }
        if (vb) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][2] * inv_b + m0, o_acc[j][3] * inv_b + m1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v4(const h3attn::Params p) { h3_attn_v4_kernel<0>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v4_nosm(const h3attn::Params p) { h3_attn_v4_kernel<32>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v4_nosm_nold(const h3attn::Params p) { h3_attn_v4_kernel<40>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v4_nosm_nopv(const h3attn::Params p) { h3_attn_v4_kernel<96>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v4_nold(const h3attn::Params p) { h3_attn_v4_kernel<8>(p); }

// =============================================================================================
// V -> fp8 e4m3, smoothed by the per-channel mean, one scale per (256-key group, head) (amax -> 448),
// transposed per head with the 32-key mma permutation (see k_sage_attn): vt u8 [H][128][s_pad], sv f32 [s_pad/256][H].
// v bf16 rows (token stride ts), n valid rows at global token tok0 (multiple of 256).
// grid (ceil(n/256), H), 256 threads, dyn smem 64*129*4.
// =============================================================================================
extern "C" __global__ void __launch_bounds__(256) k_h3_quant_v8(const bf16* __restrict__ v, int64_t ts, int n, int H, const float* __restrict__ mean,
                                                               uint8_t* __restrict__ vt, int64_t s_pad, int tok0, float* __restrict__ sv) {
    extern __shared__ float tile[];  // [64][129]
    __shared__ float red[32];
    __shared__ float mn[128];
    const int gb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    if (tid < 128) mn[tid] = mean[h * 128 + tid];
    __syncthreads();
    float amax = 0.f;
    {
        const int j = gb * 256 + tid;
        if (j < n) {
            const bf16* src = v + (int64_t)j * ts + h * 128;
#pragma unroll 4
            for (int c = 0; c < 16; ++c) {
                uint4 raw = *reinterpret_cast<const uint4*>(src + c * 8);
                const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    amax = fmaxf(amax, fabsf(__low2float(h2[e]) - mn[c * 8 + 2 * e]));
                    amax = fmaxf(amax, fabsf(__high2float(h2[e]) - mn[c * 8 + 2 * e + 1]));
                }
            }
        }
    }
    amax = block_max(amax, red);
    const float s = amax > 0.f ? amax / 448.f : 1.f;
    const float inv = 1.f / s;
    const int gt = tok0 / 256 + gb;
    if (tid == 0) sv[(int64_t)gt * H + h] = s;
    for (int sub = 0; sub < 4; ++sub) {
        const int k0 = gb * 256 + sub * 64;
        for (int e = tid; e < 64 * 64; e += 256) {
            int key = e >> 6, d2 = (e & 63) * 2;
            int j = k0 + key;
            float a = 0.f, b = 0.f;
            if (j < n) {
                bf162 x2 = *reinterpret_cast<const bf162*>(v + (int64_t)j * ts + h * 128 + d2);
                a = (__low2float(x2) - mn[d2]) * inv;
                b = (__high2float(x2) - mn[d2 + 1]) * inv;
            }
            tile[key * 129 + d2] = a;
            tile[key * 129 + d2 + 1] = b;
        }
        __syncthreads();
        for (int e = tid; e < 128 * 16; e += 256) {
            int d = e >> 4, pos0 = (e & 15) * 4;
            uint32_t packed = 0;
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                int pos = pos0 + u;
                int grp = pos >> 5, pp = pos & 31;
                int half = pp >> 4, tt = (pp & 15) >> 2, i = pp & 3;
                int key = grp * 32 + half * 16 + (i >> 1) * 8 + 2 * tt + (i & 1);
                __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(tile[key * 129 + d], __NV_SATFINITE, __NV_E4M3);
                packed |= ((uint32_t)f) << (8 * u);
            }
            *reinterpret_cast<uint32_t*>(vt + ((int64_t)h * 128 + d) * s_pad + tok0 + k0 - gb * 256 + gb * 256 + pos0) = packed;
        }
        __syncthreads();
    }
}

// =============================================================================================
// v5: ping-pong scheduling, int8 QK, fp8 (e4m3) PV with fp32 accumulators, padded smem rows.
// K tile [64 keys][144 B], V^T tile [128 dims][80 B] (64 permuted fp8 keys + pad).
// o_acc is held in units of the current 256-key group's V scale / 448.
// =============================================================================================
namespace h3v5 {
constexpr int BQ = 128, BKV = 64;
constexpr int KROW = 144, VROW = 80;
constexpr int SMEM_Q = BQ * 128;
constexpr int SMEM_K = BKV * KROW;
constexpr int SMEM_V = 128 * VROW;
constexpr int SMEM_SK = BKV * 4;
constexpr int STAGE = SMEM_K + SMEM_V + SMEM_SK;
constexpr int STAGES = 4;
constexpr int SMEM_TOTAL = SMEM_Q + STAGES * STAGE;
struct Params {
    const int8_t* q; const float* sq;
    const int8_t* k; const float* sk;
    const uint8_t* vt; int64_t s_pad; const float* sv; const float* mv;
    bf16* o; int64_t o_ts;
    int nq, nk, H;
    float scale_log2;
};
DEVI uint32_t pack_e4m3x4(float a, float b, float c, float d) {
    __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
}  // namespace h3v5

template <int ABL = 0, int NG = 2, int NST = 4>
__device__ __forceinline__ void h3_attn_v5_kernel(const h3v5::Params p) {
    using namespace h3v5;
    constexpr int BQ = 64 * NG, NT = 128 * NG, STAGES = NST;
    constexpr int SMEM_Q = BQ * 128;
    using h3attn::swz128;
    using namespace h3pp;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* St = smem + SMEM_Q;
    __shared__ __align__(8) uint64_t full[STAGES];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int ntiles = (p.nk + BKV - 1) / BKV;

    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) mbar_init(&full[s], 128);
    }
    for (int c = tid; c < BQ * 8; c += NT) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16, pred);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int lt = tid - 128 * (NG - 1);
    const uint8_t* vhead = p.vt + (int64_t)h * 128 * p.s_pad;
    auto load_kv = [&](int tile, int stage) {
        if ((ABL & 8) && tile >= STAGES) { mbar_cp_async_arrive(&full[stage]); return; }
        const int j0 = tile * BKV;
        uint8_t* ks = St + stage * STAGE;
        uint8_t* vs = ks + SMEM_K;
        float* sks = reinterpret_cast<float*>(vs + SMEM_V);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            bool pred = j < p.nk;
            cp_async_16(smem_u32(ks + row * KROW + ch * 16), p.k + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16, pred);
        }
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int d = c >> 2, ch = c & 3;
            cp_async_16(smem_u32(vs + d * VROW + ch * 16), vhead + (int64_t)d * p.s_pad + j0 + ch * 16, true);
        }
        if (lt < BKV) {
            int j = j0 + lt;
            bool pred = j < p.nk;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(sks + lt)),
                         "l"(p.sk + (int64_t)(pred ? j : 0) * H + h), "r"(pred ? 4 : 0));
        }
        mbar_cp_async_arrive(&full[stage]);
    };
    if (grp == NG - 1) {
        for (int s = 0; s < STAGES; ++s)
            if (s < ntiles) load_kv(s, s);
    }

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 16;
    const int row_a = q0 + wrow + g, row_b = row_a + 8;
    const bool va = row_a < p.nq, vb = row_b < p.nq;
    const float sq_a = (va ? p.sq[(int64_t)row_a * H + h] : 1.f) * p.scale_log2;
    const float sq_b = (vb ? p.sq[(int64_t)row_b * H + h] : 1.f) * p.scale_log2;

    uint32_t qf[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int row = wrow + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
    }
    const uint32_t st0 = smem_u32(St);
    const uint32_t k_lane = ((lidx >> 1) * 8 + lrow) * KROW + (lidx & 1) * 16;
    const uint32_t v_lane = SMEM_K + ((lidx >> 1) * 8 + lrow) * VROW + (lidx & 1) * 16;
    const uint32_t sk_lane = SMEM_K + SMEM_V + t4 * 8;

    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f;
    uint32_t pf[2][4];
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;
    float sv_cur = 1.f;

    auto pv = [&](uint32_t sbase) {
        const uint32_t vb_ = sbase + v_lane;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                h3v4::ldsm_x4(b0, b1, b2, b3, vb_ + j * 8 * VROW + ks * 32);
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_e4m3_16832(o_acc[j], pf[ks], bb0);
                mma_e4m3_16832(o_acc[j + 1], pf[ks], bb1);
            }
        }
    };

    const int next_bar = 1 + (grp + 1) % NG;
    if (!(ABL & 256) && grp == NG - 1) bar_arrive(1, 256);
    int st = 0, par = 0;
    uint32_t prev_base = 0;
    for (int t = 0; t < ntiles; ++t) {
        const uint32_t sbase = st0 + st * STAGE;
        mbar_wait(&full[st], par);
        if (!(ABL & 256)) bar_sync(1 + grp, 256);
        int32_t s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0;
        {
            const uint32_t kb = sbase + k_lane;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    h3v4::ldsm_x4(b0, b1, b2, b3, kb + j * 8 * KROW + kk * 32);
                    uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                    mma_s8_16832(s[j], qf[kk], bb0);
                    mma_s8_16832(s[j + 1], qf[kk], bb1);
                }
            }
        }
        if (ABL & 128) {
            if (!(ABL & 256)) bar_arrive(next_bar, 256);
            if (t > 0) pv(prev_base);
        } else {
            if (t > 0) pv(prev_base);
            if (!(ABL & 256)) bar_arrive(next_bar, 256);
        }
        if (t > 0) {
            const int ps = st == 0 ? STAGES - 1 : st - 1;
            if (t - 1 + STAGES < ntiles) {
                if (grp != NG - 1) {
                    bar_arrive(1 + NG + ps, NT);
                } else {
                    bar_sync(1 + NG + ps, NT);
                    load_kv(t - 1 + STAGES, ps);
                }
            }
        }
        // ---- softmax(t)
        if (ABL & 32) {
            pf[0][0] = s[0][0] & 0x38383838u; pf[0][1] = s[1][1] & 0x38383838u; pf[0][2] = s[2][2] & 0x38383838u; pf[0][3] = s[3][3] & 0x38383838u;
            pf[1][0] = s[4][0] & 0x38383838u; pf[1][1] = s[5][1] & 0x38383838u; pf[1][2] = s[6][2] & 0x38383838u; pf[1][3] = s[7][3] & 0x38383838u;
            prev_base = sbase;
            if (++st == STAGES) { st = 0; par ^= 1; }
            continue;
        }
        const int j0 = t * BKV;
        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const bool full_tile = j0 + BKV <= p.nk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float2 sk2;
            asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(sk2.x), "=f"(sk2.y) : "r"(sbase + sk_lane + j * 32));
            const float c0 = -12582912.f * sk2.x, c1 = -12582912.f * sk2.y;
            sf[j][0] = fmaf(__int_as_float(s[j][0] + 0x4B400000), sk2.x, c0);
            sf[j][1] = fmaf(__int_as_float(s[j][1] + 0x4B400000), sk2.y, c1);
            sf[j][2] = fmaf(__int_as_float(s[j][2] + 0x4B400000), sk2.x, c0);
            sf[j][3] = fmaf(__int_as_float(s[j][3] + 0x4B400000), sk2.y, c1);
            if (!full_tile) {
                int kl = j * 8 + t4 * 2;
                if (j0 + kl >= p.nk) { sf[j][0] = -INFINITY; sf[j][2] = -INFINITY; }
                if (j0 + kl + 1 >= p.nk) { sf[j][1] = -INFINITY; sf[j][3] = -INFINITY; }
            }
            mx_a = fmaxf(mx_a, fmaxf(sf[j][0], sf[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(sf[j][2], sf[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        const float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        const float alpha_a = fast_exp2((m_a - mn_a) * sq_a), alpha_b = fast_exp2((m_b - mn_b) * sq_b);
        m_a = mn_a; m_b = mn_b;
        const float nm_a = -mn_a * sq_a, nm_b = -mn_b * sq_b;
        float rs_a = 0.f, rs_b = 0.f;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            float pa[8], pb[8];
#pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                int j = ks * 4 + jj;
                pa[jj * 2] = fast_exp2(fmaf(sf[j][0], sq_a, nm_a));
                pa[jj * 2 + 1] = fast_exp2(fmaf(sf[j][1], sq_a, nm_a));
                pb[jj * 2] = fast_exp2(fmaf(sf[j][2], sq_b, nm_b));
                pb[jj * 2 + 1] = fast_exp2(fmaf(sf[j][3], sq_b, nm_b));
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) { rs_a += pa[e]; rs_b += pb[e]; }
            pf[ks][0] = pack_e4m3x4(pa[0] * 448.f, pa[1] * 448.f, pa[2] * 448.f, pa[3] * 448.f);
            pf[ks][1] = pack_e4m3x4(pb[0] * 448.f, pb[1] * 448.f, pb[2] * 448.f, pb[3] * 448.f);
            pf[ks][2] = pack_e4m3x4(pa[4] * 448.f, pa[5] * 448.f, pa[6] * 448.f, pa[7] * 448.f);
            pf[ks][3] = pack_e4m3x4(pb[4] * 448.f, pb[5] * 448.f, pb[6] * 448.f, pb[7] * 448.f);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a;
        l_b = l_b * alpha_b + rs_b;
        // o_acc rescale: softmax alpha, and the V-scale change at 256-key group boundaries
        float ra = alpha_a, rb = alpha_b;
        if ((j0 & 255) == 0) {
            const float svn = p.sv[(int64_t)(j0 >> 8) * H + h];
            ra *= sv_cur / svn;
            rb *= sv_cur / svn;
            sv_cur = svn;
        }
        if (__any_sync(0xffffffff, (ra != 1.f) || (rb != 1.f))) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o_acc[j][0] *= ra; o_acc[j][1] *= ra;
                o_acc[j][2] *= rb; o_acc[j][3] *= rb;
            }
        }
        prev_base = sbase;
        if (++st == STAGES) { st = 0; par ^= 1; }
    }
    if (!(ABL & 256)) bar_sync(1 + grp, 256);
    pv(prev_base);
    if (!(ABL & 256)) {
        bar_arrive(next_bar, 256);
        if (grp == 0) bar_sync(1, 256);
    }

    const float inv_a = l_a > 0.f ? sv_cur / (448.f * l_a) : 0.f, inv_b = l_b > 0.f ? sv_cur / (448.f * l_b) : 0.f;
    const float* mv = p.mv + h * 128;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        float m0 = mv[d], m1 = mv[d + 1];
        if (va) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][0] * inv_a + m0, o_acc[j][1] * inv_a + m1);
        }
        if (vb) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + h * 128 + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][2] * inv_b + m0, o_acc[j][3] * inv_b + m1);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5(const h3v5::Params p) { h3_attn_v5_kernel<0>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5_nold(const h3v5::Params p) { h3_attn_v5_kernel<8>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) k_h3_attn_v5g3(const h3v5::Params p) { h3_attn_v5_kernel<0, 3, 3>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5_nosm(const h3v5::Params p) { h3_attn_v5_kernel<32>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5_nosm_nold(const h3v5::Params p) { h3_attn_v5_kernel<40>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) k_h3_attn_v5g3_a(const h3v5::Params p) { h3_attn_v5_kernel<128, 3, 3>(p); }
extern "C" __global__ void __launch_bounds__(384, 1) k_h3_attn_v5g3_b(const h3v5::Params p) { h3_attn_v5_kernel<256, 3, 3>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5_a(const h3v5::Params p) { h3_attn_v5_kernel<128>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn_v5_b(const h3v5::Params p) { h3_attn_v5_kernel<256>(p); }

// =============================================================================================
// v6: like v5 (int8 QK, fp8 PV, ping-pong of two 4-warp groups, padded smem), but every warp owns 32 query rows
// (two m16 tiles): K/V fragments are reused by both m-tiles (half the ldmatrix and L2->SM traffic per FLOP).
// BQ = 256 queries per block; Q fragments are re-read from shared memory every tile to stay within 255 registers.
// =============================================================================================
template <int ABL = 0>
__device__ __forceinline__ void h3_attn_v6_kernel(const h3v5::Params p) {
    using namespace h3v5;
    using h3attn::swz128;
    using namespace h3pp;
    constexpr int NG = 2, BQ = 256, NT = 256, STAGES = 3;
    constexpr int SMEM_Q = BQ * 128;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* St = smem + SMEM_Q;
    __shared__ __align__(8) uint64_t full[STAGES];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int grp = warp >> 2;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int ntiles = (p.nk + BKV - 1) / BKV;

    if (tid == 0) {
        for (int s = 0; s < STAGES; ++s) mbar_init(&full[s], 128);
    }
    for (int c = tid; c < BQ * 8; c += NT) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16, pred);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int lt = tid - 128;
    const uint8_t* vhead = p.vt + (int64_t)h * 128 * p.s_pad;
    auto load_kv = [&](int tile, int stage) {
        const int j0 = tile * BKV;
        uint8_t* ks = St + stage * STAGE;
        uint8_t* vs = ks + SMEM_K;
        float* sks = reinterpret_cast<float*>(vs + SMEM_V);
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            bool pred = j < p.nk;
            cp_async_16(smem_u32(ks + row * KROW + ch * 16), p.k + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16, pred);
        }
#pragma unroll
        for (int it = 0; it < 4; ++it) {
            int c = lt + it * 128;
            int d = c >> 2, ch = c & 3;
            cp_async_16(smem_u32(vs + d * VROW + ch * 16), vhead + (int64_t)d * p.s_pad + j0 + ch * 16, true);
        }
        if (lt < BKV) {
            int j = j0 + lt;
            bool pred = j < p.nk;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(smem_u32(sks + lt)),
                         "l"(p.sk + (int64_t)(pred ? j : 0) * H + h), "r"(pred ? 4 : 0));
        }
        mbar_cp_async_arrive(&full[stage]);
    };
    if (grp == NG - 1) {
        for (int s = 0; s < STAGES; ++s)
            if (s < ntiles) load_kv(s, s);
    }

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    const int wrow = warp * 32;
    // rows owned by this lane: r[mt][0] = wrow + mt*16 + g, r[mt][1] = +8
    float sqr[2][2];
    bool vr[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            int r = q0 + wrow + mt * 16 + hh * 8 + g;
            vr[mt][hh] = r < p.nq;
            sqr[mt][hh] = (vr[mt][hh] ? p.sq[(int64_t)r * H + h] : 1.f) * p.scale_log2;
        }
    const uint32_t st0 = smem_u32(St);
    const uint32_t k_lane = ((lidx >> 1) * 8 + lrow) * KROW + (lidx & 1) * 16;
    const uint32_t v_lane = SMEM_K + ((lidx >> 1) * 8 + lrow) * VROW + (lidx & 1) * 16;
    const uint32_t sk_lane = SMEM_K + SMEM_V + t4 * 8;
    // Q fragment addresses (xor swizzle) for the two m-tiles and 4 k-steps: row = wrow + mt*16 + (lidx&1)*8 + lrow
    const uint32_t q_base = smem_u32(Qs) + (wrow + (lidx & 1) * 8 + lrow) * 128;
    const int q_x = lrow;  // (row & 7)

    float o_acc[2][16][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int j = 0; j < 16; ++j) o_acc[mt][j][0] = o_acc[mt][j][1] = o_acc[mt][j][2] = o_acc[mt][j][3] = 0.f;
    uint32_t pf[2][2][4];
    float m[2][2], l[2][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) { m[mt][0] = m[mt][1] = -INFINITY; l[mt][0] = l[mt][1] = 0.f; }
    float sv_cur = 1.f;

    auto pv = [&](uint32_t sbase) {
        const uint32_t vb_ = sbase + v_lane;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                h3v4::ldsm_x4(b0, b1, b2, b3, vb_ + j * 8 * VROW + ks * 32);
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_e4m3_16832(o_acc[0][j], pf[0][ks], bb0);
                mma_e4m3_16832(o_acc[0][j + 1], pf[0][ks], bb1);
                mma_e4m3_16832(o_acc[1][j], pf[1][ks], bb0);
                mma_e4m3_16832(o_acc[1][j + 1], pf[1][ks], bb1);
            }
        }
    };

    const int next_bar = 1 + (grp + 1) % NG;
    if (grp == NG - 1) bar_arrive(1, 256);
    int st = 0, par = 0;
    uint32_t prev_base = 0;
    for (int t = 0; t < ntiles; ++t) {
        const uint32_t sbase = st0 + st * STAGE;
        mbar_wait(&full[st], par);
        bar_sync(1 + grp, 256);
        int32_t s[2][8][4];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int j = 0; j < 8; ++j) s[mt][j][0] = s[mt][j][1] = s[mt][j][2] = s[mt][j][3] = 0;
        {
            const uint32_t kb = sbase + k_lane;
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                uint32_t qa[4], qb[4];
                const uint32_t qoff = (uint32_t)(((2 * kk + (lidx >> 1)) ^ q_x) << 4);
                h3v4::ldsm_x4(qa[0], qa[1], qa[2], qa[3], q_base + qoff);
                h3v4::ldsm_x4(qb[0], qb[1], qb[2], qb[3], q_base + 16 * 128 + qoff);
#pragma unroll
                for (int j = 0; j < 8; j += 2) {
                    uint32_t b0, b1, b2, b3;
                    h3v4::ldsm_x4(b0, b1, b2, b3, kb + j * 8 * KROW + kk * 32);
                    uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                    mma_s8_16832(s[0][j], qa, bb0);
                    mma_s8_16832(s[0][j + 1], qa, bb1);
                    mma_s8_16832(s[1][j], qb, bb0);
                    mma_s8_16832(s[1][j + 1], qb, bb1);
                }
            }
        }
        if (t > 0) pv(prev_base);
        bar_arrive(next_bar, 256);
        if (t > 0) {
            const int ps = st == 0 ? STAGES - 1 : st - 1;
            if (t - 1 + STAGES < ntiles) {
                if (grp != NG - 1) {
                    bar_arrive(1 + NG + ps, NT);
                } else {
                    bar_sync(1 + NG + ps, NT);
                    load_kv(t - 1 + STAGES, ps);
                }
            }
        }
        // ---- softmax(t) for both m-tiles
        const int j0 = t * BKV;
        const bool full_tile = j0 + BKV <= p.nk;
        float sk_x[8], sk_y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            asm volatile("ld.shared.v2.f32 {%0,%1}, [%2];" : "=f"(sk_x[j]), "=f"(sk_y[j]) : "r"(sbase + sk_lane + j * 32));
        }
        float alpha[2][2];
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            float sf[8][4];
            float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float c0 = -12582912.f * sk_x[j], c1 = -12582912.f * sk_y[j];
                sf[j][0] = fmaf(__int_as_float(s[mt][j][0] + 0x4B400000), sk_x[j], c0);
                sf[j][1] = fmaf(__int_as_float(s[mt][j][1] + 0x4B400000), sk_y[j], c1);
                sf[j][2] = fmaf(__int_as_float(s[mt][j][2] + 0x4B400000), sk_x[j], c0);
                sf[j][3] = fmaf(__int_as_float(s[mt][j][3] + 0x4B400000), sk_y[j], c1);
                if (!full_tile) {
                    int kl = j * 8 + t4 * 2;
                    if (j0 + kl >= p.nk) { sf[j][0] = -INFINITY; sf[j][2] = -INFINITY; }
                    if (j0 + kl + 1 >= p.nk) { sf[j][1] = -INFINITY; sf[j][3] = -INFINITY; }
                }
                mx0 = fmaxf(mx0, fmaxf(sf[j][0], sf[j][1]));
                mx1 = fmaxf(mx1, fmaxf(sf[j][2], sf[j][3]));
            }
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
            const float mn0 = fmaxf(m[mt][0], mx0), mn1 = fmaxf(m[mt][1], mx1);
            alpha[mt][0] = fast_exp2((m[mt][0] - mn0) * sqr[mt][0]);
            alpha[mt][1] = fast_exp2((m[mt][1] - mn1) * sqr[mt][1]);
            m[mt][0] = mn0; m[mt][1] = mn1;
            const float nm0 = -mn0 * sqr[mt][0], nm1 = -mn1 * sqr[mt][1];
            float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                float pa[8], pb[8];
#pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    int j = ks * 4 + jj;
                    pa[jj * 2] = fast_exp2(fmaf(sf[j][0], sqr[mt][0], nm0));
                    pa[jj * 2 + 1] = fast_exp2(fmaf(sf[j][1], sqr[mt][0], nm0));
                    pb[jj * 2] = fast_exp2(fmaf(sf[j][2], sqr[mt][1], nm1));
                    pb[jj * 2 + 1] = fast_exp2(fmaf(sf[j][3], sqr[mt][1], nm1));
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) { rs0 += pa[e]; rs1 += pb[e]; }
                pf[mt][ks][0] = pack_e4m3x4(pa[0] * 448.f, pa[1] * 448.f, pa[2] * 448.f, pa[3] * 448.f);
                pf[mt][ks][1] = pack_e4m3x4(pb[0] * 448.f, pb[1] * 448.f, pb[2] * 448.f, pb[3] * 448.f);
                pf[mt][ks][2] = pack_e4m3x4(pa[4] * 448.f, pa[5] * 448.f, pa[6] * 448.f, pa[7] * 448.f);
                pf[mt][ks][3] = pack_e4m3x4(pb[4] * 448.f, pb[5] * 448.f, pb[6] * 448.f, pb[7] * 448.f);
            }
            rs0 += __shfl_xor_sync(0xffffffff, rs0, 1);
            rs0 += __shfl_xor_sync(0xffffffff, rs0, 2);
            rs1 += __shfl_xor_sync(0xffffffff, rs1, 1);
            rs1 += __shfl_xor_sync(0xffffffff, rs1, 2);
            l[mt][0] = l[mt][0] * alpha[mt][0] + rs0;
            l[mt][1] = l[mt][1] * alpha[mt][1] + rs1;
        }
        float vsr = 1.f;
        if ((j0 & 255) == 0) {
            const float svn = p.sv[(int64_t)(j0 >> 8) * H + h];
            vsr = sv_cur / svn;
            sv_cur = svn;
        }
        const float r00 = alpha[0][0] * vsr, r01 = alpha[0][1] * vsr, r10 = alpha[1][0] * vsr, r11 = alpha[1][1] * vsr;
        if (__any_sync(0xffffffff, (r00 != 1.f) || (r01 != 1.f) || (r10 != 1.f) || (r11 != 1.f))) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                o_acc[0][j][0] *= r00; o_acc[0][j][1] *= r00; o_acc[0][j][2] *= r01; o_acc[0][j][3] *= r01;
                o_acc[1][j][0] *= r10; o_acc[1][j][1] *= r10; o_acc[1][j][2] *= r11; o_acc[1][j][3] *= r11;
            }
        }
        prev_base = sbase;
        if (++st == STAGES) { st = 0; par ^= 1; }
    }
    bar_sync(1 + grp, 256);
    pv(prev_base);
    bar_arrive(next_bar, 256);
    if (grp == 0) bar_sync(1, 256);

    const float* mv = p.mv + h * 128;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const float inv0 = l[mt][0] > 0.f ? sv_cur / (448.f * l[mt][0]) : 0.f;
        const float inv1 = l[mt][1] > 0.f ? sv_cur / (448.f * l[mt][1]) : 0.f;
        const int ra = q0 + wrow + mt * 16 + g, rb = ra + 8;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            int d = j * 8 + t4 * 2;
            float m0 = mv[d], m1 = mv[d + 1];
            if (vr[mt][0]) {
                bf16* dst = p.o + (int64_t)ra * p.o_ts + h * 128 + d;
                *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[mt][j][0] * inv0 + m0, o_acc[mt][j][1] * inv0 + m1);
            }
            if (vr[mt][1]) {
                bf16* dst = p.o + (int64_t)rb * p.o_ts + h * 128 + d;
                *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[mt][j][2] * inv1 + m0, o_acc[mt][j][3] * inv1 + m1);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 1) k_h3_attn_v6(const h3v5::Params p) { h3_attn_v6_kernel<0>(p); }

// =============================================================================================
// Fused per-head RMSNorm + partial split-half rope + H128 rotation (+ optional per-channel mean subtraction) +
// per-(token, head) int8 quantization. x: bf16 rows (token stride ts) holding H heads of 128 at column 0.
// out q8 [M][H][128], scale [M][H]. One warp per (token, head).
// =============================================================================================
extern "C" __global__ void k_h3_head_nrq(const bf16* __restrict__ x, int64_t ts, int M, int H, const bf16* __restrict__ w, float eps,
                                        const bf16* __restrict__ rope, const float* __restrict__ mean,
                                        int8_t* __restrict__ q8, float* __restrict__ scale) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)M * H) return;
    int64_t m = gw / H;
    int h = (int)(gw % H);
    const bf16* p = x + m * ts + h * 128 + lane * 4;
    uint2 raw = *reinterpret_cast<const uint2*>(p);
    const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
    float v[4] = {__low2float(h2[0]), __high2float(h2[0]), __low2float(h2[1]), __high2float(h2[1])};
    float ss = warp_sum(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
    float r = rsqrtf(ss / 128.f + eps);
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = round_bf16(v[e] * r * bf2f(w[lane * 4 + e]));
    {
        int src = lane < 12 ? lane + 12 : (lane < 24 ? lane - 12 : lane);
        float o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = __shfl_sync(0xffffffff, v[e], src);
        if (lane < 24) {
            const bf16* rp = rope + (m * 48 + (lane % 12) * 4) * 2;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float c = bf2f(rp[2 * e]), s = bf2f(rp[2 * e + 1]);
                v[e] = round_bf16(lane < 12 ? v[e] * c - o[e] * s : v[e] * c + o[e] * s);
            }
        }
    }
    {
        float a0 = v[0] + v[1], a1 = v[0] - v[1], a2 = v[2] + v[3], a3 = v[2] - v[3];
        v[0] = a0 + a2; v[2] = a0 - a2; v[1] = a1 + a3; v[3] = a1 - a3;
#pragma unroll
        for (int b = 1; b < 32; b <<= 1) {
            const bool up = lane & b;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float o = __shfl_xor_sync(0xffffffff, v[e], b);
                v[e] = up ? (o - v[e]) : (v[e] + o);
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] *= 0.08838834764831845f;
    }
    if (mean) {
        const float* mn = mean + h * 128 + lane * 4;
#pragma unroll
        for (int e = 0; e < 4; ++e) v[e] -= mn[e];
    }
    float amax = warp_max(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))));
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    char4 qv;
    qv.x = (int8_t)max(-128, min(127, __float2int_rn(v[0] * inv)));
    qv.y = (int8_t)max(-128, min(127, __float2int_rn(v[1] * inv)));
    qv.z = (int8_t)max(-128, min(127, __float2int_rn(v[2] * inv)));
    qv.w = (int8_t)max(-128, min(127, __float2int_rn(v[3] * inv)));
    *reinterpret_cast<char4*>(q8 + gw * 128 + lane * 4) = qv;
    if (lane == 0) scale[gw] = s;
}
