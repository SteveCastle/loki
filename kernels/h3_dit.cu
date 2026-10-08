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
extern "C" __global__ void k_h3_mma_bench(int mode, int iters, float* out) {
    uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, 7u, 9u}, b[2] = {threadIdx.x * 5u, 11u};
    float fc[8][4] = {};
    int32_t ic[8][4] = {};
    uint32_t hc[8][2] = {};
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (mode == 0) mma_s8_16832(ic[j], a, b);
            else if (mode == 1) mma_e4m3_16832(fc[j], a, b);
            else if (mode == 2) mma_f16_16816_f16acc(hc[j], a, b);
            else if (mode == 3) mma_bf16_16816(fc[j], a, b);
            else mma_f16_16816_f32acc(fc[j], a, b);
        }
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += fc[j][0] + (float)ic[j][0] + (float)hc[j][0];
    if (s == 12345.f) out[0] = s;
}

// =============================================================================================
// V -> fp16 (smoothed by the per-channel mean, scaled per (64-key tile, head) to |v| <= 256 so the fp16
// accumulators of the PV mma cannot overflow). v bf16 rows (token stride ts), n valid rows at global token tok0
// (multiple of 64). out: v16 [s_pad][H][128] fp16, sv f32 [s_pad/64][H]. grid (ceil(n/64), H), 256 threads.
// =============================================================================================
extern "C" __global__ void __launch_bounds__(256) k_h3_quant_v16(const bf16* __restrict__ v, int64_t ts, int n, int H, const float* __restrict__ mean,
                                                                __half* __restrict__ v16, int tok0, float* __restrict__ sv) {
    __shared__ float red[32];
    const int tb = blockIdx.x, h = blockIdx.y;
    const int tid = threadIdx.x;
    const int key = tid >> 2, q = tid & 3;  // thread: one key, 32 of its 128 dims
    const int j = tb * 64 + key;
    float vals[32];
    float amax = 0.f;
    if (j < n) {
        const bf16* src = v + (int64_t)j * ts + h * 128 + q * 32;
        const float* mn = mean + h * 128 + q * 32;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint4 raw = *reinterpret_cast<const uint4*>(src + c * 8);
            const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                vals[c * 8 + 2 * e] = __low2float(h2[e]) - mn[c * 8 + 2 * e];
                vals[c * 8 + 2 * e + 1] = __high2float(h2[e]) - mn[c * 8 + 2 * e + 1];
            }
        }
#pragma unroll
        for (int e = 0; e < 32; ++e) amax = fmaxf(amax, fabsf(vals[e]));
    } else {
#pragma unroll
        for (int e = 0; e < 32; ++e) vals[e] = 0.f;
    }
    amax = block_max(amax, red);
    const float s = amax > 0.f ? amax / 256.f : 1.f;
    const float inv = 1.f / s;
    const int gt = tok0 / 64 + tb;
    if (tid == 0) sv[(int64_t)gt * H + h] = s;
    __half* dst = v16 + ((int64_t)(tok0 + tb * 64 + key) * H + h) * 128 + q * 32;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        uint4 o;
        uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            __half2 hh = __floats2half2_rn(vals[c * 8 + 2 * e] * inv, vals[c * 8 + 2 * e + 1] * inv);
            ow[e] = *reinterpret_cast<uint32_t*>(&hh);
        }
        *reinterpret_cast<uint4*>(dst + c * 8) = o;
    }
}

// =============================================================================================
// Bidirectional attention, one segment, head dim 128:
//   S = Q8 K8^T (int8 mma, per-(token, head) scales; K smoothed by a per-channel mean, exact for softmax)
//   O = P V (fp16 mma with fp16 accumulators per 64-key tile, flushed into fp32 with the tile's V scale)
// q8 [nq][H][128], sq [nq][H]; k8 [nk][H][128], sk [nk][H]; v16 [nk_pad][H][128] (zero padded), sv [nk_pad/64][H];
// mean_v [H*128]; out bf16 rows (token stride o_ts) at column h*128.
// Block: 128 queries (8 warps x 16 rows), 64-key tiles, STAGES-deep cp.async pipeline.
// =============================================================================================
namespace h3attn {
constexpr int BQ = 128, BKV = 64, THREADS = 256;
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
}  // namespace h3attn

template <int STAGES>
__device__ void h3_attn_kernel(const h3attn::Params p) {
    using namespace h3attn;
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
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < ntiles) load_kv(s, s);
        cp_async_commit();
    }

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
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

    for (int t = 0; t < ntiles; ++t) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (t == 0) {
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                int row = warp * 16 + (lidx & 1) * 8 + lrow;
                int ch = 2 * s + (lidx >> 1);
                ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
            }
        }
        {
            int nt = t + STAGES - 1;
            if (nt < ntiles) load_kv(nt, nt % STAGES);
            cp_async_commit();
        }
        const uint8_t* ks = St + (t % STAGES) * STAGE;
        const uint8_t* vs = ks + SMEM_K;
        const float* sks = reinterpret_cast<const float*>(vs + SMEM_V);
        const int j0 = t * BKV;

        int32_t s[8][4];
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
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_s8_16832(s[j], qf[kk], bb0);
                mma_s8_16832(s[j + 1], qf[kk], bb1);
            }
        }
        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
        const bool full = j0 + BKV <= p.nk;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int kl = j * 8 + t4 * 2;
            float sk0 = sks[kl], sk1 = sks[kl + 1];
            sf[j][0] = (float)s[j][0] * (sq_a * sk0);
            sf[j][1] = (float)s[j][1] * (sq_a * sk1);
            sf[j][2] = (float)s[j][2] * (sq_b * sk0);
            sf[j][3] = (float)s[j][3] * (sq_b * sk1);
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
            float p0 = fast_exp2(sf[2 * kk][0] - mn_a), p1 = fast_exp2(sf[2 * kk][1] - mn_a);
            float p2 = fast_exp2(sf[2 * kk][2] - mn_b), p3 = fast_exp2(sf[2 * kk][3] - mn_b);
            float p4 = fast_exp2(sf[2 * kk + 1][0] - mn_a), p5 = fast_exp2(sf[2 * kk + 1][1] - mn_a);
            float p6 = fast_exp2(sf[2 * kk + 1][2] - mn_b), p7 = fast_exp2(sf[2 * kk + 1][3] - mn_b);
            rs_a += p0 + p1 + p4 + p5;
            rs_b += p2 + p3 + p6 + p7;
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

        // PV with fp16 accumulators for this tile
        uint32_t oh[16][2];
#pragma unroll
        for (int j = 0; j < 16; ++j) oh[j][0] = oh[j][1] = 0u;
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
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
        const float svt = p.sv[(int64_t)(j0 / BKV) * H + h];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            float2 lo = __half22float2(*reinterpret_cast<__half2*>(&oh[j][0]));
            float2 hi = __half22float2(*reinterpret_cast<__half2*>(&oh[j][1]));
            o_acc[j][0] = fmaf(o_acc[j][0], alpha_a, lo.x * svt);
            o_acc[j][1] = fmaf(o_acc[j][1], alpha_a, lo.y * svt);
            o_acc[j][2] = fmaf(o_acc[j][2], alpha_b, hi.x * svt);
            o_acc[j][3] = fmaf(o_acc[j][3], alpha_b, hi.y * svt);
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

extern "C" __global__ void __launch_bounds__(256) k_h3_attn2(const h3attn::Params p) { h3_attn_kernel<2>(p); }
extern "C" __global__ void __launch_bounds__(256) k_h3_attn3(const h3attn::Params p) { h3_attn_kernel<3>(p); }
