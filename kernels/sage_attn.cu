// Low-precision flash attention for the DiT (head dim 128):
//   S = Q8 K8^T on int8 tensor cores (per-token scales; K smoothed by a per-channel mean whose
//       effect on the scores is restored through a per-query correction term),
//   O = P8 V8 on fp8 (e4m3) tensor cores (V smoothed by a per-channel mean, per-64-key-tile scale).
//
// Layouts (per head h, head dim 128):
//   Q8 : int8 [nq][H][128], sq f32 [nq][H], corr f32 [2][nq][H] (= Q . mean_k(segment))
//   K8 : int8 [nk][H][128], sk f32 [nk][H]            per segment
//   V8 : e4m3 [H][128][nk_pad] "Vt": keys permuted inside each 32-key group so that the fp8 mma B
//        fragment (k = 4t..4t+3 and 16+4t..) is a contiguous 4-byte load:
//          position p (0..31) -> key: half = p/16, t = (p%16)/4, i = p%4
//                                key = half*16 + (i/2)*8 + 2t + (i%2)
//        sv f32 [nk_pad/64][H] per tile scale, mean_v f32 [H][128] per segment.
//   Two segments: segment 1 has len1 valid keys padded to len1_pad (multiple of 64); keys in
//   [len1, len1_pad) are masked. Segment 2 follows at global key index len1_pad.
// Masking: key j visible to query i iff j < limit(i) (kv_limit[i], or total when null).
// Block: 128 queries (8 warps x 16), tile 64 keys, double-buffered cp.async.
#include "common.cuh"

namespace {

constexpr int BQ = 128, THREADS = 256;
constexpr int QROWB = 128;                  // 128 int8 per query row
constexpr int KROWB = 128;                  // 128 int8 per key row
constexpr int SMEM_Q = BQ * QROWB;          // 16 KB
template <int BKV>
struct SageGeo {
    static constexpr int VROWB = BKV + 16;          // BKV fp8 keys + 16 pad per dim row
    static constexpr int SMEM_K = BKV * KROWB;
    static constexpr int SMEM_V = 128 * VROWB;
    static constexpr int SMEM_SK = BKV * 4;
    static constexpr int TOTAL = SMEM_Q + 2 * (SMEM_K + SMEM_V + SMEM_SK);
    static constexpr int NT = BKV / 8;              // key n-tiles for QK^T
    static constexpr int KS = BKV / 32;             // key k-steps for PV
};

DEVI int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }

struct SageParams {
    const int8_t* q; const float* sq; const float* corr1; const float* corr2;
    const int8_t* k1; const float* sk1; const uint8_t* v1; const float* sv1; const float* mv1; int len1; int len1_pad;
    const int8_t* k2; const float* sk2; const uint8_t* v2; const float* sv2; const float* mv2; int len2;
    bf16* o; int64_t o_ts; int o_hs;
    int nq; int H;
    const int* kv_limit;
    float scale_log2;
};

DEVI uint32_t pack_e4m3x4(float a, float b, float c, float d) {
    __nv_fp8x2_storage_t lo = __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
    __nv_fp8x2_storage_t hi = __nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3);
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

}  // namespace

template <int BKV>
__device__ void sage_attn_kernel(const SageParams p) {
    using G = SageGeo<BKV>;
    constexpr int VROWB = G::VROWB, SMEM_K = G::SMEM_K, SMEM_V = G::SMEM_V;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* Ks = Qs + SMEM_Q;
    uint8_t* Vs = Ks + 2 * SMEM_K;
    float* SKs = reinterpret_cast<float*>(Vs + 2 * SMEM_V);
    __shared__ float red[32];

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ;
    const int h = blockIdx.y;
    const int H = p.H;
    const int total = p.len1_pad + p.len2;

    // ---- block key limit
    int my_limit = 0;
    for (int r = tid; r < BQ; r += THREADS) {
        int i = q0 + r;
        int lim = 0;
        if (i < p.nq) lim = p.kv_limit ? min(p.kv_limit[i], total) : total;
        my_limit = max(my_limit, lim);
    }
    const int block_limit = (int)block_max((float)my_limit, red);
    const int ntiles = (block_limit + BKV - 1) / BKV;
    if (ntiles == 0) return;

    // ---- Q tile: 128 rows x 128 B = 1024 chunks, 4 per thread
    for (int c = tid; c < BQ * 8; c += THREADS) {
        int row = c >> 3, ch = c & 7;
        int i = q0 + row;
        bool pred = i < p.nq;
        const int8_t* src = p.q + ((int64_t)(pred ? i : 0) * H + h) * 128 + ch * 16;
        cp_async_16(smem_u32(Qs + swz128(row, ch)), src, pred);
    }
    auto load_kv = [&](int tile, int stage) {
        const int j0 = tile * BKV;
        uint8_t* ks = Ks + stage * SMEM_K;
        uint8_t* vs = Vs + stage * SMEM_V;
        float* sks = SKs + stage * BKV;
        const bool seg1 = j0 < p.len1_pad;
        // K rows: 64 x 8 chunks = 512 chunks, 2 per thread
        for (int c = tid; c < BKV * 8; c += THREADS) {
            int row = c >> 3, ch = c & 7;
            int j = j0 + row;
            const int8_t* src;
            bool pred;
            if (seg1) { pred = j < p.len1; src = p.k1 + ((int64_t)(pred ? j : 0) * H + h) * 128 + ch * 16; }
            else { int jj = j - p.len1_pad; pred = jj < p.len2; src = p.k2 + ((int64_t)(pred ? jj : 0) * H + h) * 128 + ch * 16; }
            cp_async_16(smem_u32(ks + swz128(row, ch)), src, pred);
        }
        // V^T rows: 128 dims x (BKV/16) chunks
        const uint8_t* vbase; int vlen; int jj0;
        if (seg1) { vbase = p.v1; vlen = p.len1_pad; jj0 = j0; }
        else { vbase = p.v2; vlen = (p.len2 + BKV - 1) / BKV * BKV; jj0 = j0 - p.len1_pad; }
        constexpr int VCH = BKV / 16;
        for (int c = tid; c < 128 * VCH; c += THREADS) {
            int d = c / VCH, ch = c % VCH;
            const uint8_t* src = vbase + ((int64_t)h * 128 + d) * vlen + jj0 + ch * 16;
            cp_async_16(smem_u32(vs + d * VROWB + ch * 16), src, true);
        }
        // key scales
        if (tid < BKV) {
            int j = j0 + tid;
            float s = 0.f;
            if (seg1) { if (j < p.len1) s = p.sk1[(int64_t)j * H + h]; }
            else { int jj = j - p.len1_pad; if (jj < p.len2) s = p.sk2[(int64_t)jj * H + h]; }
            sks[tid] = s;
        }
    };
    load_kv(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    const int lidx = lane >> 3, lrow = lane & 7;
    const int g = lane >> 2, t4 = lane & 3;
    // Q fragments: 4 k-steps of 32 dims
    uint32_t qf[4][4];
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        int row = warp * 16 + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + swz128(row, ch)));
    }
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    const bool va = row_a < p.nq, vb = row_b < p.nq;
    const int ia = va ? row_a : 0, ib = vb ? row_b : 0;
    const float sq_a = p.sq[(int64_t)ia * H + h] * p.scale_log2, sq_b = p.sq[(int64_t)ib * H + h] * p.scale_log2;
    const float c1a = p.corr1[(int64_t)ia * H + h] * p.scale_log2, c1b = p.corr1[(int64_t)ib * H + h] * p.scale_log2;
    const float c2a = p.corr2 ? p.corr2[(int64_t)ia * H + h] * p.scale_log2 : 0.f;
    const float c2b = p.corr2 ? p.corr2[(int64_t)ib * H + h] * p.scale_log2 : 0.f;
    int lim_a = 0, lim_b = 0;
    if (va) lim_a = p.kv_limit ? min(p.kv_limit[row_a], total) : total;
    if (vb) lim_b = p.kv_limit ? min(p.kv_limit[row_b], total) : total;

    float o_acc[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) { o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f; }
    float m_a = -INFINITY, m_b = -INFINITY;
    float l1_a = 0.f, l1_b = 0.f, l2_a = 0.f, l2_b = 0.f;  // per-segment row sums (unscaled p)
    float sv_prev = 1.f;

    for (int t = 0; t < ntiles; ++t) {
        const int stage = t & 1;
        if (t + 1 < ntiles) load_kv(t + 1, stage ^ 1);
        cp_async_commit();
        const uint8_t* ks = Ks + stage * SMEM_K;
        const uint8_t* vs = Vs + stage * SMEM_V;
        const float* sks = SKs + stage * BKV;
        const int j0 = t * BKV;
        const bool seg1 = j0 < p.len1_pad;
        const float sv_cur = seg1 ? p.sv1[(int64_t)(j0 / BKV) * H + h] : p.sv2[(int64_t)((j0 - p.len1_pad) / BKV) * H + h];
        const float ca = seg1 ? c1a : c2a, cb = seg1 ? c1b : c2b;

        // S = Q K^T : NT n-tiles x 4 k-steps, int8 mma
        int32_t s[G::NT][4];
#pragma unroll
        for (int j = 0; j < G::NT; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0; }
#pragma unroll
        for (int ks_ = 0; ks_ < 4; ++ks_) {
#pragma unroll
            for (int j = 0; j < G::NT; j += 2) {
                uint32_t b0, b1, b2, b3;
                int row = (j + (lidx >> 1)) * 8 + lrow;
                int ch = 2 * ks_ + (lidx & 1);
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + swz128(row, ch)));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_s8_16832(s[j], qf[ks_], bb0);
                mma_s8_16832(s[j + 1], qf[ks_], bb1);
            }
        }
        // dequantize + mask + row max
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
        // O += P V : k = 2 steps of 32 keys, n = 16 tiles of 8 dims. B fragments via ldmatrix on the
        // [dim][key-position] fp8 tile (16 positions = one 8x8 b16 matrix row): x4 covers 2 n-tiles.
#pragma unroll
        for (int ks_ = 0; ks_ < G::KS; ++ks_) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                int d = (j + (lidx >> 1)) * 8 + lrow;
                int off = ks_ * 32 + (lidx & 1) * 16;
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(vs + d * VROWB + off));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_e4m3_16832(o_acc[j], pf[ks_], bb0);
                mma_e4m3_16832(o_acc[j + 1], pf[ks_], bb1);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
    }

    // ---- finalize: O = (O_acc * sv_last / 448 + mean1 * l1 + mean2 * l2) / (l1 + l2)
    const float lt_a = l1_a + l2_a, lt_b = l1_b + l2_b;
    const float inv_a = lt_a > 0.f ? 1.f / lt_a : 0.f, inv_b = lt_b > 0.f ? 1.f / lt_b : 0.f;
    const float vs_a = sv_prev / 448.f, vs_b = sv_prev / 448.f;
    const float* mv1 = p.mv1 + h * 128;
    const float* mv2 = p.mv2 ? p.mv2 + h * 128 : nullptr;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        int d = j * 8 + t4 * 2;
        float m0 = mv1[d] * l1_a + (mv2 ? mv2[d] * l2_a : 0.f);
        float m1 = mv1[d + 1] * l1_a + (mv2 ? mv2[d + 1] * l2_a : 0.f);
        float n0 = mv1[d] * l1_b + (mv2 ? mv2[d] * l2_b : 0.f);
        float n1 = mv1[d + 1] * l1_b + (mv2 ? mv2[d + 1] * l2_b : 0.f);
        if (va) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + h * p.o_hs + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn((o_acc[j][0] * vs_a + m0) * inv_a, (o_acc[j][1] * vs_a + m1) * inv_a);
        }
        if (vb) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + h * p.o_hs + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn((o_acc[j][2] * vs_b + n0) * inv_b, (o_acc[j][3] * vs_b + n1) * inv_b);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) k_sage_attn_64(const SageParams p) { sage_attn_kernel<64>(p); }
extern "C" __global__ void __launch_bounds__(256) k_sage_attn_128(const SageParams p) { sage_attn_kernel<128>(p); }

// =============================================================================================
// Quantization helpers (all for [N][H][128] bf16 inputs with a token stride)
// =============================================================================================
extern "C" {

// Column sums for mean computation: sums[H*128] += sum over tokens of x[tok][h][d].
// grid.x over token chunks, block 256 threads handling 4096 columns -> each thread 16 columns.
__global__ void k_colsum_bf16(const bf16* __restrict__ x, int64_t ts, int n, int cols, float* __restrict__ sums) {
    int64_t t0 = (int64_t)blockIdx.x * 64;
    int64_t t1 = min((int64_t)n, t0 + 64);
    for (int c = threadIdx.x * 8; c < cols; c += blockDim.x * 8) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t t = t0; t < t1; ++t) {
            uint4 v = *reinterpret_cast<const uint4*>(x + t * ts + c);
            const bf162* h2 = reinterpret_cast<const bf162*>(&v);
#pragma unroll
            for (int j = 0; j < 4; ++j) { acc[2 * j] += __low2float(h2[j]); acc[2 * j + 1] += __high2float(h2[j]); }
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) atomicAdd(sums + c + j, acc[j]);
    }
}

// Per-(token, head) int8 quantization with optional per-channel mean subtraction (K smoothing).
// x bf16 [n][H][128] (token stride ts) -> q int8 [n][H][128], scale f32 [n][H]. One warp per (token, head).
__global__ void k_quant_qk_int8(const bf16* __restrict__ x, int64_t ts, int n, int H, const float* __restrict__ mean,
                                int8_t* __restrict__ q, float* __restrict__ scale) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)n * H) return;
    int64_t tok = gw / H; int h = gw % H;
    const bf16* src = x + tok * ts + h * 128 + lane * 4;
    float v[4];
    {
        uint2 raw = *reinterpret_cast<const uint2*>(src);
        const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
        v[0] = __low2float(h2[0]); v[1] = __high2float(h2[0]); v[2] = __low2float(h2[1]); v[3] = __high2float(h2[1]);
    }
    if (mean) {
        const float* m = mean + h * 128 + lane * 4;
        v[0] -= m[0]; v[1] -= m[1]; v[2] -= m[2]; v[3] -= m[3];
    }
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    amax = warp_max(amax);
    float s = amax / 127.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    int8_t o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = (int8_t)max(-128, min(127, __float2int_rn(v[j] * inv)));
    *reinterpret_cast<int32_t*>(q + gw * 128 + lane * 4) = *reinterpret_cast<int32_t*>(o);
    if (lane == 0) scale[gw] = s;
}

// Per-(query, head) correction corr[i][h] = sum_d Q[i][h][d] * mean[h][d]  (bf16 Q with token stride)
__global__ void k_q_corr(const bf16* __restrict__ q, int64_t ts, int n, int H, const float* __restrict__ mean, float* __restrict__ corr) {
    int64_t gw = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (gw >= (int64_t)n * H) return;
    int64_t tok = gw / H; int h = gw % H;
    const bf16* src = q + tok * ts + h * 128 + lane * 4;
    const float* m = mean + h * 128 + lane * 4;
    uint2 raw = *reinterpret_cast<const uint2*>(src);
    const bf162* h2 = reinterpret_cast<const bf162*>(&raw);
    float acc = __low2float(h2[0]) * m[0] + __high2float(h2[0]) * m[1] + __low2float(h2[1]) * m[2] + __high2float(h2[1]) * m[3];
    acc = warp_sum(acc);
    if (lane == 0) corr[gw] = acc;
}

// V -> fp8 transposed per head with the 32-key permutation and a per-(G-key tile, head) scale (G = 64 or 128).
// x bf16 [n][H][128] (token stride ts), mean f32 [H][128].
// out: vt uint8 [H][128][n_pad] (n_pad = ceil(n/G)*G, zero for padded keys), sv f32 [n_pad/G][H].
// One block (256 threads) per (tile, head): G keys x 128 dims.
__global__ void k_quant_v_fp8(const bf16* __restrict__ x, int64_t ts, int n, int H, const float* __restrict__ mean,
                              uint8_t* __restrict__ vt, float* __restrict__ sv, int G) {
    __shared__ float tile[128][129];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const int n_pad = (n + G - 1) / G * G;
    const int tid = threadIdx.x;
    float amax = 0.f;
    for (int e = tid; e < G * 128; e += 256) {
        int key = e >> 7, d = e & 127;
        int j = t * G + key;
        float v = 0.f;
        if (j < n) v = bf2f(x[(int64_t)j * ts + h * 128 + d]) - mean[h * 128 + d];
        tile[key][d] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = block_max(amax, red);
    float s = amax / 448.f;
    float inv = s > 0.f ? 1.f / s : 0.f;
    if (tid == 0) sv[(int64_t)t * H + h] = s > 0.f ? s : 1.f;
    __syncthreads();
    // write transposed: for d in 0..128, positions p in 0..G -> key = perm(p)
    for (int e = tid; e < 128 * G; e += 256) {
        int d = e / G, pos = e % G;
        int grp = pos >> 5, pp = pos & 31;
        int half = pp >> 4, tt = (pp & 15) >> 2, i = pp & 3;
        int key = grp * 32 + half * 16 + (i >> 1) * 8 + 2 * tt + (i & 1);
        float v = tile[key][d] * inv;
        __nv_fp8_storage_t f = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        vt[((int64_t)h * 128 + d) * n_pad + t * G + pos] = (uint8_t)f;
    }
}

__global__ void k_scale_f32(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

}  // extern "C"
