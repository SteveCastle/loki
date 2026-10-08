// Fused flash attention forward (bf16 in, f32 accumulate, bf16 out).
//
// Q: nq tokens, Hq heads; K/V: two consecutive segments (len1 from K1/V1, len2 from K2/V2)
// so a cached prefix never has to be concatenated with fresh keys. GQA: kv head = hq / (Hq/Hk).
// Masking: key j is visible to query i iff j < limit(i), where limit(i) = kv_limit[i] when
// kv_limit != null, else (causal ? i + 1 + causal_off : len1+len2).
//
// Layout: token-major with explicit strides (elements): x[tok * tok_stride + head * head_stride + d].
// Head dim D (compile-time 128 or 80) with d_real contiguous dims loaded (d_real*2 bytes, multiple of 16);
// the remaining dims are zero-filled in shared memory.
//
// Block: 128 queries (8 warps x 16 rows), KV tile 64 keys, double-buffered cp.async.
#include "common.cuh"

namespace {

constexpr int BQ = 128, BKV = 64, THREADS = 256;

template <int D>
struct Geo;
template <>
struct Geo<128> {
    static constexpr int ROWB = 256;  // bytes per row
    static constexpr int CHUNKS = 16;
    static constexpr int NT = 16;  // n-tiles (8 dims each) of the output
    static constexpr int KS = 8;   // k-steps (16 dims) for QK^T
    DEVI static int off(int row, int chunk) { return row * ROWB + ((chunk ^ (row & 7)) << 4); }
};
template <>
struct Geo<80> {
    static constexpr int ROWB = 176;  // 11 chunks, padded (conflict-free without xor)
    static constexpr int CHUNKS = 10;
    static constexpr int NT = 10;
    static constexpr int KS = 5;
    DEVI static int off(int row, int chunk) { return row * ROWB + (chunk << 4); }
};

struct AttnParams {
    const bf16* q; int64_t q_ts; int q_hs;
    const bf16* k1; const bf16* v1; int64_t kv1_ts; int kv1_hs; int len1;
    const bf16* k2; const bf16* v2; int64_t kv2_ts; int kv2_hs; int len2;
    bf16* o; int64_t o_ts; int o_hs;
    int nq, hq, hk, d_real;
    const int* kv_limit; int causal; int causal_off;
    float scale_log2;
};

template <int D>
__device__ void flash_attn_kernel(const AttnParams p) {
    using G = Geo<D>;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;                                  // BQ * ROWB
    uint8_t* Ks = smem + BQ * G::ROWB;                   // 2 * BKV * ROWB
    uint8_t* Vs = Ks + 2 * BKV * G::ROWB;                // 2 * BKV * ROWB

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ;
    const int hq = blockIdx.y;
    const int hk = hq / (p.hq / p.hk);
    const int total = p.len1 + p.len2;
    const int dchunks = p.d_real >> 3;  // 16B chunks of real data per row

    // ---- block key limit
    int my_limit = 0;
    for (int r = tid; r < BQ; r += THREADS) {
        int i = q0 + r;
        int lim = 0;
        if (i < p.nq) {
            if (p.kv_limit) lim = p.kv_limit[i];
            else if (p.causal) lim = i + 1 + p.causal_off;
            else lim = total;
        }
        my_limit = max(my_limit, min(lim, total));
    }
    __shared__ float red[32];
    int block_limit = (int)block_max((float)my_limit, red);
    const int ntiles = (block_limit + BKV - 1) / BKV;
    if (ntiles == 0) return;

    // ---- load Q tile (zero-fill rows >= nq and chunks >= dchunks)
    for (int c = tid; c < BQ * G::CHUNKS; c += THREADS) {
        int row = c / G::CHUNKS, ch = c % G::CHUNKS;
        int i = q0 + row;
        bool pred = (i < p.nq) && (ch < dchunks);
        const bf16* src = p.q + (int64_t)(pred ? i : 0) * p.q_ts + hq * p.q_hs + ch * 8;
        if (ch < dchunks) cp_async_16(smem_u32(Qs + G::off(row, ch)), src, pred);
        else *reinterpret_cast<uint4*>(Qs + G::off(row, ch)) = make_uint4(0, 0, 0, 0);
    }
    auto load_kv = [&](int tile, int stage) {
        const int j0 = tile * BKV;
        uint8_t* ks = Ks + stage * BKV * G::ROWB;
        uint8_t* vs = Vs + stage * BKV * G::ROWB;
        for (int c = tid; c < BKV * G::CHUNKS; c += THREADS) {
            int row = c / G::CHUNKS, ch = c % G::CHUNKS;
            int j = j0 + row;
            bool pred = (j < total) && (ch < dchunks);
            const bf16* ksrc; const bf16* vsrc;
            if (j < p.len1) {
                int64_t o = (int64_t)j * p.kv1_ts + hk * p.kv1_hs + ch * 8;
                ksrc = p.k1 + o; vsrc = p.v1 + o;
            } else {
                int jj = j - p.len1;
                if (jj >= p.len2) jj = 0;
                int64_t o = (int64_t)jj * p.kv2_ts + hk * p.kv2_hs + ch * 8;
                ksrc = p.k2 + o; vsrc = p.v2 + o;
            }
            if (ch < dchunks) {
                cp_async_16(smem_u32(ks + G::off(row, ch)), ksrc, pred);
                cp_async_16(smem_u32(vs + G::off(row, ch)), vsrc, pred);
            } else {
                *reinterpret_cast<uint4*>(ks + G::off(row, ch)) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(vs + G::off(row, ch)) = make_uint4(0, 0, 0, 0);
            }
        }
    };
    load_kv(0, 0);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    // ---- Q fragments in registers
    const int lidx = lane >> 3, lrow = lane & 7;
    uint32_t qf[G::KS][4];
#pragma unroll
    for (int s = 0; s < G::KS; ++s) {
        int row = warp * 16 + (lidx & 1) * 8 + lrow;
        int ch = 2 * s + (lidx >> 1);
        ldmatrix_x4(qf[s][0], qf[s][1], qf[s][2], qf[s][3], smem_u32(Qs + G::off(row, ch)));
    }

    const int g = lane >> 2, t4 = lane & 3;
    const int row_a = q0 + warp * 16 + g, row_b = row_a + 8;
    int lim_a, lim_b;
    {
        auto lim_of = [&](int i) {
            if (i >= p.nq) return 0;
            if (p.kv_limit) return min(p.kv_limit[i], total);
            if (p.causal) return min(i + 1 + p.causal_off, total);
            return total;
        };
        lim_a = lim_of(row_a); lim_b = lim_of(row_b);
    }

    float o_acc[G::NT][4];
#pragma unroll
    for (int j = 0; j < G::NT; ++j) { o_acc[j][0] = o_acc[j][1] = o_acc[j][2] = o_acc[j][3] = 0.f; }
    float m_a = -INFINITY, m_b = -INFINITY, l_a = 0.f, l_b = 0.f;

    for (int t = 0; t < ntiles; ++t) {
        const int stage = t & 1;
        if (t + 1 < ntiles) load_kv(t + 1, stage ^ 1);
        cp_async_commit();
        const uint8_t* ks = Ks + stage * BKV * G::ROWB;
        const uint8_t* vs = Vs + stage * BKV * G::ROWB;
        const int j0 = t * BKV;

        // S = Q K^T  (16 x 64 per warp): 8 n-tiles of 8 keys
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f; }
#pragma unroll
        for (int ks_ = 0; ks_ < G::KS; ++ks_) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {
                uint32_t b0, b1, b2, b3;
                int row = (j + (lidx >> 1)) * 8 + lrow;
                int ch = 2 * ks_ + (lidx & 1);
                ldmatrix_x4(b0, b1, b2, b3, smem_u32(ks + G::off(row, ch)));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(s[j], qf[ks_], bb0);
                mma_bf16_16816(s[j + 1], qf[ks_], bb1);
            }
        }
        // scale + mask + row max
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int key = j0 + j * 8 + t4 * 2;
            s[j][0] = (key < lim_a) ? s[j][0] * p.scale_log2 : -INFINITY;
            s[j][1] = (key + 1 < lim_a) ? s[j][1] * p.scale_log2 : -INFINITY;
            s[j][2] = (key < lim_b) ? s[j][2] * p.scale_log2 : -INFINITY;
            s[j][3] = (key + 1 < lim_b) ? s[j][3] * p.scale_log2 : -INFINITY;
            mx_a = fmaxf(mx_a, fmaxf(s[j][0], s[j][1]));
            mx_b = fmaxf(mx_b, fmaxf(s[j][2], s[j][3]));
        }
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 1));
        mx_a = fmaxf(mx_a, __shfl_xor_sync(0xffffffff, mx_a, 2));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 1));
        mx_b = fmaxf(mx_b, __shfl_xor_sync(0xffffffff, mx_b, 2));
        float mn_a = fmaxf(m_a, mx_a), mn_b = fmaxf(m_b, mx_b);
        float alpha_a = (m_a == -INFINITY) ? 0.f : fast_exp2(m_a - mn_a);
        float alpha_b = (m_b == -INFINITY) ? 0.f : fast_exp2(m_b - mn_b);
        float rs_a = 0.f, rs_b = 0.f;
        uint32_t pf[4][4];  // P as A fragments for 4 k-steps of 16 keys
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            float p0 = (mn_a == -INFINITY) ? 0.f : fast_exp2(s[j][0] - mn_a);
            float p1 = (mn_a == -INFINITY) ? 0.f : fast_exp2(s[j][1] - mn_a);
            float p2 = (mn_b == -INFINITY) ? 0.f : fast_exp2(s[j][2] - mn_b);
            float p3 = (mn_b == -INFINITY) ? 0.f : fast_exp2(s[j][3] - mn_b);
            rs_a += p0 + p1; rs_b += p2 + p3;
            int ks_ = j >> 1, hi = j & 1;
            pf[ks_][hi * 2 + 0] = pack_bf16x2(p0, p1);
            pf[ks_][hi * 2 + 1] = pack_bf16x2(p2, p3);
        }
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 1);
        rs_a += __shfl_xor_sync(0xffffffff, rs_a, 2);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 1);
        rs_b += __shfl_xor_sync(0xffffffff, rs_b, 2);
        l_a = l_a * alpha_a + rs_a; l_b = l_b * alpha_b + rs_b;
        m_a = mn_a; m_b = mn_b;
#pragma unroll
        for (int j = 0; j < G::NT; ++j) {
            o_acc[j][0] *= alpha_a; o_acc[j][1] *= alpha_a;
            o_acc[j][2] *= alpha_b; o_acc[j][3] *= alpha_b;
        }
        // O += P V : k = keys (4 steps of 16), n = dims (NT tiles)
#pragma unroll
        for (int ks_ = 0; ks_ < 4; ++ks_) {
#pragma unroll
            for (int j = 0; j < G::NT; j += 2) {
                uint32_t b0, b1, b2, b3;
                int key = ks_ * 16 + (lidx & 1) * 8 + lrow;
                int ch = j + (lidx >> 1);
                ldmatrix_x4_trans(b0, b1, b2, b3, smem_u32(vs + G::off(key, ch)));
                uint32_t bb0[2] = {b0, b1}, bb1[2] = {b2, b3};
                mma_bf16_16816(o_acc[j], pf[ks_], bb0);
                mma_bf16_16816(o_acc[j + 1], pf[ks_], bb1);
            }
        }
        cp_async_wait<0>();
        __syncthreads();
    }

    // ---- normalize and store
    float inv_a = (l_a > 0.f) ? 1.f / l_a : 0.f, inv_b = (l_b > 0.f) ? 1.f / l_b : 0.f;
#pragma unroll
    for (int j = 0; j < G::NT; ++j) {
        int d = j * 8 + t4 * 2;
        if (d >= p.d_real) continue;
        if (row_a < p.nq) {
            bf16* dst = p.o + (int64_t)row_a * p.o_ts + hq * p.o_hs + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][0] * inv_a, o_acc[j][1] * inv_a);
        }
        if (row_b < p.nq) {
            bf16* dst = p.o + (int64_t)row_b * p.o_ts + hq * p.o_hs + d;
            *reinterpret_cast<bf162*>(dst) = __floats2bfloat162_rn(o_acc[j][2] * inv_b, o_acc[j][3] * inv_b);
        }
    }
}

}  // namespace

extern "C" {
__global__ void __launch_bounds__(256) k_flash_attn_d128(AttnParams p) { flash_attn_kernel<128>(p); }
__global__ void __launch_bounds__(256) k_flash_attn_d80(AttnParams p) { flash_attn_kernel<80>(p); }
}
