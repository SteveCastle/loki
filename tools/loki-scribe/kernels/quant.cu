// GGUF block-quantized weights: matrix-vector products for decoding and dequantization for prefill.
//
// Weight rows are stored as GGUF blocks (K-quants: 256 elements per block), uploaded verbatim except for the two
// formats whose blocks are not 16-byte aligned, which the loader repacks:
//   Q4_K  144 B  {f16 d, f16 dmin, u8 scales[12], u8 qs[128]}                        (verbatim)
//   Q5_K  176 B  {f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128]}             (verbatim)
//   Q6_K  224 B  {u8 ql[128], u8 qh[64], i8 scales[16], f16 d, pad[14]}               (repacked from 210 B)
//   Q8_0   36 B  {f16 d, pad[2], i8 qs[32]}                    32 elements per block  (repacked from 34 B)
//
// Decoding: activations are quantized per 32-element block to int8 with an f32 scale (q8, like llama.cpp's MMVQ),
// and each warp computes one output row for up to MAXM activation rows with dp4a. Every lane owns a 32-element
// slice of one weight block per step (8 lanes per K-quant block, 4 blocks per warp step).
//   out[m, n] (=|+=) sum_k x[m, k] * W[n, k]
#include "common.cuh"

namespace {

constexpr int Q4K = 0, Q5K = 1, Q6K = 2, Q80 = 3;

template <int T> struct BlockInfo;
template <> struct BlockInfo<Q4K> { static constexpr int BYTES = 144, ELEMS = 256; };
template <> struct BlockInfo<Q5K> { static constexpr int BYTES = 176, ELEMS = 256; };
template <> struct BlockInfo<Q6K> { static constexpr int BYTES = 224, ELEMS = 256; };
template <> struct BlockInfo<Q80> { static constexpr int BYTES = 36, ELEMS = 32; };

DEVI float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }

// 6-bit scale and min of sub-block j (0..7) of a Q4_K/Q5_K block (ggml get_scale_min_k4).
DEVI void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

DEVI int dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// One lane's slice of a block, as int8 quants (values already in their signed/unsigned range) grouped in
// runs of 4 bytes, together with what is needed to scale each run. A "unit" is the work of one lane.
//   K-quants: 8 units per block; Q8_0: 1 unit per block.
template <int T> struct Unit;

// Q4_K / Q5_K unit: chunk j (64 elements = sub-blocks 2j, 2j+1), half h: elements
//   lo: j*64 + h*16 + [0,16)   (sub-block 2j)        hi: j*64 + 32 + h*16 + [0,16)   (sub-block 2j+1)
template <int T>
struct UnitK45 {
    int lo[4], hi[4];
    float dlo, mlo, dhi, mhi;  // value = dlo * q - mlo
    int elo, ehi;              // first element of each run of 16 within the block
    DEVI void load(const uint8_t* blk, int u) {
        const int j = u >> 1, h = u & 1;
        const uint32_t dm = *reinterpret_cast<const uint32_t*>(blk);
        const float d = h2f(dm & 0xFFFF), dmin = h2f(dm >> 16);
        int sc, mn;
        scale_min_k4(2 * j, blk + 4, sc, mn);
        dlo = d * sc; mlo = dmin * mn;
        scale_min_k4(2 * j + 1, blk + 4, sc, mn);
        dhi = d * sc; mhi = dmin * mn;
        elo = j * 64 + h * 16;
        ehi = elo + 32;
        const int qs_off = (T == Q4K ? 16 : 48) + j * 32 + h * 16;
        const uint4 q = *reinterpret_cast<const uint4*>(blk + qs_off);
        const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
        if (T == Q4K) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[i] = qw[i] & 0x0F0F0F0F;
                hi[i] = (qw[i] >> 4) & 0x0F0F0F0F;
            }
        } else {
            const uint4 qh = *reinterpret_cast<const uint4*>(blk + 16 + h * 16);
            const uint32_t hw[4] = {qh.x, qh.y, qh.z, qh.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                lo[i] = (qw[i] & 0x0F0F0F0F) | (((hw[i] >> (2 * j)) & 0x01010101) << 4);
                hi[i] = ((qw[i] >> 4) & 0x0F0F0F0F) | (((hw[i] >> (2 * j + 1)) & 0x01010101) << 4);
            }
        }
    }
};
template <> struct Unit<Q4K> : UnitK45<Q4K> {};
template <> struct Unit<Q5K> : UnitK45<Q5K> {};

// Q6_K unit: half n (128 elements), l = 8*(u&3): runs of 8 at n*128 + g*32 + l (g = 0..3), scale sc[n*8 + l/16 + 2g].
template <>
struct Unit<Q6K> {
    int q[4][2];
    float s[4];
    int e0;  // n*128 + l
    DEVI void load(const uint8_t* blk, int u) {
        const int n = u >> 2, l = (u & 3) * 8;
        const uint2 qla = *reinterpret_cast<const uint2*>(blk + n * 64 + l);
        const uint2 qlb = *reinterpret_cast<const uint2*>(blk + n * 64 + 32 + l);
        const uint2 qhv = *reinterpret_cast<const uint2*>(blk + 128 + n * 32 + l);
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const int8_t* sc = reinterpret_cast<const int8_t*>(blk + 192) + n * 8 + l / 16;
        const uint32_t a[2] = {qla.x, qla.y}, b[2] = {qlb.x, qlb.y}, hh[2] = {qhv.x, qhv.y};
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            q[0][w] = __vsubss4((a[w] & 0x0F0F0F0F) | ((hh[w] << 4) & 0x30303030), 0x20202020);
            q[1][w] = __vsubss4((b[w] & 0x0F0F0F0F) | ((hh[w] << 2) & 0x30303030), 0x20202020);
            q[2][w] = __vsubss4(((a[w] >> 4) & 0x0F0F0F0F) | (hh[w] & 0x30303030), 0x20202020);
            q[3][w] = __vsubss4(((b[w] >> 4) & 0x0F0F0F0F) | ((hh[w] >> 2) & 0x30303030), 0x20202020);
        }
#pragma unroll
        for (int g = 0; g < 4; ++g) s[g] = d * sc[2 * g];
        e0 = n * 128 + l;
    }
};

// Q8_0 unit: the whole 32-element block.
template <>
struct Unit<Q80> {
    int q[8];
    float d;
    DEVI void load(const uint8_t* blk, int) {
        d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const int* p = reinterpret_cast<const int*>(blk + 4);
#pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = p[i];
    }
};

template <int T> struct Units { static constexpr int PER_BLOCK = (T == Q80) ? 1 : 8; };

// ------------------------------------------------------------------------------------------------ GEMV

// Each warp computes ROWS consecutive output rows, so every activation load serves ROWS weight units and ROWS
// independent weight loads are in flight per lane.
constexpr int ROWS = 2;

template <int T, int MAXM>
__device__ void gemv_kernel(const uint8_t* __restrict__ W, const int8_t* __restrict__ xq, const float* __restrict__ xd,
                            float* __restrict__ out, int N, int K, int M, int mode) {
    constexpr int UPB = Units<T>::PER_BLOCK;
    constexpr int BB = BlockInfo<T>::BYTES, BE = BlockInfo<T>::ELEMS;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * (blockDim.x >> 5) + warp) * ROWS;
    if (n0 >= N) return;
    const int nb = K / BE;
    const int kb32 = K / 32;
    const size_t row_bytes = (size_t)nb * BB;
    float acc[ROWS][MAXM];
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
        for (int m = 0; m < MAXM; ++m) acc[r][m] = 0.f;
    const bool has1 = n0 + 1 < N;

    // flat unit index: block = idx / UPB, unit = idx % UPB; a warp covers 32 units per step
    for (int idx = lane; idx < nb * UPB; idx += 32) {
        const int ib = idx / UPB, u = idx % UPB;
        const int base = ib * BE;
        Unit<T> w[ROWS];
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const int n = (r == 1 && !has1) ? n0 : n0 + r;
            w[r].load(W + (size_t)n * row_bytes + (size_t)ib * BB, u);
        }
        if constexpr (T == Q4K || T == Q5K) {
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m >= M) break;
                const int8_t* xr = xq + (size_t)m * K + base;
                const int4 xa = *reinterpret_cast<const int4*>(xr + w[0].elo);
                const int4 xb = *reinterpret_cast<const int4*>(xr + w[0].ehi);
                const float da = xd[(size_t)m * kb32 + (base + w[0].elo) / 32];
                const float db = xd[(size_t)m * kb32 + (base + w[0].ehi) / 32];
                int ua = 0, ub = 0;
                ua = dp4a(0x01010101, xa.x, ua); ua = dp4a(0x01010101, xa.y, ua); ua = dp4a(0x01010101, xa.z, ua); ua = dp4a(0x01010101, xa.w, ua);
                ub = dp4a(0x01010101, xb.x, ub); ub = dp4a(0x01010101, xb.y, ub); ub = dp4a(0x01010101, xb.z, ub); ub = dp4a(0x01010101, xb.w, ub);
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    int sa = 0, sb = 0;
                    sa = dp4a(w[r].lo[0], xa.x, sa); sa = dp4a(w[r].lo[1], xa.y, sa); sa = dp4a(w[r].lo[2], xa.z, sa); sa = dp4a(w[r].lo[3], xa.w, sa);
                    sb = dp4a(w[r].hi[0], xb.x, sb); sb = dp4a(w[r].hi[1], xb.y, sb); sb = dp4a(w[r].hi[2], xb.z, sb); sb = dp4a(w[r].hi[3], xb.w, sb);
                    acc[r][m] += da * (w[r].dlo * (float)sa - w[r].mlo * (float)ua) + db * (w[r].dhi * (float)sb - w[r].mhi * (float)ub);
                }
            }
        } else if constexpr (T == Q6K) {
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m >= M) break;
                const int8_t* xr = xq + (size_t)m * K + base;
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    const int e = w[0].e0 + g * 32;
                    const int2 xv = *reinterpret_cast<const int2*>(xr + e);
                    const float dx = xd[(size_t)m * kb32 + (base + e) / 32];
#pragma unroll
                    for (int r = 0; r < ROWS; ++r) {
                        int s = dp4a(w[r].q[g][0], xv.x, 0);
                        s = dp4a(w[r].q[g][1], xv.y, s);
                        acc[r][m] += w[r].s[g] * dx * (float)s;
                    }
                }
            }
        } else {
#pragma unroll
            for (int m = 0; m < MAXM; ++m) {
                if (m >= M) break;
                const int* xr = reinterpret_cast<const int*>(xq + (size_t)m * K + base);
                int xv[8];
#pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = xr[i];
                const float dx = xd[(size_t)m * kb32 + base / 32];
#pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    int s = 0;
#pragma unroll
                    for (int i = 0; i < 8; ++i) s = dp4a(w[r].q[i], xv[i], s);
                    acc[r][m] += w[r].d * dx * (float)s;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        if (r == 1 && !has1) break;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m >= M) break;
            const float v = warp_sum(acc[r][m]);
            if (lane == 0) {
                float* o = out + (size_t)m * N + n0 + r;
                *o = mode ? *o + v : v;
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ dequant

// One thread per 8 consecutive outputs (16-byte stores, so a warp writes 512 contiguous bytes).
template <int T>
DEVI void dequant8(const uint8_t* __restrict__ blk, int o, float* v) {
    if constexpr (T == Q4K || T == Q5K) {
        const int j = o >> 6, c = o & 63, hi = c >> 5, l = c & 31;  // chunk, half, position
        const uint32_t dm = *reinterpret_cast<const uint32_t*>(blk);
        const float d = h2f(dm & 0xFFFF), dmin = h2f(dm >> 16);
        int sc, mn;
        scale_min_k4(2 * j + hi, blk + 4, sc, mn);
        const float ds = d * sc, ms = dmin * mn;
        const uint8_t* qs = blk + (T == Q4K ? 16 : 48) + j * 32 + l;
        const uint2 q = *reinterpret_cast<const uint2*>(qs);
        const uint32_t qw[2] = {q.x, q.y};
        uint32_t hw[2] = {0, 0};
        if (T == Q5K) {
            const uint2 h = *reinterpret_cast<const uint2*>(blk + 16 + l);
            hw[0] = h.x; hw[1] = h.y;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int qv = (qw[i >> 2] >> (8 * (i & 3) + 4 * hi)) & 0xF;
            if (T == Q5K) qv |= ((hw[i >> 2] >> (8 * (i & 3) + 2 * j + hi)) & 1) << 4;
            v[i] = ds * qv - ms;
        }
    } else if constexpr (T == Q6K) {
        const int n = o >> 7, r = o & 127, g = r >> 5, l = r & 31;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float s = d * (float)reinterpret_cast<const int8_t*>(blk + 192)[n * 8 + l / 16 + 2 * g];
        const uint2 ql = *reinterpret_cast<const uint2*>(blk + n * 64 + (g & 1) * 32 + l);
        const uint2 qh = *reinterpret_cast<const uint2*>(blk + 128 + n * 32 + l);
        const uint32_t lw[2] = {ql.x, ql.y}, hw[2] = {qh.x, qh.y};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int lo = (lw[i >> 2] >> (8 * (i & 3) + (g >= 2 ? 4 : 0))) & 0xF;
            const int h = (hw[i >> 2] >> (8 * (i & 3) + 2 * g)) & 3;
            v[i] = s * (float)((lo | (h << 4)) - 32);
        }
    } else {
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const uint32_t* q = reinterpret_cast<const uint32_t*>(blk + 4 + o);  // 36-byte blocks: 4-byte aligned only
        const uint32_t qw[2] = {q[0], q[1]};
#pragma unroll
        for (int i = 0; i < 8; ++i) v[i] = d * (float)(int8_t)((qw[i >> 2] >> (8 * (i & 3))) & 0xFF);
    }
}

template <int T>
__device__ void dequant_kernel(const uint8_t* __restrict__ W, bf16* __restrict__ out, int N, int K) {
    constexpr int BB = BlockInfo<T>::BYTES, BE = BlockInfo<T>::ELEMS;
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t e0 = t * 8;
    if (e0 >= (int64_t)N * K) return;
    const int64_t n = e0 / K;
    const int k0 = e0 % K;
    const int nb = K / BE;
    const uint8_t* blk = W + (n * nb + k0 / BE) * BB;
    float v[8];
    dequant8<T>(blk, k0 % BE, v);
    uint4 pk;
    pk.x = pack_bf16x2(v[0], v[1]);
    pk.y = pack_bf16x2(v[2], v[3]);
    pk.z = pack_bf16x2(v[4], v[5]);
    pk.w = pack_bf16x2(v[6], v[7]);
    *reinterpret_cast<uint4*>(out + e0) = pk;
}

}  // namespace

extern "C" {

// Quantize activation rows to q8: x f32 [M, K] -> xq int8 [M, K], xd f32 [M, K/32]. One warp per 32-block.
__global__ void k_quant_q8(const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd, int64_t nblocks) {
    const int64_t b = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (b >= nblocks) return;
    const float v = x[b * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float d = amax / 127.f;
    const float id = d > 0.f ? 1.f / d : 0.f;
    xq[b * 32 + lane] = (int8_t)__float2int_rn(v * id);
    if (lane == 0) xd[b] = d;
}

#define GEMV(NAME, T, MAXM)                                                                                        \
    __global__ void __launch_bounds__(256) NAME(const uint8_t* W, const int8_t* xq, const float* xd, float* out, \
                                                int N, int K, int M, int mode) {                                 \
        gemv_kernel<T, MAXM>(W, xq, xd, out, N, K, M, mode);                                                     \
    }
GEMV(k_gemv_q4k_1, Q4K, 1)
GEMV(k_gemv_q4k_2, Q4K, 2)
GEMV(k_gemv_q4k_4, Q4K, 4)
GEMV(k_gemv_q4k_8, Q4K, 8)
GEMV(k_gemv_q5k_1, Q5K, 1)
GEMV(k_gemv_q5k_2, Q5K, 2)
GEMV(k_gemv_q5k_4, Q5K, 4)
GEMV(k_gemv_q5k_8, Q5K, 8)
GEMV(k_gemv_q6k_1, Q6K, 1)
GEMV(k_gemv_q6k_2, Q6K, 2)
GEMV(k_gemv_q6k_4, Q6K, 4)
GEMV(k_gemv_q6k_8, Q6K, 8)
GEMV(k_gemv_q80_1, Q80, 1)
GEMV(k_gemv_q80_2, Q80, 2)
GEMV(k_gemv_q80_4, Q80, 4)
GEMV(k_gemv_q80_8, Q80, 8)

__global__ void k_dequant_q4k(const uint8_t* W, bf16* out, int N, int K) { dequant_kernel<Q4K>(W, out, N, K); }
__global__ void k_dequant_q5k(const uint8_t* W, bf16* out, int N, int K) { dequant_kernel<Q5K>(W, out, N, K); }
__global__ void k_dequant_q6k(const uint8_t* W, bf16* out, int N, int K) { dequant_kernel<Q6K>(W, out, N, K); }
__global__ void k_dequant_q80(const uint8_t* W, bf16* out, int N, int K) { dequant_kernel<Q80>(W, out, N, K); }

// bf16 weights, f32 activations: out[m, n] (=|+=) x[m, :] . W[n, :]. One warp per row, MAXM <= 8.
__global__ void __launch_bounds__(256) k_gemv_bf16(const bf16* __restrict__ W, const float* __restrict__ x, float* __restrict__ out,
                                                   int N, int K, int M, int mode) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = blockIdx.x * (blockDim.x >> 5) + warp;
    if (n >= N) return;
    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const bf16* row = W + (size_t)n * K;
    for (int k = lane * 2; k < K; k += 64) {
        const bf162 w2 = *reinterpret_cast<const bf162*>(row + k);
        const float w0 = __low2float(w2), w1 = __high2float(w2);
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            if (m >= M) break;
            const float2 xv = *reinterpret_cast<const float2*>(x + (size_t)m * K + k);
            acc[m] = fmaf(w0, xv.x, fmaf(w1, xv.y, acc[m]));
        }
    }
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        if (m >= M) break;
        const float v = warp_sum(acc[m]);
        if (lane == 0) {
            float* o = out + (size_t)m * N + n;
            *o = mode ? *o + v : v;
        }
    }
}

}  // extern "C"
