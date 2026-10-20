// Video Sparse Attention (VSA, FastVideo) for the MiniMax H3 DiT, as ComfyUI's BlockSparseAttention "vsa" mode runs it
// (comfy_extras/nodes_sparse_attention.py + comfy_kitchen sol_attn, eager reference backends/eager/sol_attn.py):
//   * the sequence is tiled in 64-row tiles: every non-video segment zero-padded to whole tiles ("sinks"), the target
//     video in 4x4x4 latent cubes (live rows first); block_len = live rows of a tile;
//   * fine branch: each (query tile, head) attends exactly the top-k key tiles by centroid score q_mean . k_mean
//     (k = 10% of the non-sink tiles, ties kept), the diagonal +-1 tiles and all sink tiles; sink query tiles attend
//     everything; softmax over the selected live keys only (no pooled tail);
//   * coarse branch: softmax(q_mean k_mean^T * scale) v_mean over all tiles, added per row as gate * coarse[tile] with
//     gate = to_gate_compress(h).
// The fine branch runs in k_h3_attn_v6_sparse (h3_dit.cu); these kernels build its inputs.
#include "common.cuh"

namespace {
constexpr int TILE = 64;

// float -> uint32 with the same order
__device__ __forceinline__ uint32_t ord_key(float f) {
    uint32_t b = __float_as_uint(f);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
}  // namespace

extern "C" {

// dst[map[i]] = src[i] for n bf16 rows of K (K % 8 == 0)
__global__ void k_vsa_scatter_rows_bf16(const bf16* __restrict__ src, int n, int K, const int* __restrict__ map, bf16* __restrict__ dst) {
    const int k8 = K / 8;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * k8) return;
    const int64_t r = i / k8, c = (i % k8) * 8;
    *reinterpret_cast<uint4*>(dst + (int64_t)map[r] * K + c) = *reinterpret_cast<const uint4*>(src + r * K + c);
}

// x[rows[i]] = 0 for n bf16 rows of K
__global__ void k_vsa_zero_rows_bf16(bf16* __restrict__ x, const int* __restrict__ rows, int n, int K) {
    const int k8 = K / 8;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * k8) return;
    const int64_t r = i / k8, c = (i % k8) * 8;
    *reinterpret_cast<uint4*>(x + (int64_t)rows[r] * K + c) = make_uint4(0, 0, 0, 0);
}

// Tile means of int8 per-(row, head) quantized rows: out[i][h][d] = mean_{r < len} x8[i*64 + r][h][d] * s[i*64 + r][h],
// len = blen[tile0 + i] clamped to the rows present (nrows). grid (ntiles, H), 128 threads.
__global__ void k_vsa_tile_mean_i8(const int8_t* __restrict__ x8, const float* __restrict__ s, int nrows, int H,
                                   const int* __restrict__ blen, int tile0, float* __restrict__ out) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int len = min(blen[tile0 + i], nrows - i * TILE);
    float acc = 0.f;
    for (int r = 0; r < len; ++r) {
        const int64_t row = (int64_t)i * TILE + r;
        acc += (float)x8[(row * H + h) * 128 + d] * s[row * H + h];
    }
    out[((int64_t)i * H + h) * 128 + d] = len > 0 ? acc / len : 0.f;
}

// Tile means of bf16 rows (row stride ts, head h at column h*128): as above. grid (ntiles, H), 128 threads.
__global__ void k_vsa_tile_mean_bf16(const bf16* __restrict__ v, int64_t ts, int nrows, int H, const int* __restrict__ blen,
                                     int tile0, float* __restrict__ out) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int len = min(blen[tile0 + i], nrows - i * TILE);
    float acc = 0.f;
    for (int r = 0; r < len; ++r) acc += bf2f(v[((int64_t)i * TILE + r) * ts + h * 128 + d]);
    out[((int64_t)i * H + h) * 128 + d] = len > 0 ? acc / len : 0.f;
}

// Per (group of qg <= 8 query tiles of a batch, head): scores c_j = q_mean . k_mean_j over all NT key tiles, then
//   * selection bits (W = ceil(NT/32) words): sinks (j < nsink), |j - qt| <= 1, and the kk best non-sink tiles
//     (every tile scoring >= the kk-th best, so ties are kept); a sink query tile (qt < nsink) selects everything;
//   * coarse output oc = softmax(c * scale) . v_mean.
// Query tile b of the batch is global tile qt0 + b. One warp per query tile for the top-k; the key / value means are
// read once per group. grid (ceil(ntq / qg), H), 256 threads, dyn smem qg * NT floats.
__global__ void __launch_bounds__(256) k_vsa_select(const float* __restrict__ qm, const float* __restrict__ km,
                                                    const float* __restrict__ vm, int NT, int H, int nsink, int qt0, int kk,
                                                    float scale, int ntq, int qg, uint32_t* __restrict__ bits,
                                                    float* __restrict__ oc) {
    constexpr int G = 8;
    extern __shared__ float sc[];  // [qg][NT]
    __shared__ float q[G][128];
    __shared__ float acc2[G][128];
    __shared__ float tot[G];
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int b0 = blockIdx.x * qg;
    const int ng = min(qg, ntq - b0);
    const int W = (NT + 31) / 32;
    for (int e = tid; e < G * 128; e += 256) {
        const int g = e >> 7, d = e & 127;
        q[g][d] = g < ng ? qm[((int64_t)(b0 + g) * H + h) * 128 + d] : 0.f;
    }
    __syncthreads();
    for (int j = tid; j < NT; j += 256) {
        const float4* k4 = reinterpret_cast<const float4*>(km + ((int64_t)j * H + h) * 128);
        float a[G];
#pragma unroll
        for (int g = 0; g < G; ++g) a[g] = 0.f;
#pragma unroll 4
        for (int e = 0; e < 32; ++e) {
            const float4 kv = k4[e];
#pragma unroll
            for (int g = 0; g < G; ++g)
                a[g] = fmaf(q[g][4 * e], kv.x, fmaf(q[g][4 * e + 1], kv.y, fmaf(q[g][4 * e + 2], kv.z, fmaf(q[g][4 * e + 3], kv.w, a[g]))));
        }
        for (int g = 0; g < ng; ++g) sc[g * NT + j] = a[g];
    }
    __syncthreads();

    if (warp < ng) {
        const int g = warp, qt = qt0 + b0 + g;
        float* s = sc + g * NT;
        // ---- top-k threshold over the non-sink tiles: the largest key T with count(key >= T) >= kk
        const bool dense = qt < nsink;
        const bool use_thr = !dense && kk > 0;
        uint32_t thr = 0xFFFFFFFFu;
        if (use_thr) {
            uint32_t lo = 0, hi = 0xFFFFFFFFu;
            while (lo < hi) {
                const uint32_t mid = lo + (uint32_t)((((uint64_t)hi - lo) + 1) >> 1);
                int c = 0;
                for (int j = nsink + lane; j < NT; j += 32) c += ord_key(s[j]) >= mid;
                for (int o = 16; o > 0; o >>= 1) c += __shfl_xor_sync(0xffffffff, c, o);
                if (c >= kk) lo = mid; else hi = mid - 1;
            }
            thr = lo;
        }
        uint32_t* out = bits + ((int64_t)(b0 + g) * H + h) * W;
        for (int base = 0; base < NT; base += 32) {
            const int j = base + lane;
            const bool sel = j < NT && (dense || j < nsink || abs(j - qt) <= 1 || (use_thr && ord_key(s[j]) >= thr));
            const uint32_t word = __ballot_sync(0xffffffff, sel);
            if (lane == 0) out[base >> 5] = word;
        }
        // ---- coarse weights: exp(c * scale - max), in place
        float mx = -INFINITY;
        for (int j = lane; j < NT; j += 32) mx = fmaxf(mx, s[j] * scale);
        for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
        float sum = 0.f;
        for (int j = lane; j < NT; j += 32) {
            const float e = __expf(s[j] * scale - mx);
            s[j] = e;
            sum += e;
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
        if (lane == 0) tot[g] = sum;
    }
    __syncthreads();
    // ---- coarse output: two halves of the key tiles per output dim
    const int d = tid & 127, half = tid >> 7;
    float a[G];
#pragma unroll
    for (int g = 0; g < G; ++g) a[g] = 0.f;
    for (int j = half; j < NT; j += 2) {
        const float v = vm[((int64_t)j * H + h) * 128 + d];
#pragma unroll
        for (int g = 0; g < G; ++g)
            if (g < ng) a[g] = fmaf(sc[g * NT + j], v, a[g]);
    }
    if (half)
        for (int g = 0; g < ng; ++g) acc2[g][d] = a[g];
    __syncthreads();
    if (!half)
        for (int g = 0; g < ng; ++g) oc[((int64_t)(b0 + g) * H + h) * 128 + d] = (a[g] + acc2[g][d]) / tot[g];
}

// Per (query block of 4 tiles, head): the union of the 4 tiles' selections as an ascending list of
// tile | (mask4 << 24) (bit i = query tile 4*blk + i selected it). grid (nqb, H), 32 threads.
__global__ void k_vsa_lists(const uint32_t* __restrict__ bits, int ntq, int H, int W, int cap, int* __restrict__ list,
                            int* __restrict__ cnt) {
    const int blk = blockIdx.x, h = blockIdx.y, lane = threadIdx.x;
    int* out = list + ((int64_t)blk * H + h) * cap;
    int off = 0;
    for (int base = 0; base < W; base += 32) {
        const int w = base + lane;
        uint32_t m[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int qt = blk * 4 + i;
            m[i] = (w < W && qt < ntq) ? bits[((int64_t)qt * H + h) * W + w] : 0u;
        }
        uint32_t u = m[0] | m[1] | m[2] | m[3];
        const int c = __popc(u);
        int pre = c;
        for (int o = 1; o < 32; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffff, pre, o);
            if (lane >= o) pre += t;
        }
        const int total = __shfl_sync(0xffffffff, pre, 31);
        int pos = off + pre - c;
        while (u) {
            const int bit = __ffs(u) - 1;
            u &= u - 1;
            const int mask = ((m[0] >> bit) & 1) | (((m[1] >> bit) & 1) << 1) | (((m[2] >> bit) & 1) << 2) | (((m[3] >> bit) & 1) << 3);
            out[pos++] = (w * 32 + bit) | (mask << 24);
        }
        off += total;
    }
    if (lane == 0) cnt[blk * H + h] = off;
}

// out[r][c] += gate[r][c] * oc[r / 64][c]   (bf16 rows of K = H*128, oc f32 [tiles][K])
__global__ void k_vsa_add_coarse(bf16* __restrict__ out, const bf16* __restrict__ gate, const float* __restrict__ oc, int n, int K) {
    const int k2 = K / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * k2) return;
    const int64_t r = i / k2, c = (i % k2) * 2;
    bf162* o = reinterpret_cast<bf162*>(out + r * K + c);
    const bf162 g = *reinterpret_cast<const bf162*>(gate + r * K + c);
    const float* cc = oc + (r / TILE) * K + c;
    const float2 of = __bfloat1622float2(*o), gf = __bfloat1622float2(g);
    *o = __floats2bfloat162_rn(fmaf(gf.x, cc[0], of.x), fmaf(gf.y, cc[1], of.y));
}

}  // extern "C"
