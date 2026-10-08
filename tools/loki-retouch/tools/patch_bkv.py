import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
p='kernels/sage_attn.cu'; s=open(p).read()
s=s.replace('''constexpr int BQ = 128, BKV = 64, THREADS = 256;
constexpr int QROWB = 128;                  // 128 int8 per query row
constexpr int KROWB = 128;                  // 128 int8 per key row
constexpr int VROWB = 80;                   // 64 fp8 keys + 16 pad per dim row
constexpr int SMEM_Q = BQ * QROWB;          // 16 KB
constexpr int SMEM_K = BKV * KROWB;         // 8 KB
constexpr int SMEM_V = 128 * VROWB;         // 10 KB
constexpr int SMEM_SK = BKV * 4;            // 256 B
constexpr int SMEM_TOTAL = SMEM_Q + 2 * (SMEM_K + SMEM_V + SMEM_SK);
''','''constexpr int BQ = 128, THREADS = 256;
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
''')
s=s.replace('''extern "C" __global__ void __launch_bounds__(256) k_sage_attn(const SageParams p) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* Ks = Qs + SMEM_Q;
    uint8_t* Vs = Ks + 2 * SMEM_K;
    float* SKs = reinterpret_cast<float*>(Vs + 2 * SMEM_V);''','''template <int BKV>
__device__ void sage_attn_kernel(const SageParams p) {
    using G = SageGeo<BKV>;
    constexpr int VROWB = G::VROWB, SMEM_K = G::SMEM_K, SMEM_V = G::SMEM_V;
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* Qs = smem;
    uint8_t* Ks = Qs + SMEM_Q;
    uint8_t* Vs = Ks + 2 * SMEM_K;
    float* SKs = reinterpret_cast<float*>(Vs + 2 * SMEM_V);''')
s=s.replace('''        // V^T rows: 128 dims x 4 chunks (64 keys) = 512 chunks, 2 per thread
        const uint8_t* vbase; int vlen; int jj0;
        if (seg1) { vbase = p.v1; vlen = p.len1_pad; jj0 = j0; }
        else { vbase = p.v2; vlen = (p.len2 + 63) / 64 * 64; jj0 = j0 - p.len1_pad; }
        for (int c = tid; c < 128 * 4; c += THREADS) {
            int d = c >> 2, ch = c & 3;
            const uint8_t* src = vbase + ((int64_t)h * 128 + d) * vlen + jj0 + ch * 16;
            cp_async_16(smem_u32(vs + d * VROWB + ch * 16), src, true);
        }''','''        // V^T rows: 128 dims x (BKV/16) chunks
        const uint8_t* vbase; int vlen; int jj0;
        if (seg1) { vbase = p.v1; vlen = p.len1_pad; jj0 = j0; }
        else { vbase = p.v2; vlen = (p.len2 + BKV - 1) / BKV * BKV; jj0 = j0 - p.len1_pad; }
        constexpr int VCH = BKV / 16;
        for (int c = tid; c < 128 * VCH; c += THREADS) {
            int d = c / VCH, ch = c % VCH;
            const uint8_t* src = vbase + ((int64_t)h * 128 + d) * vlen + jj0 + ch * 16;
            cp_async_16(smem_u32(vs + d * VROWB + ch * 16), src, true);
        }''')
s=s.replace('''        const float sv_cur = seg1 ? p.sv1[(int64_t)(j0 / 64) * H + h] : p.sv2[(int64_t)((j0 - p.len1_pad) / 64) * H + h];''','''        const float sv_cur = seg1 ? p.sv1[(int64_t)(j0 / BKV) * H + h] : p.sv2[(int64_t)((j0 - p.len1_pad) / BKV) * H + h];''')
s=s.replace('''        // S = Q K^T : 8 n-tiles x 4 k-steps, int8 mma
        int32_t s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0; }
#pragma unroll
        for (int ks_ = 0; ks_ < 4; ++ks_) {
#pragma unroll
            for (int j = 0; j < 8; j += 2) {''','''        // S = Q K^T : NT n-tiles x 4 k-steps, int8 mma
        int32_t s[G::NT][4];
#pragma unroll
        for (int j = 0; j < G::NT; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0; }
#pragma unroll
        for (int ks_ = 0; ks_ < 4; ++ks_) {
#pragma unroll
            for (int j = 0; j < G::NT; j += 2) {''')
s=s.replace('''        float sf[8][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j) {''','''        float sf[G::NT][4];
        float mx_a = -INFINITY, mx_b = -INFINITY;
#pragma unroll
        for (int j = 0; j < G::NT; ++j) {''')
s=s.replace('''        uint32_t pf[2][4];  // P (x448, e4m3) as A fragments for 2 k-steps of 32 keys
#pragma unroll
        for (int ks_ = 0; ks_ < 2; ++ks_) {''','''        uint32_t pf[G::KS][4];  // P (x448, e4m3) as A fragments for KS k-steps of 32 keys
#pragma unroll
        for (int ks_ = 0; ks_ < G::KS; ++ks_) {''')
s=s.replace('''#pragma unroll
        for (int ks_ = 0; ks_ < 2; ++ks_) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                int d = (j + (lidx >> 1)) * 8 + lrow;''','''#pragma unroll
        for (int ks_ = 0; ks_ < G::KS; ++ks_) {
#pragma unroll
            for (int j = 0; j < 16; j += 2) {
                uint32_t b0, b1, b2, b3;
                int d = (j + (lidx >> 1)) * 8 + lrow;''')
s=s.replace('''// =============================================================================================
// Quantization helpers''','''extern "C" __global__ void __launch_bounds__(256) k_sage_attn_64(const SageParams p) { sage_attn_kernel<64>(p); }
extern "C" __global__ void __launch_bounds__(256) k_sage_attn_128(const SageParams p) { sage_attn_kernel<128>(p); }

// =============================================================================================
// Quantization helpers''')
s=s.replace('''// V -> fp8 transposed per head with the 32-key permutation and a per-(64-key tile, head) scale.
// x bf16 [n][H][128] (token stride ts), mean f32 [H][128].
// out: vt uint8 [H][128][n_pad] (n_pad = ceil(n/64)*64, zero for padded keys), sv f32 [n_pad/64][H].
// One block (256 threads) per (tile, head): 64 keys x 128 dims.
__global__ void k_quant_v_fp8(const bf16* __restrict__ x, int64_t ts, int n, int H, const float* __restrict__ mean,
                              uint8_t* __restrict__ vt, float* __restrict__ sv) {
    __shared__ float tile[64][129];
    __shared__ float red[32];
    const int t = blockIdx.x, h = blockIdx.y;
    const int n_pad = (n + 63) / 64 * 64;
    const int tid = threadIdx.x;
    // load 64 x 128 (each thread: 32 elements)
    float amax = 0.f;
    for (int e = tid; e < 64 * 128; e += 256) {
        int key = e >> 7, d = e & 127;
        int j = t * 64 + key;''','''// V -> fp8 transposed per head with the 32-key permutation and a per-(G-key tile, head) scale (G = 64 or 128).
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
        int j = t * G + key;''')
s=s.replace('''    // write transposed: for d in 0..128, positions p in 0..64 -> key = perm(p)
    for (int e = tid; e < 128 * 64; e += 256) {
        int d = e >> 6, pos = e & 63;''','''    // write transposed: for d in 0..128, positions p in 0..G -> key = perm(p)
    for (int e = tid; e < 128 * G; e += 256) {
        int d = e / G, pos = e % G;''')
s=s.replace('''        vt[((int64_t)h * 128 + d) * n_pad + t * 64 + pos] = (uint8_t)f;''','''        vt[((int64_t)h * 128 + d) * n_pad + t * G + pos] = (uint8_t)f;''')
open(p,'w').write(s)

p='src/ops.rs'; s=open(p).read()
s=s.replace('''pub const SAGE_SMEM: u32 = (128 * 128 + 2 * (64 * 128 + 128 * 80 + 64 * 4)) as u32;

pub fn sage_init(dev: &Device) -> Result<()> {
    dev.set_max_smem("k_sage_attn", SAGE_SMEM)
}''','''/// Key tile / V quantization group size.
pub const SAGE_BKV: usize = 128;
pub const SAGE_SMEM: u32 = (128 * 128 + 2 * (SAGE_BKV * 128 + 128 * (SAGE_BKV + 16) + SAGE_BKV * 4)) as u32;
const SAGE_KERNEL: &str = if SAGE_BKV == 128 { "k_sage_attn_128" } else { "k_sage_attn_64" };

pub fn sage_init(dev: &Device) -> Result<()> {
    dev.set_max_smem(SAGE_KERNEL, SAGE_SMEM)
}''')
s=s.replace('''    let n_pad = (n + 63) / 64 * 64;
    let mean_k = Tensor::new(dev, DType::F32, &[cols])?;''','''    let n_pad = (n + SAGE_BKV - 1) / SAGE_BKV * SAGE_BKV;
    let mean_k = Tensor::new(dev, DType::F32, &[cols])?;''')
s=s.replace('''    let sv = Tensor::new(dev, DType::F32, &[n_pad / 64, h])?;
    dev.launch("k_quant_v_fp8", ((n_pad / 64) as u32, h as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(v), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(mean_v.ptr), Arg::Ptr(vt.ptr), Arg::Ptr(sv.ptr)])?;''','''    let sv = Tensor::new(dev, DType::F32, &[n_pad / SAGE_BKV, h])?;
    dev.launch("k_quant_v_fp8", ((n_pad / SAGE_BKV) as u32, h as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(v), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(mean_v.ptr), Arg::Ptr(vt.ptr), Arg::Ptr(sv.ptr), Arg::I32(SAGE_BKV as i32)])?;''')
s=s.replace('''    let f = dev.func("k_sage_attn")?;''','''    let f = dev.func(SAGE_KERNEL)?;''')
open(p,'w').write(s)
print("patched")
