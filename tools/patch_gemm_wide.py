import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))
p='kernels/gemm_int8.cu'; s=open(p).read()
s=s.replace('''constexpr int BM = 128, BN = 128, BK = 64, STAGES = 4;
constexpr int THREADS = 256;
constexpr int TILE_BYTES = BM * BK;  // 8192 (A) and BN*BK (B) the same
constexpr int SMEM_BYTES = STAGES * 2 * TILE_BYTES;  // 64 KB
''','''constexpr int BM = 128, BK = 64;
constexpr int THREADS = 256;
''')
s=s.replace('''template <typename OutT>
__device__ void gemm_i8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
                               int M, int N, int K, const float* __restrict__ sa, const float* __restrict__ sb,
                               const float* __restrict__ bias, int mode, const OutT* __restrict__ res,
                               const float* __restrict__ gate) {
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * TILE_BYTES;
''','''template <typename OutT, int BN, int STAGES>
__device__ void gemm_i8_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, OutT* __restrict__ C,
                               int M, int N, int K, const float* __restrict__ sa, const float* __restrict__ sb,
                               const float* __restrict__ bias, int mode, const OutT* __restrict__ res,
                               const float* __restrict__ gate) {
    constexpr int A_BYTES = BM * BK, B_BYTES = BN * BK;
    constexpr int WN = BN / 4;           // warp tile width (32 or 64)
    constexpr int NI = WN / 8;           // n-frags per warp (4 or 8)
    constexpr int ACH = A_BYTES / 16 / THREADS;  // A chunks per thread (2)
    constexpr int BCH = B_BYTES / 16 / THREADS;  // B chunks per thread (2 or 4)
    extern __shared__ __align__(128) uint8_t smem[];
    uint8_t* As = smem;
    uint8_t* Bs = smem + STAGES * A_BYTES;
''')
s=s.replace('''    // global -> shared copy assignment: 2 chunks per thread per tile
    int c_row[2], c_chunk[2];
    const int8_t* a_src[2];
    const int8_t* b_src[2];
    bool a_pred[2], b_pred[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int id = tid + i * THREADS;
        c_row[i] = id >> 2;
        c_chunk[i] = id & 3;
        int am = m0 + c_row[i];
        a_pred[i] = am < M;
        a_src[i] = A + (int64_t)(a_pred[i] ? am : 0) * K + c_chunk[i] * 16;
        int bn = n0 + c_row[i];
        b_pred[i] = bn < N;
        b_src[i] = B + (int64_t)(b_pred[i] ? bn : 0) * K + c_chunk[i] * 16;
    }

    auto load_tile = [&](int kt, int stage) {
        const int k0 = kt * BK;
        uint8_t* as = As + stage * TILE_BYTES;
        uint8_t* bs = Bs + stage * TILE_BYTES;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            bool kp = (k0 + c_chunk[i] * 16) < K;
            cp_async_16(smem_u32(as + swz64(c_row[i], c_chunk[i])), a_src[i] + (kp ? k0 : 0), a_pred[i] && kp);
            cp_async_16(smem_u32(bs + swz64(c_row[i], c_chunk[i])), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };

    int32_t acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;
''','''    // global -> shared copy assignment
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
        for (int i = 0; i < ACH; ++i) {
            bool kp = (k0 + a_chunk[i] * 16) < K;
            cp_async_16(smem_u32(as + swz64(a_row[i], a_chunk[i])), a_src[i] + (kp ? k0 : 0), a_pred[i] && kp);
        }
#pragma unroll
        for (int i = 0; i < BCH; ++i) {
            bool kp = (k0 + b_chunk[i] * 16) < K;
            cp_async_16(smem_u32(bs + swz64(b_row[i], b_chunk[i])), b_src[i] + (kp ? k0 : 0), b_pred[i] && kp);
        }
    };

    int32_t acc[4][NI][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0;
''')
s=s.replace('''        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * TILE_BYTES;
        const uint8_t* bs = Bs + stage * TILE_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            uint32_t afrag[4][4];
            uint32_t bfrag[4][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx >> 1);
                ldmatrix_x4(afrag[mi][0], afrag[mi][1], afrag[mi][2], afrag[mi][3], smem_u32(as + swz64(row, chunk)));
            }
#pragma unroll
            for (int nj = 0; nj < 4; nj += 2) {
                int row = wn * 32 + (nj + (lidx >> 1)) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx & 1);
                ldmatrix_x4(bfrag[nj][0], bfrag[nj][1], bfrag[nj + 1][0], bfrag[nj + 1][1], smem_u32(bs + swz64(row, chunk)));
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < 4; ++ni) mma_s8_16832(acc[mi][ni], afrag[mi], bfrag[ni]);
        }''','''        const int stage = kt % STAGES;
        const uint8_t* as = As + stage * A_BYTES;
        const uint8_t* bs = Bs + stage * B_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) {
            uint32_t afrag[4][4];
            uint32_t bfrag[NI][2];
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int row = wm * 64 + mi * 16 + (lidx & 1) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx >> 1);
                ldmatrix_x4(afrag[mi][0], afrag[mi][1], afrag[mi][2], afrag[mi][3], smem_u32(as + swz64(row, chunk)));
            }
#pragma unroll
            for (int nj = 0; nj < NI; nj += 2) {
                int row = wn * WN + (nj + (lidx >> 1)) * 8 + lrow;
                int chunk = (kk >> 4) + (lidx & 1);
                ldmatrix_x4(bfrag[nj][0], bfrag[nj][1], bfrag[nj + 1][0], bfrag[nj + 1][1], smem_u32(bs + swz64(row, chunk)));
            }
#pragma unroll
            for (int mi = 0; mi < 4; ++mi)
#pragma unroll
                for (int ni = 0; ni < NI; ++ni) mma_s8_16832(acc[mi][ni], afrag[mi], bfrag[ni]);
        }''')
s=s.replace('''#pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                int n = n0 + wn * 32 + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = (float)acc[mi][ni][half * 2 + 0] * rs * sb[n];''','''#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
                int n = n0 + wn * WN + ni * 8 + t4 * 2;
                if (n >= N) continue;
                float v0 = (float)acc[mi][ni][half * 2 + 0] * rs * sb[n];''')
s=s.replace('''extern "C" {
__global__ void __launch_bounds__(256) k_gemm_i8_bf16(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                       const float* sa, const float* sb, const float* bias, int mode,
                                                       const bf16* res, const float* gate) {
    gemm_i8_kernel<bf16>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_i8_f32(const int8_t* A, const int8_t* B, float* C, int M, int N, int K,
                                                      const float* sa, const float* sb, const float* bias, int mode,
                                                      const float* res, const float* gate) {
    gemm_i8_kernel<float>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
}''','''extern "C" {
__global__ void __launch_bounds__(256) k_gemm_i8_bf16(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                       const float* sa, const float* sb, const float* bias, int mode,
                                                       const bf16* res, const float* gate) {
    gemm_i8_kernel<bf16, 128, 4>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_i8_f32(const int8_t* A, const int8_t* B, float* C, int M, int N, int K,
                                                      const float* sa, const float* sb, const float* bias, int mode,
                                                      const float* res, const float* gate) {
    gemm_i8_kernel<float, 128, 4>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
// 128x256 tiles, 3 stages (72 KB smem): higher arithmetic intensity for large N
__global__ void __launch_bounds__(256) k_gemm_i8_bf16_w(const int8_t* A, const int8_t* B, bf16* C, int M, int N, int K,
                                                         const float* sa, const float* sb, const float* bias, int mode,
                                                         const bf16* res, const float* gate) {
    gemm_i8_kernel<bf16, 256, 3>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
__global__ void __launch_bounds__(256) k_gemm_i8_f32_w(const int8_t* A, const int8_t* B, float* C, int M, int N, int K,
                                                        const float* sa, const float* sb, const float* bias, int mode,
                                                        const float* res, const float* gate) {
    gemm_i8_kernel<float, 256, 3>(A, B, C, M, N, K, sa, sb, bias, mode, res, gate);
}
}''')
open(p,'w').write(s)

p='src/ops.rs'; s=open(p).read()
s=s.replace('''pub fn gemm_init(dev: &Device) -> Result<()> {
    for k in ["k_gemm_i8_bf16", "k_gemm_i8_f32", "k_gemm_bf16_bf16", "k_gemm_bf16_f32"] {
        dev.set_max_smem(k, GEMM_SMEM)?;
    }''','''pub const GEMM_SMEM_W: u32 = 3 * (128 * 64 + 256 * 64);
/// Use the 128x256 int8 tile when N is large enough to fill it well.
pub fn gemm_i8_wide(m: usize, n: usize) -> bool {
    n % 256 == 0 && n >= 2048 && m >= 1024
}

pub fn gemm_init(dev: &Device) -> Result<()> {
    for k in ["k_gemm_i8_bf16", "k_gemm_i8_f32", "k_gemm_bf16_bf16", "k_gemm_bf16_f32"] {
        dev.set_max_smem(k, GEMM_SMEM)?;
    }
    for k in ["k_gemm_i8_bf16_w", "k_gemm_i8_f32_w"] {
        dev.set_max_smem(k, GEMM_SMEM_W)?;
    }''')
s=s.replace('''    let kname = match out.dtype {
        DType::BF16 => "k_gemm_i8_bf16",
        DType::F32 => "k_gemm_i8_f32",
        _ => anyhow::bail!("gemm_i8: bad out dtype"),
    };
    let grid = ((((n + 127) / 128) * ((m + 127) / 128)) as u32, 1, 1);
    dev.launch(
        kname,
        grid,
        (256, 1, 1),
        GEMM_SMEM,''','''    let wide = gemm_i8_wide(m, n) && std::env::var("LOKI_GEMM_NARROW").is_err();
    let kname = match (out.dtype, wide) {
        (DType::BF16, false) => "k_gemm_i8_bf16",
        (DType::F32, false) => "k_gemm_i8_f32",
        (DType::BF16, true) => "k_gemm_i8_bf16_w",
        (DType::F32, true) => "k_gemm_i8_f32_w",
        _ => anyhow::bail!("gemm_i8: bad out dtype"),
    };
    let bn = if wide { 256 } else { 128 };
    let grid = ((((n + bn - 1) / bn) * ((m + 127) / 128)) as u32, 1, 1);
    dev.launch(
        kname,
        grid,
        (256, 1, 1),
        if wide { GEMM_SMEM_W } else { GEMM_SMEM },''')
open(p,'w').write(s)
print("patched")
