import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ---------------- (a) adaln fused into the quantizer: act 4 = LayerNorm(x)*(1+scale) ----------------
p='kernels/quant.cu'; s=open(p).read()
s=s.replace('''//   act: 0 none (K = Kin), 1 gelu_tanh (K = Kin), 2 swiglu (K = Kin/2, row = [gate|up]),
//        3 rmsnorm with weight w[K], eps (K = Kin)''','''//   act: 0 none (K = Kin), 1 gelu_tanh (K = Kin), 2 swiglu (K = Kin/2, row = [gate|up]),
//        3 rmsnorm with weight w[K], eps (K = Kin)
//        4 adaln: LayerNorm(x, no affine, eps) * (1 + scale[K]) with scale f32 (passed as `w`)''')
s=s.replace('''template <typename TI>
__device__ void quant_row(const TI* __restrict__ in, int Kin, int act, const bf16* __restrict__ w, float eps,
                          int8_t* __restrict__ out, float* __restrict__ scale_out, float* row) {''','''template <typename TI>
__device__ void quant_row(const TI* __restrict__ in, int Kin, int act, const bf16* __restrict__ w, float eps,
                          int8_t* __restrict__ out, float* __restrict__ scale_out, float* row,
                          const float* __restrict__ adascale = nullptr) {''')
s=s.replace('''    } else if (act == 3) {
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; ss += v * v; }
        float r = rsqrtf(block_sum(ss, red) / K + eps);
        for (int i = t; i < K; i += 256) row[i] = row[i] * r * bf2f(w[i]);
    } else {''','''    } else if (act == 3) {
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; ss += v * v; }
        float r = rsqrtf(block_sum(ss, red) / K + eps);
        for (int i = t; i < K; i += 256) row[i] = row[i] * r * bf2f(w[i]);
    } else if (act == 4) {
        float sum = 0.f;
        for (int i = t; i < K; i += 256) { float v = (float)in[i]; row[i] = v; sum += v; }
        float mean = block_sum(sum, red) / K;
        float ss = 0.f;
        for (int i = t; i < K; i += 256) { float v = row[i] - mean; ss += v * v; }
        float rstd = rsqrtf(block_sum(ss, red) / K + eps);
        // ComfyUI's adaln kernel hands the modulated activation to the int8 quantizer as bf16
        for (int i = t; i < K; i += 256) row[i] = round_bf16((row[i] - mean) * rstd * (1.0f + adascale[i]));
    } else {''')
s=s.replace('''__global__ void k_quant_rows_bf16(const bf16* __restrict__ in, int M, int Kin, int act, const bf16* w, float eps,
                                  int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    const int K = (act == 2) ? Kin / 2 : Kin;
    quant_row<bf16>(in + m * Kin, Kin, act, w, eps, out + m * K, scale + m, row);
}''','''__global__ void k_quant_rows_bf16(const bf16* __restrict__ in, int M, int Kin, int act, const bf16* w, float eps,
                                  int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    const int K = (act == 2) ? Kin / 2 : Kin;
    quant_row<bf16>(in + m * Kin, Kin, act, w, eps, out + m * K, scale + m, row);
}
// adaln + quantize: scale f32 [K]
__global__ void k_adaln_quant_bf16(const bf16* __restrict__ in, int M, int K, float eps, const float* __restrict__ adascale,
                                   int8_t* __restrict__ out, float* __restrict__ scale) {
    extern __shared__ float row[];
    int64_t m = blockIdx.x; if (m >= M) return;
    quant_row<bf16>(in + m * K, K, 4, nullptr, eps, out + m * K, scale + m, row, adascale);
}''')
open(p,'w').write(s)

# ---------------- (b) swiglu pairs in the int8 GEMM epilogue (mode 3) ----------------
p='kernels/gemm_int8.cu'; s=open(p).read()
s=s.replace('''//   mode 0: out = v
//   mode 1: out = res + v
//   mode 2: out = res + v * gate[n]''','''//   mode 0: out = v
//   mode 1: out = res + v
//   mode 2: out = res + v * gate[n]
//   mode 3: out[m][n/2] = silu(v[n]) * v[n+1] for even n (interleaved gate/up rows; output width N/2)''')
s=s.replace('''                int64_t off = (int64_t)m * N + n;
                if (mode == 1) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 += r0; v1 += r1;
                } else if (mode == 2) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 = r0 + v0 * gate[n]; v1 = r1 + v1 * gate[n + 1];
                }
                store2<OutT>(C + off, v0, v1);''','''                if (mode == 3) {
                    C[(int64_t)m * (N >> 1) + (n >> 1)] = (OutT)(silu_f(v0) * v1);
                    continue;
                }
                int64_t off = (int64_t)m * N + n;
                if (mode == 1) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 += r0; v1 += r1;
                } else if (mode == 2) {
                    float r0, r1; load2<OutT>(res + off, r0, r1);
                    v0 = r0 + v0 * gate[n]; v1 = r1 + v1 * gate[n + 1];
                }
                store2<OutT>(C + off, v0, v1);''')
open(p,'w').write(s)

# row gather kernels for weight interleaving
p='kernels/elementwise.cu'; s=open(p).read()
s=s.replace('''// Gather rows: out[i,:] = in[idx[i],:]  (f32)''','''// Gather rows of int8 [rows, K]: out[i,:] = in[idx[i],:]  (16B vectors; K % 16 == 0)
__global__ void k_gather_rows_i8(const int8_t* __restrict__ in, const int* __restrict__ idx, int8_t* __restrict__ out, int rows, int K) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t k16 = K / 16;
    if (i >= (int64_t)rows * k16) return;
    int64_t r = i / k16; int c = (i % k16) * 16;
    *reinterpret_cast<uint4*>(out + r * K + c) = *reinterpret_cast<const uint4*>(in + (int64_t)idx[r] * K + c);
}

// Gather rows: out[i,:] = in[idx[i],:]  (f32)''')
open(p,'w').write(s)

# ---------------- Rust side ----------------
p='src/ops.rs'; s=open(p).read()
s=s.replace('''/// int8 GEMM: out[M,N] = (aq[M,K] . wq[N,K]^T) * sa[m] * sw[n] (+bias) with epilogue.''','''/// AdaLN fused into the int8 quantizer: q = quant(rotate(LN(x) * (1 + scale)))
pub fn adaln_quant(dev: &Device, x: &Tensor, scale: &Tensor, eps: f32, out_q: &Tensor, out_s: &Tensor) -> Result<()> {
    let m = x.shape[0];
    let k: usize = x.shape[1..].iter().product();
    ensure!(x.dtype == DType::BF16 && k % 256 == 0);
    let smem = (k * 4) as u32;
    dev.launch("k_adaln_quant_bf16", (m as u32, 1, 1), (256, 1, 1), smem, &[Arg::Ptr(x.ptr), Arg::I32(m as i32), Arg::I32(k as i32), Arg::F32(eps), Arg::Ptr(scale.ptr), Arg::Ptr(out_q.ptr), Arg::Ptr(out_s.ptr)])
}

/// Gather int8 rows: out[i] = w[idx[i]]
pub fn gather_rows_i8(dev: &Device, w: &Tensor, idx: &Tensor, out: &Tensor) -> Result<()> {
    let k = w.shape[1];
    let rows = idx.numel();
    dev.launch_n("k_gather_rows_i8", rows * (k / 16), &[Arg::Ptr(w.ptr), Arg::Ptr(idx.ptr), Arg::Ptr(out.ptr), Arg::I32(rows as i32), Arg::I32(k as i32)])
}

/// int8 GEMM: out[M,N] = (aq[M,K] . wq[N,K]^T) * sa[m] * sw[n] (+bias) with epilogue.''')
s=s.replace('''    /// out = res + v * gate[n]
    AddResGated(&'a Tensor, &'a Tensor),
}''','''    /// out = res + v * gate[n]
    AddResGated(&'a Tensor, &'a Tensor),
    /// out[m][n/2] = silu(v[n]) * v[n+1] (weights interleaved gate/up; output width N/2)
    SwiGluPairs,
}''')
s=s.replace('''    ensure!(out.numel() == m * n, "gemm_i8: out shape");
    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
    };''','''    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
        Epi::SwiGluPairs => (3, 0u64, 0u64),
    };
    ensure!(out.numel() == if mode == 3 { m * n / 2 } else { m * n }, "gemm_i8: out shape");''')
s=s.replace('''    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
    };
    let kname = match out.dtype {
        DType::BF16 => "k_gemm_bf16_bf16",''','''    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
        Epi::SwiGluPairs => anyhow::bail!("gemm_bf16: SwiGluPairs unsupported"),
    };
    let kname = match out.dtype {
        DType::BF16 => "k_gemm_bf16_bf16",''')
open(p,'w').write(s)

p='src/weights.rs'; s=open(p).read()
s=s.replace('''pub fn expect_shape(''','''impl QLinear {
    /// Reorder rows [gate(0..H) | up(H..2H)] into interleaved pairs (gate_j, up_j) for the fused SwiGLU epilogue.
    pub fn interleave_gate_up(&self, dev: &Device) -> Result<QLinear> {
        let h = self.n / 2;
        let idx: Vec<i32> = (0..self.n).map(|r| if r % 2 == 0 { (r / 2) as i32 } else { (h + r / 2) as i32 }).collect();
        let idx_t = Tensor::from_buf(dev.upload(&idx)?, DType::F32, &[self.n]);
        let w = Tensor::new(dev, DType::I8, &[self.n, self.k])?;
        crate::ops::gather_rows_i8(dev, &self.w, &idx_t, &w)?;
        let scale = Tensor::new(dev, DType::F32, &[self.n])?;
        crate::ops::gather_rows_f32(dev, &self.scale.reshape(&[self.n, 1]), &idx_t, &scale.reshape(&[self.n, 1]))?;
        Ok(QLinear { w, scale, n: self.n, k: self.k })
    }
}

pub fn expect_shape(''')
open(p,'w').write(s)

p='src/dit.rs'; s=open(p).read()
s=s.replace('''                gate_up: l.qlinear(&format!("{p}.img_mlp.gate_up"))?,''','''                gate_up: l.qlinear(&format!("{p}.img_mlp.gate_up"))?.interleave_gate_up(&dev)?,''')
# target pass
s=s.replace('''            pr.time(dev, "adaln", || ops::adaln(dev, &x, &m.scale1, &m.scale1, 0, EPS, &hbuf))?;
            pr.time(dev, "quant", || ops::quant_rows(dev, &hbuf, QuantAct::None, None, 0.0, &xq, &xs))?;
            pr.time(dev, "gemm qkv",''','''            pr.time(dev, "adaln+quant", || ops::adaln_quant(dev, &x, &m.scale1, EPS, &xq, &xs))?;
            pr.time(dev, "gemm qkv",''')
s=s.replace('''            pr.time(dev, "adaln", || ops::adaln(dev, &x, &m.scale2, &m.scale2, 0, EPS, &hbuf))?;
            pr.time(dev, "quant", || ops::quant_rows(dev, &hbuf, QuantAct::None, None, 0.0, &xq, &xs))?;
            pr.time(dev, "gemm gate_up", || ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::Store, &gu))?;
            pr.time(dev, "quant swiglu", || ops::quant_rows(dev, &gu, QuantAct::SwiGlu, None, 0.0, &hq, &xs))?;''','''            pr.time(dev, "adaln+quant", || ops::adaln_quant(dev, &x, &m.scale2, EPS, &xq, &xs))?;
            pr.time(dev, "gemm gate_up", || ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::SwiGluPairs, &gu))?;
            pr.time(dev, "quant", || ops::quant_rows(dev, &gu, QuantAct::None, None, 0.0, &hq, &xs))?;''')
s=s.replace('''        let hbuf = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let xq = Tensor::new(dev, DType::I8, &[nt, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[nt])?;
        let qkv = Tensor::new(dev, DType::BF16, &[nt, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[nt, 2 * MLP])?;''','''        let hbuf = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let xq = Tensor::new(dev, DType::I8, &[nt, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[nt])?;
        let qkv = Tensor::new(dev, DType::BF16, &[nt, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[nt, MLP])?;''')
# prefix pass
s=s.replace('''            ops::adaln(dev, &x, scale1, scale1, p, EPS, &hbuf)?;
            ops::quant_rows(dev, &hbuf, QuantAct::None, None, 0.0, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.qkv.w, &b.qkv.scale, None, Epi::Store, &qkv)?;''','''            ops::adaln_quant(dev, &x, scale1, EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.qkv.w, &b.qkv.scale, None, Epi::Store, &qkv)?;''')
s=s.replace('''            ops::adaln(dev, &x, scale2, scale2, p, EPS, &hbuf)?;
            ops::quant_rows(dev, &hbuf, QuantAct::None, None, 0.0, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::Store, &gu)?;
            ops::quant_rows(dev, &gu, QuantAct::SwiGlu, None, 0.0, &hq, &xs)?;''','''            ops::adaln_quant(dev, &x, scale2, EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::SwiGluPairs, &gu)?;
            ops::quant_rows(dev, &gu, QuantAct::None, None, 0.0, &hq, &xs)?;''')
s=s.replace('''        let hbuf = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        let xq = Tensor::new(dev, DType::I8, &[p, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[p])?;
        let qkv = Tensor::new(dev, DType::BF16, &[p, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[p, 2 * MLP])?;''','''        let xq = Tensor::new(dev, DType::I8, &[p, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[p])?;
        let qkv = Tensor::new(dev, DType::BF16, &[p, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[p, MLP])?;''')
open(p,'w').write(s)
print("patched")
