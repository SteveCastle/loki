//! GGUF block quants: CPU dequantization (embedding rows, tests), block repacking for the GPU kernels, and `Linear`,
//! a weight matrix kept quantized on the GPU (see kernels/quant.cu for the layouts).
use crate::cuda::{Arg, Device};
use crate::gguf::{Gguf, GgmlType};
use crate::ops::{self, Epi};
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Result};

fn f16(b: &[u8]) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
}

fn scale_min_k4(j: usize, q: &[u8]) -> (f32, f32) {
    if j < 4 {
        ((q[j] & 63) as f32, (q[j + 4] & 63) as f32)
    } else {
        (((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f32, ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f32)
    }
}

/// Dequantize one block (file layout) into `y` (256 or 32 values).
fn dequant_block(ty: GgmlType, b: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::Q4K | GgmlType::Q5K => {
            let d = f16(&b[0..2]);
            let dmin = f16(&b[2..4]);
            let sc = &b[4..16];
            let (qh, qs) = if ty == GgmlType::Q5K { (Some(&b[16..48]), &b[48..176]) } else { (None, &b[16..144]) };
            for j in 0..4 {
                let (s1, m1) = scale_min_k4(2 * j, sc);
                let (s2, m2) = scale_min_k4(2 * j + 1, sc);
                for l in 0..32 {
                    let q = qs[j * 32 + l];
                    let (mut lo, mut hi) = ((q & 0xF) as f32, (q >> 4) as f32);
                    if let Some(qh) = qh {
                        if qh[l] & (1 << (2 * j)) != 0 {
                            lo += 16.0;
                        }
                        if qh[l] & (1 << (2 * j + 1)) != 0 {
                            hi += 16.0;
                        }
                    }
                    y[j * 64 + l] = d * s1 * lo - dmin * m1;
                    y[j * 64 + 32 + l] = d * s2 * hi - dmin * m2;
                }
            }
        }
        GgmlType::Q6K => {
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let d = f16(&b[208..210]);
            for n in 0..2 {
                for l in 0..32 {
                    let is = l / 16;
                    let (a, c, h) = (ql[n * 64 + l], ql[n * 64 + 32 + l], qh[n * 32 + l]);
                    let q1 = ((a & 0xF) | ((h & 3) << 4)) as i32 - 32;
                    let q2 = ((c & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((a >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((c >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                    let s = |i: usize| sc[n * 8 + i] as i8 as f32;
                    y[n * 128 + l] = d * s(is) * q1 as f32;
                    y[n * 128 + 32 + l] = d * s(is + 2) * q2 as f32;
                    y[n * 128 + 64 + l] = d * s(is + 4) * q3 as f32;
                    y[n * 128 + 96 + l] = d * s(is + 6) * q4 as f32;
                }
            }
        }
        GgmlType::Q8_0 => {
            let d = f16(&b[0..2]);
            for i in 0..32 {
                y[i] = d * b[2 + i] as i8 as f32;
            }
        }
        _ => unreachable!(),
    }
}

/// Dequantize `numel` values of a tensor's raw data on the CPU.
pub fn dequant_cpu(ty: GgmlType, data: &[u8], numel: usize) -> Result<Vec<f32>> {
    let (be, bb) = ty.block()?;
    let mut out = vec![0f32; numel];
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = f16(c);
            }
        }
        GgmlType::BF16 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(2)) {
                *o = half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32();
            }
        }
        _ => {
            for (i, blk) in data.chunks_exact(bb).enumerate().take(numel / be) {
                dequant_block(ty, blk, &mut out[i * be..(i + 1) * be]);
            }
        }
    }
    Ok(out)
}

/// One row (`cols` values) of a 2-D tensor, dequantized.
pub fn dequant_row(ty: GgmlType, data: &[u8], row: usize, cols: usize) -> Result<Vec<f32>> {
    let (be, bb) = ty.block()?;
    let rb = cols / be * bb;
    dequant_cpu(ty, &data[row * rb..(row + 1) * rb], cols)
}

/// Weight kinds as kept on the GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Q4K,
    Q5K,
    Q6K,
    Q80,
    Bf16,
}

impl Kind {
    fn tag(self) -> &'static str {
        match self {
            Kind::Q4K => "q4k",
            Kind::Q5K => "q5k",
            Kind::Q6K => "q6k",
            Kind::Q80 => "q80",
            Kind::Bf16 => "bf16",
        }
    }
}

/// Repack file blocks into the GPU layout (Q6_K 210 -> 224 bytes, Q8_0 34 -> 36 bytes; others verbatim).
fn repack(ty: GgmlType, data: &[u8]) -> Vec<u8> {
    match ty {
        GgmlType::Q6K => {
            let mut out = vec![0u8; data.len() / 210 * 224];
            for (o, b) in out.chunks_exact_mut(224).zip(data.chunks_exact(210)) {
                o[..210].copy_from_slice(b);
            }
            out
        }
        GgmlType::Q8_0 => {
            let mut out = vec![0u8; data.len() / 34 * 36];
            for (o, b) in out.chunks_exact_mut(36).zip(data.chunks_exact(34)) {
                o[0..2].copy_from_slice(&b[0..2]);
                o[4..36].copy_from_slice(&b[2..34]);
            }
            out
        }
        _ => data.to_vec(),
    }
}

/// q8 activations for the GEMV kernels plus the f32 rows they came from.
pub struct Act<'a> {
    pub x: &'a Tensor,  // f32 [M, K]
    pub xq: &'a Tensor, // i8 [M, K]
    pub xd: &'a Tensor, // f32 [M, K/32]
}

/// Quantize f32 rows x [M, K] into q8 (xq int8 [M, K], xd f32 [M, K/32]).
pub fn quant_q8(dev: &Device, x: &Tensor, m: usize, k: usize, xq: &Tensor, xd: &Tensor) -> Result<()> {
    ensure!(k % 32 == 0 && xq.numel() >= m * k && xd.numel() >= m * k / 32);
    let nb = m * k / 32;
    dev.launch_n("k_quant_q8", nb * 32, &[Arg::Ptr(x.ptr), Arg::Ptr(xq.ptr), Arg::Ptr(xd.ptr), Arg::I64(nb as i64)])
}

/// A weight matrix W [N, K] (row n = output n), quantized or bf16, on the GPU.
pub struct Linear {
    pub kind: Kind,
    pub w: Tensor,
    pub n: usize,
    pub k: usize,
}

impl Linear {
    pub fn load(dev: &Device, g: &Gguf, name: &str) -> Result<Linear> {
        let t = g.tensor(name)?;
        ensure!(t.dims.len() >= 2, "{name} is not a matrix");
        let (n, k) = t.rows_cols();
        let data = g.data(t);
        let (kind, w) = match t.ty {
            GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Q8_0 => {
                let kind = match t.ty {
                    GgmlType::Q4K => Kind::Q4K,
                    GgmlType::Q5K => Kind::Q5K,
                    GgmlType::Q6K => Kind::Q6K,
                    _ => Kind::Q80,
                };
                let bytes = repack(t.ty, data);
                (kind, Tensor::from_buf(dev.upload(&bytes)?, DType::U8, &[bytes.len()]))
            }
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                let v = dequant_cpu(t.ty, data, n * k)?;
                let b: Vec<u16> = v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
                (Kind::Bf16, Tensor::from_bf16(dev, &b, &[n, k])?)
            }
            other => bail!("{name}: tensor type {} is not supported (use a Q4_K/Q5_K/Q6_K/Q8_0/F16/BF16 GGUF)", other.name()),
        };
        if kind != Kind::Bf16 {
            ensure!(k % 256 == 0 || (kind == Kind::Q80 && k % 32 == 0), "{name}: K={k} is not a multiple of the block size");
        }
        Ok(Linear { kind, w, n, k })
    }

    pub fn bytes(&self) -> usize {
        self.w.bytes()
    }

    /// out[m, :] (=|+=) a[m, :] . W^T for m < M <= 8 (decoding).
    pub fn gemv(&self, dev: &Device, a: &Act, m: usize, out: &Tensor, add: bool) -> Result<()> {
        ensure!(m >= 1 && m <= 8, "gemv: M={m}");
        ensure!(out.numel() >= m * self.n);
        if self.kind == Kind::Bf16 {
            let grid = ((self.n + 7) / 8) as u32;
            return dev.launch("k_gemv_bf16", (grid, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(self.w.ptr), Arg::Ptr(a.x.ptr), Arg::Ptr(out.ptr), Arg::I32(self.n as i32), Arg::I32(self.k as i32), Arg::I32(m as i32), Arg::I32(add as i32)]);
        }
        let mm = if m <= 1 { 1 } else if m <= 2 { 2 } else if m <= 4 { 4 } else { 8 };
        let name = format!("k_gemv_{}_{mm}", self.kind.tag());
        // 8 warps per block, 2 rows per warp
        let grid = ((self.n + 15) / 16) as u32;
        dev.launch(&name, (grid, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(self.w.ptr), Arg::Ptr(a.xq.ptr), Arg::Ptr(a.xd.ptr), Arg::Ptr(out.ptr), Arg::I32(self.n as i32), Arg::I32(self.k as i32), Arg::I32(m as i32), Arg::I32(add as i32)])
    }

    /// The weight as bf16 [N, K]: itself, or dequantized into `scratch` (>= N*K bf16).
    pub fn bf16<'a>(&'a self, dev: &Device, scratch: &Tensor) -> Result<Tensor> {
        if self.kind == Kind::Bf16 {
            return Ok(self.w.clone());
        }
        ensure!(scratch.dtype == DType::BF16 && scratch.numel() >= self.n * self.k, "dequant scratch too small");
        let units = self.n * self.k / 8;
        let name = format!("k_dequant_{}", self.kind.tag());
        dev.launch_n(&name, units, &[Arg::Ptr(self.w.ptr), Arg::Ptr(scratch.ptr), Arg::I32(self.n as i32), Arg::I32(self.k as i32)])?;
        Ok(Tensor { buf: scratch.buf.clone(), ptr: scratch.ptr, dtype: DType::BF16, shape: vec![self.n, self.k] })
    }

    /// out[M, N] = a[M, K] (bf16) . W^T with an epilogue (prefill): dequantize, then the bf16 tensor-core GEMM.
    pub fn gemm(&self, dev: &Device, a: &Tensor, out: &Tensor, epi: Epi, scratch: &Tensor) -> Result<()> {
        let w = self.bf16(dev, scratch)?;
        ops::gemm_bf16(dev, a, &w, None, ops::Act::None, epi, out)
    }
}
