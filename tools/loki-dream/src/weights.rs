//! Loading safetensors weights onto the GPU, including ComfyUI int8 ConvRot linears. Float weights stored as fp16 (some
//! fine-tunes are published that way) are converted to bf16 on the host as they are uploaded.
use crate::cuda::Device;
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Context, Result};
use std::sync::Arc;

/// An int8 ConvRot linear: weight [N, K] int8, per-row scale [N] f32.
pub struct QLinear {
    pub w: Tensor,
    pub scale: Tensor,
    pub n: usize,
    pub k: usize,
}

/// fp16 bytes -> bf16 bytes (round to nearest even), split across threads: a 10 GB checkpoint is ~5G values.
fn f16_to_bf16(src: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; src.len()];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16);
    let chunk = (src.len() / 2).div_ceil(threads).max(1 << 16) * 2;
    std::thread::scope(|s| {
        for (i, o) in src.chunks(chunk).zip(out.chunks_mut(chunk)) {
            s.spawn(move || {
                for (a, b) in i.chunks_exact(2).zip(o.chunks_exact_mut(2)) {
                    let v = half::f16::from_bits(u16::from_le_bytes([a[0], a[1]])).to_f32();
                    b.copy_from_slice(&half::bf16::from_f32(v).to_bits().to_le_bytes());
                }
            });
        }
    });
    out
}

pub struct Loader<'a> {
    pub st: &'a SafeTensors,
    pub dev: Arc<Device>,
    pub uploaded: usize,
}

impl<'a> Loader<'a> {
    pub fn new(st: &'a SafeTensors, dev: Arc<Device>) -> Loader<'a> {
        Loader { st, dev, uploaded: 0 }
    }

    fn upload_raw(&mut self, name: &str, dtype: DType, shape: Vec<usize>) -> Result<Tensor> {
        let bytes = self.st.bytes(name)?;
        let t = Tensor::new(&self.dev, dtype, &shape)?;
        ensure!(t.bytes() == bytes.len(), "tensor {name}: byte size mismatch {} vs {}", t.bytes(), bytes.len());
        self.dev.htod_at(t.ptr, bytes)?;
        self.uploaded += bytes.len();
        Ok(t)
    }

    /// A float tensor's bytes as bf16: as stored, or converted from fp16.
    fn bf16_bytes(&self, name: &str) -> Result<std::borrow::Cow<'a, [u8]>> {
        let st: &'a SafeTensors = self.st;
        let info = st.info(name)?;
        let bytes = st.bytes(name)?;
        match info.dtype.as_str() {
            "BF16" => Ok(std::borrow::Cow::Borrowed(bytes)),
            "F16" => Ok(std::borrow::Cow::Owned(f16_to_bf16(bytes))),
            other => bail!("tensor {name}: expected BF16 or F16, got {other}"),
        }
    }

    /// bf16 tensor (stored as bf16, or fp16 converted).
    pub fn bf16(&mut self, name: &str) -> Result<Tensor> {
        let shape = self.st.info(name)?.shape.clone();
        let bytes = self.bf16_bytes(name)?;
        let t = Tensor::new(&self.dev, DType::BF16, &shape)?;
        ensure!(t.bytes() == bytes.len(), "tensor {name}: byte size mismatch {} vs {}", t.bytes(), bytes.len());
        self.dev.htod_at(t.ptr, &bytes)?;
        self.uploaded += bytes.len();
        Ok(t)
    }
    /// Any float tensor converted to f32 on the host (used for biases and small vectors).
    pub fn f32(&mut self, name: &str) -> Result<Tensor> {
        let v = self.st.f32s(name)?;
        let shape = self.st.info(name)?.shape.clone();
        self.uploaded += v.len() * 4;
        Tensor::from_f32(&self.dev, &v, &shape)
    }
    pub fn f32_opt(&mut self, name: &str) -> Result<Option<Tensor>> {
        if self.st.has(name) {
            Ok(Some(self.f32(name)?))
        } else {
            Ok(None)
        }
    }

    /// ComfyUI int8 ConvRot linear (`<prefix>.weight`, `.weight_scale`, `.comfy_quant`).
    pub fn qlinear(&mut self, prefix: &str) -> Result<QLinear> {
        let meta = self.st.bytes(&format!("{prefix}.comfy_quant")).with_context(|| format!("{prefix} quant metadata"))?;
        let meta: serde_json::Value = serde_json::from_slice(meta)?;
        ensure!(meta["format"] == "int8_tensorwise", "{prefix}: unsupported quant format {}", meta["format"]);
        ensure!(meta["convrot"] == true && meta["convrot_groupsize"] == 256, "{prefix}: expected convrot groupsize 256");
        let winfo = self.st.info(&format!("{prefix}.weight"))?.clone();
        ensure!(winfo.dtype == "I8" && winfo.shape.len() == 2, "{prefix}: expected 2D I8 weight");
        let (n, k) = (winfo.shape[0], winfo.shape[1]);
        let w = self.upload_raw(&format!("{prefix}.weight"), DType::I8, vec![n, k])?;
        let sinfo = self.st.info(&format!("{prefix}.weight_scale"))?.clone();
        ensure!(sinfo.dtype == "F32" && sinfo.shape.iter().product::<usize>() == n, "{prefix}: bad weight_scale");
        let scale = self.upload_raw(&format!("{prefix}.weight_scale"), DType::F32, vec![n])?;
        Ok(QLinear { w, scale, n, k })
    }

    /// Several int8 linears with identical K concatenated along N (e.g. q|k|v or gate|up).
    pub fn qlinear_cat(&mut self, prefixes: &[&str]) -> Result<QLinear> {
        let mut n_total = 0;
        let mut k = 0;
        for p in prefixes {
            let meta: serde_json::Value = serde_json::from_slice(self.st.bytes(&format!("{p}.comfy_quant"))?)?;
            ensure!(meta["format"] == "int8_tensorwise" && meta["convrot"] == true, "{p}: unsupported quant format");
            let info = self.st.info(&format!("{p}.weight"))?;
            ensure!(info.dtype == "I8");
            if k == 0 {
                k = info.shape[1];
            }
            ensure!(info.shape[1] == k, "qlinear_cat: K mismatch");
            n_total += info.shape[0];
        }
        let w = Tensor::new(&self.dev, DType::I8, &[n_total, k])?;
        let scale = Tensor::new(&self.dev, DType::F32, &[n_total])?;
        let mut off = 0;
        for p in prefixes {
            let wb = self.st.bytes(&format!("{p}.weight"))?;
            self.dev.htod_at(w.ptr + (off * k) as u64, wb)?;
            let sb = self.st.bytes(&format!("{p}.weight_scale"))?;
            self.dev.htod_at(scale.ptr + (off * 4) as u64, sb)?;
            off += wb.len() / k;
            self.uploaded += wb.len() + sb.len();
        }
        Ok(QLinear { w, scale, n: n_total, k })
    }

    /// bf16 linears concatenated along N (bf16 or fp16 converted).
    pub fn bf16_cat(&mut self, names: &[&str]) -> Result<Tensor> {
        let mut n_total = 0;
        let mut k = 0;
        for name in names {
            let info = self.st.info(name)?;
            ensure!(info.dtype == "BF16" || info.dtype == "F16", "tensor {name}: expected BF16 or F16, got {}", info.dtype);
            let kk: usize = info.shape[1..].iter().product();
            if k == 0 {
                k = kk;
            }
            ensure!(kk == k);
            n_total += info.shape[0];
        }
        let t = Tensor::new(&self.dev, DType::BF16, &[n_total, k])?;
        let mut off = 0;
        for name in names {
            let b = self.bf16_bytes(name)?;
            self.dev.htod_at(t.ptr + (off * k * 2) as u64, &b)?;
            off += b.len() / (k * 2);
            self.uploaded += b.len();
        }
        Ok(t)
    }
    /// f32 vectors concatenated.
    pub fn f32_cat(&mut self, names: &[&str]) -> Result<Tensor> {
        let mut v = Vec::new();
        for n in names {
            v.extend(self.st.f32s(n)?);
        }
        let len = v.len();
        self.uploaded += len * 4;
        Tensor::from_f32(&self.dev, &v, &[len])
    }
}

impl QLinear {
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

pub fn expect_shape(t: &Tensor, shape: &[usize], what: &str) -> Result<()> {
    if t.shape != shape {
        bail!("{what}: expected shape {:?}, got {:?}", shape, t.shape);
    }
    Ok(())
}
