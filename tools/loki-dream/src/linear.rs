//! Linear layers from any of the weight formats the supported checkpoints ship in, in one of the engine's two compute
//! formats: int8 ConvRot (W8A8, Hadamard-256 rotated rows with per-row scales; the int8 GEMM) or bf16.
//!
//! Stored formats: bf16, fp16, fp32, ComfyUI "scaled fp8" (e4m3 with a per-tensor `weight_scale`, the quantization recorded
//! in a `.comfy_quant` tensor or the header's `_quantization_metadata`; plain fp8 casts without a scale too), ComfyUI
//! NVFP4 (e2m1 pairs with blocked fp8 scales per 16 values and a per-tensor `weight_scale_2`) and ComfyUI int8 ConvRot.
//! Float, fp8 and NVFP4 weights are converted on the GPU as they load: dequantized to bf16, then (for int8) rotated and
//! quantized per row by the same kernel that quantizes activations, so both sides of the GEMM agree on the rotation. An
//! int8 ConvRot weight stays int8 whatever is asked (its rotation cannot be undone cheaply). Layers a quantizer marked
//! `full_precision_matrix_mult` (precision-sensitive) stay bf16. mxfp8 / int4 are refused.
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, Epi, QuantAct};
use crate::tensor::{DType, Tensor};
use crate::weights::{Loader, QLinear};
use anyhow::{bail, ensure, Context, Result};

/// Compute format of a linear.
pub enum Lin {
    I8(QLinear),
    Bf16(Tensor),
}

impl Lin {
    pub fn n(&self) -> usize {
        match self {
            Lin::I8(q) => q.n,
            Lin::Bf16(t) => t.shape[0],
        }
    }
    pub fn k(&self) -> usize {
        match self {
            Lin::I8(q) => q.k,
            Lin::Bf16(t) => t.shape[1],
        }
    }
    pub fn is_int8(&self) -> bool {
        matches!(self, Lin::I8(_))
    }
    pub fn bytes(&self) -> usize {
        match self {
            Lin::I8(q) => q.w.bytes() + q.scale.bytes(),
            Lin::Bf16(t) => t.bytes(),
        }
    }

    /// out[M, N] (bf16 or f32) = epi(x[M, K] (bf16) . W^T (+ bias f32 [N])). int8 weights quantize `x` on the fly.
    pub fn forward(&self, dev: &Device, x: &Tensor, bias: Option<&Tensor>, epi: Epi, out: &Tensor) -> Result<()> {
        match self {
            Lin::Bf16(w) => ops::gemm_bf16(dev, x, w, bias, Act::None, epi, out),
            Lin::I8(q) => {
                let m = x.shape[0];
                let xq = Tensor::new(dev, DType::I8, &[m, q.k])?;
                let xs = Tensor::new(dev, DType::F32, &[m])?;
                ops::quant_rows(dev, x, QuantAct::None, None, 0.0, &xq, &xs)?;
                ops::gemm_i8(dev, &xq, &xs, &q.w, &q.scale, bias, epi, out)
            }
        }
    }

    /// out = W . swiglu(gu) for gu = [gate | up] rows (bf16 [M, 2F]); int8 weights fold the SwiGLU into the quantizer.
    pub fn forward_swiglu(&self, dev: &Device, gu: &Tensor, epi: Epi, out: &Tensor) -> Result<()> {
        let m = gu.shape[0];
        let f = gu.numel() / m / 2;
        match self {
            Lin::Bf16(w) => {
                let a = Tensor::new(dev, DType::BF16, &[m, f])?;
                ops::swiglu_round(dev, gu, &a)?;
                ops::gemm_bf16(dev, &a, w, None, Act::None, epi, out)
            }
            Lin::I8(q) => {
                let xq = Tensor::new(dev, DType::I8, &[m, f])?;
                let xs = Tensor::new(dev, DType::F32, &[m])?;
                ops::quant_rows(dev, gu, QuantAct::SwiGlu, None, 0.0, &xq, &xs)?;
                ops::gemm_i8(dev, &xq, &xs, &q.w, &q.scale, None, epi, out)
            }
        }
    }

    /// Like `forward` with an already quantized input (int8 weights only; bf16 weights use `xb`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_q(&self, dev: &Device, xq: Option<(&Tensor, &Tensor)>, xb: Option<&Tensor>, bias: Option<&Tensor>, epi: Epi, out: &Tensor) -> Result<()> {
        match self {
            Lin::Bf16(w) => ops::gemm_bf16(dev, xb.context("bf16 linear without bf16 input")?, w, bias, Act::None, epi, out),
            Lin::I8(q) => {
                let (aq, sa) = xq.context("int8 linear without quantized input")?;
                ops::gemm_i8(dev, aq, sa, &q.w, &q.scale, bias, epi, out)
            }
        }
    }
}

/// What the caller wants a linear in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    Int8,
    Bf16,
}

/// How a weight is stored.
#[derive(Clone, Debug, PartialEq)]
pub enum Stored {
    Float(String),
    /// e4m3 with a per-tensor scale
    Fp8(f32),
    Int8ConvRot,
    /// ComfyUI NVFP4: e2m1 pairs, blocked fp8 scales per 16 values, a per-tensor scale
    Nvfp4(f32),
}

/// The quantization record of `prefix` (from `prefix.comfy_quant` or the header's `_quantization_metadata`).
fn quant_meta(l: &Loader, prefix: &str) -> Result<Option<serde_json::Value>> {
    let cq = format!("{prefix}.comfy_quant");
    if l.st.has(&cq) {
        return Ok(Some(serde_json::from_slice(l.st.bytes(&cq)?).with_context(|| format!("{cq}"))?));
    }
    if let Some(m) = l.st.metadata.get("_quantization_metadata") {
        let v: serde_json::Value = serde_json::from_str(m).context("_quantization_metadata")?;
        if let Some(layers) = v.get("layers") {
            for p in crate::arch::PREFIXES {
                if let Some(bare) = prefix.strip_prefix(p) {
                    if let Some(x) = layers.get(bare) {
                        return Ok(Some(x.clone()));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// A scalar f32 tensor (or [1]).
fn scalar(l: &Loader, name: &str) -> Result<f32> {
    let v = l.st.f32s(name)?;
    ensure!(v.len() == 1, "{name}: expected a per-tensor scale, got {} values (block-scaled formats are not supported)", v.len());
    Ok(v[0])
}

/// How `prefix.weight` is stored, and whether its quantizer wants it computed in full precision.
pub fn stored(l: &Loader, prefix: &str) -> Result<(Stored, bool)> {
    let wname = format!("{prefix}.weight");
    let info = l.st.info(&wname)?;
    let meta = quant_meta(l, prefix)?;
    let full = meta.as_ref().and_then(|m| m.get("full_precision_matrix_mult")).and_then(|v| v.as_bool()).unwrap_or(false);
    if let Some(m) = &meta {
        let fmt = m.get("format").and_then(|f| f.as_str()).unwrap_or("");
        match fmt {
            "int8_tensorwise" => {
                ensure!(m["convrot"] == true && m["convrot_groupsize"] == 256, "{prefix}: int8 without ConvRot (groupsize 256) is not supported");
                return Ok((Stored::Int8ConvRot, full));
            }
            "float8_e4m3fn" => {}
            "nvfp4" => return Ok((Stored::Nvfp4(scalar(l, &format!("{prefix}.weight_scale_2"))?), full)),
            "" => {}
            other => bail!("{prefix}: weight format {other} is not supported (bf16, fp16, fp8 e4m3 and int8 ConvRot are)"),
        }
    }
    Ok(match info.dtype.as_str() {
        "BF16" | "F16" | "F32" => (Stored::Float(info.dtype.clone()), full),
        "F8_E4M3" => {
            let scale = [format!("{prefix}.weight_scale"), format!("{prefix}.scale_weight")].into_iter().find(|n| l.st.has(n));
            (Stored::Fp8(match scale { Some(n) => scalar(l, &n)?, None => 1.0 }), full)
        }
        "I8" => (Stored::Int8ConvRot, full),
        "U8" if l.st.has(&format!("{prefix}.weight_scale_2")) => (Stored::Nvfp4(scalar(l, &format!("{prefix}.weight_scale_2"))?), full),
        other => bail!("{prefix}: weight dtype {other} is not supported (bf16, fp16, fp8 e4m3 and int8 ConvRot are)"),
    })
}

/// `prefix.weight` dequantized to a bf16 device tensor [N, K] (2D; higher-rank weights are flattened after dim 0).
pub fn weight_bf16(l: &mut Loader, prefix: &str) -> Result<Tensor> {
    let (st, _) = stored(l, prefix)?;
    let wname = format!("{prefix}.weight");
    let info = l.st.info(&wname)?.clone();
    let n = info.shape[0];
    let mut k: usize = info.shape[1..].iter().product::<usize>().max(1);
    if matches!(st, Stored::Nvfp4(_)) {
        k *= 2;
    }
    let dev = l.dev.clone();
    let out = Tensor::new(&dev, DType::BF16, &[n, k])?;
    let bytes = l.st.bytes(&wname)?;
    l.uploaded += bytes.len();
    match st {
        Stored::Float(dt) if dt == "BF16" => dev.htod_at(out.ptr, bytes)?,
        Stored::Float(dt) => {
            let raw = Tensor::new(&dev, DType::U8, &[bytes.len()])?;
            dev.htod_at(raw.ptr, bytes)?;
            let kern = if dt == "F16" { "k_f16_to_bf16" } else { "k_f32_to_bf16" };
            dev.launch_n(kern, n * k, &[Arg::Ptr(raw.ptr), Arg::Ptr(out.ptr), Arg::I64((n * k) as i64)])?;
        }
        Stored::Fp8(scale) => {
            let raw = Tensor::new(&dev, DType::U8, &[bytes.len()])?;
            dev.htod_at(raw.ptr, bytes)?;
            dev.launch_n("k_fp8_to_bf16", n * k, &[Arg::Ptr(raw.ptr), Arg::F32(scale), Arg::Ptr(out.ptr), Arg::I64((n * k) as i64)])?;
        }
        Stored::Nvfp4(g) => {
            let sname = format!("{prefix}.weight_scale");
            let sinfo = l.st.info(&sname)?.clone();
            ensure!(sinfo.dtype == "F8_E4M3" && sinfo.shape.len() == 2 && sinfo.shape[0] >= n && sinfo.shape[1] * 16 >= k && sinfo.shape[0] % 128 == 0 && sinfo.shape[1] % 4 == 0, "{prefix}: unexpected nvfp4 block scales {:?}", sinfo.shape);
            let raw = Tensor::new(&dev, DType::U8, &[bytes.len()])?;
            dev.htod_at(raw.ptr, bytes)?;
            let sb = l.st.bytes(&sname)?;
            let sc = Tensor::new(&dev, DType::U8, &[sb.len()])?;
            dev.htod_at(sc.ptr, sb)?;
            l.uploaded += sb.len();
            dev.launch_n("k_nvfp4_to_bf16", n * k, &[Arg::Ptr(raw.ptr), Arg::Ptr(sc.ptr), Arg::I32(sinfo.shape[1] as i32), Arg::F32(g), Arg::Ptr(out.ptr), Arg::I32(n as i32), Arg::I32(k as i32), Arg::I32(1) /* comfy/float.py packs the first value into the high nibble */])?;
        }
        Stored::Int8ConvRot => bail!("{prefix}: an int8 ConvRot weight cannot be read as bf16"),
    }
    Ok(out)
}

/// Rows of bf16 weights [N, K] -> int8 ConvRot.
pub fn quantize(dev: &Device, w: &Tensor) -> Result<QLinear> {
    let (n, k) = (w.shape[0], w.shape[1]);
    let q = Tensor::new(dev, DType::I8, &[n, k])?;
    let s = Tensor::new(dev, DType::F32, &[n])?;
    ops::quant_rows(dev, w, QuantAct::None, None, 0.0, &q, &s)?;
    Ok(QLinear { w: q, scale: s, n, k })
}

/// Whether a K can take the int8 path (Hadamard groups of 256, one row in shared memory).
pub fn int8_ok(k: usize) -> bool {
    k % 256 == 0 && k <= 24576
}

/// One linear (`prefix.weight`) in the wanted format.
pub fn lin(l: &mut Loader, prefix: &str, want: Want) -> Result<Lin> {
    lin_cat(l, &[prefix], want)
}

/// Several linears with the same K concatenated along N (q|k|v, gate|up, ...).
pub fn lin_cat(l: &mut Loader, prefixes: &[&str], want: Want) -> Result<Lin> {
    let mut kinds = Vec::with_capacity(prefixes.len());
    for p in prefixes {
        kinds.push(stored(l, p)?);
    }
    let any_i8 = kinds.iter().any(|(s, _)| *s == Stored::Int8ConvRot);
    if any_i8 {
        ensure!(kinds.iter().all(|(s, _)| *s == Stored::Int8ConvRot), "{}: mixed int8 and float parts cannot be fused", prefixes.join(", "));
        return Ok(Lin::I8(if prefixes.len() == 1 { l.qlinear(prefixes[0])? } else { l.qlinear_cat(prefixes)? }));
    }
    let full = kinds.iter().any(|(_, f)| *f);
    let parts: Vec<Tensor> = prefixes.iter().map(|p| weight_bf16(l, p)).collect::<Result<_>>()?;
    let k = parts[0].shape[1];
    ensure!(parts.iter().all(|t| t.shape[1] == k), "{}: K mismatch", prefixes.join(", "));
    let w = if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        let n: usize = parts.iter().map(|t| t.shape[0]).sum();
        let w = Tensor::new(&l.dev, DType::BF16, &[n, k])?;
        let mut off = 0;
        for t in &parts {
            l.dev.dtod(w.ptr + off as u64, t.ptr, t.bytes())?;
            off += t.bytes();
        }
        w
    };
    if want == Want::Int8 && !full && int8_ok(k) {
        Ok(Lin::I8(quantize(&l.dev, &w)?))
    } else {
        Ok(Lin::Bf16(w))
    }
}

/// An optional bias as f32 (`prefix.bias`).
pub fn bias(l: &mut Loader, prefix: &str) -> Result<Option<Tensor>> {
    l.f32_opt(&format!("{prefix}.bias"))
}
/// Biases of fused linears, concatenated (zeros where a part has none); None if no part has one.
pub fn bias_cat(l: &mut Loader, prefixes: &[&str], widths: &[usize]) -> Result<Option<Tensor>> {
    if !prefixes.iter().any(|p| l.st.has(&format!("{p}.bias"))) {
        return Ok(None);
    }
    let mut v = Vec::new();
    for (p, &w) in prefixes.iter().zip(widths) {
        let n = format!("{p}.bias");
        if l.st.has(&n) {
            v.extend(l.st.f32s(&n)?);
        } else {
            v.extend(std::iter::repeat(0f32).take(w));
        }
    }
    let len = v.len();
    Ok(Some(Tensor::from_f32(&l.dev, &v, &[len])?))
}

/// A small float tensor (norm weights, embeddings ...) as bf16, whatever its stored float type (fp8 included).
pub fn small_bf16(l: &mut Loader, name: &str) -> Result<Tensor> {
    let info = l.st.info(name)?.clone();
    if info.dtype == "BF16" {
        return l.bf16(name);
    }
    let v: Vec<u16> = if info.dtype == "F8_E4M3" {
        l.st.bytes(name)?.iter().map(|&b| crate::tensor::bf16_bits(fp8_e4m3_to_f32(b))).collect()
    } else {
        l.st.f32s(name)?.iter().map(|&x| crate::tensor::bf16_bits(x)).collect()
    };
    l.uploaded += v.len() * 2;
    Tensor::from_bf16(&l.dev, &v, &info.shape)
}

/// Host-side e4m3fn decode.
pub fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 7) as f32;
    if e == 0xF && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fp8_decode() {
        assert_eq!(fp8_e4m3_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3_to_f32(0xB8), -1.0);
        assert_eq!(fp8_e4m3_to_f32(0x7E), 448.0);
        assert_eq!(fp8_e4m3_to_f32(0x01), 2f32.powi(-9));
    }
}
