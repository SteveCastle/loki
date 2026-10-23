//! Loading the SAM 3.1 safetensors (fp16 as published by Comfy-Org) onto the GPU.
//!
//! Matrices go up as bf16 (converted on the device), biases / norm weights / small tables as f32. Linear layers are
//! `[N, K]` row-major (PyTorch layout), which is what the bf16 GEMM consumes directly.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Context, Result};
use std::cell::Cell;
use std::path::Path;
use std::sync::Arc;

/// y = x W^T + b with W bf16 [N, K], b f32 [N].
#[derive(Clone)]
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub n: usize,
    pub k: usize,
}

/// LayerNorm affine parameters (f32).
#[derive(Clone)]
pub struct Norm {
    pub w: Tensor,
    pub b: Tensor,
    pub eps: f32,
}

/// Small host-side linear layer (f32), for heads whose outputs are a handful of numbers.
#[derive(Clone)]
pub struct HostLinear {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub n: usize,
    pub k: usize,
}
impl HostLinear {
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let m = x.len() / self.k;
        let mut out = vec![0f32; m * self.n];
        for r in 0..m {
            let xr = &x[r * self.k..(r + 1) * self.k];
            for j in 0..self.n {
                let wr = &self.w[j * self.k..(j + 1) * self.k];
                let mut acc = 0f32;
                for i in 0..self.k {
                    acc += xr[i] * wr[i];
                }
                out[r * self.n + j] = acc + self.b[j];
            }
        }
        out
    }
}
/// Host MLP: relu between layers, none after the last.
#[derive(Clone)]
pub struct HostMlp(pub Vec<HostLinear>);
impl HostMlp {
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        for (i, l) in self.0.iter().enumerate() {
            h = l.apply(&h);
            if i + 1 < self.0.len() {
                for v in h.iter_mut() {
                    *v = v.max(0.0);
                }
            }
        }
        h
    }
}

pub struct Weights {
    pub st: SafeTensors,
    pub dev: Arc<Device>,
    pub uploaded: Cell<usize>,
}

impl Weights {
    pub fn open(path: &Path, dev: Arc<Device>) -> Result<Weights> {
        let st = SafeTensors::open(path)?;
        Ok(Weights { st, dev, uploaded: Cell::new(0) })
    }
    pub fn has(&self, name: &str) -> bool {
        self.st.has(name)
    }
    pub fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.st.info(name)?.shape.clone())
    }
    /// Any float tensor as host f32.
    pub fn host(&self, name: &str) -> Result<Vec<f32>> {
        self.st.f32s(name)
    }
    /// Raw fp16 tensor uploaded and converted to bf16 on the device, original shape.
    pub fn bf16(&self, name: &str) -> Result<Tensor> {
        let info = self.st.info(name)?.clone();
        let n: usize = info.shape.iter().product();
        let out = Tensor::new(&self.dev, DType::BF16, &info.shape)?;
        match info.dtype.as_str() {
            "F16" => {
                let tmp = Tensor::new(&self.dev, DType::F16, &info.shape)?;
                self.dev.htod_at(tmp.ptr, self.st.bytes(name)?)?;
                self.dev.launch_n("k_f16_to_bf16", n, &[Arg::Ptr(tmp.ptr), Arg::Ptr(out.ptr), Arg::I64(n as i64)])?;
            }
            "BF16" => self.dev.htod_at(out.ptr, self.st.bytes(name)?)?,
            _ => {
                let v = self.st.f32s(name)?;
                let b: Vec<u16> = v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
                self.dev.htod_at(out.ptr, as_bytes(&b))?;
            }
        }
        self.uploaded.set(self.uploaded.get() + n * 2);
        Ok(out)
    }
    /// Float tensor as device f32.
    pub fn f32(&self, name: &str) -> Result<Tensor> {
        let v = self.st.f32s(name)?;
        let shape = self.st.info(name)?.shape.clone();
        self.uploaded.set(self.uploaded.get() + v.len() * 4);
        Tensor::from_f32(&self.dev, &v, &shape)
    }
    pub fn linear(&self, prefix: &str) -> Result<Linear> {
        let wname = format!("{prefix}.weight");
        let shape = self.shape(&wname).with_context(|| format!("linear {prefix}"))?;
        ensure!(shape.len() == 2, "{prefix}: expected a 2-D weight, got {shape:?}");
        let w = self.bf16(&wname)?;
        let bname = format!("{prefix}.bias");
        let b = if self.has(&bname) { Some(self.f32(&bname)?) } else { None };
        Ok(Linear { w, b, n: shape[0], k: shape[1] })
    }
    /// One of the q/k/v slices of a fused `in_proj_weight [3E, E]` (`which` 0, 1, 2), or all three (`None`).
    pub fn in_proj(&self, prefix: &str, which: Option<usize>) -> Result<Linear> {
        let w = self.host(&format!("{prefix}.in_proj_weight"))?;
        let b = self.host(&format!("{prefix}.in_proj_bias"))?;
        let shape = self.shape(&format!("{prefix}.in_proj_weight"))?;
        let (n3, k) = (shape[0], shape[1]);
        let e = n3 / 3;
        let (r0, r1) = match which {
            Some(i) => (i * e, (i + 1) * e),
            None => (0, n3),
        };
        self.linear_from_host(&w[r0 * k..r1 * k], Some(&b[r0..r1]), r1 - r0, k)
    }
    /// Several linears with the same K stacked along N (e.g. separate q/k projections fused into one GEMM).
    pub fn linear_cat(&self, prefixes: &[&str]) -> Result<Linear> {
        let mut w = Vec::new();
        let mut b = Vec::new();
        let mut n = 0;
        let mut k = 0;
        for p in prefixes {
            let s = self.shape(&format!("{p}.weight"))?;
            if k == 0 {
                k = s[1];
            }
            ensure!(s[1] == k, "linear_cat: K mismatch at {p}");
            n += s[0];
            w.extend(self.host(&format!("{p}.weight"))?);
            b.extend(self.host(&format!("{p}.bias"))?);
        }
        self.linear_from_host(&w, Some(&b), n, k)
    }
    pub fn linear_from_host(&self, w: &[f32], b: Option<&[f32]>, n: usize, k: usize) -> Result<Linear> {
        ensure!(w.len() == n * k, "linear_from_host: {} != {n}x{k}", w.len());
        let wb: Vec<u16> = w.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
        let wt = Tensor::from_bf16(&self.dev, &wb, &[n, k])?;
        let bt = match b {
            Some(b) => Some(Tensor::from_f32(&self.dev, b, &[n])?),
            None => None,
        };
        self.uploaded.set(self.uploaded.get() + n * k * 2);
        Ok(Linear { w: wt, b: bt, n, k })
    }
    pub fn norm(&self, prefix: &str, eps: f32) -> Result<Norm> {
        Ok(Norm { w: self.f32(&format!("{prefix}.weight"))?, b: self.f32(&format!("{prefix}.bias"))?, eps })
    }
    pub fn host_linear(&self, prefix: &str) -> Result<HostLinear> {
        let s = self.shape(&format!("{prefix}.weight"))?;
        let b = if self.has(&format!("{prefix}.bias")) { self.host(&format!("{prefix}.bias"))? } else { vec![0.0; s[0]] };
        Ok(HostLinear { w: self.host(&format!("{prefix}.weight"))?, b, n: s[0], k: s[1] })
    }
    pub fn host_mlp(&self, prefix: &str, layers: usize) -> Result<HostMlp> {
        let mut v = Vec::new();
        for i in 0..layers {
            v.push(self.host_linear(&format!("{prefix}.layers.{i}"))?);
        }
        Ok(HostMlp(v))
    }
    /// Conv2d weight [Co, Ci, KH, KW] -> implicit-GEMM layout bf16 [Co, Kpad] with k = (ky*KW + kx)*Ci + ci,
    /// Kpad rounded up to a multiple of 32.
    pub fn conv(&self, prefix: &str) -> Result<Conv> {
        let wname = format!("{prefix}.weight");
        let s = self.shape(&wname)?;
        ensure!(s.len() == 4, "{prefix}: conv weight must be 4-D");
        let (co, ci, kh, kw) = (s[0], s[1], s[2], s[3]);
        let w = self.host(&wname)?;
        let k = kh * kw * ci;
        let kpad = (k + 31) / 32 * 32;
        let mut g = vec![0f32; co * kpad];
        for o in 0..co {
            for c in 0..ci {
                for y in 0..kh {
                    for x in 0..kw {
                        g[o * kpad + (y * kw + x) * ci + c] = w[((o * ci + c) * kh + y) * kw + x];
                    }
                }
            }
        }
        let gb: Vec<u16> = g.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
        let bname = format!("{prefix}.bias");
        let b = if self.has(&bname) { Some(self.f32(&bname)?) } else { None };
        Ok(Conv { w: Tensor::from_bf16(&self.dev, &gb, &[co, kpad])?, b, co, ci, kh, kw, kpad })
    }
    /// ConvTranspose2d(k=2, s=2) weight [Ci, Co, 2, 2] as a GEMM producing (dy, dx, co) per input pixel:
    /// bf16 [4*Co, Ci] and bias repeated 4 times.
    pub fn conv_t2(&self, prefix: &str) -> Result<Linear> {
        let s = self.shape(&format!("{prefix}.weight"))?;
        ensure!(s.len() == 4 && s[2] == 2 && s[3] == 2, "{prefix}: expected ConvTranspose2d k=2 weight");
        let (ci, co) = (s[0], s[1]);
        let w = self.host(&format!("{prefix}.weight"))?;
        let b = self.host(&format!("{prefix}.bias"))?;
        let mut g = vec![0f32; 4 * co * ci];
        let mut gb = vec![0f32; 4 * co];
        for d in 0..4 {
            let (dy, dx) = (d / 2, d % 2);
            for o in 0..co {
                gb[d * co + o] = b[o];
                for c in 0..ci {
                    g[(d * co + o) * ci + c] = w[((c * co + o) * 2 + dy) * 2 + dx];
                }
            }
        }
        self.linear_from_host(&g, Some(&gb), 4 * co, ci)
    }
}

/// Conv2d in implicit-GEMM layout.
#[derive(Clone)]
pub struct Conv {
    pub w: Tensor,
    pub b: Option<Tensor>,
    pub co: usize,
    pub ci: usize,
    pub kh: usize,
    pub kw: usize,
    pub kpad: usize,
}

pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

pub fn expect_shape(t: &Tensor, shape: &[usize], what: &str) -> Result<()> {
    if t.shape != shape {
        bail!("{what}: expected shape {:?}, got {:?}", shape, t.shape);
    }
    Ok(())
}
