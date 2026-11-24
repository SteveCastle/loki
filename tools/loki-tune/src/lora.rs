//! The trained LoRA: every MOSS-Image block linear (q, k, v, o, ffn_up_gate, ffn_down of all 42 blocks), the same set
//! the Kromoss turbo LoRAs cover and loki-dream / ComfyUI load.
//!
//! Parameters live in one flat f32 buffer on the GPU (with gradients and both Adam moments next to it), so the optimizer
//! is one kernel over everything. Per linear: down (A) [R, K] then up (B) [N, R]. q, k and v are three LoRAs; they are
//! trained on the fused qkv projection as a stacked down [3r, D] and a block-diagonal up [3D, 3r] whose off-diagonal
//! blocks are zero and stay zero (their gradients are masked). Each step the weights are cast to bf16 for the GEMMs
//! (up and its transpose pre-scaled by alpha / rank).
use crate::gpu;
use crate::safetensors_out::{self, Out};
use anyhow::{ensure, Context, Result};
use loki_dream::cuda::Device;
use loki_dream::dit::{DIM, FFN, LAYERS};
use loki_dream::tensor::{DType, Tensor};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Qkv,
    O,
    UpGate,
    Down,
}
impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Qkv, Kind::O, Kind::UpGate, Kind::Down];
    /// (input width K, output width N)
    pub fn dims(self) -> (usize, usize) {
        match self {
            Kind::Qkv => (DIM, 3 * DIM),
            Kind::O => (DIM, DIM),
            Kind::UpGate => (DIM, 2 * FFN),
            Kind::Down => (FFN, DIM),
        }
    }
}

/// One linear's parameters in the flat buffer (offsets in f32 elements).
#[derive(Clone, Debug)]
pub struct Lin {
    pub layer: usize,
    pub kind: Kind,
    /// rank of the stored matrices (3 r for the fused qkv)
    pub r: usize,
    pub k: usize,
    pub n: usize,
    pub down: usize,
    pub up: usize,
}

/// The bf16 copies one training step uses.
pub struct Mat {
    /// A [R, K]
    pub down: Tensor,
    /// A^T [K, R]
    pub down_t: Tensor,
    /// B * scale [N, R]
    pub up: Tensor,
    /// (B * scale)^T [R, N]
    pub up_t: Tensor,
}

pub struct LoraSet {
    pub rank: usize,
    pub alpha: f32,
    pub lins: Vec<Lin>,
    pub n_params: usize,
    /// params, grads, Adam m, Adam v: f32 [n_params] each
    pub p: Tensor,
    pub g: Tensor,
    pub m: Tensor,
    pub v: Tensor,
    pub mats: Vec<Mat>,
}

/// splitmix64: the deterministic stream LoRA init and the training plan draw from.
#[derive(Clone, Debug)]
pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// uniform in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize % n.max(1)
    }
}

fn layout(rank: usize) -> (Vec<Lin>, usize) {
    let mut lins = Vec::new();
    let mut off = 0usize;
    for layer in 0..LAYERS {
        for kind in Kind::ALL {
            let (k, n) = kind.dims();
            let r = if kind == Kind::Qkv { 3 * rank } else { rank };
            let down = off;
            off += r * k;
            let up = off;
            off += n * r;
            lins.push(Lin { layer, kind, r, k, n, down, up });
        }
    }
    (lins, off)
}

fn ptr(t: &Tensor, off: usize) -> u64 {
    t.ptr + (off * 4) as u64
}

impl LoraSet {
    /// Fresh LoRA: A ~ U(-1/sqrt(K), 1/sqrt(K)) (PEFT's kaiming-uniform), B = 0, so training starts at the base model.
    pub fn new(dev: &Device, rank: usize, alpha: f32, seed: u64) -> Result<LoraSet> {
        ensure!(rank >= 8 && rank % 8 == 0 && rank <= 256, "--rank must be a multiple of 8 between 8 and 256");
        let (lins, n) = layout(rank);
        let mut host = vec![0f32; n];
        let mut rng = Rng::new(seed.wrapping_add(0x4c6f_5241));
        for l in &lins {
            let bound = 1.0 / (l.k as f64).sqrt();
            for x in &mut host[l.down..l.down + l.r * l.k] {
                *x = ((rng.uniform() * 2.0 - 1.0) * bound) as f32;
            }
        }
        Self::from_host(dev, rank, alpha, &host, None)
    }

    fn from_host(dev: &Device, rank: usize, alpha: f32, p: &[f32], mv: Option<(&[f32], &[f32])>) -> Result<LoraSet> {
        let (lins, n) = layout(rank);
        ensure!(p.len() == n, "LoRA state holds {} parameters, rank {rank} needs {n}", p.len());
        let pt = Tensor::from_f32(dev, p, &[n])?;
        let g = Tensor::zeros(dev, DType::F32, &[n])?;
        let (m, v) = match mv {
            Some((m, v)) => (Tensor::from_f32(dev, m, &[n])?, Tensor::from_f32(dev, v, &[n])?),
            None => (Tensor::zeros(dev, DType::F32, &[n])?, Tensor::zeros(dev, DType::F32, &[n])?),
        };
        let mut mats = Vec::with_capacity(lins.len());
        for l in &lins {
            mats.push(Mat {
                down: Tensor::new(dev, DType::BF16, &[l.r, l.k])?,
                down_t: Tensor::new(dev, DType::BF16, &[l.k, l.r])?,
                up: Tensor::new(dev, DType::BF16, &[l.n, l.r])?,
                up_t: Tensor::new(dev, DType::BF16, &[l.r, l.n])?,
            });
        }
        let mut set = LoraSet { rank, alpha, lins, n_params: n, p: pt, g, m, v, mats };
        set.cast(dev)?;
        Ok(set)
    }

    pub fn scale(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn get(&self, layer: usize, kind: Kind) -> (&Lin, &Mat) {
        let i = layer * 4 + Kind::ALL.iter().position(|&k| k == kind).unwrap();
        (&self.lins[i], &self.mats[i])
    }

    /// Gradient buffers of a linear: (down, up) device addresses.
    pub fn grad_ptrs(&self, l: &Lin) -> (u64, u64) {
        (ptr(&self.g, l.down), ptr(&self.g, l.up))
    }

    /// Refresh the bf16 copies from the f32 parameters (after every optimizer step).
    pub fn cast(&mut self, dev: &Device) -> Result<()> {
        let s = self.scale();
        for (l, m) in self.lins.iter().zip(&self.mats) {
            gpu::lora_cast(dev, ptr(&self.p, l.down), &m.down, l.r, l.k, 1.0, false)?;
            gpu::lora_cast(dev, ptr(&self.p, l.down), &m.down_t, l.r, l.k, 1.0, true)?;
            gpu::lora_cast(dev, ptr(&self.p, l.up), &m.up, l.n, l.r, s, false)?;
            gpu::lora_cast(dev, ptr(&self.p, l.up), &m.up_t, l.n, l.r, s, true)?;
        }
        Ok(())
    }

    /// Mask the fused qkv up-gradients to their q / k / v diagonal blocks.
    pub fn mask_grads(&self, dev: &Device) -> Result<()> {
        for l in self.lins.iter().filter(|l| l.kind == Kind::Qkv) {
            gpu::mask_blockdiag(dev, ptr(&self.g, l.up), DIM, self.rank)?;
        }
        Ok(())
    }

    /// Global L2 norm of the accumulated gradients.
    pub fn grad_norm(&self, dev: &Device) -> Result<f64> {
        Ok(gpu::sumsq(dev, self.g.ptr, self.n_params)?.sqrt())
    }

    pub fn zero_grads(&self, dev: &Device) -> Result<()> {
        dev.memset_at(self.g.ptr, self.n_params * 4)
    }

    /// One AdamW step (`gscale` multiplies the accumulated gradients: 1/accumulation x clipping), then recast.
    #[allow(clippy::too_many_arguments)]
    pub fn step(&mut self, dev: &Device, lr: f32, wd: f32, step: usize, gscale: f32) -> Result<()> {
        gpu::adamw(dev, self.p.ptr, self.g.ptr, self.m.ptr, self.v.ptr, self.n_params, lr, 0.9, 0.999, 1e-8, wd, step, gscale)?;
        self.cast(dev)
    }

    // ------------------------------------------------------------------------------------------------ files
    /// The inference LoRA, kohya / Kromoss keys (`diffusion_model.layers.N.proj.q.lora_down.weight`, `.lora_up.weight`,
    /// `.alpha`), bf16. `meta` becomes the safetensors `__metadata__`.
    pub fn save_lora(&self, dev: &Device, path: &Path, meta: &BTreeMap<String, String>) -> Result<()> {
        self.export(dev, &self.p, true, path, meta)
    }

    /// A flat buffer laid out like the parameters (the parameters themselves, or their gradients) in the LoRA key layout.
    pub fn export(&self, dev: &Device, buf: &Tensor, bf16: bool, path: &Path, meta: &BTreeMap<String, String>) -> Result<()> {
        let p = buf.to_f32_vec(dev)?;
        let r = self.rank;
        let mut out = Out::default();
        let dt = if bf16 { "BF16" } else { "F32" };
        let bf = |v: &[f32]| -> Vec<u8> {
            if bf16 {
                v.iter().flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes()).collect()
            } else {
                v.iter().flat_map(|x| x.to_le_bytes()).collect()
            }
        };
        for l in &self.lins {
            let pre = format!("diffusion_model.layers.{}.proj", l.layer);
            let down = &p[l.down..l.down + l.r * l.k];
            let up = &p[l.up..l.up + l.n * l.r];
            let mut put = |name: &str, d: &[f32], u: Vec<f32>, n: usize, k: usize| {
                out.add(&format!("{pre}.{name}.lora_down.weight"), dt, &[r, k], bf(d));
                out.add(&format!("{pre}.{name}.lora_up.weight"), dt, &[n, r], bf(&u));
                out.add(&format!("{pre}.{name}.alpha"), "F32", &[], self.alpha.to_le_bytes().to_vec());
            };
            match l.kind {
                Kind::Qkv => {
                    for (i, name) in ["q", "k", "v"].iter().enumerate() {
                        let d = &down[i * r * DIM..(i + 1) * r * DIM];
                        // rows i*D.. of the block-diagonal up, columns i*r..
                        let mut u = vec![0f32; DIM * r];
                        for row in 0..DIM {
                            let src = (i * DIM + row) * 3 * r + i * r;
                            u[row * r..(row + 1) * r].copy_from_slice(&up[src..src + r]);
                        }
                        put(name, d, u, DIM, DIM);
                    }
                }
                Kind::O => put("o", down, up.to_vec(), l.n, l.k),
                Kind::UpGate => put("ffn_up_gate", down, up.to_vec(), l.n, l.k),
                Kind::Down => put("ffn_down", down, up.to_vec(), l.n, l.k),
            }
        }
        safetensors_out::write(path, &out, meta)
    }

    /// Optimizer state (f32 params + Adam moments) for resuming.
    pub fn save_state(&self, dev: &Device, path: &Path, meta: &BTreeMap<String, String>) -> Result<()> {
        let n = self.n_params;
        let mut out = Out::default();
        for (name, t) in [("params", &self.p), ("adam_m", &self.m), ("adam_v", &self.v)] {
            let v = t.to_f32_vec(dev)?;
            out.add(name, "F32", &[n], v.iter().flat_map(|x| x.to_le_bytes()).collect());
        }
        safetensors_out::write(path, &out, meta)
    }

    pub fn load_state(dev: &Device, path: &Path, rank: usize, alpha: f32) -> Result<LoraSet> {
        let st = loki_dream::safetensors::SafeTensors::open(path).with_context(|| format!("opening {}", path.display()))?;
        let p = st.f32s("params")?;
        let m = st.f32s("adam_m")?;
        let v = st.f32s("adam_v")?;
        Self::from_host(dev, rank, alpha, &p, Some((&m, &v)))
    }
}

/// The JSON object stored under a state file's `loki_tune` metadata key.
pub fn state_meta(path: &Path) -> Result<Value> {
    let meta = safetensors_out::read_metadata(path)?;
    let s = meta.get("loki_tune").with_context(|| format!("{} is not a loki-tune checkpoint state", path.display()))?;
    Ok(serde_json::from_str(s)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_counts() {
        let (lins, n) = layout(16);
        assert_eq!(lins.len(), LAYERS * 4);
        // per block: qkv 48x3072 + 9216x48, o 2x16x3072, up_gate 16x3072 + 16384x16, down 16x8192 + 3072x16
        let per = 48 * 3072 + 9216 * 48 + 2 * 16 * 3072 + 16 * 3072 + 16384 * 16 + 16 * 8192 + 3072 * 16;
        assert_eq!(n, per * LAYERS);
        assert_eq!(lins[4].layer, 1);
        assert_eq!(lins[1].down, lins[0].up + 9216 * 48);
    }

    #[test]
    fn rng_is_deterministic_and_uniform() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let xs: Vec<f64> = (0..1000).map(|_| a.uniform()).collect();
        assert!(xs.iter().zip((0..1000).map(|_| b.uniform())).all(|(x, y)| *x == y));
        let mean = xs.iter().sum::<f64>() / 1000.0;
        assert!((mean - 0.5).abs() < 0.05);
    }
}
