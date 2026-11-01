//! What the flow sampler needs from a diffusion transformer, and the pieces the new architectures share.
use crate::cuda::Device;
use crate::linear::{self, Want};
use crate::ops::{self, Act, Epi};
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::Loader;
use anyhow::Result;

/// A flow-matching velocity model over token-major latents [tokens, channels()].
pub trait Model {
    /// Step-independent state of one prompt at one latent size.
    type Run;
    /// Per-step (timestep) conditioning, shared by the positive and negative branch.
    type Mod;
    fn channels(&self) -> usize;
    fn modulation(&self, sigma: f32) -> Result<Self::Mod>;
    /// `x` bf16 [tokens, channels] -> velocity `out` bf16 [tokens, channels].
    fn forward(&self, run: &Self::Run, x: &Tensor, m: &Self::Mod, out: &Tensor) -> Result<()>;
}

pub fn round_bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

/// Sinusoidal timestep embedding [cos | sin] of `dim` values, frequencies exp(-ln(10000) i / (dim/2)), computed in f32
/// and rounded to bf16 (diffusers `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)` and its look-alikes).
pub fn sinusoid_bf16(dev: &Device, t: f32, dim: usize) -> Result<Tensor> {
    let half = dim / 2;
    let mut e = vec![0u16; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        let a = t * f;
        e[i] = bf16_bits(a.cos());
        e[half + i] = bf16_bits(a.sin());
    }
    Tensor::from_bf16(dev, &e, &[1, dim])
}

/// A small dense layer (time embeddings, modulation): bf16 weight [N, K] + optional f32 bias.
pub struct Dense {
    pub w: Tensor,
    pub b: Option<Tensor>,
}
impl Dense {
    pub fn load(l: &mut Loader, prefix: &str) -> Result<Dense> {
        let w = linear::weight_bf16(l, prefix)?;
        Ok(Dense { w, b: linear::bias(l, prefix)? })
    }
    pub fn n(&self) -> usize {
        self.w.shape[0]
    }
    /// out = act(x . W^T + b), bf16
    pub fn apply(&self, dev: &Device, x: &Tensor, act: Act) -> Result<Tensor> {
        let out = Tensor::new(dev, DType::BF16, &[x.shape[0], self.n()])?;
        ops::gemm_bf16(dev, x, &self.w, self.b.as_ref(), Act::None, Epi::Store, &out)?;
        if act != Act::None {
            ops::act_inplace(dev, &out, act)?;
        }
        Ok(out)
    }
}

/// A [1, k*D] bf16 row split into k f32 vectors of D (holding the bf16 values).
pub fn split_row(dev: &Device, row: &Tensor, parts: usize) -> Result<Vec<Tensor>> {
    let v = row.to_f32_vec(dev)?;
    let d = v.len() / parts;
    (0..parts).map(|i| Tensor::from_f32(dev, &v[i * d..(i + 1) * d], &[d])).collect()
}

/// Linears of a DiT block in the run's precision.
pub fn want(int8: bool) -> Want {
    if int8 {
        Want::Int8
    } else {
        Want::Bf16
    }
}

/// x (bf16) through SiLU, as a new tensor.
pub fn silu_copy(dev: &Device, x: &Tensor) -> Result<Tensor> {
    let y = Tensor::new(dev, DType::BF16, &x.shape)?;
    dev.dtod(y.ptr, x.ptr, x.bytes())?;
    ops::act_inplace(dev, &y, Act::Silu)?;
    Ok(y)
}

/// Rope table [n, 64, 2] (cos, sin) f32 for per-token frequency angles `angle(token, f)`.
pub fn rope_table(dev: &Device, n: usize, angle: impl Fn(usize, usize) -> f64) -> Result<Tensor> {
    let mut tab = vec![0f32; n * 128];
    for t in 0..n {
        for f in 0..64 {
            let a = angle(t, f);
            tab[t * 128 + f * 2] = a.cos() as f32;
            tab[t * 128 + f * 2 + 1] = a.sin() as f32;
        }
    }
    Tensor::from_f32(dev, &tab, &[n, 64, 2])
}

pub fn bias_or_none(l: &mut Loader, prefix: &str) -> Result<Option<Tensor>> {
    linear::bias(l, prefix)
}

/// Copy `src` [rows, w] into columns col0.. of `dst` [rows, W] (bf16; dst pre-zeroed).
pub fn put_cols(dev: &Device, dst: &Tensor, col0: usize, src: &Tensor) -> Result<()> {
    let rows = src.shape[0];
    let w = src.numel() / rows;
    let dw = dst.numel() / dst.shape[0];
    ops::add_cols(dev, dst.ptr, dw, col0, src.ptr, w, rows, w)
}

#[cfg(test)]
mod tests {
    #[test]
    fn rounding() {
        assert_eq!(super::round_bf16(1.0), 1.0);
    }
}
