//! Sampling loop of `MOSSImagePipeline.__call__`: fp32 latents, flow-matching Euler on the official schedule, CFG with
//! the negative branch run with block 22 skipped (`uncond_skip_layers=(22,)`), plus SDEdit img2img and latent inpainting.
use crate::cuda::Device;
use crate::dit::{Dit, Run, LATENT};
use crate::ops;
use crate::rng;
use crate::sampler;
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Context, Result};

#[derive(Clone, Debug)]
pub struct Settings {
    pub steps: usize,
    pub cfg: f32,
    pub shift: f32,
    pub seed: u64,
    /// blocks skipped in the negative branch (official default [22])
    pub skip_layers: Vec<usize>,
    /// img2img strength (1 = from pure noise)
    pub strength: f32,
}

/// The initial noise for a seed, token-major [h*w, 128] (drawn as torch.randn([1, 128, h, w]) on the CPU).
pub fn noise_tokens(seed: u64, h: usize, w: usize) -> Vec<f32> {
    let chw = rng::torch_randn(seed, LATENT * h * w);
    let mut out = vec![0f32; chw.len()];
    for c in 0..LATENT {
        for p in 0..h * w {
            out[p * LATENT + c] = chw[c * h * w + p];
        }
    }
    out
}

pub struct Inputs<'a> {
    pub cond: &'a Run,
    /// None when CFG is off (cfg == 1)
    pub uncond: Option<&'a Run>,
    /// normalized init latent tokens f32 [h*w, 128] (img2img / inpaint)
    pub x0: Option<&'a Tensor>,
    /// latent-resolution mask f32 [h*w] (1 = regenerate)
    pub mask: Option<&'a Tensor>,
    /// override of the initial noise (tests)
    pub noise: Option<Vec<f32>>,
}

/// Run the sampler; returns the final latent tokens f32 [h*w, 128].
pub fn sample(dev: &Device, dit: &Dit, s: &Settings, inp: &Inputs, progress: &dyn Fn(usize, usize)) -> Result<Tensor> {
    let (h, w) = (inp.cond.h, inp.cond.w);
    let nt = h * w;
    let sig = sampler::sigmas(s.steps, s.shift);
    let n_steps = sig.len() - 1;
    let start = if inp.x0.is_some() { sampler::img2img_start(n_steps, s.strength) } else { 0 };
    ensure!(start < n_steps, "--strength {} leaves no steps to run with {} steps", s.strength, n_steps);
    let noise_v = match &inp.noise {
        Some(v) => v.clone(),
        None => noise_tokens(s.seed, h, w),
    };
    let noise = Tensor::from_f32(dev, &noise_v, &[nt, LATENT])?;
    let x = Tensor::new(dev, DType::F32, &[nt, LATENT])?;
    match inp.x0 {
        Some(x0) => ops::noise_mix(dev, &x, x0, &noise, sig[start])?,
        None => dev.dtod(x.ptr, noise.ptr, noise.bytes())?,
    }
    let xb = Tensor::new(dev, DType::BF16, &[nt, LATENT])?;
    let vc = Tensor::new(dev, DType::BF16, &[nt, LATENT])?;
    let vu = Tensor::new(dev, DType::BF16, &[nt, LATENT])?;
    let total = n_steps - start;
    for (k, i) in (start..n_steps).enumerate() {
        crate::worker::check_cancel()?;
        progress(k, total);
        let t = sampler::timestep(sig[i]);
        let m = dit.modulation(t)?;
        ops::to_bf16(dev, &x, &xb)?;
        dit.forward(inp.cond, &xb, &m, &[], &vc).with_context(|| format!("dit step {i}"))?;
        let uncond = match inp.uncond {
            Some(u) => {
                dit.forward(u, &xb, &m, &s.skip_layers, &vu).with_context(|| format!("dit step {i} (negative)"))?;
                Some(&vu)
            }
            None => None,
        };
        ops::cfg_euler(dev, &x, &vc, uncond, s.cfg, sig[i + 1] - sig[i])?;
        if let (Some(mask), Some(x0)) = (inp.mask, inp.x0) {
            ops::inpaint_blend(dev, &x, x0, &noise, mask, sig[i + 1])?;
        }
        if k % 4 == 3 {
            dev.sync()?;
        }
    }
    progress(total, total);
    dit.prof.report();
    Ok(x)
}
