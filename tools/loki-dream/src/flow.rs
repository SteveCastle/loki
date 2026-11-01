//! Flow-matching sampling for Krea 2, FLUX.2 klein and Anima: diffusers' `FlowMatchEulerDiscreteScheduler` schedules
//! (`linspace(1, 1/n, n)` shifted statically or by a resolution-dependent `mu` with the exponential time shift), Euler
//! steps on fp32 latents, classifier-free guidance `u + cfg (c - u)`, SDEdit img2img and latent inpainting.
use crate::cuda::Device;
use crate::model::Model;
use crate::ops;
use crate::rng;
use crate::sampler;
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Context, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shift {
    /// sigma' = s * sigma / (1 + (s - 1) sigma)
    Static(f32),
    /// exponential time shift: sigma' = e^mu / (e^mu + 1/sigma - 1)
    Dynamic(f32),
}

/// Sigmas for `steps` steps, ending with the terminal 0.
pub fn sigmas(steps: usize, shift: Shift) -> Vec<f32> {
    let n = steps.max(1);
    let mut out: Vec<f32> = (0..n)
        .map(|i| {
            // np.linspace(1.0, 1/n, n) in f64 (i * step + start, the last point exact), then .astype(float32)
            let stop = 1.0 / n as f64;
            let s = if i == n - 1 { stop } else { i as f64 * ((stop - 1.0) / (n - 1).max(1) as f64) + 1.0 } as f32;
            match shift {
                Shift::Static(k) => k * s / (1.0 + (k - 1.0) * s),
                Shift::Dynamic(mu) => {
                    let e = (mu as f64).exp() as f32;
                    e / (e + (1.0 / s - 1.0))
                }
            }
        })
        .collect();
    out.push(0.0);
    out
}

/// FLUX.2's `compute_empirical_mu` (diffusers pipeline_flux2).
pub fn flux2_mu(seq: usize, steps: usize) -> f32 {
    let (a1, b1) = (8.73809524e-05f64, 1.89833333f64);
    let (a2, b2) = (0.00016927f64, 0.45666666f64);
    let s = seq as f64;
    if seq > 4300 {
        return (a2 * s + b2) as f32;
    }
    let m200 = a2 * s + b2;
    let m10 = a1 * s + b1;
    let a = (m200 - m10) / 190.0;
    let b = m200 - 200.0 * a;
    (a * steps as f64 + b) as f32
}

/// Krea 2's resolution-dependent mu (`calculate_shift` with base 256 / max 6400 tokens, shift 0.5 .. 1.15).
pub fn krea2_mu(seq: usize) -> f32 {
    let (b0, b1, s0, s1) = (256f64, 6400f64, 0.5f64, 1.15f64);
    let m = (s1 - s0) / (b1 - b0);
    (seq as f64 * m + (s0 - m * b0)) as f32
}

/// Noise for 16-channel latents patched 2x2 (Krea 2, Anima): torch.randn([16, 2h, 2w]) -> tokens [h*w, 64], c*4+dy*2+dx.
pub fn noise_patched16(seed: u64, h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (2 * h, 2 * w);
    let chw = rng::torch_randn(seed, 16 * hh * ww);
    let mut out = vec![0f32; chw.len()];
    for c in 0..16 {
        for y in 0..hh {
            for x in 0..ww {
                let t = (y / 2) * w + x / 2;
                out[t * 64 + c * 4 + (y % 2) * 2 + x % 2] = chw[(c * hh + y) * ww + x];
            }
        }
    }
    out
}

pub struct Inputs<'a, R> {
    pub cond: &'a R,
    pub uncond: Option<&'a R>,
    /// normalized init latent tokens f32 (img2img / inpaint)
    pub x0: Option<&'a Tensor>,
    /// latent-resolution mask f32 [tokens] (1 = regenerate)
    pub mask: Option<&'a Tensor>,
    pub noise: Vec<f32>,
}

/// Run the sampler over `sig` (with terminal 0); returns the final latent tokens f32 [tokens, channels].
#[allow(clippy::too_many_arguments)]
pub fn sample<M: Model>(dev: &Device, m: &M, sig: &[f32], cfg: f32, strength: f32, inp: &Inputs<M::Run>, progress: &dyn Fn(usize, usize)) -> Result<Tensor> {
    let c = m.channels();
    let nt = inp.noise.len() / c;
    let n_steps = sig.len() - 1;
    let start = if inp.x0.is_some() { sampler::img2img_start(n_steps, strength) } else { 0 };
    ensure!(start < n_steps, "--strength {strength} leaves no steps to run with {n_steps} steps");
    let noise = Tensor::from_f32(dev, &inp.noise, &[nt, c])?;
    let x = Tensor::new(dev, DType::F32, &[nt, c])?;
    match inp.x0 {
        Some(x0) => ops::noise_mix(dev, &x, x0, &noise, sig[start])?,
        None => dev.dtod(x.ptr, noise.ptr, noise.bytes())?,
    }
    let xb = Tensor::new(dev, DType::BF16, &[nt, c])?;
    let vc = Tensor::new(dev, DType::BF16, &[nt, c])?;
    let vu = Tensor::new(dev, DType::BF16, &[nt, c])?;
    let total = n_steps - start;
    for (k, i) in (start..n_steps).enumerate() {
        crate::worker::check_cancel()?;
        progress(k, total);
        let md = m.modulation(sig[i])?;
        ops::to_bf16(dev, &x, &xb)?;
        m.forward(inp.cond, &xb, &md, &vc).with_context(|| format!("step {i}"))?;
        let uncond = match inp.uncond {
            Some(u) => {
                m.forward(u, &xb, &md, &vu).with_context(|| format!("step {i} (negative)"))?;
                Some(&vu)
            }
            None => None,
        };
        ops::cfg_euler(dev, &x, &vc, uncond, cfg, sig[i + 1] - sig[i])?;
        if let (Some(mask), Some(x0)) = (inp.mask, inp.x0) {
            ops::inpaint_blend(dev, &x, x0, &noise, mask, sig[i + 1])?;
        }
        if k % 4 == 3 {
            dev.sync()?;
        }
    }
    progress(total, total);
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_shift_schedule() {
        let s = sigmas(4, Shift::Static(3.0));
        assert_eq!(s.len(), 5);
        assert_eq!(s[0], 1.0);
        assert_eq!(s[4], 0.0);
        // 0.25 -> 3*0.25/(1+2*0.25) = 0.5
        assert!((s[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn dynamic_shift_is_identity_at_mu_zero() {
        let s = sigmas(4, Shift::Dynamic(0.0));
        assert!((s[1] - 0.75).abs() < 1e-6 && (s[3] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn mus() {
        assert!((krea2_mu(256) - 0.5).abs() < 1e-6);
        assert!((krea2_mu(6400) - 1.15).abs() < 1e-6);
        assert!((flux2_mu(5000, 4) - (0.00016927 * 5000.0 + 0.45666666) as f32).abs() < 1e-6);
    }

    #[test]
    fn patched_noise_layout() {
        let n = noise_patched16(1, 2, 3);
        let chw = rng::torch_randn(1, 16 * 4 * 6);
        // token (1, 2) channel c=5, dy=1, dx=0 <- pixel (3, 4)
        assert_eq!(n[(1 * 3 + 2) * 64 + 5 * 4 + 2], chw[(5 * 4 + 3) * 6 + 4]);
    }
}
