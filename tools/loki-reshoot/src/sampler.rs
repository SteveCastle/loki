//! Flow-matching schedule (ModelSamplingAV shift + `simple` scheduler) and the `res_multistep` sampler
//! (ComfyUI k_diffusion, eta = 0), operating on the packed (video, carried-audio) latent pair.
use crate::cuda::{Arg, Device};
use crate::tensor::{DType, Tensor};
use anyhow::Result;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

pub const SHIFT_VIDEO: f64 = 12.0;
pub const SHIFT_AUDIO: f64 = 3.0;

fn snr_shift(alpha: f64, t: f64) -> f64 {
    if alpha == 1.0 {
        t
    } else {
        alpha * t / (1.0 + (alpha - 1.0) * t)
    }
}

/// ComfyUI `simple_scheduler` on ModelSamplingDiscreteFlow(shift): `steps` sigmas descending from 1, then 0.
pub fn simple_sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let table: Vec<f32> = (1..=1000).map(|i| snr_shift(shift, i as f64 / 1000.0) as f32).collect();
    let ss = table.len() as f64 / steps as f64;
    let mut sig: Vec<f64> = (0..steps).map(|x| table[table.len() - 1 - (x as f64 * ss) as usize] as f64).collect();
    sig.push(0.0);
    sig
}

/// Standard-normal noise on the device (f32), reproducible from `seed`.
pub fn randn(dev: &Device, shape: &[usize], seed: u64) -> Result<Tensor> {
    let n: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let v: Vec<f32> = (0..n).map(|_| StandardNormal.sample(&mut rng)).collect();
    Tensor::from_f32(dev, &v, shape)
}

fn axpby3(dev: &Device, out: &Tensor, x: &Tensor, d: &Tensor, o: Option<&Tensor>, a: f32, b: f32, c: f32) -> Result<()> {
    dev.launch_n(
        "k_axpby3_f32",
        out.numel(),
        &[Arg::Ptr(out.ptr), Arg::Ptr(x.ptr), Arg::Ptr(d.ptr), Arg::Ptr(o.map_or(0, |t| t.ptr)), Arg::I64(out.numel() as i64), Arg::F32(a), Arg::F32(b), Arg::F32(c)],
    )
}

/// A model evaluation: given the current (video, audio-carried) latents and sigma, write the velocity
/// outputs (ComfyUI `forward()` convention, so that denoised = x - out * sigma).
pub type Model<'a> = dyn FnMut(&Tensor, &Tensor, f32, &Tensor, &Tensor) -> Result<()> + 'a;

/// Run res_multistep over `sigmas` in place on (`xv`, `xa`). `progress(i, n)` is called after each step.
pub fn res_multistep(dev: &Device, sigmas: &[f64], xv: &Tensor, xa: &Tensor, model: &mut Model, progress: &dyn Fn(usize, usize)) -> Result<()> {
    let steps = sigmas.len() - 1;
    let (vv, va) = (Tensor::new(dev, DType::F32, &xv.shape)?, Tensor::new(dev, DType::F32, &xa.shape)?);
    let (mut dv, mut da) = (Tensor::new(dev, DType::F32, &xv.shape)?, Tensor::new(dev, DType::F32, &xa.shape)?);
    let (mut ov, mut oa) = (Tensor::new(dev, DType::F32, &xv.shape)?, Tensor::new(dev, DType::F32, &xa.shape)?);
    let mut have_old = false;
    for i in 0..steps {
        let (s, sn) = (sigmas[i], sigmas[i + 1]);
        model(xv, xa, s as f32, &vv, &va)?;
        // denoised = x - v * sigma
        axpby3(dev, &dv, xv, &vv, None, 1.0, -(s as f32), 0.0)?;
        axpby3(dev, &da, xa, &va, None, 1.0, -(s as f32), 0.0)?;
        if sn == 0.0 || !have_old {
            // Euler step to sigma_next: x = x*(sn/s) + denoised*(1 - sn/s)
            let r = sn / s;
            axpby3(dev, xv, xv, &dv, None, r as f32, (1.0 - r) as f32, 0.0)?;
            axpby3(dev, xa, xa, &da, None, r as f32, (1.0 - r) as f32, 0.0)?;
        } else {
            // old_sigma_down of the previous step is sigmas[i] (eta = 0), so t_old == t
            let (t, t_next, t_prev) = (-s.ln(), -sn.ln(), -sigmas[i - 1].ln());
            let t_old = t;
            let h = t_next - t;
            let c2 = (t_prev - t_old) / h;
            let phi1 = |x: f64| x.exp_m1() / x;
            let phi2 = |x: f64| (phi1(x) - 1.0) / x;
            let (p1, p2) = (phi1(-h), phi2(-h));
            let nan0 = |v: f64| if v.is_finite() { v } else { 0.0 };
            let b1 = nan0(p1 - p2 / c2);
            let b2 = nan0(p2 / c2);
            let a = (-h).exp();
            axpby3(dev, xv, xv, &dv, Some(&ov), a as f32, (h * b1) as f32, (h * b2) as f32)?;
            axpby3(dev, xa, xa, &da, Some(&oa), a as f32, (h * b1) as f32, (h * b2) as f32)?;
        }
        std::mem::swap(&mut dv, &mut ov);
        std::mem::swap(&mut da, &mut oa);
        have_old = true;
        dev.sync()?;
        progress(i + 1, steps);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_matches_comfy() {
        let s = simple_sigmas(20, 12.0);
        assert_eq!(s.len(), 21);
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert_eq!(s[20], 0.0);
        // table[999 - 50] = shift(0.95)
        let want = 12.0 * 0.95 / (1.0 + 11.0 * 0.95);
        assert!((s[1] - want).abs() < 1e-6, "{} vs {}", s[1], want);
        assert!(s.windows(2).all(|w| w[0] > w[1]));
    }
}
