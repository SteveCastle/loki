//! Flow-matching schedules (ModelSamplingAV shift + the `simple` / `sgm_uniform` schedulers, partial denoise) and the
//! `res_multistep` (ComfyUI k_diffusion, eta = 0), `euler` and `er_sde` samplers, operating on the packed (video, carried-audio) latent pair.
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

/// ComfyUI `sgm_uniform` (normal_scheduler with sgm=True) on ModelSamplingDiscreteFlow(shift): timesteps linear from
/// timestep(sigma_max) = 1000 to timestep(sigma_min) = 1000 * shift(0.001), `steps` of them (the endpoint dropped), each
/// mapped back through the shifted sigma, then 0.
pub fn sgm_uniform_sigmas(steps: usize, shift: f64) -> Vec<f64> {
    let sigma_min = snr_shift(shift, 1.0 / 1000.0) as f32 as f64;
    let (start, end) = (1000.0f64, sigma_min * 1000.0);
    let mut sig: Vec<f64> = (0..steps)
        .map(|i| {
            // torch.linspace(start, end, steps + 1) in f32
            let ts = (start + (end - start) * i as f64 / steps as f64) as f32 as f64;
            snr_shift(shift, ts / 1000.0) as f32 as f64
        })
        .collect();
    sig.push(0.0);
    sig
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Simple,
    SgmUniform,
}

/// ComfyUI `BasicScheduler(scheduler, steps, denoise)`: below denoise 1 the schedule of int(steps / denoise) steps is
/// computed and its last `steps` steps are taken (so the run starts at that schedule's sigma, not at 1).
pub fn schedule(kind: Scheduler, steps: usize, denoise: f64, shift: f64) -> Vec<f64> {
    let total = if denoise < 1.0 { ((steps as f64 / denoise.max(1e-6)) as usize).max(steps) } else { steps };
    let all = match kind {
        Scheduler::Simple => simple_sigmas(total, shift),
        Scheduler::SgmUniform => sgm_uniform_sigmas(total, shift),
    };
    all[all.len() - (steps + 1)..].to_vec()
}

/// Flow-matching `noise_scaling` (ComfyUI CONST): x = sigma * noise + (1 - sigma) * latent, in place on `latent`.
pub fn noise_latent(dev: &Device, latent: &Tensor, noise: &Tensor, sigma: f64) -> Result<()> {
    axpby3(dev, latent, latent, noise, None, (1.0 - sigma) as f32, sigma as f32, 0.0)
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

/// Run the Euler sampler over `sigmas` in place on (`xv`, `xa`): what few-step distilled models (FastH3) are trained for.
/// `progress(i, n)` is called after each step.
pub fn euler(dev: &Device, sigmas: &[f64], xv: &Tensor, xa: &Tensor, model: &mut Model, progress: &dyn Fn(usize, usize)) -> Result<()> {
    let steps = sigmas.len() - 1;
    let (vv, va) = (Tensor::new(dev, DType::F32, &xv.shape)?, Tensor::new(dev, DType::F32, &xa.shape)?);
    for i in 0..steps {
        let (s, sn) = (sigmas[i], sigmas[i + 1]);
        model(xv, xa, s as f32, &vv, &va)?;
        if std::env::var_os("LOKI_DEBUG_LATENT").is_some() {
            debug_frames(dev, &format!("step {} v", i + 1), &vv)?;
        }
        // x + (sn - s) * v  ==  x*(sn/s) + denoised*(1 - sn/s)
        axpby3(dev, xv, xv, &vv, None, 1.0, (sn - s) as f32, 0.0)?;
        axpby3(dev, xa, xa, &va, None, 1.0, (sn - s) as f32, 0.0)?;
        dev.sync()?;
        if std::env::var_os("LOKI_DEBUG_LATENT").is_some() {
            debug_frames(dev, &format!("step {} x", i + 1), xv)?;
        }
        progress(i + 1, steps);
    }
    Ok(())
}

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

/// ComfyUI `er_sde` (VP ER-SDE-Solver-3, s_noise 1, max_stage 3) on the flow (CONST) parameterization: stochastic, so
/// fresh noise is drawn after every step but the last (seeded from `seed`). This is the two-pass workflow's sampler.
pub fn er_sde(dev: &Device, sigmas: &[f64], xv: &Tensor, xa: &Tensor, seed: u64, model: &mut Model, progress: &dyn Fn(usize, usize)) -> Result<()> {
    let steps = sigmas.len() - 1;
    let sig = er_sde_sigmas(sigmas);
    // half-logSNR = -logit(sigma); er_lambda = exp(-half_log_snr) = sigma / (1 - sigma)
    let lam: Vec<f64> = sig.iter().map(|&s| s / (1.0 - s)).collect();
    let ns = |x: f64| x * ((x.powf(0.3)).exp() + 10.0);
    let points = 200usize;
    let mk = |t: &Tensor| Tensor::new(dev, DType::F32, &t.shape);
    let (vv, va) = (mk(xv)?, mk(xa)?);
    let (mut dv, mut da) = (mk(xv)?, mk(xa)?); // denoised
    let (mut odv, mut oda) = (mk(xv)?, mk(xa)?); // old denoised
    let (mut ddv, mut dda) = (mk(xv)?, mk(xa)?); // denoised_d
    let (mut oddv, mut odda) = (mk(xv)?, mk(xa)?); // old denoised_d
    for i in 0..steps {
        let s = sig[i];
        model(xv, xa, s as f32, &vv, &va)?;
        axpby3(dev, &dv, xv, &vv, None, 1.0, -(s as f32), 0.0)?;
        axpby3(dev, &da, xa, &va, None, 1.0, -(s as f32), 0.0)?;
        let stage = (i + 1).min(3);
        if sig[i + 1] == 0.0 {
            dev.dtod(xv.ptr, dv.ptr, xv.bytes())?;
            dev.dtod(xa.ptr, da.ptr, xa.bytes())?;
        } else {
            let (ls, lt) = (lam[i], lam[i + 1]);
            let alpha_s = sig[i] / ls;
            let alpha_t = sig[i + 1] / lt;
            let r_alpha = alpha_t / alpha_s;
            let r = ns(lt) / ns(ls);
            // stage 1 (Euler)
            let (a1, b1) = ((r_alpha * r) as f32, (alpha_t * (1.0 - r)) as f32);
            axpby3(dev, xv, xv, &dv, None, a1, b1, 0.0)?;
            axpby3(dev, xa, xa, &da, None, a1, b1, 0.0)?;
            if stage >= 2 {
                let dt = lt - ls;
                let step = -dt / points as f64;
                let pos: Vec<f64> = (0..points).map(|k| lt + k as f64 * step).collect();
                let s_int: f64 = pos.iter().map(|&p| 1.0 / ns(p)).sum::<f64>() * step;
                // denoised_d = (denoised - old_denoised) / (ls - lam[i-1])
                let inv = (1.0 / (ls - lam[i - 1])) as f32;
                axpby3(dev, &ddv, &dv, &odv, None, inv, -inv, 0.0)?;
                axpby3(dev, &dda, &da, &oda, None, inv, -inv, 0.0)?;
                let c2 = (alpha_t * (dt + s_int * ns(lt))) as f32;
                axpby3(dev, xv, xv, &ddv, None, 1.0, c2, 0.0)?;
                axpby3(dev, xa, xa, &dda, None, 1.0, c2, 0.0)?;
                if stage >= 3 {
                    let s_u: f64 = pos.iter().map(|&p| (p - ls) / ns(p)).sum::<f64>() * step;
                    // denoised_u = (denoised_d - old_denoised_d) / ((ls - lam[i-2]) / 2); folded into the update
                    let inv_u = 2.0 / (ls - lam[i - 2]);
                    let c3 = alpha_t * ((dt * dt) / 2.0 + s_u * ns(lt)) * inv_u;
                    axpby3(dev, xv, xv, &ddv, Some(&oddv), 1.0, c3 as f32, -(c3 as f32))?;
                    axpby3(dev, xa, xa, &dda, Some(&odda), 1.0, c3 as f32, -(c3 as f32))?;
                }
                std::mem::swap(&mut ddv, &mut oddv);
                std::mem::swap(&mut dda, &mut odda);
            }
            let amp = (lt * lt - ls * ls * r * r).sqrt();
            let amp = if amp.is_finite() { amp } else { 0.0 };
            let g = (alpha_t * amp) as f32;
            if g != 0.0 {
                // randn uploads synchronously into a stream-ordered allocation: let queued kernels finish first
                dev.sync()?;
                let step_seed = seed ^ 0xE5DE_0000_0000_0000 ^ ((i as u64 + 1) << 20);
                let nv = randn(dev, &xv.shape, step_seed)?;
                let na = randn(dev, &xa.shape, step_seed ^ 0x9E37_79B9_7F4A_7C15)?;
                axpby3(dev, xv, xv, &nv, None, 1.0, g, 0.0)?;
                axpby3(dev, xa, xa, &na, None, 1.0, g, 0.0)?;
            }
        }
        std::mem::swap(&mut dv, &mut odv);
        std::mem::swap(&mut da, &mut oda);
        dev.sync()?;
        progress(i + 1, steps);
    }
    Ok(())
}

/// The sigmas er_sde actually evaluates the model at: ComfyUI's `offset_first_sigma_for_snr` moves a first sigma of 1
/// (infinite logSNR) to percent_to_sigma(1e-4) = shift(1 - 1e-4).
pub fn er_sde_sigmas(sigmas: &[f64]) -> Vec<f64> {
    let mut s = sigmas.to_vec();
    if s.len() > 1 && s[0] >= 1.0 {
        s[0] = snr_shift(SHIFT_VIDEO, 1.0 - 1e-4);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sgm_uniform_and_partial_schedules() {
        // sgm_uniform: timesteps linspace(1000, 1000*sigma_min, 9)[:-1]
        let s = sgm_uniform_sigmas(8, 12.0);
        assert_eq!(s.len(), 9);
        assert!((s[0] - 1.0).abs() < 1e-6);
        let smin = 12.0 * 0.001 / (1.0 + 11.0 * 0.001);
        let ts1 = 1000.0 + (smin * 1000.0 - 1000.0) / 8.0;
        let want = 12.0 * (ts1 / 1000.0) / (1.0 + 11.0 * (ts1 / 1000.0));
        assert!((s[1] - want).abs() < 1e-5, "{} vs {}", s[1], want);
        assert_eq!(s[8], 0.0);
        // BasicScheduler(simple, 3, denoise 0.3): last 3 steps of the 10-step schedule
        let p = schedule(Scheduler::Simple, 3, 0.3, 12.0);
        let full = simple_sigmas(10, 12.0);
        assert_eq!(p, full[7..].to_vec());
        assert!((p[0] - 12.0 * 0.3 / (1.0 + 11.0 * 0.3)).abs() < 1e-6);
        assert_eq!(schedule(Scheduler::Simple, 8, 1.0, 12.0), simple_sigmas(8, 12.0));
    }
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

/// LOKI_DEBUG_LATENT: per latent frame std / max |x| / non-finite count of a [C, T, H, W] f32 tensor (diagnostics).
fn debug_frames(dev: &Device, what: &str, t: &Tensor) -> Result<()> {
    dev.sync()?;
    let v = t.to_f32_vec(dev)?;
    let (c, tt) = (t.shape[0], t.shape[1]);
    let hw: usize = t.shape[2..].iter().product();
    let mut s = String::new();
    for f in 0..tt {
        let (mut sum, mut sq, mut mx, mut bad) = (0f64, 0f64, 0f32, 0usize);
        for ch in 0..c {
            for &x in &v[(ch * tt + f) * hw..(ch * tt + f + 1) * hw] {
                if !x.is_finite() {
                    bad += 1;
                    continue;
                }
                sum += x as f64;
                sq += (x as f64) * (x as f64);
                mx = mx.max(x.abs());
            }
        }
        let n = (c * hw - bad).max(1) as f64;
        let std = (sq / n - (sum / n).powi(2)).max(0.0).sqrt();
        s += &format!(" {f}:{std:.2}/{mx:.0}{}", if bad > 0 { format!("!{bad}") } else { String::new() });
    }
    crate::info!("{what}:{s}");
    Ok(())
}
