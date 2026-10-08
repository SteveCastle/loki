//! End-to-end pipeline: reference image -> (VAE encode, Qwen3-VL conditioning) -> flow-matching sampling
//! with the Qwen Image 2.1 DiT -> VAE decode -> PNG.
use crate::cuda::Device;
use crate::dit::{self, Dit, RefLatent, Run};
use crate::image::{reference_size, Rgb8};
use crate::ops;
use crate::tensor::{DType, Tensor};
use crate::text_encoder::{Conditioning, TextEncoder};
use crate::vae::Vae;
use anyhow::{ensure, Context, Result};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use std::sync::Arc;

/// How reference images are sized before the model sees them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefPolicy {
    /// Generic editing: keep the input's own size (rounded to 32); only shrink it to the output's pixel
    /// budget when it is larger than the output (reference tokens are attended by every target token).
    Native,
    /// The wallpaper presets: shrink to half the output's budget, upscale small inputs to `MIN_REFERENCE`.
    Wallpaper,
}

#[derive(Clone, Debug)]
pub struct Settings {
    /// size the model generates at (multiples of 16)
    pub target_w: usize,
    pub target_h: usize,
    /// size of the delivered image; differs from the target by < 16 px per side when the requested size is not a
    /// multiple of 16 (the result is then resampled with Lanczos to exactly this size)
    pub out_w: usize,
    pub out_h: usize,
    pub steps: usize,
    pub shift: f32,
    pub seed: u64,
    /// ComfyUI "resolution": 0 = automatic (per `ref_policy`), else the reference's size parameter
    pub ref_resolution: usize,
    pub ref_policy: RefPolicy,
    /// the final prompt text (`<imageN>` tags refer to the references in order)
    pub prompt: String,
}

/// Smallest reference the model sees by default (ComfyUI's node default resolution). A tiny reference gives
/// the DiT only a handful of spatially aligned tokens to copy from and the output re-draws the geometry
/// instead of enlarging it (a 400 px frame: SSIM 0.91 to the aligned source at native size, 0.94 at 1024).
pub const MIN_REFERENCE: usize = 1024;

/// ComfyUI "simple" scheduler over ModelSamplingFlux(shift): sigma(t) = e^mu / (e^mu + 1/t - 1) with mu = shift,
/// 10000 discrete timesteps, sampled from the top: sigmas[-(1 + floor(i * 10000/steps))], then 0.
pub fn simple_sigmas(steps: usize, shift: f32) -> Vec<f32> {
    let total = 10000usize;
    let ss = total as f64 / steps as f64;
    let mut out = Vec::with_capacity(steps + 1);
    for i in 0..steps {
        let idx = total - 1 - (i as f64 * ss) as usize; // 0-based index into sigmas (ascending t)
        let t = (idx + 1) as f64 / total as f64;
        let mu = shift as f64;
        let s = mu.exp() / (mu.exp() + (1.0 / t - 1.0));
        out.push(s as f32);
    }
    out.push(0.0);
    out
}

/// Per-image prepared inputs (phase 1 of a batch).
pub struct Encoded {
    pub cond: Conditioning,
    /// the reference latents, in slot order
    pub refs: Vec<RefLatent>,
}

pub struct Pipeline {
    pub dev: Arc<Device>,
    pub settings: Settings,
}

impl Pipeline {
    pub fn new(dev: Arc<Device>, settings: Settings) -> Result<Pipeline> {
        ops::gemm_init(&dev)?;
        ops::attn_init(&dev)?;
        Vae::init_kernels(&dev)?;
        Ok(Pipeline { dev, settings })
    }

    /// Resize the input like TextEncodeQwenImage21 does (lanczos to multiples of 32). With
    /// ref_resolution 0 the native size is kept unless it exceeds half the output's pixel budget, in which
    /// case the reference is scaled to that budget (keeps the KV cache and attention cost bounded: the
    /// reference tokens are attended by every target token at every step), or falls below MIN_REFERENCE
    /// pixels per side, in which case it is upscaled to that.
    pub fn prepare_reference(&self, img: &Rgb8) -> Rgb8 {
        let mut res = self.settings.ref_resolution;
        if res == 0 {
            let target = self.settings.target_w * self.settings.target_h;
            match self.settings.ref_policy {
                RefPolicy::Wallpaper => {
                    let budget = target / 2;
                    if img.w * img.h > budget {
                        res = (budget as f64).sqrt() as usize;
                        crate::info!("  reference {}x{} is larger than the output; scaling it to about {} pixels per side", img.w, img.h, res);
                    } else if img.w * img.h < MIN_REFERENCE * MIN_REFERENCE {
                        res = MIN_REFERENCE;
                    }
                }
                RefPolicy::Native => {
                    if img.w * img.h > target {
                        res = (target as f64).sqrt() as usize;
                        crate::info!("  reference {}x{} is larger than the output; scaling it to about {} pixels per side", img.w, img.h, res);
                    }
                }
            }
        }
        let (nw, nh) = reference_size(img.w, img.h, res);
        img.resize_lanczos(nw, nh)
    }

    /// Encode one job: `imgs[0]` is `<image1>`, further images `<image2>`, ... (all sent to the text encoder and VAE).
    pub fn encode(&self, te: &TextEncoder, vae: &Vae, imgs: &[Rgb8]) -> Result<Encoded> {
        let dev = &self.dev;
        ensure!(!imgs.is_empty(), "at least one input image is needed");
        let rs: Vec<Rgb8> = imgs.iter().map(|i| self.prepare_reference(i)).collect();
        let t0 = std::time::Instant::now();
        let cond = te.encode(&self.settings.prompt, &rs).context("text encoder")?;
        dev.sync()?;
        let t_te = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let mut refs = Vec::with_capacity(rs.len());
        for (r, &slot) in rs.iter().zip(&cond.slots) {
            let lat = vae.encode(&r.to_f32(), r.h, r.w).context("vae encode")?;
            let (rh, rw) = (r.h / 16, r.w / 16);
            let latent = Tensor::new(dev, DType::BF16, &[rh * rw, 64])?;
            dit::latent_norm_in(dev, &lat, &vae.latents_mean, &vae.latents_std, &latent)?;
            dev.sync()?;
            crate::info!("  <image{}> {}x{}: {} latent tokens (slot {})", refs.len() + 1, r.w, r.h, rh * rw, slot);
            refs.push(RefLatent { slot, latent, rh, rw });
        }
        crate::info!("  {} text tokens; text encoder {:.2}s, vae encode {:.2}s", cond.context.shape[0], t_te, t0.elapsed().as_secs_f64());
        Ok(Encoded { cond, refs })
    }

    /// Flow-matching Euler sampling. Returns the denormalized latent [Nt, 64] bf16 for the VAE.
    pub fn sample(&self, model: &Dit, enc: &Encoded, vae_mean: &Tensor, vae_std: &Tensor, progress: &dyn Fn(usize, usize)) -> Result<Tensor> {
        let dev = &self.dev;
        let s = &self.settings;
        ensure!(s.target_w % 16 == 0 && s.target_h % 16 == 0, "target size must be a multiple of 16");
        let (h, w) = (s.target_h / 16, s.target_w / 16);
        let nt = h * w;
        let mut run: Run = model.prepare(&enc.cond.context, &enc.refs, h, w).context("preparing dit run")?;
        // noise (f32, token-major [Nt, 64])
        let mut rng = rand::rngs::StdRng::seed_from_u64(s.seed);
        let noise: Vec<f32> = (0..nt * 64).map(|_| StandardNormal.sample(&mut rng)).collect();
        let sigmas = simple_sigmas(s.steps, s.shift);
        let x: Vec<f32> = noise.iter().map(|v| v * sigmas[0]).collect();
        let x = Tensor::from_f32(dev, &x, &[nt, 64])?;
        let xb = Tensor::new(dev, DType::BF16, &[nt, 64])?;
        let v = Tensor::new(dev, DType::BF16, &[nt, 64])?;
        for i in 0..s.steps {
            progress(i, s.steps);
            ops::to_bf16(dev, &x, &xb)?;
            model.forward(&mut run, &xb, sigmas[i], &v).with_context(|| format!("dit step {i}"))?;
            // x += v * (sigma_next - sigma)
            dit::axpy(dev, &x, &v, sigmas[i + 1] - sigmas[i])?;
            if i % 4 == 3 {
                dev.sync()?; // keep the host from running too far ahead; also surfaces errors early
            }
        }
        progress(s.steps, s.steps);
        model.prof.report();
        let out = Tensor::new(dev, DType::BF16, &[nt, 64])?;
        dit::latent_norm_out(dev, &x, vae_mean, vae_std, &out)?;
        Ok(out)
    }

    pub fn decode(&self, vae: &Vae, latent: &Tensor) -> Result<Rgb8> {
        let s = &self.settings;
        let (h, w) = (s.target_h / 16, s.target_w / 16);
        let rgb = vae.decode(latent, h, w).context("vae decode")?;
        Ok(Rgb8 { w: s.target_w, h: s.target_h, data: rgb })
    }
}
