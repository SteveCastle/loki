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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    Landscape,
    Portrait,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub target_w: usize,
    pub target_h: usize,
    pub orientation: Orientation,
    pub steps: usize,
    pub shift: f32,
    pub seed: u64,
    /// ComfyUI "resolution": 0 keeps the reference at native size (rounded to 32), at least MIN_REFERENCE
    pub ref_resolution: usize,
    pub prompt_override: Option<String>,
    /// sequence mode (the frames of a clip): the stricter, framing-grounded prompt
    pub seq: bool,
}

/// Smallest reference the model sees by default (ComfyUI's node default resolution). A tiny reference gives
/// the DiT only a handful of spatially aligned tokens to copy from and the output re-draws the geometry
/// instead of enlarging it (a 400 px frame: SSIM 0.91 to the aligned source at native size, 0.94 at 1024).
pub const MIN_REFERENCE: usize = 1024;

fn canvas_name(o: Orientation, w: usize, h: usize) -> String {
    let g = gcd(w, h);
    let ratio = format!("{}:{}", w / g, h / g);
    match o {
        Orientation::Landscape => format!("{ratio} landscape desktop wallpaper"),
        Orientation::Portrait => format!("{ratio} portrait phone wallpaper"),
    }
}

/// The restoration paragraph. `fidelity` (sequence mode) asks for a geometrically faithful enlargement: a
/// free restoration re-draws eyes, fingers and contours of a low-resolution frame, which reads as morphing
/// between the frames of a clip.
fn restoration_paragraph(fidelity: bool) -> &'static str {
    if fidelity {
        "Enlarge <image1> into a clean, high-resolution version of the same photograph, as a faithful super-resolution of exactly what <image1> shows. Every edge, contour, and object must stay exactly where it is in <image1>, with the same shape, size, position, and angle: do not redraw, re-pose, re-light, or reinterpret anything, do not open or close eyes or mouths, and do not change the expression, the fingers, the teeth, the tongue, the hair, or any object. Only add the fine detail that the low resolution lost, removing compression artifacts, blockiness, and noise, and keeping intentional blur as it is."
    } else {
        "Restore <image1> into a clean, clear, high-resolution version of the original photograph. Remove grain, unwanted noise, compression artifacts, blockiness, and muddy detail. Correct unintended softness and defocus so the main subject appears naturally in focus, with believable texture and smooth tonal transitions. Preserve intentional depth of field and background blur."
    }
}

/// The framing paragraph of sequence mode: every frame of a clip must land on the canvas identically, so
/// the placement is spelled out (independently restored frames otherwise drift in zoom by about +-15%).
fn framing_paragraph(o: Orientation) -> &'static str {
    match o {
        Orientation::Landscape => "This is one frame of a video clip whose frames are processed one by one with the same settings, so every frame must be framed identically. Place <image1> at the centre of the canvas at the one scale where its full height exactly fills the height of the output, cropping nothing of it, so that the added surroundings lie only to its left and right. Do not zoom in or out, shift, rotate, or re-crop: scaled to the same size, the output must align exactly with <image1>.",
        Orientation::Portrait => "This is one frame of a video clip whose frames are processed one by one with the same settings, so every frame must be framed identically. Place <image1> at the centre of the canvas at the one scale where its full width exactly fills the width of the output, cropping nothing of it, so that the added surroundings lie only above and below it. Do not zoom in or out, shift, rotate, or re-crop: scaled to the same size, the output must align exactly with <image1>.",
    }
}

/// The baked-in restoration + outpainting prompt, parameterized by the target orientation; `seq` picks the
/// strict restoration wording and adds the framing paragraph.
pub fn build_prompt(o: Orientation, w: usize, h: usize, seq: bool) -> String {
    let canvas = canvas_name(o, w, h);
    let extend = match o {
        Orientation::Landscape => "to the left and right of the original image, and above and below where needed",
        Orientation::Portrait => "above and below the original image, and along the sides where needed",
    };
    let composition = match o {
        Orientation::Landscape => "landscape",
        Orientation::Portrait => "portrait",
    };
    let restoration = restoration_paragraph(seq);
    let framing = if seq { format!("{}\n\n", framing_paragraph(o)) } else { String::new() };
    format!(
        "{restoration}\n\n\
Keep the original subject identity, facial features, expression, pose, proportions, clothing, objects, and scene details. Preserve the photograph\u{2019}s original lighting, colors, exposure, and atmosphere. Use faithful restoration without beauty retouching, stylization, artificial textures, digital sharpening, halos, or exaggerated contrast.\n\n\
Expand the canvas into a {canvas} by outpainting beyond the boundaries of <image1>. Keep the original image region intact in its content and geometry, applying only the requested restoration within it. Preserve the subjects\u{2019} original size and spatial relationships. Do not stretch, squeeze, warp, crop, reposition, or redraw the original scene to fit the new ratio.\n\n\
Create the additional space by extending the surrounding environment {extend}. Continue the existing perspective, lighting, textures, depth of field, and background structures seamlessly. Add only plausible environmental continuation, without introducing new focal subjects or duplicating existing people or objects.\n\n\
{framing}\
Deliver a clean, faithfully restored photograph within a naturally expanded {composition} composition, with seamless transitions between the original image and the added surroundings."
    )
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

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

    pub fn prompt(&self) -> String {
        let s = &self.settings;
        s.prompt_override.clone().unwrap_or_else(|| build_prompt(s.orientation, s.target_w, s.target_h, s.seq))
    }

    /// Resize the input like TextEncodeQwenImage21 does (lanczos to multiples of 32). With
    /// ref_resolution 0 the native size is kept unless it exceeds half the output's pixel budget, in which
    /// case the reference is scaled to that budget (keeps the KV cache and attention cost bounded: the
    /// reference tokens are attended by every target token at every step), or falls below MIN_REFERENCE
    /// pixels per side, in which case it is upscaled to that.
    pub fn prepare_reference(&self, img: &Rgb8) -> Rgb8 {
        let mut res = self.settings.ref_resolution;
        if res == 0 {
            let budget = self.settings.target_w * self.settings.target_h / 2;
            if img.w * img.h > budget {
                res = (budget as f64).sqrt() as usize;
                eprintln!("  reference {}x{} is larger than the output; scaling it to about {} pixels per side", img.w, img.h, res);
            } else if img.w * img.h < MIN_REFERENCE * MIN_REFERENCE {
                res = MIN_REFERENCE;
            }
        }
        let (nw, nh) = reference_size(img.w, img.h, res);
        img.resize_lanczos(nw, nh)
    }

    pub fn encode(&self, te: &TextEncoder, vae: &Vae, img: &Rgb8) -> Result<Encoded> {
        let dev = &self.dev;
        let r = self.prepare_reference(img);
        let prompt = self.prompt();
        let t0 = std::time::Instant::now();
        let cond = te.encode(&prompt, std::slice::from_ref(&r)).context("text encoder")?;
        dev.sync()?;
        let t_te = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        let lat = vae.encode(&r.to_f32(), r.h, r.w).context("vae encode")?;
        let (rh, rw) = (r.h / 16, r.w / 16);
        let latent = Tensor::new(dev, DType::BF16, &[rh * rw, 64])?;
        dit::latent_norm_in(dev, &lat, &vae.latents_mean, &vae.latents_std, &latent)?;
        dev.sync()?;
        let slot = cond.slots[0];
        eprintln!("  reference {}x{} -> {} text tokens (slot {}), {} latent tokens; text encoder {:.2}s, vae encode {:.2}s", r.w, r.h, cond.context.shape[0], slot, rh * rw, t_te, t0.elapsed().as_secs_f64());
        Ok(Encoded { cond, refs: vec![RefLatent { slot, latent, rh, rw }] })
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
