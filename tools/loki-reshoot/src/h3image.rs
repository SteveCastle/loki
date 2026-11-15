//! MiniMax-H3-Image: text-to-image and multi-reference image editing in one checkpoint (ComfyUI
//! `MiniMaxH3ImageConditioning` -> BasicGuider (cfg 1) -> `euler` / `simple` / 6 steps -> image VAE decode).
//!
//! The engine of loki-reshoot's H3 family run on one latent frame: the Qwen3-VL-32B conditioning encoder (`te_h3`,
//! pictures labelled "Picture i: "), the H3 DiT without its audio stream (`dit_h3::Dit::forward_image`, 6 PDD output
//! heads on the checkpoint's own sigma grid) and the image VAE (`vae_image`). loki-dream (text to image) and
//! loki-retouch (edits) call it through this module; the caller's residency cache and cancel flag come in as a `Host`.
//!
//! Phases keep a 24 GB card bounded: text encoder -> VAE (reference latents) -> DiT -> VAE decode, each dropped after
//! its phase on the command line (a worker keeps what fits).
use crate::cuda::Device;
use crate::dit_h3::{Dit, DitInputs, RefBlock, RefKind};
use crate::image::Rgb8;
use crate::models::{self, ModelSpec};
use crate::te_h3::{RefItem, TextEncoder};
use crate::tensor::{DType, Tensor};
use crate::vae_image::ImageVae;
use anyhow::{anyhow, bail, ensure, Context, Result};
use std::any::Any;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

pub const MODEL_NAME: &str = "MiniMax H3 Image";
/// The model's canvas grid (2x2 DiT patch over 16x latents).
pub const CANVAS_MULTIPLE: usize = 32;
/// Distilled step count (the PDD head bank).
pub const DEFAULT_STEPS: usize = 6;
/// The ComfyUI node's default canvas side without references (4 MP; the official pipeline's minimum generation area).
pub const DEFAULT_SIDE: usize = 2048;
/// The official pipeline's `min_generation_area`: smaller requests are drawn on a canvas this large, then resampled.
pub const MIN_GENERATION_AREA: usize = 2048 * 2048;
/// Above ~20 MP the model leaves its trained range (the release table tops out at 6272x2688 = 16.9 MP).
pub const MAX_PIXELS: usize = 6272 * 3584;
/// Reference pictures (`<Picture 1>` .. `<Picture 16>`).
pub const MAX_REFS: usize = 16;
/// References: the text encoder sees them at a 1024 short edge, the VAE within a 1024-2048 short edge.
pub const REF_SHORT_EDGE: usize = 1024;
pub const REF_VAE_MAX_SHORT_EDGE: usize = 2048;

/// The release's recommended ~16 MP sizes (W, H) by aspect ratio.
pub const NATIVE_SIZES: [(&str, usize, usize); 9] = [
    ("1:1", 4096, 4096),
    ("4:3", 4736, 3552),
    ("3:4", 3552, 4736),
    ("3:2", 5024, 3360),
    ("2:3", 3360, 5024),
    ("16:9", 5472, 3072),
    ("9:16", 3072, 5472),
    ("21:9", 6272, 2688),
    ("9:21", 2688, 6272),
];

pub const HF_H3_IMAGE: &str = "https://huggingface.co/Comfy-Org/MiniMax-H3-Image/resolve/main/";
pub const DIT: ModelSpec = ModelSpec { what: "MiniMax H3 Image diffusion model", patterns: &["minimax_h3_image", "int8"], repo: HF_H3_IMAGE, hf_path: "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors", size_gb: 21.0 };
pub const VAE: ModelSpec = ModelSpec { what: "MiniMax H3 Image VAE", patterns: &["minimax_h3_image_vae"], repo: HF_H3_IMAGE, hf_path: "vae/minimax_h3_image_vae_fp16.safetensors", size_gb: 1.2 };
/// The H3 family's Qwen3-VL-32B conditioning encoder (shared with loki-reshoot's video model).
pub const TEXT_ENCODER: ModelSpec = models::TEXT_ENCODER;

/// Whether a diffusion model file is MiniMax-H3-Image (an H3 DiT without the audio stream), from its tensor names.
pub fn is_h3_image(has: &dyn Fn(&str) -> bool) -> bool {
    let with = |n: &str| ["", "model.diffusion_model.", "diffusion_model."].iter().any(|p| has(&format!("{p}{n}")));
    with("video_patch_proj.weight") && with("condition_proj.weight") && !with("audio_patch_proj.weight")
}

/// Whether the file is an H3 audio-video DiT (a loki-reshoot model, not an image one).
pub fn is_h3_video(has: &dyn Fn(&str) -> bool) -> bool {
    let with = |n: &str| ["", "model.diffusion_model.", "diffusion_model."].iter().any(|p| has(&format!("{p}{n}")));
    with("video_patch_proj.weight") && with("condition_proj.weight") && with("audio_patch_proj.weight")
}

// ---- sizes (ports of nodes_minimax_h3.py) ------------------------------------------------------------------------

fn round_grid(x: f64) -> usize {
    ((x / CANVAS_MULTIPLE as f64).round() as usize * CANVAS_MULTIPLE).max(CANVAS_MULTIPLE)
}

/// `_size_short_edge`: aspect-preserving scale putting the short edge inside [low, high], on the canvas grid.
pub fn size_short_edge(w: usize, h: usize, low: usize, high: usize) -> (usize, usize) {
    let short = w.min(h) as f64;
    let scale = short.clamp(low as f64, high as f64) / short;
    (round_grid(w as f64 * scale), round_grid(h as f64 * scale))
}

/// The VAE copy of a reference picture (also the default canvas of an edit: Picture 1's).
pub fn ref_vae_size(w: usize, h: usize) -> (usize, usize) {
    size_short_edge(w, h, REF_SHORT_EDGE, REF_VAE_MAX_SHORT_EDGE)
}

/// The text encoder's copy of a reference picture.
pub fn ref_vision_size(w: usize, h: usize) -> (usize, usize) {
    size_short_edge(w, h, REF_SHORT_EDGE, REF_SHORT_EDGE)
}

/// The canvas a requested output is drawn on: the request on the 32 px grid, scaled up (ratio kept) to the official
/// pipeline's minimum generation area when it is smaller; the result is then resampled to the request.
pub fn generation_canvas(w: usize, h: usize) -> (usize, usize) {
    let area = (w * h) as f64;
    let s = if area < MIN_GENERATION_AREA as f64 { (MIN_GENERATION_AREA as f64 / area).sqrt() } else { 1.0 };
    (round_grid(w as f64 * s), round_grid(h as f64 * s))
}

// ---- schedule --------------------------------------------------------------------------------------------------

/// ComfyUI `MiniMaxH3ImageModelSampling.sigma`: linear in the PDD grid index (`grid` decreasing from 1 to 0).
fn grid_sigma(grid: &[f64], t: f64) -> f64 {
    let asc: Vec<f64> = grid.iter().rev().copied().collect();
    let n = asc.len() - 1;
    let pos = t.clamp(0.0, 1.0) * n as f64;
    let i = (pos.floor() as usize).min(n - 1);
    asc[i] + (asc[i + 1] - asc[i]) * (pos - i as f64)
}

/// ComfyUI `simple` scheduler over that model sampling (1000 discrete timesteps), `steps` + 1 sigmas ending in 0.
/// `denoise` < 1 keeps the last `steps` of a longer schedule (KSampler img2img).
pub fn schedule(grid: &[f64], steps: usize, denoise: f64) -> Vec<f64> {
    let total = if denoise < 1.0 { ((steps as f64) / denoise.max(1e-4)) as usize } else { steps };
    // f32 like torch's buffer
    let table: Vec<f64> = (1..=1000).map(|k| grid_sigma(grid, k as f64 / 1000.0) as f32 as f64).collect();
    let ss = 1000.0 / total as f64;
    let mut s: Vec<f64> = (0..total).map(|x| table[999 - (x as f64 * ss) as usize]).collect();
    s.push(0.0);
    s.split_off(s.len() - (steps + 1))
}

// ---- host hooks ------------------------------------------------------------------------------------------------

/// What the calling tool provides: its model residency cache (worker mode), cancellation and progress reporting.
pub trait Host {
    /// Load a model through the caller's cache (`key` names the model and its loaded form, `file` its weights).
    fn load_any(&self, key: &str, file: &Path, load: &mut dyn FnMut() -> Result<Rc<dyn Any>>) -> Result<Rc<dyn Any>>;
    /// Fails once the caller cancelled the run (checked every sampling step).
    fn check_cancel(&self) -> Result<()> {
        Ok(())
    }
    /// Sampling progress: step `i` of `n` done.
    fn progress(&self, _i: usize, _n: usize) {}
}

/// A plain command-line host: every load is a fresh load, dropped after its phase.
pub struct Plain;
impl Host for Plain {
    fn load_any(&self, _key: &str, _file: &Path, load: &mut dyn FnMut() -> Result<Rc<dyn Any>>) -> Result<Rc<dyn Any>> {
        load()
    }
}

fn load<T: 'static>(host: &dyn Host, key: &str, file: &Path, mut f: impl FnMut() -> Result<T>) -> Result<Rc<T>> {
    let any = host.load_any(key, file, &mut || Ok(Rc::new(f()?) as Rc<dyn Any>))?;
    any.downcast::<T>().map_err(|_| anyhow!("model cache: {key} holds another type"))
}

thread_local! {
    static DEVICE: std::cell::RefCell<Option<Arc<Device>>> = const { std::cell::RefCell::new(None) };
}

/// The GPU context of this engine: a fresh one per call on the command line, one for the life of a resident worker
/// (`persistent`), so cached models and new buffers share a stream.
pub fn device(persistent: bool) -> Result<Arc<Device>> {
    if !persistent {
        return Device::new(0);
    }
    if let Some(d) = DEVICE.with(|c| c.borrow().clone()) {
        return Ok(d);
    }
    let d = Device::new(0)?;
    DEVICE.with(|c| *c.borrow_mut() = Some(d.clone()));
    Ok(d)
}

// ---- run -------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Paths {
    pub dit: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

impl Paths {
    /// Resolve each file: explicit path / bare name, else the model folders, else a download.
    pub fn resolve(dit: Option<PathBuf>, text_encoder: Option<PathBuf>, vae: Option<PathBuf>) -> Result<Paths> {
        Ok(Paths { dit: models::resolve(dit, &DIT)?, text_encoder: models::resolve(text_encoder, &TEXT_ENCODER)?, vae: models::resolve(vae, &VAE)? })
    }
}

pub struct Request {
    pub prompt: String,
    /// `<Picture 1>`, `<Picture 2>`, ... in order (Picture 1 is the image an edit changes); empty = text to image
    pub images: Vec<Rgb8>,
    /// the canvas the model draws on (multiples of 32)
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    /// img2img: start from this image (any size; resized to the canvas) re-noised to `strength` of the schedule
    pub init: Option<(Rgb8, f64)>,
}

pub struct Output {
    /// at the canvas size
    pub image: Rgb8,
    /// the sigmas sampled
    pub sigmas: Vec<f64>,
}

pub fn check_size(w: usize, h: usize) -> Result<()> {
    ensure!(w % CANVAS_MULTIPLE == 0 && h % CANVAS_MULTIPLE == 0 && w >= 256 && h >= 256, "the canvas must be multiples of {CANVAS_MULTIPLE} px and at least 256 px, got {w}x{h}");
    ensure!(w * h <= MAX_PIXELS, "{w}x{h} is {:.1} MP; MiniMax H3 Image draws up to ~{:.0} MP (6272x3584)", (w * h) as f64 / 1e6, MAX_PIXELS as f64 / 1e6);
    Ok(())
}

fn load_te(dev: &Arc<Device>, host: &dyn Host, p: &Path) -> Result<Rc<TextEncoder>> {
    load(host, &format!("h3-text-encoder {}", p.display()), p, || TextEncoder::load(dev.clone(), p)).context("loading the H3 text encoder")
}
fn load_vae(dev: &Arc<Device>, host: &dyn Host, p: &Path) -> Result<Rc<ImageVae>> {
    load(host, &format!("h3-image-vae {}", p.display()), p, || ImageVae::load(dev.clone(), p)).context("loading the H3 Image VAE")
}
fn load_dit(dev: &Arc<Device>, host: &dyn Host, p: &Path) -> Result<Rc<Dit>> {
    let d = load(host, &format!("h3-image-dit {}", p.display()), p, || Dit::load(dev.clone(), p)).context("loading the H3 Image DiT")?;
    ensure!(d.is_image(), "{} is an H3 audio-video model (loki-reshoot's), not MiniMax H3 Image", p.display());
    Ok(d)
}

/// Every model of a run, loaded in the order a run uses them (worker warm-up).
pub fn warm(dev: &Arc<Device>, host: &dyn Host, paths: &Paths) -> Result<()> {
    load_te(dev, host, &paths.text_encoder)?;
    load_vae(dev, host, &paths.vae)?;
    load_dit(dev, host, &paths.dit)?;
    Ok(())
}

/// The exact text the encoder gets, with each picture's vision block shown as `[Picture i]` (for --show-prompt).
pub fn show_prompt(prompt: &str, n_images: usize) -> String {
    let mut s = String::new();
    for i in 1..=n_images {
        s.push_str(&format!("Picture {i}: <|vision_start|>[Picture {i}]<|vision_end|>"));
    }
    s + prompt
}

/// One request (see `generate_many`).
pub fn generate(dev: &Arc<Device>, host: &dyn Host, paths: &Paths, req: &Request) -> Result<Output> {
    let mut out = None;
    generate_many(dev, host, paths, std::slice::from_ref(req), &mut |_, o| {
        out = Some(o);
        Ok(())
    })?;
    out.context("no output")
}

/// Several requests with one load of each model: every conditioning first (text encoder), then the reference latents
/// (VAE), then every sample (DiT), then the decodes, handing each finished image to `done(request index, output)`.
pub fn generate_many(dev: &Arc<Device>, host: &dyn Host, paths: &Paths, reqs: &[Request], done: &mut dyn FnMut(usize, Output) -> Result<()>) -> Result<()> {
    let t_all = std::time::Instant::now();
    for req in reqs {
        ensure!(req.images.len() <= MAX_REFS, "at most {MAX_REFS} reference pictures");
        ensure!(req.steps >= 1, "--steps must be at least 1");
        check_size(req.width, req.height)?;
    }

    // ---- 1. conditioning (pictures as the text encoder sees them: 1024 short edge)
    let t0 = std::time::Instant::now();
    let mut conds = Vec::with_capacity(reqs.len());
    {
        let te = load_te(dev, host, &paths.text_encoder)?;
        for req in reqs {
            host.check_cancel()?;
            let vision: Vec<RefItem> = req
                .images
                .iter()
                .map(|im| {
                    let (vw, vh) = ref_vision_size(im.w, im.h);
                    let v = if (vw, vh) == (im.w, im.h) { im.clone() } else { im.resize_lanczos(vw, vh) };
                    RefItem::Image { w: v.w, h: v.h, rgb: v.data }
                })
                .collect();
            conds.push(te.encode_image(&req.prompt, &vision).context("text encoding")?);
        }
        dev.sync()?;
    }
    crate::info!("text encoder: {} prompt(s), {} tokens in {:.1}s", reqs.len(), conds.iter().map(|c| c.tags.len()).sum::<usize>(), t0.elapsed().as_secs_f64());

    // ---- 2. reference latents (1024-2048 short edge) and img2img starts
    let mut refs: Vec<Vec<RefBlock>> = Vec::with_capacity(reqs.len());
    let mut inits: Vec<Option<Tensor>> = Vec::with_capacity(reqs.len());
    if reqs.iter().any(|r| !r.images.is_empty() || r.init.is_some()) {
        let t0 = std::time::Instant::now();
        let vae = load_vae(dev, host, &paths.vae)?;
        for req in reqs {
            host.check_cancel()?;
            let mut blocks = Vec::new();
            for im in &req.images {
                let (ew, eh) = ref_vae_size(im.w, im.h);
                let r = if (ew, eh) == (im.w, im.h) { im.clone() } else { im.resize_lanczos(ew, eh) };
                let z = vae.encode(&r.data, r.h, r.w).context("encoding a reference picture")?;
                blocks.push(RefBlock { kind: RefKind::Image, latent_t: 1, latent_h: eh / 16, latent_w: ew / 16, ref_audio_t: 0, video: Some(z), audio: None });
            }
            refs.push(blocks);
            inits.push(match &req.init {
                Some((img, _)) => {
                    let r = if (img.w, img.h) == (req.width, req.height) { img.clone() } else { img.resize_lanczos(req.width, req.height) };
                    Some(vae.encode(&r.data, r.h, r.w).context("encoding the init image")?)
                }
                None => None,
            });
        }
        dev.sync()?;
        crate::info!("reference latents: {} picture(s) in {:.1}s", reqs.iter().map(|r| r.images.len()).sum::<usize>(), t0.elapsed().as_secs_f64());
    } else {
        refs.resize_with(reqs.len(), Vec::new);
        inits.resize_with(reqs.len(), || None);
    }

    // ---- 3. sampling: euler on the PDD grid (cfg 1: one DiT pass per step)
    let mut latents: Vec<(Tensor, Vec<f64>)> = Vec::with_capacity(reqs.len());
    {
        let dit = load_dit(dev, host, &paths.dit)?;
        let grid = dit.pdd_grid().map(|g| g.to_vec()).context("this H3 Image checkpoint declares no PDD sigma grid (pdd_sigmas)")?;
        for (ri, req) in reqs.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let (lh, lw) = (req.height / 16, req.width / 16);
            crate::info!("{MODEL_NAME}: {}x{} ({:.1} MP), {} steps, seed {}, {} reference picture(s)", req.width, req.height, (req.width * req.height) as f64 / 1e6, req.steps, req.seed, req.images.len());
            let denoise = req.init.as_ref().map_or(1.0, |(_, s)| s.clamp(0.0, 1.0));
            ensure!(denoise > 0.0, "--strength 0 leaves nothing to draw");
            let sigmas = schedule(&grid, req.steps, denoise);
            let inp = DitInputs { text: &conds[ri].context, text_tags: &conds[ri].tags, refs: &refs[ri], latent_t: 1, latent_h: lh, latent_w: lw, audio_t: 0, seed: req.seed, cond_noise_aug: None };
            let mut run = dit.prepare(&inp).context("preparing the DiT run")?;
            run.sigmas = sigmas.clone();
            // torch.randn on the CPU like ComfyUI's prepare_noise, so a seed means the same there
            let noise = crate::rng::torch_randn(req.seed, 24 * lh * lw);
            let x = Tensor::from_f32(dev, &noise, &[24, 1, lh, lw])?;
            if let Some(z0) = &inits[ri] {
                // CONST noise scaling: sigma * noise + (1 - sigma) * latent
                let s0 = sigmas[0] as f32;
                axpby(dev, &x, &x, z0, s0, 1.0 - s0)?;
            }
            let v = Tensor::new(dev, DType::F32, &[24, 1, lh, lw])?;
            let steps = sigmas.len() - 1;
            for i in 0..steps {
                host.check_cancel()?;
                let (s, sn) = (sigmas[i], sigmas[i + 1]);
                dit.forward_image(&mut run, &x, s as f32, &v)?;
                // euler: x + (sn - s) * v (the model output is the flow velocity, negated by the model)
                axpby(dev, &x, &x, &v, 1.0, (sn - s) as f32)?;
                dev.sync()?;
                host.progress(i + 1, steps);
            }
            drop(run);
            crate::info!("  sampled {steps} steps in {:.1}s", t0.elapsed().as_secs_f64());
            latents.push((x, sigmas));
        }
        dit.prof.report();
    }
    drop(refs);
    drop(inits);
    drop(conds);

    // ---- 4. decode
    let vae = load_vae(dev, host, &paths.vae)?;
    for (ri, (x, sigmas)) in latents.into_iter().enumerate() {
        let t0 = std::time::Instant::now();
        let req = &reqs[ri];
        let rgb = vae.decode(&x, req.height / 16, req.width / 16).context("decoding")?;
        drop(x);
        crate::info!("  decoded in {:.1}s", t0.elapsed().as_secs_f64());
        done(ri, Output { image: Rgb8 { w: req.width, h: req.height, data: rgb }, sigmas })?;
    }
    crate::info!("{MODEL_NAME}: {} image(s) in {:.1}s", reqs.len(), t_all.elapsed().as_secs_f64());
    Ok(())
}

/// out = a * x + b * y (f32, device)
fn axpby(dev: &Device, out: &Tensor, x: &Tensor, y: &Tensor, a: f32, b: f32) -> Result<()> {
    if x.numel() != y.numel() || out.numel() != x.numel() {
        bail!("axpby: size mismatch");
    }
    dev.launch_n(
        "k_axpby3_f32",
        out.numel(),
        &[
            crate::cuda::Arg::Ptr(out.ptr),
            crate::cuda::Arg::Ptr(x.ptr),
            crate::cuda::Arg::Ptr(y.ptr),
            crate::cuda::Arg::Ptr(0),
            crate::cuda::Arg::I64(out.numel() as i64),
            crate::cuda::Arg::F32(a),
            crate::cuda::Arg::F32(b),
            crate::cuda::Arg::F32(0.0),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    const GRID: [f64; 7] = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0];

    #[test]
    fn schedule_matches_comfy_simple() {
        // calculate_sigmas(model_sampling, "simple", 6) of the release checkpoint (ComfyUI 0.45)
        let want = [1.0, 0.9676535725593567, 0.8877084255218506, 0.75, 0.5336428284645081, 0.23261038959026337, 0.0];
        let got = schedule(&GRID, 6, 1.0);
        assert_eq!(got.len(), 7);
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{got:?}");
        }
        // denoise keeps the tail of a longer schedule
        let s = schedule(&GRID, 3, 0.5);
        assert_eq!(s.len(), 4);
        assert!((s[3]).abs() < 1e-12 && s[0] < 0.8);
    }

    #[test]
    fn sizes() {
        assert_eq!(ref_vae_size(1920, 1080), (1920, 1088));
        assert_eq!(ref_vae_size(800, 600), (1376, 1024));
        assert_eq!(ref_vae_size(6000, 4000), (3072, 2048));
        assert_eq!(ref_vision_size(6000, 4000), (1536, 1024));
        assert_eq!(generation_canvas(1024, 1024), (2048, 2048));
        assert_eq!(generation_canvas(4096, 4096), (4096, 4096));
        assert_eq!(generation_canvas(1920, 1080), (2720, 1536));
    }

    #[test]
    fn detects_by_tensor_names() {
        let img = ["video_patch_proj.weight", "condition_proj.weight"];
        assert!(is_h3_image(&|n| img.contains(&n)));
        let vid = ["video_patch_proj.weight", "condition_proj.weight", "audio_patch_proj.weight"];
        assert!(!is_h3_image(&|n| vid.contains(&n)) && is_h3_video(&|n| vid.contains(&n)));
    }
}
