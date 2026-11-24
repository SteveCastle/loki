//! End-to-end reference-to-video pipeline (ComfyUI `video_minimax_h3_r2v`):
//! references -> Qwen3-VL conditioning + VAE latents -> DiT sampling (res_multistep, or Euler for the distilled FastH3)
//! -> video/audio decode -> mp4.
//!
//! Two-pass mode (`Request::two_pass`, the ErosMax "two-pass optimized" ComfyUI workflow): pass 1 samples at a low
//! resolution, the learned latent upscaler (`upscaler`) enlarges the video latent to the output size, and pass 2 re-noises
//! it to a partial-denoise sigma and refines it in a few steps. The soundtrack is pass 1's (pass 2 still sees the
//! re-noised audio, as the joint model needs it, but its audio output is discarded, as in the workflow).
//!
//! Phases keep GPU memory bounded on a 24 GB card: text encoder (evicted) -> VAE encode (evicted) -> DiT (evicted) -> VAE decode.
use crate::cuda::Device;
use crate::dit_h3::{Dit, DitInputs, Guide, RefBlock, RefKind};
use crate::extend::{self, Extend};
use crate::fit::{fit_to_canvas, Fit};
use crate::image::Rgb8;
use crate::media::{self, Ffmpeg, VideoWriter, AUDIO_SR, FPS};
use crate::sampler;
use crate::te_h3::{RefItem, TextEncoder};
use crate::tensor::{DType, Tensor};
use crate::upscaler::Upscaler;
use crate::vae_audio::AudioVae;
use crate::vae_video::VideoVae;
use anyhow::{ensure, Context, Result};
use std::path::PathBuf;
use std::sync::Arc;

// ---- model capabilities (see README "Limits") -------------------------------------------------------------
pub const CANVAS_MULTIPLE: usize = 32;
/// Trained duration range of the model: ~5.2 s .. ~15 s at 24 fps (124 .. 362 frames); shorter works but is outside training.
pub const MIN_TRAINED_FRAMES: usize = 124;
pub const MAX_FRAMES: usize = 362;
/// "Up to 2K": 1920x1088 is the largest size in ComfyUI's template table; hard cap a bit above (2048x1152).
pub const WARN_PIXELS: usize = 1920 * 1088;
pub const MAX_PIXELS: usize = 2048 * 1152;
pub const MAX_REF_IMAGES: usize = 9;
pub const MAX_REF_VIDEOS: usize = 3;
pub const MAX_REF_AUDIOS: usize = 3;
const REF_IMAGE_SHORT_EDGE: f64 = 2048.0;
const BASE_SHORT_EDGE: f64 = 768.0;
const CANVAS_MAX_PIXELS: f64 = 768.0 * 1344.0;
const MAX_REF_AUDIO_SECONDS: f64 = 30.0;
const AUDIO_LATENT_FPS: f64 = 40.0;
const AUDIO_SHIFT_SCALE: f32 = (sampler::SHIFT_VIDEO / sampler::SHIFT_AUDIO) as f32;

#[derive(Clone, Debug)]
pub struct MediaSpec {
    pub path: PathBuf,
    pub start: f64,
    pub dur: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefImageSize {
    Match,
    Max,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplerKind {
    Euler,
    ResMultistep,
    ErSde,
}

/// The refine stage of a two-pass run.
#[derive(Clone, Debug)]
pub struct TwoPass {
    /// pass 1 canvas (multiples of 32); the output canvas is the request's width x height
    pub first_width: usize,
    pub first_height: usize,
    pub refine_steps: usize,
    /// ComfyUI BasicScheduler denoise of the refine schedule (`simple`)
    pub refine_denoise: f64,
    pub refine_sampler: SamplerKind,
}

#[derive(Clone, Debug)]
pub struct Request {
    pub prompt: String,
    pub ref_images: Vec<PathBuf>,
    pub ref_videos: Vec<MediaSpec>,
    pub ref_audios: Vec<MediaSpec>,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub steps: usize,
    pub seed: u64,
    pub ref_image_size: RefImageSize,
    /// fit the FIRST reference image to the output canvas (pad/crop/stretch) before it becomes <Picture 1>
    pub fit_first: Option<Fit>,
    pub video_audio: bool,
    pub no_audio: bool,
    /// Video Sparse Attention on a model that supports it (FastH3); otherwise ignored
    pub sparse: bool,
    /// A few-step distilled (turbo) model: plain Euler steps instead of res_multistep. Implied by PDD output heads.
    pub distilled: bool,
    /// Sampler of the (first) pass; None = the recipe's (Euler when distilled, else res_multistep)
    pub sampler: Option<SamplerKind>,
    pub scheduler: sampler::Scheduler,
    pub two_pass: Option<TwoPass>,
    /// a long video of chained chunks (see `extend`); None = one run (up to ~15 s)
    pub extend: Option<Extend>,
    pub crf: u32,
    pub out: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub dit: PathBuf,
    pub text_encoder: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
    /// the latent upscaler (two-pass runs only)
    pub upscaler: Option<PathBuf>,
}

// ---- geometry helpers (ports of nodes_minimax_h3.py) --------------------------------------------------------
/// Smallest frame count >= n with n % 17 == 5.
pub fn align_frame_count(mut n: usize) -> usize {
    while n % 17 != 5 {
        n += 1;
    }
    n
}
pub fn video_latent_t(frames: usize) -> usize {
    if frames <= 5 {
        2
    } else {
        (frames - 5) / 17 * 5 + 2
    }
}
/// (frame_count, latent_t, audio_latent_t) for a requested length.
pub fn temporal_shape(length: usize) -> (usize, usize, usize) {
    let fc = align_frame_count(length.max(5));
    let duration = fc as f64 / FPS as f64;
    let audio_t = round_half_even(duration * AUDIO_LATENT_FPS);
    (fc, video_latent_t(fc), audio_t)
}
fn round_half_even(x: f64) -> usize {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        (r - 1.0) as usize
    } else {
        r as usize
    }
}
fn round32(x: f64) -> usize {
    (((x / CANVAS_MULTIPLE as f64).round() as usize) * CANVAS_MULTIPLE).max(CANVAS_MULTIPLE)
}
/// 768-short-edge canvas with a 768*1344 area cap, per-axis rounded to 32 (used for reference videos).
pub fn adapt_canvas(w: usize, h: usize) -> (usize, usize) {
    let ratio = w as f64 / h as f64;
    let (mut nw, mut nh) = if ratio >= 1.0 { (BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE) } else { (BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio) };
    if nw * nh > CANVAS_MAX_PIXELS {
        let s = (CANVAS_MAX_PIXELS / (nw * nh)).sqrt();
        nw *= s;
        nh *= s;
    }
    (round32(nw), round32(nh))
}
/// Reference image size: aspect-preserving, downscale only; "match" = generation pixel area, "max" = 2048 short edge.
pub fn ref_image_dims(w: usize, h: usize, gen_w: usize, gen_h: usize, mode: RefImageSize) -> (usize, usize) {
    let scale = match mode {
        RefImageSize::Match => (((gen_w * gen_h) as f64) / ((w * h) as f64)).sqrt().min(1.0),
        RefImageSize::Max => (REF_IMAGE_SHORT_EDGE / w.min(h) as f64).min(1.0),
    };
    (round32(w as f64 * scale), round32(h as f64 * scale))
}

/// Frame counts of the request's chunks: [all frames] for a plain run, else the long video's chunks (`extend::plan`).
pub fn chunk_plan(req: &Request) -> Result<Vec<usize>> {
    let (fc, _, _) = temporal_shape(req.frames);
    match &req.extend {
        None => Ok(vec![fc]),
        Some(e) => {
            ensure!(req.two_pass.is_none(), "long videos run one pass for now (--passes 1)");
            extend::plan(fc, e.chunk_frames, e.context_frames)
        }
    }
}

/// Validate the requested output geometry/length against the model's capabilities.
pub fn check_limits(width: usize, height: usize, frames: usize) -> Result<()> {
    ensure!(width % CANVAS_MULTIPLE == 0 && height % CANVAS_MULTIPLE == 0 && width >= 64 && height >= 64, "size must be multiples of {CANVAS_MULTIPLE} (>= 64), got {width}x{height}");
    ensure!(width * height <= MAX_PIXELS, "{width}x{height} is {:.2} MP; the model supports up to ~2K (2048x1152, {:.2} MP)", (width * height) as f64 / 1e6, MAX_PIXELS as f64 / 1e6);
    if width * height > WARN_PIXELS {
        crate::info!("warning: {width}x{height} is above the largest size in ComfyUI's template table (1920x1088); expect long run times and high VRAM use");
    }
    ensure!(frames <= MAX_FRAMES, "{frames} frames ({:.1} s) exceeds the model's ~15 s limit ({MAX_FRAMES} frames)", frames as f64 / FPS as f64);
    if frames < MIN_TRAINED_FRAMES {
        crate::info!("note: {frames} frames ({:.1} s) is shorter than the trained range (124-362 frames, 5.2-15 s); it runs, but quality may be lower", frames as f64 / FPS as f64);
    }
    Ok(())
}

// ---- prepared reference data ---------------------------------------------------------------------------------
struct Image {
    w: usize,
    h: usize,
    rgb: Vec<u8>,
}
struct Video {
    w: usize,
    h: usize,
    n: usize,
    rgb: Vec<u8>,
    /// planar stereo 32 kHz soundtrack (paired `<Audio j>`), if any
    audio: Option<(Vec<f32>, usize)>,
}

fn load_image(ff: &Ffmpeg, path: &std::path::Path) -> Result<Rgb8> {
    match Rgb8::load(path) {
        Ok(i) => Ok(i),
        Err(first) => match ff.decode_image(path) {
            Ok((data, w, h)) => Ok(Rgb8 { w, h, data }),
            Err(_) => Err(first),
        },
    }
}

/// `record` is the generation record (compact JSON, see `genmeta`) stamped into the output's `comment` tag.
// Every model goes through the worker's residency cache (a plain load on the command line, dropped after its phase).
fn load_text_encoder(dev: &Arc<Device>, p: &std::path::Path) -> Result<std::rc::Rc<TextEncoder>> {
    crate::worker::load(&format!("text-encoder {}", p.display()), p, || TextEncoder::load(dev.clone(), p)).context("loading text encoder")
}
fn load_video_vae(dev: &Arc<Device>, p: &std::path::Path) -> Result<std::rc::Rc<VideoVae>> {
    crate::worker::load(&format!("video-vae {}", p.display()), p, || VideoVae::load(dev.clone(), p)).context("loading video VAE")
}
fn load_audio_vae(dev: &Arc<Device>, p: &std::path::Path) -> Result<std::rc::Rc<AudioVae>> {
    crate::worker::load(&format!("audio-vae {}", p.display()), p, || AudioVae::load(dev.clone(), p)).context("loading audio VAE")
}
fn load_dit(dev: &Arc<Device>, p: &std::path::Path, sparse: bool) -> Result<std::rc::Rc<Dit>> {
    // VSA from the first step by default (as FastH3 is trained); LOKI_VSA_SIGMA=s keeps steps above sigma s dense
    let vsa_sigma = std::env::var("LOKI_VSA_SIGMA").ok().and_then(|v| v.parse::<f32>().ok()).unwrap_or(f32::INFINITY);
    let sigma = sparse.then_some(vsa_sigma);
    crate::worker::load(&format!("dit {} {sigma:?}", p.display()), p, || Dit::load_opts(dev.clone(), p, sigma)).context("loading DiT")
}

/// Worker warm-up: every model of a run, loaded in the order a run uses them (what does not fit is evicted again).
pub fn warm(dev: &Arc<Device>, paths: &Paths, sparse: bool) -> Result<()> {
    load_text_encoder(dev, &paths.text_encoder)?;
    load_video_vae(dev, &paths.video_vae)?;
    load_audio_vae(dev, &paths.audio_vae)?;
    load_dit(dev, &paths.dit, sparse)?;
    Ok(())
}

pub fn run(dev: Arc<Device>, req: &Request, paths: &Paths, ff: &Ffmpeg, record: Option<&str>) -> Result<()> {
    let t_all = std::time::Instant::now();
    ensure!(req.ref_images.len() <= MAX_REF_IMAGES, "at most {MAX_REF_IMAGES} reference images");
    ensure!(req.ref_videos.len() <= MAX_REF_VIDEOS, "at most {MAX_REF_VIDEOS} reference videos");
    ensure!(req.ref_audios.len() <= MAX_REF_AUDIOS, "at most {MAX_REF_AUDIOS} reference audio clips");
    let (frame_count, latent_t, audio_t) = temporal_shape(req.frames);
    let chunks = chunk_plan(req)?;
    // the longest single run (= the whole video unless it is a long one)
    let run_frames = *chunks.iter().max().unwrap();
    check_limits(req.width, req.height, run_frames)?;
    let (lh, lw) = (req.height / 16, req.width / 16);
    // the canvas the references are sized for and pass 1 samples on (the workflow's MiniMaxH3ReferenceToVideo runs at
    // the first-pass size)
    let (gen_w, gen_h) = req.two_pass.as_ref().map_or((req.width, req.height), |tp| (tp.first_width, tp.first_height));
    if let Some(tp) = &req.two_pass {
        check_limits(gen_w, gen_h, frame_count)?;
        ensure!(gen_w <= req.width && gen_h <= req.height, "two-pass: the first pass ({gen_w}x{gen_h}) must not be larger than the output ({}x{})", req.width, req.height);
        crate::info!(
            "output: {}x{} @ {} fps, {} frames ({:.2} s), latent {}x{}x{} (+{} audio frames), seed {}; two passes: {} steps at {gen_w}x{gen_h}, latent upscale, {} refine steps (denoise {})",
            req.width, req.height, FPS, frame_count, frame_count as f64 / FPS as f64, latent_t, lh, lw, audio_t, req.seed, req.steps, tp.refine_steps, tp.refine_denoise
        );
    } else {
        crate::info!(
            "output: {}x{} @ {} fps, {} frames ({:.2} s), latent {}x{}x{} (+{} audio frames), {} steps, seed {}",
            req.width, req.height, FPS, frame_count, frame_count as f64 / FPS as f64, latent_t, lh, lw, audio_t, req.steps, req.seed
        );
    }
    if let Some(e) = &req.extend {
        let secs: Vec<String> = chunks.iter().map(|&c| format!("{:.1}", c as f64 / FPS as f64)).collect();
        crate::info!("long video: {} chunks of {} s, each continuing the last {} frames of the one before", chunks.len(), secs.join(" / "), e.context_frames);
    }

    // ---- 1. load / resize references (CPU) -----------------------------------------------------------------
    let mut images: Vec<Image> = Vec::new();
    for p in &req.ref_images {
        let mut img = load_image(ff, p).with_context(|| format!("reference image {}", p.display()))?;
        if let (Some(f), true) = (req.fit_first, images.is_empty()) {
            let (fitted, padded) = fit_to_canvas(&img, req.width, req.height, f);
            crate::info!("first reference {}x{} fitted to the {}x{} canvas ({:?}{})", img.w, img.h, req.width, req.height, f, if padded { ", black bars to be filled by the model" } else { "" });
            img = fitted;
        }
        let (tw, th) = ref_image_dims(img.w, img.h, gen_w, gen_h, req.ref_image_size);
        let r = img.resize_lanczos(tw, th);
        images.push(Image { w: tw, h: th, rgb: r.data });
    }
    let mut videos: Vec<Video> = Vec::new();
    for v in &req.ref_videos {
        let info = ff.probe(&v.path).with_context(|| format!("reference video {}", v.path.display()))?;
        ensure!(info.has_video && info.width > 0, "{} has no video stream", v.path.display());
        let (mut cw, mut ch) = adapt_canvas(info.width, info.height);
        if info.width * info.height < cw * ch {
            cw = round32(info.width as f64);
            ch = round32(info.height as f64);
        }
        let (mut rgb, mut n) = ff.decode_video(&v.path, v.start, v.dur, cw, ch, run_frames)?;
        ensure!(n >= 5, "reference video {} has only {n} frames (need at least 5, ~0.2 s)", v.path.display());
        while n % 17 != 5 {
            n -= 1;
        }
        rgb.truncate(n * cw * ch * 3);
        let secs = n as f64 / FPS as f64;
        let audio = if req.video_audio && info.has_audio {
            let want = round_half_even(secs * AUDIO_SR as f64);
            Some(ff.decode_audio(&v.path, v.start, Some(secs), Some(want))?)
        } else {
            None
        };
        crate::info!("ref video {}: {}x{} -> canvas {cw}x{ch}, {n} frames ({secs:.2} s){}", v.path.display(), info.width, info.height, if audio.is_some() { " + soundtrack" } else { "" });
        videos.push(Video { w: cw, h: ch, n, rgb, audio });
    }
    let mut audios: Vec<(Vec<f32>, usize)> = Vec::new();
    for a in &req.ref_audios {
        let dur = a.dur.map(|d| d.min(MAX_REF_AUDIO_SECONDS)).or(Some(MAX_REF_AUDIO_SECONDS));
        let (wav, n) = ff.decode_audio(&a.path, a.start, dur, None).with_context(|| format!("reference audio {}", a.path.display()))?;
        ensure!(n >= AUDIO_SR / 20, "reference audio {} is empty", a.path.display());
        crate::info!("ref audio {}: {:.2} s", a.path.display(), n as f64 / AUDIO_SR as f64);
        audios.push((wav, n));
    }
    print_tags(req, &videos);

    // ---- 2. conditioning (text encoder, evicted afterwards) -------------------------------------------------
    let mut items: Vec<RefItem> = Vec::new();
    for im in &images {
        items.push(RefItem::Image { w: im.w, h: im.h, rgb: im.rgb.clone() });
    }
    for v in &videos {
        if v.audio.is_some() {
            items.push(RefItem::Audio);
        }
        // the encoder sees the clip at 2 fps with timestamps k/2 seconds
        let mut frames = Vec::new();
        for i in (0..v.n).step_by(FPS / 2) {
            let off = i * v.w * v.h * 3;
            frames.push((v.w, v.h, v.rgb[off..off + v.w * v.h * 3].to_vec()));
        }
        let ts: Vec<f32> = (0..frames.len()).map(|k| k as f32 / 2.0).collect();
        items.push(RefItem::Video { frames, timestamps: ts });
    }
    for _ in &audios {
        items.push(RefItem::Audio);
    }
    let t0 = std::time::Instant::now();
    // a long video's chunk prompts are encoded here too, while the text encoder is loaded
    let prompts: Vec<&str> = std::iter::once(req.prompt.as_str()).chain(req.extend.iter().flat_map(|e| e.prompts.iter().map(|p| p.as_str()))).collect();
    let conds = {
        let te = load_text_encoder(&dev, &paths.text_encoder)?;
        let mut cs = Vec::new();
        for p in &prompts {
            let c = te.encode(p, &items).context("text encoding")?;
            dev.sync()?;
            cs.push(c);
        }
        crate::info!("text encoder: {} tokens in {:.1}s", cs.iter().map(|c| c.tags.len().to_string()).collect::<Vec<_>>().join(" + "), t0.elapsed().as_secs_f64());
        cs
    }; // text encoder dropped here
    dev.sync()?;

    // ---- 3. reference latents (VAEs) -------------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let mut refs: Vec<RefBlock> = Vec::new();
    {
        let vvae = load_video_vae(&dev, &paths.video_vae)?;
        let avae = load_audio_vae(&dev, &paths.audio_vae)?;
        for im in &images {
            let z = vvae.encode(&im.rgb, 1, im.h, im.w).context("encoding reference image")?;
            refs.push(RefBlock { kind: RefKind::Image, latent_t: 1, latent_h: im.h / 16, latent_w: im.w / 16, ref_audio_t: 0, video: Some(z), audio: None });
        }
        for v in &videos {
            let z = vvae.encode(&v.rgb, v.n, v.h, v.w).context("encoding reference video")?;
            let tl = z.shape[1];
            ensure!(tl == video_latent_t(v.n), "reference video latent frames {tl} != expected {}", video_latent_t(v.n));
            let (audio, rt, kind) = match &v.audio {
                Some((wav, n)) => {
                    let a = encode_ref_audio(&avae, wav, *n).context("encoding reference video soundtrack")?;
                    let rt = a.shape[2];
                    (Some(a), rt, RefKind::VideoAudio)
                }
                None => (None, 0, RefKind::Video),
            };
            refs.push(RefBlock { kind, latent_t: tl, latent_h: v.h / 16, latent_w: v.w / 16, ref_audio_t: rt, video: Some(z), audio });
        }
        for (wav, n) in &audios {
            let a = encode_ref_audio(&avae, wav, *n).context("encoding reference audio")?;
            let rt = a.shape[2];
            refs.push(RefBlock { kind: RefKind::Audio, latent_t: 0, latent_h: 0, latent_w: 0, ref_audio_t: rt, video: None, audio: Some(a) });
        }
        dev.sync()?;
    } // VAEs dropped
    dev.sync()?;
    crate::info!("reference latents: {} block(s) in {:.1}s", refs.len(), t0.elapsed().as_secs_f64());

    // ---- 4. DiT sampling ----------------------------------------------------------------------------------------
    let (xv, xa, latent_t, audio_t) = {
        let t0 = std::time::Instant::now();
        // VSA from the first step by default (as FastH3 is trained); LOKI_VSA_SIGMA=s keeps steps above sigma s dense
        let dit = load_dit(&dev, &paths.dit, req.sparse)?;
        // a few-step distilled model (--model fast, or any with PDD heads) is trained for plain Euler steps; the base
        // model for res_multistep
        let distilled = req.distilled || dit.pdd_heads() > 1;
        let first_sampler = req.sampler.unwrap_or(if distilled { SamplerKind::Euler } else { SamplerKind::ResMultistep });
        let refine_steps = req.two_pass.as_ref().map_or(0, |tp| tp.refine_steps);
        let total_steps = req.steps * chunks.len() + refine_steps;
        let show_bar = std::io::IsTerminal::is_terminal(&std::io::stderr()) && !crate::log::quiet();
        let bar = if show_bar { indicatif::ProgressBar::new(total_steps as u64) } else { indicatif::ProgressBar::hidden() };
        bar.set_style(indicatif::ProgressStyle::with_template("  sampling {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
        // one sampling pass over `sigmas` of the target `t` x `h` x `w` (+ `at` audio frames) in xv / xa, conditioned on
        // `cond` and `guides`; progress counts on from `done`
        struct Target<'a> {
            cond: &'a crate::te_h3::Conditioning,
            guides: &'a [Guide],
            t: usize,
            h: usize,
            w: usize,
            at: usize,
            seed: u64,
        }
        let sample_pass = |tg: &Target, xv: &Tensor, xa: &Tensor, sigmas: &[f64], kind: SamplerKind, done: usize| -> Result<()> {
            let inp = DitInputs {
                text: &tg.cond.context,
                text_tags: &tg.cond.tags,
                guides: tg.guides,
                refs: &refs,
                latent_t: tg.t,
                latent_h: tg.h,
                latent_w: tg.w,
                audio_t: tg.at,
                seed: tg.seed,
                cond_noise_aug: None,
            };
            let mut run = dit.prepare(&inp).context("preparing DiT run")?;
            // the sigmas the model is evaluated at (PDD heads pick their interval from this schedule)
            run.sigmas = if kind == SamplerKind::ErSde { sampler::er_sde_sigmas(sigmas) } else { sigmas.to_vec() };
            let mut model = |v: &Tensor, a: &Tensor, sigma: f32, ov: &Tensor, oa: &Tensor| -> Result<()> {
                crate::worker::check_cancel()?;
                dit.forward(&mut run, v, a, sigma, ov, oa)
            };
            let progress = |i: usize, _n: usize| {
                bar.set_position((done + i) as u64);
                if !show_bar {
                    crate::info!("step {}/{total_steps}", done + i); // machine-readable progress for callers that capture stderr
                }
            };
            match kind {
                SamplerKind::Euler => sampler::euler(&dev, sigmas, xv, xa, &mut model, &progress),
                SamplerKind::ResMultistep => sampler::res_multistep(&dev, sigmas, xv, xa, &mut model, &progress),
                SamplerKind::ErSde => sampler::er_sde(&dev, sigmas, xv, xa, tg.seed ^ (done as u64), &mut model, &progress),
            }
        };
        let (lh1, lw1) = (gen_h / 16, gen_w / 16);
        let sigmas = sampler::schedule(req.scheduler, req.steps, 1.0, sampler::SHIFT_VIDEO);
        let out = if let Some(ext) = &req.extend {
            let (xv, xa) = sample_chunks(&dev, ext, &chunks, (lh1, lw1), req.seed, &mut |tg_guides, t, at, k, xv, xa| {
                let tg = Target { cond: &conds[ext.prompt_index(k)], guides: tg_guides, t, h: lh1, w: lw1, at, seed: chunk_seed(req.seed, k) };
                sample_pass(&tg, xv, xa, &sigmas, first_sampler, k * req.steps)
            })?;
            let (t, at) = (xv.shape[1], xa.shape[2]);
            (xv, xa, t, at)
        } else {
            let xv = sampler::randn(&dev, &[24, latent_t, lh1, lw1], req.seed)?;
            let xa = sampler::randn(&dev, &[32, 2, audio_t], req.seed ^ 0x9E37_79B9_7F4A_7C15)?;
            let tg = Target { cond: &conds[0], guides: &[], t: latent_t, h: lh1, w: lw1, at: audio_t, seed: req.seed };
            sample_pass(&tg, &xv, &xa, &sigmas, first_sampler, 0)?;
            match &req.two_pass {
            None => (xv, xa, latent_t, audio_t),
            Some(tp) => {
                crate::info!("pass 1: {} steps at {gen_w}x{gen_h} in {:.1}s", req.steps, t0.elapsed().as_secs_f64());
                let t1 = std::time::Instant::now();
                let up_path = paths.upscaler.as_ref().context("two-pass run without a latent upscaler")?;
                let xv_up = {
                    let up = Upscaler::load(dev.clone(), up_path).context("loading the latent upscaler")?;
                    up.upscale(&xv, lh, lw).context("latent upscale")?
                };
                drop(xv);
                dev.sync()?;
                crate::info!("latent upscale {lw1}x{lh1} -> {lw}x{lh} in {:.1}s", t1.elapsed().as_secs_f64());
                // LOKI_TWO_PASS_SKIP_REFINE (diagnostics): decode the upscaled pass-1 latent as is
                if std::env::var_os("LOKI_TWO_PASS_SKIP_REFINE").is_some() {
                    crate::info!("LOKI_TWO_PASS_SKIP_REFINE: no pass 2");
                } else {
                    // pass 1's soundtrack is the result's; pass 2 refines a re-noised copy alongside the video
                    let xa2 = Tensor::new(&dev, DType::F32, &xa.shape)?;
                    dev.dtod(xa2.ptr, xa.ptr, xa.bytes())?;
                    let sig2 = sampler::schedule(sampler::Scheduler::Simple, tp.refine_steps, tp.refine_denoise, sampler::SHIFT_VIDEO);
                    // (randn uploads synchronously: the stream must be idle, see upscaler.rs)
                    dev.sync()?;
                    let nv = sampler::randn(&dev, &xv_up.shape, req.seed ^ 0x2BA5_5000_0000_0001)?;
                    let na = sampler::randn(&dev, &xa2.shape, req.seed ^ 0x2BA5_5000_0000_0002)?;
                    sampler::noise_latent(&dev, &xv_up, &nv, sig2[0])?;
                    sampler::noise_latent(&dev, &xa2, &na, sig2[0])?;
                    dev.sync()?;
                    drop((nv, na));
                    let t2 = std::time::Instant::now();
                    let tg = Target { cond: &conds[0], guides: &[], t: latent_t, h: lh, w: lw, at: audio_t, seed: req.seed };
                    sample_pass(&tg, &xv_up, &xa2, &sig2, tp.refine_sampler, req.steps)?;
                    crate::info!("pass 2: {} refine steps at {}x{} (from sigma {:.3}) in {:.1}s", tp.refine_steps, req.width, req.height, sig2[0], t2.elapsed().as_secs_f64());
                }
                (xv_up, xa, latent_t, audio_t)
            }
            }
        };
        bar.finish_and_clear();
        crate::info!("sampled {} steps in {:.1}s", total_steps, t0.elapsed().as_secs_f64());
        dit.prof.report();
        out
    }; // DiT dropped
    dev.sync()?;
    drop(refs);
    drop(conds);

    // ---- 5. decode + mux ---------------------------------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    if let Some(d) = req.out.parent() {
        if !d.as_os_str().is_empty() {
            std::fs::create_dir_all(d)?;
        }
    }
    let mut wav_path = None;
    if !req.no_audio {
        // the carried audio variable is (sigma_v / sigma_a) * x_audio => x_audio = y / audio_scale at sigma = 0
        let za = Tensor::new(&dev, DType::F32, &xa.shape)?;
        dev.launch_n(
            "k_axpby3_f32",
            za.numel(),
            &[
                crate::cuda::Arg::Ptr(za.ptr),
                crate::cuda::Arg::Ptr(xa.ptr),
                crate::cuda::Arg::Ptr(xa.ptr),
                crate::cuda::Arg::Ptr(0),
                crate::cuda::Arg::I64(za.numel() as i64),
                crate::cuda::Arg::F32(1.0 / AUDIO_SHIFT_SCALE),
                crate::cuda::Arg::F32(0.0),
                crate::cuda::Arg::F32(0.0),
            ],
        )?;
        let avae = load_audio_vae(&dev, &paths.audio_vae)?;
        let wav = decode_audio(&dev, &avae, &za, audio_t).context("decoding audio")?;
        let p = if req.out == std::path::Path::new("-") {
            std::env::temp_dir().join(format!("loki-reshoot-{}.wav", std::process::id()))
        } else {
            req.out.with_extension("tmp.wav")
        };
        media::write_wav_f32(&p, &wav, audio_t * 800, AUDIO_SR)?;
        wav_path = Some(p);
    }
    drop(xa);
    let mut writer = VideoWriter::new(ff, &req.out, req.width, req.height, wav_path.as_deref(), req.crf, record)?;
    {
        let vvae = load_video_vae(&dev, &paths.video_vae)?;
        let total = vvae
            .decode(&xv, latent_t, lh, lw, &mut |rgb, _n| writer.write(rgb))
            .context("decoding video")?;
        ensure!(total == frame_count, "decoder produced {total} frames, expected {frame_count}");
    }
    writer.finish()?;
    if let Some(p) = wav_path {
        let _ = std::fs::remove_file(p);
    }
    crate::info!("decoded + encoded in {:.1}s", t0.elapsed().as_secs_f64());
    crate::info!("wrote {} (total {:.1}s)", req.out.display(), t_all.elapsed().as_secs_f64());
    Ok(())
}

/// Audio latent frames decoded in one call: every plain run (<= ~603); a long video's soundtrack goes in overlapping
/// windows (the decoder's memory grows with the length), cross-faded in the middle of each overlap.
const AUDIO_DECODE_WINDOW: usize = 640;
const AUDIO_DECODE_OVERLAP: usize = 48;
const AUDIO_DECODE_FADE: usize = 16;

/// Planar stereo waveform [2, t * 800] of the audio latent `za` [32, 2, t].
fn decode_audio(dev: &Arc<Device>, avae: &AudioVae, za: &Tensor, t: usize) -> Result<Vec<f32>> {
    if t <= AUDIO_DECODE_WINDOW {
        return avae.decode(za, t);
    }
    const SPF: usize = 800; // samples per audio latent frame
    let z = za.to_f32_vec(dev)?;
    let n = t * SPF;
    let mut out = vec![0f32; 2 * n];
    let mut start = 0;
    loop {
        let end = (start + AUDIO_DECODE_WINDOW).min(t);
        let w = end - start;
        let mut sub = Vec::with_capacity(64 * w);
        for s in 0..64 {
            sub.extend_from_slice(&z[s * t + start..s * t + end]);
        }
        dev.sync()?;
        let wav = avae.decode(&Tensor::from_f32(dev, &sub, &[32, 2, w])?, w)?;
        // the previous window's samples up to the fade, then a linear cross-fade, then this window's
        let c0 = if start == 0 { 0 } else { (start + (AUDIO_DECODE_OVERLAP - AUDIO_DECODE_FADE) / 2) * SPF };
        let c1 = if start == 0 { 0 } else { c0 + AUDIO_DECODE_FADE * SPF };
        for ch in 0..2 {
            for g in c0..end * SPF {
                let v = wav[ch * w * SPF + g - start * SPF];
                let o = &mut out[ch * n + g];
                *o = if g < c1 {
                    let a = (g - c0) as f32 / (c1 - c0) as f32;
                    *o * (1.0 - a) + v * a
                } else {
                    v
                };
            }
        }
        if end == t {
            return Ok(out);
        }
        start = end - AUDIO_DECODE_OVERLAP;
    }
}

/// Noise seed of chunk `k` of a long video (chunk 1 keeps the run's seed: it is the plain run of its length).
fn chunk_seed(seed: u64, k: usize) -> u64 {
    seed.wrapping_add(k as u64)
}

/// Samples a long video chunk by chunk (see `extend`). `sample(guides, latent_t, audio_t, k, xv, xa)` denoises chunk k
/// from the noise in xv / xa in place. The kept latents are collected on the host (the DiT keeps the VRAM). Returns the
/// whole video latent [24, T, h, w] and its carried audio [32, 2, Ta].
#[allow(clippy::type_complexity)]
fn sample_chunks(
    dev: &Arc<Device>,
    ext: &Extend,
    chunks: &[usize],
    (lh, lw): (usize, usize),
    seed: u64,
    sample: &mut dyn FnMut(&[Guide], usize, usize, usize, &Tensor, &Tensor) -> Result<()>,
) -> Result<(Tensor, Tensor)> {
    let area = lh * lw;
    let ctx = ext.context_frames;
    let (ctx_slots, ctx_ticks) = (video_latent_t(ctx), extend::context_audio_ticks(ctx));
    // the long latents so far, per video channel [T, h, w] and per audio channel x stream [Ta] (carried audio variable)
    let mut video: Vec<Vec<f32>> = vec![Vec::new(); 24];
    let mut audio: Vec<Vec<f32>> = vec![Vec::new(); 64];
    let mut frames = 0usize;
    for (k, &fk) in chunks.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let (_, lt, at) = temporal_shape(fk);
        // (uploads are synchronous: the stream must be idle, see upscaler.rs)
        dev.sync()?;
        let mut guides = Vec::new();
        let (mut skip_v, mut skip_a) = (0, 0);
        if k > 0 {
            // the clean tail of what is there so far, anchored at the chunk's frame 0 (start = 5k latent frames in, a
            // latent-cycle boundary: the long latent always has 5k+2)
            let have = video[0].len() / area;
            let mut v = Vec::with_capacity(24 * ctx_slots * area);
            for ch in &video {
                v.extend_from_slice(&ch[(have - ctx_slots) * area..]);
            }
            let have_a = audio[0].len();
            let (audio_offset, first_new) = extend::audio_join(frames - ctx, have_a, ctx_ticks);
            let mut a = Vec::with_capacity(64 * ctx_ticks);
            for s in &audio {
                // guides are clean latents: the carried variable over the audio scale
                a.extend(s[have_a - ctx_ticks..].iter().map(|y| y / AUDIO_SHIFT_SCALE));
            }
            guides.push(Guide {
                frame: 0,
                video: Some(Tensor::from_f32(dev, &v, &[24, ctx_slots, lh, lw])?),
                audio: Some(Tensor::from_f32(dev, &a, &[32, 2, ctx_ticks])?),
                audio_offset,
            });
            (skip_v, skip_a) = (ctx_slots, first_new.min(at));
        }
        let s = chunk_seed(seed, k);
        let xv = sampler::randn(dev, &[24, lt, lh, lw], s)?;
        let xa = sampler::randn(dev, &[32, 2, at], s ^ 0x9E37_79B9_7F4A_7C15)?;
        sample(&guides, lt, at, k, &xv, &xa)?;
        drop(guides);
        let hv = xv.to_f32_vec(dev)?;
        for (ch, dst) in video.iter_mut().enumerate() {
            dst.extend_from_slice(&hv[(ch * lt + skip_v) * area..(ch + 1) * lt * area]);
        }
        let ha = xa.to_f32_vec(dev)?;
        for (s, dst) in audio.iter_mut().enumerate() {
            dst.extend_from_slice(&ha[s * at + skip_a..(s + 1) * at]);
        }
        frames += if k == 0 { fk } else { fk - ctx };
        crate::info!("chunk {}/{}: {fk} frames ({:.1} s of video so far) in {:.1}s", k + 1, chunks.len(), frames as f64 / FPS as f64, t0.elapsed().as_secs_f64());
    }
    let t = video[0].len() / area;
    ensure!(t == video_latent_t(frames), "long video latent has {t} frames, expected {} for {frames} frames", video_latent_t(frames));
    let ta = audio[0].len();
    let xv = Tensor::from_f32(dev, &video.concat(), &[24, t, lh, lw])?;
    let xa = Tensor::from_f32(dev, &audio.concat(), &[32, 2, ta])?;
    Ok((xv, xa))
}

/// ComfyUI crops (centred) the reference waveform to a multiple of 800 samples before the audio VAE.
fn encode_ref_audio(avae: &AudioVae, wav: &[f32], n: usize) -> Result<Tensor> {
    let (start, len) = crate::vae_audio::comfy_crop_window(n);
    ensure!(len > 0, "reference audio is shorter than one latent frame (25 ms)");
    let mut c = Vec::with_capacity(2 * len);
    c.extend_from_slice(&wav[start..start + len]);
    c.extend_from_slice(&wav[n + start..n + start + len]);
    avae.encode(&c, len)
}

fn print_tags(req: &Request, videos: &[Video]) {
    crate::info!("prompt tags (use them in the prompt to point at a reference):");
    for (i, p) in req.ref_images.iter().enumerate() {
        crate::info!("  <Picture {}>  {}", i + 1, p.display());
    }
    let mut audio_n = 0;
    for (i, v) in req.ref_videos.iter().enumerate() {
        if videos[i].audio.is_some() {
            audio_n += 1;
            crate::info!("  <Audio {}>    soundtrack of {}", audio_n, v.path.display());
        }
        crate::info!("  <Video {}>    {}", i + 1, v.path.display());
    }
    for a in &req.ref_audios {
        audio_n += 1;
        crate::info!("  <Audio {}>    {}", audio_n, a.path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_shapes() {
        assert_eq!(temporal_shape(124), (124, 37, 207));
        assert_eq!(temporal_shape(22), (22, 7, 37));
        assert_eq!(temporal_shape(24), (39, 12, 65));
        assert_eq!(video_latent_t(5), 2);
    }
    #[test]
    fn canvas() {
        assert_eq!(adapt_canvas(1920, 1080), (1344, 768));
        assert_eq!(adapt_canvas(512, 512), (768, 768));
    }
}
