//! End-to-end reference-to-video pipeline (ComfyUI `video_minimax_h3_r2v`):
//! references -> Qwen3-VL conditioning + VAE latents -> DiT sampling (res_multistep) -> video/audio decode -> mp4.
//!
//! Phases keep GPU memory bounded on a 24 GB card: text encoder (evicted) -> VAE encode (evicted) -> DiT (evicted) -> VAE decode.
use crate::cuda::Device;
use crate::dit_h3::{Dit, DitInputs, RefBlock, RefKind};
use crate::fit::{fit_to_canvas, Fit};
use crate::image::Rgb8;
use crate::media::{self, Ffmpeg, VideoWriter, AUDIO_SR, FPS};
use crate::sampler;
use crate::te_h3::{RefItem, TextEncoder};
use crate::tensor::{DType, Tensor};
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
    pub crf: u32,
    pub out: PathBuf,
}

#[derive(Clone, Debug)]
pub struct Paths {
    pub dit: PathBuf,
    pub text_encoder: PathBuf,
    pub video_vae: PathBuf,
    pub audio_vae: PathBuf,
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

pub fn run(dev: Arc<Device>, req: &Request, paths: &Paths, ff: &Ffmpeg) -> Result<()> {
    let t_all = std::time::Instant::now();
    ensure!(req.ref_images.len() <= MAX_REF_IMAGES, "at most {MAX_REF_IMAGES} reference images");
    ensure!(req.ref_videos.len() <= MAX_REF_VIDEOS, "at most {MAX_REF_VIDEOS} reference videos");
    ensure!(req.ref_audios.len() <= MAX_REF_AUDIOS, "at most {MAX_REF_AUDIOS} reference audio clips");
    let (frame_count, latent_t, audio_t) = temporal_shape(req.frames);
    check_limits(req.width, req.height, frame_count)?;
    let (lh, lw) = (req.height / 16, req.width / 16);
    crate::info!(
        "output: {}x{} @ {} fps, {} frames ({:.2} s), latent {}x{}x{} (+{} audio frames), {} steps, seed {}",
        req.width, req.height, FPS, frame_count, frame_count as f64 / FPS as f64, latent_t, lh, lw, audio_t, req.steps, req.seed
    );

    // ---- 1. load / resize references (CPU) -----------------------------------------------------------------
    let mut images: Vec<Image> = Vec::new();
    for p in &req.ref_images {
        let mut img = load_image(ff, p).with_context(|| format!("reference image {}", p.display()))?;
        if let (Some(f), true) = (req.fit_first, images.is_empty()) {
            let (fitted, padded) = fit_to_canvas(&img, req.width, req.height, f);
            crate::info!("first reference {}x{} fitted to the {}x{} canvas ({:?}{})", img.w, img.h, req.width, req.height, f, if padded { ", black bars to be filled by the model" } else { "" });
            img = fitted;
        }
        let (tw, th) = ref_image_dims(img.w, img.h, req.width, req.height, req.ref_image_size);
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
        let (mut rgb, mut n) = ff.decode_video(&v.path, v.start, v.dur, cw, ch, frame_count)?;
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
    let cond = {
        let te = TextEncoder::load(dev.clone(), &paths.text_encoder).context("loading text encoder")?;
        let c = te.encode(&req.prompt, &items).context("text encoding")?;
        dev.sync()?;
        crate::info!("text encoder: {} tokens in {:.1}s", c.tags.len(), t0.elapsed().as_secs_f64());
        c
    }; // text encoder dropped here
    dev.sync()?;

    // ---- 3. reference latents (VAEs) -------------------------------------------------------------------------
    let t0 = std::time::Instant::now();
    let mut refs: Vec<RefBlock> = Vec::new();
    {
        let vvae = VideoVae::load(dev.clone(), &paths.video_vae).context("loading video VAE")?;
        let avae = AudioVae::load(dev.clone(), &paths.audio_vae).context("loading audio VAE")?;
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
    let (xv, xa) = {
        let t0 = std::time::Instant::now();
        let dit = Dit::load(dev.clone(), &paths.dit).context("loading DiT")?;
        let inp = DitInputs {
            text: &cond.context,
            text_tags: &cond.tags,
            refs: &refs,
            latent_t,
            latent_h: lh,
            latent_w: lw,
            audio_t,
            seed: req.seed,
            cond_noise_aug: None,
        };
        let mut run = dit.prepare(&inp).context("preparing DiT run")?;
        let xv = sampler::randn(&dev, &[24, latent_t, lh, lw], req.seed)?;
        let xa = sampler::randn(&dev, &[32, 2, audio_t], req.seed ^ 0x9E37_79B9_7F4A_7C15)?;
        let sigmas = sampler::simple_sigmas(req.steps, sampler::SHIFT_VIDEO);
        let bar = indicatif::ProgressBar::new(req.steps as u64);
        bar.set_style(indicatif::ProgressStyle::with_template("  sampling {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
        sampler::res_multistep(
            &dev,
            &sigmas,
            &xv,
            &xa,
            &mut |v, a, sigma, ov, oa| dit.forward(&mut run, v, a, sigma, ov, oa),
            &|i, _| bar.set_position(i as u64),
        )?;
        bar.finish_and_clear();
        crate::info!("sampled {} steps in {:.1}s", req.steps, t0.elapsed().as_secs_f64());
        (xv, xa)
    }; // DiT dropped
    dev.sync()?;
    drop(refs);
    drop(cond);

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
        let avae = AudioVae::load(dev.clone(), &paths.audio_vae)?;
        let wav = avae.decode(&za, audio_t).context("decoding audio")?;
        let p = if req.out == std::path::Path::new("-") {
            std::env::temp_dir().join(format!("loki-reshoot-{}.wav", std::process::id()))
        } else {
            req.out.with_extension("tmp.wav")
        };
        media::write_wav_f32(&p, &wav, audio_t * 800, AUDIO_SR)?;
        wav_path = Some(p);
    }
    drop(xa);
    let mut writer = VideoWriter::new(ff, &req.out, req.width, req.height, wav_path.as_deref(), req.crf)?;
    {
        let vvae = VideoVae::load(dev.clone(), &paths.video_vae).context("loading video VAE")?;
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
