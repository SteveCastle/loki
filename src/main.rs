//! h3ref2va: MiniMax H3 reference-to-video (images / videos / audio -> video with sound) on a standalone CUDA engine.
use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use h3ref2va::cuda::Device;
use h3ref2va::media::{Ffmpeg, FPS};
use h3ref2va::models;
use h3ref2va::pipeline::{self, MediaSpec, Paths, RefImageSize, Request};
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum RefSize {
    Match,
    Max,
}

#[derive(Parser, Debug)]
#[command(
    name = "h3ref2va",
    version,
    about = "MiniMax H3 reference-to-video on a standalone CUDA engine: any mix of reference images, videos and audio -> mp4 with native audio",
    after_help = "REFERENCE TAGS\n  In the prompt, point at references by tag, in the order given on the command line:\n  <Picture 1>.. (--ref-image), <Video 1>.. (--ref-video), <Audio 1>.. (soundtracks of the reference videos first, then --ref-audio).\n  The tags are printed at start-up.\n\nLIMITS\n  length  : 5.2-15 s trained (124-362 frames @ 24 fps); shorter runs, > 15.08 s is refused\n  size    : multiples of 32, up to ~2K (2048x1152); ComfyUI's table tops out at 1920x1088\n  refs    : up to 9 images, 3 videos (each 5+ frames, ~0.2-15 s), 3 audio clips\n\nMODELS\n  Looked up next to the binary, in ./models, $H3_MODELS; missing ones are downloaded from Hugging Face (~42 GB)."
)]
struct Cli {
    /// Prompt text (use <Picture 1>, <Video 1>, <Audio 1> tags to address the references).
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a text file.
    #[arg(long)]
    prompt_file: Option<PathBuf>,
    /// Reference image (repeatable, up to 9). Any format the `image` crate or ffmpeg can read.
    #[arg(short = 'i', long = "ref-image")]
    ref_image: Vec<PathBuf>,
    /// Reference video FILE[@START[,DURATION]] (repeatable, up to 3). Seconds; resampled to 24 fps; its soundtrack is used too.
    #[arg(short = 'v', long = "ref-video")]
    ref_video: Vec<String>,
    /// Reference audio FILE[@START[,DURATION]] (repeatable, up to 3), e.g. music or a voice to clone.
    #[arg(short = 'a', long = "ref-audio")]
    ref_audio: Vec<String>,
    /// Do not use the soundtracks of the reference videos.
    #[arg(long)]
    no_video_audio: bool,
    /// Output mp4 (default: h3_<seed>.mp4 in the current directory).
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Duration in seconds (snapped up to the model's 17k+5 frame grid at 24 fps). Default 5.
    #[arg(short, long, conflicts_with = "frames")]
    duration: Option<f64>,
    /// Exact frame count instead of --duration (snapped up to 17k+5: 5, 22, 39, ... 124, ... 362).
    #[arg(long)]
    frames: Option<usize>,
    /// Output size WxH (multiples of 32). Default: the first reference's aspect at --megapixels.
    #[arg(long)]
    size: Option<String>,
    /// Aspect ratio W:H for the default size (e.g. 16:9). Default: from the first reference image/video, else 16:9.
    #[arg(long)]
    aspect: Option<String>,
    /// Pixel budget for the default size, in megapixels.
    #[arg(long, default_value_t = 0.4)]
    megapixels: f64,
    /// Sampling steps (res_multistep, `simple` schedule, shift 12).
    #[arg(long, default_value_t = 20)]
    steps: usize,
    /// Noise seed (default: random, printed).
    #[arg(long)]
    seed: Option<u64>,
    /// Reference image sizing: `match` scales each down to the generation's pixel area, `max` keeps up to 2048 px short edge
    /// (better identity fidelity, several times slower).
    #[arg(long, value_enum, default_value_t = RefSize::Match)]
    ref_image_size: RefSize,
    /// Do not generate/mux audio.
    #[arg(long)]
    no_audio: bool,
    /// x264 CRF of the output.
    #[arg(long, default_value_t = 18)]
    crf: u32,
    /// Diffusion model file. Default: auto-discover (or download).
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Qwen3-VL text encoder file.
    #[arg(long)]
    text_encoder: Option<PathBuf>,
    /// Video VAE file.
    #[arg(long)]
    video_vae: Option<PathBuf>,
    /// Audio VAE file.
    #[arg(long)]
    audio_vae: Option<PathBuf>,
    /// ffmpeg executable (ffprobe next to it). Default: next to the binary, PATH, or auto-download on Windows.
    #[arg(long)]
    ffmpeg: Option<PathBuf>,
}

/// `FILE[@START[,DURATION]]`; the `@` suffix is only taken when it parses as numbers (paths may contain '@').
fn parse_media(s: &str) -> Result<MediaSpec> {
    if let Some((path, spec)) = s.rsplit_once('@') {
        let mut it = spec.splitn(2, ',');
        let start = it.next().and_then(|a| a.trim().parse::<f64>().ok());
        let dur = it.next().map(|b| b.trim().parse::<f64>());
        match (start, dur) {
            (Some(st), None) => return Ok(MediaSpec { path: PathBuf::from(path), start: st, dur: None }),
            (Some(st), Some(Ok(d))) => return Ok(MediaSpec { path: PathBuf::from(path), start: st, dur: Some(d) }),
            _ => {}
        }
    }
    Ok(MediaSpec { path: PathBuf::from(s), start: 0.0, dur: None })
}

fn parse_size(s: &str) -> Result<(usize, usize)> {
    let (a, b) = s.split_once(['x', 'X']).context("--size must look like WIDTHxHEIGHT")?;
    Ok((a.trim().parse().context("bad width")?, b.trim().parse().context("bad height")?))
}

fn parse_aspect(s: &str) -> Result<f64> {
    let (a, b) = s.split_once([':', '/']).context("--aspect must look like W:H")?;
    let (a, b): (f64, f64) = (a.trim().parse()?, b.trim().parse()?);
    ensure!(a > 0.0 && b > 0.0, "bad aspect");
    Ok(a / b)
}

/// Size for `ratio` (w/h) at about `mp` megapixels, rounded to multiples of 32.
fn size_for(ratio: f64, mp: f64) -> (usize, usize) {
    let area = mp * 1e6;
    let w = (area * ratio).sqrt();
    let h = area / w;
    let r = |x: f64| ((x / 32.0).round() as usize * 32).max(64);
    (r(w), r(h))
}

fn main() {
    if let Err(e) = run() {
        let msg = format!("{e:#}");
        eprintln!("error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications (a running ComfyUI holds its models in VRAM) or lower --size / --duration / --ref-image-size.");
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let prompt = match (&cli.prompt, &cli.prompt_file) {
        (Some(p), _) => p.clone(),
        (None, Some(f)) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
        (None, None) => bail!("give a prompt with --prompt or --prompt-file"),
    };
    let ff = Ffmpeg::discover(cli.ffmpeg.as_deref())?;
    let ref_videos: Vec<MediaSpec> = cli.ref_video.iter().map(|s| parse_media(s)).collect::<Result<_>>()?;
    let ref_audios: Vec<MediaSpec> = cli.ref_audio.iter().map(|s| parse_media(s)).collect::<Result<_>>()?;
    for p in cli.ref_image.iter().chain(ref_videos.iter().map(|m| &m.path)).chain(ref_audios.iter().map(|m| &m.path)) {
        ensure!(Path::new(p).is_file(), "reference file {} not found", p.display());
    }

    let frames = match (cli.frames, cli.duration) {
        (Some(f), _) => f,
        (None, Some(d)) => {
            ensure!(d > 0.0, "--duration must be positive");
            ((d * FPS as f64).round() as usize).max(5)
        }
        (None, None) => 5 * FPS,
    };
    // output size
    let (width, height) = if let Some(s) = &cli.size {
        parse_size(s)?
    } else {
        let ratio = if let Some(a) = &cli.aspect {
            parse_aspect(a)?
        } else if let Some(p) = cli.ref_image.first() {
            let i = image_dims(&ff, p)?;
            i.0 as f64 / i.1 as f64
        } else if let Some(v) = ref_videos.first() {
            let i = ff.probe(&v.path)?;
            i.width as f64 / i.height.max(1) as f64
        } else {
            16.0 / 9.0
        };
        size_for(ratio, cli.megapixels)
    };
    let seed = cli.seed.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        (t as u64) & 0xFFFF_FFFF
    });
    let out = cli.out.clone().unwrap_or_else(|| PathBuf::from(format!("h3_{seed}.mp4")));
    let req = Request {
        prompt,
        ref_images: cli.ref_image.clone(),
        ref_videos,
        ref_audios,
        width,
        height,
        frames,
        steps: cli.steps,
        seed,
        ref_image_size: match cli.ref_image_size {
            RefSize::Match => RefImageSize::Match,
            RefSize::Max => RefImageSize::Max,
        },
        video_audio: !cli.no_video_audio,
        no_audio: cli.no_audio,
        crf: cli.crf,
        out,
    };
    // fail on bad geometry before touching models
    let (fc, _, _) = pipeline::temporal_shape(req.frames);
    pipeline::check_limits(req.width, req.height, fc)?;

    let paths = Paths {
        dit: models::resolve(cli.dit.clone(), &models::DIT)?,
        text_encoder: models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?,
        video_vae: models::resolve(cli.video_vae.clone(), &models::VIDEO_VAE)?,
        audio_vae: models::resolve(cli.audio_vae.clone(), &models::AUDIO_VAE)?,
    };
    eprintln!("  dit: {}\n  text encoder: {}\n  video vae: {}\n  audio vae: {}", paths.dit.display(), paths.text_encoder.display(), paths.video_vae.display(), paths.audio_vae.display());
    let dev = Device::new(0)?;
    eprintln!("  gpu: {} MB free of {} MB", dev.free_mem()? >> 20, dev.total_mem >> 20);
    pipeline::run(dev, &req, &paths, &ff)
}

fn image_dims(ff: &Ffmpeg, p: &Path) -> Result<(usize, usize)> {
    if let Ok((w, h)) = image::image_dimensions(p) {
        return Ok((w as usize, h as usize));
    }
    let i = ff.probe(p)?;
    ensure!(i.width > 0, "cannot read the size of {}", p.display());
    Ok((i.width, i.height))
}
