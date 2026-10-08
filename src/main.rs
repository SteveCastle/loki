//! loki-reshoot: MiniMax H3 reference-to-video (images / videos / audio -> video with sound) on a standalone CUDA engine.
use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use loki_reshoot::cuda::Device;
use loki_reshoot::media::{Ffmpeg, FPS};
use loki_reshoot::models;
use loki_reshoot::fit::{self, Fit};
use loki_reshoot::pipeline::{self, MediaSpec, Paths, RefImageSize, Request};
use loki_reshoot::presets::{self, Shake};
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum RefSize {
    Match,
    Max,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ShakeArg {
    None,
    Subtle,
    Handheld,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FitArg {
    Auto,
    Pad,
    Crop,
    Stretch,
}

#[derive(Parser, Debug)]
#[command(
    name = "loki-reshoot",
    version,
    about = "loki-reshoot: AI video with sound from reference images, videos and audio (MiniMax H3) on a standalone CUDA engine. Part of the loki- toolkit (see also loki-retouch, image editing).",
    after_help = concat!(
        "REFERENCE TAGS
  In the prompt, point at references by tag, in the order given on the command line:
  <Picture 1>.. (--ref-image), <Video 1>.. (--ref-video), <Audio 1>.. (soundtracks of the reference videos first, then --ref-audio).
  The tags are printed at start-up.

LIMITS
  length  : 5.2-15 s trained (124-362 frames @ 24 fps); shorter runs, > 15.08 s is refused
  size    : multiples of 32, up to ~2K (2048x1152); ComfyUI's table tops out at 1920x1088
  refs    : up to 9 images, 3 videos (each 5+ frames, ~0.2-15 s), 3 audio clips

COMPOSING (loki- toolkit conventions, shared with loki-retouch)
  `-` as a reference path reads that one input (image, video or audio) from stdin; `-o -` writes the mp4 itself to stdout
  (fragmented mp4); otherwise the path of the written file is printed on stdout (`--json`: one JSON object instead).
  Progress goes to stderr only (`-q` silences it); exit status 0 = ok, 1 = error, 2 = usage error.
    loki-retouch --preset restore -o - old.jpg | loki-reshoot --animate - -d 5 -o - | ffmpeg -i - -vf scale=720:-2 small.mp4

MODELS
  Looked up next to the binary, in ./models, $LOKI_MODELS; missing ones are downloaded from Hugging Face (~42 GB).

",
        include_str!("../docs/PROMPT_CHEATSHEET.txt")
    )
)]
struct Cli {
    /// Prompt text (use <Picture 1>, <Video 1>, <Audio 1> tags to address the references). Long prompts: --prompt-file.
    /// Read the PROMPT WRITING section below (or `--prompt-guide`) before writing one: this model needs a detailed, structured prompt.
    #[arg(short, long)]
    prompt: Option<String>,
    /// QUICK MODE "living photo": animate IMAGE with natural ambient life, subtle resting movement and (by default) a subtle
    /// camera shake, keeping the photo's identity and framing. Adds the image as <Picture 1>, uses --native size and 5 s unless
    /// overridden. Describe the subject with --describe for best results; --prompt adds extra direction.
    #[arg(long, value_name = "IMAGE")]
    animate: Option<PathBuf>,
    /// With --animate: one sentence naming the subject and setting (e.g. "the young woman in a black swimsuit taking a mirror
    /// selfie in a sunlit room"). Anchors identity and what may move.
    #[arg(long)]
    describe: Option<String>,
    /// With --animate: camera behaviour.
    #[arg(long, value_enum, default_value_t = ShakeArg::Subtle)]
    shake: ShakeArg,
    /// Use the model's native canvas: snap the first reference's aspect to the nearest supported ratio (1:1 768x768, 4:3 1024x768,
    /// 3:4 768x1024, 16:9 1344x768, 9:16 768x1344) and fit the first image to it (see --fit). Overridden by --size.
    #[arg(long)]
    native: bool,
    /// With --native: how the first reference image is fitted to the canvas when its ratio is not exactly the supported one:
    /// `auto` (default: centre-crop when the ratios are within ~12%, else pad), `crop`, `pad` (black bars; the model tends to KEEP
    /// them, so use it only for large mismatches you want to letterbox) or `stretch`.
    #[arg(long, value_enum, default_value_t = FitArg::Auto)]
    fit: FitArg,
    /// Print the final prompt (after --animate expansion) and exit without generating.
    #[arg(long)]
    show_prompt: bool,
    /// Silence progress/diagnostics on stderr (errors still print).
    #[arg(short, long)]
    quiet: bool,
    /// Print one JSON object describing the result on stdout instead of the bare output path.
    #[arg(long)]
    json: bool,
    /// Print MiniMax's full reference-mode prompt-writing guide and exit.
    #[arg(long)]
    prompt_guide: bool,
    /// Read the prompt from a text file (`-` = stdin).
    #[arg(short = 'P', long, value_name = "FILE")]
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
    /// Output mp4, or `-` for the mp4 on stdout (default: reshoot_<seed>.mp4 in the current directory).
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
    if cli.prompt_guide {
        println!("{}", include_str!("../docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"));
        return Ok(());
    }
    let mut cli = cli;
    loki_reshoot::log::set_quiet(cli.quiet);
    let _stdin_tmp = TempFile(resolve_stdin(&mut cli)?);
    if let Some(img) = cli.animate.clone() {
        cli.ref_image.insert(0, img);
        if cli.size.is_none() && cli.aspect.is_none() {
            cli.native = true;
        }
    }
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
        if cli.native {
            // the canvas rule ComfyUI uses for reference videos: 768 short edge, area cap 768*1344
            let (bw, bh) = if let Some(p) = cli.ref_image.first() {
                image_dims(&ff, p)?
            } else if let Some(v) = ref_videos.first() {
                let i = ff.probe(&v.path)?;
                (i.width, i.height)
            } else {
                (1920, 1080)
            };
            let ratio = match &cli.aspect {
                Some(a) => parse_aspect(a)?,
                None => bw as f64 / bh as f64,
            };
            let (label, r) = fit::best_ratio((ratio * 10000.0) as usize, 10000);
            let (w, h) = fit::canvas_for_ratio(r);
            eprintln!("native canvas: nearest supported ratio {label} -> {w}x{h}");
            (w, h)
        } else {
            size_for(ratio, cli.megapixels)
        }
    };
    // resolve --fit auto against the first image: crop when the mismatch is small, else pad
    let fit_resolved = match cli.fit {
        FitArg::Pad => Fit::Pad,
        FitArg::Crop => Fit::Crop,
        FitArg::Stretch => Fit::Stretch,
        FitArg::Auto => match cli.ref_image.first() {
            Some(p) => {
                let (ow, oh) = image_dims(&ff, p)?;
                if ((ow as f64 / oh as f64) / (width as f64 / height as f64)).ln().abs() <= 0.12 {
                    Fit::Crop
                } else {
                    Fit::Pad
                }
            }
            None => Fit::Crop,
        },
    };
    // is the first reference going to be padded with black bars on the canvas?
    let padded_first = cli.native
        && fit_resolved == Fit::Pad
        && match cli.ref_image.first() {
            Some(p) => {
                let (ow, oh) = image_dims(&ff, p)?;
                fit::pad_extent(ow, oh, width, height) != (width, height)
            }
            None => false,
        };
    let prompt = match (&cli.prompt, &cli.prompt_file) {
        _ if cli.animate.is_some() => {
            let extra = match (&cli.prompt, &cli.prompt_file) {
                (Some(p), _) => Some(p.clone()),
                (None, Some(f)) => Some(std::fs::read_to_string(f)?),
                _ => None,
            };
            let shake = match cli.shake {
                ShakeArg::None => Shake::None,
                ShakeArg::Subtle => Shake::Subtle,
                ShakeArg::Handheld => Shake::Handheld,
            };
            presets::animate_prompt(cli.describe.as_deref(), shake, extra.as_deref(), padded_first)
        }
        (Some(p), _) => p.clone(),
        (None, Some(f)) => std::fs::read_to_string(f).with_context(|| format!("reading {}", f.display()))?,
        (None, None) => bail!("give a prompt with --prompt or --prompt-file"),
    };
    if cli.show_prompt {
        println!("{prompt}");
        return Ok(());
    }
    let seed = cli.seed.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        (t as u64) & 0xFFFF_FFFF
    });
    let out = cli.out.clone().unwrap_or_else(|| PathBuf::from(format!("reshoot_{seed}.mp4")));
    let to_stdout = out == Path::new("-");
    if to_stdout {
        use std::io::IsTerminal;
        ensure!(!std::io::stdout().is_terminal(), "refusing to write a video to the terminal: pipe stdout (| ffmpeg -i - ...) or use -o FILE");
    }
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
        fit_first: if cli.native { Some(fit_resolved) } else { None },
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
    let out_path = req.out.clone();
    pipeline::run(dev, &req, &paths, &ff)?;
    if !to_stdout {
        if cli.json {
            let (fc, _, _) = pipeline::temporal_shape(req.frames);
            let j = serde_json::json!({
                "output": out_path.display().to_string(), "width": req.width, "height": req.height, "frames": fc, "fps": FPS,
                "seconds": fc as f64 / FPS as f64, "seed": req.seed, "steps": req.steps, "audio": !req.no_audio,
            });
            println!("{j}");
        } else {
            println!("{}", out_path.display());
        }
    }
    Ok(())
}

/// `-` as a reference (or prompt file) means stdin; at most one such input per run. The bytes go to a temp file so ffmpeg and
/// the image decoder can sniff the format from the content.
/// Removes the stdin temp file when dropped.
struct TempFile(Option<PathBuf>);
impl Drop for TempFile {
    fn drop(&mut self) {
        if let Some(p) = &self.0 {
            let _ = std::fs::remove_file(p);
        }
    }
}

fn resolve_stdin(cli: &mut Cli) -> Result<Option<PathBuf>> {
    use std::io::Read;
    let mut uses = 0usize;
    let is_dash = |p: &Path| p == Path::new("-");
    let strip = |s: &str| -> bool { s == "-" || s.starts_with("-@") };
    uses += cli.ref_image.iter().filter(|p| is_dash(p)).count() + cli.animate.iter().filter(|p| is_dash(p)).count();
    uses += cli.ref_video.iter().filter(|s| strip(s)).count() + cli.ref_audio.iter().filter(|s| strip(s)).count();
    uses += cli.prompt_file.iter().filter(|p| is_dash(p)).count();
    ensure!(uses <= 1, "stdin (`-`) can carry only one input per run");
    if uses == 0 {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    std::io::stdin().lock().read_to_end(&mut bytes).context("reading stdin")?;
    ensure!(!bytes.is_empty(), "no data on stdin");
    if cli.prompt_file.as_deref().map_or(false, |p| is_dash(p)) {
        cli.prompt = Some(String::from_utf8(bytes).context("prompt on stdin is not UTF-8")?);
        cli.prompt_file = None;
        return Ok(None);
    }
    let tmp = std::env::temp_dir().join(format!("loki-reshoot-stdin-{}", std::process::id()));
    std::fs::write(&tmp, &bytes)?;
    for p in cli.ref_image.iter_mut().chain(cli.animate.iter_mut()) {
        if is_dash(p) {
            *p = tmp.clone();
        }
    }
    for s in cli.ref_video.iter_mut().chain(cli.ref_audio.iter_mut()) {
        if s == "-" {
            *s = tmp.display().to_string();
        } else if let Some(rest) = s.strip_prefix("-@") {
            *s = format!("{}@{rest}", tmp.display());
        }
    }
    Ok(Some(tmp))
}

fn image_dims(ff: &Ffmpeg, p: &Path) -> Result<(usize, usize)> {
    if let Ok((w, h)) = image::image_dimensions(p) {
        return Ok((w as usize, h as usize));
    }
    let i = ff.probe(p)?;
    ensure!(i.width > 0, "cannot read the size of {}", p.display());
    Ok((i.width, i.height))
}
