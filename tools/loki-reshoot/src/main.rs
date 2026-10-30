//! loki-reshoot: MiniMax H3 reference-to-video (images / videos / audio -> video with sound) on a standalone CUDA engine.
use anyhow::{bail, ensure, Context, Result};
use clap::{CommandFactory, Parser, ValueEnum};
use loki_reshoot::worker;
use loki_reshoot::genmeta;
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

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum ModelArg {
    /// MiniMax H3 ref2va: the original model, best quality (20 steps, res_multistep)
    H3,
    /// FastH3 Omni: FastVideo's 8-step distilled finetune of H3 with sparse attention, ~4-5x faster sampling, may lose some quality
    Fast,
}

impl ModelArg {
    fn default_steps(self) -> usize {
        match self {
            ModelArg::H3 => 20,
            ModelArg::Fast => 8,
        }
    }
    fn name(self) -> &'static str {
        match self {
            ModelArg::H3 => "MiniMax H3",
            ModelArg::Fast => "FastH3 Omni",
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FitArg {
    Auto,
    Pad,
    Crop,
    Stretch,
}

#[derive(Parser, Debug, Clone)]
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
  --model h3 (default): MiniMax H3, 20 steps. --model fast: FastH3 Omni, an 8-step distilled finetune run with Video Sparse
  Attention: ~4-5x faster sampling, ~2.5x end to end at 1024x768 / 5 s (quality may be a little lower; --dense-attention). Both share the text encoder and VAEs; fast only adds its own diffusion model (22 GB).
  Looked up next to the binary, in ./models, $LOKI_MODELS (a path list); missing ones are downloaded from Hugging Face
  (~42 GB for h3, +22 GB for fast).
  Swapping the diffusion model: --dit FILE runs any same-architecture H3 finetune (int8 ConvRot, bf16 linears are
  quantized as they load) with --model's recipe: h3 = 20 res_multistep steps, fast = 8 Euler steps for few-step (turbo)
  finetunes. $LOKI_RESHOOT_DIT / $LOKI_RESHOOT_FAST_DIT set a machine default per model. A bare NAME is looked up in the
  model folders.

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
    /// Your own JSON object, stored verbatim as the `meta` field of the generation record stamped into the result (e.g. an
    /// app's recipe for the run). Left out of the record's `args`.
    #[arg(long, value_name = "JSON")]
    meta: Option<String>,
    /// Read --meta from a file.
    #[arg(long, value_name = "FILE")]
    meta_file: Option<String>,
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
    /// Which diffusion model generates: `h3` (original, best quality) or `fast` (FastH3 Omni, 8 steps + sparse attention, ~4-5x faster sampling).
    #[arg(long, value_enum, default_value_t = ModelArg::H3)]
    model: ModelArg,
    /// With --model fast: full attention instead of Video Sparse Attention (VSA, the default for FastH3: each query attends
    /// its 10% most relevant video cubes plus all text/reference/audio tokens, ~1.5-2x faster per step).
    #[arg(long)]
    dense_attention: bool,
    /// Sampling steps (`simple` schedule, shift 12). Default 20 for h3 (res_multistep), 8 for fast (Euler; it is distilled
    /// for exactly 8).
    #[arg(long)]
    steps: Option<usize>,
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
    /// Diffusion model file or bare file name (looked up in the model folders): any same-architecture H3 finetune, sampled
    /// with --model's recipe (a FastH3 file is recognised as such). Default: $LOKI_RESHOOT_DIT (h3) /
    /// $LOKI_RESHOOT_FAST_DIT (fast), else --model's own file, auto-discovered (or downloaded).
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
    /// Run as a resident worker: read one request per line on stdin (a JSON object whose "args" are a command line for
    /// this tool), keep the models on the GPU between requests, answer with JSON events on stdout. For supervisors such
    /// as loki-worker; see CONVENTIONS.md, "Worker mode".
    #[arg(long)]
    worker: bool,
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
    if let Err(e) = run(std::env::args().collect()) {
        let msg = format!("{e:#}");
        eprintln!("error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications (a running ComfyUI holds its models in VRAM) or lower --size / --duration / --ref-image-size.");
        }
        std::process::exit(1);
    }
}

/// The diffusion model: --dit, else the machine-wide swap for --model ($LOKI_RESHOOT_DIT / $LOKI_RESHOOT_FAST_DIT: a
/// same-architecture finetune), else --model's own file.
fn dit_choice(cli: &Cli) -> Option<PathBuf> {
    let var = if cli.model == ModelArg::Fast { "LOKI_RESHOOT_FAST_DIT" } else { "LOKI_RESHOOT_DIT" };
    cli.dit.clone().or_else(|| std::env::var_os(var).filter(|v| !v.is_empty()).map(PathBuf::from))
}

fn model_paths(cli: &Cli) -> Result<Paths> {
    Ok(Paths {
        dit: models::resolve(dit_choice(cli), if cli.model == ModelArg::Fast { &models::FAST_DIT } else { &models::DIT })?,
        text_encoder: models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?,
        video_vae: models::resolve(cli.video_vae.clone(), &models::VIDEO_VAE)?,
        audio_vae: models::resolve(cli.audio_vae.clone(), &models::AUDIO_VAE)?,
    })
}

fn run(argv: Vec<String>) -> Result<()> {
    let Some(cli) = worker::parse_args::<Cli>(Cli::command(), argv)? else { return Ok(()) };
    if cli.worker {
        ensure!(!worker::active(), "--worker inside a worker request");
        loki_reshoot::log::set_quiet(cli.quiet);
        return worker::serve(&worker::Tool {
            name: "loki-reshoot",
            version: env!("CARGO_PKG_VERSION"),
            model: "MiniMax H3",
            run: &|argv| run(argv),
            // {"cmd": "warm", "model": "fast"} warms FastH3's diffusion model instead of H3's
            warm: &|req| {
                let mut base = cli.clone();
                if req.get("model").and_then(|m| m.as_str()) == Some("fast") {
                    base.model = ModelArg::Fast;
                }
                pipeline::warm(&worker::device()?, &model_paths(&base)?, base.model == ModelArg::Fast)
            },
            custom: None,
        });
    }
    if cli.prompt_guide {
        worker::out_line(include_str!("../docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md"));
        return Ok(());
    }
    let mut cli = cli;
    loki_reshoot::log::set_quiet(cli.quiet);
    let meta = genmeta::read_meta(cli.meta.as_deref(), cli.meta_file.as_deref()).map_err(anyhow::Error::msg)?;
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
        worker::out_line(&prompt);
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
        ensure!(!worker::active(), "-o - (mp4 on stdout) is not available in --worker mode; give a file");
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
        steps: cli.steps.unwrap_or(cli.model.default_steps()),
        seed,
        ref_image_size: match cli.ref_image_size {
            RefSize::Match => RefImageSize::Match,
            RefSize::Max => RefImageSize::Max,
        },
        fit_first: if cli.native { Some(fit_resolved) } else { None },
        video_audio: !cli.no_video_audio,
        no_audio: cli.no_audio,
        sparse: !cli.dense_attention,
        distilled: cli.model == ModelArg::Fast,
        crf: cli.crf,
        out,
    };
    // fail on bad geometry before touching models
    let (fc, _, _) = pipeline::temporal_shape(req.frames);
    pipeline::check_limits(req.width, req.height, fc)?;
    let mut record = generation_record(&cli, &req, fc, fit_resolved, &ff, _stdin_tmp.0.as_deref());
    record.set("meta", meta);

    let paths = model_paths(&cli)?;
    record.set("dit", paths.dit.file_name().map(|f| f.to_string_lossy().into_owned()));
    // VSA needs the model's coarse gates: a fast finetune without them runs dense
    if record.get("attention").and_then(|a| a.as_str()) == Some("vsa") {
        let gated = loki_reshoot::safetensors::SafeTensors::open(&paths.dit).map(|st| st.has("blocks.0.attn.to_gate_compress.weight")).unwrap_or(true);
        if !gated {
            record.set("attention", "dense");
        }
    }
    eprintln!("  dit: {}\n  text encoder: {}\n  video vae: {}\n  audio vae: {}", paths.dit.display(), paths.text_encoder.display(), paths.video_vae.display(), paths.audio_vae.display());
    let dev = worker::device()?;
    eprintln!("  gpu: {} MB free of {} MB", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let out_path = req.out.clone();
    pipeline::run(dev, &req, &paths, &ff, Some(&record.to_json()))?;
    if !to_stdout {
        if cli.json {
            let j = serde_json::json!({
                "output": out_path.display().to_string(), "width": req.width, "height": req.height, "frames": fc, "fps": FPS,
                "seconds": fc as f64 / FPS as f64, "seed": req.seed, "steps": req.steps, "audio": !req.no_audio,
                "generation": record.to_value(),
            });
            worker::out_line(&j.to_string());
        } else {
            worker::out_line(&out_path.display().to_string());
        }
    }
    Ok(())
}

/// The generation record stamped into the output's `comment` tag (see `genmeta`): the final prompt, every setting that
/// shaped the result, and the references by prompt tag. `stdin_tmp` is the temp file a `-` reference was spooled to; it is
/// reported as `-` since the temp path means nothing later.
fn generation_record(cli: &Cli, req: &Request, frame_count: usize, fit: Fit, ff: &Ffmpeg, stdin_tmp: Option<&Path>) -> genmeta::Record {
    use serde_json::{json, Value};
    let show = |p: &Path| -> String {
        if stdin_tmp == Some(p) { "-".to_string() } else { p.display().to_string() }
    };
    let mut inputs: Vec<Value> = Vec::new();
    for (i, p) in req.ref_images.iter().enumerate() {
        let mut v = json!({ "tag": format!("<Picture {}>", i + 1), "path": show(p) });
        if let Ok((w, h)) = image_dims(ff, p) {
            v["width"] = w.into();
            v["height"] = h.into();
        }
        inputs.push(v);
    }
    let mut audio_n = 0;
    for (i, m) in req.ref_videos.iter().enumerate() {
        let mut v = json!({ "tag": format!("<Video {}>", i + 1), "path": show(&m.path), "start": m.start });
        if let Some(d) = m.dur {
            v["duration"] = d.into();
        }
        if req.video_audio && ff.probe(&m.path).map(|i| i.has_audio).unwrap_or(false) {
            audio_n += 1;
            v["audio_tag"] = format!("<Audio {audio_n}>").into();
        }
        inputs.push(v);
    }
    for m in &req.ref_audios {
        audio_n += 1;
        let mut v = json!({ "tag": format!("<Audio {audio_n}>"), "path": show(&m.path), "start": m.start });
        if let Some(d) = m.dur {
            v["duration"] = d.into();
        }
        inputs.push(v);
    }
    let mut rec = genmeta::Record::new("loki-reshoot", env!("CARGO_PKG_VERSION"), cli.model.name());
    rec.set("model_id", format!("{:?}", cli.model).to_lowercase())
        .set("attention", if cli.model == ModelArg::Fast { Some(if req.sparse { "vsa" } else { "dense" }) } else { None })
        .set("prompt", req.prompt.clone())
        .set("seed", req.seed)
        .set("steps", req.steps)
        .set("width", req.width)
        .set("height", req.height)
        .set("frames", frame_count)
        .set("fps", FPS)
        .set("seconds", (frame_count as f64 / FPS as f64 * 1000.0).round() / 1000.0)
        .set("audio", !req.no_audio)
        .set("crf", req.crf)
        .set("ref_image_size", match req.ref_image_size { RefImageSize::Match => "match", RefImageSize::Max => "max" })
        .set("native", if cli.native { Some(true) } else { None })
        .set("fit", if cli.native { Some(format!("{fit:?}").to_lowercase()) } else { None })
        .set("animate", if cli.animate.is_some() { Some(true) } else { None })
        .set("describe", if cli.animate.is_some() { cli.describe.clone() } else { None })
        .set("shake", if cli.animate.is_some() { Some(format!("{:?}", cli.shake).to_lowercase()) } else { None })
        .set("inputs", inputs);
    rec
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
    worker::forbid_stdin()?;
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
