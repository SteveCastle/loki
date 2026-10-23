//! Command-line front end of loki-dream: text -> image with Kromoss (MOSS-Image), plus img2img, inpainting and LoRAs.
//!
//! loki- toolkit conventions: results are written to files whose paths are printed on stdout (one per line), or one JSON
//! object for the run with `--json`, or the PNG itself with `-o -`; progress and diagnostics go to stderr only (`-q`
//! silences them); exit status 0 ok, 1 runtime error, 2 usage error.
use crate::cuda::Device;
use crate::dit::Dit;
use crate::genmeta;
use crate::image::{composite, Mask, Rgb8};
use crate::lora::{self, LoraSpec};
use crate::models;
use crate::pipeline::{self, Inputs, Settings};
use crate::tensor::Tensor;
use crate::text_encoder::{self, TextEncoder};
use crate::vae::Vae;
use anyhow::{bail, ensure, Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser};
use std::collections::HashMap;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

pub const MODEL_NAME: &str = "Kromoss (MOSS-Image)";

const HELP: &str = "\
OUTPUT
  stdout carries the result and nothing else: the path of every image written, one per line (--json: one JSON object
  {\"images\":[{\"path\",\"seed\",\"width\",\"height\",\"generation\"}],\"prompt\",...} for the whole run instead); `-o -` writes the
  PNG itself (one image). Progress (`step i/n` lines when stderr is not a terminal), timings and warnings go to stderr;
  -q silences them. Every PNG carries its generation record (JSON in an iTXt chunk `loki`: prompt, seed, steps, cfg, size,
  inputs, command line, and --meta).
  -o FILE.png   one image (with --count N or several prompts: FILE_<seed>.png next to it)
  -o DIR        an existing folder (or a path ending in / or \\): files are named dream_<seed>.png, never overwriting
  (none)        dream_<seed>.png in the current folder (img2img: <init>_dream_<seed>.png next to the init image)

SIZE (default 1024x1024; the model works in multiples of 16 pixels; about 1 megapixel is its home ground)
  --size WxH | --aspect W:H [--megapixels MP (default 1)] | --megapixels MP (square)
  A size that is not a multiple of 16 is generated at the nearest multiple and Lanczos-resampled to exactly what you
  asked (--snap keeps the multiple of 16). With --init and no size flag: the init image's own size.

SAMPLING
  Full model: 50 steps, CFG 3, flow-matching Euler (shift 5), the negative branch with block 22 skipped (official
  defaults). --turbo: the 8-step DMD2 LoRA, 8 steps, CFG 1, no negative branch (about 10x faster). --seed N reproduces
  an image; the noise is drawn exactly like torch.randn on the CPU, so a seed means the same as in diffusers/ComfyUI.

IMG2IMG / INPAINT
  --init IMG --strength F    noise the encoded image to F (0..1, default 0.6) and denoise the rest of the schedule
                             (the last F of the steps, like diffusers). The shift-5 schedule keeps colours and layout
                             up to about 0.9; use 0.9-0.97 to change the subject while keeping the composition.
  --mask MASK (with --init)  white = regenerate, black = keep, grey = in between; any size (resized to the init). The
                             kept area is re-imposed every step and composited back at full resolution.

LAYOUT PROMPTS
  MOSS-Image understands a structured layout prompt: a compact JSON object given as the prompt, e.g.
  {\"summary\":{\"caption\":\"a poster for a jazz night\"},\"items\":[{\"label\":\"text\",\"caption\":\"title\",\"text\":\"BLUE NOTE\",
  \"bbox\":[-400,-300,-250,300]},{\"label\":\"object\",\"caption\":\"a saxophone\",\"bbox\":[-150,-200,450,200]}],
  \"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}
  bbox = [y0, x0, y1, x1] in pixels with the origin at the image centre; label text|object|character; optional
  \"palette\". It goes through the text encoder like any prompt (see --show-prompt).

EXAMPLES
  PROG -p \"a tabby cat holding a sign that says KROMOSS\"                 # -> dream_<seed>.png
  PROG --turbo --count 4 -o out/ -p \"a lighthouse in a storm, oil painting\"
  PROG --aspect 16:9 --seed 7 -p \"a misty mountain lake at dawn\" -o lake.png
  PROG --init photo.jpg --strength 0.5 -p \"the same scene in winter\"
  PROG --init photo.jpg --mask sky.png --strength 1 -p \"a sky full of hot air balloons\"
  PROG --lora style.safetensors:0.8 -p \"...\"                             # any MOSS-Image LoRA (kohya/ComfyUI/diffusers keys)
  PROG --turbo -o - -p \"a red apple\" | loki-reshoot --animate - -d 5 -o apple.mp4
";

#[derive(Parser, Debug)]
#[command(version, about = "loki-dream: text-to-image on a standalone CUDA engine (Kromoss, a fine-tune of MOSS-Image, with the Qwen3.5-9B text encoder and the FLUX.2 VAE). Text to image, img2img, inpainting, LoRAs, 8-step turbo. Part of the loki- toolkit.", after_help = HELP)]
struct Cli {
    /// What to draw. Repeatable: several prompts in one run (models load once).
    #[arg(short, long)]
    prompt: Vec<String>,
    /// Read the prompt from a file (`-` = stdin); the whole file is one prompt.
    #[arg(short = 'P', long, value_name = "FILE")]
    prompt_file: Option<String>,
    /// A file with one prompt per line (blank lines and lines starting with # are skipped).
    #[arg(long, value_name = "FILE")]
    batch: Option<PathBuf>,
    /// Negative prompt (ignored with --turbo or --cfg 1). Default: empty, as the official pipeline.
    #[arg(long, value_name = "TEXT")]
    negative: Option<String>,
    /// Print the exact text(s) the text encoder gets (chat template included) and exit.
    #[arg(long)]
    show_prompt: bool,

    /// Output size WxH (multiples of 16 are native; others are resampled). Default 1024x1024.
    #[arg(long, value_name = "WxH")]
    size: Option<String>,
    /// Aspect ratio W:H at --megapixels (default 1).
    #[arg(long, value_name = "W:H")]
    aspect: Option<String>,
    /// Pixel count in megapixels (square unless --aspect).
    #[arg(long, value_name = "MP")]
    megapixels: Option<f64>,
    /// Deliver the model's multiple-of-16 size instead of resampling to the exact requested size.
    #[arg(long)]
    snap: bool,

    /// Sampling steps (default 50; 8 with --turbo).
    #[arg(long)]
    steps: Option<usize>,
    /// Classifier-free guidance scale (default 3; 1 with --turbo = no negative branch).
    #[arg(long)]
    cfg: Option<f32>,
    /// Noise seed (default: random). Several images use seed, seed+1, ...
    #[arg(long)]
    seed: Option<u64>,
    /// Images per prompt (one model load for all).
    #[arg(long, default_value_t = 1)]
    count: usize,
    /// 8-step turbo: apply the DMD2 LoRA, 8 steps, CFG 1, no negative branch.
    #[arg(long)]
    turbo: bool,
    /// Turbo LoRA flavour: dmd2 (default: best PartiPrompts score, more contrast and detail) or dmd.
    #[arg(long, value_name = "dmd2|dmd", default_value = "dmd2", hide = true)]
    turbo_lora: String,
    /// Apply a LoRA (repeatable): FILE or FILE:STRENGTH. Kromoss/MOSS-format (kohya) and ComfyUI/diffusers keys.
    #[arg(long = "lora", value_name = "FILE[:SCALE]")]
    loras: Vec<String>,
    /// Scheduler shift (official 5).
    #[arg(long, default_value_t = 5.0)]
    shift: f32,
    /// Blocks skipped in the negative branch, comma separated, or `none` (official default 22).
    #[arg(long, value_name = "LIST", default_value = "22")]
    skip_layers: String,

    /// img2img: start from this image (its size is the default output size).
    #[arg(long, value_name = "IMAGE")]
    init: Option<PathBuf>,
    /// img2img strength: how much of the schedule is redrawn (0..1).
    #[arg(long, default_value_t = 0.6)]
    strength: f32,
    /// Inpainting mask for --init: white = regenerate, black = keep; soft edges respected; any size.
    #[arg(long, value_name = "MASK")]
    mask: Option<PathBuf>,

    /// Output PNG path, an existing folder, or `-` for the PNG on stdout.
    #[arg(short, long)]
    out: Option<String>,
    /// Silence progress/diagnostics on stderr (errors still print).
    #[arg(short, long)]
    quiet: bool,
    /// Print one JSON object describing the run (images, seeds, sizes, generation records) instead of paths.
    #[arg(long)]
    json: bool,
    /// Your own JSON object, stored verbatim as the `meta` field of each image's generation record. Left out of `args`.
    #[arg(long, value_name = "JSON")]
    meta: Option<String>,
    /// Read --meta from a file.
    #[arg(long, value_name = "FILE")]
    meta_file: Option<String>,

    /// Diffusion model precision: int8 (ConvRot W8A8, 5.4 GB, default when present) or bf16 (10.2 GB).
    #[arg(long, value_name = "int8|bf16")]
    precision: Option<String>,
    /// Kromoss diffusion model file (overrides discovery; its format decides the precision).
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Qwen3.5-9B text encoder file.
    #[arg(long)]
    text_encoder: Option<PathBuf>,
    /// FLUX.2 small VAE file.
    #[arg(long)]
    vae: Option<PathBuf>,
}

pub fn main(prog: &'static str) {
    if let Err(e) = run(prog) {
        let msg = format!("{e:#}");
        eprintln!("{prog}: error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications, or lower --size / --megapixels.");
        }
        std::process::exit(1);
    }
}

fn round16(x: f64) -> usize {
    (((x / 16.0).round() as usize) * 16).max(16)
}

/// ((generated w, h), (delivered w, h))
fn target_size(cli: &Cli, init: Option<(usize, usize)>) -> Result<((usize, usize), (usize, usize))> {
    ensure!(!(cli.size.is_some() && (cli.aspect.is_some() || cli.megapixels.is_some())), "use --size, or --aspect/--megapixels, not both");
    let out = if let Some(s) = &cli.size {
        let (a, b) = s.split_once(['x', 'X']).context("--size must look like WIDTHxHEIGHT")?;
        (a.trim().parse::<usize>().context("bad --size width")?, b.trim().parse::<usize>().context("bad --size height")?)
    } else if cli.aspect.is_some() || cli.megapixels.is_some() {
        let ratio = match &cli.aspect {
            Some(a) => {
                let (x, y) = a.split_once(':').context("--aspect must look like W:H (e.g. 16:9)")?;
                let (x, y) = (x.trim().parse::<f64>().context("bad --aspect")?, y.trim().parse::<f64>().context("bad --aspect")?);
                ensure!(x > 0.0 && y > 0.0, "bad --aspect");
                x / y
            }
            None => init.map(|(w, h)| w as f64 / h as f64).unwrap_or(1.0),
        };
        let mp = cli.megapixels.unwrap_or(1.0);
        ensure!(mp > 0.0 && mp <= 16.0, "--megapixels must be in (0, 16]");
        let w = (mp * 1e6 * ratio).sqrt();
        let (w, h) = (round16(w), round16(mp * 1e6 / w));
        (w, h)
    } else if let Some(i) = init {
        i
    } else {
        (1024, 1024)
    };
    ensure!(out.0 >= 64 && out.1 >= 64, "output size {}x{} is too small (64 px minimum)", out.0, out.1);
    ensure!(out.0 * out.1 <= 16 << 20, "output size {}x{} is too large (16 megapixels maximum)", out.0, out.1);
    let gen = (round16(out.0 as f64), round16(out.1 as f64));
    Ok((gen, if cli.snap { gen } else { out }))
}

fn read_stdin() -> Result<String> {
    let mut v = String::new();
    std::io::stdin().lock().read_to_string(&mut v).context("reading stdin")?;
    Ok(v)
}

#[derive(Clone)]
enum Dest {
    Stdout,
    File(PathBuf),
}

/// A file name that does not exist yet: base.png, base_1.png, ...
fn unique(dir: &Path, base: &str, taken: &mut Vec<PathBuf>) -> PathBuf {
    let mut p = dir.join(format!("{base}.png"));
    let mut i = 1;
    while p.exists() || taken.contains(&p) {
        p = dir.join(format!("{base}_{i}.png"));
        i += 1;
    }
    taken.push(p.clone());
    p
}

fn run(prog: &'static str) -> Result<()> {
    let cmd = Cli::command().name(prog).bin_name(prog).after_help(HELP.replace("PROG", prog));
    let cli = Cli::from_arg_matches(&cmd.get_matches())?;
    crate::log::set_quiet(cli.quiet);
    let meta = genmeta::read_meta(cli.meta.as_deref(), cli.meta_file.as_deref()).map_err(anyhow::Error::msg)?;

    // ---- prompts
    let mut prompts: Vec<String> = cli.prompt.clone();
    if let Some(f) = &cli.prompt_file {
        prompts.push(if f == "-" { read_stdin()? } else { std::fs::read_to_string(f).with_context(|| format!("reading {f}"))? });
    }
    if let Some(b) = &cli.batch {
        let text = std::fs::read_to_string(b).with_context(|| format!("reading {}", b.display()))?;
        prompts.extend(text.lines().map(|l| l.trim()).filter(|l| !l.is_empty() && !l.starts_with('#')).map(|l| l.to_string()));
    }
    // the chat template trims the prompt; the record and file names see the same text
    let prompts: Vec<String> = prompts.iter().map(|p| p.trim().to_string()).collect();
    ensure!(!prompts.is_empty(), "give a prompt with -p TEXT, -P FILE or --batch FILE");
    ensure!(cli.count >= 1, "--count must be at least 1");
    let skip_layers: Vec<usize> = if cli.skip_layers.trim() == "none" || cli.skip_layers.trim().is_empty() {
        Vec::new()
    } else {
        cli.skip_layers.split(',').map(|s| s.trim().parse::<usize>().context("--skip-layers: comma-separated block indices or `none`")).collect::<Result<_>>()?
    };
    ensure!(skip_layers.iter().all(|&l| l < crate::dit::LAYERS), "--skip-layers: block indices must be < {}", crate::dit::LAYERS);
    let steps = cli.steps.unwrap_or(if cli.turbo { 8 } else { 50 });
    ensure!((1..=1000).contains(&steps), "--steps must be in 1..1000");
    let cfg = if cli.turbo {
        if cli.cfg.map(|c| c != 1.0).unwrap_or(false) {
            crate::info!("warning: --turbo runs at CFG 1; --cfg {} ignored", cli.cfg.unwrap());
        }
        1.0
    } else {
        cli.cfg.unwrap_or(3.0)
    };
    ensure!(cfg >= 0.0 && cfg.is_finite(), "--cfg must be >= 0");
    if cli.turbo && cli.negative.is_some() {
        crate::info!("warning: --turbo has no negative branch; --negative ignored");
    }
    let negative = if cfg != 1.0 { Some(cli.negative.clone().unwrap_or_default()) } else { None };
    if cli.show_prompt {
        for p in &prompts {
            print!("{}", text_encoder::template(p));
            println!();
        }
        if let Some(n) = &negative {
            println!("--- negative (CFG {cfg}, blocks {:?} skipped) ---", skip_layers);
            print!("{}", text_encoder::template(n));
            println!();
        }
        return Ok(());
    }
    ensure!(cli.mask.is_none() || cli.init.is_some(), "--mask needs --init (the image to inpaint)");
    ensure!((0.0..=1.0).contains(&cli.strength), "--strength must be in 0..1");
    let loras: Vec<LoraSpec> = cli.loras.iter().map(|s| LoraSpec::parse(s)).collect::<Result<_>>()?;
    for l in &loras {
        ensure!(l.path.is_file(), "LoRA {} not found", l.path.display());
    }

    // ---- inputs and sizes
    let init_img = match &cli.init {
        Some(p) => Some(Rgb8::load(p)?),
        None => None,
    };
    let ((gw, gh), (ow, oh)) = target_size(&cli, init_img.as_ref().map(|i| (i.w, i.h)))?;
    let init_gen = init_img.as_ref().map(|i| i.resize_lanczos(gw, gh));
    let mask_gen = match &cli.mask {
        Some(p) => Some(Mask::load(p)?.resize(gw, gh)),
        None => None,
    };

    // ---- seeds and destinations
    let base_seed = cli.seed.unwrap_or_else(|| {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        ((t ^ (std::process::id() as u128) << 20) % 4_294_967_296) as u64
    });
    struct Job {
        prompt: usize,
        seed: u64,
        dest: Dest,
    }
    let n_images = prompts.len() * cli.count;
    let mut jobs: Vec<Job> = Vec::with_capacity(n_images);
    let mut taken: Vec<PathBuf> = Vec::new();
    let init_stem = cli.init.as_ref().map(|p| p.file_stem().unwrap_or_default().to_string_lossy().to_string());
    for (pi, _) in prompts.iter().enumerate() {
        for c in 0..cli.count {
            let seed = base_seed.wrapping_add((pi * cli.count + c) as u64);
            let name = match &init_stem {
                Some(s) => format!("{s}_dream_{seed}"),
                None => format!("dream_{seed}"),
            };
            let dest = match cli.out.as_deref() {
                Some("-") => {
                    ensure!(n_images == 1, "-o - (PNG on stdout) needs exactly one image");
                    ensure!(!std::io::stdout().is_terminal(), "refusing to write a PNG to a terminal; redirect stdout or use -o FILE");
                    Dest::Stdout
                }
                Some(o) => {
                    let op = PathBuf::from(o);
                    let is_dir = op.is_dir() || o.ends_with('/') || o.ends_with('\\');
                    if is_dir {
                        std::fs::create_dir_all(&op).with_context(|| format!("creating {}", op.display()))?;
                        Dest::File(unique(&op, &name, &mut taken))
                    } else if n_images == 1 {
                        Dest::File(op)
                    } else {
                        let dir = op.parent().map(|d| d.to_path_buf()).unwrap_or_default();
                        let stem = op.file_stem().unwrap_or_default().to_string_lossy().to_string();
                        Dest::File(unique(&dir, &format!("{stem}_{seed}"), &mut taken))
                    }
                }
                None => {
                    let dir = match (&cli.init, &init_stem) {
                        (Some(p), Some(_)) => p.parent().map(|d| d.to_path_buf()).unwrap_or_default(),
                        _ => PathBuf::new(),
                    };
                    Dest::File(unique(&dir, &name, &mut taken))
                }
            };
            jobs.push(Job { prompt: pi, seed, dest });
        }
    }

    // ---- models
    let dit_path = match (&cli.dit, cli.precision.as_deref()) {
        (Some(p), _) => models::resolve(Some(p.clone()), &models::DIT_INT8)?,
        (None, Some("bf16")) => models::resolve(None, &models::DIT_BF16)?,
        (None, Some("int8")) => models::resolve(None, &models::DIT_INT8)?,
        (None, None) => match models::find_local(&models::DIT_INT8).or_else(|| models::find_local(&models::DIT_BF16)) {
            Some(p) => p,
            None => models::resolve(None, &models::DIT_INT8)?,
        },
        (None, Some(o)) => bail!("--precision must be int8 or bf16, not {o}"),
    };
    let te_path = models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?;
    let vae_path = models::resolve(cli.vae.clone(), &models::VAE)?;
    let mut all_loras = loras.clone();
    if cli.turbo {
        let spec = match cli.turbo_lora.as_str() {
            "dmd2" => &models::TURBO_LORA,
            "dmd" => &models::TURBO_LORA_DMD,
            o => bail!("--turbo-lora must be dmd2 or dmd, not {o}"),
        };
        all_loras.insert(0, LoraSpec { path: models::resolve(None, spec)?, strength: 1.0 });
    }
    crate::info!("{prog}: {} image(s), {}x{}{}, {} steps, CFG {}{}", n_images, ow, oh, if (gw, gh) != (ow, oh) { format!(" (generated at {gw}x{gh})") } else { String::new() }, steps, cfg, if cli.turbo { " (turbo)" } else { "" });
    crate::info!("  dit: {}\n  text encoder: {}\n  vae: {}", dit_path.display(), te_path.display(), vae_path.display());

    let t_dev = std::time::Instant::now();
    let dev = Device::new(0)?;
    crate::info!("  gpu: {} MB free of {} MB (cuda init {:.1}s)", dev.free_mem()? >> 20, dev.total_mem >> 20, t_dev.elapsed().as_secs_f64());
    crate::ops::gemm_init(&dev)?;
    crate::ops::attn_init(&dev)?;
    Vae::init_kernels(&dev)?;
    let t_all = std::time::Instant::now();

    // ---- phase 1: text encoder (streamed layer by layer); every prompt and the negative once
    let mut texts: Vec<String> = prompts.clone();
    if let Some(n) = &negative {
        texts.push(n.clone());
    }
    let t0 = std::time::Instant::now();
    let feats = {
        let te = TextEncoder::open(dev.clone(), &te_path)?;
        let show = !cli.quiet && std::io::stderr().is_terminal();
        let bar = if show { indicatif::ProgressBar::new(text_encoder::LAYERS as u64) } else { indicatif::ProgressBar::hidden() };
        bar.set_style(indicatif::ProgressStyle::with_template("  text encoder {bar:30} {pos}/{len} layers").unwrap());
        let out = te.encode(&texts, &|l| bar.set_position(l as u64)).context("text encoder")?;
        bar.finish_and_clear();
        te.close();
        out
    };
    crate::info!("  text encoder: {} prompt(s), {} tokens, {:.1}s", texts.len(), feats.iter().map(|f| f.tokens.len()).sum::<usize>(), t0.elapsed().as_secs_f64());

    // ---- phase 2: VAE (kept for the decodes); encode the init image
    let vae = Vae::load(dev.clone(), &vae_path)?;
    let (lh, lw) = (gh / 16, gw / 16);
    let x0 = match &init_gen {
        Some(img) => Some(vae.encode(&img.to_f32(), gh, gw).context("vae encode")?),
        None => None,
    };
    let mask_lat = match &mask_gen {
        Some(m) => Some(Tensor::from_f32(&dev, &m.max_down(16).data, &[lh * lw])?),
        None => None,
    };

    // ---- phase 3: DiT (+ LoRAs); sample, decode, write each image as soon as it is ready
    let mut dit = Dit::load(dev.clone(), &dit_path, None)?;
    if !all_loras.is_empty() {
        let t0 = std::time::Instant::now();
        dit.lora = lora::load(&dev, &all_loras, crate::dit::LAYERS)?;
        crate::info!("  loras ready in {:.1}s", t0.elapsed().as_secs_f64());
    }
    let mut runs: HashMap<usize, crate::dit::Run> = HashMap::new();
    let uncond = match &negative {
        Some(_) => Some(dit.prepare(&feats[prompts.len()].hidden, lh, lw)?),
        None => None,
    };
    let show_bar = !cli.quiet && std::io::stderr().is_terminal();
    let stdout_is_image = jobs.iter().any(|j| matches!(j.dest, Dest::Stdout));
    let mut results: Vec<serde_json::Value> = Vec::new();
    let settings_base = Settings { steps, cfg, shift: cli.shift, seed: 0, skip_layers: skip_layers.clone(), strength: if cli.init.is_some() { cli.strength } else { 1.0 } };
    let lora_record: Vec<serde_json::Value> = all_loras.iter().map(|l| serde_json::json!({"path": l.path.display().to_string(), "strength": l.strength})).collect();
    for (ji, job) in jobs.iter().enumerate() {
        if !runs.contains_key(&job.prompt) {
            runs.insert(job.prompt, dit.prepare(&feats[job.prompt].hidden, lh, lw)?);
        }
        let run = &runs[&job.prompt];
        let mut s = settings_base.clone();
        s.seed = job.seed;
        crate::info!("image {}/{}: seed {} {}x{}", ji + 1, jobs.len(), job.seed, ow, oh);
        let t0 = std::time::Instant::now();
        let bar = if show_bar { indicatif::ProgressBar::new(steps as u64) } else { indicatif::ProgressBar::hidden() };
        bar.set_style(indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
        let inputs = Inputs { cond: run, uncond: uncond.as_ref(), x0: x0.as_ref(), mask: mask_lat.as_ref(), noise: None };
        let lat = pipeline::sample(&dev, &dit, &s, &inputs, &|n, total| {
            bar.set_length(total as u64);
            bar.set_position(n as u64);
            if !show_bar {
                crate::info!("step {n}/{total}");
            }
        })?;
        bar.finish_and_clear();
        let t_sample = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let rgb = vae.decode(&lat, lh, lw).context("vae decode")?;
        drop(lat);
        let mut img = Rgb8 { w: gw, h: gh, data: rgb };
        if let (Some(m), Some(init)) = (&mask_gen, &init_gen) {
            img = composite(&img, init, m);
        }
        if (img.w, img.h) != (ow, oh) {
            img = img.resize_lanczos(ow, oh);
        }
        crate::info!("  sampled in {:.1}s, decoded in {:.1}s", t_sample, t1.elapsed().as_secs_f64());
        // ---- generation record
        let mut rec = genmeta::Record::new(prog, env!("CARGO_PKG_VERSION"), MODEL_NAME);
        let mut inputs_rec = Vec::new();
        if let (Some(p), Some(i)) = (&cli.init, &init_img) {
            inputs_rec.push(serde_json::json!({"tag": "init", "path": p.display().to_string(), "width": i.w, "height": i.h}));
        }
        if let Some(p) = &cli.mask {
            inputs_rec.push(serde_json::json!({"tag": "mask", "path": p.display().to_string()}));
        }
        rec.set("prompt", prompts[job.prompt].clone())
            .set("negative", negative.clone())
            .set("seed", job.seed)
            .set("steps", steps)
            .set("cfg", cfg as f64)
            .set("shift", cli.shift as f64)
            .set("turbo", if cli.turbo { Some(true) } else { None })
            .set("sampler", "euler")
            .set("skip_layers", if negative.is_some() { Some(skip_layers.clone()) } else { None })
            .set("precision", if dit.int8 { "int8" } else { "bf16" })
            .set("width", ow)
            .set("height", oh)
            .set("generated_width", if (gw, gh) != (ow, oh) { Some(gw) } else { None })
            .set("generated_height", if (gw, gh) != (ow, oh) { Some(gh) } else { None })
            .set("strength", if cli.init.is_some() { Some(cli.strength as f64) } else { None })
            .set("loras", if lora_record.is_empty() { None } else { Some(lora_record.clone()) })
            .set("inputs", if inputs_rec.is_empty() { None } else { Some(inputs_rec) })
            .set("meta", meta.clone());
        let text = [(genmeta::PNG_KEYWORD.to_string(), rec.to_json())];
        match &job.dest {
            Dest::Stdout => {
                let mut so = std::io::stdout().lock();
                so.write_all(&img.png_bytes(&text)?)?;
                so.flush()?;
                crate::info!("wrote PNG to stdout");
            }
            Dest::File(path) => {
                if let Some(d) = path.parent() {
                    if !d.as_os_str().is_empty() {
                        std::fs::create_dir_all(d)?;
                    }
                }
                img.save_png(path, &text)?;
                crate::info!("wrote {}", path.display());
                if cli.json {
                    results.push(serde_json::json!({"path": path.display().to_string(), "seed": job.seed, "width": ow, "height": oh, "prompt": prompts[job.prompt], "generation": rec.to_value()}));
                } else if !stdout_is_image {
                    println!("{}", path.display());
                    std::io::stdout().flush()?;
                }
            }
        }
    }
    if cli.json {
        let j = serde_json::json!({
            "images": results,
            "prompt": if prompts.len() == 1 { serde_json::Value::from(prompts[0].clone()) } else { serde_json::Value::from(prompts.clone()) },
            "negative": negative, "steps": steps, "cfg": cfg, "turbo": cli.turbo, "width": ow, "height": oh,
        });
        println!("{j}");
    }
    drop(dit);
    dev.sync()?;
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}
