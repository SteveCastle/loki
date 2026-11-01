//! Command-line front end of loki-dream: text -> image with Kromoss (MOSS-Image), Krea 2, FLUX.2 klein or Anima, plus
//! img2img, inpainting and (MOSS-Image) LoRAs. The diffusion model file decides the architecture (`arch::detect`).
//!
//! loki- toolkit conventions: results are written to files whose paths are printed on stdout (one per line), or one JSON
//! object for the run with `--json`, or the PNG itself with `-o -`; progress and diagnostics go to stderr only (`-q`
//! silences them); exit status 0 ok, 1 runtime error, 2 usage error.
use crate::arch::{self, Arch};
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
use clap::{CommandFactory, Parser};
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

MODELS (the diffusion model file decides the architecture; it is detected from the file's tensors)
  --model FILE|NAME (alias --dit) picks it; --arch NAME picks an architecture's default model instead (downloaded if
  missing). Each architecture brings its own text encoder and VAE, found in the model folders or downloaded.
  moss    Kromoss / MOSS-Image (default)  Qwen3.5-9B + FLUX.2 small VAE   50 steps CFG 3; --turbo: 8-step DMD2 LoRA
  krea2   Krea 2 (raw / turbo)             Qwen3-VL-4B + Qwen-Image VAE   raw 28 steps CFG 5.5; turbo 8 steps CFG 1
  flux2   FLUX.2 klein 4B / 9B             Qwen3-4B / 8B + FLUX.2 VAE     base 50 steps CFG 4; distilled 4 steps CFG 1
  anima   Anima (base / aesthetic / turbo) Qwen3-0.6B + Qwen-Image VAE    30 steps CFG 4; turbo 10 steps CFG 1
  Which variant a checkpoint is goes by its name: Krea 2 is turbo unless named raw/base (its fine-tunes are turbo-based),
  klein is distilled unless named base, Anima is base unless named turbo; --turbo forces the distilled settings and
  --steps / --cfg override any default. Other architectures (FLUX.1, SDXL, Qwen-Image, ...) are refused by name.

SIZE (default 1024x1024; the models work in multiples of 16 pixels; about 1 megapixel is their home ground)
  --size WxH | --aspect W:H [--megapixels MP (default 1)] | --megapixels MP (square)
  A size that is not a multiple of 16 is generated at the nearest multiple and Lanczos-resampled to exactly what you
  asked (--snap keeps the multiple of 16). With --init and no size flag: the init image's own size.

SAMPLING
  Flow-matching Euler with each model's official schedule (MOSS-Image: shift 5 snapped to the training grid, the negative
  branch with block 22 skipped; Krea 2 and FLUX.2: resolution-dependent shift; Anima: shift 3). --cfg is classic
  classifier-free guidance (uncond + cfg * (cond - uncond); 1 = no negative branch). --seed N reproduces an image; the
  noise is drawn exactly like torch.randn on the CPU, so a seed means the same as in diffusers/ComfyUI.

IMG2IMG / INPAINT
  --init IMG --strength F    noise the encoded image to F (0..1, default 0.6) and denoise the rest of the schedule
                             (the last F of the steps, like diffusers).
  --mask MASK (with --init)  white = regenerate, black = keep, grey = in between; any size (resized to the init). The
                             kept area is re-imposed every step and composited back at full resolution.

LAYOUT PROMPTS (MOSS-Image)
  MOSS-Image understands a structured layout prompt: a compact JSON object given as the prompt, e.g.
  {\"summary\":{\"caption\":\"a poster for a jazz night\"},\"items\":[{\"label\":\"text\",\"caption\":\"title\",\"text\":\"BLUE NOTE\",
  \"bbox\":[-400,-300,-250,300]},{\"label\":\"object\",\"caption\":\"a saxophone\",\"bbox\":[-150,-200,450,200]}],
  \"bbox_coord_space\":\"abs_centralized\",\"bbox_format\":\"yxyx\"}
  bbox = [y0, x0, y1, x1] in pixels with the origin at the image centre; label text|object|character; optional
  \"palette\". It goes through the text encoder like any prompt (see --show-prompt).

EXAMPLES
  PROG -p \"a tabby cat holding a sign that says KROMOSS\"                 # -> dream_<seed>.png
  PROG --turbo --count 4 -o out/ -p \"a lighthouse in a storm, oil painting\"
  PROG --model krea2_turbo_fp8_scaled -p \"a misty mountain lake at dawn\" --aspect 16:9
  PROG --model flux-2-klein-4b -p \"a red apple on a wooden table\"
  PROG --arch anima -p \"masterpiece, best quality, 1girl, silver hair, night city\"
  PROG --init photo.jpg --mask sky.png --strength 1 -p \"a sky full of hot air balloons\"
  PROG --lora style.safetensors:0.8 -p \"...\"                             # any MOSS-Image LoRA (kohya/ComfyUI/diffusers keys)
  PROG --turbo -o - -p \"a red apple\" | loki-reshoot --animate - -d 5 -o apple.mp4
";

#[derive(Parser, Debug)]
#[command(version, about = "loki-dream: text-to-image on a standalone CUDA engine: Kromoss (MOSS-Image), Krea 2, FLUX.2 klein and Anima, each with its own text encoder and VAE. Text to image, img2img, inpainting, LoRAs (MOSS-Image), turbo. Part of the loki- toolkit.", after_help = HELP)]
pub(crate) struct Cli {
    /// What to draw. Repeatable: several prompts in one run (models load once).
    #[arg(short, long)]
    pub(crate) prompt: Vec<String>,
    /// Read the prompt from a file (`-` = stdin); the whole file is one prompt.
    #[arg(short = 'P', long, value_name = "FILE")]
    pub(crate) prompt_file: Option<String>,
    /// A file with one prompt per line (blank lines and lines starting with # are skipped).
    #[arg(long, value_name = "FILE")]
    pub(crate) batch: Option<PathBuf>,
    /// Negative prompt (ignored with CFG 1). Default: empty, as the official pipelines.
    #[arg(long, value_name = "TEXT")]
    pub(crate) negative: Option<String>,
    /// Print the exact text(s) the text encoder gets (chat template included) and exit.
    #[arg(long)]
    pub(crate) show_prompt: bool,

    /// Output size WxH (multiples of 16 are native; others are resampled). Default 1024x1024.
    #[arg(long, value_name = "WxH")]
    pub(crate) size: Option<String>,
    /// Aspect ratio W:H at --megapixels (default 1).
    #[arg(long, value_name = "W:H")]
    pub(crate) aspect: Option<String>,
    /// Pixel count in megapixels (square unless --aspect).
    #[arg(long, value_name = "MP")]
    pub(crate) megapixels: Option<f64>,
    /// Deliver the model's multiple-of-16 size instead of resampling to the exact requested size.
    #[arg(long)]
    pub(crate) snap: bool,

    /// Sampling steps (default: the model's own, see MODELS).
    #[arg(long)]
    pub(crate) steps: Option<usize>,
    /// Classifier-free guidance scale (default: the model's own; 1 = no negative branch).
    #[arg(long)]
    pub(crate) cfg: Option<f32>,
    /// Noise seed (default: random). Several images use seed, seed+1, ...
    #[arg(long)]
    pub(crate) seed: Option<u64>,
    /// Images per prompt (one model load for all).
    #[arg(long, default_value_t = 1)]
    pub(crate) count: usize,
    /// Fast mode. MOSS-Image: the 8-step DMD2 LoRA, 8 steps, CFG 1. Krea 2 / FLUX.2 / Anima: the distilled
    /// checkpoint's settings (use with a turbo / distilled model; such files are recognised by name anyway).
    #[arg(long)]
    pub(crate) turbo: bool,
    /// Turbo LoRA flavour: dmd2 (default: best PartiPrompts score, more contrast and detail) or dmd.
    #[arg(long, value_name = "dmd2|dmd", default_value = "dmd2", hide = true)]
    pub(crate) turbo_lora: String,
    /// Apply a LoRA (repeatable): FILE or FILE:STRENGTH. MOSS-Image only: Kromoss/MOSS-format (kohya) and ComfyUI/diffusers keys.
    #[arg(long = "lora", value_name = "FILE[:SCALE]")]
    pub(crate) loras: Vec<String>,
    /// Scheduler shift (MOSS-Image 5, Anima 3; Krea 2 and FLUX.2 shift by resolution unless this is given).
    #[arg(long)]
    pub(crate) shift: Option<f32>,
    /// MOSS-Image: blocks skipped in the negative branch, comma separated, or `none` (official default 22).
    #[arg(long, value_name = "LIST")]
    pub(crate) skip_layers: Option<String>,

    /// img2img: start from this image (its size is the default output size).
    #[arg(long, value_name = "IMAGE")]
    pub(crate) init: Option<PathBuf>,
    /// img2img strength: how much of the schedule is redrawn (0..1).
    #[arg(long, default_value_t = 0.6)]
    pub(crate) strength: f32,
    /// Inpainting mask for --init: white = regenerate, black = keep; soft edges respected; any size.
    #[arg(long, value_name = "MASK")]
    pub(crate) mask: Option<PathBuf>,

    /// Output PNG path, an existing folder, or `-` for the PNG on stdout.
    #[arg(short, long)]
    pub(crate) out: Option<String>,
    /// Silence progress/diagnostics on stderr (errors still print).
    #[arg(short, long)]
    pub(crate) quiet: bool,
    /// Print one JSON object describing the run (images, seeds, sizes, generation records) instead of paths.
    #[arg(long)]
    pub(crate) json: bool,
    /// Your own JSON object, stored verbatim as the `meta` field of each image's generation record. Left out of `args`.
    #[arg(long, value_name = "JSON")]
    pub(crate) meta: Option<String>,
    /// Read --meta from a file.
    #[arg(long, value_name = "FILE")]
    pub(crate) meta_file: Option<String>,

    /// MOSS-Image: which Kromoss file (int8 ConvRot 5.4 GB, default when present, or bf16 10.2 GB). Other architectures:
    /// the compute format of the diffusion model (default: int8 for quantized files and for models too large for the
    /// GPU in bf16, else bf16).
    #[arg(long, value_name = "int8|bf16")]
    pub(crate) precision: Option<String>,
    /// Diffusion model file (alias --model): Kromoss or a MOSS-Image fine-tune, a Krea 2, FLUX.2 klein or Anima
    /// checkpoint (bf16, fp16, fp8 or int8 ConvRot). The architecture is detected from the file. A bare file name is
    /// looked up in the models folders. Default: $LOKI_DREAM_DIT, else --arch's default model.
    #[arg(long, visible_alias = "model", short = 'm')]
    pub(crate) dit: Option<PathBuf>,
    /// Architecture whose default model to use when no --model is given: moss (default), krea2, flux2, anima.
    /// With --model it must match the file.
    #[arg(long, value_name = "NAME")]
    pub(crate) arch: Option<String>,
    /// Text encoder file (the architecture's own: Qwen3.5-9B, Qwen3-VL-4B, Qwen3-4B / 8B or Qwen3-0.6B).
    #[arg(long)]
    pub(crate) text_encoder: Option<PathBuf>,
    /// VAE file (FLUX.2 small / full VAE or Qwen-Image VAE, by architecture).
    #[arg(long)]
    pub(crate) vae: Option<PathBuf>,

    /// List the diffusion models in the model folders this tool can run (architecture, variant, default settings,
    /// whether their text encoder and VAE are here), plus each architecture's default model as a download, and exit.
    /// With --json: one object {"models":[...],"default","search_dirs"}.
    #[arg(long)]
    pub(crate) list_models: bool,

    /// Run as a resident worker: read one request per line on stdin (a JSON object whose "args" are a command line for
    /// this tool), keep the models on the GPU between requests, answer with JSON events on stdout. For supervisors such
    /// as loki-worker; see CONVENTIONS.md, "Worker mode".
    #[arg(long)]
    pub(crate) worker: bool,
}

pub fn main(prog: &'static str) {
    if let Err(e) = run(prog, std::env::args().collect()) {
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
    crate::worker::forbid_stdin()?;
    let mut v = String::new();
    std::io::stdin().lock().read_to_string(&mut v).context("reading stdin")?;
    Ok(v)
}

#[derive(Clone)]
pub(crate) enum Dest {
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

// The VAE and the DiT go through the worker's residency cache (a plain load on the command line). The DiT sits in a
// RefCell: each run sets its own LoRAs on it.
fn load_vae(dev: &std::sync::Arc<Device>, p: &Path) -> Result<std::rc::Rc<Vae>> {
    crate::worker::load(&format!("vae {}", p.display()), p, || Vae::load(dev.clone(), p))
}
fn load_dit(dev: &std::sync::Arc<Device>, p: &Path) -> Result<std::rc::Rc<std::cell::RefCell<Dit>>> {
    crate::worker::load(&format!("dit {}", p.display()), p, || Ok(std::cell::RefCell::new(Dit::load(dev.clone(), p, None)?)))
}

/// Which diffusion model a run uses, before anything is downloaded.
pub(crate) enum ModelChoice {
    /// a local file (--model / --dit or $LOKI_DREAM_DIT), its architecture detected
    File(PathBuf),
    /// the architecture's default model, resolved (found or downloaded) by `resolve_model_choice`
    Default,
    /// one default model named by its file (--model anima-base-v1.0.safetensors before it was ever downloaded, as
    /// `--list-models` offers it): found or downloaded like `Default`
    Spec(&'static models::Spec),
}

/// The default models a bare --model name can download, by architecture.
const DEFAULT_SPECS: [(Arch, &models::Spec); 6] = [
    (Arch::Moss, &models::DIT_INT8),
    (Arch::Moss, &models::DIT_BF16),
    (Arch::Krea2, &models::KREA2_DIT),
    (Arch::Flux2, &models::KLEIN_DIT),
    (Arch::Anima, &models::ANIMA_DIT),
    (Arch::Anima, &models::ANIMA_TURBO_DIT),
];

/// A bare file name (no folder) that is one of the default models' own: what it would download.
fn default_named(p: &Path) -> Option<(Arch, &'static models::Spec)> {
    if p.components().count() != 1 {
        return None;
    }
    let name = p.to_string_lossy().to_lowercase();
    let name = name.strip_suffix(".safetensors").unwrap_or(&name).to_string();
    DEFAULT_SPECS.iter().find(|(_, spec)| {
        let f = spec.hf_path.rsplit('/').next().unwrap_or_default().to_lowercase();
        f.strip_suffix(".safetensors").unwrap_or(&f) == name
    }).map(|&(a, spec)| (a, spec))
}

/// The diffusion model and its architecture: --model/--dit (detected), else $LOKI_DREAM_DIT (a machine-wide default),
/// else --arch's default model (Kromoss: --precision's file, or found locally int8 first, or downloaded). Reads local
/// headers only; nothing is downloaded until `resolve_model_choice`.
pub(crate) fn choose_model(cli: &Cli) -> Result<(ModelChoice, Arch)> {
    let want = cli.arch.as_deref().map(Arch::parse).transpose()?;
    if let Some(p) = &cli.dit {
        let path = match models::resolve_model(p) {
            Ok(path) => path,
            // not on disk, but a default model's name: download it rather than fail
            Err(e) => match default_named(p) {
                Some((a, spec)) => {
                    if let Some(w) = want {
                        ensure!(w == a, "--arch {} but {} is a {} model", w.id(), p.display(), a.name());
                    }
                    return Ok((ModelChoice::Spec(spec), a));
                }
                None => return Err(e),
            },
        };
        let a = arch::detect_file(&path)?;
        if let Some(w) = want {
            ensure!(w == a, "--arch {} but {} is a {} model", w.id(), path.display(), a.name());
        }
        return Ok((ModelChoice::File(path), a));
    }
    // the machine default: skipped when --arch chooses, and for a MOSS-Image model when --precision picks a Kromoss file
    // (for the other architectures --precision is the compute format of the same model)
    if let Some(p) = std::env::var_os("LOKI_DREAM_DIT").filter(|v| !v.is_empty()).map(PathBuf::from) {
        if want.is_none() {
            let path = models::resolve_model(&p).context("$LOKI_DREAM_DIT")?;
            let a = arch::detect_file(&path)?;
            if a != Arch::Moss || cli.precision.is_none() {
                return Ok((ModelChoice::File(path), a));
            }
        }
    }
    Ok((ModelChoice::Default, want.unwrap_or(Arch::Moss)))
}

/// The default model's spec for an architecture.
fn default_spec(cli: &Cli, a: Arch) -> Result<&'static models::Spec> {
    Ok(match a {
        Arch::Moss => match cli.precision.as_deref() {
            Some("bf16") => &models::DIT_BF16,
            Some("int8") | None => &models::DIT_INT8,
            Some(o) => bail!("--precision must be int8 or bf16, not {o}"),
        },
        Arch::Krea2 => &models::KREA2_DIT,
        Arch::Flux2 => &models::KLEIN_DIT,
        Arch::Anima => &models::ANIMA_DIT,
    })
}

/// The file name a choice will have (to read the variant from before a download).
pub(crate) fn choice_name(cli: &Cli, c: &ModelChoice, a: Arch) -> Result<PathBuf> {
    Ok(match c {
        ModelChoice::File(p) => p.clone(),
        ModelChoice::Default => PathBuf::from(default_spec(cli, a)?.hf_path.rsplit('/').next().unwrap_or_default()),
        ModelChoice::Spec(spec) => PathBuf::from(spec.hf_path.rsplit('/').next().unwrap_or_default()),
    })
}

/// The choice as a local file, downloading the default model when it is missing.
pub(crate) fn resolve_model_choice(cli: &Cli, c: ModelChoice, a: Arch) -> Result<PathBuf> {
    let path = match c {
        ModelChoice::File(p) => return Ok(p),
        ModelChoice::Default if a == Arch::Moss && cli.precision.is_none() => match models::find_local(&models::DIT_INT8).or_else(|| models::find_local(&models::DIT_BF16)) {
            Some(p) => p,
            None => models::resolve(None, &models::DIT_INT8)?,
        },
        ModelChoice::Default => models::resolve(None, default_spec(cli, a)?)?,
        ModelChoice::Spec(spec) => models::resolve(None, spec)?,
    };
    let detected = arch::detect_file(&path)?;
    ensure!(detected == a, "{} is a {} model, not {}", path.display(), detected.name(), a.name());
    Ok(path)
}

/// `choose_model` then `resolve_model_choice`.
pub(crate) fn dit_path(cli: &Cli) -> Result<(PathBuf, Arch)> {
    let (c, a) = choose_model(cli)?;
    Ok((resolve_model_choice(cli, c, a)?, a))
}

/// The name to stamp as the record's `model` when `dit` is not one of Kromoss's own files (a same-architecture fine-tune
/// swapped in with --dit or $LOKI_DREAM_DIT): the file's name without `.safetensors`. None for Kromoss itself.
fn swapped_model_name(dit: &Path) -> Option<String> {
    if models::is_spec_file(&models::DIT_INT8, dit) || models::is_spec_file(&models::DIT_BF16, dit) {
        return None;
    }
    Some(file_stem(dit))
}

pub(crate) fn file_stem(p: &Path) -> String {
    let name = p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    name.strip_suffix(".safetensors").unwrap_or(&name).to_string()
}

/// One image to make.
pub(crate) struct Job {
    pub(crate) prompt: usize,
    pub(crate) seed: u64,
    pub(crate) dest: Dest,
}

/// Everything decided before a model loads, shared by every architecture's run.
pub(crate) struct Plan {
    pub(crate) prog: &'static str,
    pub(crate) prompts: Vec<String>,
    pub(crate) negative: Option<String>,
    pub(crate) steps: usize,
    pub(crate) cfg: f32,
    pub(crate) jobs: Vec<Job>,
    /// generated (multiple of 16) and delivered size
    pub(crate) gen: (usize, usize),
    pub(crate) out: (usize, usize),
    pub(crate) init_img: Option<Rgb8>,
    pub(crate) init_gen: Option<Rgb8>,
    pub(crate) mask_gen: Option<Mask>,
    pub(crate) meta: Option<serde_json::Value>,
    pub(crate) dit: PathBuf,
    pub(crate) arch: Arch,
    pub(crate) stdout_is_image: bool,
}

/// Finish one image: composite an inpaint, resample to the delivered size, stamp the record, write it, report it.
pub(crate) fn deliver(cli: &Cli, plan: &Plan, job: &Job, mut img: Rgb8, rec: &mut genmeta::Record, results: &mut Vec<serde_json::Value>) -> Result<()> {
    let (ow, oh) = plan.out;
    let (gw, gh) = plan.gen;
    if let (Some(m), Some(init)) = (&plan.mask_gen, &plan.init_gen) {
        img = composite(&img, init, m);
    }
    if (img.w, img.h) != (ow, oh) {
        img = img.resize_lanczos(ow, oh);
    }
    let mut inputs_rec = Vec::new();
    if let (Some(p), Some(i)) = (&cli.init, &plan.init_img) {
        inputs_rec.push(serde_json::json!({"tag": "init", "path": p.display().to_string(), "width": i.w, "height": i.h}));
    }
    if let Some(p) = &cli.mask {
        inputs_rec.push(serde_json::json!({"tag": "mask", "path": p.display().to_string()}));
    }
    rec.set("dit", plan.dit.file_name().map(|f| f.to_string_lossy().into_owned()))
        .set("width", ow)
        .set("height", oh)
        .set("generated_width", if (gw, gh) != (ow, oh) { Some(gw) } else { None })
        .set("generated_height", if (gw, gh) != (ow, oh) { Some(gh) } else { None })
        .set("strength", if cli.init.is_some() { Some(cli.strength as f64) } else { None })
        .set("inputs", if inputs_rec.is_empty() { None } else { Some(inputs_rec) })
        .set("meta", plan.meta.clone());
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
                results.push(serde_json::json!({"path": path.display().to_string(), "seed": job.seed, "width": ow, "height": oh, "prompt": plan.prompts[job.prompt], "generation": rec.to_value()}));
            } else if !plan.stdout_is_image {
                crate::worker::out_line(&path.display().to_string());
                std::io::stdout().flush()?;
            }
        }
    }
    Ok(())
}

/// The run's JSON object (--json).
pub(crate) fn finish_json(cli: &Cli, plan: &Plan, results: Vec<serde_json::Value>, turbo: bool) {
    if cli.json {
        let prompts = &plan.prompts;
        let j = serde_json::json!({
            "images": results,
            "prompt": if prompts.len() == 1 { serde_json::Value::from(prompts[0].clone()) } else { serde_json::Value::from(prompts.clone()) },
            "negative": plan.negative, "steps": plan.steps, "cfg": plan.cfg, "turbo": turbo, "width": plan.out.0, "height": plan.out.1,
            "architecture": plan.arch.id(),
        });
        crate::worker::out_line(&j.to_string());
    }
}

/// A progress bar on a terminal, `step i/n` lines otherwise.
pub(crate) fn step_progress(cli: &Cli, steps: usize) -> (indicatif::ProgressBar, bool) {
    let show = !cli.quiet && std::io::stderr().is_terminal();
    let bar = if show { indicatif::ProgressBar::new(steps as u64) } else { indicatif::ProgressBar::hidden() };
    bar.set_style(indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
    (bar, show)
}

fn run(prog: &'static str, argv: Vec<String>) -> Result<()> {
    let cmd = Cli::command().name(prog).bin_name(prog).after_help(HELP.replace("PROG", prog));
    let Some(cli) = crate::worker::parse_args::<Cli>(cmd, argv)? else { return Ok(()) };
    crate::log::set_quiet(cli.quiet);
    if cli.worker {
        ensure!(!crate::worker::active(), "--worker inside a worker request");
        return crate::worker::serve(&crate::worker::Tool {
            name: prog,
            version: env!("CARGO_PKG_VERSION"),
            model: MODEL_NAME,
            run: &|argv| run(prog, argv),
            warm: &|_| {
                let (dit_path, a) = dit_path(&cli)?;
                if a != Arch::Moss {
                    return crate::engines::warm(&cli, a, &dit_path);
                }
                models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?;
                let vae_path = models::resolve(cli.vae.clone(), &models::VAE)?;
                let dev = crate::worker::device()?;
                crate::ops::gemm_init(&dev)?;
                crate::ops::attn_init(&dev)?;
                Vae::init_kernels(&dev)?;
                load_vae(&dev, &vae_path)?;
                load_dit(&dev, &dit_path)?;
                Ok(())
            },
            custom: None,
        });
    }
    if cli.list_models {
        let v = crate::catalog::list();
        if cli.json {
            crate::worker::out_line(&v.to_string());
        } else {
            for l in crate::catalog::table(&v) {
                crate::worker::out_line(&l);
            }
        }
        return Ok(());
    }
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
    // the chat templates trim the prompt; the record and file names see the same text
    let prompts: Vec<String> = prompts.iter().map(|p| p.trim().to_string()).collect();
    ensure!(!prompts.is_empty(), "give a prompt with -p TEXT, -P FILE or --batch FILE");
    ensure!(cli.count >= 1, "--count must be at least 1");

    // ---- the model decides the architecture and its defaults
    let (choice, a) = choose_model(&cli)?;
    let distilled = cli.turbo || crate::engines::is_distilled(a, &choice_name(&cli, &choice, a)?);
    let (def_steps, def_cfg) = match a {
        Arch::Moss => (if cli.turbo { 8 } else { 50 }, if cli.turbo { 1.0 } else { 3.0 }),
        _ => crate::engines::defaults(a, distilled),
    };
    let steps = cli.steps.unwrap_or(def_steps);
    ensure!((1..=1000).contains(&steps), "--steps must be in 1..1000");
    let cfg = if a == Arch::Moss && cli.turbo {
        if cli.cfg.map(|c| c != 1.0).unwrap_or(false) {
            crate::info!("warning: --turbo runs at CFG 1; --cfg {} ignored", cli.cfg.unwrap());
        }
        1.0
    } else {
        cli.cfg.unwrap_or(def_cfg)
    };
    ensure!(cfg >= 0.0 && cfg.is_finite(), "--cfg must be >= 0");
    if cfg == 1.0 && cli.negative.is_some() {
        crate::info!("warning: CFG 1 has no negative branch; --negative ignored");
    }
    let negative = if cfg != 1.0 { Some(cli.negative.clone().unwrap_or_default()) } else { None };
    if a != Arch::Moss {
        ensure!(cli.loras.is_empty(), "--lora is supported for MOSS-Image models only ({} model given)", a.name());
        ensure!(cli.skip_layers.is_none(), "--skip-layers is a MOSS-Image setting ({} model given)", a.name());
    }
    if cli.show_prompt {
        let shown = |p: &str| -> Vec<String> { if a == Arch::Moss { vec![text_encoder::template(p)] } else { crate::engines::show_prompt(a, p) } };
        for p in &prompts {
            for l in shown(p) {
                crate::worker::out_line(&l);
            }
        }
        if let Some(n) = &negative {
            crate::worker::out_line(&format!("--- negative (CFG {cfg}) ---"));
            for l in shown(n) {
                crate::worker::out_line(&l);
            }
        }
        return Ok(());
    }
    ensure!(cli.mask.is_none() || cli.init.is_some(), "--mask needs --init (the image to inpaint)");
    ensure!((0.0..=1.0).contains(&cli.strength), "--strength must be in 0..1");
    let dit = resolve_model_choice(&cli, choice, a)?;

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
                    ensure!(!crate::worker::active(), "-o - (PNG on stdout) is not available in --worker mode; give a file or folder");
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
    let stdout_is_image = jobs.iter().any(|j| matches!(j.dest, Dest::Stdout));
    let plan = Plan { prog, prompts, negative, steps, cfg, jobs, gen: (gw, gh), out: (ow, oh), init_img, init_gen, mask_gen, meta, dit, arch: a, stdout_is_image };
    match a {
        Arch::Moss => run_moss(&cli, &plan),
        _ => crate::engines::run(&cli, &plan, distilled),
    }
}

/// MOSS-Image (Kromoss): Qwen3.5-9B, the MOSS DiT with LoRAs and the official snapped schedule, the FLUX.2 small VAE.
fn run_moss(cli: &Cli, plan: &Plan) -> Result<()> {
    let prog = plan.prog;
    let (prompts, negative, steps, cfg) = (&plan.prompts, &plan.negative, plan.steps, plan.cfg);
    let ((gw, gh), (ow, oh)) = (plan.gen, plan.out);
    let skip_layers: Vec<usize> = match cli.skip_layers.as_deref().map(str::trim) {
        None => vec![22],
        Some("none") | Some("") => Vec::new(),
        Some(s) => s.split(',').map(|s| s.trim().parse::<usize>().context("--skip-layers: comma-separated block indices or `none`")).collect::<Result<_>>()?,
    };
    ensure!(skip_layers.iter().all(|&l| l < crate::dit::LAYERS), "--skip-layers: block indices must be < {}", crate::dit::LAYERS);
    let shift = cli.shift.unwrap_or(5.0);
    let loras: Vec<LoraSpec> = cli.loras.iter().map(|s| LoraSpec::parse(s)).collect::<Result<_>>()?;
    for l in &loras {
        ensure!(l.path.is_file(), "LoRA {} not found", l.path.display());
    }

    // ---- models
    let dit_path = &plan.dit;
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
    crate::info!("{prog}: {} image(s), {}x{}{}, {} steps, CFG {}{}", plan.jobs.len(), ow, oh, if (gw, gh) != (ow, oh) { format!(" (generated at {gw}x{gh})") } else { String::new() }, steps, cfg, if cli.turbo { " (turbo)" } else { "" });
    crate::info!("  dit: {}\n  text encoder: {}\n  vae: {}", dit_path.display(), te_path.display(), vae_path.display());

    let t_dev = std::time::Instant::now();
    let dev = crate::worker::device()?;
    crate::info!("  gpu: {} MB free of {} MB (cuda init {:.1}s)", dev.free_mem()? >> 20, dev.total_mem >> 20, t_dev.elapsed().as_secs_f64());
    crate::ops::gemm_init(&dev)?;
    crate::ops::attn_init(&dev)?;
    Vae::init_kernels(&dev)?;
    let t_all = std::time::Instant::now();

    // ---- phase 1: text encoder (streamed layer by layer); every prompt and the negative once
    let mut texts: Vec<String> = prompts.clone();
    if let Some(n) = negative {
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
    let vae = load_vae(&dev, &vae_path)?;
    let (lh, lw) = (gh / 16, gw / 16);
    let x0 = match &plan.init_gen {
        Some(img) => Some(vae.encode(&img.to_f32(), gh, gw).context("vae encode")?),
        None => None,
    };
    let mask_lat = match &plan.mask_gen {
        Some(m) => Some(Tensor::from_f32(&dev, &m.max_down(16).data, &[lh * lw])?),
        None => None,
    };

    // ---- phase 3: DiT (+ LoRAs); sample, decode, write each image as soon as it is ready
    let dit_cell = load_dit(&dev, dit_path)?;
    let mut dit = dit_cell.borrow_mut();
    // always assigned: a resident DiT still carries the previous run's LoRAs
    dit.lora = Vec::new();
    if !all_loras.is_empty() {
        let t0 = std::time::Instant::now();
        dit.lora = lora::load(&dev, &all_loras, crate::dit::LAYERS)?;
        crate::info!("  loras ready in {:.1}s", t0.elapsed().as_secs_f64());
    }
    let mut runs: HashMap<usize, crate::dit::Run> = HashMap::new();
    let uncond = match negative {
        Some(_) => Some(dit.prepare(&feats[prompts.len()].hidden, lh, lw)?),
        None => None,
    };
    let mut results: Vec<serde_json::Value> = Vec::new();
    let settings_base = Settings { steps, cfg, shift, seed: 0, skip_layers: skip_layers.clone(), strength: if cli.init.is_some() { cli.strength } else { 1.0 } };
    let lora_record: Vec<serde_json::Value> = all_loras.iter().map(|l| serde_json::json!({"path": l.path.display().to_string(), "strength": l.strength})).collect();
    for (ji, job) in plan.jobs.iter().enumerate() {
        if !runs.contains_key(&job.prompt) {
            runs.insert(job.prompt, dit.prepare(&feats[job.prompt].hidden, lh, lw)?);
        }
        let run = &runs[&job.prompt];
        let mut s = settings_base.clone();
        s.seed = job.seed;
        crate::info!("image {}/{}: seed {} {}x{}", ji + 1, plan.jobs.len(), job.seed, ow, oh);
        let t0 = std::time::Instant::now();
        let (bar, show_bar) = step_progress(cli, steps);
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
        crate::info!("  sampled in {:.1}s, decoded in {:.1}s", t_sample, t1.elapsed().as_secs_f64());
        // ---- generation record
        let swapped = swapped_model_name(dit_path);
        let mut rec = genmeta::Record::new(prog, env!("CARGO_PKG_VERSION"), swapped.as_deref().unwrap_or(MODEL_NAME));
        rec.set("base_model", swapped.as_ref().map(|_| MODEL_NAME));
        rec.set("prompt", prompts[job.prompt].clone())
            .set("negative", negative.clone())
            .set("seed", job.seed)
            .set("steps", steps)
            .set("cfg", cfg as f64)
            .set("shift", shift as f64)
            .set("turbo", if cli.turbo { Some(true) } else { None })
            .set("sampler", "euler")
            .set("skip_layers", if negative.is_some() { Some(skip_layers.clone()) } else { None })
            .set("precision", if dit.int8 { "int8" } else { "bf16" })
            .set("loras", if lora_record.is_empty() { None } else { Some(lora_record.clone()) });
        deliver(cli, plan, job, Rgb8 { w: gw, h: gh, data: rgb }, &mut rec, &mut results)?;
    }
    finish_json(cli, plan, results, cli.turbo);
    drop(dit);
    dev.sync()?;
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_models_bare_name_downloads_it() {
        assert_eq!(default_named(Path::new("anima-base-v1.0.safetensors")).map(|(a, _)| a), Some(Arch::Anima));
        assert_eq!(default_named(Path::new("anima-turbo-v1.1")).map(|(_, s)| s.hf_path), Some(models::ANIMA_TURBO_DIT.hf_path));
        assert_eq!(default_named(Path::new("FLUX-2-KLEIN-4B")).map(|(a, _)| a), Some(Arch::Flux2));
        assert_eq!(default_named(Path::new("krea2_turbo_int8_convrot.safetensors")).map(|(a, _)| a), Some(Arch::Krea2));
        // a path, or any other name, is never a download
        assert!(default_named(Path::new("models/anima-turbo-v1.1.safetensors")).is_none());
        assert!(default_named(Path::new("my-finetune.safetensors")).is_none());
    }

    #[test]
    fn swapped_model_name_names_finetunes_only() {
        assert_eq!(swapped_model_name(Path::new("C:/models/kromoss_int8_convrot.safetensors")), None);
        assert_eq!(swapped_model_name(Path::new("/m/kromoss_bf16.safetensors")), None);
        assert_eq!(swapped_model_name(Path::new("C:/models/my_moss_finetune_fp16.safetensors")).as_deref(), Some("my_moss_finetune_fp16"));
    }

    #[test]
    fn model_is_an_alias_of_dit() {
        let c = Cli::try_parse_from(["loki-dream", "--model", "x.safetensors", "-p", "a"]).unwrap();
        assert_eq!(c.dit.as_deref(), Some(Path::new("x.safetensors")));
        let c = Cli::try_parse_from(["loki-dream", "-m", "y", "-p", "a"]).unwrap();
        assert_eq!(c.dit.as_deref(), Some(Path::new("y")));
    }
}
