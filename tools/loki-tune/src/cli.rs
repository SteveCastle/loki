//! The loki-tune command line: a run folder per LoRA, from images to a trained, checkpointed LoRA.
use crate::cache::Cache;
use crate::data::{self, Item};
use crate::lora::{self, LoraSet};
use crate::plan::{self, Timesteps};
use crate::progress::hms;
use crate::train::Trainer;
use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use loki_dream::cuda::Device;
use loki_dream::dit::{Dit, LATENT, LAYERS};
use loki_dream::genmeta;
use loki_dream::models;
use loki_dream::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PROG: &str = "loki-tune";

const AFTER_HELP: &str = r#"INPUTS
  Images, folders (searched recursively), list files: .txt / .lst = one image path per line, .jsonl = one
  {"path": "...", "caption": "..."} per line (relative paths are relative to the list), and - = such a list on stdin.
  Captions: a list's caption, else a sidecar text file (photo.txt or photo.jpg.txt), else (--captions auto)
  loki-scribe writes a long and a short caption (each step trains on one of them). --trigger WORD puts WORD first
  in every caption. --kind decides what the captions leave to the trigger:
    character  the person is only "TRIGGER (a CLASS)": no face, hair, body or age; clothing, pose, setting described
    style      the content described in full, the style never named (no medium, technique, palette, artist)
    object     the object is only "TRIGGER": not its design; placement, surroundings, lighting described
    concept    the pose / action / effect is only "TRIGGER"; everything else described
    general    everything described (the default)
  Review the captions in RUN/run.json with --dry-run first; edit them there or as sidecar .txt files.

RUN FOLDER (-o DIR, default ./NAME)
  run.json                          the settings, every image with its caption and training size
  checkpoints/step-000250/          NAME-step000250.safetensors (the LoRA at that step) + state.safetensors
  samples/step-000250_0.png         preview images per checkpoint (with --turbo, 8 steps; step-000000 = no LoRA)
  log.jsonl                         one line per step: loss, learning rate, gradient norm, seconds
  NAME.safetensors                  the final LoRA
  cache/                            text features and latents (deleted when training completes; --keep-cache)

CHECKPOINTS AND RESUMING
  A checkpoint is written every --save-every steps and when the run is stopped (Ctrl-C once: finish the step,
  save, exit; twice: exit now; or create a file named STOP in the run folder). --resume DIR continues from the
  run's latest checkpoint, --resume DIR/checkpoints/step-000500 from that one; resuming an earlier checkpoint branches into DIR-from500 (or -o)
  and leaves the original untouched. Training settings (--steps, --lr, --save-every, samples) may be changed
  on resume; the images, model, rank and resolution are the run's.

OUTPUT
  stdout: the path of the final LoRA (or the last checkpoint's when stopped early); --json one object
  {"lora", "run", "steps", "checkpoints": [...], "samples": [...], "loss"}.
  stderr: progress; without a terminal, one line per step: "step 12/1000 loss 0.4123 avg 0.4510 lr 1.0e-4 ...".
  The LoRA uses Kromoss's key layout (diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}) and loads
  in loki-dream (--lora FILE[:SCALE], bf16 and int8 models alike) and ComfyUI.

EXAMPLES
  loki-tune --kind character --trigger ohwx --class woman photos/   # character LoRA, captions by loki-scribe
  loki-tune --kind style --trigger inkwash -o runs/inkwash art/     # style LoRA
  loki-tune --model my_moss_finetune.safetensors set.jsonl           # any MOSS-Image checkpoint as the base
  loki-tune --resume runs/style --steps 3000                         # stop with Ctrl-C, continue later, extend
  loki-tune --resume runs/style/checkpoints/step-000750              # branch from an earlier checkpoint
  loki-dream --turbo --lora runs/style/inkwash.safetensors -p "ohwx, ..."
"#;

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Captions {
    /// list / sidecar captions, loki-scribe for the rest
    Auto,
    /// list / sidecar captions only (images without one train on the trigger alone)
    Given,
    /// no captions: every image trains on the trigger alone
    None,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum TimestepArg {
    Moss,
    Uniform,
    LogitNormal,
}

#[derive(Parser, Debug)]
#[command(name = "loki-tune", version, about = "Train LoRAs for MOSS-Image (Kromoss) models from a folder of images (standalone CUDA trainer)", after_help = AFTER_HELP)]
pub struct Cli {
    /// Images, folders, list files (.txt / .lst / .jsonl) or - (a list on stdin)
    pub inputs: Vec<String>,
    /// Run folder (default ./NAME)
    #[arg(short, long)]
    pub out: Option<PathBuf>,
    /// LoRA name (file names); default: the trigger, else the first input's folder name
    #[arg(long)]
    pub name: Option<String>,
    /// Trigger word put first in every caption
    #[arg(long)]
    pub trigger: Option<String>,
    /// What the LoRA teaches; decides how loki-scribe captions the images [default: general]
    #[arg(long, value_enum)]
    pub kind: Option<data::Kind>,
    /// What the trigger is (woman, dog, mug...): the captions call the subject "TRIGGER, a CLASS,"
    #[arg(long)]
    pub class: Option<String>,
    /// What every image shares, which the captions leave to the trigger (default: loki-scribe finds it for
    /// character, style and object LoRAs)
    #[arg(long)]
    pub traits: Option<String>,
    /// The MOSS-Image checkpoint to train against (path or name in the model folders; default Kromoss bf16, else int8)
    #[arg(short, long, alias = "dit")]
    pub model: Option<PathBuf>,
    /// LoRA rank (multiple of 8) [default: 16]
    #[arg(long)]
    pub rank: Option<usize>,
    /// LoRA alpha [default: the rank]
    #[arg(long)]
    pub alpha: Option<f32>,
    /// Learning rate [default: 1e-4]
    #[arg(long)]
    pub lr: Option<f32>,
    /// Optimizer steps [default: about 50 epochs (70 for styles), 1000-3000]
    #[arg(long)]
    pub steps: Option<usize>,
    /// Images per optimizer step (gradient accumulation) [default: 1]
    #[arg(long)]
    pub batch: Option<usize>,
    /// Learning-rate warmup steps [default: 50]
    #[arg(long)]
    pub warmup: Option<usize>,
    /// Cosine learning-rate decay to 10% (default: constant)
    #[arg(long)]
    pub cosine: bool,
    /// Training size: about RES x RES pixels per image, aspect kept [default: 1024]
    #[arg(long)]
    pub resolution: Option<usize>,
    /// Noise-level sampling [default: moss]
    #[arg(long, value_enum)]
    pub timesteps: Option<TimestepArg>,
    /// Share of steps drawn from the clean half (sigma < 0.5) with --timesteps moss [default: 0.3]
    #[arg(long)]
    pub clean_share: Option<f64>,
    /// Probability of training a step on the empty caption [default: 0]
    #[arg(long)]
    pub caption_dropout: Option<f64>,
    /// Where captions come from [default: auto]
    #[arg(long, value_enum)]
    pub captions: Option<Captions>,
    /// The instruction loki-scribe captions with
    #[arg(long)]
    pub caption_prompt: Option<String>,
    /// Also train on mirrored images (not for text, logos or asymmetric subjects)
    #[arg(long)]
    pub flip: bool,
    /// AdamW weight decay [default: 0.01]
    #[arg(long)]
    pub weight_decay: Option<f32>,
    /// Gradient clipping (global norm) [default: 1]
    #[arg(long)]
    pub grad_clip: Option<f32>,
    /// Checkpoint every N steps [default: 250]
    #[arg(long)]
    pub save_every: Option<usize>,
    /// Preview prompt (repeatable) [default: the first two captions]
    #[arg(long = "sample")]
    pub samples: Vec<String>,
    /// Previews every N steps [default: --save-every]
    #[arg(long)]
    pub sample_every: Option<usize>,
    /// Preview steps; 8 or fewer use the 8-step DMD2 turbo LoRA at CFG 1, more use CFG 3 [default: 8]
    #[arg(long)]
    pub sample_steps: Option<usize>,
    /// Preview size [default: RES x RES]
    #[arg(long)]
    pub sample_size: Option<String>,
    /// No previews
    #[arg(long)]
    pub no_samples: bool,
    /// Continue a run (its folder: latest checkpoint) or branch from a checkpoint folder
    #[arg(long)]
    pub resume: Option<PathBuf>,
    /// Keep cache/ when the run completes
    #[arg(long)]
    pub keep_cache: bool,
    /// Seed for the LoRA init, data order, noise and noise levels [default: 0]
    #[arg(long)]
    pub seed: Option<u64>,
    /// Qwen3.5-9B text encoder override
    #[arg(long)]
    pub text_encoder: Option<PathBuf>,
    /// FLUX.2 small VAE override
    #[arg(long)]
    pub vae: Option<PathBuf>,
    /// Caption and plan the run (RUN/run.json), print it and exit; start it with --resume RUN
    #[arg(long)]
    pub dry_run: bool,
    /// Caller's JSON object, stored verbatim in the LoRA's generation record (`meta`)
    #[arg(long)]
    pub meta: Option<String>,
    /// No progress on stderr
    #[arg(short, long)]
    pub quiet: bool,
    /// One JSON object on stdout
    #[arg(long)]
    pub json: bool,
}

/// Everything a run is, saved as run.json.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    pub loki_tune: u32,
    pub version: String,
    pub name: String,
    pub model: PathBuf,
    pub trigger: Option<String>,
    #[serde(default = "general")]
    pub kind: data::Kind,
    #[serde(default)]
    pub class: Option<String>,
    /// shared traits the captions leave out (see --traits)
    #[serde(default)]
    pub traits: Option<String>,
    pub rank: usize,
    pub alpha: f32,
    pub lr: f32,
    pub steps: usize,
    pub batch: usize,
    pub warmup: usize,
    pub cosine: bool,
    pub resolution: usize,
    pub timesteps: Timesteps,
    pub clean_share: f64,
    pub caption_dropout: f64,
    pub captions: Captions,
    pub flip: bool,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub seed: u64,
    pub save_every: usize,
    pub sample_every: usize,
    pub samples: Vec<String>,
    pub sample_steps: usize,
    pub sample_size: (usize, usize),
    /// the cache folder when it is not this run's (a branch shares its origin's)
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub meta: Option<serde_json::Value>,
    pub items: Vec<Item>,
}

impl Config {
    fn texts(&self) -> Vec<String> {
        let mut t: Vec<String> = self.items.iter().map(|it| data::with_trigger(&it.caption, self.trigger.as_deref())).collect();
        t.extend(self.items.iter().filter_map(|it| it.short.as_deref()).map(|s| data::with_trigger(s, self.trigger.as_deref())));
        t.push(String::new());
        t.extend(self.samples.iter().cloned());
        t
    }
}

fn general() -> data::Kind {
    data::Kind::General
}

static STOP: AtomicUsize = AtomicUsize::new(0);

fn parse_size(s: &str) -> Result<(usize, usize)> {
    let (w, h) = s.split_once(['x', 'X']).context("size: WxH")?;
    let (w, h): (usize, usize) = (w.trim().parse()?, h.trim().parse()?);
    ensure!(w >= 256 && h >= 256 && w % 16 == 0 && h % 16 == 0, "size {w}x{h}: multiples of 16, at least 256");
    Ok((w, h))
}

/// The default base: Kromoss bf16 when present (bf16 gradients track an f32 reference far better than the int8
/// model's), else the int8 file, else download the bf16 one.
fn default_model() -> Result<PathBuf> {
    if let Some(p) = models::find_local(&models::DIT_BF16) {
        return Ok(p);
    }
    if let Some(p) = models::find_local(&models::DIT_INT8) {
        return Ok(p);
    }
    models::resolve(None, &models::DIT_BF16)
}

fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '_' }).collect();
    let s = s.trim_matches(['_', '.']).to_string();
    if s.is_empty() {
        "lora".into()
    } else {
        s
    }
}

fn ckpt_name(step: usize) -> String {
    format!("step-{step:06}")
}

/// Checkpoints of a run, sorted by step.
fn checkpoints(run: &Path) -> Vec<(usize, PathBuf)> {
    let mut v: Vec<(usize, PathBuf)> = std::fs::read_dir(run.join("checkpoints"))
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| {
                    let p = e.path();
                    let step = p.file_name()?.to_str()?.strip_prefix("step-")?.parse().ok()?;
                    p.join("state.safetensors").is_file().then_some((step, p))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn read_config(run: &Path) -> Result<Config> {
    let p = run.join("run.json");
    let s = std::fs::read_to_string(&p).with_context(|| format!("{} is not a loki-tune run (no run.json)", run.display()))?;
    serde_json::from_str(&s).with_context(|| format!("reading {}", p.display()))
}

fn write_config(run: &Path, c: &Config) -> Result<()> {
    let p = run.join("run.json");
    let tmp = run.join("run.json.part");
    std::fs::write(&tmp, serde_json::to_string_pretty(c)?)?;
    std::fs::rename(&tmp, &p).with_context(|| format!("writing {}", p.display()))
}

struct Start {
    run: PathBuf,
    cfg: Config,
    /// (step, checkpoint folder) to continue from
    from: Option<(usize, PathBuf)>,
}

/// Apply the settings that may change on resume.
fn overrides(cli: &Cli, c: &mut Config) -> Result<()> {
    if let Some(v) = cli.steps {
        c.steps = v;
    }
    if let Some(v) = cli.lr {
        c.lr = v;
    }
    if let Some(v) = cli.warmup {
        c.warmup = v;
    }
    if cli.cosine {
        c.cosine = true;
    }
    if let Some(v) = cli.save_every {
        // previews follow the checkpoints unless their own interval is given
        if c.sample_every == c.save_every && cli.sample_every.is_none() {
            c.sample_every = v.max(1);
        }
        c.save_every = v.max(1);
    }
    if let Some(v) = cli.sample_every {
        c.sample_every = v;
    }
    if !cli.samples.is_empty() {
        c.samples = cli.samples.clone();
    }
    if cli.no_samples {
        c.samples.clear();
    }
    if let Some(v) = cli.sample_steps {
        c.sample_steps = v.max(1);
    }
    if let Some(s) = &cli.sample_size {
        c.sample_size = parse_size(s)?;
    }
    if let Some(v) = cli.caption_dropout {
        c.caption_dropout = v;
    }
    if let Some(v) = cli.weight_decay {
        c.weight_decay = v;
    }
    if let Some(v) = cli.grad_clip {
        c.grad_clip = v;
    }
    Ok(())
}

fn resume(cli: &Cli, at: &Path) -> Result<Start> {
    ensure!(cli.inputs.is_empty(), "--resume continues a run on its own images; give no inputs");
    for (set, flag) in [(cli.rank.is_some(), "--rank"), (cli.alpha.is_some(), "--alpha"), (cli.model.is_some(), "--model"), (cli.resolution.is_some(), "--resolution"), (cli.trigger.is_some(), "--trigger"), (cli.kind.is_some(), "--kind"), (cli.class.is_some(), "--class"), (cli.traits.is_some(), "--traits"), (cli.flip, "--flip"), (cli.batch.is_some(), "--batch"), (cli.seed.is_some(), "--seed")] {
        ensure!(!set, "{flag} is fixed by the run being resumed");
    }
    let (run, chosen) = if at.join("run.json").is_file() {
        let last = checkpoints(at).pop();
        (at.to_path_buf(), last)
    } else if at.join("state.safetensors").is_file() {
        let run = at.parent().and_then(Path::parent).context("a checkpoint folder lives in RUN/checkpoints/")?.to_path_buf();
        let step = lora::state_meta(&at.join("state.safetensors"))?["step"].as_u64().context("checkpoint without a step")? as usize;
        (run, Some((step, at.to_path_buf())))
    } else {
        bail!("{}: neither a run folder (run.json) nor a checkpoint folder (state.safetensors)", at.display());
    };
    let mut cfg = read_config(&run)?;
    overrides(cli, &mut cfg)?;
    let latest = checkpoints(&run).last().map(|c| c.0);
    let Some((step, ck)) = chosen else {
        // a run stopped before its first checkpoint: start it again in place
        return Ok(Start { run, cfg, from: None });
    };
    let branch = cli.out.clone().or_else(|| (Some(step) != latest).then(|| PathBuf::from(format!("{}-from{step}", run.display()))));
    match branch {
        None => {
            if let Some(o) = &cli.out {
                ensure!(o == &run, "-o: a run continues in its own folder");
            }
            Ok(Start { run, cfg, from: Some((step, ck)) })
        }
        Some(dir) => {
            ensure!(!dir.join("run.json").exists(), "{} already holds a run; --resume it, or choose another -o", dir.display());
            let dst = dir.join("checkpoints").join(ckpt_name(step));
            std::fs::create_dir_all(&dst)?;
            for e in std::fs::read_dir(&ck)? {
                let p = e?.path();
                std::fs::copy(&p, dst.join(p.file_name().unwrap()))?;
            }
            cfg.cache_dir = Some(cfg.cache_dir.clone().unwrap_or_else(|| run.join("cache")));
            write_config(&dir, &cfg)?;
            crate::info!("{PROG}: branching from {} at step {step} into {}", run.display(), dir.display());
            Ok(Start { run: dir, cfg, from: Some((step, dst)) })
        }
    }
}

fn new_run(cli: &Cli) -> Result<Start> {
    ensure!(!cli.inputs.is_empty(), "no training images: give image files, folders or a list (see --help)");
    let found = data::collect(&cli.inputs)?;
    let name = sanitize(&cli.name.clone().or_else(|| cli.trigger.clone()).unwrap_or_else(|| {
        let first = PathBuf::from(&cli.inputs[0]);
        let dir = if first.is_dir() { first } else { first.parent().map(Path::to_path_buf).unwrap_or_default() };
        dir.canonicalize().ok().and_then(|d| d.file_name().map(|f| f.to_string_lossy().into_owned())).unwrap_or_else(|| "lora".into())
    }));
    let run = cli.out.clone().unwrap_or_else(|| PathBuf::from(&name));
    ensure!(!run.join("run.json").exists(), "{} already holds a run: continue it with --resume {}, or choose another -o", run.display(), run.display());
    let resolution = cli.resolution.unwrap_or(1024);
    ensure!((256..=2048).contains(&resolution), "--resolution {resolution}: 256..2048");
    let captions = cli.captions.unwrap_or(Captions::Auto);
    let mut items = Vec::with_capacity(found.len());
    for (p, cap) in found {
        let (w, h) = data::dims(&p)?;
        let (bw, bh) = data::bucket(w, h, resolution * resolution);
        if w * h < resolution * resolution / 2 {
            crate::info!("  note: {} is {w}x{h}, small for {resolution}px training (trained at {bw}x{bh})", p.display());
        }
        let (caption, source) = match captions {
            Captions::None => (String::new(), "none"),
            _ => match cap {
                Some(c) => (c, "list"),
                None => match data::sidecar(&p) {
                    Some(c) => (c, "sidecar"),
                    None => (String::new(), "none"),
                },
            },
        };
        items.push(Item { path: data::plain_path(p.canonicalize().unwrap_or(p)), caption, short: None, caption_source: source.into(), width: w, height: h, bw, bh });
    }
    let batch = cli.batch.unwrap_or(1).max(1);
    let rank = cli.rank.unwrap_or(16);
    let model = match &cli.model {
        Some(m) => models::resolve_model(m)?,
        None => default_model()?,
    };
    let meta = match &cli.meta {
        Some(m) => {
            let v: serde_json::Value = serde_json::from_str(m).context("--meta: not JSON")?;
            ensure!(v.is_object(), "--meta must be a JSON object");
            Some(v)
        }
        None => None,
    };
    let save_every = cli.save_every.unwrap_or(250).max(1);
    let cfg = Config {
        loki_tune: 1,
        version: env!("CARGO_PKG_VERSION").into(),
        name,
        model,
        trigger: cli.trigger.clone().filter(|t| !t.trim().is_empty()),
        kind: cli.kind.unwrap_or(data::Kind::General),
        class: cli.class.clone().filter(|c| !c.trim().is_empty()),
        traits: cli.traits.clone().filter(|c| !c.trim().is_empty()),
        rank,
        alpha: cli.alpha.unwrap_or(rank as f32),
        lr: cli.lr.unwrap_or(1e-4),
        steps: cli.steps.unwrap_or_else(|| plan::default_steps(items.len(), batch, cli.kind == Some(data::Kind::Style))),
        batch,
        warmup: cli.warmup.unwrap_or(50),
        cosine: cli.cosine,
        resolution,
        timesteps: match cli.timesteps.unwrap_or(TimestepArg::Moss) {
            TimestepArg::Moss => Timesteps::Moss,
            TimestepArg::Uniform => Timesteps::Uniform,
            TimestepArg::LogitNormal => Timesteps::LogitNormal,
        },
        clean_share: cli.clean_share.unwrap_or(0.3).clamp(0.0, 0.99),
        caption_dropout: cli.caption_dropout.unwrap_or(0.0).clamp(0.0, 1.0),
        captions,
        flip: cli.flip,
        weight_decay: cli.weight_decay.unwrap_or(0.01),
        grad_clip: cli.grad_clip.unwrap_or(1.0),
        seed: cli.seed.unwrap_or(0),
        save_every,
        sample_every: cli.sample_every.unwrap_or(save_every),
        samples: Vec::new(),
        sample_steps: cli.sample_steps.unwrap_or(8).max(1),
        sample_size: match &cli.sample_size {
            Some(s) => parse_size(s)?,
            None => (resolution / 16 * 16, resolution / 16 * 16),
        },
        cache_dir: None,
        meta,
        items,
    };
    ensure!(cfg.rank % 8 == 0 && (8..=256).contains(&cfg.rank), "--rank must be a multiple of 8 between 8 and 256");
    Ok(Start { run, cfg, from: None })
}

/// Fill in loki-scribe captions where the run has none (with --captions auto) and the default preview prompts.
fn finish_captions(cli: &Cli, cfg: &mut Config) -> Result<()> {
    // a caption an older build stored as loki-scribe's raw JSON answer
    for it in cfg.items.iter_mut().filter(|it| it.caption_source == "scribe" && it.caption.contains("\"long\"")) {
        let (long, short) = data::parse_caption(&it.caption);
        if !long.contains("\"long\"") {
            crate::info!("  caption of {} repaired", it.path.display());
            it.caption = long;
            if it.short.is_none() {
                it.short = short;
            }
        }
    }
    if cfg.captions == Captions::Auto {
        let missing: Vec<PathBuf> = cfg.items.iter().filter(|it| it.caption.is_empty()).map(|it| it.path.clone()).collect();
        if !missing.is_empty() {
            if cfg.traits.is_none() && cli.caption_prompt.is_none() {
                if let Some(tp) = data::traits_prompt(cfg.kind) {
                    let all: Vec<PathBuf> = cfg.items.iter().map(|it| it.path.clone()).collect();
                    crate::info!("  captions: finding what the {} image(s) share", all.len());
                    let t = data::scribe_traits(&all, tp, cli.quiet)?;
                    crate::info!("  shared traits (left to the trigger): {t}");
                    cfg.traits = Some(t).filter(|t| !t.is_empty());
                }
            }
            crate::info!("  captions: describing {} image(s) with loki-scribe", missing.len());
            let prompt = cli.caption_prompt.clone().unwrap_or_else(|| data::caption_prompt(cfg.kind, cfg.trigger.as_deref(), cfg.class.as_deref(), cfg.traits.as_deref()));
            let caps = data::scribe(&missing, &prompt, cli.quiet)?;
            for it in cfg.items.iter_mut().filter(|it| it.caption.is_empty()) {
                if let Some((long, short)) = caps.get(&it.path) {
                    it.caption = long.clone();
                    it.short = short.clone();
                    it.caption_source = "scribe".into();
                }
            }
        }
    }
    if cfg.samples.is_empty() && !cli.no_samples {
        let first = cfg.items.first().map(|it| data::with_trigger(it.short.as_deref().unwrap_or(&it.caption), cfg.trigger.as_deref()));
        cfg.samples = data::default_samples(cfg.kind, cfg.trigger.as_deref(), cfg.class.as_deref(), first.as_deref());
    }
    Ok(())
}

fn print_plan(run: &Path, c: &Config, from: usize) {
    eprintln!("{PROG}: {} -> {}", c.name, run.display());
    eprintln!("  base model: {}", c.model.display());
    eprintln!("  kind: {:?}{}{}", c.kind, c.trigger.as_ref().map(|t| format!(", trigger {t:?}")).unwrap_or_default(), c.class.as_ref().map(|k| format!(" ({k})")).unwrap_or_default());
    if let Some(t) = &c.traits {
        eprintln!("  shared traits (left out of the captions): {t}");
    }
    for p in &c.samples {
        eprintln!("  preview: {p}");
    }
    eprintln!("  {} image(s) at ~{}px, rank {} alpha {}, lr {:.1e}{}, {} steps x {} image(s){}", c.items.len(), c.resolution, c.rank, c.alpha, c.lr, if c.cosine { " cosine" } else { "" }, c.steps, c.batch, if from > 0 { format!(", from step {from}") } else { String::new() });
    eprintln!("  noise levels: {:?}{}, caption dropout {}, flip {}", c.timesteps, if c.timesteps == Timesteps::Moss { format!(" (clean share {})", c.clean_share) } else { String::new() }, c.caption_dropout, c.flip);
    eprintln!("  checkpoints every {} steps; {} preview prompt(s) every {} steps", c.save_every, c.samples.len(), c.sample_every);
    for it in &c.items {
        let cap = data::with_trigger(&it.caption, c.trigger.as_deref());
        let short: String = cap.chars().take(100).collect();
        eprintln!("    {}x{}  {}  [{}] {}{}", it.bw, it.bh, it.path.display(), it.caption_source, short, if cap.chars().count() > 100 { "..." } else { "" });
    }
}

// ------------------------------------------------------------------------------------------------ previews
struct Previewer {
    vae: loki_dream::vae::Vae,
    turbo: Option<PathBuf>,
}

impl Previewer {
    fn render(&self, dev: &Arc<Device>, dit: &mut Dit, cache: &Cache, c: &Config, lora_file: Option<&Path>, step: usize, dir: &Path) -> Result<Vec<PathBuf>> {
        use loki_dream::pipeline::{sample, Inputs, Settings};
        std::fs::create_dir_all(dir)?;
        let mut specs = Vec::new();
        if let Some(t) = &self.turbo {
            specs.push(loki_dream::lora::LoraSpec { path: t.clone(), strength: 1.0 });
        }
        if let Some(f) = lora_file {
            specs.push(loki_dream::lora::LoraSpec { path: f.to_path_buf(), strength: 1.0 });
        }
        let quiet = loki_dream::log::quiet();
        loki_dream::log::set_quiet(true);
        let loaded = if specs.is_empty() { Ok(Vec::new()) } else { loki_dream::lora::load(dev, &specs, LAYERS) };
        loki_dream::log::set_quiet(quiet);
        dit.lora = loaded?;
        let (w, h) = c.sample_size;
        let (lh, lw) = (h / 16, w / 16);
        let turbo = self.turbo.is_some();
        let cfg = if turbo { 1.0 } else { 3.0 };
        let uncond = if turbo { None } else { Some(dit.prepare(&cache.load_text(dev, "")?, lh, lw)?) };
        let mut out = Vec::new();
        for (i, p) in c.samples.iter().enumerate() {
            let run = dit.prepare(&cache.load_text(dev, p)?, lh, lw)?;
            let s = Settings { steps: c.sample_steps, cfg, shift: 5.0, seed: 42 + i as u64, skip_layers: vec![22], strength: 1.0 };
            let lat = sample(dev, dit, &s, &Inputs { cond: &run, uncond: uncond.as_ref(), x0: None, mask: None, noise: None }, &|_, _| {})?;
            let rgb = self.vae.decode(&lat, lh, lw)?;
            let mut rec = genmeta::Record::new(PROG, env!("CARGO_PKG_VERSION"), "MOSS-Image");
            rec.set("prompt", p.clone()).set("seed", 42 + i as u64).set("steps", c.sample_steps).set("cfg", cfg as f64).set("turbo", turbo).set("dit", c.model.display().to_string()).set("training_step", step);
            if let Some(f) = lora_file {
                rec.set("loras", serde_json::json!([{"path": f.display().to_string(), "strength": 1.0}]));
            }
            let path = dir.join(format!("{}_{i}.png", ckpt_name(step)));
            loki_dream::image::Rgb8 { w, h, data: rgb }.save_png(&path, &[(genmeta::PNG_KEYWORD.to_string(), rec.to_json())])?;
            out.push(path);
        }
        dit.lora = Vec::new();
        Ok(out)
    }
}

// ------------------------------------------------------------------------------------------------ training
fn lora_meta(c: &Config, step: usize, loss: Option<f64>) -> BTreeMap<String, String> {
    let mut rec = genmeta::Record::new(PROG, env!("CARGO_PKG_VERSION"), "MOSS-Image LoRA");
    rec.set("base_model", c.model.file_name().map(|f| f.to_string_lossy().into_owned()))
        .set("name", c.name.clone())
        .set("trigger", c.trigger.clone())
        .set("rank", c.rank)
        .set("alpha", c.alpha as f64)
        .set("lr", c.lr as f64)
        .set("step", step)
        .set("steps", c.steps)
        .set("batch", c.batch)
        .set("resolution", c.resolution)
        .set("images", c.items.len())
        .set("loss", loss)
        .set("meta", c.meta.clone());
    let mut m = BTreeMap::new();
    m.insert("loki".into(), rec.to_json());
    // kohya-style keys other tools read
    m.insert("ss_network_dim".into(), c.rank.to_string());
    m.insert("ss_network_alpha".into(), c.alpha.to_string());
    m.insert("ss_base_model_version".into(), "moss_image".into());
    m.insert("ss_steps".into(), step.to_string());
    m.insert("ss_num_train_images".into(), c.items.len().to_string());
    m.insert("ss_learning_rate".into(), c.lr.to_string());
    m.insert("ss_resolution".into(), format!("({}, {})", c.resolution, c.resolution));
    if let Some(t) = &c.trigger {
        m.insert("ss_tag_frequency".into(), serde_json::json!({ "trigger": { t.clone(): c.items.len() } }).to_string());
    }
    m
}

struct Saved {
    lora: PathBuf,
}

fn save_checkpoint(dev: &Device, run: &Path, c: &Config, set: &LoraSet, step: usize, loss: Option<f64>) -> Result<Saved> {
    let dir = run.join("checkpoints").join(ckpt_name(step));
    std::fs::create_dir_all(&dir)?;
    let lora = dir.join(format!("{}-step{step:06}.safetensors", c.name));
    set.save_lora(dev, &lora, &lora_meta(c, step, loss))?;
    let mut meta = BTreeMap::new();
    meta.insert("loki_tune".into(), serde_json::json!({"step": step, "rank": c.rank, "alpha": c.alpha, "loss": loss}).to_string());
    set.save_state(dev, &dir.join("state.safetensors"), &meta)?;
    Ok(Saved { lora })
}

pub fn main() {
    let cli = Cli::parse();
    loki_dream::log::set_quiet(cli.quiet);
    match run(&cli) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("{PROG}: error: {e:#}");
            std::process::exit(1);
        }
    }
}

fn run(cli: &Cli) -> Result<i32> {
    let Start { run, mut cfg, from } = match &cli.resume {
        Some(at) => resume(cli, at)?,
        None => new_run(cli)?,
    };
    let start_step = from.as_ref().map(|f| f.0).unwrap_or(0);
    ensure!(start_step < cfg.steps || cli.dry_run, "the checkpoint is at step {start_step} of {}; give a larger --steps to continue", cfg.steps);
    std::fs::create_dir_all(&run).with_context(|| format!("creating {}", run.display()))?;
    finish_captions(cli, &mut cfg)?;
    write_config(&run, &cfg)?;
    let _ = std::fs::remove_file(run.join("STOP"));
    if cli.dry_run {
        print_plan(&run, &cfg, start_step);
        eprintln!("{PROG}: planned; review or edit the captions in {}, then start with: loki-tune --resume {}", run.join("run.json").display(), run.display());
        return Ok(0);
    }
    if !cli.quiet {
        print_plan(&run, &cfg, start_step);
    }
    ctrlc::set_handler(|| {
        if STOP.fetch_add(1, Ordering::SeqCst) == 0 {
            eprintln!("\n{PROG}: stopping after this step (checkpoint will be saved); Ctrl-C again to quit now");
        } else {
            std::process::exit(130);
        }
    })
    .ok();

    let t_all = std::time::Instant::now();
    let dev = crate::gpu::device()?;
    crate::info!("  gpu: {} MB free of {} MB", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let cache = match &cfg.cache_dir {
        Some(d) => {
            std::fs::create_dir_all(d)?;
            Cache { dir: d.clone() }
        }
        None => Cache::new(&run)?,
    };
    // ---- preprocessing
    let te_path = models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?;
    let vae_path = models::resolve(cli.vae.clone(), &models::VAE)?;
    cache.encode_texts(&dev, &te_path, &cfg.texts(), cli.quiet)?;
    cache.encode_images(&dev, &vae_path, &cfg.items, cfg.flip, cli.quiet)?;
    dev.trim_pool();

    // ---- model, LoRA, optimizer state
    let t0 = std::time::Instant::now();
    let mut dit = Dit::load(dev.clone(), &cfg.model, None).with_context(|| format!("loading {}", cfg.model.display()))?;
    crate::info!("  base model: {} ({}) loaded in {:.1}s", cfg.model.display(), if dit.int8 { "int8" } else { "bf16" }, t0.elapsed().as_secs_f64());
    let mut set = match &from {
        Some((_, ck)) => LoraSet::load_state(&dev, &ck.join("state.safetensors"), cfg.rank, cfg.alpha)?,
        None => LoraSet::new(&dev, cfg.rank, cfg.alpha, cfg.seed)?,
    };
    crate::info!("  lora: rank {} alpha {}, {:.1}M parameters", cfg.rank, cfg.alpha, set.n_params as f64 / 1e6);
    let trainer = Trainer::new(&dev, &dit)?;
    let previews = if cfg.samples.is_empty() {
        None
    } else {
        let turbo = if cfg.sample_steps <= 8 { Some(models::resolve(None, &models::TURBO_LORA)?) } else { None };
        Some(Previewer { vae: loki_dream::vae::Vae::load(dev.clone(), &vae_path)?, turbo })
    };
    let mut samples: Vec<PathBuf> = Vec::new();
    let mut ckpts: Vec<PathBuf> = Vec::new();
    if let (Some(pv), 0) = (&previews, start_step) {
        let t = std::time::Instant::now();
        samples.extend(pv.render(&dev, &mut dit, &cache, &cfg, None, 0, &run.join("samples"))?);
        crate::info!("  previews without the LoRA: {:.1}s", t.elapsed().as_secs_f64());
    }

    // ---- the loop
    let n = cfg.items.len();
    let mut log = std::fs::OpenOptions::new().create(true).append(true).open(run.join("log.jsonl"))?;
    let empty = cache.load_text(&dev, "")?;
    let mut ema: Option<f64> = None;
    let mut last_saved = start_step;
    let mut step = start_step;
    let t_loop = std::time::Instant::now();
    let mut stopped = false;
    while step < cfg.steps {
        let ts = std::time::Instant::now();
        let mut loss_sum = 0.0f64;
        for j in 0..cfg.batch {
            let micro = step * cfg.batch + j;
            let idx = plan::item(cfg.seed, micro, n);
            let it = &cfg.items[idx];
            let mut rng = plan::micro_rng(cfg.seed, micro);
            let sigma = plan::sigma(&mut rng, cfg.timesteps, cfg.clean_share);
            let drop = rng.uniform() < cfg.caption_dropout;
            let short = it.short.is_some() && rng.uniform() < 0.5;
            let flipped = cfg.flip && rng.uniform() < 0.5;
            let x0 = cache.load_latent(&dev, it, flipped)?;
            let noise: Vec<f32> = (0..x0.numel()).map(|_| rng.normal() as f32).collect();
            let noise = Tensor::from_f32(&dev, &noise, &[x0.numel() / LATENT, LATENT])?;
            let text = if short { it.short.as_deref().unwrap() } else { &it.caption };
            let feats = if drop { empty.clone() } else { cache.load_text(&dev, &data::with_trigger(text, cfg.trigger.as_deref()))? };
            let r = dit.prepare(&feats, it.bh / 16, it.bw / 16)?;
            let out = trainer.step(&dev, &dit, &set, &r, &x0, &noise, sigma).with_context(|| format!("training on {}", it.path.display()))?;
            ensure!(out.loss.is_finite(), "the loss is {} at step {} ({}); lower --lr", out.loss, step + 1, it.path.display());
            loss_sum += out.loss as f64;
        }
        set.mask_grads(&dev)?;
        let gnorm = set.grad_norm(&dev)? / cfg.batch as f64;
        let clip = if cfg.grad_clip > 0.0 && gnorm > cfg.grad_clip as f64 { cfg.grad_clip as f64 / gnorm } else { 1.0 };
        let lr = plan::lr(cfg.lr, step, cfg.steps, cfg.warmup, cfg.cosine);
        set.step(&dev, lr, cfg.weight_decay, step + 1, (clip / cfg.batch as f64) as f32)?;
        dev.sync()?;
        step += 1;
        let loss = loss_sum / cfg.batch as f64;
        ema = Some(match ema {
            Some(e) => e * 0.95 + loss * 0.05,
            None => loss,
        });
        let secs = ts.elapsed().as_secs_f64();
        let done = (step - start_step) as f64;
        let rate = t_loop.elapsed().as_secs_f64() / done;
        crate::info!("step {step}/{} loss {loss:.4} avg {:.4} lr {lr:.1e} grad {gnorm:.3} {secs:.2}s/step eta {}", cfg.steps, ema.unwrap(), hms(rate * (cfg.steps - step) as f64));
        writeln!(log, "{}", serde_json::json!({"step": step, "loss": loss, "avg": ema, "lr": lr, "grad_norm": gnorm, "secs": secs}))?;
        // Ctrl-C, or a STOP file in the run folder (how a supervisor such as the media server asks for a checkpoint)
        let stop_file = run.join("STOP");
        let stop_now = STOP.load(Ordering::SeqCst) > 0 || stop_file.exists();
        if stop_now {
            let _ = std::fs::remove_file(&stop_file);
        }
        let save = step % cfg.save_every == 0 || step == cfg.steps || stop_now;
        if save {
            let s = save_checkpoint(&dev, &run, &cfg, &set, step, ema)?;
            crate::info!("  checkpoint {}", s.lora.display());
            last_saved = step;
            ckpts.push(s.lora.clone());
            if let Some(pv) = &previews {
                if !stop_now && cfg.sample_every > 0 && (step % cfg.sample_every == 0 || step == cfg.steps) {
                    let t = std::time::Instant::now();
                    samples.extend(pv.render(&dev, &mut dit, &cache, &cfg, Some(&s.lora), step, &run.join("samples"))?);
                    crate::info!("  previews: {:.1}s", t.elapsed().as_secs_f64());
                }
            }
        }
        if stop_now {
            stopped = true;
            break;
        }
    }
    let _ = last_saved;
    // ---- result
    let last = ckpts.last().cloned().or_else(|| checkpoints(&run).last().map(|(s, d)| d.join(format!("{}-step{s:06}.safetensors", cfg.name))));
    let result = if !stopped && step >= cfg.steps {
        let final_path = run.join(format!("{}.safetensors", cfg.name));
        if let Some(l) = &last {
            std::fs::copy(l, &final_path).with_context(|| format!("writing {}", final_path.display()))?;
        }
        if !cli.keep_cache && cfg.cache_dir.is_none() {
            let _ = std::fs::remove_dir_all(&cache.dir);
        }
        crate::info!("{PROG}: done, {} steps in {}", cfg.steps - start_step, hms(t_all.elapsed().as_secs_f64()));
        Some(final_path)
    } else {
        crate::info!("{PROG}: stopped at step {step}; continue with: loki-tune --resume {}", run.display());
        last
    };
    if cli.json {
        let v = serde_json::json!({"lora": result, "run": run, "steps": step, "stopped": stopped, "checkpoints": ckpts, "samples": samples, "loss": ema});
        println!("{v}");
    } else if let Some(p) = &result {
        println!("{}", p.display());
    }
    Ok(if stopped { 130 } else { 0 })
}
