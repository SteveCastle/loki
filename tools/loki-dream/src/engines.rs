//! The Krea 2, FLUX.2 klein and Anima runs: each architecture's text conditioning (Qwen3 text encoder, templates,
//! hidden-state taps, Anima's T5 query tokens), its VAE, its default settings and schedule, then the shared flow sampler.
use crate::anima::Anima;
use crate::arch::Arch;
use crate::cli::{self, Cli, Plan};
use crate::cuda::Device;
use crate::flow::{self, Shift};
use crate::flux2::Flux2;
use crate::genmeta;
use crate::image::Rgb8;
use crate::krea2::Krea2;
use crate::model::{self, Model};
use crate::models::{self, Spec};
use crate::qwen3::{Prompt, Qwen3};
use crate::qvae::QVae;
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::tokenizer::{self, Tokenizer};
use crate::vae::Vae;
use anyhow::{bail, ensure, Context, Result};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

// ---- Krea 2's prompt layout (pipeline_krea2.py)
const KREA2_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const KREA2_SUFFIX: &str = "<|im_end|>\n<|im_start|>assistant\n";
const KREA2_DROP: usize = 34;
const KREA2_MAX: usize = 512;

/// FLUX.2 klein's chat template (`apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`).
pub fn klein_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Whether a checkpoint is a step-distilled variant, by its file name. Krea 2: distilled unless named raw / base
/// (Turbo is the released default and community fine-tunes build on it); FLUX.2 klein: distilled unless named base;
/// Anima: base unless named turbo / distilled.
pub fn is_distilled(a: Arch, path: &Path) -> bool {
    let n = path.file_name().map(|f| f.to_string_lossy().to_lowercase()).unwrap_or_default();
    match a {
        Arch::Moss => false,
        Arch::Krea2 => !(n.contains("raw") || n.contains("base")),
        Arch::Anima => ["turbo", "distill", "lightning"].iter().any(|k| n.contains(k)),
        Arch::Flux2 => !n.contains("base"),
        Arch::Sdxl => crate::sdxl::is_distilled(path),
        // a 6-step PDD model: its output heads are distilled on its own sigma grid
        Arch::H3Image => true,
    }
}

/// (steps, cfg) defaults.
pub fn defaults(a: Arch, distilled: bool) -> (usize, f32) {
    match (a, distilled) {
        (Arch::Krea2, false) => (28, 5.5),
        (Arch::Krea2, true) => (8, 1.0),
        (Arch::Flux2, false) => (50, 4.0),
        (Arch::Flux2, true) => (4, 1.0),
        (Arch::Anima, false) => (30, 4.0),
        (Arch::Anima, true) => (10, 1.0),
        (Arch::Sdxl, false) => (30, 6.0),
        (Arch::Sdxl, true) => (8, 1.0),
        (Arch::Moss, _) => (50, 3.0),
        (Arch::H3Image, _) => (loki_reshoot::h3image::DEFAULT_STEPS, 1.0),
    }
}

/// The text(s) the encoders get, for --show-prompt.
pub fn show_prompt(a: Arch, p: &str) -> Vec<String> {
    match a {
        Arch::Krea2 => vec![format!("{KREA2_PREFIX}{p}{KREA2_SUFFIX}")],
        Arch::Flux2 => vec![klein_template(p)],
        // Qwen3-0.6B and the adapter's T5 query tokens both read the raw text
        Arch::Anima => vec![p.to_string()],
        Arch::Moss => vec![crate::text_encoder::template(p)],
        // what CLIP reads, by weight: each segment on its own line with its weight
        Arch::Sdxl => vec![crate::clip::weighted_segments(p).into_iter().map(|(t, w)| if w == 1.0 { t } else { format!("[{t}]x{w}") }).collect()],
        // no chat template: the pictures' labelled vision blocks, then the prompt as given
        Arch::H3Image => vec![loki_reshoot::h3image::show_prompt(p, 0)],
    }
}

pub(crate) fn te_spec(a: Arch, dit: &Path) -> Result<&'static Spec> {
    Ok(match a {
        Arch::Krea2 => &models::KREA2_TE,
        Arch::Anima => &models::ANIMA_TE,
        Arch::Flux2 => {
            let st = SafeTensors::open(dit)?;
            let p = crate::arch::prefix_of(&st, "txt_in.weight").context("FLUX.2: no txt_in")?;
            match st.info(&format!("{p}txt_in.weight"))?.shape[1] {
                7680 => &models::KLEIN4B_TE,
                12288 => &models::KLEIN9B_TE,
                k => bail!("FLUX.2 model with a {k}-wide text input: only klein 4B (Qwen3-4B) and 9B (Qwen3-8B) are supported"),
            }
        }
        Arch::Moss => &models::TEXT_ENCODER,
        Arch::Sdxl => bail!("an SDXL checkpoint carries its own text encoders"),
        Arch::H3Image => &models::H3_TE,
    })
}

/// The VAE an architecture downloads (None: SDXL's is inside the checkpoint).
pub(crate) fn vae_spec(a: Arch) -> Option<&'static Spec> {
    match a {
        Arch::Krea2 | Arch::Anima => Some(&models::QWEN_IMAGE_VAE),
        Arch::Flux2 => Some(&models::FLUX2_VAE),
        Arch::Moss => Some(&models::VAE),
        Arch::Sdxl => None,
        Arch::H3Image => Some(&models::H3_IMAGE_VAE),
    }
}

/// int8 or bf16 for a diffusion model: --precision, else int8 for quantized checkpoints and for float ones too large to
/// sit in bf16 next to the activations, else bf16.
fn use_int8(cli: &Cli, dit: &Path, dev: &Device) -> Result<bool> {
    match cli.precision.as_deref() {
        Some("int8") => return Ok(true),
        Some("bf16") => return Ok(false),
        Some(o) => bail!("--precision must be int8 or bf16, not {o}"),
        None => {}
    }
    let st = SafeTensors::open(dit)?;
    let quantized = st.tensors.values().any(|t| t.dtype == "F8_E4M3" || t.dtype == "I8");
    if quantized {
        return Ok(true);
    }
    let bytes = std::fs::metadata(dit).map(|m| m.len() as f64).unwrap_or(0.0);
    let per_param = if st.tensors.values().any(|t| t.dtype == "F32") && !st.tensors.values().any(|t| t.dtype == "BF16" || t.dtype == "F16") { 4.0 } else { 2.0 };
    Ok(bytes / per_param * 2.0 > dev.total_mem as f64 * 0.45)
}

/// A loaded diffusion model of one of the flow architectures.
enum Engine {
    Krea2(Krea2),
    Flux2(Flux2),
    Anima(Anima),
}
impl Engine {
    fn int8(&self) -> bool {
        match self {
            Engine::Krea2(m) => m.int8,
            Engine::Flux2(m) => m.int8,
            Engine::Anima(m) => m.int8,
        }
    }
}

enum AnyVae {
    Flux2(Vae),
    Qwen(QVae),
}
impl AnyVae {
    fn encode(&self, rgb: &[f32], h: usize, w: usize) -> Result<Tensor> {
        match self {
            AnyVae::Flux2(v) => v.encode(rgb, h, w),
            AnyVae::Qwen(v) => v.encode(rgb, h, w),
        }
    }
    fn decode(&self, t: &Tensor, h: usize, w: usize) -> Result<Vec<u8>> {
        match self {
            AnyVae::Flux2(v) => v.decode(t, h, w),
            AnyVae::Qwen(v) => v.decode(t, h, w),
        }
    }
}

fn load_engine(dev: &Arc<Device>, a: Arch, dit: &Path, int8: bool) -> Result<Rc<Engine>> {
    crate::worker::load(&format!("dit {} {} {}", a.id(), dit.display(), if int8 { "int8" } else { "bf16" }), dit, || {
        Ok(match a {
            Arch::Krea2 => Engine::Krea2(Krea2::load(dev.clone(), dit, int8)?),
            Arch::Flux2 => Engine::Flux2(Flux2::load(dev.clone(), dit, int8)?),
            Arch::Anima => Engine::Anima(Anima::load(dev.clone(), dit, int8)?),
            Arch::Moss | Arch::Sdxl | Arch::H3Image => bail!("{} runs through its own path", a.name()),
        })
    })
}

fn load_any_vae(dev: &Arc<Device>, a: Arch, p: &Path) -> Result<Rc<AnyVae>> {
    crate::worker::load(&format!("vae {} {}", a.id(), p.display()), p, || {
        Ok(match a {
            Arch::Flux2 => AnyVae::Flux2(Vae::load(dev.clone(), p)?),
            _ => AnyVae::Qwen(QVae::load(dev.clone(), p)?),
        })
    })
}

fn init_kernels(dev: &Device) -> Result<()> {
    crate::ops::gemm_init(dev)?;
    crate::ops::attn_init(dev)?;
    Vae::init_kernels(dev)
}

/// Worker warm-up: fetch the architecture's files and load its DiT and VAE.
pub(crate) fn warm(cli: &Cli, a: Arch, dit: &Path) -> Result<()> {
    if a == Arch::Sdxl {
        return crate::sdxl::warm(cli, dit);
    }
    if a == Arch::H3Image {
        return crate::h3::warm(cli, dit);
    }
    models::resolve(cli.text_encoder.clone(), te_spec(a, dit)?)?;
    let vae = models::resolve(cli.vae.clone(), vae_spec(a).context("VAE")?)?;
    let dev = crate::worker::device()?;
    init_kernels(&dev)?;
    load_any_vae(&dev, a, &vae)?;
    load_engine(&dev, a, dit, use_int8(cli, dit, &dev)?)?;
    Ok(())
}

/// Conditioning of one text in the form its DiT takes.
pub enum Cond {
    /// Krea 2: [T * 12, 2560] tapped states, token-major
    Krea2(Tensor),
    /// FLUX.2: [512, 3 * hidden]
    Flux2(Tensor),
    /// Anima: Qwen3 last hidden state [S, 1024] (masked rows zeroed), valid rows, T5 ids
    Anima(Tensor, usize, Vec<u32>),
}

/// Run the text encoder over every text (prompts then the negative) and shape its output per architecture.
pub fn encode_texts(dev: &Arc<Device>, a: Arch, te_path: &Path, texts: &[String], quiet: bool) -> Result<Vec<Cond>> {
    let tok = Tokenizer::qwen3()?;
    let te = Qwen3::open(dev.clone(), te_path)?;
    let c = te.cfg.clone();
    let mut prompts = Vec::with_capacity(texts.len());
    let mut t5_ids = Vec::new();
    match a {
        Arch::Krea2 => {
            ensure!(c.hidden == 2560 && c.layers >= 35, "{}: Krea 2 needs the Qwen3-VL-4B text encoder (hidden 2560, 36 layers), got hidden {} / {} layers", te_path.display(), c.hidden, c.layers);
            let prefix = tok.encode(KREA2_PREFIX);
            ensure!(prefix.len() == KREA2_DROP, "Krea 2 prompt prefix tokenizes to {} tokens, expected {KREA2_DROP}", prefix.len());
            let suffix = tok.encode(KREA2_SUFFIX);
            for t in texts {
                let mut ids = tok.encode(&format!("{KREA2_PREFIX}{t}"));
                if ids.len() > KREA2_MAX + KREA2_DROP - suffix.len() {
                    crate::info!("warning: prompt truncated to {} tokens (Krea 2's limit)", KREA2_MAX - suffix.len());
                    ids.truncate(KREA2_MAX + KREA2_DROP - suffix.len());
                }
                ids.extend(&suffix);
                let n = ids.len();
                prompts.push(Prompt { tokens: ids, valid: n });
            }
        }
        Arch::Flux2 => {
            for t in texts {
                let mut ids = tok.encode(&klein_template(t));
                if ids.len() > flow_max_text() {
                    crate::info!("warning: prompt truncated to {} tokens (FLUX.2's limit)", flow_max_text());
                    ids.truncate(flow_max_text());
                }
                let valid = ids.len();
                ids.resize(flow_max_text(), tokenizer::qwen3::ENDOFTEXT);
                prompts.push(Prompt { tokens: ids, valid });
            }
        }
        Arch::Anima => {
            ensure!(c.hidden == 1024, "{}: Anima needs the Qwen3-0.6B text encoder (hidden 1024), got hidden {}", te_path.display(), c.hidden);
            let t5 = crate::t5tok::T5Tokenizer::new()?;
            for t in texts {
                let mut ids = tok.encode(t);
                ids.truncate(512);
                let valid = ids.len();
                if ids.is_empty() {
                    ids.push(0);
                }
                prompts.push(Prompt { tokens: ids, valid });
                let mut q = t5.encode(t);
                if q.len() > 512 {
                    q.truncate(511);
                    q.push(crate::t5tok::EOS);
                }
                t5_ids.push(q);
            }
        }
        Arch::Moss | Arch::Sdxl | Arch::H3Image => unreachable!(),
    }
    let taps: Vec<usize> = match a {
        Arch::Krea2 => crate::krea2::TAPS.to_vec(),
        Arch::Flux2 => crate::flux2::TAPS.to_vec(),
        _ => Vec::new(),
    };
    let show = !quiet && std::io::stderr().is_terminal();
    let bar = if show { indicatif::ProgressBar::new(c.layers as u64) } else { indicatif::ProgressBar::hidden() };
    bar.set_style(indicatif::ProgressStyle::with_template("  text encoder {bar:30} {pos}/{len} layers").unwrap());
    let outs = te.encode(&prompts, &taps, a == Arch::Anima, &|l| bar.set_position(l as u64)).context("text encoder")?;
    bar.finish_and_clear();
    te.close();
    let mut conds = Vec::with_capacity(outs.len());
    for (i, (o, p)) in outs.into_iter().zip(&prompts).enumerate() {
        conds.push(match a {
            Arch::Krea2 => {
                let t = p.tokens.len() - KREA2_DROP;
                let l = taps.len();
                let x = Tensor::zeros(dev, DType::BF16, &[t, l * c.hidden])?;
                for (k, tap) in o.taps.iter().enumerate() {
                    model::put_cols(dev, &x, k * c.hidden, &tap.rows(KREA2_DROP, t))?;
                }
                Cond::Krea2(x.reshape(&[t * l, c.hidden]))
            }
            Arch::Flux2 => {
                let t = p.tokens.len();
                let x = Tensor::zeros(dev, DType::BF16, &[t, taps.len() * c.hidden])?;
                for (k, tap) in o.taps.iter().enumerate() {
                    model::put_cols(dev, &x, k * c.hidden, tap)?;
                }
                Cond::Flux2(x)
            }
            _ => {
                let last = o.last.context("last hidden state")?;
                if p.valid < p.tokens.len() {
                    let s: Vec<f32> = (0..p.tokens.len()).map(|r| if r < p.valid { 1.0 } else { 0.0 }).collect();
                    crate::ops::row_scale(dev, &last, &Tensor::from_f32(dev, &s, &[s.len()])?)?;
                }
                Cond::Anima(last, p.valid, std::mem::take(&mut t5_ids[i]))
            }
        });
    }
    Ok(conds)
}

fn flow_max_text() -> usize {
    crate::flux2::MAX_TEXT
}

/// Everything about a run's sampling the record and the sampler need.
struct Sched {
    sigmas: Vec<f32>,
    shift: Shift,
}

fn schedule(a: Arch, cli: &Cli, steps: usize, distilled: bool, tokens: usize) -> Sched {
    let shift = match (cli.shift, a) {
        (Some(s), _) => Shift::Static(s),
        (None, Arch::Krea2) => Shift::Dynamic(if distilled { 1.15 } else { flow::krea2_mu(tokens) }),
        (None, Arch::Flux2) => Shift::Dynamic(flow::flux2_mu(tokens, steps)),
        (None, _) => Shift::Static(3.0),
    };
    Sched { sigmas: flow::sigmas(steps, shift), shift }
}

fn sample_with<M: Model>(dev: &Device, m: &M, sig: &[f32], cfg: f32, strength: f32, cond: &M::Run, uncond: Option<&M::Run>, x0: Option<&Tensor>, mask: Option<&Tensor>, noise: Vec<f32>, cli: &Cli, steps: usize) -> Result<Tensor> {
    let (bar, show_bar) = cli::step_progress(cli, steps);
    let inp = flow::Inputs { cond, uncond, x0, mask, noise };
    let lat = flow::sample(dev, m, sig, cfg, strength, &inp, &|n, total| {
        bar.set_length(total as u64);
        bar.set_position(n as u64);
        if !show_bar {
            crate::info!("step {n}/{total}");
        }
    })?;
    bar.finish_and_clear();
    Ok(lat)
}

/// A prepared prompt for one engine.
enum Prepared {
    Krea2(crate::krea2::Run),
    Flux2(crate::flux2::Run),
    Anima(crate::anima::Run),
}

fn prepare(e: &Engine, c: &Cond, lh: usize, lw: usize) -> Result<Prepared> {
    Ok(match (e, c) {
        (Engine::Krea2(m), Cond::Krea2(x)) => Prepared::Krea2(m.prepare(x, lh, lw)?),
        (Engine::Flux2(m), Cond::Flux2(x)) => Prepared::Flux2(m.prepare(x, lh, lw)?),
        (Engine::Anima(m), Cond::Anima(src, valid, t5)) => {
            let ctx = m.adapter(src, *valid, t5)?;
            Prepared::Anima(m.prepare(&ctx, lh, lw)?)
        }
        _ => bail!("conditioning does not match the model"),
    })
}

pub(crate) fn run(cli: &Cli, plan: &Plan, distilled: bool) -> Result<()> {
    let a = plan.arch;
    let prog = plan.prog;
    let ((gw, gh), (ow, oh)) = (plan.gen, plan.out);
    let (steps, cfg) = (plan.steps, plan.cfg);
    let te_path = models::resolve(cli.text_encoder.clone(), te_spec(a, &plan.dit)?)?;
    let vae_path = models::resolve(cli.vae.clone(), vae_spec(a).context("VAE")?)?;
    crate::info!("{prog}: {} {}{}, {} image(s), {}x{}{}, {} steps, CFG {}", a.name(), cli::file_stem(&plan.dit), if distilled { " (distilled)" } else { "" }, plan.jobs.len(), ow, oh, if (gw, gh) != (ow, oh) { format!(" (generated at {gw}x{gh})") } else { String::new() }, steps, cfg);
    crate::info!("  dit: {}\n  text encoder: {}\n  vae: {}", plan.dit.display(), te_path.display(), vae_path.display());
    let t_dev = std::time::Instant::now();
    let dev = crate::worker::device()?;
    crate::info!("  gpu: {} MB free of {} MB (cuda init {:.1}s)", dev.free_mem()? >> 20, dev.total_mem >> 20, t_dev.elapsed().as_secs_f64());
    init_kernels(&dev)?;
    let t_all = std::time::Instant::now();

    // ---- phase 1: text encoder over the prompts and the negative
    let mut texts: Vec<String> = plan.prompts.clone();
    if let Some(n) = &plan.negative {
        texts.push(n.clone());
    }
    let t0 = std::time::Instant::now();
    let conds = encode_texts(&dev, a, &te_path, &texts, cli.quiet)?;
    crate::info!("  text encoder: {} text(s), {:.1}s", texts.len(), t0.elapsed().as_secs_f64());

    // ---- phase 2: VAE; encode the init image
    let vae = load_any_vae(&dev, a, &vae_path)?;
    let (lh, lw) = (gh / 16, gw / 16);
    let x0 = match &plan.init_gen {
        Some(img) => Some(vae.encode(&img.to_f32(), gh, gw).context("vae encode")?),
        None => None,
    };
    let mask_lat = match &plan.mask_gen {
        Some(m) => Some(Tensor::from_f32(&dev, &m.max_down(16).data, &[lh * lw])?),
        None => None,
    };

    // ---- phase 3: DiT; sample, decode, write each image as soon as it is ready
    let int8 = use_int8(cli, &plan.dit, &dev)?;
    let engine = load_engine(&dev, a, &plan.dit, int8)?;
    if let Engine::Flux2(f) = &*engine {
        let te = Qwen3::open(dev.clone(), &te_path)?;
        ensure!(te.cfg.hidden == f.te_hidden(), "{}: this FLUX.2 model needs a text encoder with hidden size {} (klein 4B: Qwen3-4B, 9B: Qwen3-8B), got {}", te_path.display(), f.te_hidden(), te.cfg.hidden);
        te.close();
    }
    let sched = schedule(a, cli, steps, distilled, lh * lw);
    let strength = if cli.init.is_some() { cli.strength } else { 1.0 };
    let uncond = match &plan.negative {
        Some(_) => Some(prepare(&engine, &conds[plan.prompts.len()], lh, lw)?),
        None => None,
    };
    let mut runs: HashMap<usize, Prepared> = HashMap::new();
    let mut results: Vec<serde_json::Value> = Vec::new();
    for (ji, job) in plan.jobs.iter().enumerate() {
        if !runs.contains_key(&job.prompt) {
            runs.insert(job.prompt, prepare(&engine, &conds[job.prompt], lh, lw)?);
        }
        let run = &runs[&job.prompt];
        crate::info!("image {}/{}: seed {} {}x{}", ji + 1, plan.jobs.len(), job.seed, ow, oh);
        let t0 = std::time::Instant::now();
        let lat = match (&*engine, run, uncond.as_ref()) {
            (Engine::Krea2(m), Prepared::Krea2(c), u) => {
                let u = match u {
                    Some(Prepared::Krea2(u)) => Some(u),
                    _ => None,
                };
                sample_with(&dev, m, &sched.sigmas, cfg, strength, c, u, x0.as_ref(), mask_lat.as_ref(), flow::noise_patched16(job.seed, lh, lw), cli, steps)?
            }
            (Engine::Flux2(m), Prepared::Flux2(c), u) => {
                let u = match u {
                    Some(Prepared::Flux2(u)) => Some(u),
                    _ => None,
                };
                sample_with(&dev, m, &sched.sigmas, cfg, strength, c, u, x0.as_ref(), mask_lat.as_ref(), crate::pipeline::noise_tokens(job.seed, lh, lw), cli, steps)?
            }
            (Engine::Anima(m), Prepared::Anima(c), u) => {
                let u = match u {
                    Some(Prepared::Anima(u)) => Some(u),
                    _ => None,
                };
                sample_with(&dev, m, &sched.sigmas, cfg, strength, c, u, x0.as_ref(), mask_lat.as_ref(), flow::noise_patched16(job.seed, lh, lw), cli, steps)?
            }
            _ => bail!("internal: prepared prompt does not match the model"),
        };
        let t_sample = t0.elapsed().as_secs_f64();
        if let Engine::Krea2(m) = &*engine {
            m.prof.report();
        }
        let t1 = std::time::Instant::now();
        let rgb = vae.decode(&lat, lh, lw).context("vae decode")?;
        drop(lat);
        crate::info!("  sampled in {:.1}s, decoded in {:.1}s", t_sample, t1.elapsed().as_secs_f64());
        let mut rec = genmeta::Record::new(prog, env!("CARGO_PKG_VERSION"), &cli::file_stem(&plan.dit));
        rec.set("base_model", a.name())
            .set("architecture", a.id())
            .set("prompt", plan.prompts[job.prompt].clone())
            .set("negative", plan.negative.clone())
            .set("seed", job.seed)
            .set("steps", steps)
            .set("cfg", cfg as f64)
            .set("shift", match sched.shift {
                Shift::Static(s) => serde_json::json!(s as f64),
                Shift::Dynamic(mu) => serde_json::json!({"mu": mu as f64}),
            })
            .set("turbo", if distilled { Some(true) } else { None })
            .set("sampler", "euler")
            .set("precision", if engine.int8() { "int8" } else { "bf16" })
            .set("text_encoder", te_path.file_name().map(|f| f.to_string_lossy().into_owned()));
        cli::deliver(cli, plan, job, Rgb8 { w: gw, h: gh, data: rgb }, &mut rec, &mut results)?;
    }
    cli::finish_json(cli, plan, results, distilled);
    drop(engine);
    dev.sync()?;
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}

/// The text encoder a model would use (for diagnostics and tests).
pub fn text_encoder_for(a: Arch, dit: &Path) -> Result<PathBuf> {
    models::resolve(None, te_spec(a, dit)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distilled_by_name() {
        assert!(is_distilled(Arch::Krea2, Path::new("krea2_turbo_fp8_scaled.safetensors")));
        assert!(is_distilled(Arch::Krea2, Path::new("some_krea2_finetune.safetensors")));
        assert!(!is_distilled(Arch::Krea2, Path::new("krea2_raw_bf16.safetensors")));
        assert!(is_distilled(Arch::Flux2, Path::new("flux-2-klein-9b-fp8.safetensors")));
        assert!(!is_distilled(Arch::Flux2, Path::new("flux-2-klein-base-4b-fp8.safetensors")));
        assert!(is_distilled(Arch::Anima, Path::new("anima-turbo-v1.1.safetensors")));
        assert!(!is_distilled(Arch::Anima, Path::new("anima_baseV10.safetensors")));
    }

    #[test]
    fn krea2_prefix_is_34_tokens() {
        let t = Tokenizer::qwen3().unwrap();
        assert_eq!(t.encode(KREA2_PREFIX).len(), KREA2_DROP);
        assert_eq!(t.encode(KREA2_SUFFIX).len(), 5);
    }
}
