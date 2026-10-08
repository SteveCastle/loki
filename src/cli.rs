//! Command-line front end: a generic Qwen Image 2.1 image-edit tool with named presets (`--preset 4kify`, ...).
//!
//! loki- toolkit conventions (shared with loki-reshoot): images in from files/globs/directories or stdin (`-`), the result out as a file (its path is
//! printed on stdout, or one JSON object per job with `--json`) or as PNG bytes on stdout (`-o -`); progress and
//! diagnostics go to stderr only and `--quiet` silences them; a non-zero exit status signals failure.
use crate::cuda::Device;
use crate::dit::Dit;
use crate::image::Rgb8;
use crate::models;
use crate::pipeline::{Encoded, Pipeline, Settings};
use crate::presets::{self, Preset};
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;
use anyhow::{bail, ensure, Context, Result};
use clap::{CommandFactory, FromArgMatches, Parser};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

const HELP: &str = "\
OUTPUT
  A written file's path is printed on stdout (one line per input; `--json` prints one JSON object per input instead);
  `-o -` writes the PNG itself to stdout. Everything else (progress, timings) goes to stderr; `-q` silences it.

PRESETS (--preset NAME; see --list-presets)
  A preset bundles a prompt, an output size and reference sizing. Without one, PROG is a plain image editor:
  give --prompt; the output keeps the input's size and ratio unless a size flag says otherwise.
  Explicit flags always win over the preset: --size over its size, --prompt replaces its prompt (--append extends it).

SIZE (default: same as the input; a preset's own size wins over that, an explicit flag wins over the preset)
  --size same | WxH    --scale 2 (2x each side, ratio kept)    --width PX / --height PX / --long-edge PX (ratio kept)
  --megapixels MP      --upscale [N] (= --preset upscale --scale N)    --snap (keep the model's multiple-of-16 size)
  Sizes that are not multiples of 16 are generated at the nearest multiple and Lanczos-resampled to exactly what you asked.
  The model both upscales (faithful super-resolution) and changes content (any --prompt): they combine, e.g. --scale 2 -p \"add snow\".

REFERENCES
  The input is <image1>; every --ref FILE adds <image2>, <image3>, ... to each job (multi-image edits and compositions).
  The prompt can mention them: \"put the jacket from <image2> on the person in <image1>\".

EXAMPLES
  PROG -p \"make it night, with rain\" photo.jpg                     # edit at the input's size -> photo_edit.png
  PROG -p \"put the jacket of <image2> on <image1>\" -r jacket.png me.png
  PROG --upscale 2 small.jpg                                         # faithful 2x super-resolution, ratio kept
  PROG --scale 1.5 -p \"replace the sky with a sunset\" photo.jpg      # content edit + 1.5x size
  PROG --preset 4kify photo.jpg                                       # restore + outpaint to a 4K wallpaper
  PROG --preset restore -o - old.jpg | PROG -p \"colorize\" - -o - | PROG --preset 4kify - -o wall.png
  PROG --preset 4kify -q -o walls/ shots/*.jpg | xargs -n1 echo       # batch: paths of the results on stdout
";

#[derive(Parser, Debug)]
#[command(version, about = "loki-retouch: AI image editing on a standalone CUDA engine (Qwen Image 2.1). Edit content, upscale, restore, composite, outpaint; wallpapers via --preset 4kify. Part of the loki- toolkit (see also loki-reshoot, images -> video).", after_help = HELP)]
struct Cli {
    /// Input image(s): files, directories, globs ("photos/*.jpg"), or `-` for an image on stdin. Each input is one job; models
    /// are loaded once for all of them.
    #[arg(value_name = "INPUT")]
    inputs: Vec<String>,

    /// Edit instruction. Replaces the preset's prompt if a preset is active. `<image1>`, `<image2>`... refer to the inputs.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Read the prompt from a file (`-` = stdin).
    #[arg(short = 'P', long, value_name = "FILE")]
    prompt_file: Option<String>,
    /// Text appended to the final prompt (handy to extend a preset).
    #[arg(long)]
    append: Option<String>,
    /// Extra reference image added to every job as <image2>, <image3>, ... (repeatable).
    #[arg(short, long = "ref", value_name = "FILE")]
    refs: Vec<PathBuf>,

    /// Preset: 4kify, 4kify-phone, upscale, restore, or `none`. See --list-presets.
    #[arg(long, value_name = "NAME")]
    preset: Option<String>,
    /// List the presets and exit.
    #[arg(long)]
    list_presets: bool,
    /// Preset modifier for the frames of a clip: stricter faithful-enlargement prompt, framing pinned, one seed for all inputs.
    #[arg(long)]
    seq: bool,
    /// Legacy alias for `--preset 4kify-phone`.
    #[arg(long, hide = true)]
    phone: bool,
    /// Print the final prompt (for the first input's size) and exit.
    #[arg(long)]
    show_prompt: bool,

    /// Output size: `WxH`, or `same` for the first input's size. The default is the preset's size, else the input's own size.
    /// Sizes that are not multiples of 16 are generated at the nearest multiple and resampled to exactly this size.
    #[arg(long, value_name = "WxH|same")]
    size: Option<String>,
    /// Output size as a multiple of the input's size, ratio kept (2 = 2x each side; 0.5 = half). Beats a preset's size.
    #[arg(long, value_name = "FACTOR")]
    scale: Option<f64>,
    /// Shorthand for `--preset upscale --scale N`: faithful super-resolution by N (default 2).
    #[arg(long, value_name = "N", num_args = 0..=1, default_missing_value = "2")]
    upscale: Option<f64>,
    /// Output width in pixels, input ratio kept.
    #[arg(long, value_name = "PX")]
    width: Option<usize>,
    /// Output height in pixels, input ratio kept.
    #[arg(long, value_name = "PX")]
    height: Option<usize>,
    /// Output long edge in pixels, input ratio kept.
    #[arg(long, value_name = "PX")]
    long_edge: Option<usize>,
    /// Output size at this many megapixels, input ratio kept.
    #[arg(long, value_name = "MP")]
    megapixels: Option<f64>,
    /// Deliver the model's native multiple-of-16 size instead of resampling to the exact requested size.
    #[arg(long)]
    snap: bool,
    /// Output PNG path (single input), a directory (several inputs), or `-` for PNG on stdout. Default: next to the input as
    /// <name>_edit.png (preset suffix, e.g. _4k); an input from stdin defaults to stdout.
    #[arg(short, long)]
    out: Option<String>,

    /// Sampling steps.
    #[arg(long, default_value_t = 25)]
    steps: usize,
    /// Noise seed (incremented per input; held fixed with --seq).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Flow shift of the sampling schedule.
    #[arg(long, default_value_t = 0.69)]
    shift: f32,
    /// Size parameter of the references seen by the model (0 = automatic: the input's own size, capped at the output's pixel
    /// budget; wallpaper presets: see README).
    #[arg(long, default_value_t = 0)]
    ref_resolution: usize,

    /// Silence progress/diagnostics on stderr (errors still print).
    #[arg(short, long)]
    quiet: bool,
    /// Print one JSON object per result on stdout instead of the bare path.
    #[arg(long)]
    json: bool,

    /// Qwen Image 2.1 diffusion model (int8 convrot safetensors). Default: auto-discover next to the binary, or download.
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Qwen3-VL 8B text encoder (int8 convrot safetensors).
    #[arg(long)]
    text_encoder: Option<PathBuf>,
    /// Qwen Image 2.1 VAE (bf16 safetensors).
    #[arg(long)]
    vae: Option<PathBuf>,
}

/// Entry point shared by the binaries: `prog` names the tool in help, `default_preset` is applied unless `--preset` says otherwise.
pub fn main(prog: &'static str, default_preset: Option<&'static str>) {
    if let Err(e) = run(prog, default_preset) {
        let msg = format!("{e:#}");
        eprintln!("{prog}: error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications (a running ComfyUI holds its models in VRAM), or lower --size / --ref-resolution.");
        }
        std::process::exit(1);
    }
}

enum Source {
    Path(PathBuf),
    Stdin(Rgb8),
}

impl Source {
    fn name(&self) -> String {
        match self {
            Source::Path(p) => p.display().to_string(),
            Source::Stdin(_) => "<stdin>".to_string(),
        }
    }
    fn load(&self) -> Result<Rgb8> {
        match self {
            Source::Path(p) => Rgb8::load(p),
            Source::Stdin(i) => Ok(i.clone()),
        }
    }
    fn dims(&self) -> Result<(usize, usize)> {
        match self {
            Source::Path(p) => {
                let (w, h) = image::image_dimensions(p).with_context(|| format!("reading {}", p.display()))?;
                Ok((w as usize, h as usize))
            }
            Source::Stdin(i) => Ok((i.w, i.h)),
        }
    }
}

enum Dest {
    Stdout,
    File(PathBuf),
}

fn round16(x: f64) -> usize {
    (((x / 16.0).round() as usize) * 16).max(16)
}

/// (generated size, delivered size). Priority: --size; one of --scale/--width/--height/--long-edge/--megapixels
/// (all keep the input's ratio); the preset's fixed size; the preset's default scale; the input's own size.
fn target_size(cli: &Cli, preset: Option<&Preset>, first: (usize, usize)) -> Result<((usize, usize), (usize, usize))> {
    let (iw, ih) = (first.0 as f64, first.1 as f64);
    let ratio = iw / ih;
    let exact = |w: f64, h: f64| -> (usize, usize) { ((w.round() as usize).max(16), (h.round() as usize).max(16)) };
    let scale = cli.scale.or(cli.upscale);
    let n_scaling = [scale.is_some(), cli.width.is_some(), cli.height.is_some(), cli.long_edge.is_some(), cli.megapixels.is_some()].iter().filter(|b| **b).count();
    ensure!(n_scaling <= 1, "use only one of --scale/--upscale, --width, --height, --long-edge, --megapixels");
    let out: (usize, usize) = if let Some(s) = &cli.size {
        if s == "same" {
            (first.0, first.1)
        } else {
            let (a, b) = s.split_once(['x', 'X']).context("--size must look like WIDTHxHEIGHT or `same`")?;
            (a.trim().parse::<usize>().context("bad width")?, b.trim().parse::<usize>().context("bad height")?)
        }
    } else if let Some(f) = scale {
        ensure!(f > 0.0, "--scale must be positive");
        exact(iw * f, ih * f)
    } else if let Some(w) = cli.width {
        exact(w as f64, w as f64 / ratio)
    } else if let Some(h) = cli.height {
        exact(h as f64 * ratio, h as f64)
    } else if let Some(l) = cli.long_edge {
        if ratio >= 1.0 { exact(l as f64, l as f64 / ratio) } else { exact(l as f64 * ratio, l as f64) }
    } else if let Some(mp) = cli.megapixels {
        ensure!(mp > 0.0, "--megapixels must be positive");
        let w = (mp * 1e6 * ratio).sqrt();
        exact(w, mp * 1e6 / w)
    } else if let Some(sz) = preset.and_then(|p| p.size()) {
        sz
    } else if let Some(f) = preset.and_then(|p| p.default_scale()) {
        exact(iw * f, ih * f)
    } else {
        (first.0, first.1)
    };
    ensure!(out.0 >= 16 && out.1 >= 16, "output size {}x{} is too small", out.0, out.1);
    let gen = (round16(out.0 as f64), round16(out.1 as f64));
    Ok((gen, if cli.snap { gen } else { out }))
}

fn read_stdin() -> Result<Vec<u8>> {
    let mut v = Vec::new();
    std::io::stdin().lock().read_to_end(&mut v).context("reading stdin")?;
    Ok(v)
}

fn expand_inputs(inputs: &[String]) -> Result<Vec<Source>> {
    let exts = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
    let is_img = |p: &Path| p.extension().map(|e| exts.contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false);
    let mut out = Vec::new();
    for inp in inputs {
        let p = Path::new(inp);
        if inp == "-" {
            let bytes = read_stdin()?;
            ensure!(!bytes.is_empty(), "no image data on stdin");
            out.push(Source::Stdin(Rgb8::from_bytes(&bytes).context("decoding the image on stdin")?));
        } else if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_img(p)).collect();
            v.sort();
            out.extend(v.into_iter().map(Source::Path));
        } else if p.is_file() {
            out.push(Source::Path(p.to_path_buf()));
        } else if inp.contains('*') || inp.contains('?') {
            // simple glob: directory part is literal, file part has wildcards
            let (dir, pat) = match p.parent() {
                Some(d) if !d.as_os_str().is_empty() => (d.to_path_buf(), p.file_name().unwrap().to_string_lossy().to_string()),
                _ => (PathBuf::from("."), inp.clone()),
            };
            let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
                .with_context(|| format!("reading {}", dir.display()))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|f| f.is_file() && glob_match(&pat.to_lowercase(), &f.file_name().unwrap().to_string_lossy().to_lowercase()))
                .collect();
            v.sort();
            if v.is_empty() {
                crate::info!("warning: no files match {inp}");
            }
            out.extend(v.into_iter().map(Source::Path));
        } else {
            bail!("input {inp} not found");
        }
    }
    ensure!(!out.is_empty(), "no input images");
    Ok(out)
}

fn glob_match(pat: &str, s: &str) -> bool {
    let p: Vec<char> = pat.chars().collect();
    let t: Vec<char> = s.chars().collect();
    fn go(p: &[char], t: &[char]) -> bool {
        match (p.first(), t.first()) {
            (None, None) => true,
            (Some('*'), _) => go(&p[1..], t) || (!t.is_empty() && go(p, &t[1..])),
            (Some('?'), Some(_)) => go(&p[1..], &t[1..]),
            (Some(a), Some(b)) if a == b => go(&p[1..], &t[1..]),
            _ => false,
        }
    }
    go(&p, &t)
}

fn destination(out: &Option<String>, src: &Source, njobs: usize, suffix: &str) -> Result<Dest> {
    match (out.as_deref(), src) {
        (Some("-"), _) => {
            ensure!(njobs == 1, "-o - (PNG on stdout) needs exactly one input");
            Ok(Dest::Stdout)
        }
        (Some(o), _) => {
            let o = PathBuf::from(o);
            if njobs == 1 && o.extension().is_some() && !o.is_dir() {
                Ok(Dest::File(o))
            } else {
                let stem = match src {
                    Source::Path(p) => p.file_stem().unwrap_or_default().to_string_lossy().to_string(),
                    Source::Stdin(_) => "stdin".to_string(),
                };
                Ok(Dest::File(o.join(format!("{stem}{suffix}.png"))))
            }
        }
        (None, Source::Stdin(_)) => {
            ensure!(njobs == 1, "several inputs including stdin need -o DIR");
            Ok(Dest::Stdout)
        }
        (None, Source::Path(p)) => {
            let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
            Ok(Dest::File(p.with_file_name(format!("{stem}{suffix}.png"))))
        }
    }
}

fn run(prog: &'static str, default_preset: Option<&'static str>) -> Result<()> {
    let cmd = Cli::command().name(prog).bin_name(prog).after_help(HELP.replace("PROG", prog));
    let cli = Cli::from_arg_matches(&cmd.get_matches())?;
    crate::log::set_quiet(cli.quiet);

    if cli.list_presets {
        for (n, d) in presets::PRESETS {
            println!("{n:<12} {d}");
        }
        return Ok(());
    }

    // preset resolution: --preset NAME | --phone | the binary's default
    let preset_name: Option<String> = match (cli.preset.as_deref().or(if cli.upscale.is_some() { Some("upscale") } else { None }), cli.phone) {
        (Some("none"), _) => None,
        (Some(n), false) => Some(n.to_string()),
        (Some(n), true) if n == "4kify" || n == "4kify-phone" => Some("4kify-phone".to_string()),
        (Some(n), true) => bail!("--phone conflicts with --preset {n}"),
        (None, true) => Some("4kify-phone".to_string()),
        (None, false) => default_preset.map(|s| s.to_string()),
    };
    let preset = match &preset_name {
        Some(n) => Some(Preset::parse(n, cli.seq)?),
        None => {
            ensure!(!cli.seq, "--seq modifies a preset; pick one with --preset");
            None
        }
    };

    // prompt: explicit text replaces the preset's; otherwise the preset's; --append extends either
    let explicit_prompt = match (&cli.prompt, &cli.prompt_file) {
        (Some(_), Some(_)) => bail!("use --prompt or --prompt-file, not both"),
        (Some(p), None) => Some(p.clone()),
        (None, Some(f)) if f == "-" => {
            ensure!(!cli.inputs.iter().any(|i| i == "-"), "stdin cannot carry both the prompt and an input image");
            Some(String::from_utf8(read_stdin()?).context("prompt on stdin is not UTF-8")?)
        }
        (None, Some(f)) => Some(std::fs::read_to_string(f).with_context(|| format!("reading {f}"))?),
        (None, None) => None,
    };
    ensure!(explicit_prompt.is_some() || preset.is_some(), "give an edit instruction with --prompt / --prompt-file, or choose a --preset (see --list-presets)");
    let make_prompt = |w: usize, h: usize| -> String {
        let mut p = match (&explicit_prompt, &preset) {
            (Some(t), _) => t.trim().to_string(),
            (None, Some(pr)) => pr.prompt(w, h),
            (None, None) => unreachable!(),
        };
        if let Some(a) = &cli.append {
            p = format!("{p}\n\n{}", a.trim());
        }
        p
    };

    if cli.show_prompt && cli.inputs.is_empty() {
        // no input needed when the preset (or --size) fixes the output size
        let ((w, h), _) = target_size(&cli, preset.as_ref(), (1024, 1024))?;
        println!("{}", make_prompt(w, h));
        return Ok(());
    }
    ensure!(!cli.inputs.is_empty(), "no input image (give files, a directory, a glob, or `-` for stdin)");
    let sources = expand_inputs(&cli.inputs)?;
    let first_dims = sources[0].dims()?;
    if cli.show_prompt {
        let ((w, h), _) = target_size(&cli, preset.as_ref(), first_dims)?;
        println!("{}", make_prompt(w, h));
        return Ok(());
    }
    for r in &cli.refs {
        ensure!(r.is_file(), "reference image {} not found", r.display());
    }
    let suffix = preset.as_ref().map(|p| p.suffix()).unwrap_or("_edit");
    let dests: Vec<Dest> = sources.iter().map(|s| destination(&cli.out, s, sources.len(), suffix)).collect::<Result<_>>()?;
    let fixed_seed = preset.as_ref().map(|p| p.seq).unwrap_or(false);
    let policy = preset.as_ref().map(|p| p.ref_policy()).unwrap_or(crate::pipeline::RefPolicy::Native);

    let dit_path = models::resolve(cli.dit.clone(), &models::DIT)?;
    let te_path = models::resolve(cli.text_encoder.clone(), &models::TEXT_ENCODER)?;
    let vae_path = models::resolve(cli.vae.clone(), &models::VAE)?;
    crate::info!("{prog}: {} input(s){}{}", sources.len(), preset_name.as_ref().map(|n| format!(", preset {n}")).unwrap_or_default(), if cli.refs.is_empty() { String::new() } else { format!(", {} extra reference(s)", cli.refs.len()) });
    crate::info!("  dit: {}\n  text encoder: {}\n  vae: {}", dit_path.display(), te_path.display(), vae_path.display());

    let dev = Device::new(0)?;
    crate::info!("  gpu: {} MB free of {} MB", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let base = Settings { target_w: 0, target_h: 0, out_w: 0, out_h: 0, steps: cli.steps, shift: cli.shift, seed: cli.seed, ref_resolution: cli.ref_resolution, ref_policy: policy, prompt: String::new() };
    let _init = Pipeline::new(dev.clone(), base.clone())?; // initializes the GEMM/attention/VAE kernels
    let t_all = std::time::Instant::now();
    let ref_imgs: Vec<Rgb8> = cli.refs.iter().map(|p| Rgb8::load(p)).collect::<Result<_>>()?;

    // ---- phase 1: load the VAE and text encoder; encode every input
    let vae = Vae::load(dev.clone(), &vae_path)?;
    let te = TextEncoder::load(dev.clone(), &te_path)?;
    let mut encoded: Vec<(usize, Settings, Encoded)> = Vec::new();
    let mut failed = 0usize;
    for (i, src) in sources.iter().enumerate() {
        crate::info!("encoding {}", src.name());
        let res = (|| -> Result<(Settings, Encoded)> {
            let img = src.load()?;
            let ((w, h), (ow, oh)) = target_size(&cli, preset.as_ref(), (img.w, img.h))?;
            let mut s = base.clone();
            s.target_w = w;
            s.target_h = h;
            s.out_w = ow;
            s.out_h = oh;
            s.prompt = make_prompt(w, h);
            s.seed = if fixed_seed { cli.seed } else { cli.seed.wrapping_add(i as u64) };
            let p = Pipeline { dev: dev.clone(), settings: s.clone() };
            let mut imgs = vec![img];
            imgs.extend(ref_imgs.iter().cloned());
            let enc = p.encode(&te, &vae, &imgs)?;
            Ok((s, enc))
        })();
        match res {
            Ok((s, e)) => encoded.push((i, s, e)),
            Err(e) => {
                eprintln!("{prog}: skipping {}: {e:#}", src.name());
                failed += 1;
            }
        }
    }
    drop(te);
    dev.sync()?;
    if encoded.is_empty() {
        bail!("nothing to do");
    }

    // ---- phase 2: load the DiT; sample, decode and write each result as soon as it is ready
    let model = Dit::load(dev.clone(), &dit_path)?;
    let show_bar = !cli.quiet && std::io::stderr().is_terminal();
    let stdout_is_image = dests.iter().any(|d| matches!(d, Dest::Stdout));
    for (i, settings, enc) in &encoded {
        let src = &sources[*i];
        let pipe_i = Pipeline { dev: dev.clone(), settings: settings.clone() };
        crate::info!("sampling {} -> {}x{} (seed {})", src.name(), settings.out_w, settings.out_h, settings.seed);
        let t0 = std::time::Instant::now();
        let bar = if show_bar { indicatif::ProgressBar::new(settings.steps as u64) } else { indicatif::ProgressBar::hidden() };
        bar.set_style(indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
        let lat = pipe_i.sample(&model, enc, &vae.latents_mean, &vae.latents_std, &|n, _| bar.set_position(n as u64))?;
        bar.finish_and_clear();
        crate::info!("  sampled in {:.1}s", t0.elapsed().as_secs_f64());
        let t0 = std::time::Instant::now();
        let mut out_img = pipe_i.decode(&vae, &lat)?;
        drop(lat);
        if (out_img.w, out_img.h) != (settings.out_w, settings.out_h) {
            crate::info!("  resampling {}x{} -> exactly {}x{}", out_img.w, out_img.h, settings.out_w, settings.out_h);
            out_img = out_img.resize_lanczos(settings.out_w, settings.out_h);
        }
        match &dests[*i] {
            Dest::Stdout => {
                let mut so = std::io::stdout().lock();
                so.write_all(&out_img.png_bytes()?)?;
                so.flush()?;
                crate::info!("wrote PNG to stdout ({:.1}s decode+encode)", t0.elapsed().as_secs_f64());
            }
            Dest::File(path) => {
                if let Some(d) = path.parent() {
                    if !d.as_os_str().is_empty() {
                        std::fs::create_dir_all(d)?;
                    }
                }
                out_img.save_png(path)?;
                crate::info!("wrote {} ({:.1}s decode+save)", path.display(), t0.elapsed().as_secs_f64());
                if cli.json {
                    let j = serde_json::json!({
                        "input": src.name(), "output": path.display().to_string(), "width": settings.out_w, "height": settings.out_h,
                        "seed": settings.seed, "steps": settings.steps, "preset": preset_name,
                    });
                    println!("{j}");
                } else if !stdout_is_image {
                    println!("{}", path.display());
                }
            }
        }
    }
    drop(model);
    dev.sync()?;
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    if failed > 0 {
        bail!("{failed} of {} input(s) failed", sources.len());
    }
    Ok(())
}
