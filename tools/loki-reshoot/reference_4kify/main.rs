//! 4kify: restore a photo and outpaint it into a 4K desktop (or phone) wallpaper with Qwen Image 2.1,
//! using a self-contained CUDA inference engine. `--seq` treats the inputs as the frames of a clip:
//! one seed for all of them and a prompt that pins the framing to the input.
use anyhow::{bail, Context, Result};
use clap::Parser;
use fourkify::cuda::Device;
use fourkify::dit::Dit;
use fourkify::image::Rgb8;
use fourkify::pipeline::{build_prompt, Encoded, Orientation, Pipeline, Settings};
use fourkify::text_encoder::TextEncoder;
use fourkify::vae::Vae;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "4kify", version, about = "Restore + outpaint photos into 4K wallpapers (Qwen Image 2.1, standalone CUDA engine)")]
struct Cli {
    /// Qwen Image 2.1 diffusion model (int8 convrot safetensors). Default: auto-discover next to the binary.
    #[arg(long)]
    dit: Option<PathBuf>,
    /// Qwen3-VL 8B text encoder (int8 convrot safetensors). Default: auto-discover next to the binary.
    #[arg(long)]
    text_encoder: Option<PathBuf>,
    /// Qwen Image 2.1 VAE (bf16 safetensors). Default: auto-discover next to the binary.
    #[arg(long)]
    vae: Option<PathBuf>,
    /// Make a vertical phone wallpaper (1296x2800, iPhone-friendly 9:19.5) instead of a 3840x2160 desktop one.
    #[arg(long)]
    phone: bool,
    /// Override the output size, e.g. 2560x1440 (multiples of 16).
    #[arg(long)]
    size: Option<String>,
    /// Output file (single input) or directory (multiple inputs). Default: <input>_4k.png next to the input.
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Sampling steps.
    #[arg(long, default_value_t = 25)]
    steps: usize,
    /// Noise seed (per image; incremented for each additional input, held fixed with --seq).
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Reference resolution passed to the text encoder/VAE (0 = native size rounded to 32, upscaled to at
    /// least 1024 for small inputs and capped at the output's pixel budget).
    #[arg(long, default_value_t = 0)]
    ref_resolution: usize,
    /// Replace the baked-in prompt entirely (must mention <image1>).
    #[arg(long)]
    prompt: Option<String>,
    /// Print the prompt that would be used and exit.
    #[arg(long)]
    show_prompt: bool,
    /// Sequence mode for the frames of a clip: one seed for every input and a stricter prompt that asks for
    /// a faithful enlargement framed identically on every frame (see --seq --show-prompt).
    #[arg(long)]
    seq: bool,
    /// Input image(s): files, directories, or globs (e.g. "photos/*.jpg"). The last arguments.
    #[arg(required_unless_present = "show_prompt")]
    inputs: Vec<String>,
}

const HF_BASE: &str = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/";

/// Download `HF_BASE + hf_path` to `dest`, resuming a partial `.part` file if one exists.
fn download(hf_path: &str, dest: &Path) -> Result<()> {
    use std::io::{Read, Write};
    let url = format!("{HF_BASE}{hf_path}");
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut req = ureq::get(&url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        // 416: the .part file already holds everything
        Err(ureq::Error::Status(416, _)) => {
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(e) => bail!("downloading {url}: {e}"),
    };
    let resumed = resp.status() == 206;
    let start = if resumed { have } else { 0 };
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).map(|n| n + start);
    let pb = indicatif::ProgressBar::new(total.unwrap_or(0));
    pb.set_style(
        indicatif::ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}")
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
    );
    pb.set_position(start);
    let mut file = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("downloading {url} (re-run to resume)"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
    }
    file.flush()?;
    drop(file);
    pb.finish();
    if let Some(t) = total {
        let got = std::fs::metadata(&part)?.len();
        if got != t {
            bail!("download of {url} was truncated ({got}/{t} bytes); re-run to resume");
        }
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}

/// Resolve a model file: explicit path, else search next to the binary / ./models, else download from HF.
fn discover(explicit: Option<PathBuf>, patterns: &[&str], what: &str, hf_path: &str) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("{what}: {} does not exist", p.display());
        }
        return Ok(p);
    }
    if let Some(p) = find_local(patterns) {
        return Ok(p);
    }
    let base = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| PathBuf::from("."));
    let dest = base.join("models").join(hf_path.rsplit('/').next().unwrap());
    eprintln!("{what} not found locally; downloading {hf_path} from Hugging Face to {}", dest.display());
    download(hf_path, &dest)?;
    Ok(dest)
}

fn find_local(patterns: &[&str]) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
            dirs.push(d.join("models"));
        }
    }
    dirs.push(PathBuf::from("."));
    dirs.push(PathBuf::from("models"));
    for d in &dirs {
        if let Ok(rd) = std::fs::read_dir(d) {
            let mut names: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            names.sort();
            for pat in patterns {
                for n in &names {
                    let f = n.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
                    if f.ends_with(".safetensors") && pat.split('*').all(|part| f.contains(part)) {
                        return Some(n.clone());
                    }
                }
            }
        }
    }
    None
}

fn expand_inputs(inputs: &[String]) -> Result<Vec<PathBuf>> {
    let exts = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"];
    let is_img = |p: &Path| p.extension().map(|e| exts.contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false);
    let mut out = Vec::new();
    for inp in inputs {
        let p = Path::new(inp);
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| is_img(p)).collect();
            v.sort();
            out.extend(v);
        } else if p.is_file() {
            out.push(p.to_path_buf());
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
                eprintln!("warning: no files match {inp}");
            }
            out.extend(v);
        } else {
            bail!("input {inp} not found");
        }
    }
    if out.is_empty() {
        bail!("no input images");
    }
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

fn main() -> Result<()> {
    if let Err(e) = run() {
        let msg = format!("{e:#}");
        eprintln!("error: {msg}");
        if msg.contains("allocating") || msg.contains("OUT_OF_MEMORY") {
            eprintln!("hint: the GPU ran out of memory. Close other GPU applications (e.g. a running ComfyUI holds its models in VRAM),
      or lower --ref-resolution (e.g. 1536) to shrink the reference image's share of the attention cache.");
        }
        std::process::exit(1);
    }
    Ok(())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    let (mut tw, mut th, orientation) = if cli.phone { (1296usize, 2800usize, Orientation::Portrait) } else { (3840usize, 2160usize, Orientation::Landscape) };
    if let Some(s) = &cli.size {
        let (a, b) = s.split_once('x').context("--size must look like WIDTHxHEIGHT")?;
        tw = a.trim().parse::<usize>().context("bad width")?;
        th = b.trim().parse::<usize>().context("bad height")?;
        if tw % 16 != 0 || th % 16 != 0 {
            bail!("--size must be multiples of 16 (got {tw}x{th})");
        }
    }
    let orientation = if cli.size.is_some() { if th > tw { Orientation::Portrait } else { Orientation::Landscape } } else { orientation };
    if let Some(p) = &cli.prompt {
        if !p.contains("<image1>") {
            bail!("--prompt must mention <image1>");
        }
    }
    let settings = Settings { target_w: tw, target_h: th, orientation, steps: cli.steps, shift: 0.69, seed: cli.seed, ref_resolution: cli.ref_resolution, prompt_override: cli.prompt.clone(), seq: cli.seq };
    if cli.show_prompt {
        println!("{}", build_prompt(orientation, tw, th, cli.seq));
        return Ok(());
    }
    let inputs = expand_inputs(&cli.inputs)?;
    let dit_path = discover(cli.dit.clone(), &["qwen_image_2.1*int8", "qwen_image_2.1*convrot", "qwen_image_2.1"], "diffusion model", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors")?;
    let te_path = discover(cli.text_encoder.clone(), &["qwen3vl*8b", "qwen3vl"], "text encoder", "text_encoders/qwen3vl_8b_int8_convrot.safetensors")?;
    let vae_path = discover(cli.vae.clone(), &["qwen_image_2.1*vae", "qwen_image*vae"], "VAE", "vae/qwen_image_2.1_vae_bf16.safetensors")?;
    eprintln!("4kify: {} image(s) -> {}x{} {:?}{}", inputs.len(), tw, th, orientation, if cli.seq { " (sequence)" } else { "" });
    eprintln!("  dit: {}\n  text encoder: {}\n  vae: {}", dit_path.display(), te_path.display(), vae_path.display());

    let dev = Device::new(0)?;
    eprintln!("  gpu: {} MB free of {} MB", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let pipe = Pipeline::new(dev.clone(), settings)?;
    let t_all = std::time::Instant::now();

    // ---- phase 1: load the VAE and text encoder; encode every input
    let vae = Vae::load(dev.clone(), &vae_path)?;
    let te = TextEncoder::load(dev.clone(), &te_path)?;
    let mut encoded: Vec<(PathBuf, Encoded)> = Vec::new();
    for p in &inputs {
        eprintln!("encoding {}", p.display());
        let img = match Rgb8::load(p) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("  skipping: {e:#}");
                continue;
            }
        };
        match pipe.encode(&te, &vae, &img) {
            Ok(e) => encoded.push((p.clone(), e)),
            Err(e) => eprintln!("  skipping: {e:#}"),
        }
    }
    drop(te);
    dev.sync()?;
    if encoded.is_empty() {
        bail!("nothing to do");
    }

    // ---- phase 2: load the DiT; sample, decode and write each image in turn
    let model = Dit::load(dev.clone(), &dit_path)?;
    for (i, (p, enc)) in encoded.iter().enumerate() {
        let mut pipe_i = Pipeline { dev: dev.clone(), settings: pipe.settings.clone() };
        pipe_i.settings.seed = if cli.seq { pipe.settings.seed } else { pipe.settings.seed.wrapping_add(i as u64) };
        eprintln!("sampling {} (seed {})", p.display(), pipe_i.settings.seed);
        let t0 = std::time::Instant::now();
        let bar = indicatif::ProgressBar::new(pipe.settings.steps as u64);
        bar.set_style(indicatif::ProgressStyle::with_template("  {bar:40} {pos}/{len} steps  {elapsed_precise} eta {eta_precise}").unwrap());
        let lat = pipe_i.sample(&model, enc, &vae.latents_mean, &vae.latents_std, &|i, _| bar.set_position(i as u64))?;
        bar.finish_and_clear();
        eprintln!("  sampled in {:.1}s", t0.elapsed().as_secs_f64());
        let t0 = std::time::Instant::now();
        let out_img = pipe.decode(&vae, &lat)?;
        drop(lat);
        let out_path = output_path(&cli.out, p, inputs.len() > 1, cli.phone);
        if let Some(d) = out_path.parent() {
            if !d.as_os_str().is_empty() {
                std::fs::create_dir_all(d)?;
            }
        }
        out_img.save_png(&out_path)?;
        eprintln!("wrote {} ({:.1}s decode+save)", out_path.display(), t0.elapsed().as_secs_f64());
    }
    drop(model);
    dev.sync()?;
    eprintln!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}

fn output_path(out: &Option<PathBuf>, input: &Path, multi: bool, phone: bool) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let suffix = if phone { "_phone" } else { "_4k" };
    match out {
        Some(o) if !multi && (o.extension().is_some() && !o.is_dir()) => o.clone(),
        Some(o) => o.join(format!("{stem}{suffix}.png")),
        None => input.with_file_name(format!("{stem}{suffix}.png")),
    }
}
