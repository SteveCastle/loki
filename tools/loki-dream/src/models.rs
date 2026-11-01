//! Model file discovery (next to the binary / its models/ folder / $LOKI_MODELS / ./ / ./models) and resumable download
//! from Hugging Face (ungated repos: lodestones/Kromoss for MOSS-Image, Comfy-Org's Krea 2 and FLUX.2 klein repos,
//! circlestone-labs/Anima). File names are kept as published, so one shared `models/` folder serves every loki- tool and
//! the media server (and a ComfyUI models tree can be listed in $LOKI_MODELS).
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};


pub struct Spec {
    pub what: &'static str,
    /// Hugging Face repo the file is downloaded from
    pub repo: &'static str,
    pub patterns: &'static [&'static str],
    pub hf_path: &'static str,
    pub size_gb: f32,
}

const KROMOSS: &str = "lodestones/Kromoss";

pub const DIT_INT8: Spec = Spec { repo: KROMOSS, what: "Kromoss diffusion model (int8)", patterns: &["kromoss*int8", "kromoss*convrot"], hf_path: "diffusion_models/kromoss_int8_convrot.safetensors", size_gb: 5.4 };
pub const DIT_BF16: Spec = Spec { repo: KROMOSS, what: "Kromoss diffusion model (bf16)", patterns: &["kromoss_bf16", "kromoss*bf16"], hf_path: "diffusion_models/kromoss_bf16.safetensors", size_gb: 10.2 };
pub const TEXT_ENCODER: Spec = Spec { repo: KROMOSS, what: "Qwen3.5-9B text encoder", patterns: &["qwen3.5_9b", "qwen3.5*9b", "qwen3_5*9b"], hf_path: "text_encoders/qwen3.5_9b_bf16.safetensors", size_gb: 19.3 };
pub const VAE: Spec = Spec { repo: KROMOSS, what: "FLUX.2 small VAE", patterns: &["flux2-vae_small", "flux2*vae*small"], hf_path: "vae/flux2-vae_small.safetensors", size_gb: 0.25 };
pub const TURBO_LORA: Spec = Spec { repo: KROMOSS, what: "8-step DMD2 turbo LoRA", patterns: &["8-steps-dmd2-lora"], hf_path: "8-steps-dmd2-lora.safetensors", size_gb: 1.2 };
pub const TURBO_LORA_DMD: Spec = Spec { repo: KROMOSS, what: "8-step DMD turbo LoRA", patterns: &["8-steps-dmd-lora"], hf_path: "8-steps-dmd-lora.safetensors", size_gb: 1.2 };

// ---- Krea 2 (Comfy-Org/Krea-2: the official ComfyUI files; the krea/ repos are gated)
pub const KREA2_DIT: Spec = Spec { repo: "Comfy-Org/Krea-2", what: "Krea 2 Turbo diffusion model (int8)", patterns: &["krea2_turbo_int8", "krea2*turbo*int8", "krea2_turbo"], hf_path: "diffusion_models/krea2_turbo_int8_convrot.safetensors", size_gb: 13.5 };
pub const KREA2_TE: Spec = Spec { repo: "Comfy-Org/Krea-2", what: "Qwen3-VL-4B text encoder", patterns: &["qwen3vl_4b_bf16", "qwen3vl_4b", "qwen3-vl-4b", "qwen3_vl_4b", "qwen3vl4b"], hf_path: "text_encoders/qwen3vl_4b_bf16.safetensors", size_gb: 8.9 };
pub const QWEN_IMAGE_VAE: Spec = Spec { repo: "Comfy-Org/Krea-2", what: "Qwen-Image VAE", patterns: &["qwen_image_vae"], hf_path: "vae/qwen_image_vae.safetensors", size_gb: 0.25 };

// ---- FLUX.2 klein (Comfy-Org/Flux-2-Klein-4B / -9B)
pub const KLEIN_DIT: Spec = Spec { repo: "Comfy-Org/Flux-2-Klein-4B", what: "FLUX.2 klein 4B diffusion model", patterns: &["flux-2-klein-4b.", "flux-2-klein-4b"], hf_path: "split_files/diffusion_models/flux-2-klein-4b.safetensors", size_gb: 7.8 };
pub const KLEIN4B_TE: Spec = Spec { repo: "Comfy-Org/Flux-2-Klein-4B", what: "Qwen3-4B text encoder", patterns: &["qwen_3_4b.", "qwen_3_4b", "qwen3-4b", "qwen3_4b"], hf_path: "split_files/text_encoders/qwen_3_4b.safetensors", size_gb: 8.0 };
pub const KLEIN9B_TE: Spec = Spec { repo: "Comfy-Org/Flux-2-Klein-9B", what: "Qwen3-8B text encoder", patterns: &["qwen_3_8b.", "qwen_3_8b", "qwen3-8b", "qwen3_8b"], hf_path: "split_files/text_encoders/qwen_3_8b.safetensors", size_gb: 16.4 };
pub const FLUX2_VAE: Spec = Spec { repo: "Comfy-Org/Flux-2-Klein-4B", what: "FLUX.2 VAE", patterns: &["flux2-vae.safetensors", "flux2_vae.safetensors"], hf_path: "split_files/vae/flux2-vae.safetensors", size_gb: 0.34 };

// ---- Anima (circlestone-labs/Anima)
pub const ANIMA_DIT: Spec = Spec { repo: "circlestone-labs/Anima", what: "Anima diffusion model", patterns: &["anima-base", "anima_base"], hf_path: "split_files/diffusion_models/anima-base-v1.0.safetensors", size_gb: 4.2 };
/// the distilled 10-step Anima, downloaded only when named (`--model anima-turbo-v1.1`)
pub const ANIMA_TURBO_DIT: Spec = Spec { repo: "circlestone-labs/Anima", what: "Anima Turbo diffusion model", patterns: &["anima-turbo", "anima_turbo"], hf_path: "split_files/diffusion_models/anima-turbo-v1.1.safetensors", size_gb: 4.2 };
pub const ANIMA_TE: Spec = Spec { repo: "circlestone-labs/Anima", what: "Qwen3-0.6B text encoder", patterns: &["qwen_3_06b", "qwen3-0.6b", "qwen3_0.6b", "qwen3_06b"], hf_path: "split_files/text_encoders/qwen_3_06b_base.safetensors", size_gb: 1.2 };

/// Download `https://huggingface.co/<repo>/resolve/main/<hf_path>` to `dest`, resuming a partial `.part` file if one exists.
fn download(repo: &str, hf_path: &str, dest: &Path) -> Result<()> {
    use std::io::{Read, Write};
    let url = format!("https://huggingface.co/{repo}/resolve/main/{hf_path}");
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
    // Replicas sharing one models volume (serverless workers) must not write the same .part file at once: one
    // downloads, the others wait for it (a lock nobody has touched for 2 minutes is stale and taken over).
    let lock = PathBuf::from(format!("{}.lock", dest.display()));
    struct Unlock(PathBuf);
    impl Drop for Unlock {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let fresh = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).map(|t| t.elapsed().map(|e| e.as_secs() < 120).unwrap_or(true)).unwrap_or(false);
    let mut announced = false;
    let _unlock = loop {
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&lock) {
            Ok(_) => break Unlock(lock.clone()),
            Err(_) if dest.exists() => return Ok(()),
            Err(_) if !fresh(&part) && !fresh(&lock) => {
                let _ = std::fs::remove_file(&lock);
            }
            Err(_) => {
                if !announced {
                    eprintln!("  another process is downloading {}; waiting for it", dest.display());
                    announced = true;
                }
                std::thread::sleep(std::time::Duration::from_secs(5));
            }
        }
    };
    if dest.exists() {
        return Ok(());
    }
    let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    let mut req = ureq::get(&url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, _)) => {
            std::fs::rename(&part, dest)?;
            return Ok(());
        }
        Err(e) => bail!("downloading {url}: {e}"),
    };
    let resumed = resp.status() == 206;
    let start = if resumed { have } else { 0 };
    let total = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).map(|n| n + start);
    let pb = if crate::log::quiet() { indicatif::ProgressBar::hidden() } else { indicatif::ProgressBar::new(total.unwrap_or(0)) };
    pb.set_style(indicatif::ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}").unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()));
    pb.set_position(start);
    let mut file = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut reader = resp.into_reader();
    let tty = std::io::IsTerminal::is_terminal(&std::io::stderr());
    let (mut got, mut last_pct) = (start, 0u64);
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("downloading {url} (re-run to resume)"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        pb.inc(n as u64);
        got += n as u64;
        // without a terminal the bar draws nothing: log every 5% instead (supervisors relay stderr to remote callers)
        if !tty {
            if let Some(t) = total.filter(|t| *t > 0) {
                let pct = got * 100 / t;
                if pct >= last_pct + 5 || got == t {
                    last_pct = pct - pct % 5;
                    eprintln!("  downloading {}: {pct}% ({:.1}/{:.1} GB)", dest.file_name().unwrap_or_default().to_string_lossy(), got as f64 / 1e9, t as f64 / 1e9);
                }
            }
        }
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

/// Folders searched for model files, in order.
pub fn search_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.to_path_buf());
            dirs.push(d.join("models"));
        }
    }
    if let Some(v) = std::env::var_os("LOKI_MODELS") {
        dirs.extend(std::env::split_paths(&v));
    }
    dirs.push(PathBuf::from("."));
    dirs.push(PathBuf::from("models"));
    dirs
}

/// Whether `path` is `spec`'s own file: a `.safetensors` whose (lowercase) name matches one of the spec's patterns.
pub fn is_spec_file(spec: &Spec, path: &Path) -> bool {
    let f = path.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
    f.ends_with(".safetensors") && spec.patterns.iter().any(|pat| pat.split('*').all(|part| f.contains(part)))
}

pub fn find_local(spec: &Spec) -> Option<PathBuf> {
    for d in search_dirs() {
        if let Ok(rd) = std::fs::read_dir(&d) {
            let mut names: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).collect();
            names.sort();
            for pat in spec.patterns {
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

/// Where missing model files are downloaded: `$LOKI_MODELS_DOWNLOAD` (containers point it at their models volume), else
/// `<binary dir>/models`.
fn download_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("LOKI_MODELS_DOWNLOAD").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("models"))).unwrap_or_else(|| PathBuf::from("models"))
}

/// A bare file name (no folder) looked up in the model folders, with or without its `.safetensors` extension.
fn find_named(p: &Path) -> Option<PathBuf> {
    if p.components().count() != 1 {
        return None;
    }
    let name = p.as_os_str().to_string_lossy();
    let names = [name.to_string(), format!("{name}.safetensors")];
    search_dirs().into_iter().flat_map(|d| names.iter().map(move |n| d.join(n)).collect::<Vec<_>>()).find(|c| c.is_file())
}

/// A diffusion model given on the command line: a path, or a bare file name looked up in the model folders.
pub fn resolve_model(p: &Path) -> Result<PathBuf> {
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if let Some(found) = find_named(p) {
        return Ok(found);
    }
    bail!("diffusion model {} does not exist (a bare file name is looked up in: {})", p.display(), search_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "))
}

/// Resolve a model file: explicit path (or bare file name in the model folders), else search, else download from
/// Hugging Face into models/ next to the binary.
pub fn resolve(explicit: Option<PathBuf>, spec: &Spec) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p);
        }
        if let Some(found) = find_named(&p) {
            return Ok(found);
        }
        bail!("{}: {} does not exist (a bare file name is looked up in: {})", spec.what, p.display(), search_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", "));
    }
    if let Some(p) = find_local(spec) {
        return Ok(p);
    }
    let dest = download_dir().join(spec.hf_path.rsplit('/').next().unwrap());
    // announced even with -q: a multi-GB download should never be a surprise
    eprintln!("{} not found locally; downloading {} ({:.1} GB) from Hugging Face ({}) to {}", spec.what, spec.hf_path, spec.size_gb, spec.repo, dest.display());
    download(spec.repo, spec.hf_path, &dest)?;
    Ok(dest)
}
