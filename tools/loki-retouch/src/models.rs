//! Model file discovery (next to the binary / ./models) and resumable download from Hugging Face.
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

const HF_BASE: &str = "https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/";

pub struct Spec {
    pub what: &'static str,
    pub patterns: &'static [&'static str],
    pub hf_path: &'static str,
}

pub const DIT: Spec = Spec { what: "diffusion model", patterns: &["qwen_image_2.1*int8", "qwen_image_2.1*convrot", "qwen_image_2.1"], hf_path: "diffusion_models/qwen_image_2.1_int8_convrot.safetensors" };
pub const TEXT_ENCODER: Spec = Spec { what: "text encoder", patterns: &["qwen3vl*8b", "qwen3vl"], hf_path: "text_encoders/qwen3vl_8b_int8_convrot.safetensors" };
pub const VAE: Spec = Spec { what: "VAE", patterns: &["qwen_image_2.1*vae", "qwen_image*vae"], hf_path: "vae/qwen_image_2.1_vae_bf16.safetensors" };

/// Download `HF_BASE + hf_path` to `dest`, resuming a partial `.part` file if one exists.
fn download(hf_path: &str, dest: &Path) -> Result<()> {
    use std::io::{Read, Write};
    let url = format!("{HF_BASE}{hf_path}");
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
    let pb = if crate::log::quiet() { indicatif::ProgressBar::hidden() } else { indicatif::ProgressBar::new(total.unwrap_or(0)) };
    pb.set_style(
        indicatif::ProgressStyle::with_template("{bar:40} {bytes}/{total_bytes} {bytes_per_sec} eta {eta}")
            .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar()),
    );
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

/// Where missing model files are downloaded: `$LOKI_MODELS_DOWNLOAD` (containers point it at their models volume), else
/// `<binary dir>/models`.
fn download_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("LOKI_MODELS_DOWNLOAD").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("models"))).unwrap_or_else(|| PathBuf::from("models"))
}

/// Resolve a model file: explicit path, else search next to the binary / ./models, else download from HF.
pub fn resolve(explicit: Option<PathBuf>, spec: &Spec) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if !p.exists() {
            bail!("{}: {} does not exist", spec.what, p.display());
        }
        return Ok(p);
    }
    if let Some(p) = find_local(spec.patterns) {
        return Ok(p);
    }
    let dest = download_dir().join(spec.hf_path.rsplit('/').next().unwrap());
    eprintln!("{} not found locally; downloading {} from Hugging Face to {}", spec.what, spec.hf_path, dest.display());
    download(spec.hf_path, &dest)?;
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
    if let Some(d) = std::env::var_os("LOKI_MODELS") {
        dirs.extend(std::env::split_paths(&d).filter(|p| !p.as_os_str().is_empty()));
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
