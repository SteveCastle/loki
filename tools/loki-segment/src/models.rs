//! Model file discovery (next to the binary / ./models) and resumable download from Hugging Face.
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Comfy-Org's ungated fp16 repackaging of the official checkpoint (facebook/sam3.1 `sam3.1_multiplex.pt`, gated).
const HF_BASE: &str = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/";

pub struct Spec {
    pub what: &'static str,
    pub patterns: &'static [&'static str],
    pub hf_path: &'static str,
}

pub const SAM31: Spec = Spec { what: "SAM 3.1 model", patterns: &["sam3.1*multiplex", "sam3.1"], hf_path: "checkpoints/sam3.1_multiplex_fp16.safetensors" };

/// Expected size of the published file (a quick sanity check against truncated copies).
pub const SAM31_BYTES: u64 = 1_745_546_848;

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
    let pb = if crate::log::quiet() { indicatif::ProgressBar::hidden() } else { indicatif::ProgressBar::new(total.unwrap_or(0)) };
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
    let base = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| PathBuf::from("."));
    let dest = base.join("models").join(spec.hf_path.rsplit('/').next().unwrap());
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
    if let Some(list) = std::env::var_os("LOKI_MODELS") {
        dirs.extend(std::env::split_paths(&list));
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
