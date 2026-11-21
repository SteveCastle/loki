//! Model files: GGUF language models and their vision projectors (mmproj), found locally or downloaded.
//!
//! A model is named by (first match wins):
//!   * a path to a `.gguf` file;
//!   * a Hugging Face style `publisher/repo/file.gguf` (also LM Studio's on-disk layout): looked up as that relative
//!     path in every model folder, else downloaded from huggingface.co/publisher/repo;
//!   * `publisher/repo`: the repo's Q4_K_M file (or its only model file), locally or downloaded;
//!   * a bare name such as `qwen3.8-27b-uncensored`: the model folder or file whose name contains it (case-insensitive).
//! Model folders: next to the binary, its `models/`, `$LOKI_MODELS` (a path list), `./models`, and LM Studio's
//! (`~/.lmstudio/models`, `~/.cache/lm-studio/models`).
//! The vision projector is `--mmproj`, else the `mmproj*.gguf` beside the model (F16 preferred).
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// The default model: Qwen3.8 27B, Q4_K_M with its BF16 vision projector.
pub const DEFAULT_MODEL: &str = "lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf";

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| PathBuf::from("."))
}

fn home() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).filter(|h| !h.is_empty()).map(PathBuf::from)
}

/// Folders searched for models, in order.
pub fn search_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let exe = exe_dir();
    dirs.push(exe.join("models"));
    dirs.push(exe.clone());
    if let Some(d) = std::env::var_os("LOKI_MODELS") {
        dirs.extend(std::env::split_paths(&d).filter(|p| !p.as_os_str().is_empty()));
    }
    dirs.push(PathBuf::from("models"));
    if let Some(h) = home() {
        dirs.push(h.join(".lmstudio").join("models"));
        dirs.push(h.join(".cache").join("lm-studio").join("models"));
    }
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| seen.insert(d.clone()));
    dirs
}

/// Where missing files are downloaded: `$LOKI_MODELS_DOWNLOAD`, else `<binary dir>/models`.
fn download_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("LOKI_MODELS_DOWNLOAD").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    exe_dir().join("models")
}

fn is_gguf(p: &Path) -> bool {
    p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false)
}
fn is_mmproj(p: &Path) -> bool {
    p.file_name().map(|f| f.to_string_lossy().to_lowercase().contains("mmproj")).unwrap_or(false)
}

fn norm(s: &str) -> String {
    s.to_lowercase().chars().filter(|c| c.is_alphanumeric() || *c == '.').collect()
}

/// The language-model GGUF in a folder: Q4_K_M if several, else the largest (never an mmproj or a split part > 1).
fn pick_in_dir(dir: &Path) -> Option<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && is_gguf(p) && !is_mmproj(p)).collect();
    files.sort();
    if let Some(q) = files.iter().find(|p| p.file_name().unwrap().to_string_lossy().to_lowercase().contains("q4_k_m")) {
        return Some(q.clone());
    }
    files.into_iter().max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
}

/// Model folders (or files) up to three levels deep under the search dirs whose name contains `name`.
fn find_by_name(name: &str) -> Option<PathBuf> {
    let want = norm(name);
    if want.is_empty() {
        return None;
    }
    let mut best: Option<(usize, PathBuf)> = None;
    let visit = |p: &Path, best: &mut Option<(usize, PathBuf)>| {
        let f = norm(&p.file_name().unwrap_or_default().to_string_lossy());
        if f.contains(&want) {
            let cand = if p.is_dir() { pick_in_dir(p) } else if is_gguf(p) && !is_mmproj(p) { Some(p.to_path_buf()) } else { None };
            if let Some(c) = cand {
                // the closest name wins (shortest extra text)
                let extra = f.len() - want.len();
                if best.as_ref().map(|b: &(usize, PathBuf)| extra < b.0).unwrap_or(true) {
                    *best = Some((extra, c));
                }
            }
        }
    };
    for d in search_dirs() {
        let Ok(l1) = std::fs::read_dir(&d) else { continue };
        for e1 in l1.filter_map(|e| e.ok()) {
            let p1 = e1.path();
            visit(&p1, &mut best);
            if !p1.is_dir() {
                continue;
            }
            let Ok(l2) = std::fs::read_dir(&p1) else { continue };
            for e2 in l2.filter_map(|e| e.ok()) {
                let p2 = e2.path();
                visit(&p2, &mut best);
                if p2.is_dir() {
                    if let Ok(l3) = std::fs::read_dir(&p2) {
                        for e3 in l3.filter_map(|e| e.ok()) {
                            visit(&e3.path(), &mut best);
                        }
                    }
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    best.map(|b| b.1)
}

/// Resolve the language model.
pub fn resolve_model(spec: Option<&str>) -> Result<PathBuf> {
    let spec = spec.map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).or_else(|| std::env::var("LOKI_SCRIBE_MODEL").ok().filter(|s| !s.trim().is_empty())).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let p = Path::new(&spec);
    if p.is_file() {
        return Ok(p.to_path_buf());
    }
    if p.is_dir() {
        return pick_in_dir(p).with_context(|| format!("no model .gguf in {}", p.display()));
    }
    let parts: Vec<&str> = spec.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
    if parts.len() >= 2 && !p.is_absolute() {
        for d in search_dirs() {
            let cand = d.join(&spec);
            if cand.is_file() {
                return Ok(cand);
            }
            // a managed copy keeps the files flat in its folder ($LOKI_MODELS)
            if parts.len() > 2 && d.join(parts[parts.len() - 1]).is_file() {
                return Ok(d.join(parts[parts.len() - 1]));
            }
            if cand.is_dir() {
                if let Some(f) = pick_in_dir(&cand) {
                    return Ok(f);
                }
            }
        }
        let repo = format!("{}/{}", parts[0], parts[1]);
        let file = if parts.len() > 2 { parts[2..].join("/") } else { pick_remote(&repo, false)? };
        let dest = download_dir().join(parts[0]).join(parts[1]).join(&file);
        if !dest.is_file() {
            crate::info!("model {spec} not found locally; downloading {repo}/{file} from Hugging Face to {}", dest.display());
            download(&format!("https://huggingface.co/{repo}/resolve/main/{file}"), &dest)?;
        }
        return Ok(dest);
    }
    if let Some(f) = find_by_name(&spec) {
        return Ok(f);
    }
    bail!(
        "model {spec:?} not found (searched {}); pass a .gguf path or a Hugging Face publisher/repo[/file.gguf]",
        search_dirs().iter().map(|d| d.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// The vision projector for `model`: explicit, else beside the model, else (for a downloaded model) from its repo.
/// None: a text-only model.
pub fn resolve_mmproj(explicit: Option<&str>, model: &Path) -> Result<Option<PathBuf>> {
    if let Some(e) = explicit.map(str::trim).filter(|s| !s.is_empty()) {
        if e.eq_ignore_ascii_case("none") {
            return Ok(None);
        }
        let p = Path::new(e);
        if p.is_file() {
            return Ok(Some(p.to_path_buf()));
        }
        let parts: Vec<&str> = e.split(['/', '\\']).filter(|s| !s.is_empty()).collect();
        if parts.len() >= 3 && !p.is_absolute() {
            for d in search_dirs() {
                if d.join(e).is_file() {
                    return Ok(Some(d.join(e)));
                }
            }
            let dest = download_dir().join(e);
            download(&format!("https://huggingface.co/{}/{}/resolve/main/{}", parts[0], parts[1], parts[2..].join("/")), &dest)?;
            return Ok(Some(dest));
        }
        bail!("--mmproj {e}: not found");
    }
    let dir = model.parent().unwrap_or(Path::new("."));
    let mut cands: Vec<PathBuf> = std::fs::read_dir(dir).map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && is_gguf(p) && is_mmproj(p)).collect()).unwrap_or_default();
    cands.sort();
    let rank = |p: &PathBuf| {
        let f = p.file_name().unwrap().to_string_lossy().to_lowercase();
        if f.contains("f16") && !f.contains("bf16") {
            0
        } else if f.contains("bf16") {
            1
        } else if f.contains("f32") {
            2
        } else {
            3
        }
    };
    if let Some(p) = cands.iter().min_by_key(|p| rank(p)) {
        return Ok(Some(p.clone()));
    }
    // a model downloaded into <download>/<publisher>/<repo>/: fetch the repo's projector too
    if let (Some(repo_dir), Ok(dl)) = (model.parent(), download_dir().canonicalize()) {
        if let Ok(rd) = repo_dir.canonicalize() {
            if let Ok(rel) = rd.strip_prefix(&dl) {
                let parts: Vec<String> = rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
                if parts.len() == 2 {
                    let repo = format!("{}/{}", parts[0], parts[1]);
                    if let Ok(f) = pick_remote(&repo, true) {
                        let dest = repo_dir.join(&f);
                        crate::info!("downloading the vision projector {repo}/{f}");
                        download(&format!("https://huggingface.co/{repo}/resolve/main/{f}"), &dest)?;
                        return Ok(Some(dest));
                    }
                }
            }
        }
    }
    Ok(None)
}

/// A file of a Hugging Face repo: the Q4_K_M model (`mmproj` false) or the F16/BF16 projector.
fn pick_remote(repo: &str, mmproj: bool) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let resp = ureq::get(&url).call().with_context(|| format!("listing {repo} on Hugging Face"))?;
    let body: serde_json::Value = serde_json::from_reader(resp.into_reader())?;
    let files: Vec<String> = body.get("siblings").and_then(|s| s.as_array()).map(|a| a.iter().filter_map(|f| f.get("rfilename").and_then(|v| v.as_str()).map(str::to_string)).collect()).unwrap_or_default();
    let ggufs: Vec<&String> = files.iter().filter(|f| f.to_lowercase().ends_with(".gguf") && f.to_lowercase().contains("mmproj") == mmproj).collect();
    let pref: &[&str] = if mmproj { &["f16", "bf16", "f32"] } else { &["q4_k_m", "q5_k_m", "q6_k", "q8_0"] };
    for p in pref {
        if let Some(f) = ggufs.iter().find(|f| {
            let l = f.to_lowercase();
            l.contains(p) && !(mmproj && *p == "f16" && l.contains("bf16"))
        }) {
            return Ok((*f).clone());
        }
    }
    ggufs.first().map(|f| (*f).clone()).with_context(|| format!("{repo} has no {} .gguf", if mmproj { "mmproj" } else { "model" }))
}

/// Download `url` to `dest`, resuming a partial `.part` file; one process at a time per file (a `.lock` beside it).
pub fn download(url: &str, dest: &Path) -> Result<()> {
    use std::io::{Read, Write};
    if let Some(dir) = dest.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let part = PathBuf::from(format!("{}.part", dest.display()));
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
    let mut req = ureq::get(url);
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
    let mut file = std::fs::OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(&part)?;
    let mut reader = resp.into_reader();
    let (mut got, mut last_pct) = (start, u64::MAX);
    let mut buf = vec![0u8; 1 << 20];
    let name = dest.file_name().unwrap_or_default().to_string_lossy().into_owned();
    loop {
        let n = reader.read(&mut buf).with_context(|| format!("downloading {url} (re-run to resume)"))?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        got += n as u64;
        if let Some(t) = total.filter(|t| *t > 0) {
            let pct = got * 100 / t;
            if last_pct == u64::MAX || pct >= last_pct + 5 || got == t {
                last_pct = pct - pct % 5;
                // announced even with -q: large downloads are never silent (CONVENTIONS.md)
                eprintln!("  downloading {name}: {pct}% ({:.1}/{:.1} GB)", got as f64 / 1e9, t as f64 / 1e9);
            }
        }
    }
    file.flush()?;
    drop(file);
    if let Some(t) = total {
        let got = std::fs::metadata(&part)?.len();
        if got != t {
            bail!("download of {url} was truncated ({got}/{t} bytes); re-run to resume");
        }
    }
    std::fs::rename(&part, dest)?;
    Ok(())
}
