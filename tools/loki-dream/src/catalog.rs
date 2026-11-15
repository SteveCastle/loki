//! `--list-models`: the diffusion models this engine can run, for front ends that let a user pick one.
//!
//! Every `.safetensors` file directly in the model folders (`models::search_dirs`) whose tensors detect as a supported
//! architecture is listed (text encoders, VAEs, LoRAs and other architectures are skipped), with what a picker needs:
//! the architecture, the variant (distilled or not) and the settings it runs with by default, the stored weight format,
//! and whether the text encoder and VAE it needs are here or will be downloaded on first use. Each architecture's
//! default model is listed too when it is not present, as a download. Only safetensors headers are read.
//!
//! A display name comes from a `<stem>.civitai.info` file next to the model when there is one (ComfyUI model managers
//! write those), else from the file name.
use crate::arch::{self, Arch};
use crate::engines;
use crate::models::{self, Spec};
use crate::safetensors::SafeTensors;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// The architectures' default models, in the order they are offered.
fn default_specs() -> [(Arch, &'static Spec); 6] {
    [(Arch::Moss, &models::DIT_INT8), (Arch::Krea2, &models::KREA2_DIT), (Arch::Flux2, &models::KLEIN_DIT), (Arch::Anima, &models::ANIMA_DIT), (Arch::Sdxl, &models::SDXL_BASE), (Arch::H3Image, &models::H3_IMAGE_DIT)]
}

/// How a checkpoint stores its big weights: the dtype of most 2D weight bytes (int8 ConvRot, fp8, nvfp4, bf16, ...).
fn stored_format(st: &SafeTensors) -> &'static str {
    let mut by: Vec<(&str, usize)> = Vec::new();
    for t in st.tensors.values() {
        if t.shape.len() != 2 {
            continue;
        }
        let k = match t.dtype.as_str() {
            "I8" => "int8",
            "F8_E4M3" | "F8_E5M2" => "fp8",
            "U8" => "nvfp4",
            "BF16" => "bf16",
            "F16" => "fp16",
            "F32" => "fp32",
            _ => continue,
        };
        let n = t.data_offsets.1 - t.data_offsets.0;
        match by.iter_mut().find(|(d, _)| *d == k) {
            Some(e) => e.1 += n,
            None => by.push((k, n)),
        }
    }
    by.into_iter().max_by_key(|e| e.1).map(|e| e.0).unwrap_or("unknown")
}

/// A display name from the civitai.info next to the model ("<model name> <version name>"), if any.
fn civitai_label(path: &Path) -> Option<(String, Option<String>)> {
    let stem = path.file_stem()?.to_string_lossy().to_string();
    let info = path.with_file_name(format!("{stem}.civitai.info"));
    let v: Value = serde_json::from_slice(&std::fs::read(info).ok()?).ok()?;
    let model = v.get("model").and_then(|m| m.get("name")).and_then(|n| n.as_str()).map(str::trim).filter(|s| !s.is_empty());
    let version = v.get("name").and_then(|n| n.as_str()).map(str::trim).filter(|s| !s.is_empty());
    let label = match (model, version) {
        (Some(m), Some(ver)) if !m.to_lowercase().contains(&ver.to_lowercase()) => format!("{m} {ver}"),
        (Some(m), _) => m.to_string(),
        (None, Some(ver)) => ver.to_string(),
        (None, None) => return None,
    };
    let base = v.get("baseModel").and_then(|b| b.as_str()).map(str::to_string);
    Some((label, base))
}

fn gb(x: f32) -> f64 {
    ((x as f64) * 10.0).round() / 10.0
}

/// What the run needs beyond the checkpoint: a local file, or the size of the download that will fetch it.
fn need(spec: &Spec) -> Value {
    match models::find_local(spec) {
        Some(p) => json!({"file": p.file_name().map(|f| f.to_string_lossy().into_owned()), "present": true}),
        None => json!({"file": spec.hf_path.rsplit('/').next(), "present": false, "download_gb": gb(spec.size_gb), "from": spec.repo}),
    }
}

fn entry(path: &Path, a: Arch, st: Option<&SafeTensors>, local: bool, download: Option<&Spec>) -> Value {
    let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = file.strip_suffix(".safetensors").unwrap_or(&file).to_string();
    let distilled = engines::is_distilled(a, path);
    let (steps, cfg) = match a {
        Arch::Moss => (50, 3.0),
        _ => engines::defaults(a, distilled),
    };
    let civitai = if local { civitai_label(path) } else { None };
    let te = if local || a != Arch::Flux2 { engines::te_spec(a, path).ok().map(need) } else { Some(need(&models::KLEIN4B_TE)) };
    let mut e = json!({
        "file": file,
        "label": civitai.as_ref().map(|c| c.0.clone()).unwrap_or_else(|| stem.clone()),
        "architecture": a.id(),
        "architecture_name": a.name(),
        "distilled": distilled,
        "steps": steps,
        "cfg": cfg,
        // MOSS-Image: --turbo applies the DMD2 LoRA; the other architectures have distilled checkpoints instead
        "turbo": a == Arch::Moss,
        "loras": a == Arch::Moss,
        // reference pictures (--ref) the model takes: MiniMax H3 Image's <Picture n>
        "references": if a == Arch::H3Image { loki_reshoot::h3image::MAX_REFS } else { 0 },
        "local": local,
        "text_encoder": te,
        // SDXL: both inside the checkpoint (null)
        "vae": engines::vae_spec(a).map(need),
    });
    let o = e.as_object_mut().unwrap();
    if let Some(st) = st {
        o.insert("format".into(), json!(stored_format(st)));
    }
    if local {
        o.insert("path".into(), json!(path.display().to_string()));
        o.insert("size_gb".into(), json!(((std::fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f64 / 1e9) * 10.0).round() / 10.0));
        if let Some(base) = civitai.and_then(|c| c.1) {
            o.insert("base_model".into(), json!(base));
        }
    }
    if let Some(spec) = download {
        o.insert("download_gb".into(), json!(gb(spec.size_gb)));
        o.insert("from".into(), json!(spec.repo));
    }
    e
}

/// The catalog as one JSON object: `models` (local ones first, by architecture then name), `default` (the file a run
/// without --model uses) and `search_dirs`.
pub fn list() -> Value {
    let dirs = models::search_dirs();
    let mut seen: HashSet<String> = HashSet::new();
    let mut found: Vec<(Arch, String, Value)> = Vec::new();
    for d in &dirs {
        let Ok(rd) = std::fs::read_dir(d) else { continue };
        let mut paths: Vec<PathBuf> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_file() && p.extension().map(|x| x.eq_ignore_ascii_case("safetensors")).unwrap_or(false)).collect();
        paths.sort();
        for p in paths {
            let name = p.file_name().map(|f| f.to_string_lossy().to_lowercase()).unwrap_or_default();
            // a bare name resolves to the first folder that has it: later copies are unreachable by name
            if !seen.insert(name) {
                continue;
            }
            let Ok(st) = SafeTensors::open(&p) else { continue };
            let Ok(a) = arch::detect(&st) else { continue };
            let e = entry(&p, a, Some(&st), true, None);
            found.push((a, e["label"].as_str().unwrap_or_default().to_lowercase(), e));
            st.drop_in_background();
        }
    }
    let order = |a: Arch| default_specs().iter().position(|(x, _)| *x == a).unwrap_or(9);
    found.sort_by(|x, y| order(x.0).cmp(&order(y.0)).then(x.1.cmp(&y.1)));
    let mut out: Vec<Value> = found.into_iter().map(|f| f.2).collect();
    // each architecture's default model, when it is not here yet
    for (a, spec) in default_specs() {
        if models::find_local(spec).is_none() && !(a == Arch::Moss && models::find_local(&models::DIT_BF16).is_some()) {
            let file = spec.hf_path.rsplit('/').next().unwrap_or_default();
            out.push(entry(Path::new(file), a, None, false, Some(spec)));
        }
    }
    let default = std::env::var_os("LOKI_DREAM_DIT")
        .filter(|v| !v.is_empty())
        .map(|v| PathBuf::from(v).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())
        .or_else(|| models::find_local(&models::DIT_INT8).or_else(|| models::find_local(&models::DIT_BF16)).and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned())))
        .unwrap_or_else(|| models::DIT_INT8.hf_path.rsplit('/').next().unwrap_or_default().to_string());
    json!({"models": out, "default": default, "search_dirs": dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>()})
}

/// The catalog as a table for a terminal.
pub fn table(v: &Value) -> Vec<String> {
    let mut lines = vec![format!("{:<46} {:<16} {:<9} {:<6} {:>5} {:>4}  {}", "model (--model NAME)", "architecture", "variant", "format", "steps", "cfg", "status")];
    for m in v["models"].as_array().into_iter().flatten() {
        let missing: Vec<String> = ["text_encoder", "vae"]
            .iter()
            .filter(|k| m[**k]["present"] == false)
            .map(|k| format!("{} {} GB", if *k == "vae" { "VAE" } else { "text encoder" }, m[*k]["download_gb"]))
            .collect();
        let status = if m["local"] == false {
            format!("download {} GB{}", m["download_gb"], if missing.is_empty() { String::new() } else { format!(" + {}", missing.join(" + ")) })
        } else if missing.is_empty() {
            "ready".into()
        } else {
            format!("first run downloads {}", missing.join(" + "))
        };
        let def = if m["file"] == v["default"] { " (default)" } else { "" };
        lines.push(format!(
            "{:<46} {:<16} {:<9} {:<6} {:>5} {:>4}  {}{}",
            m["file"].as_str().unwrap_or_default().trim_end_matches(".safetensors"),
            m["architecture_name"].as_str().unwrap_or_default(),
            if m["distilled"] == true { "distilled" } else { "base" },
            m["format"].as_str().unwrap_or("-"),
            m["steps"],
            m["cfg"],
            status,
            def
        ));
    }
    lines
}
