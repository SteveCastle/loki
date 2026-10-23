//! Video frames through ffmpeg/ffprobe (found next to the binary, in ./ffmpeg, or on PATH).
use crate::image::Rgb8;
use anyhow::{bail, ensure, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn find_tool(name: &str) -> Option<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
        dirs.push(d.clone());
        dirs.push(d.join("ffmpeg"));
        dirs.push(d.join("ffmpeg").join("bin"));
    }
    dirs.push(PathBuf::from("ffmpeg").join("bin"));
    for d in dirs {
        let p = d.join(exe_name(name));
        if p.is_file() {
            return Some(p);
        }
    }
    let ok = Command::new(exe_name(name)).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
    ok.then(|| PathBuf::from(exe_name(name)))
}

pub struct VideoInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
}

pub fn probe(path: &Path) -> Result<VideoInfo> {
    let ffprobe = find_tool("ffprobe").context("ffprobe not found (put ffmpeg/ffprobe next to loki-segment or on PATH)")?;
    let out = Command::new(ffprobe)
        .args(["-v", "error", "-select_streams", "v:0", "-print_format", "json", "-show_streams"])
        .arg(path)
        .output()
        .context("running ffprobe")?;
    ensure!(out.status.success(), "ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let s = &v["streams"][0];
    let rate = |s: &str| -> f64 {
        match s.split_once('/') {
            Some((a, b)) => a.parse::<f64>().unwrap_or(0.0) / b.parse::<f64>().unwrap_or(1.0).max(1e-9),
            None => s.parse::<f64>().unwrap_or(0.0),
        }
    };
    let (w, h) = (s["width"].as_u64().unwrap_or(0) as usize, s["height"].as_u64().unwrap_or(0) as usize);
    if w == 0 || h == 0 {
        bail!("{} has no video stream", path.display());
    }
    let mut fps = rate(s["avg_frame_rate"].as_str().unwrap_or("0"));
    if fps <= 0.0 {
        fps = rate(s["r_frame_rate"].as_str().unwrap_or("0"));
    }
    Ok(VideoInfo { width: w, height: h, fps })
}

/// Every frame of the first video stream at native size (no frame-rate conversion), as RGB.
pub fn decode_frames(path: &Path, info: &VideoInfo, max_frames: usize) -> Result<Vec<Rgb8>> {
    let ffmpeg = find_tool("ffmpeg").context("ffmpeg not found (put ffmpeg/ffprobe next to loki-segment or on PATH)")?;
    let mut cmd = Command::new(ffmpeg);
    cmd.args(["-v", "error", "-nostdin", "-i"]).arg(path).args(["-map", "0:v:0", "-an", "-fps_mode", "passthrough"]);
    if max_frames > 0 {
        cmd.args(["-frames:v", &max_frames.to_string()]);
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    let out = cmd.stderr(Stdio::piped()).stdout(Stdio::piped()).output().context("running ffmpeg")?;
    ensure!(out.status.success(), "ffmpeg failed decoding {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
    let fsz = info.width * info.height * 3;
    ensure!(fsz > 0 && out.stdout.len() % fsz == 0, "unexpected ffmpeg output size");
    Ok(out.stdout.chunks_exact(fsz).map(|c| Rgb8 { w: info.width, h: info.height, data: c.to_vec() }).collect())
}
