//! Media I/O through ffmpeg/ffprobe: probing, decoding reference videos/audio, muxing the result.
//!
//! ffmpeg is discovered next to the binary, in `./ffmpeg`, then on PATH; on Windows it is downloaded
//! (a static "essentials" build) into `<binary dir>/ffmpeg` when it cannot be found.
use anyhow::{bail, ensure, Context, Result};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

pub const FPS: usize = 24;
pub const AUDIO_SR: usize = 32000;

#[derive(Clone, Debug)]
pub struct Ffmpeg {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
}

#[derive(Clone, Debug)]
pub struct MediaInfo {
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub duration: f64,
    pub has_video: bool,
    pub has_audio: bool,
}

fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

fn candidate_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())) {
        dirs.push(d.clone());
        dirs.push(d.join("ffmpeg"));
        dirs.push(d.join("ffmpeg").join("bin"));
    }
    dirs.push(PathBuf::from("."));
    dirs.push(PathBuf::from("ffmpeg"));
    dirs.push(PathBuf::from("ffmpeg").join("bin"));
    dirs
}

fn find_tool(name: &str) -> Option<PathBuf> {
    for d in candidate_dirs() {
        let p = d.join(exe_name(name));
        if p.is_file() {
            return Some(p);
        }
    }
    // PATH
    let ok = Command::new(exe_name(name)).arg("-version").stdout(Stdio::null()).stderr(Stdio::null()).status().map(|s| s.success()).unwrap_or(false);
    if ok {
        return Some(PathBuf::from(exe_name(name)));
    }
    None
}

impl Ffmpeg {
    /// `explicit` may point at the ffmpeg executable (ffprobe is looked up next to it).
    pub fn discover(explicit: Option<&Path>) -> Result<Ffmpeg> {
        if let Some(p) = explicit {
            ensure!(p.is_file(), "--ffmpeg {} does not exist", p.display());
            let probe = p.with_file_name(exe_name("ffprobe"));
            let ffprobe = if probe.is_file() { probe } else { find_tool("ffprobe").context("ffprobe not found next to --ffmpeg or on PATH")? };
            return Ok(Ffmpeg { ffmpeg: p.to_path_buf(), ffprobe });
        }
        if let (Some(a), Some(b)) = (find_tool("ffmpeg"), find_tool("ffprobe")) {
            return Ok(Ffmpeg { ffmpeg: a, ffprobe: b });
        }
        if cfg!(windows) {
            eprintln!("ffmpeg not found; downloading a static build (one time)...");
            return download_ffmpeg();
        }
        bail!("ffmpeg/ffprobe not found: install them (e.g. `apt install ffmpeg`) or pass --ffmpeg PATH");
    }

    pub fn probe(&self, path: &Path) -> Result<MediaInfo> {
        let out = Command::new(&self.ffprobe)
            .args(["-v", "error", "-print_format", "json", "-show_streams", "-show_format"])
            .arg(path)
            .output()
            .context("running ffprobe")?;
        ensure!(out.status.success(), "ffprobe failed on {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
        let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
        let streams = v["streams"].as_array().cloned().unwrap_or_default();
        let mut info = MediaInfo { width: 0, height: 0, fps: 0.0, duration: 0.0, has_video: false, has_audio: false };
        let parse_rate = |s: &str| -> f64 {
            match s.split_once('/') {
                Some((a, b)) => a.parse::<f64>().unwrap_or(0.0) / b.parse::<f64>().unwrap_or(1.0).max(1e-9),
                None => s.parse::<f64>().unwrap_or(0.0),
            }
        };
        for s in &streams {
            match s["codec_type"].as_str() {
                Some("video") if !info.has_video => {
                    // still-image "videos" (cover art) have a disposition flag; keep the first real video stream
                    info.has_video = true;
                    info.width = s["width"].as_u64().unwrap_or(0) as usize;
                    info.height = s["height"].as_u64().unwrap_or(0) as usize;
                    info.fps = parse_rate(s["avg_frame_rate"].as_str().unwrap_or("0"));
                    if let Some(d) = s["duration"].as_str().and_then(|d| d.parse::<f64>().ok()) {
                        info.duration = d;
                    }
                }
                Some("audio") => info.has_audio = true,
                _ => {}
            }
        }
        if let Some(d) = v["format"]["duration"].as_str().and_then(|d| d.parse::<f64>().ok()) {
            if info.duration <= 0.0 || d > info.duration {
                info.duration = d;
            }
        }
        Ok(info)
    }

    /// Decode a video segment to RGB24 frames at `FPS`, scaled (Lanczos, plain stretch) to `w`x`h`.
    /// Returns (frames, count). `max_frames` caps the output.
    pub fn decode_video(&self, path: &Path, start: f64, dur: Option<f64>, w: usize, h: usize, max_frames: usize) -> Result<(Vec<u8>, usize)> {
        let mut cmd = Command::new(&self.ffmpeg);
        cmd.args(["-v", "error", "-nostdin"]);
        if start > 0.0 {
            cmd.args(["-ss", &format!("{start}")]);
        }
        cmd.arg("-i").arg(path);
        if let Some(d) = dur {
            cmd.args(["-t", &format!("{d}")]);
        }
        cmd.args(["-an", "-vf", &format!("fps={FPS},scale={w}:{h}:flags=lanczos"), "-frames:v", &max_frames.to_string(), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
        let out = cmd.stderr(Stdio::piped()).stdout(Stdio::piped()).output().context("running ffmpeg")?;
        ensure!(out.status.success(), "ffmpeg failed decoding {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
        let fsz = w * h * 3;
        ensure!(out.stdout.len() % fsz == 0, "unexpected ffmpeg output size");
        let n = out.stdout.len() / fsz;
        Ok((out.stdout, n))
    }

    /// Decode (a segment of) the audio of `path` to planar stereo f32 at 32 kHz: `[2, n]`. Mono is duplicated.
    /// `pad_to` (samples) zero-pads/truncates to an exact length when given.
    pub fn decode_audio(&self, path: &Path, start: f64, dur: Option<f64>, pad_to: Option<usize>) -> Result<(Vec<f32>, usize)> {
        let mut cmd = Command::new(&self.ffmpeg);
        cmd.args(["-v", "error", "-nostdin"]);
        if start > 0.0 {
            cmd.args(["-ss", &format!("{start}")]);
        }
        cmd.arg("-i").arg(path);
        if let Some(d) = dur {
            cmd.args(["-t", &format!("{d}")]);
        }
        cmd.args(["-vn", "-ac", "2", "-ar", &AUDIO_SR.to_string(), "-f", "f32le", "-"]);
        let out = cmd.stderr(Stdio::piped()).stdout(Stdio::piped()).output().context("running ffmpeg")?;
        ensure!(out.status.success(), "ffmpeg failed decoding audio of {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
        let inter: Vec<f32> = out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut n = inter.len() / 2;
        if let Some(p) = pad_to {
            n = p;
        }
        let mut planar = vec![0f32; 2 * n];
        for i in 0..(inter.len() / 2).min(n) {
            planar[i] = inter[2 * i];
            planar[n + i] = inter[2 * i + 1];
        }
        Ok((planar, n))
    }

    /// First frame of an image/animation that the `image` crate cannot read (gif, avif, ...), as RGB24.
    pub fn decode_image(&self, path: &Path) -> Result<(Vec<u8>, usize, usize)> {
        let info = self.probe(path)?;
        ensure!(info.has_video && info.width > 0, "{} has no image/video stream", path.display());
        let out = Command::new(&self.ffmpeg)
            .args(["-v", "error", "-nostdin", "-i"])
            .arg(path)
            .args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .context("running ffmpeg")?;
        ensure!(out.status.success() && out.stdout.len() == info.width * info.height * 3, "ffmpeg could not decode {}", path.display());
        Ok((out.stdout, info.width, info.height))
    }
}

/// Streaming mp4 writer: raw RGB24 frames in, H.264 (+ AAC from a wav file) out.
pub struct VideoWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    pub frames: usize,
    frame_bytes: usize,
}

impl VideoWriter {
    pub fn new(ff: &Ffmpeg, out: &Path, w: usize, h: usize, wav: Option<&Path>, crf: u32) -> Result<VideoWriter> {
        let mut cmd = Command::new(&ff.ffmpeg);
        cmd.args(["-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{w}x{h}"), "-r", &FPS.to_string(), "-i", "-"]);
        if let Some(a) = wav {
            cmd.arg("-i").arg(a);
        }
        cmd.args(["-c:v", "libx264", "-preset", "medium", "-crf", &crf.to_string(), "-pix_fmt", "yuv420p"]);
        if wav.is_some() {
            cmd.args(["-c:a", "aac", "-b:a", "192k"]);
        }
        cmd.args(["-movflags", "+faststart"]).arg(out);
        let mut child = cmd.stdin(Stdio::piped()).stderr(Stdio::inherit()).spawn().context("starting ffmpeg (encoder)")?;
        let stdin = child.stdin.take();
        Ok(VideoWriter { child, stdin, frames: 0, frame_bytes: w * h * 3 })
    }
    pub fn write(&mut self, rgb: &[u8]) -> Result<()> {
        ensure!(rgb.len() % self.frame_bytes == 0, "partial frame");
        self.stdin.as_mut().context("writer closed")?.write_all(rgb).context("ffmpeg encoder closed its input (see its error above)")?;
        self.frames += rgb.len() / self.frame_bytes;
        Ok(())
    }
    pub fn finish(mut self) -> Result<()> {
        drop(self.stdin.take());
        let st = self.child.wait()?;
        ensure!(st.success(), "ffmpeg encoder failed ({st})");
        Ok(())
    }
}

/// Write planar stereo f32 `[2, n]` as an interleaved 32-bit-float wav.
pub fn write_wav_f32(path: &Path, planar: &[f32], n: usize, sr: usize) -> Result<()> {
    ensure!(planar.len() == 2 * n);
    let data_len = n * 2 * 4;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&((36 + data_len) as u32).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&3u16.to_le_bytes())?; // IEEE float
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&(sr as u32).to_le_bytes())?;
    f.write_all(&((sr * 8) as u32).to_le_bytes())?;
    f.write_all(&8u16.to_le_bytes())?;
    f.write_all(&32u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&(data_len as u32).to_le_bytes())?;
    for i in 0..n {
        f.write_all(&planar[i].to_le_bytes())?;
        f.write_all(&planar[n + i].to_le_bytes())?;
    }
    f.flush()?;
    Ok(())
}

#[cfg(windows)]
fn download_ffmpeg() -> Result<Ffmpeg> {
    let base = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.to_path_buf())).unwrap_or_else(|| PathBuf::from("."));
    let dir = base.join("ffmpeg");
    std::fs::create_dir_all(&dir)?;
    let url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
    let resp = ureq::get(url).call().map_err(|e| anyhow::anyhow!("downloading {url}: {e}"))?;
    let mut zip_bytes = Vec::new();
    resp.into_reader().read_to_end(&mut zip_bytes)?;
    let mut ar = zip::ZipArchive::new(std::io::Cursor::new(zip_bytes))?;
    for name in ["ffmpeg.exe", "ffprobe.exe"] {
        let mut found = false;
        for i in 0..ar.len() {
            let mut f = ar.by_index(i)?;
            if f.name().ends_with(&format!("/bin/{name}")) {
                let mut out = std::fs::File::create(dir.join(name))?;
                std::io::copy(&mut f, &mut out)?;
                found = true;
                break;
            }
        }
        ensure!(found, "{name} not found in the downloaded archive");
    }
    Ok(Ffmpeg { ffmpeg: dir.join("ffmpeg.exe"), ffprobe: dir.join("ffprobe.exe") })
}

#[cfg(not(windows))]
fn download_ffmpeg() -> Result<Ffmpeg> {
    bail!("automatic ffmpeg download is only implemented on Windows")
}
