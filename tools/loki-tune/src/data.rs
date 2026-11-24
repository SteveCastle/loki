//! The training set: which images, their captions, and the size each one is trained at.
//!
//! Inputs are image files, folders (searched recursively for images) and list files: a `.txt` / `.lst` given on the
//! command line holds one path per line, a `.jsonl` one `{"path": ..., "caption": ...}` object per line (how a caller
//! such as the media server hands over its own captions), and `-` reads such a list from stdin. Relative paths in a
//! list are relative to the list file.
//!
//! Captions, first found wins: the list's `caption`, a sidecar text file (`photo.txt` or `photo.jpg.txt`, kohya's
//! convention), then (with `--captions auto`) a description written by loki-scribe, cached in the run folder.
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::io::BufRead;
use std::path::{Path, PathBuf};

pub const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "jfif"];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Item {
    pub path: PathBuf,
    /// the caption as found or written (without the trigger)
    pub caption: String,
    /// a one-sentence caption next to the long one (loki-scribe captions): each step trains on one of the two
    #[serde(default)]
    pub short: Option<String>,
    /// where it came from: list | sidecar | scribe | none
    pub caption_source: String,
    /// source size
    pub width: usize,
    pub height: usize,
    /// training size (multiples of 16)
    pub bw: usize,
    pub bh: usize,
}

pub fn is_image(p: &Path) -> bool {
    p.extension().and_then(|e| e.to_str()).map(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false)
}

#[derive(Deserialize)]
struct ListLine {
    path: PathBuf,
    #[serde(default)]
    caption: Option<String>,
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out)?;
        } else if is_image(&p) {
            out.push(p);
        }
    }
    Ok(())
}

fn read_list(text: &str, base: &Path, jsonl: bool, out: &mut Vec<(PathBuf, Option<String>)>) -> Result<()> {
    for (i, line) in text.lines().enumerate() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (p, cap) = if jsonl || line.starts_with('{') {
            let l: ListLine = serde_json::from_str(line).with_context(|| format!("list line {}: expected {{\"path\": ..., \"caption\": ...}}", i + 1))?;
            (l.path, l.caption)
        } else {
            (PathBuf::from(line.trim_matches('"')), None)
        };
        let p = if p.is_relative() { base.join(p) } else { p };
        out.push((p, cap));
    }
    Ok(())
}

/// Expand the command-line inputs to (image, caption from a list) pairs, deduplicated, in the order given.
pub fn collect(inputs: &[String]) -> Result<Vec<(PathBuf, Option<String>)>> {
    let mut found: Vec<(PathBuf, Option<String>)> = Vec::new();
    for inp in inputs {
        if inp == "-" {
            let mut s = String::new();
            for l in std::io::stdin().lock().lines() {
                s.push_str(&l?);
                s.push('\n');
            }
            read_list(&s, &std::env::current_dir()?, false, &mut found)?;
            continue;
        }
        let p = PathBuf::from(inp);
        if p.is_dir() {
            let mut v = Vec::new();
            walk(&p, &mut v)?;
            ensure!(!v.is_empty(), "{}: no images ({}) in this folder", p.display(), IMAGE_EXTS.join(", "));
            found.extend(v.into_iter().map(|x| (x, None)));
        } else if p.is_file() && is_image(&p) {
            found.push((p, None));
        } else if p.is_file() {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            ensure!(matches!(ext.as_str(), "txt" | "lst" | "jsonl" | "json"), "{}: not an image, a folder or a list file (.txt / .lst / .jsonl)", p.display());
            let text = std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            read_list(&text, p.parent().unwrap_or(Path::new(".")), ext == "jsonl" || ext == "json", &mut found)?;
        } else {
            bail!("{}: no such file or folder", p.display());
        }
    }
    let mut seen = HashSet::new();
    found.retain(|(p, _)| seen.insert(p.canonicalize().unwrap_or_else(|_| p.clone())));
    for (p, _) in &found {
        ensure!(p.is_file(), "{}: no such image", p.display());
        ensure!(is_image(p), "{}: not a supported image ({})", p.display(), IMAGE_EXTS.join(", "));
    }
    ensure!(!found.is_empty(), "no training images given");
    Ok(found)
}

/// A sidecar caption next to an image: `name.txt`, else `name.jpg.txt`.
pub fn sidecar(p: &Path) -> Option<String> {
    for c in [p.with_extension("txt"), PathBuf::from(format!("{}.txt", p.display()))] {
        if let Ok(s) = std::fs::read_to_string(&c) {
            let s = s.trim().trim_start_matches('\u{feff}').trim().to_string();
            if !s.is_empty() {
                return Some(s);
            }
        }
    }
    None
}

/// Training size for an image: about `area` pixels (never more than the image has), its aspect ratio, sides multiples
/// of 16 (one latent token per 16x16 pixels), at least 256.
pub fn bucket(w: usize, h: usize, area: usize) -> (usize, usize) {
    let a = area.min(w * h) as f64;
    let ar = w as f64 / h as f64;
    let bw = ((a * ar).sqrt() / 16.0).floor() as usize * 16;
    let bh = ((a / ar).sqrt() / 16.0).floor() as usize * 16;
    (bw.max(256), bh.max(256))
}

/// Decode an image upright (EXIF orientation applied), transparency composited over white.
pub fn load_rgb(p: &Path) -> Result<loki_dream::image::Rgb8> {
    use image::ImageDecoder;
    let mut dec = image::ImageReader::open(p).with_context(|| format!("opening {}", p.display()))?.with_guessed_format()?.into_decoder().with_context(|| format!("decoding {}", p.display()))?;
    let orient = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(dec).with_context(|| format!("decoding {}", p.display()))?;
    img.apply_orientation(orient);
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width() as usize, rgba.height() as usize);
    let mut data = Vec::with_capacity(w * h * 3);
    for px in rgba.pixels() {
        let a = px[3] as u32;
        for c in 0..3 {
            data.push(((px[c] as u32 * a + 255 * (255 - a) + 127) / 255) as u8);
        }
    }
    Ok(loki_dream::image::Rgb8 { w, h, data })
}

/// Only the dimensions (upright), without decoding the pixels where the format allows.
pub fn dims(p: &Path) -> Result<(usize, usize)> {
    use image::ImageDecoder;
    let mut dec = image::ImageReader::open(p).with_context(|| format!("opening {}", p.display()))?.with_guessed_format()?.into_decoder().with_context(|| format!("reading {}", p.display()))?;
    let (w, h) = dec.dimensions();
    let o = dec.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    use image::metadata::Orientation::*;
    Ok(match o {
        Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH => (h as usize, w as usize),
        _ => (w as usize, h as usize),
    })
}

/// Scale to cover (bw, bh) with Lanczos, then crop the centre.
pub fn fit(img: &loki_dream::image::Rgb8, bw: usize, bh: usize) -> loki_dream::image::Rgb8 {
    let s = (bw as f64 / img.w as f64).max(bh as f64 / img.h as f64);
    let (nw, nh) = (((img.w as f64 * s).round() as usize).max(bw), ((img.h as f64 * s).round() as usize).max(bh));
    let r = if (nw, nh) == (img.w, img.h) { img.clone() } else { img.resize_lanczos(nw, nh) };
    let (x0, y0) = ((nw - bw) / 2, (nh - bh) / 2);
    let mut data = Vec::with_capacity(bw * bh * 3);
    for y in 0..bh {
        let row = ((y0 + y) * nw + x0) * 3;
        data.extend_from_slice(&r.data[row..row + bw * 3]);
    }
    loki_dream::image::Rgb8 { w: bw, h: bh, data }
}

pub fn mirror(img: &loki_dream::image::Rgb8) -> loki_dream::image::Rgb8 {
    let mut data = vec![0u8; img.data.len()];
    for y in 0..img.h {
        for x in 0..img.w {
            let (s, d) = ((y * img.w + x) * 3, (y * img.w + img.w - 1 - x) * 3);
            data[d..d + 3].copy_from_slice(&img.data[s..s + 3]);
        }
    }
    loki_dream::image::Rgb8 { w: img.w, h: img.h, data }
}

/// The text a caption is trained with: the trigger first (unless the caption already names it).
pub fn with_trigger(caption: &str, trigger: Option<&str>) -> String {
    match trigger.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) if caption.contains(t) => caption.to_string(),
        Some(t) if caption.is_empty() => t.to_string(),
        Some(t) => format!("{t}, {caption}"),
        None => caption.to_string(),
    }
}

/// What a LoRA teaches, which decides how its images are captioned. Community practice: the caption describes what
/// should stay promptable, the trigger carries what the LoRA should learn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum Kind {
    /// a person or character: identity left to the trigger, everything else described
    Character,
    /// a look (medium, technique, palette, rendering): content described, the style never named
    Style,
    /// a product or object: its design left to the trigger, its surroundings described
    Object,
    /// a pose, action, effect or composition: named only by the trigger
    Concept,
    /// everything described plainly
    General,
}

/// How the captions name the subject: `zkx, a man` on first mention, then `zkx` (the trigger alone when no class).
fn naming(trigger: Option<&str>, class: Option<&str>, what: &str) -> String {
    let t = trigger.map(str::trim).filter(|t| !t.is_empty());
    let c = class.map(str::trim).filter(|c| !c.is_empty());
    match (t, c) {
        (Some(t), Some(c)) => format!("Call the {what} \"{t}\": write \"{t}, a {c},\" the first time and \"{t}\" (or a pronoun) after that"),
        (Some(t), None) => format!("Call the {what} \"{t}\" (or a pronoun after the first mention)"),
        (None, Some(c)) => format!("Call the {what} \"the {c}\""),
        (None, None) => format!("Call the {what} plainly by what it is"),
    }
}

/// The loki-scribe instruction for a kind. `traits`: what every image of the set shares (found by `scribe_traits`),
/// which the captions must leave to the trigger. Captions come back as JSON {"long", "short"}: MOSS-Image is trained
/// on several caption lengths per image (its user guide's long / medium / short captions), and a mix of caption
/// lengths trains a LoRA that answers both detailed and terse prompts.
pub fn caption_prompt(kind: Kind, trigger: Option<&str>, class: Option<&str>, traits: Option<&str>) -> String {
    let t = trigger.map(str::trim).filter(|t| !t.is_empty());
    let shared = traits.map(str::trim).filter(|s| !s.is_empty());
    let rule = match kind {
        Kind::Character => format!(
            "This image is one of a set that teaches an image model a specific person or character. {}. Do NOT describe who they are: no face \
shape, facial features, eye colour, skin tone, ethnicity, age, body type, hair or beard{}; the name carries all of that. DO describe everything \
that changes between pictures: clothing and accessories, pose, gesture, facial expression, what they are doing, other people, the setting and \
background, lighting, camera angle, framing and distance, and the medium (photo, illustration, render...).",
            naming(t, class, "person"),
            shared.map(|s| format!(", and none of these traits they always have: {s}")).unwrap_or_default()
        ),
        Kind::Style => format!(
            "This image is one of a set that teaches an image model an art or photographic style{}. Describe the CONTENT completely and plainly: \
the subjects, their appearance and clothing, poses, objects, the setting, the composition and framing. Do NOT describe the style in any way: no \
medium, technique, brushwork, line quality, rendering, colour grading, palette, mood words, artist names, or words like \"stylized\", \
\"painterly\" or \"cinematic\"{}; the style is what the model must learn without being told.",
            t.map(|t| format!(" called \"{t}\" (start the caption with \"{t}\")")).unwrap_or_default(),
            shared.map(|s| format!(", and none of these traits of the style: {s}")).unwrap_or_default()
        ),
        Kind::Object => format!(
            "This image is one of a set that teaches an image model a specific object or product. {}. Do NOT describe its own design: no shape, \
materials, colours, markings, logos or labels that belong to it{}; the name carries all of that. DO describe everything around it: where it is \
and how it is placed, the camera angle and distance, hands or people using it, other objects, the surface, background, lighting and the medium \
(photo, render...).",
            naming(t, class, "object"),
            shared.map(|s| format!(", and none of these traits it always has: {s}")).unwrap_or_default()
        ),
        Kind::Concept => format!(
            "This image is one of a set that teaches an image model a concept (a pose, action, effect or composition). {}, without describing \
the concept in other words{}. Describe everything else plainly: who or what is in the image, their appearance and clothing, the setting, \
lighting, camera angle and the medium.",
            naming(t, class, "concept"),
            shared.map(|s| format!(" (it is: {s})")).unwrap_or_default()
        ),
        Kind::General => "Describe the image completely and plainly: the main subject and what they look like, clothing, pose and expression, \
what is happening, the setting and background, lighting, camera angle and framing, and the medium or style (photo, illustration, render...)."
            .to_string(),
    };
    format!(
        "Write training captions for this image for a text-to-image model. {rule} Transcribe any visible text exactly, in quotes. If there \
is a signature, watermark, logo or artist name, say so and where it is (for example \"a small signature in the lower right corner\"), so it is \
not learned as part of the picture. Use plain, \
specific natural language: no preamble, no opinions, no markdown, no lists. Answer with only a JSON object: \
{{\"long\": \"one detailed paragraph\", \"short\": \"one short sentence\"}}."
    )
}

/// The instruction that finds what a set's images share (the traits the captions leave to the trigger).
pub fn traits_prompt(kind: Kind) -> Option<&'static str> {
    Some(match kind {
        Kind::Character => "These pictures all show the same person or character. List the visual traits that are the same in every picture and \
identify them: face, hair, beard, eye colour, skin, build, age, and any glasses, scars, tattoos, jewellery or clothing item they wear in \
every picture. Ignore anything that changes between pictures. Answer with one line: a comma-separated list of short phrases, nothing else.",
        Kind::Style => "These pictures all share one visual style. List what makes up the style, the same in every picture: medium, technique, \
line and brushwork, rendering, colour palette and grading, lighting habits, texture, framing habits. Ignore the subjects and content. Answer \
with one line: a comma-separated list of short phrases, nothing else.",
        Kind::Object => "These pictures all show the same object or product. List the traits of the object itself that are the same in every \
picture: shape, materials, colours, markings, logos, labels, parts. Ignore its surroundings. Answer with one line: a comma-separated list of \
short phrases, nothing else.",
        Kind::Concept | Kind::General => return None,
    })
}

/// What the images of a set share, from up to 8 of them seen together by loki-scribe.
pub fn scribe_traits(paths: &[PathBuf], prompt: &str, quiet: bool) -> Result<String> {
    let exe = std::env::var("LOKI_SCRIBE").unwrap_or_else(|_| "loki-scribe".to_string());
    let step = paths.len().div_ceil(8).max(1);
    let pick: Vec<&PathBuf> = paths.iter().step_by(step).take(8).collect();
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(["--json", "--together", "--temp", "0.2", "--dry-multiplier", "0.8", "--max-tokens", "300", "--max-image-tokens", "768", "-p", prompt]);
    if quiet {
        cmd.arg("-q");
    }
    cmd.args(&pick);
    cmd.stdout(std::process::Stdio::piped()).stderr(if quiet { std::process::Stdio::null() } else { std::process::Stdio::inherit() });
    let res = cmd.output().with_context(|| format!("running {exe} to find the traits the images share"))?;
    ensure!(res.status.success(), "{exe} failed (exit {})", res.status);
    let out = String::from_utf8_lossy(&res.stdout);
    let text = out.lines().filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok()).find_map(|v| v["text"].as_str().map(str::to_string)).context("loki-scribe gave no answer")?;
    Ok(text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").trim_matches(['.', '"']).to_string())
}

/// A path without Windows' verbatim prefix (`\\?\C:\...` from canonicalize).
pub fn plain_path(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") => PathBuf::from(rest),
        _ => p,
    }
}

/// Preview prompts: one training caption (does it reproduce?) and one new scene (does it generalize?).
pub fn default_samples(kind: Kind, trigger: Option<&str>, class: Option<&str>, first: Option<&str>) -> Vec<String> {
    let t = trigger.map(str::trim).filter(|t| !t.is_empty());
    let who = match (t, class.map(str::trim).filter(|c| !c.is_empty())) {
        (Some(t), Some(c)) => format!("{t}, a {c},"),
        (Some(t), None) => t.to_string(),
        (None, Some(c)) => format!("the {c}"),
        (None, None) => String::new(),
    };
    let novel = match kind {
        Kind::Character if !who.is_empty() => Some(format!("A photo of {who} sitting on a park bench feeding pigeons on an autumn afternoon, warm low sunlight.")),
        Kind::Object if !who.is_empty() => Some(format!("A photo of {who} on a kitchen counter next to a bowl of lemons, soft window light.")),
        Kind::Style => Some(format!("{}A lighthouse on a rocky coast at dusk, waves breaking below, a small boat in the distance.", t.map(|t| format!("{t}. ")).unwrap_or_default())),
        _ => None,
    };
    first.map(str::to_string).into_iter().chain(novel).filter(|s| !s.is_empty()).collect()
}

/// One string field of a JSON object that may be cut off or carry stray escapes: the text after `"key": "` up to the
/// closing quote (or the end), unescaped.
fn loose_field(t: &str, key: &str) -> Option<String> {
    let k = t.find(&format!("\"{key}\""))?;
    let rest = &t[k + key.len() + 2..];
    let rest = rest.trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => break,
            '\\' => match chars.next() {
                Some('n') => out.push(' '),
                Some('t') => out.push(' '),
                Some(o) => out.push(o),
                None => break,
            },
            c => out.push(c),
        }
    }
    let s = out.split_whitespace().collect::<Vec<_>>().join(" ");
    (!s.is_empty()).then(|| trim_runaway(&s))
}

/// An answer cut off by the token limit usually ran into a loop ("her soul cherished but not fully honored, her
/// spirit..."): keep it to its last full sentence (else clause) within ~900 characters.
fn trim_runaway(s: &str) -> String {
    const MAX: usize = 900;
    if s.len() <= MAX {
        return s.to_string();
    }
    let mut end = MAX;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    let head = &s[..end];
    match head.rfind(". ").or_else(|| head.rfind(", ")) {
        Some(i) if i > MAX / 3 => format!("{}.", head[..i].trim_end_matches(['.', ','])),
        _ => format!("{}.", head.trim_end()),
    }
}

/// A loki-scribe answer as (long, short): the JSON object (possibly fenced, cut off by the token limit, or with
/// stray escapes), else the whole text as the long caption.
pub fn parse_caption(text: &str) -> (String, Option<String>) {
    let t = text.trim();
    let body = t.find('{').and_then(|a| t.rfind('}').filter(|&b| b > a).map(|b| &t[a..=b]));
    if let Some(b) = body {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(b) {
            let long = v["long"].as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
            let short = v["short"].as_str().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
            match (long, short) {
                (Some(l), s) => return (l, s),
                (None, Some(s)) => return (s, None),
                _ => {}
            }
        }
    }
    if t.contains("\"long\"") || t.contains("\"short\"") {
        match (loose_field(t, "long"), loose_field(t, "short")) {
            (Some(l), s) => return (l, s),
            (None, Some(s)) => return (s, None),
            _ => {}
        }
    }
    (t.to_string(), None)
}

/// Caption images with loki-scribe (on PATH, or $LOKI_SCRIBE). Returns path -> (long, short).
pub fn scribe(paths: &[PathBuf], prompt: &str, quiet: bool) -> Result<BTreeMap<PathBuf, (String, Option<String>)>> {
    let exe = std::env::var("LOKI_SCRIBE").unwrap_or_else(|_| "loki-scribe".to_string());
    let mut out = BTreeMap::new();
    for chunk in paths.chunks(64) {
        let mut cmd = std::process::Command::new(&exe);
        cmd.args(["--json", "--temp", "0.2", "--dry-multiplier", "0.8", "--max-tokens", "600", "-p", prompt]);
        if quiet {
            cmd.arg("-q");
        }
        cmd.args(chunk);
        cmd.stdout(std::process::Stdio::piped()).stderr(if quiet { std::process::Stdio::null() } else { std::process::Stdio::inherit() });
        let res = cmd.output().with_context(|| format!("running {exe} to caption the images (install loki-scribe, or give captions: sidecar .txt files or a .jsonl list, or --captions none)"))?;
        ensure!(res.status.success(), "{exe} failed captioning the images (exit {})", res.status);
        for line in String::from_utf8_lossy(&res.stdout).lines() {
            let v: serde_json::Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if let (Some(i), Some(t)) = (v["index"].as_u64(), v["text"].as_str()) {
                if let Some(p) = chunk.get(i as usize) {
                    out.insert(p.clone(), parse_caption(t));
                }
            }
        }
    }
    for p in paths {
        ensure!(out.contains_key(p), "loki-scribe returned no caption for {}", p.display());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_keep_area_and_aspect() {
        assert_eq!(bucket(1024, 1024, 1 << 20), (1024, 1024));
        assert_eq!(bucket(4000, 3000, 1 << 20), (1168, 880));
        let (w, h) = bucket(3000, 4000, 1 << 20);
        assert!(w * h <= 1 << 20 && w % 16 == 0 && h % 16 == 0 && h > w);
        // small images are not upscaled past their own pixel count
        let (w, h) = bucket(640, 480, 1 << 20);
        assert!(w <= 640 && h <= 480 && w >= 624);
        assert_eq!(bucket(100, 100, 1 << 20), (256, 256));
    }

    #[test]
    fn trigger_goes_first_once() {
        assert_eq!(with_trigger("a woman in a red coat", Some("ohwx")), "ohwx, a woman in a red coat");
        assert_eq!(with_trigger("ohwx in a red coat", Some("ohwx")), "ohwx in a red coat");
        assert_eq!(with_trigger("", Some("ohwx")), "ohwx");
        assert_eq!(with_trigger("a cat", None), "a cat");
    }

    #[test]
    fn lists_resolve_relative_paths_and_captions() {
        let mut out = Vec::new();
        read_list("a.png\n# note\n\n{\"path\": \"b.jpg\", \"caption\": \"two\"}\n", Path::new("/d"), false, &mut out).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], (Path::new("/d").join("a.png"), None));
        assert_eq!(out[1], (Path::new("/d").join("b.jpg"), Some("two".to_string())));
    }

    #[test]
    fn captions_parse_json_or_text() {
        assert_eq!(parse_caption("```json\n{\"long\": \"a b\", \"short\": \"a\"}\n```"), ("a b".to_string(), Some("a".to_string())));
        assert_eq!(parse_caption("just text"), ("just text".to_string(), None));
        // cut off by the token limit, and with an escape JSON does not allow
        assert_eq!(
            parse_caption("{\"long\": \"a woman in a \\'red\\' coat, standing\", \"short\": \"a woman in a red co"),
            ("a woman in a 'red' coat, standing".to_string(), Some("a woman in a red co".to_string()))
        );
        assert_eq!(parse_caption("{\"long\": \"cut off here"), ("cut off here".to_string(), None));
        let looping = format!("{{\"long\": \"a woman in a pink top. Her hair is long, {}", "her heart valued but not fully treasured, ".repeat(40));
        let (l, _) = parse_caption(&looping);
        assert!(l.len() <= 901 && l.starts_with("a woman in a pink top.") && l.ends_with('.'), "{l}");
        let p = caption_prompt(Kind::Character, Some("ohwx"), Some("woman"), Some("red hair, freckles"));
        assert!(p.contains("\"ohwx, a woman,\"") && p.contains("red hair, freckles"));
        assert!(caption_prompt(Kind::Style, None, None, None).contains("Do NOT describe the style"));
        assert!(traits_prompt(Kind::General).is_none());
        let s = default_samples(Kind::Character, Some("ohwx"), Some("woman"), Some("ohwx, a woman, in a cafe"));
        assert_eq!(s.len(), 2);
        assert!(s[1].contains("ohwx, a woman,"));
        assert_eq!(plain_path(PathBuf::from(r"\\?\C:\x\y.png")), PathBuf::from(r"C:\x\y.png"));
    }

    #[test]
    fn fit_crops_the_centre() {
        let img = loki_dream::image::Rgb8 { w: 4, h: 2, data: (0..24).collect() };
        let f = fit(&img, 2, 2);
        assert_eq!((f.w, f.h), (2, 2));
        assert_eq!(&f.data[..3], &img.data[3..6]);
    }
}

