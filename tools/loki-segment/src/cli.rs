//! Command-line front end: argument parsing, prompt assembly, mask output, `--json`, and dispatch to the image,
//! video and `--serve` paths.
use crate::cuda::Device;
use crate::genmeta::{self, Record};
use crate::image::Rgb8;
use crate::model::{mask_bbox, ms, Encoded, Model, ObjPrompt, Segment, MODEL_NAME};
use crate::{info, models};
use anyhow::{bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const TOOL: &str = "loki-segment";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const AFTER_HELP: &str = r#"PROMPTS
  -p/--prompt TEXT     a concept ("person", "the sky", "red car"): every matching instance becomes an object,
                       best first (joint detector score > --threshold, default 0.5). Repeatable: one concept each.
  --point X,Y / --neg X,Y   positive / negative clicks in input pixels: together they pick ONE object.
  --box X,Y,W,H        a box around one object (input pixels). With clicks: clicks + box = one object.
                       Several --box without clicks: one object per box.
  --prompts FILE.json  full spec, a list of objects (use `-` for stdin):
                       [{"obj": 1, "frame": 0, "text": "dog"}, {"obj": 2, "points": [[410,220,1],[400,300,0]]},
                        {"obj": 3, "box": [10,20,200,150], "text": "label only"}]
                       An entry with points/box is one object (its "text" is just its label); an entry with only
                       "text" yields every instance of that concept. Video: entries may be on different frames, and
                       several entries with the same "obj" on different frames are corrections (each adds a
                       conditioning frame to that object's memory). "candidate": k (clicks/box) uses the k-th
                       ambiguity candidate (0 = best, the order --multimask lists them in) as the object's mask.
  --multimask          clicks/box: also write SAM's 3 ambiguity candidates (sub-part / part / whole), best first.

OUTPUT
  Masks are 8-bit grayscale PNGs at the input resolution with SOFT edges (sigmoid of the upsampled logits;
  threshold at 128 for a hard mask, or pass --binary). --alpha writes the input as RGBA with the mask as alpha.
  Image:  one object  -> -o FILE.png, or -o DIR/<id>.png, default <input>_mask.png next to the input.
          several     -> -o DIR/<id>.png, or FILE_<id>.png for -o FILE.png, default <input>_mask_<id>.png.
          -o -        -> the PNG bytes on stdout (exactly one object; refuses a terminal).
          --multimask: candidates beside each click/box mask as <mask>_c1.png .. _c3.png.
  Video:  DIR/<id>/<frame:05>.png per object (default <input>_masks/); --multimask candidates as
          DIR/<id>/<frame:05>_c1.png .. on each prompted frame.
  stdout: the path of every file written, one per line; with --json a single JSON object instead:
    {"width":W,"height":H,"frames":N,"fps":F,"model":"SAM 3.1",
     "objects":[{"id":1,"label":"person","score":0.93,"bbox":[x,y,w,h],"mask":"path.png"}]}
    (with --multimask, click/box objects add "candidates":[{"mask":..,"score":..,"bbox":[..]}, ...] best first,
     score = sigmoid of the decoder's IoU prediction; video candidates also carry "frame")
    (video objects carry "frames":"DIR" instead of "mask" and "prompt_frames":[..]; bbox is on their first prompted
     frame; the top level adds "prompt_frame" (first) and "prompt_frames"; "objects":[] when nothing found)
  stderr, video: one progress line per frame step, "track 37/301 (95 ms/frame)", also when not a terminal.
  stderr: progress and timings (-q silences). Exit status 0 ok (also when nothing matched), 1 error, 2 usage.
  Every PNG carries the loki generation record (iTXt "loki"; see tools/CONVENTIONS.md).

SERVE (interactive editors)
  --serve reads JSON lines on stdin and answers exactly one JSON line per request on stdout:
    {"cmd":"open","path":"img.jpg"}           -> {"ok":true,"width":..,"height":..,"frames":1,"ms":..}
    {"cmd":"segment","text":"dog","out":"dir"} -> same object as --json (fields: text|prompt, point, points
        [[x,y,l]..], neg, box, prompts [...], threshold, out, alpha, binary, frame, multimask, candidate)
        on an open video: previews frame "frame" only (~100 ms for a new frame, ~10 ms again on the same one)
    {"cmd":"track", ...}                       -> video: propagate the prompts (any frames) through the clip
    {"cmd":"close"}                            -> drop the open image/video ({"ok":true})
  Errors come back as {"error":"..."} and the session continues; EOF ends it. Models load once; an open image is
  encoded once, so each segment request costs only the prompt decoders (~5-20 ms + PNG writing).

EXAMPLES
  loki-segment -p person photo.jpg                       # every person -> photo_mask_1.png, photo_mask_2.png ...
  loki-segment -p "the sky" --json -o masks/ photo.jpg   # JSON with ids, scores, boxes, mask paths
  loki-segment --point 410,220 --neg 380,400 photo.jpg   # the object under the click -> photo_mask.png
  loki-segment --box 100,80,300,420 --alpha -o cut.png photo.jpg
  loki-segment -p dog --frame 0 -o roto/ clip.mp4        # video: roto/<id>/00000.png ... (temporally tracked)
  cat img.png | loki-segment -p cat -o - - > cat_mask.png
"#;

#[derive(Parser, Debug)]
#[command(name = "loki-segment", version, about = "Segment Anything 3.1 (concepts, clicks, boxes; images and video) as one native CUDA binary", after_help = AFTER_HELP)]
pub struct Cli {
    /// Image or video file, or - for an image on stdin (not used with --serve)
    pub input: Option<String>,
    /// Concept prompt, repeatable ("person", "sky") -> every matching instance
    #[arg(short = 'p', long = "prompt")]
    pub prompt: Vec<String>,
    /// Concept prompts from a file, one per line (- = stdin)
    #[arg(short = 'P', long = "prompt-file")]
    pub prompt_file: Option<String>,
    /// Positive click X,Y in input pixels (repeatable)
    #[arg(long = "point", value_name = "X,Y", allow_hyphen_values = true)]
    pub point: Vec<String>,
    /// Negative click X,Y in input pixels (repeatable)
    #[arg(long = "neg", value_name = "X,Y", allow_hyphen_values = true)]
    pub neg: Vec<String>,
    /// Box X,Y,W,H in input pixels (repeatable)
    #[arg(long = "box", value_name = "X,Y,W,H", allow_hyphen_values = true)]
    pub bbox: Vec<String>,
    /// Video: frame the clicks/boxes/concepts are on
    #[arg(long, default_value_t = 0)]
    pub frame: usize,
    /// Full prompt spec (JSON list of objects); - = stdin
    #[arg(long = "prompts", value_name = "FILE.json")]
    pub prompts: Option<String>,
    /// Every object (automatic mask generation from a grid of clicks)
    #[arg(long)]
    pub everything: bool,
    /// Detector score threshold for concept prompts (joint class x presence probability)
    #[arg(long, default_value_t = 0.5)]
    pub threshold: f32,
    /// Video: the clip is a seamless loop (propagate across the wrap so first and last frames agree)
    #[arg(long = "loop")]
    pub looped: bool,
    /// Video: only propagate forward from the prompted frame
    #[arg(long = "no-bidir")]
    pub no_bidir: bool,
    /// Video: temporal smoothing of mask logits over N frames (0 = off)
    #[arg(long, default_value_t = 0)]
    pub smooth: usize,
    /// Video: when the tracker's mask candidates score within F of each other, keep the one that agrees with the
    /// previous frame (stops near-ties flipping a limb in and out); 0 = stock argmax
    #[arg(long, default_value_t = 0.15)]
    pub sticky: f32,
    /// Output file, directory, or - (PNG to stdout)
    #[arg(short = 'o', long = "out")]
    pub out: Option<String>,
    /// Clicks/boxes: also write SAM's 3 ambiguity candidates (sub-part / part / whole), best first, as
    /// <mask>_c1.png .. _c3.png, listed under "candidates" in --json (video: on the prompted frames)
    #[arg(long)]
    pub multimask: bool,
    /// Write the input as RGBA with the mask as alpha (a cut-out) instead of a gray mask
    #[arg(long)]
    pub alpha: bool,
    /// Hard 0/255 masks instead of soft edges
    #[arg(long)]
    pub binary: bool,
    /// Print one JSON object (objects, scores, boxes, mask paths) instead of paths
    #[arg(long)]
    pub json: bool,
    /// Silence progress on stderr
    #[arg(short, long)]
    pub quiet: bool,
    /// JSON-lines session on stdin/stdout (models load once; see SERVE below)
    #[arg(long)]
    pub serve: bool,
    /// Model file (default: discovered next to the binary, ./models, $LOKI_MODELS; downloaded when missing)
    #[arg(long = "model", value_name = "PATH")]
    pub model: Option<PathBuf>,
    /// Caller's JSON object stored verbatim in the generation record ("meta")
    #[arg(long)]
    pub meta: Option<String>,
    /// Read --meta from a file
    #[arg(long = "meta-file")]
    pub meta_file: Option<String>,
}

pub fn main() {
    let cli = Cli::parse();
    crate::log::set_quiet(cli.quiet);
    let r = if cli.serve { crate::serve::run(&cli) } else { run(&cli) };
    if let Err(e) = r {
        eprintln!("{TOOL}: error: {e:#}");
        std::process::exit(1);
    }
}

fn usage(msg: &str) -> ! {
    eprintln!("{TOOL}: error: {msg}\n\nFor more information, try '--help'.");
    std::process::exit(2);
}

pub fn parse_nums(s: &str, n: usize, what: &str) -> Result<Vec<f32>> {
    let v: Vec<f32> = s.split(',').map(|t| t.trim().parse::<f32>()).collect::<std::result::Result<_, _>>().with_context(|| format!("{what} {s:?}: expected {n} comma-separated numbers"))?;
    if v.len() != n {
        bail!("{what} {s:?}: expected {n} comma-separated numbers");
    }
    Ok(v)
}

/// Object prompts from the flags (concepts, clicks/boxes, --prompts file).
pub fn prompts_from_cli(cli: &Cli) -> Result<Vec<ObjPrompt>> {
    let mut out = Vec::new();
    let mut texts = cli.prompt.clone();
    if let Some(f) = &cli.prompt_file {
        let s = if f == "-" { read_stdin_string()? } else { std::fs::read_to_string(f).with_context(|| format!("reading {f}"))? };
        texts.extend(s.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()));
    }
    for t in texts {
        out.push(ObjPrompt { text: Some(t), frame: Some(cli.frame), ..Default::default() });
    }
    let mut pts = Vec::new();
    for p in &cli.point {
        let v = parse_nums(p, 2, "--point")?;
        pts.push(vec![v[0], v[1], 1.0]);
    }
    for p in &cli.neg {
        let v = parse_nums(p, 2, "--neg")?;
        pts.push(vec![v[0], v[1], 0.0]);
    }
    let boxes: Vec<Vec<f32>> = cli.bbox.iter().map(|b| parse_nums(b, 4, "--box")).collect::<Result<_>>()?;
    if boxes.len() > 1 && !pts.is_empty() {
        bail!("clicks with several --box are ambiguous; use --prompts to say which click belongs to which object");
    }
    if boxes.len() > 1 {
        for b in boxes {
            out.push(ObjPrompt { bbox: Some(b), frame: Some(cli.frame), ..Default::default() });
        }
    } else if !pts.is_empty() || !boxes.is_empty() {
        out.push(ObjPrompt { points: pts, bbox: boxes.into_iter().next(), frame: Some(cli.frame), ..Default::default() });
    }
    if let Some(f) = &cli.prompts {
        let s = if f == "-" { read_stdin_string()? } else { std::fs::read_to_string(f).with_context(|| format!("reading {f}"))? };
        let v: Vec<ObjPrompt> = serde_json::from_str(&s).with_context(|| format!("parsing {f} (a JSON list of objects)"))?;
        out.extend(v.into_iter().map(|mut e| {
            if e.frame.is_none() {
                e.frame = Some(cli.frame);
            }
            e
        }));
    }
    Ok(out)
}

fn read_stdin_string() -> Result<String> {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s)?;
    Ok(s)
}

pub fn is_video_path(p: &str) -> bool {
    let l = p.to_lowercase();
    [".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".mpg", ".mpeg", ".wmv"].iter().any(|e| l.ends_with(e))
}

pub fn load_model(explicit: Option<PathBuf>) -> Result<Model> {
    let path = models::resolve(explicit, &models::SAM31)?;
    let dev = Device::new(0)?;
    let t = Instant::now();
    let m = Model::load(&path, dev)?;
    m.dev.sync()?;
    info!("loaded {} in {:.2}s ({} MB on the GPU)", path.display(), t.elapsed().as_secs_f64(), m.weights.uploaded.get() >> 20);
    Ok(m)
}

/// Where the masks of an image go.
pub struct OutPlan {
    /// directory mode: <dir>/<id>.png
    pub dir: Option<PathBuf>,
    /// file mode: base path (one object: as is; several: <stem>_<id>.png)
    pub file: Option<PathBuf>,
    pub stdout: bool,
}
impl OutPlan {
    pub fn new(out: Option<&str>, input: Option<&str>) -> Result<OutPlan> {
        match out {
            Some("-") => Ok(OutPlan { dir: None, file: None, stdout: true }),
            Some(o) => {
                let p = PathBuf::from(o);
                let is_dir = p.is_dir() || o.ends_with('/') || o.ends_with('\\') || p.extension().is_none();
                if is_dir {
                    Ok(OutPlan { dir: Some(p), file: None, stdout: false })
                } else {
                    Ok(OutPlan { dir: None, file: Some(p), stdout: false })
                }
            }
            None => match input {
                Some(i) if i != "-" => {
                    let ip = Path::new(i);
                    let stem = ip.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "image".into());
                    Ok(OutPlan { dir: None, file: Some(ip.with_file_name(format!("{stem}_mask.png"))), stdout: false })
                }
                _ => bail!("an image read from stdin needs -o (a file, a directory, or - for stdout)"),
            },
        }
    }
    pub fn path_for(&self, id: u32, n: usize) -> PathBuf {
        if let Some(d) = &self.dir {
            return d.join(format!("{id}.png"));
        }
        let f = self.file.as_ref().unwrap();
        if n == 1 {
            return f.clone();
        }
        let stem = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        f.with_file_name(format!("{stem}_{id}.png"))
    }
}

pub struct WriteOpts {
    pub alpha: bool,
    pub binary: bool,
    pub multimask: bool,
}

/// Write the masks of one image and return the JSON objects.
pub fn write_masks(model: &Model, img: &Rgb8, segs: &[Segment], plan: &OutPlan, opts: &WriteOpts, rec: &Record) -> Result<(Vec<Value>, Vec<PathBuf>)> {
    let mut objs = Vec::new();
    let mut paths = Vec::new();
    if plan.stdout && segs.len() != 1 {
        bail!("-o - writes exactly one mask but {} objects were found; use -o DIR", segs.len());
    }
    if let Some(d) = &plan.dir {
        std::fs::create_dir_all(d).with_context(|| format!("creating {}", d.display()))?;
    }
    let encode = |logits: &crate::tensor::Tensor, obj: Value| -> Result<(Vec<u8>, Option<[usize; 4]>)> {
        let mut m = model.mask_u8(logits, img.w, img.h)?;
        if opts.binary {
            for v in m.iter_mut() {
                *v = if *v >= 128 { 255 } else { 0 };
            }
        }
        let bbox = mask_bbox(&m, img.w, img.h);
        let mut r = rec.clone();
        r.set("object", obj);
        let text = vec![(genmeta::PNG_KEYWORD.to_string(), r.to_json())];
        let bytes = if opts.alpha { crate::image::rgba_png_bytes(img, &m, &text)? } else { crate::image::gray_png_bytes(img.w, img.h, &m, &text)? };
        Ok((bytes, bbox))
    };
    for s in segs {
        let (bytes, bbox) = encode(&s.logits, json!({"id": s.id, "label": s.label, "score": s.score}))?;
        let mask_val = if plan.stdout {
            let mut so = std::io::stdout().lock();
            if so.is_terminal() {
                bail!("refusing to write PNG data to a terminal; redirect stdout or use -o FILE");
            }
            so.write_all(&bytes)?;
            so.flush()?;
            Value::from("-")
        } else {
            let p = plan.path_for(s.id, segs.len());
            if let Some(parent) = p.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&p, &bytes).with_context(|| format!("writing {}", p.display()))?;
            paths.push(p.clone());
            Value::from(p.display().to_string())
        };
        let mut o = json!({
            "id": s.id,
            "label": s.label,
            "score": (s.score as f64 * 1e4).round() / 1e4,
            "bbox": bbox.map(|b| json!(b)).unwrap_or(Value::Null),
            "mask": mask_val,
        });
        if opts.multimask && !s.candidates.is_empty() && !plan.stdout {
            // derive from the mask path's own spelling so separators stay consistent with "mask"
            let base = plan.path_for(s.id, segs.len()).display().to_string();
            let stem = base.strip_suffix(".png").unwrap_or(&base).to_string();
            let mut cands = Vec::new();
            for (k, (score, lg)) in s.candidates.iter().enumerate() {
                let (bytes, bbox) = encode(lg, json!({"id": s.id, "label": s.label, "candidate": k, "score": score}))?;
                let p = PathBuf::from(format!("{stem}_c{}.png", k + 1));
                std::fs::write(&p, &bytes).with_context(|| format!("writing {}", p.display()))?;
                cands.push(json!({"mask": p.display().to_string(), "score": (*score as f64 * 1e4).round() / 1e4, "bbox": bbox.map(|b| json!(b)).unwrap_or(Value::Null)}));
            }
            o["candidates"] = Value::Array(cands);
        }
        objs.push(o);
    }
    Ok((objs, paths))
}

pub fn base_record(prompts: &[ObjPrompt], threshold: f32, input: &str, w: usize, h: usize, meta: Option<&Value>) -> Record {
    let mut r = Record::new(TOOL, VERSION, MODEL_NAME);
    r.set("prompts", serde_json::to_value(prompts).unwrap_or(Value::Null));
    r.set("threshold", threshold as f64);
    r.set("width", w as u64).set("height", h as u64);
    r.set("inputs", json!([{"path": input, "width": w, "height": h}]));
    if let Some(m) = meta {
        r.set("meta", m.clone());
    }
    r
}

fn run(cli: &Cli) -> Result<()> {
    let Some(input) = cli.input.clone() else { usage("an INPUT image or video is required (or --serve)") };
    let meta = genmeta::read_meta(cli.meta.as_deref(), cli.meta_file.as_deref()).map_err(|e| anyhow::anyhow!(e))?;
    let prompts = match prompts_from_cli(cli) {
        Ok(p) => p,
        Err(e) => usage(&format!("{e:#}")),
    };
    if prompts.is_empty() && !cli.everything {
        usage("nothing to segment: give -p TEXT, --point/--neg, --box, --prompts FILE or --everything");
    }
    if is_video_path(&input) {
        return crate::video::run_cli(cli, &input, &prompts, meta.as_ref());
    }
    let img = if input == "-" {
        let mut b = Vec::new();
        std::io::stdin().read_to_end(&mut b)?;
        Rgb8::from_bytes(&b)?
    } else {
        Rgb8::load(Path::new(&input))?
    };
    let plan = OutPlan::new(cli.out.as_deref(), Some(&input))?;
    let model = load_model(cli.model.clone())?;
    let t = Instant::now();
    let enc = model.encode(&img, false)?;
    model.dev.sync()?;
    info!("encoded {}x{} in {:.0} ms", img.w, img.h, ms(t));
    let t = Instant::now();
    let segs = run_prompts(&model, &enc, &prompts, cli.threshold, cli.everything, cli.multimask)?;
    model.dev.sync()?;
    info!("{} object(s) in {:.0} ms", segs.len(), ms(t));
    let rec = base_record(&prompts, cli.threshold, &input, img.w, img.h, meta.as_ref());
    let t = Instant::now();
    let (objs, paths) = write_masks(&model, &img, &segs, &plan, &WriteOpts { alpha: cli.alpha, binary: cli.binary, multimask: cli.multimask }, &rec)?;
    info!("wrote {} mask(s) in {:.0} ms", objs.len(), ms(t));
    if segs.is_empty() {
        info!("nothing matched");
    }
    if cli.json {
        let v = json!({"width": img.w, "height": img.h, "frames": 1, "fps": Value::Null, "model": MODEL_NAME, "objects": objs});
        println!("{v}");
    } else if !plan.stdout {
        for p in paths {
            println!("{}", p.display());
        }
    }
    Ok(())
}

/// Prompts plus optional automatic mask generation.
pub fn run_prompts(model: &Model, enc: &Encoded, prompts: &[ObjPrompt], threshold: f32, everything: bool, multimask: bool) -> Result<Vec<Segment>> {
    let mut segs = model.segment(enc, prompts, threshold, multimask)?;
    if everything {
        let first = segs.iter().map(|s| s.id).max().unwrap_or(0) + 1;
        segs.extend(crate::everything::generate(model, enc, first)?);
    }
    Ok(segs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cli(args: &[&str]) -> Cli {
        Cli::parse_from(std::iter::once("loki-segment").chain(args.iter().copied()))
    }

    #[test]
    fn clicks_and_a_box_form_one_object() {
        let p = prompts_from_cli(&cli(&["--point", "10,20", "--neg", "30,40", "--box", "1,2,3,4", "in.jpg"])).unwrap();
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].points, vec![vec![10.0, 20.0, 1.0], vec![30.0, 40.0, 0.0]]);
        assert_eq!(p[0].bbox, Some(vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn concepts_and_boxes_are_separate_objects() {
        let p = prompts_from_cli(&cli(&["-p", "person", "-p", "sky", "--box", "0,0,5,5", "--box", "5,5,5,5", "--frame", "3", "in.mp4"])).unwrap();
        assert_eq!(p.len(), 4);
        assert_eq!(p[0].text.as_deref(), Some("person"));
        assert!(p.iter().all(|o| o.frame == Some(3)));
        assert!(prompts_from_cli(&cli(&["--point", "1,1", "--box", "0,0,5,5", "--box", "5,5,5,5", "in.jpg"])).is_err());
        assert!(prompts_from_cli(&cli(&["--point", "1", "in.jpg"])).is_err());
    }

    #[test]
    fn output_plans() {
        let f = OutPlan::new(None, Some("dir/photo.jpg")).unwrap();
        assert_eq!(f.path_for(1, 1), PathBuf::from("dir/photo_mask.png"));
        assert_eq!(f.path_for(2, 3), PathBuf::from("dir/photo_mask_2.png"));
        let d = OutPlan::new(Some("masks/"), Some("photo.jpg")).unwrap();
        assert_eq!(d.path_for(7, 1), PathBuf::from("masks/").join("7.png"));
        assert!(OutPlan::new(Some("-"), Some("-")).unwrap().stdout);
        assert!(OutPlan::new(None, Some("-")).is_err());
    }

    #[test]
    fn serve_requests_parse_like_flags() {
        let r: Value = serde_json::from_str(r#"{"cmd":"segment","text":["dog","cat"],"frame":4}"#).unwrap();
        let p = crate::serve::prompts_from_json(&r).unwrap();
        assert_eq!(p.len(), 2);
        let r: Value = serde_json::from_str(r#"{"cmd":"segment","text":"dog","point":[5,6],"neg":[[1,2]],"box":[0,0,9,9]}"#).unwrap();
        let p = crate::serve::prompts_from_json(&r).unwrap();
        assert_eq!(p.len(), 1, "text next to clicks labels the clicked object");
        assert_eq!(p[0].text.as_deref(), Some("dog"));
        assert_eq!(p[0].points, vec![vec![5.0, 6.0, 1.0], vec![1.0, 2.0, 0.0]]);
        let r: Value = serde_json::from_str(r#"{"cmd":"segment","point":[5,6],"candidate":2}"#).unwrap();
        assert_eq!(crate::serve::prompts_from_json(&r).unwrap()[0].candidate, Some(2));
        let r: Value = serde_json::from_str(r#"{"cmd":"track","prompts":[{"obj":1,"frame":0,"points":[[1,1,1]]},{"obj":1,"frame":40,"box":[0,0,9,9],"candidate":1}]}"#).unwrap();
        let p = crate::serve::prompts_from_json(&r).unwrap();
        assert_eq!((p[1].obj, p[1].frame, p[1].candidate), (Some(1), Some(40), Some(1)));
        let r: Value = serde_json::from_str(r#"{"cmd":"track","frame":9,"prompts":[{"obj":3,"points":[[1,1,1]]},{"text":"x","frame":2}]}"#).unwrap();
        let p = crate::serve::prompts_from_json(&r).unwrap();
        assert_eq!((p[0].frame, p[1].frame), (Some(9), Some(2)));
    }
}
