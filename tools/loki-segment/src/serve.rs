//! `--serve`: a JSON-lines session. Models load once; `open` encodes an image (or decodes a video); `segment` and
//! `track` answer against that encoding. One JSON line out per line in; errors are `{"error": "..."}`.
use crate::cli::{base_record, load_model, run_prompts, write_masks, Cli, OutPlan, WriteOpts};
use crate::image::Rgb8;
use crate::model::{ms, Encoded, Model, ObjPrompt, MODEL_NAME};
use crate::info;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;

pub enum Session {
    None,
    Image { path: String, img: Rgb8, enc: Encoded },
    Video(crate::video::VideoSession),
}

/// Object prompts from a request object (same vocabulary as the CLI flags and `--prompts`).
pub fn prompts_from_json(req: &Value) -> Result<Vec<ObjPrompt>> {
    let frame = req.get("frame").and_then(|v| v.as_u64()).map(|v| v as usize);
    let mut out = Vec::new();
    for key in ["text", "prompt"] {
        match req.get(key) {
            Some(Value::String(s)) => out.push(ObjPrompt { text: Some(s.clone()), frame, ..Default::default() }),
            Some(Value::Array(a)) => {
                for s in a {
                    if let Some(s) = s.as_str() {
                        out.push(ObjPrompt { text: Some(s.to_string()), frame, ..Default::default() });
                    }
                }
            }
            _ => {}
        }
    }
    let nums = |v: &Value| -> Option<Vec<f32>> { v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect()) };
    let mut pts: Vec<Vec<f32>> = Vec::new();
    let add_pts = |v: Option<&Value>, label: f32, pts: &mut Vec<Vec<f32>>| {
        if let Some(Value::Array(a)) = v {
            if a.first().map(|x| x.is_number()).unwrap_or(false) {
                if let Some(p) = nums(&Value::Array(a.clone())) {
                    pts.push(vec![p[0], p[1], if p.len() > 2 { p[2] } else { label }]);
                }
            } else {
                for e in a {
                    if let Some(p) = nums(e) {
                        if p.len() >= 2 {
                            pts.push(vec![p[0], p[1], if p.len() > 2 { p[2] } else { label }]);
                        }
                    }
                }
            }
        }
    };
    add_pts(req.get("point"), 1.0, &mut pts);
    add_pts(req.get("points"), 1.0, &mut pts);
    add_pts(req.get("neg"), 0.0, &mut pts);
    let mut boxes: Vec<Vec<f32>> = Vec::new();
    if let Some(Value::Array(a)) = req.get("box") {
        if a.first().map(|x| x.is_number()).unwrap_or(false) {
            boxes.push(nums(&Value::Array(a.clone())).unwrap_or_default());
        } else {
            boxes.extend(a.iter().filter_map(nums));
        }
    }
    if boxes.len() > 1 && !pts.is_empty() {
        bail!("clicks with several boxes are ambiguous; use \"prompts\" to group them per object");
    }
    let obj = req.get("obj").and_then(|v| v.as_u64()).map(|v| v as u32);
    if boxes.len() > 1 {
        for b in boxes {
            out.push(ObjPrompt { bbox: Some(b), frame, ..Default::default() });
        }
    } else if !pts.is_empty() || !boxes.is_empty() {
        // a "text" next to clicks labels the clicked object instead of searching for the concept
        let label = out.iter().find_map(|p| p.text.clone());
        if label.is_some() {
            out.retain(|p| p.text.is_none());
        }
        let candidate = req.get("candidate").and_then(|v| v.as_u64()).map(|v| v as usize);
        out.push(ObjPrompt { obj, points: pts, bbox: boxes.into_iter().next(), frame, text: label, candidate });
    }
    if let Some(p) = req.get("prompts") {
        let v: Vec<ObjPrompt> = serde_json::from_value(p.clone()).context("\"prompts\" must be a list of objects")?;
        // entries without their own "frame" use the request's
        out.extend(v.into_iter().map(|mut e| {
            if e.frame.is_none() {
                e.frame = frame;
            }
            e
        }));
    }
    Ok(out)
}

fn handle(model: &Model, sess: &mut Session, req: &Value, cli: &Cli) -> Result<Value> {
    let cmd = req.get("cmd").and_then(|v| v.as_str()).unwrap_or("");
    match cmd {
        "open" => {
            let path = req.get("path").and_then(|v| v.as_str()).context("open needs \"path\"")?.to_string();
            *sess = Session::None; // release the previous one first
            let t = Instant::now();
            if crate::cli::is_video_path(&path) {
                let v = crate::video::VideoSession::open(model, &path)?;
                let r = json!({"ok": true, "width": v.width, "height": v.height, "frames": v.frames.len(), "fps": v.fps, "ms": ms(t).round()});
                *sess = Session::Video(v);
                Ok(r)
            } else {
                let img = Rgb8::load(Path::new(&path))?;
                let enc = model.encode(&img, false)?;
                model.dev.sync()?;
                let r = json!({"ok": true, "width": img.w, "height": img.h, "frames": 1, "fps": Value::Null, "ms": ms(t).round()});
                *sess = Session::Image { path, img, enc };
                Ok(r)
            }
        }
        "segment" | "track" => {
            let prompts = prompts_from_json(req)?;
            let threshold = req.get("threshold").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(cli.threshold);
            let everything = req.get("everything").and_then(|v| v.as_bool()).unwrap_or(false);
            let opts = WriteOpts {
                alpha: req.get("alpha").and_then(|v| v.as_bool()).unwrap_or(cli.alpha),
                binary: req.get("binary").and_then(|v| v.as_bool()).unwrap_or(cli.binary),
                multimask: req.get("multimask").and_then(|v| v.as_bool()).unwrap_or(cli.multimask),
            };
            let out = req.get("out").and_then(|v| v.as_str());
            if prompts.is_empty() && !everything {
                bail!("nothing to segment: give text, point(s), neg, box, prompts or everything");
            }
            match sess {
                Session::None => bail!("no image open: send {{\"cmd\":\"open\",\"path\":...}} first"),
                Session::Image { path, img, enc } => {
                    let t = Instant::now();
                    let segs = run_prompts(model, enc, &prompts, threshold, everything, opts.multimask)?;
                    let plan = OutPlan::new(out, Some(path))?;
                    if plan.stdout {
                        bail!("\"out\": \"-\" is not available in --serve (stdout carries the replies)");
                    }
                    let rec = base_record(&prompts, threshold, path, img.w, img.h, req.get("meta"));
                    let (objs, _) = write_masks(model, img, &segs, &plan, &opts, &rec)?;
                    Ok(json!({"width": img.w, "height": img.h, "frames": 1, "fps": Value::Null, "model": MODEL_NAME, "objects": objs, "ms": ms(t).round()}))
                }
                Session::Video(v) => {
                    v.meta = req.get("meta").cloned();
                    let vopts = crate::video::TrackOpts::from_json(req, cli);
                    // on a video, "segment" previews the prompted frame only; "track" propagates through the clip
                    let t = Instant::now();
                    let mut r = v.track(model, &prompts, threshold, out, &opts, &vopts, cmd == "segment")?;
                    r["ms"] = json!(ms(t).round());
                    Ok(r)
                }
            }
        }
        "close" => {
            *sess = Session::None;
            Ok(json!({"ok": true}))
        }
        "info" => Ok(json!({"ok": true, "tool": crate::cli::TOOL, "version": crate::cli::VERSION, "model": MODEL_NAME})),
        "" => bail!("missing \"cmd\""),
        other => bail!("unknown cmd {other:?} (open, segment, track, close, info)"),
    }
}

pub fn run(cli: &Cli) -> Result<()> {
    let model = load_model(cli.model.clone())?;
    let mut sess = Session::None;
    if let Some(input) = &cli.input {
        let r = handle(&model, &mut sess, &json!({"cmd": "open", "path": input}), cli)?;
        info!("opened {input}: {r}");
    }
    info!("ready (JSON lines on stdin)");
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(req) => match handle(&model, &mut sess, &req, cli) {
                Ok(v) => v,
                Err(e) => json!({"error": format!("{e:#}")}),
            },
            Err(e) => json!({"error": format!("bad JSON: {e}")}),
        };
        writeln!(stdout, "{reply}")?;
        stdout.flush()?;
    }
    Ok(())
}
