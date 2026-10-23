//! Video: frames through ffmpeg, SAM 3.1 tracking from one prompted frame, and the temporal-stability layer on top
//! of the stock tracker:
//!   - bidirectional: a backward pass from the prompted frame covers the frames before it (default on);
//!   - loop (`--loop`): both passes run all the way around the clip, so every frame gets a forward and a backward
//!     estimate; they are merged with weights falling off with the distance to the prompt, which makes the last
//!     frame agree with the first;
//!   - smoothing (`--smooth N`): a triangular temporal filter over the mask logits (wrapping in loop mode).
use crate::cli::{base_record, Cli, WriteOpts};
use crate::genmeta::{self, Record};
use crate::image::Rgb8;
use crate::info;
use crate::interactive::Point;
use crate::model::{mask_bbox, ms, Model, ObjPrompt, MODEL_NAME};
use crate::tensor::{DType, Tensor};
use crate::tracker::{FrameFeats, FrameMem, SLOTS};
use crate::vit::IMG;
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

const LOW: usize = 288;
const P: usize = LOW * LOW;

pub struct TrackOpts {
    pub looped: bool,
    pub bidir: bool,
    pub smooth: usize,
    /// candidate-selection hysteresis margin (0 = stock argmax)
    pub sticky: f32,
}
impl TrackOpts {
    pub fn from_json(req: &Value, cli: &Cli) -> TrackOpts {
        TrackOpts {
            looped: req.get("loop").and_then(|v| v.as_bool()).unwrap_or(cli.looped),
            bidir: !req.get("no_bidir").and_then(|v| v.as_bool()).unwrap_or(cli.no_bidir),
            smooth: req.get("smooth").and_then(|v| v.as_u64()).map(|v| v as usize).unwrap_or(cli.smooth),
            sticky: req.get("sticky").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(cli.sticky),
        }
    }
}

pub struct VideoSession {
    pub path: String,
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub frames: Vec<Rgb8>,
    cache: HashMap<usize, FrameFeats>,
    cache_limit: usize,
    /// validation hook: already-preprocessed model inputs (CHW f32, 1008^2) used instead of `frames`
    pub ref_inputs: Option<Vec<Vec<f32>>>,
    /// reproduce the official demo's multi-object add path exactly (validation)
    pub parity: bool,
    /// validation: dump tracker intermediates of the first frames here
    pub debug: Option<std::cell::RefCell<crate::dump::Dump>>,
    /// the last prompted frame's full encoding (interactive clicking on one frame in --serve)
    last_enc: Option<(usize, std::rc::Rc<crate::model::Encoded>)>,
    /// caller's --meta object for the generation record
    pub meta: Option<Value>,
}

/// A tracked object: its conditioning-frame output and identity.
pub struct CondObj {
    pub id: u32,
    pub label: String,
    pub score: f32,
    pub logits: Tensor,
    pub obj_score: f32,
    pub ptr: Vec<f32>,
    /// found by a concept prompt (multiplexed with its siblings)
    pub concept: bool,
}

/// Tracking result: the objects and, per frame, their (merged, smoothed) low-res logits [288*288] per object.
pub struct Tracked {
    pub objs: Vec<CondObj>,
    pub frame: usize,
    pub logits: BTreeMap<usize, Vec<Vec<f32>>>,
}

/// Per frame, per object: low-res logits on the host, plus the pass distance they came from.
type Estimates = BTreeMap<usize, Vec<(usize, Vec<Vec<f32>>)>>;

impl VideoSession {
    /// A session over frames already in memory (e.g. a JPEG sequence).
    pub fn from_frames(path: &str, frames: Vec<Rgb8>, fps: f64) -> VideoSession {
        let (width, height) = (frames[0].w, frames[0].h);
        VideoSession { path: path.to_string(), width, height, fps, frames, cache: HashMap::new(), cache_limit: 560, ref_inputs: None, parity: false, debug: None, last_enc: None, meta: None }
    }

    pub fn open(_model: &Model, path: &str) -> Result<VideoSession> {
        let t = Instant::now();
        let info = crate::media::probe(Path::new(path))?;
        let frames = crate::media::decode_frames(Path::new(path), &info, 0)?;
        if frames.is_empty() {
            bail!("{path}: no frames decoded");
        }
        info!("decoded {} frames {}x{} @ {:.3} fps in {:.0} ms", frames.len(), info.width, info.height, info.fps, ms(t));
        // tracker features of a frame take ~10.5 MB; only loop mode revisits frames (forward and backward pass), so
        // a modest cache (~1.6 GB) is enough to save most of the second visits on short loops
        let limit = std::env::var("LOKI_SEGMENT_CACHE_FRAMES").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
        Ok(VideoSession { path: path.to_string(), width: info.width, height: info.height, fps: info.fps, frames, cache: HashMap::new(), cache_limit: limit, ref_inputs: None, parity: false, debug: None, last_enc: None, meta: None })
    }

    fn dbg(&self, model: &Model, name: &str, t: &Tensor) -> Result<()> {
        if let Some(d) = &self.debug {
            let v = t.to_f32_vec(&model.dev)?;
            d.borrow_mut().write(name, &t.shape, &v)?;
        }
        Ok(())
    }
    fn dbg_host(&self, name: &str, shape: &[usize], v: &[f32]) -> Result<()> {
        if let Some(d) = &self.debug {
            d.borrow_mut().write(name, shape, v)?;
        }
        Ok(())
    }

    fn feats(&mut self, model: &Model, t: usize) -> Result<FrameFeats> {
        if let Some(f) = self.cache.get(&t) {
            return Ok(f.clone());
        }
        let x = self.input(model, t)?;
        let trunk = model.vit.forward(&x)?;
        let prop = model.neck_prop.forward(&model.dev, &trunk)?;
        let f = model.tracker.frame_feats(&prop)?;
        if self.cache.len() < self.cache_limit {
            self.cache.insert(t, f.clone());
        }
        Ok(f)
    }

    fn input(&self, model: &Model, t: usize) -> Result<Tensor> {
        match &self.ref_inputs {
            Some(r) => Tensor::from_f32(&model.dev, &r[t], &[3, IMG, IMG]),
            None => model.preprocess(&self.frames[t]),
        }
    }

    /// Resolve the prompts on their frame into conditioning objects.
    fn condition(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, frame: usize) -> Result<(Vec<CondObj>, FrameFeats)> {
        let enc = match &self.last_enc {
            Some((t, e)) if *t == frame => e.clone(),
            _ => {
                let img = &self.frames[frame];
                let x = self.input(model, frame)?;
                let e = std::rc::Rc::new(model.encode_tensor(&x, img.w, img.h, true)?);
                self.last_enc = Some((frame, e.clone()));
                e
            }
        };
        let img = &self.frames[frame];
        let feats = model.tracker.frame_feats(enc.prop.as_ref().unwrap())?;
        self.cache.insert(frame, feats.clone());
        let (sx, sy) = (IMG as f32 / img.w as f32, IMG as f32 / img.h as f32);
        let mut objs: Vec<CondObj> = Vec::new();
        let mut next_id = prompts.iter().filter_map(|p| p.obj).max().unwrap_or(0) + 1;
        let mut fresh = || {
            next_id += 1;
            next_id - 1
        };
        for p in prompts {
            if !p.points.is_empty() || p.bbox.is_some() {
                let mut pts: Vec<Point> = p
                    .points
                    .iter()
                    .map(|v| Point { x: v.first().copied().unwrap_or(0.0) * sx, y: v.get(1).copied().unwrap_or(0.0) * sy, label: v.get(2).map(|l| *l as i32).unwrap_or(1) })
                    .collect();
                if let Some(b) = &p.bbox {
                    if b.len() != 4 {
                        bail!("a box is [x, y, w, h]");
                    }
                    pts.push(Point { x: b[0] * sx, y: b[1] * sy, label: 2 });
                    pts.push(Point { x: (b[0] + b[2]) * sx, y: (b[1] + b[3]) * sy, label: 3 });
                }
                let o = model.inter.decode(&enc.sam, &pts, pts.len() == 1)?;
                let id = p.obj.unwrap_or_else(&mut fresh);
                let label = p.text.clone().unwrap_or_else(|| if p.bbox.is_some() { "box".into() } else { "click".into() });
                let score = crate::detector::sigmoid(o.obj_score);
                if self.parity && !objs.is_empty() {
                    // official demo: objects added to an existing bucket enter through the mask-input path
                    let (lg, ptr, sc) = self.mask_object_from_logits(model, &enc, &o.mask)?;
                    objs.push(CondObj { id, label, score, logits: lg, obj_score: sc, ptr, concept: false });
                } else {
                    let (ptr, sc) = model.tracker.cond_object(&o);
                    let lg = self.via_video_res(model, &o.mask)?;
                    objs.push(CondObj { id, label, score, logits: lg, obj_score: sc, ptr, concept: false });
                }
            } else if let Some(t) = &p.text {
                // concept: every detected instance enters the tracker through the mask-input path
                for (i, (score, logits)) in model.concept(&enc, t, threshold)?.into_iter().enumerate() {
                    let id = match p.obj {
                        Some(o) if i == 0 => o,
                        _ => fresh(),
                    };
                    let (lg, ptr, sc) = self.mask_object(model, &enc, &logits)?;
                    objs.push(CondObj { id, label: t.clone(), score, logits: lg, obj_score: sc, ptr, concept: true });
                }
            } else {
                bail!("a prompt needs text, points or a box");
            }
        }
        if objs.len() > SLOTS {
            info!("{} objects found; tracking the first {SLOTS}", objs.len());
            objs.truncate(SLOTS);
        }
        Ok((objs, feats))
    }

    /// The official demo stores a prompted frame's masks at video resolution and brings them back to 288 for
    /// the tracker (bilinear up, antialiased down), which softens the logits slightly; do the same.
    fn via_video_res(&self, model: &Model, logits288: &Tensor) -> Result<Tensor> {
        let dev = &model.dev;
        let (w, h) = (self.width, self.height);
        let up = Tensor::new(dev, DType::F32, &[w * h])?;
        dev.launch_n(
            "k_resize_logits_f32",
            w * h,
            &[crate::cuda::Arg::Ptr(logits288.ptr), crate::cuda::Arg::I32(1), crate::cuda::Arg::I32(LOW as i32), crate::cuda::Arg::I32(LOW as i32), crate::cuda::Arg::Ptr(up.ptr), crate::cuda::Arg::I32(h as i32), crate::cuda::Arg::I32(w as i32)],
        )?;
        let v = up.to_f32_vec(dev)?;
        let back = crate::image::resample_plane(&v, w, h, LOW, LOW);
        Tensor::from_f32(dev, &back, &[P])
    }

    /// Official multi-object add (`add_new_masks_to_existing_state` -> `_use_mask_as_output` on the clicked
    /// object's own logits): output = AA-pool(logits*20-10, 72) brought back to 288; pointer from a decode prompted
    /// with those logits as a mask; score +-10.
    fn mask_object_from_logits(&self, model: &Model, enc: &crate::model::Encoded, logits288: &Tensor) -> Result<(Tensor, Vec<f32>, f32)> {
        let dev = &model.dev;
        let lo = logits288.to_f32_vec(dev)?;
        let appearing = lo.iter().any(|v| *v > 0.0);
        let hi: Vec<f32> = lo.iter().map(|v| v * 20.0 - 10.0).collect();
        let l72 = crate::image::resample_plane(&hi, LOW, LOW, 72, 72);
        let back = crate::image::resample_plane(&l72, 72, 72, LOW, LOW);
        let dense = model.inter.mask_prompt_from(&lo, LOW);
        let o = model.inter.decode_with(&enc.sam, &[Point { x: 0.0, y: 0.0, label: -1 }], false, Some(&dense))?;
        let (mut ptr, _) = model.tracker.cond_object(&o);
        if !appearing {
            ptr = model.tracker.no_obj_pointer(&ptr);
        }
        Ok((Tensor::from_f32(dev, &back, &[P])?, ptr, if appearing { 10.0 } else { -10.0 }))
    }

    /// `_use_mask_as_output`: a detector mask (288 logits) as a conditioning mask: the output is the binary mask
    /// (x20 - 10), the pointer comes from the interactive decoder prompted with the mask.
    fn mask_object(&self, model: &Model, enc: &crate::model::Encoded, logits288: &Tensor) -> Result<(Tensor, Vec<f32>, f32)> {
        let dev = &model.dev;
        let lo = logits288.to_f32_vec(dev)?;
        // binary mask at 1008: bilinear of the logits > 0
        let up = Tensor::new(dev, DType::F32, &[IMG * IMG])?;
        dev.launch_n(
            "k_resize_logits_f32",
            IMG * IMG,
            &[crate::cuda::Arg::Ptr(logits288.ptr), crate::cuda::Arg::I32(1), crate::cuda::Arg::I32(LOW as i32), crate::cuda::Arg::I32(LOW as i32), crate::cuda::Arg::Ptr(up.ptr), crate::cuda::Arg::I32(IMG as i32), crate::cuda::Arg::I32(IMG as i32)],
        )?;
        let m1008: Vec<f32> = up.to_f32_vec(dev)?.iter().map(|v| if *v > 0.0 { 1.0 } else { 0.0 }).collect();
        let appearing = m1008.iter().any(|v| *v > 0.0);
        let dense = model.inter.mask_prompt(&m1008);
        let o = model.inter.decode_with(&enc.sam, &[Point { x: 0.0, y: 0.0, label: -1 }], false, Some(&dense))?;
        let (mut ptr, _) = model.tracker.cond_object(&o);
        if !appearing {
            ptr = model.tracker.no_obj_pointer(&ptr);
        }
        // output mask: binary x 20 - 10 (low res: the area-average of the 1008 mask, as the official downsample)
        let bin: Vec<f32> = crate::image::resample_plane(&m1008, IMG, IMG, LOW, LOW).iter().map(|v| v * 20.0 - 10.0).collect();
        let _ = lo;
        let sc = if appearing { 10.0 } else { -10.0 };
        Ok((Tensor::from_f32(dev, &bin, &[P])?, ptr, sc))
    }

    /// One propagation pass over `order` (physical frame indices, pass step i+1 = order[i]) for every bucket at
    /// once (each frame's backbone runs once); the conditioning frame is pass step 0. Returns per visited frame,
    /// per bucket, the per-object logits (host).
    #[allow(clippy::too_many_arguments)]
    fn run_pass(&mut self, model: &Model, conds: &[(FrameMem, Vec<Vec<f32>>)], order: &[usize], label: &str, debug: bool, sticky: f32) -> Result<Vec<(usize, Vec<Vec<Vec<f32>>>)>> {
        let num_frames = self.frames.len();
        let nb = conds.len();
        let mut mems: Vec<BTreeMap<usize, FrameMem>> = vec![BTreeMap::new(); nb];
        let mut prev: Vec<Vec<Vec<f32>>> = conds.iter().map(|c| c.1.clone()).collect();
        let mut out = Vec::new();
        let t0 = Instant::now();
        for (i, &t) in order.iter().enumerate() {
            let step = i + 1;
            let feats = self.feats(model, t)?;
            let mut per_bucket = Vec::with_capacity(nb);
            for b in 0..nb {
                let cond = &conds[b].0;
                let n = conds[b].1.len();
                let mut mem: Vec<(&FrameMem, i64)> = vec![(cond, step as i64)];
                for t_pos in 1..7usize {
                    let t_rel = 7 - t_pos;
                    if step >= t_rel + 1 {
                        if let Some(m) = mems[b].get(&(step - t_rel)) {
                            mem.push((m, t_pos as i64));
                        }
                    }
                }
                let mut ptrs: Vec<(&[f32], usize)> = vec![(&cond.ptr, step)];
                for t_diff in 1..num_frames.min(16) {
                    if t_diff > step {
                        break;
                    }
                    if let Some(m) = mems[b].get(&(step - t_diff)) {
                        ptrs.push((&m.ptr, t_diff));
                    }
                }
                let pix = model.tracker.attend_memory(&feats, &mem, &ptrs, num_frames)?;
                let (objs, ptr) = model.tracker.decode(&feats, &pix, n, if sticky > 0.0 { Some((&prev[b], sticky)) } else { None })?;
                if debug && b == 0 && step <= 3 {
                    self.dbg(model, &format!("f{t}_pixmem"), &pix)?;
                    self.dbg_host(&format!("f{t}_objptr"), &[SLOTS, 256], &ptr)?;
                }
                let lrefs: Vec<&Tensor> = objs.iter().map(|o| &o.logits).collect();
                let scores: Vec<f32> = objs.iter().map(|o| o.score).collect();
                let maskmem = model.tracker.encode_memory(&feats, &lrefs, &scores, &vec![false; n], false)?;
                if debug && b == 0 && step <= 3 {
                    self.dbg(model, &format!("f{t}_maskmem"), &maskmem)?;
                }
                mems[b].insert(step, FrameMem { maskmem, feat: feats.feat.clone(), ptr });
                if step > 16 {
                    mems[b].remove(&(step - 16));
                }
                let mut lg = Vec::with_capacity(n);
                for o in &objs {
                    lg.push(o.logits.to_f32_vec(&model.dev)?);
                }
                prev[b] = lg.clone();
                per_bucket.push(lg);
            }
            out.push((t, per_bucket));
            if !crate::log::quiet() && (step % 10 == 0 || step == order.len()) {
                eprintln!("  {label} {step}/{} ({:.0} ms/frame)", order.len(), ms(t0) / step as f64);
            }
        }
        Ok(out)
    }

    /// Track the prompts through the clip and write DIR/<id>/<frame>.png. `single`: only the prompted frame.
    #[allow(clippy::too_many_arguments)]
    pub fn track(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, out: Option<&str>, opts: &WriteOpts, vopts: &TrackOpts, single: bool) -> Result<Value> {
        let t_all = Instant::now();
        let Tracked { objs, frame: f, logits: merged } = self.compute(model, prompts, threshold, vopts, single)?;
        let nf = self.frames.len();
        let dir = match out {
            Some(o) => PathBuf::from(o),
            None => {
                let ip = Path::new(&self.path);
                let stem = ip.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "video".into());
                ip.with_file_name(format!("{stem}_masks"))
            }
        };
        // write masks
        let t = Instant::now();
        let mut rec = base_record(prompts, threshold, &self.path, self.width, self.height, self.meta.as_ref());
        rec.set("frames", nf as u64).set("fps", self.fps).set("loop", vopts.looped).set("bidir", vopts.bidir).set("smooth", vopts.smooth as u64).set("sticky", vopts.sticky as f64);
        let written = self.write(model, &objs, &merged, &dir, opts, &rec, single)?;
        info!("wrote {written} mask(s) in {:.0} ms; total {:.1} s", ms(t), t_all.elapsed().as_secs_f64());
        let mut jobjs = Vec::new();
        for (o_i, o) in objs.iter().enumerate() {
            let m = model.mask_u8(&Tensor::from_f32(&model.dev, &merged[&f][o_i], &[P])?, self.width, self.height)?;
            jobjs.push(json!({
                "id": o.id,
                "label": o.label,
                "score": (o.score as f64 * 1e4).round() / 1e4,
                "bbox": mask_bbox(&m, self.width, self.height).map(|b| json!(b)).unwrap_or(Value::Null),
                "frames": dir.join(o.id.to_string()).display().to_string(),
            }));
        }
        Ok(json!({"width": self.width, "height": self.height, "frames": nf, "fps": self.fps, "model": MODEL_NAME, "prompt_frame": f, "objects": jobjs}))
    }

    /// Condition on the prompted frame, propagate, merge and smooth (no output written).
    pub fn compute(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, vopts: &TrackOpts, single: bool) -> Result<Tracked> {
        let frames: Vec<usize> = prompts.iter().map(|p| p.frame.unwrap_or(0)).collect();
        let f = frames.first().copied().unwrap_or(0);
        if frames.iter().any(|x| *x != f) {
            bail!("all video prompts must be on the same frame for now (got frames {frames:?})");
        }
        let nf = self.frames.len();
        if f >= nf {
            bail!("--frame {f} is past the end of the clip ({nf} frames)");
        }
        let t = Instant::now();
        let (objs, feats) = self.condition(model, prompts, threshold, f)?;
        info!("prompted frame {f}: {} object(s) in {:.0} ms", objs.len(), ms(t));
        let n = objs.len();
        let mut est: Estimates = BTreeMap::new();
        let cond_logits: Vec<Vec<f32>> = objs.iter().map(|o| o.logits.to_f32_vec(&model.dev)).collect::<Result<_>>()?;
        est.entry(f).or_default().push((0, cond_logits));
        if n > 0 && !single {
            // buckets: the official demo gives every clicked object its own bucket (independent memory) and
            // multiplexes a batch of detector instances 16 to a bucket
            let mut buckets: Vec<Vec<usize>> = Vec::new();
            for (i, o) in objs.iter().enumerate() {
                match buckets.last_mut() {
                    Some(b) if o.concept && objs[b[0]].concept && objs[b[0]].label == o.label && b.len() < SLOTS => b.push(i),
                    _ => buckets.push(vec![i]),
                }
            }
            // the prompted frame's memory: non-overlap across every object, then encoded per bucket
            let lrefs: Vec<&Tensor> = objs.iter().map(|o| &o.logits).collect();
            let hi = model.tracker.hires(&lrefs, true)?;
            let fwd: Vec<usize> = if vopts.looped { (1..nf).map(|k| (f + k) % nf).collect() } else { (f + 1..nf).collect() };
            let bwd: Vec<usize> = if vopts.looped { (1..nf).map(|k| (f + nf - k) % nf).collect() } else { (0..f).rev().collect() };
            let mut conds = Vec::new();
            for (bi, b) in buckets.iter().enumerate() {
                let scores: Vec<f32> = b.iter().map(|i| objs[*i].obj_score).collect();
                let rows = hi.rows(b[0], b.len());
                let maskmem = model.tracker.encode_hires(&feats, &rows, &scores, &vec![true; b.len()])?;
                let mut ptr = vec![0f32; SLOTS * 256];
                for (s, i) in b.iter().enumerate() {
                    ptr[s * 256..(s + 1) * 256].copy_from_slice(&objs[*i].ptr);
                }
                if bi == 0 {
                    self.dbg(model, "c0_maskmem", &maskmem)?;
                    self.dbg_host("c0_objptr", &[SLOTS, 256], &ptr)?;
                }
                let cl: Vec<Vec<f32>> = b.iter().map(|i| est[&f][0].1[*i].clone()).collect();
                conds.push((FrameMem { maskmem, feat: feats.feat.clone(), ptr }, cl));
            }
            if buckets.len() > 1 {
                info!("{} objects in {} tracker buckets", n, buckets.len());
            }
            for (order, label, on) in [(&fwd, "forward", true), (&bwd, "backward", vopts.bidir || vopts.looped)] {
                if !on || order.is_empty() {
                    continue;
                }
                for (i, (t, per_bucket)) in self.run_pass(model, &conds, order, label, true, vopts.sticky)?.into_iter().enumerate() {
                    let key = i + 1 + if label == "backward" { 1_000_000 } else { 0 };
                    let mut all = vec![Vec::new(); n];
                    for (b, lg) in buckets.iter().zip(per_bucket) {
                        for (k, oi) in b.iter().enumerate() {
                            all[*oi] = lg[k].clone();
                        }
                    }
                    est.entry(t).or_default().push((key, all));
                }
            }
            // pass keys encode the direction; turn them back into distances
            for list in est.values_mut() {
                for (d, _) in list.iter_mut() {
                    if *d >= 1_000_000 {
                        *d -= 1_000_000;
                    }
                }
            }
        }
        // merge the estimates of each frame (weights fall off with the distance to the prompt)
        let mut merged: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
        for (t, list) in &est {
            if list.len() == 1 {
                merged.insert(*t, list[0].1.clone());
                continue;
            }
            let ws: Vec<f32> = list.iter().map(|(d, _)| 1.0 / (1.0 + *d as f32)).collect();
            let wsum: f32 = ws.iter().sum();
            let mut m = vec![vec![0f32; P]; n];
            for ((_, lg), w) in list.iter().zip(&ws) {
                for o in 0..n {
                    // absent objects carry -1024: clamp so one absent estimate does not swamp a present one
                    for (acc, v) in m[o].iter_mut().zip(&lg[o]) {
                        *acc += w / wsum * v.max(-32.0);
                    }
                }
            }
            merged.insert(*t, m);
        }
        if vopts.smooth > 1 && merged.len() > 2 {
            merged = smooth(&merged, n, vopts.smooth, vopts.looped, nf, f);
        }
        Ok(Tracked { objs, frame: f, logits: merged })
    }

    #[allow(clippy::too_many_arguments)]
    fn write(&self, model: &Model, objs: &[CondObj], merged: &BTreeMap<usize, Vec<Vec<f32>>>, dir: &Path, opts: &WriteOpts, rec: &Record, single: bool) -> Result<usize> {
        let (w, h) = (self.width, self.height);
        let nf = self.frames.len();
        for o in objs {
            std::fs::create_dir_all(dir.join(o.id.to_string())).with_context(|| format!("creating {}", dir.display()))?;
        }
        let empty = vec![0u8; w * h];
        let mut jobs: Vec<(PathBuf, Vec<u8>, usize, String)> = Vec::new();
        let mut written = 0;
        let flush = |jobs: &mut Vec<(PathBuf, Vec<u8>, usize, String)>| -> Result<()> {
            let chunk = std::mem::take(jobs);
            let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
            let per = (chunk.len() + nt - 1) / nt.max(1);
            std::thread::scope(|sc| -> Result<()> {
                let mut hs = Vec::new();
                for part in chunk.chunks(per.max(1)) {
                    hs.push(sc.spawn(move || -> Result<()> {
                        for (path, m, t, meta) in part {
                            let _ = t;
                            let text = vec![(genmeta::PNG_KEYWORD.to_string(), meta.clone())];
                            let bytes = crate::image::gray_png_bytes(w, h, m, &text)?;
                            std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
                        }
                        Ok(())
                    }));
                }
                for hnd in hs {
                    hnd.join().map_err(|_| anyhow::anyhow!("PNG writer panicked"))??;
                }
                Ok(())
            })
        };
        for t in 0..nf {
            if single && !merged.contains_key(&t) {
                continue;
            }
            for (oi, o) in objs.iter().enumerate() {
                let mut m = match merged.get(&t) {
                    Some(lg) => model.mask_u8(&Tensor::from_f32(&model.dev, &lg[oi], &[P])?, w, h)?,
                    None => empty.clone(),
                };
                if opts.binary {
                    m.iter_mut().for_each(|v| *v = if *v >= 128 { 255 } else { 0 });
                }
                let mut r = rec.clone();
                r.set("object", json!({"id": o.id, "label": o.label})).set("frame", t as u64);
                jobs.push((dir.join(o.id.to_string()).join(format!("{t:05}.png")), m, t, r.to_json()));
                written += 1;
            }
            if jobs.len() >= 64 {
                flush(&mut jobs)?;
            }
        }
        flush(&mut jobs)?;
        Ok(written)
    }
}

/// Triangular temporal filter of radius n/2 over each object's logits (wrapping when looped); the prompted
/// frame is kept as is.
fn smooth(m: &BTreeMap<usize, Vec<Vec<f32>>>, nobj: usize, n: usize, looped: bool, nf: usize, keep: usize) -> BTreeMap<usize, Vec<Vec<f32>>> {
    let r = (n / 2).max(1) as i64;
    let mut out = BTreeMap::new();
    for (&t, cur) in m {
        if t == keep {
            out.insert(t, cur.clone());
            continue;
        }
        let mut acc = vec![vec![0f32; P]; nobj];
        let mut wsum = 0f32;
        for d in -r..=r {
            let mut u = t as i64 + d;
            if looped {
                u = u.rem_euclid(nf as i64);
            }
            if u < 0 || u >= nf as i64 {
                continue;
            }
            let Some(lg) = m.get(&(u as usize)) else { continue };
            let w = (r + 1 - d.abs()) as f32;
            wsum += w;
            for o in 0..nobj {
                for (a, v) in acc[o].iter_mut().zip(&lg[o]) {
                    *a += w * v.max(-32.0);
                }
            }
        }
        for o in 0..nobj {
            for a in acc[o].iter_mut() {
                *a /= wsum;
            }
        }
        out.insert(t, acc);
    }
    out
}

pub fn run_cli(cli: &Cli, input: &str, prompts: &[ObjPrompt], meta: Option<&Value>) -> Result<()> {
    if cli.out.as_deref() == Some("-") {
        bail!("video masks are a directory of PNGs; -o - is for images");
    }
    if cli.everything {
        bail!("--everything is not available for video");
    }
    let model = crate::cli::load_model(cli.model.clone())?;
    let mut sess = VideoSession::open(&model, input)?;
    sess.meta = meta.cloned();
    let vopts = TrackOpts { looped: cli.looped, bidir: !cli.no_bidir, smooth: cli.smooth, sticky: cli.sticky };
    let r = sess.track(&model, prompts, cli.threshold, cli.out.as_deref(), &WriteOpts { alpha: cli.alpha, binary: cli.binary }, &vopts, false)?;
    if cli.json {
        println!("{r}");
    } else {
        for o in r["objects"].as_array().cloned().unwrap_or_default() {
            println!("{}", o["frames"].as_str().unwrap_or(""));
        }
    }
    Ok(())
}

