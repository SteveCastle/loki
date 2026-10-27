//! Video: frames through ffmpeg, SAM 3.1 tracking from one or more prompted frames, and the temporal-stability layer
//! on top of the stock tracker:
//!   - bidirectional: a forward pass from the first prompted frame and a backward pass from the last one; frames
//!     between prompts get both estimates, merged by distance to the conditioning frame (default on);
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

/// One conditioning (prompted) frame of an object.
pub struct CondEntry {
    /// low-res logits [288*288] on the device
    pub logits: Tensor,
    /// object score logit fed to the memory encoder
    pub obj_score: f32,
    /// object pointer [256]
    pub ptr: Vec<f32>,
    /// reported score (detector probability for concepts, presence probability for clicks)
    pub score: f32,
    /// clicks/box: the 3 ambiguity candidates, best first (score, logits)
    pub candidates: Vec<(f32, Tensor)>,
}

/// A tracked object and the frames it was prompted on.
pub struct VObj {
    pub id: u32,
    pub label: String,
    /// instances of one concept prompt share a tracker bucket (multiplexed); None = own bucket
    pub group: Option<usize>,
    pub conds: BTreeMap<usize, CondEntry>,
}

/// Tracking result: the objects and, per frame, their (merged, smoothed) low-res logits [288*288] per object.
pub struct Tracked {
    pub objs: Vec<VObj>,
    /// every prompted frame, ascending
    pub cond_frames: Vec<usize>,
    pub logits: BTreeMap<usize, Vec<Vec<f32>>>,
}

/// A bucket's conditioning frames: frame -> (memory, logits of the bucket's objects).
type BucketConds = BTreeMap<usize, (FrameMem, Vec<Vec<f32>>)>;

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

    /// Resolve the prompts of one frame into conditioning entries: (object id, label, concept group, entry).
    /// `next_id` hands out ids to objects without one; concept prompts get group ids from `next_group`.
    #[allow(clippy::type_complexity)]
    fn condition(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, frame: usize, next_id: &mut u32, next_group: &mut usize, need_ptr: bool) -> Result<(Vec<(u32, String, Option<usize>, CondEntry)>, FrameFeats)> {
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
        let mut out: Vec<(u32, String, Option<usize>, CondEntry)> = Vec::new();
        let mut fresh = || {
            *next_id += 1;
            *next_id - 1
        };
        let mut geo_on_frame = 0;
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
                let mut o = model.inter.decode(&enc.sam, &pts, pts.len() == 1)?;
                let candidates = o.ranked().into_iter().map(|k| Ok((crate::detector::sigmoid(o.ious[k]), o.row(&model.dev, k)?))).collect::<Result<Vec<_>>>()?;
                if let Some(c) = p.candidate {
                    o.choose(&model.dev, c)?;
                }
                let id = p.obj.unwrap_or_else(&mut fresh);
                let label = p.text.clone().unwrap_or_else(|| if p.bbox.is_some() { "box".into() } else { "click".into() });
                let score = crate::detector::sigmoid(o.obj_score);
                let entry = if self.parity && geo_on_frame > 0 {
                    // official demo: objects added to an existing bucket enter through the mask-input path
                    let (lg, ptr, sc) = self.mask_object_from_logits(model, &enc, &o.mask)?;
                    CondEntry { logits: lg, obj_score: sc, ptr, score, candidates }
                } else {
                    let (ptr, sc) = model.tracker.cond_object(&o);
                    let lg = self.via_video_res(model, &o.mask)?;
                    CondEntry { logits: lg, obj_score: sc, ptr, score, candidates }
                };
                geo_on_frame += 1;
                out.push((id, label, None, entry));
            } else if let Some(t) = &p.text {
                // concept: every detected instance enters the tracker through the mask-input path
                let group = *next_group;
                *next_group += 1;
                for (i, (score, logits)) in model.concept(&enc, t, threshold)?.into_iter().enumerate() {
                    let id = match p.obj {
                        Some(o) if i == 0 => o,
                        _ => fresh(),
                    };
                    let (lg, ptr, sc) = self.mask_object(model, &enc, &logits, need_ptr)?;
                    out.push((id, t.clone(), Some(group), CondEntry { logits: lg, obj_score: sc, ptr, score, candidates: Vec::new() }));
                }
            } else {
                bail!("a prompt needs text, points or a box");
            }
        }
        Ok((out, feats))
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
    /// `need_ptr`: false for a preview (the pointer, a host-side mask-prompt pass, is only needed for tracking).
    fn mask_object(&self, model: &Model, enc: &crate::model::Encoded, logits288: &Tensor, need_ptr: bool) -> Result<(Tensor, Vec<f32>, f32)> {
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
        let mut ptr = Vec::new();
        if need_ptr {
            let dense = model.inter.mask_prompt(&m1008);
            let o = model.inter.decode_with(&enc.sam, &[Point { x: 0.0, y: 0.0, label: -1 }], false, Some(&dense))?;
            ptr = model.tracker.cond_object(&o).0;
            if !appearing {
                ptr = model.tracker.no_obj_pointer(&ptr);
            }
        }
        // output mask: binary x 20 - 10 (low res: the area-average of the 1008 mask, as the official downsample)
        let bin: Vec<f32> = crate::image::resample_plane(&m1008, IMG, IMG, LOW, LOW).iter().map(|v| v * 20.0 - 10.0).collect();
        let _ = lo;
        let sc = if appearing { 10.0 } else { -10.0 };
        Ok((Tensor::from_f32(dev, &bin, &[P])?, ptr, sc))
    }

    /// One propagation pass over `order` (physical frame indices; position k in the pass = order[k]) for every bucket
    /// at once, so each frame's backbone runs once. A bucket starts at its first conditioning frame in the pass; at
    /// its conditioning frames the output is the prompt's mask. Memory per frame, as upstream: the (up to 4) closest
    /// conditioning frames, before or after, then the 6 most recent tracked frames; object pointers of the
    /// conditioning frames and the last 15 frames. Returns per position: frame, per bucket Some((distance to the last
    /// conditioning frame, per-object logits)) or None before the bucket starts.
    #[allow(clippy::type_complexity)]
    fn run_pass(&mut self, model: &Model, conds: &[BucketConds], order: &[usize], sticky: f32, progress: &mut Progress) -> Result<Vec<(usize, Vec<Option<(usize, Vec<Vec<f32>>)>>)>> {
        let num_frames = self.frames.len();
        let nb = conds.len();
        let pos_of: HashMap<usize, usize> = order.iter().enumerate().map(|(k, f)| (*f, k)).collect();
        // per bucket: conditioning position -> frame
        let cpos: Vec<BTreeMap<usize, usize>> = conds.iter().map(|c| c.keys().filter_map(|f| pos_of.get(f).map(|k| (*k, *f))).collect()).collect();
        let start: Vec<Option<usize>> = cpos.iter().map(|c| c.keys().next().copied()).collect();
        let mut mems: Vec<BTreeMap<usize, FrameMem>> = vec![BTreeMap::new(); nb];
        let mut prev: Vec<Vec<Vec<f32>>> = vec![Vec::new(); nb];
        let mut last_cond: Vec<usize> = vec![0; nb];
        let mut out = Vec::with_capacity(order.len());
        let mut dbg_steps = 0;
        for (k, &t) in order.iter().enumerate() {
            let mut feats: Option<FrameFeats> = None;
            let mut per_bucket = Vec::with_capacity(nb);
            for b in 0..nb {
                let Some(s) = start[b] else {
                    per_bucket.push(None);
                    continue;
                };
                if k < s {
                    per_bucket.push(None);
                    continue;
                }
                if let Some(f) = cpos[b].get(&k) {
                    let lg = conds[b][f].1.clone();
                    prev[b] = lg.clone();
                    last_cond[b] = k;
                    per_bucket.push(Some((0, lg)));
                    continue;
                }
                if feats.is_none() {
                    feats = Some(self.feats(model, t)?);
                }
                let fe = feats.as_ref().unwrap();
                let n = conds[b].values().next().map(|c| c.1.len()).unwrap_or(0);
                // conditioning frames: up to 4 closest (closest before, closest after, then by distance)
                let all: Vec<usize> = cpos[b].keys().copied().collect();
                let mut sel: Vec<usize> = Vec::new();
                if all.len() <= 4 {
                    sel = all.clone();
                } else {
                    if let Some(c) = all.iter().copied().filter(|c| *c < k).max() {
                        sel.push(c);
                    }
                    if let Some(c) = all.iter().copied().filter(|c| *c >= k).min() {
                        sel.push(c);
                    }
                    let mut rest: Vec<usize> = all.iter().copied().filter(|c| !sel.contains(c)).collect();
                    rest.sort_by_key(|c| (*c as i64 - k as i64).abs());
                    for c in rest {
                        if sel.len() >= 4 {
                            break;
                        }
                        sel.push(c);
                    }
                }
                let cond_mem = |p: usize| -> Option<&FrameMem> { cpos[b].get(&p).map(|f| &conds[b][f].0) };
                let mut mem: Vec<(&FrameMem, i64)> = sel.iter().map(|c| (cond_mem(*c).unwrap(), k as i64 - *c as i64)).collect();
                for t_pos in 1..7usize {
                    let t_rel = 7 - t_pos;
                    if k >= t_rel {
                        let p = k - t_rel;
                        if let Some(m) = mems[b].get(&p) {
                            mem.push((m, t_pos as i64));
                        } else if !sel.contains(&p) {
                            if let Some(m) = cond_mem(p) {
                                mem.push((m, t_pos as i64));
                            }
                        }
                    }
                }
                let mut ptrs: Vec<(&[f32], usize)> = sel.iter().map(|c| (cond_mem(*c).unwrap().ptr.as_slice(), (k as i64 - *c as i64).unsigned_abs() as usize)).collect();
                for t_diff in 1..num_frames.min(16) {
                    if t_diff > k {
                        break;
                    }
                    let p = k - t_diff;
                    if let Some(m) = mems[b].get(&p) {
                        ptrs.push((&m.ptr, t_diff));
                    } else if !sel.contains(&p) {
                        if let Some(m) = cond_mem(p) {
                            ptrs.push((&m.ptr, t_diff));
                        }
                    }
                }
                let pix = model.tracker.attend_memory(fe, &mem, &ptrs, num_frames)?;
                let (objs, ptr) = model.tracker.decode(fe, &pix, n, if sticky > 0.0 && prev[b].len() == n { Some((&prev[b], sticky)) } else { None })?;
                let dbg = b == 0 && dbg_steps < 3;
                if dbg {
                    self.dbg(model, &format!("f{t}_pixmem"), &pix)?;
                    self.dbg_host(&format!("f{t}_objptr"), &[SLOTS, 256], &ptr)?;
                }
                let lrefs: Vec<&Tensor> = objs.iter().map(|o| &o.logits).collect();
                let scores: Vec<f32> = objs.iter().map(|o| o.score).collect();
                let maskmem = model.tracker.encode_memory(fe, &lrefs, &scores, &vec![false; n], false)?;
                if dbg {
                    self.dbg(model, &format!("f{t}_maskmem"), &maskmem)?;
                    dbg_steps += 1;
                }
                mems[b].insert(k, FrameMem { maskmem, feat: fe.feat.clone(), ptr });
                if k >= 16 {
                    mems[b].remove(&(k - 16));
                }
                let mut lg = Vec::with_capacity(n);
                for o in &objs {
                    lg.push(o.logits.to_f32_vec(&model.dev)?);
                }
                prev[b] = lg.clone();
                per_bucket.push(Some((k - last_cond[b], lg)));
            }
            out.push((t, per_bucket));
            progress.tick();
            crate::worker::check_cancel()?;
        }
        Ok(out)
    }

    /// Track the prompts through the clip and write DIR/<id>/<frame>.png. `single`: only the prompted frames.
    #[allow(clippy::too_many_arguments)]
    pub fn track(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, out: Option<&str>, opts: &WriteOpts, vopts: &TrackOpts, single: bool) -> Result<Value> {
        let t_all = Instant::now();
        let Tracked { objs, cond_frames, logits: merged } = self.compute(model, prompts, threshold, vopts, single)?;
        let nf = self.frames.len();
        let dir = match out {
            Some(o) => PathBuf::from(o),
            None => {
                let ip = Path::new(&self.path);
                let stem = ip.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "video".into());
                ip.with_file_name(format!("{stem}_masks"))
            }
        };
        let t = Instant::now();
        let mut rec = base_record(prompts, threshold, &self.path, self.width, self.height, self.meta.as_ref());
        rec.set("frames", nf as u64).set("fps", self.fps).set("loop", vopts.looped).set("bidir", vopts.bidir).set("smooth", vopts.smooth as u64).set("sticky", vopts.sticky as f64);
        let written = self.write(model, &objs, &merged, &dir, opts, &rec, single)?;
        info!("wrote {written} mask(s) in {:.0} ms; total {:.1} s", ms(t), t_all.elapsed().as_secs_f64());
        let mut jobjs = Vec::new();
        for (o_i, o) in objs.iter().enumerate() {
            let (f0, first) = o.conds.iter().next().unwrap();
            let m = model.mask_u8(&Tensor::from_f32(&model.dev, &merged[f0][o_i], &[P])?, self.width, self.height)?;
            let mut j = json!({
                "id": o.id,
                "label": o.label,
                "score": (first.score as f64 * 1e4).round() / 1e4,
                "bbox": mask_bbox(&m, self.width, self.height).map(|b| json!(b)).unwrap_or(Value::Null),
                "frames": dir.join(o.id.to_string()).display().to_string(),
                "prompt_frames": o.conds.keys().collect::<Vec<_>>(),
            });
            if opts.multimask {
                let mut cands = Vec::new();
                for (f, c) in &o.conds {
                    for (k, (score, lg)) in c.candidates.iter().enumerate() {
                        let mut cm = model.mask_u8(lg, self.width, self.height)?;
                        if opts.binary {
                            cm.iter_mut().for_each(|v| *v = if *v >= 128 { 255 } else { 0 });
                        }
                        let p = dir.join(o.id.to_string()).join(format!("{f:05}_c{}.png", k + 1));
                        let mut r = rec.clone();
                        r.set("object", json!({"id": o.id, "label": o.label, "candidate": k, "score": score})).set("frame", *f as u64);
                        let bytes = crate::image::gray_png_bytes(self.width, self.height, &cm, &[(genmeta::PNG_KEYWORD.to_string(), r.to_json())])?;
                        std::fs::write(&p, bytes).with_context(|| format!("writing {}", p.display()))?;
                        cands.push(json!({"frame": f, "mask": p.display().to_string(), "score": (*score as f64 * 1e4).round() / 1e4,
                            "bbox": mask_bbox(&cm, self.width, self.height).map(|b| json!(b)).unwrap_or(Value::Null)}));
                    }
                }
                if !cands.is_empty() {
                    j["candidates"] = Value::Array(cands);
                }
            }
            jobjs.push(j);
        }
        let f0 = cond_frames.first().copied().unwrap_or(0);
        Ok(json!({"width": self.width, "height": self.height, "frames": nf, "fps": self.fps, "model": MODEL_NAME, "prompt_frame": f0, "prompt_frames": cond_frames, "objects": jobjs}))
    }

    /// Condition on every prompted frame, propagate forward and backward, merge and smooth (nothing written).
    pub fn compute(&mut self, model: &Model, prompts: &[ObjPrompt], threshold: f32, vopts: &TrackOpts, single: bool) -> Result<Tracked> {
        let nf = self.frames.len();
        // normalize: default frame 0; entries for the same object on the same frame are merged (clicks add up)
        let mut entries: BTreeMap<usize, Vec<ObjPrompt>> = BTreeMap::new();
        for p in prompts {
            let f = p.frame.unwrap_or(0);
            if f >= nf {
                bail!("frame {f} is past the end of the clip ({nf} frames)");
            }
            let list = entries.entry(f).or_default();
            let geo = !p.points.is_empty() || p.bbox.is_some();
            if let (Some(id), true) = (p.obj, geo) {
                if let Some(e) = list.iter_mut().find(|e| e.obj == Some(id) && (!e.points.is_empty() || e.bbox.is_some())) {
                    e.points.extend(p.points.iter().cloned());
                    if p.bbox.is_some() {
                        e.bbox = p.bbox.clone();
                    }
                    if e.text.is_none() {
                        e.text = p.text.clone();
                    }
                    if p.candidate.is_some() {
                        e.candidate = p.candidate;
                    }
                    continue;
                }
            }
            let mut q = p.clone();
            q.frame = Some(f);
            list.push(q);
        }
        let mut next_id = prompts.iter().filter_map(|p| p.obj).max().unwrap_or(0) + 1;
        let mut next_group = 0usize;
        let mut objs: Vec<VObj> = Vec::new();
        let mut feats_at: BTreeMap<usize, FrameFeats> = BTreeMap::new();
        for (&f, list) in &entries {
            let t = Instant::now();
            let (found, feats) = self.condition(model, list, threshold, f, &mut next_id, &mut next_group, !single)?;
            info!("prompted frame {f}: {} object(s) in {:.0} ms", found.len(), ms(t));
            feats_at.insert(f, feats);
            for (id, label, group, entry) in found {
                match objs.iter_mut().find(|o| o.id == id) {
                    Some(o) => {
                        // another prompted frame for a known object: it gets its own bucket (its own memory)
                        o.group = None;
                        o.conds.insert(f, entry);
                    }
                    None => objs.push(VObj { id, label, group, conds: BTreeMap::from([(f, entry)]) }),
                }
            }
        }
        let n = objs.len();
        let cond_frames: Vec<usize> = feats_at.keys().copied().collect();
        // the prompt masks are the outputs of their frames
        let mut est: BTreeMap<usize, Vec<Vec<(usize, Vec<f32>)>>> = BTreeMap::new();
        for (oi, o) in objs.iter().enumerate() {
            for (f, c) in &o.conds {
                est.entry(*f).or_insert_with(|| vec![Vec::new(); n])[oi].push((0, c.logits.to_f32_vec(&model.dev)?));
            }
        }
        if n > 0 && !single {
            // buckets: every clicked / multi-frame object alone (own memory, as the official demo does); the
            // instances of one concept prompt multiplexed, 16 to a bucket
            let mut buckets: Vec<Vec<usize>> = Vec::new();
            for (i, o) in objs.iter().enumerate() {
                match buckets.iter_mut().find(|b| o.group.is_some() && objs[b[0]].group == o.group && b.len() < SLOTS) {
                    Some(b) => b.push(i),
                    None => buckets.push(vec![i]),
                }
            }
            if buckets.len() > 1 {
                info!("{n} objects in {} tracker buckets", buckets.len());
            }
            // conditioning memories: per prompted frame, non-overlap across the objects prompted there, then per bucket
            let mut bconds: Vec<BucketConds> = vec![BTreeMap::new(); buckets.len()];
            let mut first_dbg = true;
            for (&f, feats) in &feats_at {
                let here: Vec<usize> = (0..n).filter(|i| objs[*i].conds.contains_key(&f)).collect();
                let lrefs: Vec<&Tensor> = here.iter().map(|i| &objs[*i].conds[&f].logits).collect();
                let hi = model.tracker.hires(&lrefs, true)?;
                let p_hi = IMG * IMG;
                for (bi, b) in buckets.iter().enumerate() {
                    if !b.iter().all(|i| objs[*i].conds.contains_key(&f)) {
                        continue;
                    }
                    let rows = Tensor::new(&model.dev, DType::F32, &[b.len(), p_hi])?;
                    for (r, i) in b.iter().enumerate() {
                        let src = here.iter().position(|h| h == i).unwrap();
                        model.dev.dtod(rows.ptr + (r * p_hi * 4) as u64, hi.ptr + (src * p_hi * 4) as u64, p_hi * 4)?;
                    }
                    let scores: Vec<f32> = b.iter().map(|i| objs[*i].conds[&f].obj_score).collect();
                    let maskmem = model.tracker.encode_hires(feats, &rows, &scores, &vec![true; b.len()])?;
                    let mut ptr = vec![0f32; SLOTS * 256];
                    for (s, i) in b.iter().enumerate() {
                        ptr[s * 256..(s + 1) * 256].copy_from_slice(&objs[*i].conds[&f].ptr);
                    }
                    if bi == 0 && first_dbg {
                        self.dbg(model, "c0_maskmem", &maskmem)?;
                        self.dbg_host("c0_objptr", &[SLOTS, 256], &ptr)?;
                        first_dbg = false;
                    }
                    let lg: Vec<Vec<f32>> = b.iter().map(|i| est[&f][*i][0].1.clone()).collect();
                    bconds[bi].insert(f, (FrameMem { maskmem, feat: feats.feat.clone(), ptr }, lg));
                }
            }
            let (fmin, fmax) = (cond_frames[0], *cond_frames.last().unwrap());
            let fwd: Vec<usize> = if vopts.looped { (0..nf).map(|k| (fmin + k) % nf).collect() } else { (fmin..nf).collect() };
            let bwd: Vec<usize> = if vopts.looped { (0..nf).map(|k| (fmax + nf - k) % nf).collect() } else { (0..=fmax).rev().collect() };
            let do_bwd = vopts.bidir || vopts.looped;
            let mut progress = Progress::new(fwd.len() + if do_bwd { bwd.len() } else { 0 });
            for (order, on) in [(&fwd, true), (&bwd, do_bwd)] {
                if !on {
                    continue;
                }
                for (t, per_bucket) in self.run_pass(model, &bconds, order, vopts.sticky, &mut progress)? {
                    for (b, r) in buckets.iter().zip(per_bucket) {
                        let Some((d, lg)) = r else { continue };
                        if d == 0 {
                            continue; // the prompt mask itself, already recorded
                        }
                        let slot = est.entry(t).or_insert_with(|| vec![Vec::new(); n]);
                        for (k, oi) in b.iter().enumerate() {
                            slot[*oi].push((d, lg[k].clone()));
                        }
                    }
                }
            }
        }
        // merge each frame's estimates: a prompt mask wins; otherwise weights fall off with the distance to the
        // conditioning frame each estimate was propagated from
        let mut merged: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
        for (t, per_obj) in &est {
            let mut m = Vec::with_capacity(n);
            for list in per_obj {
                if list.is_empty() {
                    m.push(vec![crate::interactive::NO_OBJ_SCORE; P]);
                } else if let Some((_, lg)) = list.iter().find(|(d, _)| *d == 0) {
                    m.push(lg.clone());
                } else if list.len() == 1 {
                    m.push(list[0].1.clone());
                } else {
                    let ws: Vec<f32> = list.iter().map(|(d, _)| 1.0 / (1.0 + *d as f32)).collect();
                    let wsum: f32 = ws.iter().sum();
                    let mut acc = vec![0f32; P];
                    for ((_, lg), w) in list.iter().zip(&ws) {
                        // absent objects carry -1024: clamp so one absent estimate does not swamp a present one
                        for (a, v) in acc.iter_mut().zip(lg) {
                            *a += w / wsum * v.max(-32.0);
                        }
                    }
                    m.push(acc);
                }
            }
            merged.insert(*t, m);
        }
        if vopts.smooth > 1 && merged.len() > 2 {
            merged = smooth(&merged, n, vopts.smooth, vopts.looped, nf, &cond_frames);
        }
        Ok(Tracked { objs, cond_frames, logits: merged })
    }

    #[allow(clippy::too_many_arguments)]
    fn write(&self, model: &Model, objs: &[VObj], merged: &BTreeMap<usize, Vec<Vec<f32>>>, dir: &Path, opts: &WriteOpts, rec: &Record, single: bool) -> Result<usize> {
        let (w, h) = (self.width, self.height);
        let nf = self.frames.len();
        for o in objs {
            std::fs::create_dir_all(dir.join(o.id.to_string())).with_context(|| format!("creating {}", dir.display()))?;
        }
        let empty = vec![0u8; w * h];
        let mut jobs: Vec<(PathBuf, Vec<u8>, String)> = Vec::new();
        let mut written = 0;
        let flush = |jobs: &mut Vec<(PathBuf, Vec<u8>, String)>| -> Result<()> {
            let chunk = std::mem::take(jobs);
            let nt = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(12);
            let per = (chunk.len() + nt - 1) / nt.max(1);
            std::thread::scope(|sc| -> Result<()> {
                let mut hs = Vec::new();
                for part in chunk.chunks(per.max(1)) {
                    hs.push(sc.spawn(move || -> Result<()> {
                        for (path, m, meta) in part {
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
                jobs.push((dir.join(o.id.to_string()).join(format!("{t:05}.png")), m, r.to_json()));
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

/// Progress of a tracking run on stderr, one line per frame step: `track 37/124` (also when stderr is not a
/// terminal, so callers can parse `(\d+)/(\d+)` from the last line). Silenced by -q.
pub struct Progress {
    done: usize,
    total: usize,
    t0: Instant,
}
impl Progress {
    fn new(total: usize) -> Progress {
        let p = Progress { done: 0, total, t0: Instant::now() };
        if !crate::log::quiet() {
            eprintln!("track 0/{total}");
        }
        p
    }
    fn tick(&mut self) {
        self.done += 1;
        if !crate::log::quiet() {
            eprintln!("track {}/{} ({:.0} ms/frame)", self.done, self.total, ms(self.t0) / self.done as f64);
        }
    }
}

/// Triangular temporal filter of radius n/2 over each object's logits (wrapping when looped); prompted frames are
/// kept as they are.
fn smooth(m: &BTreeMap<usize, Vec<Vec<f32>>>, nobj: usize, n: usize, looped: bool, nf: usize, keep: &[usize]) -> BTreeMap<usize, Vec<Vec<f32>>> {
    let r = (n / 2).max(1) as i64;
    let mut out = BTreeMap::new();
    for (&t, cur) in m {
        if keep.contains(&t) {
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
    let r = sess.track(&model, prompts, cli.threshold, cli.out.as_deref(), &WriteOpts { alpha: cli.alpha, binary: cli.binary, multimask: cli.multimask }, &vopts, false)?;
    if cli.json {
        crate::worker::out_line(&r.to_string());
    } else {
        for o in r["objects"].as_array().cloned().unwrap_or_default() {
            crate::worker::out_line(o["frames"].as_str().unwrap_or(""));
        }
    }
    Ok(())
}

