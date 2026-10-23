//! The engine facade: loads SAM 3.1 once, encodes an image (trunk + necks + per-head preparation, the expensive
//! part, ~70 ms), then answers any number of prompts against that encoding (concept text: ~15 ms; clicks: ~3 ms).
//! `--serve` keeps one of these alive across requests.
use crate::cuda::{Arg, Device};
use crate::detector::{DetImage, Detector};
use crate::image::Rgb8;
use crate::interactive::{Interactive, Point, SamImage};
use crate::tensor::{DType, Tensor};
use crate::text::{TextEncoder, TextFeatures};
use crate::vit::{Neck, Vit, IMG};
use crate::weights::Weights;
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

pub const MODEL_NAME: &str = "SAM 3.1";

pub struct Model {
    pub dev: Arc<Device>,
    pub weights: Weights,
    pub vit: Vit,
    pub neck_det: Neck,
    pub neck_inter: Neck,
    pub neck_prop: Neck,
    pub text: TextEncoder,
    pub det: Detector,
    pub inter: Interactive,
    pub tracker: crate::tracker::Tracker,
    text_cache: std::cell::RefCell<HashMap<String, TextFeatures>>,
}

/// One encoded image (or video frame).
pub struct Encoded {
    pub width: usize,
    pub height: usize,
    pub det: DetImage,
    pub sam: SamImage,
    /// propagation neck levels (288, 144, 72), kept for video tracking
    pub prop: Option<Vec<Tensor>>,
}

/// One object to segment, in input-image pixels. Mirrors the `--prompts` JSON entries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObjPrompt {
    #[serde(default)]
    pub obj: Option<u32>,
    #[serde(default)]
    pub frame: Option<usize>,
    #[serde(default)]
    pub text: Option<String>,
    /// [[x, y, label], ...] label 1 positive / 0 negative
    #[serde(default)]
    pub points: Vec<Vec<f32>>,
    /// [x, y, w, h]
    #[serde(default, rename = "box")]
    pub bbox: Option<Vec<f32>>,
}

/// A segmented object: soft mask logits at 288x288 (model space) plus its metadata.
pub struct Segment {
    pub id: u32,
    pub label: String,
    pub score: f32,
    /// f32 logits [288*288] on the device
    pub logits: Tensor,
}

impl Model {
    pub fn load(path: &Path, dev: Arc<Device>) -> Result<Model> {
        crate::ops::init(&dev)?;
        crate::ops::attn_init(&dev)?;
        let w = Weights::open(path, dev.clone())?;
        let vb = "detector.backbone.vision_backbone";
        let vit = Vit::load(&w, &format!("{vb}.trunk"))?;
        let neck_det = Neck::load(&w, &format!("{vb}.convs"))?;
        let neck_inter = Neck::load(&w, &format!("{vb}.interactive_convs"))?;
        let neck_prop = Neck::load(&w, &format!("{vb}.propagation_convs"))?;
        let text = TextEncoder::load(&w, "detector.backbone.language_backbone")?;
        let det = Detector::load(&w, "detector")?;
        let inter = Interactive::load(&w, "tracker.model")?;
        let tracker = crate::tracker::Tracker::load(&w, "tracker.model")?;
        Ok(Model { dev, weights: w, vit, neck_det, neck_inter, neck_prop, text, det, inter, tracker, text_cache: Default::default() })
    }

    /// Resize (antialiased bilinear, squashed to 1008x1008), normalize to [-1, 1], CHW f32 on the device.
    pub fn preprocess(&self, img: &Rgb8) -> Result<Tensor> {
        let px = crate::image::resize_bilinear_aa(img, IMG, IMG);
        let mut chw = vec![0f32; 3 * IMG * IMG];
        for i in 0..IMG * IMG {
            for c in 0..3 {
                chw[c * IMG * IMG + i] = (px[i * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        Tensor::from_f32(&self.dev, &chw, &[3, IMG, IMG])
    }

    pub fn encode(&self, img: &Rgb8, with_propagation: bool) -> Result<Encoded> {
        let x = self.preprocess(img)?;
        self.encode_tensor(&x, img.w, img.h, with_propagation)
    }

    pub fn encode_tensor(&self, x: &Tensor, width: usize, height: usize, with_propagation: bool) -> Result<Encoded> {
        let dev = &*self.dev;
        let trunk = self.vit.forward(x)?;
        let fpn = self.neck_det.forward(dev, &trunk)?;
        let det = self.det.prepare(&fpn)?;
        let ifpn = self.neck_inter.forward(dev, &trunk)?;
        let sam = self.inter.prepare(&ifpn)?;
        let prop = if with_propagation { Some(self.neck_prop.forward(dev, &trunk)?) } else { None };
        Ok(Encoded { width, height, det, sam, prop })
    }

    pub fn text_features(&self, text: &str) -> Result<TextFeatures> {
        if let Some(t) = self.text_cache.borrow().get(text) {
            return Ok(t.clone());
        }
        let t = self.text.encode(text)?;
        self.text_cache.borrow_mut().insert(text.to_string(), t.clone());
        Ok(t)
    }

    /// Every instance of a concept with joint score above `threshold`, best first.
    pub fn concept(&self, enc: &Encoded, text: &str, threshold: f32) -> Result<Vec<(f32, Tensor)>> {
        let tf = self.text_features(text)?;
        let d = self.det.detect(&enc.det, &tf)?;
        let mut keep: Vec<(usize, f32)> = d.probs.iter().enumerate().filter(|(_, p)| **p > threshold).map(|(i, p)| (i, *p)).collect();
        keep.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if keep.is_empty() {
            return Ok(Vec::new());
        }
        let idx: Vec<usize> = keep.iter().map(|k| k.0).collect();
        let m = self.det.masks(&d, &idx)?;
        let p = 288 * 288;
        Ok(keep.iter().enumerate().map(|(i, k)| (k.1, m.rows(i, 1).reshape(&[p]))).collect())
    }

    /// One object from clicks / a box (input pixels).
    pub fn clicks(&self, enc: &Encoded, points: &[(f32, f32, i32)], bbox: Option<[f32; 4]>) -> Result<(f32, Tensor, crate::interactive::SamOutput)> {
        let (sx, sy) = (IMG as f32 / enc.width as f32, IMG as f32 / enc.height as f32);
        let mut pts: Vec<Point> = points.iter().map(|&(x, y, l)| Point { x: x * sx, y: y * sy, label: l }).collect();
        if let Some(b) = bbox {
            pts.push(Point { x: b[0] * sx, y: b[1] * sy, label: 2 });
            pts.push(Point { x: (b[0] + b[2]) * sx, y: (b[1] + b[3]) * sy, label: 3 });
        }
        let multimask = pts.len() == 1;
        let o = self.inter.decode(&enc.sam, &pts, multimask)?;
        // score: the decoder's object-presence probability (its IoU head is an uncalibrated regression)
        Ok((crate::detector::sigmoid(o.obj_score), o.mask.clone(), o))
    }

    /// Resolve a list of object prompts on one encoded image into segments (ids assigned in order; a concept
    /// prompt yields one segment per detected instance).
    pub fn segment(&self, enc: &Encoded, prompts: &[ObjPrompt], threshold: f32) -> Result<Vec<Segment>> {
        let mut out = Vec::new();
        let mut next_id = prompts.iter().filter_map(|p| p.obj).max().unwrap_or(0) + 1;
        for p in prompts {
            let has_geo = !p.points.is_empty() || p.bbox.is_some();
            if has_geo {
                let pts: Vec<(f32, f32, i32)> = p
                    .points
                    .iter()
                    .map(|v| (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).map(|l| *l as i32).unwrap_or(1)))
                    .collect();
                let bbox = match &p.bbox {
                    Some(b) if b.len() == 4 => Some([b[0], b[1], b[2], b[3]]),
                    Some(_) => bail!("a box is [x, y, w, h]"),
                    None => None,
                };
                let (score, logits, _) = self.clicks(enc, &pts, bbox)?;
                let id = p.obj.unwrap_or_else(|| {
                    next_id += 1;
                    next_id - 1
                });
                let label = p.text.clone().unwrap_or_else(|| if p.bbox.is_some() { "box".into() } else { "click".into() });
                out.push(Segment { id, label, score, logits });
            } else if let Some(t) = &p.text {
                let found = self.concept(enc, t, threshold)?;
                let n = found.len();
                for (i, (score, logits)) in found.into_iter().enumerate() {
                    // a fixed obj id names the first (best) instance; the others get fresh ids
                    let id = match p.obj {
                        Some(o) if i == 0 => o,
                        _ => {
                            next_id += 1;
                            next_id - 1
                        }
                    };
                    let label = if n > 1 { t.clone() } else { t.clone() };
                    out.push(Segment { id, label, score, logits });
                }
            } else {
                bail!("a prompt needs text, points or a box");
            }
        }
        Ok(out)
    }

    /// Upsample 288x288 logits to (w, h) and return the soft mask as u8 (sigmoid * 255).
    pub fn mask_u8(&self, logits: &Tensor, w: usize, h: usize) -> Result<Vec<u8>> {
        let dev = &*self.dev;
        let out = Tensor::new(dev, DType::U8, &[h * w])?;
        dev.launch_n(
            "k_resize_logits_u8",
            w * h,
            &[Arg::Ptr(logits.ptr), Arg::I32(1), Arg::I32(288), Arg::I32(288), Arg::Ptr(out.ptr), Arg::I32(h as i32), Arg::I32(w as i32)],
        )?;
        dev.dtoh_at::<u8>(out.ptr, w * h)
    }
}

/// Tight [x, y, w, h] box of mask pixels >= 128 (None when empty).
pub fn mask_bbox(m: &[u8], w: usize, h: usize) -> Option<[usize; 4]> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..h {
        let row = &m[y * w..(y + 1) * w];
        if let (Some(a), Some(b)) = (row.iter().position(|v| *v >= 128), row.iter().rposition(|v| *v >= 128)) {
            x0 = x0.min(a);
            x1 = x1.max(b);
            y0 = y0.min(y);
            y1 = y1.max(y);
        }
    }
    if x0 == usize::MAX {
        None
    } else {
        Some([x0, y0, x1 - x0 + 1, y1 - y0 + 1])
    }
}

pub fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}
