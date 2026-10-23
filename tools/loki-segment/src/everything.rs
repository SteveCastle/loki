//! `--everything`: automatic mask generation in the style of SAM's AMG: a grid of single positive clicks through the
//! interactive decoder (3 candidates each), filtered by predicted IoU and stability score, de-duplicated by mask IoU.
use crate::model::{Encoded, Model, Segment};
use crate::interactive::Point;
use crate::tensor::{DType, Tensor};
use crate::vit::IMG;
use anyhow::Result;

const GRID: usize = 16;
const LOW: usize = 288;
/// The interactive decoder's IoU head behaves like a logit (values ~1-4): candidates below this are dropped.
const PRED_IOU_MIN: f32 = 0.86;
const STABILITY_MIN: f32 = 0.92;
const NMS_IOU: f32 = 0.7;
const MIN_AREA_FRAC: f32 = 0.0008;

struct Cand {
    score: f32,
    bits: Vec<u64>,
    area: usize,
    logits: Vec<f32>,
}

fn bits_of(l: &[f32]) -> (Vec<u64>, usize) {
    let mut b = vec![0u64; (l.len() + 63) / 64];
    let mut n = 0;
    for (i, v) in l.iter().enumerate() {
        if *v > 0.0 {
            b[i / 64] |= 1 << (i % 64);
            n += 1;
        }
    }
    (b, n)
}

pub fn generate(model: &Model, enc: &Encoded, first_id: u32) -> Result<Vec<Segment>> {
    let p = LOW * LOW;
    let mut cands: Vec<Cand> = Vec::new();
    for gy in 0..GRID {
        for gx in 0..GRID {
            let x = (gx as f32 + 0.5) / GRID as f32 * IMG as f32;
            let y = (gy as f32 + 0.5) / GRID as f32 * IMG as f32;
            let o = model.inter.decode(&enc.sam, &[Point { x, y, label: 1 }], true)?;
            if o.obj_score <= 0.0 {
                continue;
            }
            let all = o.all.to_f32_vec(&model.dev)?;
            for k in 1..4 {
                let iou = o.ious[k];
                if iou < PRED_IOU_MIN {
                    continue;
                }
                let l = &all[k * p..(k + 1) * p];
                let hi = l.iter().filter(|v| **v > 1.0).count();
                let lo = l.iter().filter(|v| **v > -1.0).count();
                if lo == 0 || (hi as f32 / lo as f32) < STABILITY_MIN {
                    continue;
                }
                let (bits, area) = bits_of(l);
                if (area as f32) < MIN_AREA_FRAC * p as f32 || area as f32 > 0.95 * p as f32 {
                    continue;
                }
                cands.push(Cand { score: iou, bits, area, logits: l.to_vec() });
            }
        }
    }
    cands.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept: Vec<Cand> = Vec::new();
    for c in cands {
        let dup = kept.iter().any(|k| {
            let inter: u32 = k.bits.iter().zip(&c.bits).map(|(a, b)| (a & b).count_ones()).sum();
            let uni = k.area + c.area - inter as usize;
            uni > 0 && inter as f32 / uni as f32 > NMS_IOU
        });
        if !dup {
            kept.push(c);
        }
    }
    // largest first reads naturally as layers (background-ish regions, then parts)
    kept.sort_by(|a, b| b.area.cmp(&a.area));
    let mut out = Vec::new();
    for (i, c) in kept.into_iter().enumerate() {
        let t = Tensor::from_f32(&model.dev, &c.logits, &[p])?;
        debug_assert_eq!(t.dtype, DType::F32);
        out.push(Segment { id: first_id + i as u32, label: "object".into(), score: crate::detector::sigmoid(c.score), logits: t, candidates: Vec::new() });
    }
    Ok(out)
}
