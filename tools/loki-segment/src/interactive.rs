//! The interactive (SAM-style) head of the SAM 3.1 tracker: point/box prompt encoder and the two-way transformer
//! mask decoder (`interactive_sam_prompt_encoder` / `interactive_sam_mask_decoder`), fed by the `interactive_convs`
//! neck plus `interactivity_no_mem_embed` (an image with no memory). Boxes are two corner points (labels 2, 3),
//! as SAM 2 does.
use crate::cuda::Device;
use crate::ops::{self, Act, Epi};
use crate::tensor::{DType, Tensor};
use crate::weights::{HostMlp, Linear, Norm, Weights};
use anyhow::Result;
use std::sync::Arc;

const D: usize = 256;
const SIDE: usize = 72;
const HW: usize = SIDE * SIDE;
pub const IMG: f32 = 1008.0;
pub const NO_OBJ_SCORE: f32 = -1024.0;

/// SAM `Attention` with separate projections (internal dim = 256 / downsample).
pub struct SamAttn {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
impl SamAttn {
    pub fn load(w: &Weights, p: &str) -> Result<SamAttn> {
        Ok(SamAttn { q: w.linear(&format!("{p}.q_proj"))?, k: w.linear(&format!("{p}.k_proj"))?, v: w.linear(&format!("{p}.v_proj"))?, out: w.linear(&format!("{p}.out_proj"))? })
    }
    /// out_proj(attn(q, k, v)) (+ res) -> f32
    pub fn forward(&self, dev: &Device, q: &Tensor, k: &Tensor, v: &Tensor, res: Option<&Tensor>) -> Result<Tensor> {
        let qp = ops::linear_new(dev, q, &self.q, Act::None, DType::BF16)?;
        let kp = ops::linear_new(dev, k, &self.k, Act::None, DType::BF16)?;
        let vp = ops::linear_new(dev, v, &self.v, Act::None, DType::BF16)?;
        let a = ops::mha(dev, &qp, &kp, &vp, 8, None, None)?;
        let nq = q.numel() / D;
        let out = Tensor::new(dev, DType::F32, &[nq, D])?;
        match res {
            Some(r) => ops::linear(dev, &a, &self.out, Act::None, Epi::AddRes(r), &out)?,
            None => ops::linear(dev, &a, &self.out, Act::None, Epi::Store, &out)?,
        }
        Ok(out)
    }
}

pub struct TwoWayLayer {
    self_attn: SamAttn,
    norm1: Norm,
    t2i: SamAttn,
    norm2: Norm,
    lin1: Linear,
    lin2: Linear,
    norm3: Norm,
    i2t: SamAttn,
    norm4: Norm,
    skip_pe: bool,
}

pub struct TwoWay {
    layers: Vec<TwoWayLayer>,
    final_attn: SamAttn,
    norm_final: Norm,
}
impl TwoWay {
    pub fn load(w: &Weights, p: &str) -> Result<TwoWay> {
        let mut layers = Vec::new();
        for i in 0..2 {
            let l = format!("{p}.layers.{i}");
            layers.push(TwoWayLayer {
                self_attn: SamAttn::load(w, &format!("{l}.self_attn"))?,
                norm1: w.norm(&format!("{l}.norm1"), 1e-5)?,
                t2i: SamAttn::load(w, &format!("{l}.cross_attn_token_to_image"))?,
                norm2: w.norm(&format!("{l}.norm2"), 1e-5)?,
                lin1: w.linear(&format!("{l}.mlp.lin1"))?,
                lin2: w.linear(&format!("{l}.mlp.lin2"))?,
                norm3: w.norm(&format!("{l}.norm3"), 1e-5)?,
                i2t: SamAttn::load(w, &format!("{l}.cross_attn_image_to_token"))?,
                norm4: w.norm(&format!("{l}.norm4"), 1e-5)?,
                skip_pe: i == 0,
            });
        }
        Ok(TwoWay { layers, final_attn: SamAttn::load(w, &format!("{p}.final_attn_token_to_image"))?, norm_final: w.norm(&format!("{p}.norm_final_attn"), 1e-5)? })
    }
    /// queries f32 [T, 256] (also the query PE), keys f32 [HW, 256], key_pe f32 [HW, 256] -> (queries, keys)
    pub fn forward(&self, dev: &Device, tokens: &Tensor, keys: &Tensor, key_pe: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut q = tokens.clone();
        let mut k = keys.clone();
        for l in &self.layers {
            if l.skip_pe {
                let qb = ops::to_bf16(dev, &q)?;
                q = l.self_attn.forward(dev, &qb, &qb, &qb, None)?;
            } else {
                let qp = ops::add_new(dev, &q, Some(tokens), Act::None, DType::BF16)?;
                let qb = ops::to_bf16(dev, &q)?;
                q = l.self_attn.forward(dev, &qp, &qp, &qb, Some(&q))?;
            }
            q = ops::layernorm_new(dev, &q, &l.norm1, DType::F32)?;
            let qp = ops::add_new(dev, &q, Some(tokens), Act::None, DType::BF16)?;
            let kp = ops::add_new(dev, &k, Some(key_pe), Act::None, DType::BF16)?;
            let kb = ops::to_bf16(dev, &k)?;
            q = l.t2i.forward(dev, &qp, &kp, &kb, Some(&q))?;
            q = ops::layernorm_new(dev, &q, &l.norm2, DType::F32)?;
            let qb = ops::to_bf16(dev, &q)?;
            let h = ops::linear_new(dev, &qb, &l.lin1, Act::Relu, DType::BF16)?;
            let q2 = Tensor::new(dev, DType::F32, &q.shape)?;
            ops::linear(dev, &h, &l.lin2, Act::None, Epi::AddRes(&q), &q2)?;
            q = ops::layernorm_new(dev, &q2, &l.norm3, DType::F32)?;
            let qp = ops::add_new(dev, &q, Some(tokens), Act::None, DType::BF16)?;
            let qb = ops::to_bf16(dev, &q)?;
            k = l.i2t.forward(dev, &kp, &qp, &qb, Some(&k))?;
            k = ops::layernorm_new(dev, &k, &l.norm4, DType::F32)?;
        }
        let qp = ops::add_new(dev, &q, Some(tokens), Act::None, DType::BF16)?;
        let kp = ops::add_new(dev, &k, Some(key_pe), Act::None, DType::BF16)?;
        let kb = ops::to_bf16(dev, &k)?;
        q = self.final_attn.forward(dev, &qp, &kp, &kb, Some(&q))?;
        q = ops::layernorm_new(dev, &q, &self.norm_final, DType::F32)?;
        Ok((q, k))
    }
}

/// Random-Fourier positional encoding (SAM PositionEmbeddingRandom): coords in [0,1] -> [sin(128), cos(128)].
pub fn fourier_pe(g: &[f32], x: f32, y: f32) -> Vec<f32> {
    let (cx, cy) = (2.0 * x - 1.0, 2.0 * y - 1.0);
    let n = g.len() / 2;
    let mut out = vec![0f32; 2 * n];
    for i in 0..n {
        let a = 2.0 * std::f32::consts::PI * (cx * g[i] + cy * g[n + i]);
        out[i] = a.sin();
        out[n + i] = a.cos();
    }
    out
}
pub fn dense_pe(g: &[f32], side: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(side * side * 256);
    for y in 0..side {
        for x in 0..side {
            out.extend(fourier_pe(g, (x as f32 + 0.5) / side as f32, (y as f32 + 0.5) / side as f32));
        }
    }
    out
}

/// A click / box corner in model space (1008 px): label 1 positive, 0 negative, 2/3 box corners, -1 padding.
#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: f32,
    pub y: f32,
    pub label: i32,
}

pub struct SamOutput {
    /// chosen low-res mask logits f32 [288*288] (NO_OBJ_SCORE when the object is judged absent)
    pub mask: Tensor,
    pub iou: f32,
    pub obj_score: f32,
    /// all candidate masks [M, 288*288] and their IoU predictions, as the decoder returned them
    pub all: Tensor,
    pub ious: Vec<f32>,
    /// mask token used for the object pointer (video), f32 [256]
    pub token: Vec<f32>,
}

/// Interactive decoder (also the shape of the multiplex decoder's transformer).
pub struct Interactive {
    dev: Arc<Device>,
    gauss: Vec<f32>,
    point_emb: Vec<Vec<f32>>,
    not_a_point: Vec<f32>,
    no_mask: Vec<f32>,
    no_mem: Tensor,
    transformer: TwoWay,
    out_tokens: Vec<f32>,
    dc1: Linear,
    ln1: Norm,
    dc2: Linear,
    pub conv_s0: Linear,
    pub conv_s1: Linear,
    hyper: Vec<(Linear, Linear, Linear)>,
    iou_head: HostMlp,
    obj_head: HostMlp,
    pe: Tensor,
    mask_path: MaskPath,
}

/// Host-side weights of the mask-prompt path.
struct MaskPath {
    ds_w: Vec<f32>,
    ds_b: f32,
    c1w: Vec<f32>,
    c1b: Vec<f32>,
    n1w: Vec<f32>,
    n1b: Vec<f32>,
    c2w: Vec<f32>,
    c2b: Vec<f32>,
    n2w: Vec<f32>,
    n2b: Vec<f32>,
    c3w: Vec<f32>,
    c3b: Vec<f32>,
}

/// Conv2d k=2 s=2 on a host HWC plane [n, n, ci] -> [n/2, n/2, co]; weight [co, ci, 2, 2].
fn conv_k2s2(x: &[f32], n: usize, ci: usize, w: &[f32], b: &[f32], co: usize) -> Vec<f32> {
    let m = n / 2;
    let mut out = vec![0f32; m * m * co];
    for y in 0..m {
        for xx in 0..m {
            for o in 0..co {
                let mut s = b[o];
                for c in 0..ci {
                    for ky in 0..2 {
                        for kx in 0..2 {
                            s += w[((o * ci + c) * 2 + ky) * 2 + kx] * x[((2 * y + ky) * n + 2 * xx + kx) * ci + c];
                        }
                    }
                }
                out[(y * m + xx) * co + o] = s;
            }
        }
    }
    out
}

/// LayerNorm2d (eps 1e-6) over channels then exact GELU, host HWC.
fn ln2d_gelu(x: &[f32], c: usize, w: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = x.to_vec();
    for px in out.chunks_exact_mut(c) {
        let mean = px.iter().sum::<f32>() / c as f32;
        let var = px.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let r = 1.0 / (var + 1e-6).sqrt();
        for (i, v) in px.iter_mut().enumerate() {
            let y = (*v - mean) * r * w[i] + b[i];
            *v = 0.5 * y * (1.0 + erf(y / std::f32::consts::SQRT_2));
        }
    }
    out
}

/// erf via the Numerical Recipes erfc approximation (|error| < 1.2e-7).
fn erf(x: f32) -> f32 {
    let x = x as f64;
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let p = -x * x - 1.26551223
        + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    let y = 1.0 - t * p.exp();
    (if x >= 0.0 { y } else { -y }) as f32
}

/// Per-image state for the interactive head.
pub struct SamImage {
    /// image embedding + no-memory embed + no-mask dense embed, f32 [5184, 256]
    pub src: Tensor,
    /// conv_s0 of the 288 level bf16 [288*288, 32], conv_s1 of the 144 level bf16 [144*144, 64]
    pub s0: Tensor,
    pub s1: Tensor,
}

impl Interactive {
    pub fn load(w: &Weights, t: &str) -> Result<Interactive> {
        let pe = format!("{t}.interactive_sam_prompt_encoder");
        let dp = format!("{t}.interactive_sam_mask_decoder");
        let gauss = w.host(&format!("{pe}.pe_layer.positional_encoding_gaussian_matrix"))?;
        let mut out_tokens = w.host(&format!("{dp}.obj_score_token.weight"))?;
        out_tokens.extend(w.host(&format!("{dp}.iou_token.weight"))?);
        out_tokens.extend(w.host(&format!("{dp}.mask_tokens.weight"))?);
        let conv1x1 = |name: &str| -> Result<Linear> {
            let s = w.shape(&format!("{name}.weight"))?;
            w.linear_from_host(&w.host(&format!("{name}.weight"))?, Some(&w.host(&format!("{name}.bias"))?), s[0], s[1])
        };
        let mut hyper = Vec::new();
        for i in 0..4 {
            let h = format!("{dp}.output_hypernetworks_mlps.{i}.layers");
            hyper.push((w.linear(&format!("{h}.0"))?, w.linear(&format!("{h}.1"))?, w.linear(&format!("{h}.2"))?));
        }
        let dev = w.dev.clone();
        let pe_t = Tensor::from_f32(&dev, &dense_pe(&gauss, SIDE), &[HW, D])?;
        Ok(Interactive {
            point_emb: (0..4).map(|i| w.host(&format!("{pe}.point_embeddings.{i}.weight"))).collect::<Result<_>>()?,
            not_a_point: w.host(&format!("{pe}.not_a_point_embed.weight"))?,
            no_mask: w.host(&format!("{pe}.no_mask_embed.weight"))?,
            no_mem: w.f32(&format!("{t}.interactivity_no_mem_embed"))?.reshape(&[D]),
            transformer: TwoWay::load(w, &format!("{dp}.transformer"))?,
            out_tokens,
            dc1: w.conv_t2(&format!("{dp}.output_upscaling.0"))?,
            ln1: w.norm(&format!("{dp}.output_upscaling.1"), 1e-6)?,
            dc2: w.conv_t2(&format!("{dp}.output_upscaling.3"))?,
            conv_s0: conv1x1(&format!("{dp}.conv_s0"))?,
            conv_s1: conv1x1(&format!("{dp}.conv_s1"))?,
            hyper,
            iou_head: w.host_mlp(&format!("{dp}.iou_prediction_head"), 3)?,
            obj_head: w.host_mlp(&format!("{dp}.pred_obj_score_head"), 3)?,
            pe: pe_t,
            mask_path: MaskPath {
                ds_w: w.host(&format!("{t}.interactive_mask_downsample.weight"))?,
                ds_b: w.host(&format!("{t}.interactive_mask_downsample.bias"))?[0],
                c1w: w.host(&format!("{pe}.mask_downscaling.0.weight"))?,
                c1b: w.host(&format!("{pe}.mask_downscaling.0.bias"))?,
                n1w: w.host(&format!("{pe}.mask_downscaling.1.weight"))?,
                n1b: w.host(&format!("{pe}.mask_downscaling.1.bias"))?,
                c2w: w.host(&format!("{pe}.mask_downscaling.3.weight"))?,
                c2b: w.host(&format!("{pe}.mask_downscaling.3.bias"))?,
                n2w: w.host(&format!("{pe}.mask_downscaling.4.weight"))?,
                n2b: w.host(&format!("{pe}.mask_downscaling.4.bias"))?,
                c3w: w.host(&format!("{pe}.mask_downscaling.6.weight"))?,
                c3b: w.host(&format!("{pe}.mask_downscaling.6.bias"))?,
            },
            gauss,
            dev,
        })
    }

    /// Per-image state from the interactive neck (levels 288, 144, 72).
    pub fn prepare(&self, fpn: &[Tensor]) -> Result<SamImage> {
        let dev = &*self.dev;
        let s0 = ops::linear_new(dev, &fpn[0], &self.conv_s0, Act::None, DType::BF16)?;
        let s1 = ops::linear_new(dev, &fpn[1], &self.conv_s1, Act::None, DType::BF16)?;
        let a = ops::add_new(dev, &fpn[2], Some(&self.no_mem), Act::None, DType::F32)?;
        let nm = Tensor::from_f32(dev, &self.no_mask, &[D])?;
        let src = ops::add_new(dev, &a, Some(&nm), Act::None, DType::F32)?;
        Ok(SamImage { src, s0, s1 })
    }

    /// Sparse prompt embeddings for the points (+ one padding point, as the 3.1 tracker always pads).
    pub fn embed_points(&self, pts: &[Point]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut all = pts.to_vec();
        all.push(Point { x: 0.0, y: 0.0, label: -1 });
        for p in &all {
            let mut e = fourier_pe(&self.gauss, (p.x + 0.5) / IMG, (p.y + 0.5) / IMG);
            if p.label == -1 {
                e = self.not_a_point.clone();
            } else if (0..4).contains(&p.label) {
                for (v, a) in e.iter_mut().zip(&self.point_emb[p.label as usize]) {
                    *v += a;
                }
            }
            out.extend(e);
        }
        out
    }

    /// Decode the mask for one object from clicks/box corners. `multimask` = SAM's ambiguity mode (1 click).
    pub fn decode(&self, img: &SamImage, pts: &[Point], multimask: bool) -> Result<SamOutput> {
        self.decode_with(img, pts, multimask, None)
    }

    /// Mask prompt (SAM `mask_downscaling` after the tracker's `interactive_mask_downsample`), on the host:
    /// binary mask at 1008x1008 -> dense prompt embedding f32 [72*72, 256].
    pub fn mask_prompt(&self, m1008: &[f32]) -> Vec<f32> {
        self.mask_prompt_from(m1008, 1008)
    }

    /// Mask prompt from an `side` x `side` mask (any scale: binary or logits).
    pub fn mask_prompt_from(&self, m: &[f32], side: usize) -> Vec<f32> {
        let mp = &self.mask_path;
        // interactive_mask_downsample: Conv2d(1, 1, k=4, s=4)
        let n0 = side / 4;
        let mut a = vec![0f32; n0 * n0];
        for y in 0..n0 {
            for x in 0..n0 {
                let mut s = mp.ds_b;
                for ky in 0..4 {
                    for kx in 0..4 {
                        s += mp.ds_w[ky * 4 + kx] * m[(4 * y + ky) * side + 4 * x + kx];
                    }
                }
                a[y * n0 + x] = s;
            }
        }
        // F.interpolate(.., 288, bilinear, antialias=True)
        let up = crate::image::resample_plane(&a, n0, n0, 288, 288);
        // mask_downscaling: conv(1->4, k2 s2) LN2d GELU conv(4->16, k2 s2) LN2d GELU conv(16->256, k1)
        let c1 = conv_k2s2(&up, 288, 1, &mp.c1w, &mp.c1b, 4);
        let c1 = ln2d_gelu(&c1, 4, &mp.n1w, &mp.n1b);
        let c2 = conv_k2s2(&c1, 144, 4, &mp.c2w, &mp.c2b, 16);
        let c2 = ln2d_gelu(&c2, 16, &mp.n2w, &mp.n2b);
        let mut out = vec![0f32; HW * D];
        for p in 0..HW {
            for o in 0..D {
                let mut s = mp.c3b[o];
                for c in 0..16 {
                    s += mp.c3w[o * 16 + c] * c2[p * 16 + c];
                }
                out[p * D + o] = s;
            }
        }
        out
    }

    /// `dense`: optional dense prompt embedding [5184, 256] replacing the no-mask embedding.
    pub fn decode_with(&self, img: &SamImage, pts: &[Point], multimask: bool, dense: Option<&[f32]>) -> Result<SamOutput> {
        let dev = &*self.dev;
        let src_owned;
        let src = match dense {
            Some(d) => {
                // img.src already carries no_mask_embed: swap it for the mask embedding
                let mut delta = d.to_vec();
                for p in 0..HW {
                    for c in 0..D {
                        delta[p * D + c] -= self.no_mask[c];
                    }
                }
                let dt = Tensor::from_f32(dev, &delta, &[HW, D])?;
                src_owned = ops::add_new(dev, &img.src, Some(&dt), Act::None, DType::F32)?;
                &src_owned
            }
            None => &img.src,
        };
        let mut tok = self.out_tokens.clone();
        tok.extend(self.embed_points(pts));
        let nt = tok.len() / D;
        let tokens = Tensor::from_f32(dev, &tok, &[nt, D])?;
        let (hs, keys) = self.transformer.forward(dev, &tokens, src, &self.pe)?;
        let hsv = hs.to_f32_vec(dev)?;
        // upscaling: dc1 (+ s1) -> LN2d -> GELU -> dc2 (+ s0) -> GELU
        let kb = ops::to_bf16(dev, &keys)?;
        let u1 = ops::conv_t2(dev, &kb, SIDE, SIDE, &self.dc1, Act::None)?;
        let u1 = ops::add_new(dev, &u1, Some(&img.s1), Act::None, DType::F32)?;
        let u1 = ops::layernorm_new(dev, &u1, &self.ln1, DType::F32)?;
        let u1 = ops::add_new(dev, &u1, None, Act::GeluErf, DType::BF16)?;
        let u2 = ops::conv_t2(dev, &u1, 2 * SIDE, 2 * SIDE, &self.dc2, Act::None)?;
        let up = ops::add_new(dev, &u2, Some(&img.s0), Act::GeluErf, DType::BF16)?;
        // hypernetworks on the 4 mask tokens
        let nm = 4;
        let hyper = Tensor::new(dev, DType::BF16, &[nm, 32])?;
        for i in 0..nm {
            let t = Tensor::from_f32(dev, &hsv[(2 + i) * D..(3 + i) * D], &[1, D])?;
            let t = ops::to_bf16(dev, &t)?;
            let (a, b, c) = &self.hyper[i];
            let h = ops::linear_new(dev, &t, a, Act::Relu, DType::BF16)?;
            let h = ops::linear_new(dev, &h, b, Act::Relu, DType::BF16)?;
            let h = ops::linear_new(dev, &h, c, Act::None, DType::F32)?;
            let hb = ops::to_bf16(dev, &h)?;
            dev.dtod(hyper.ptr + (i * 32 * 2) as u64, hb.ptr, 64)?;
        }
        let p = 4 * HW * 4;
        let masks = Tensor::new(dev, DType::F32, &[nm, p])?;
        ops::gemm(dev, &hyper, &up, None, Act::None, Epi::Store, &masks)?;
        let ious = self.iou_head.apply(&hsv[D..2 * D]);
        let obj_score = self.obj_head.apply(&hsv[0..D])[0];
        // selection
        let pick = if multimask {
            1 + (0..3).max_by(|a, b| ious[1 + a].partial_cmp(&ious[1 + b]).unwrap()).unwrap()
        } else {
            // dynamic multimask via stability (delta 0.05, threshold 0.98)
            let m0 = masks.rows(0, 1).to_f32_vec(dev)?;
            let ai = m0.iter().filter(|v| **v > 0.05).count() as f32;
            let au = m0.iter().filter(|v| **v > -0.05).count() as f32;
            let stab = if au > 0.0 { ai / au } else { 1.0 };
            if stab >= 0.98 {
                0
            } else {
                1 + (0..3).max_by(|a, b| ious[1 + a].partial_cmp(&ious[1 + b]).unwrap()).unwrap()
            }
        };
        let mask = Tensor::new(dev, DType::F32, &[p])?;
        if obj_score > 0.0 {
            dev.dtod(mask.ptr, masks.ptr + (pick * p * 4) as u64, p * 4)?;
        } else {
            ops::fill_f32(dev, &mask, NO_OBJ_SCORE)?;
        }
        let token = hsv[(2 + pick) * D..(3 + pick) * D].to_vec();
        Ok(SamOutput { mask, iou: ious[pick], obj_score, all: masks, ious, token })
    }
}
