//! The SAM 3 detector (promptable concept segmentation): geometry/CLS prompt encoder, 6-layer text-image fusion
//! encoder, 6-layer DETR decoder (200 queries + presence token, box refinement, log box relative-position bias),
//! dot-product scoring and the pixel-decoder segmentation head. Follows `sam3/model/{encoder,decoder,
//! geometry_encoders,maskformer_segmentation}.py` and `Sam3Image.forward_grounding` (inference, no DAC).
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, Epi};
use crate::tensor::{DType, Tensor};
use crate::text::TextFeatures;
use crate::weights::{HostLinear, HostMlp, Linear, Norm, Weights};
use anyhow::Result;
use std::sync::Arc;

pub const D: usize = 256;
const HEADS: usize = 8;
pub const NQ: usize = 200;
const SIDE: usize = 72;
const HW: usize = SIDE * SIDE;

/// nn.MultiheadAttention weights split into q / k / v / out projections.
pub struct Mha {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
}
impl Mha {
    pub fn load(w: &Weights, prefix: &str) -> Result<Mha> {
        Ok(Mha { q: w.in_proj(prefix, Some(0))?, k: w.in_proj(prefix, Some(1))?, v: w.in_proj(prefix, Some(2))?, out: w.linear(&format!("{prefix}.out_proj"))? })
    }
    /// out_proj(attention(q_in Wq, k_in Wk, v_in Wv)) (+ res); inputs bf16, result f32 (`res` added when given).
    pub fn forward(&self, dev: &Device, q_in: &Tensor, k_in: &Tensor, v_in: &Tensor, kmask: Option<&Tensor>, bias: Option<(&Tensor, usize, usize)>, res: Option<&Tensor>) -> Result<Tensor> {
        let q = ops::linear_new(dev, q_in, &self.q, Act::None, DType::BF16)?;
        let k = ops::linear_new(dev, k_in, &self.k, Act::None, DType::BF16)?;
        let v = ops::linear_new(dev, v_in, &self.v, Act::None, DType::BF16)?;
        let a = ops::mha(dev, &q, &k, &v, HEADS, kmask, bias)?;
        let nq = q.numel() / D;
        let out = Tensor::new(dev, DType::F32, &[nq, D])?;
        match res {
            Some(r) => ops::linear(dev, &a, &self.out, Act::None, Epi::AddRes(r), &out)?,
            None => ops::linear(dev, &a, &self.out, Act::None, Epi::Store, &out)?,
        }
        Ok(out)
    }
}

/// Pre-norm encoder layer (fusion encoder and geometry encoder share the structure).
struct EncLayer {
    norm1: Norm,
    self_attn: Mha,
    norm2: Norm,
    cross: Mha,
    norm3: Norm,
    lin1: Linear,
    lin2: Linear,
}
impl EncLayer {
    fn load(w: &Weights, p: &str) -> Result<EncLayer> {
        Ok(EncLayer {
            norm1: w.norm(&format!("{p}.norm1"), 1e-5)?,
            self_attn: Mha::load(w, &format!("{p}.self_attn"))?,
            norm2: w.norm(&format!("{p}.norm2"), 1e-5)?,
            cross: Mha::load(w, &format!("{p}.cross_attn_image"))?,
            norm3: w.norm(&format!("{p}.norm3"), 1e-5)?,
            lin1: w.linear(&format!("{p}.linear1"))?,
            lin2: w.linear(&format!("{p}.linear2"))?,
        })
    }
    /// x f32 [n, 256] updated in place.
    ///  self-attn: q = k = LN1(x) (+ pos if given), v = LN1(x)      (fusion encoder: pos_enc_at_attn)
    ///  cross:     q = LN2(x), k = mem_k, v = mem_v (mem_kmask over keys)
    #[allow(clippy::too_many_arguments)]
    fn forward(&self, dev: &Device, x: &mut Tensor, pos: Option<&Tensor>, self_kmask: Option<&Tensor>, mem_k: &Tensor, mem_v: &Tensor, mem_kmask: Option<&Tensor>) -> Result<()> {
        let h = Tensor::new(dev, DType::BF16, &x.shape)?;
        let hp = Tensor::new(dev, DType::BF16, &x.shape)?;
        match pos {
            Some(p) => ops::layernorm(dev, x, Some(&self.norm1), 1e-5, &h, Some((p, &hp)))?,
            None => ops::layernorm(dev, x, Some(&self.norm1), 1e-5, &h, None)?,
        }
        let qk = if pos.is_some() { &hp } else { &h };
        *x = self.self_attn.forward(dev, qk, qk, &h, self_kmask, None, Some(&*x))?;
        ops::layernorm(dev, x, Some(&self.norm2), 1e-5, &h, None)?;
        *x = self.cross.forward(dev, &h, mem_k, mem_v, mem_kmask, None, Some(&*x))?;
        ops::layernorm(dev, x, Some(&self.norm3), 1e-5, &h, None)?;
        let f = ops::linear_new(dev, &h, &self.lin1, Act::Relu, DType::BF16)?;
        ops::linear(dev, &f, &self.lin2, Act::None, Epi::AddRes(&*x), &*x)?;
        Ok(())
    }
}

/// Post-norm decoder layer: self-attn (queries + presence), text cross-attn, image cross-attn with box RPB, FFN.
struct DecLayer {
    self_attn: Mha,
    norm2: Norm,
    ca_text: Mha,
    catext_norm: Norm,
    cross: Mha,
    norm1: Norm,
    lin1: Linear,
    lin2: Linear,
    norm3: Norm,
}

struct Mlp {
    layers: Vec<Linear>,
}
impl Mlp {
    fn load(w: &Weights, p: &str, n: usize) -> Result<Mlp> {
        Ok(Mlp { layers: (0..n).map(|i| w.linear(&format!("{p}.layers.{i}"))).collect::<Result<_>>()? })
    }
    /// bf16 in -> f32 out (relu between layers)
    fn forward(&self, dev: &Device, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for (i, l) in self.layers.iter().enumerate() {
            let last = i + 1 == self.layers.len();
            h = ops::linear_new(dev, &h, l, if last { Act::None } else { Act::Relu }, if last { DType::F32 } else { DType::BF16 })?;
        }
        Ok(h)
    }
}

/// f32 small linear (any output width) on the device.
struct SmallLinear {
    w: Tensor,
    b: Tensor,
    n: usize,
    k: usize,
}
impl SmallLinear {
    fn load(w: &Weights, p: &str) -> Result<SmallLinear> {
        let s = w.shape(&format!("{p}.weight"))?;
        Ok(SmallLinear { w: w.f32(&format!("{p}.weight"))?, b: w.f32(&format!("{p}.bias"))?, n: s[0], k: s[1] })
    }
    fn forward(&self, dev: &Device, x: &Tensor, act: Act) -> Result<Tensor> {
        let m = x.numel() / self.k;
        let out = Tensor::new(dev, DType::F32, &[m, self.n])?;
        let xt = if x.dtype == DType::BF16 { 1 } else { 0 };
        dev.launch_n(
            "k_linear_small",
            m * self.n,
            &[Arg::Ptr(x.ptr), Arg::I32(xt), Arg::Ptr(self.w.ptr), Arg::Ptr(self.b.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(self.n as i32), Arg::I32(self.k as i32), Arg::I32(act.code())],
        )?;
        Ok(out)
    }
}

pub struct Detector {
    dev: Arc<Device>,
    // geometry encoder (CLS token only: points/boxes exemplars are not wired yet)
    geo_cls: Vec<f32>,
    geo_final_proj: Linear,
    geo_norm: Norm,
    geo_layers: Vec<EncLayer>,
    geo_encode_norm: Norm,
    enc_layers: Vec<EncLayer>,
    dec_layers: Vec<DecLayer>,
    dec_norm: Norm,
    query_embed: Vec<f32>,
    ref_points: Vec<f32>,
    ref_point_head: Mlp,
    bbox_embed: (Linear, Linear, SmallLinear),
    rpb_x: [Tensor; 4],
    rpb_y: [Tensor; 4],
    presence_token: Vec<f32>,
    presence_head: HostMlp,
    presence_out_norm: Norm,
    // scoring
    score_mlp: (Linear, Linear, Norm),
    score_prompt_proj: HostLinear,
    score_hs_proj: Linear,
    // segmentation head
    seg_cross: Mha,
    seg_cross_norm: Norm,
    pix_convs: Vec<crate::weights::Conv>,
    pix_norms: Vec<Norm>,
    inst_head: Linear,
    sem_head: Linear,
    mask_embed: Mlp,
    pos72: Tensor,
}

/// Image-dependent detector state, computed once per image and reused for every concept prompt.
pub struct DetImage {
    /// neck level 2 (72x72) bf16 [5184, 256]
    pub feat: Tensor,
    /// feat + pos, bf16
    pub feat_pos: Tensor,
    /// f32 copy of feat (encoder input)
    pub feat_f32: Tensor,
    /// FPN levels 0 (288) and 1 (144), bf16
    pub fpn0: Tensor,
    pub fpn1: Tensor,
    /// encoded geometry CLS prompt token f32 [1, 256]
    pub geo: Tensor,
}

/// Result of one concept prompt on one image.
pub struct Detections {
    /// joint probability sigmoid(class) * sigmoid(presence), per query
    pub probs: Vec<f32>,
    /// joint logits (inverse sigmoid of probs, clamped to +-10), per query (for comparison with the reference)
    pub logits: Vec<f32>,
    pub presence_logit: f32,
    /// cxcywh normalized, per query
    pub boxes: Vec<[f32; 4]>,
    /// last decoder layer (normed), bf16 [200, 256]
    pub hs: Tensor,
    /// per-pixel instance embeddings bf16 [288*288, 256]
    pub inst: Tensor,
    /// pixel embedding bf16 [288*288, 256] (input of the semantic head)
    pub pix: Tensor,
    /// fusion encoder output f32 [5184, 256]
    pub memory: Tensor,
}

impl Detector {
    pub fn load(w: &Weights, p: &str) -> Result<Detector> {
        let dev = w.dev.clone();
        let g = format!("{p}.geometry_encoder");
        let geo_layers = (0..3).map(|i| EncLayer::load(w, &format!("{g}.encode.{i}"))).collect::<Result<_>>()?;
        let enc_layers = (0..6).map(|i| EncLayer::load(w, &format!("{p}.transformer.encoder.layers.{i}"))).collect::<Result<_>>()?;
        let dp = format!("{p}.transformer.decoder");
        let mut dec_layers = Vec::new();
        for i in 0..6 {
            let l = format!("{dp}.layers.{i}");
            dec_layers.push(DecLayer {
                self_attn: Mha::load(w, &format!("{l}.self_attn"))?,
                norm2: w.norm(&format!("{l}.norm2"), 1e-5)?,
                ca_text: Mha::load(w, &format!("{l}.ca_text"))?,
                catext_norm: w.norm(&format!("{l}.catext_norm"), 1e-5)?,
                cross: Mha::load(w, &format!("{l}.cross_attn"))?,
                norm1: w.norm(&format!("{l}.norm1"), 1e-5)?,
                lin1: w.linear(&format!("{l}.linear1"))?,
                lin2: w.linear(&format!("{l}.linear2"))?,
                norm3: w.norm(&format!("{l}.norm3"), 1e-5)?,
            });
        }
        let ld = |name: &str| -> Result<Tensor> { w.f32(name) };
        let rpb = |axis: &str| -> Result<[Tensor; 4]> {
            Ok([
                ld(&format!("{dp}.boxRPB_embed_{axis}.layers.0.weight"))?,
                ld(&format!("{dp}.boxRPB_embed_{axis}.layers.0.bias"))?,
                ld(&format!("{dp}.boxRPB_embed_{axis}.layers.1.weight"))?,
                ld(&format!("{dp}.boxRPB_embed_{axis}.layers.1.bias"))?,
            ])
        };
        let s = format!("{p}.dot_prod_scoring");
        let sh = format!("{p}.segmentation_head");
        let inst = w.host(&format!("{sh}.instance_seg_head.weight"))?;
        let inst_b = w.host(&format!("{sh}.instance_seg_head.bias"))?;
        let mut sem = w.host(&format!("{sh}.semantic_seg_head.weight"))?; // [1, 256, 1, 1]
        let mut sem_b = w.host(&format!("{sh}.semantic_seg_head.bias"))?;
        sem.resize(8 * D, 0.0);
        sem_b.resize(8, 0.0);
        let pos72 = Tensor::from_f32(&dev, &crate::vit::sine_pos_2d(SIDE, SIDE), &[HW, D])?;
        Ok(Detector {
            geo_cls: w.host(&format!("{g}.cls_embed.weight"))?,
            geo_final_proj: w.linear(&format!("{g}.final_proj"))?,
            geo_norm: w.norm(&format!("{g}.norm"), 1e-5)?,
            geo_layers,
            geo_encode_norm: w.norm(&format!("{g}.encode_norm"), 1e-5)?,
            enc_layers,
            dec_layers,
            dec_norm: w.norm(&format!("{dp}.norm"), 1e-5)?,
            query_embed: w.host(&format!("{dp}.query_embed.weight"))?,
            ref_points: w.host(&format!("{dp}.reference_points.weight"))?,
            ref_point_head: Mlp::load(w, &format!("{dp}.ref_point_head"), 2)?,
            bbox_embed: (w.linear(&format!("{dp}.bbox_embed.layers.0"))?, w.linear(&format!("{dp}.bbox_embed.layers.1"))?, SmallLinear::load(w, &format!("{dp}.bbox_embed.layers.2"))?),
            rpb_x: rpb("x")?,
            rpb_y: rpb("y")?,
            presence_token: w.host(&format!("{dp}.presence_token.weight"))?,
            presence_head: w.host_mlp(&format!("{dp}.presence_token_head"), 3)?,
            presence_out_norm: w.norm(&format!("{dp}.presence_token_out_norm"), 1e-5)?,
            score_mlp: (w.linear(&format!("{s}.prompt_mlp.layers.0"))?, w.linear(&format!("{s}.prompt_mlp.layers.1"))?, w.norm(&format!("{s}.prompt_mlp.out_norm"), 1e-5)?),
            score_prompt_proj: w.host_linear(&format!("{s}.prompt_proj"))?,
            score_hs_proj: w.linear(&format!("{s}.hs_proj"))?,
            seg_cross: Mha::load(w, &format!("{sh}.cross_attend_prompt"))?,
            seg_cross_norm: w.norm(&format!("{sh}.cross_attn_norm"), 1e-5)?,
            pix_convs: (0..3).map(|i| w.conv(&format!("{sh}.pixel_decoder.conv_layers.{i}"))).collect::<Result<_>>()?,
            pix_norms: (0..3).map(|i| w.norm(&format!("{sh}.pixel_decoder.norms.{i}"), 1e-5)).collect::<Result<_>>()?,
            inst_head: w.linear_from_host(&inst, Some(&inst_b), D, D)?,
            sem_head: w.linear_from_host(&sem, Some(&sem_b), 8, D)?,
            mask_embed: Mlp::load(w, &format!("{sh}.mask_predictor.mask_embed"), 3)?,
            pos72,
            dev,
        })
    }

    /// Per-image state from the detector neck (levels 288, 144, 72).
    pub fn prepare(&self, fpn: &[Tensor]) -> Result<DetImage> {
        let dev = &*self.dev;
        let feat = fpn[2].clone();
        let feat_pos = ops::add_new(dev, &feat, Some(&self.pos72), Act::None, DType::BF16)?;
        let feat_f32 = ops::to_f32(dev, &feat)?;
        // geometry encoder with only the CLS token: norm(final_proj(cls)) then 3 layers against the image
        let cls = Tensor::from_f32(dev, &self.geo_cls, &[1, D])?;
        let cls_b = ops::to_bf16(dev, &cls)?;
        let proj = ops::linear_new(dev, &cls_b, &self.geo_final_proj, Act::None, DType::F32)?;
        let mut geo = ops::layernorm_new(dev, &proj, &self.geo_norm, DType::F32)?;
        for l in &self.geo_layers {
            l.forward(dev, &mut geo, None, None, &feat_pos, &feat, None)?;
        }
        let geo = ops::layernorm_new(dev, &geo, &self.geo_encode_norm, DType::F32)?;
        Ok(DetImage { feat, feat_pos, feat_f32, fpn0: fpn[0].clone(), fpn1: fpn[1].clone(), geo })
    }

    /// Run one concept prompt. `with_pixels`: also build the per-pixel instance embeddings (needed for masks).
    pub fn detect(&self, img: &DetImage, text: &TextFeatures) -> Result<Detections> {
        let dev = &*self.dev;
        // ---- prompt = [text (32, padded), geo cls (1)]
        let np = text.valid.len() + 1;
        let prompt = Tensor::new(dev, DType::F32, &[np, D])?;
        dev.dtod(prompt.ptr, text.feats.ptr, text.feats.bytes())?;
        dev.dtod(prompt.ptr + text.feats.bytes() as u64, img.geo.ptr, D * 4)?;
        let mut valid: Vec<u8> = text.valid.iter().map(|v| *v as u8).collect();
        valid.push(1);
        let kmask = Tensor::from_buf(dev.upload(&valid)?, DType::U8, &[np]);
        let prompt_b = ops::to_bf16(dev, &prompt)?;

        // ---- fusion encoder over the 72x72 image tokens
        let mut memory = img.feat_f32.clone(); // replaced (not written) by the first layer's attention
        for l in &self.enc_layers {
            l.forward(dev, &mut memory, Some(&self.pos72), None, &prompt_b, &prompt_b, Some(&kmask))?;
        }
        let mem_b = ops::to_bf16(dev, &memory)?;
        let mem_pos = ops::add_new(dev, &memory, Some(&self.pos72), Act::None, DType::BF16)?;

        // ---- decoder
        let nt = NQ + 1;
        let mut tgt_h = Vec::with_capacity(nt * D);
        tgt_h.extend_from_slice(&self.presence_token);
        tgt_h.extend_from_slice(&self.query_embed);
        let mut tgt = Tensor::from_f32(dev, &tgt_h, &[nt, D])?;
        let refs: Vec<f32> = self.ref_points.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let mut refb = Tensor::from_f32(dev, &refs, &[NQ, 4])?;
        let qpos_full = Tensor::zeros(dev, DType::F32, &[nt, D])?;
        let bias = Tensor::new(dev, DType::F32, &[HEADS, nt, HW])?;
        let bx = Tensor::new(dev, DType::F32, &[NQ, SIDE, HEADS])?;
        let by = Tensor::new(dev, DType::F32, &[NQ, SIDE, HEADS])?;
        let mut hs = Tensor::new(dev, DType::BF16, &[NQ, D])?;
        let mut final_delta = None;
        for (li, l) in self.dec_layers.iter().enumerate() {
            // query positions from the reference boxes
            let sine = Tensor::new(dev, DType::F32, &[NQ, 512])?;
            dev.launch_n("k_sine_embed_boxes", NQ * 512, &[Arg::Ptr(refb.ptr), Arg::Ptr(sine.ptr), Arg::I32(NQ as i32)])?;
            let sine_b = ops::to_bf16(dev, &sine)?;
            let qpos = self.ref_point_head.forward(dev, &sine_b)?;
            dev.dtod(qpos_full.ptr + (D * 4) as u64, qpos.ptr, qpos.bytes())?;
            // box relative position bias
            for (axis, out, wts) in [(0, &bx, &self.rpb_x), (1, &by, &self.rpb_y)] {
                dev.launch_n(
                    "k_rpb_axis",
                    NQ * SIDE,
                    &[
                        Arg::Ptr(refb.ptr),
                        Arg::I32(NQ as i32),
                        Arg::I32(SIDE as i32),
                        Arg::I32(axis),
                        Arg::Ptr(wts[0].ptr),
                        Arg::Ptr(wts[1].ptr),
                        Arg::Ptr(wts[2].ptr),
                        Arg::Ptr(wts[3].ptr),
                        Arg::I32(HEADS as i32),
                        Arg::Ptr(out.ptr),
                    ],
                )?;
            }
            dev.launch_n("k_rpb_bias", HEADS * nt * HW, &[Arg::Ptr(bx.ptr), Arg::Ptr(by.ptr), Arg::I32(NQ as i32), Arg::I32(SIDE as i32), Arg::I32(SIDE as i32), Arg::I32(HEADS as i32), Arg::Ptr(bias.ptr)])?;

            // self-attention (post-norm)
            let tq = ops::add_new(dev, &tgt, Some(&qpos_full), Act::None, DType::BF16)?;
            let tv = ops::to_bf16(dev, &tgt)?;
            let t2 = l.self_attn.forward(dev, &tq, &tq, &tv, None, None, Some(&tgt))?;
            tgt = ops::layernorm_new(dev, &t2, &l.norm2, DType::F32)?;
            // text cross-attention
            let tq = ops::add_new(dev, &tgt, Some(&qpos_full), Act::None, DType::BF16)?;
            let t2 = l.ca_text.forward(dev, &tq, &prompt_b, &prompt_b, Some(&kmask), None, Some(&tgt))?;
            tgt = ops::layernorm_new(dev, &t2, &l.catext_norm, DType::F32)?;
            // image cross-attention with the box bias
            let tq = ops::add_new(dev, &tgt, Some(&qpos_full), Act::None, DType::BF16)?;
            let t2 = l.cross.forward(dev, &tq, &mem_pos, &mem_b, None, Some((&bias, nt * HW, HW)), Some(&tgt))?;
            tgt = ops::layernorm_new(dev, &t2, &l.norm1, DType::F32)?;
            // FFN
            let tb = ops::to_bf16(dev, &tgt)?;
            let f = ops::linear_new(dev, &tb, &l.lin1, Act::Relu, DType::BF16)?;
            let t2 = Tensor::new(dev, DType::F32, &tgt.shape)?;
            ops::linear(dev, &f, &l.lin2, Act::None, Epi::AddRes(&tgt), &t2)?;
            tgt = ops::layernorm_new(dev, &t2, &l.norm3, DType::F32)?;

            // box refinement from the normed queries
            let q = tgt.rows(1, NQ);
            hs = ops::layernorm_new(dev, &q, &self.dec_norm, DType::BF16)?;
            let h1 = ops::linear_new(dev, &hs, &self.bbox_embed.0, Act::Relu, DType::BF16)?;
            let h2 = ops::linear_new(dev, &h1, &self.bbox_embed.1, Act::Relu, DType::BF16)?;
            let delta = self.bbox_embed.2.forward(dev, &h2, Act::None)?;
            let newref = Tensor::new(dev, DType::F32, &[NQ, 4])?;
            dev.launch_n("k_box_refine", NQ * 4, &[Arg::Ptr(refb.ptr), Arg::Ptr(delta.ptr), Arg::I32((NQ * 4) as i32), Arg::Ptr(newref.ptr)])?;
            if li + 1 == self.dec_layers.len() {
                final_delta = Some(newref);
            } else {
                refb = newref;
            }
        }
        let boxes_t = final_delta.unwrap();

        // ---- presence logit (host)
        let pres = tgt.rows(0, 1);
        let pres_n = ops::layernorm_new(dev, &pres, &self.presence_out_norm, DType::F32)?;
        let presence_logit = self.presence_head.apply(&pres_n.to_f32_vec(dev)?)[0];

        // ---- dot-product scoring
        let h = ops::linear_new(dev, &prompt_b, &self.score_mlp.0, Act::Relu, DType::BF16)?;
        let pm = Tensor::new(dev, DType::F32, &[np, D])?;
        ops::linear(dev, &h, &self.score_mlp.1, Act::None, Epi::AddRes(&prompt), &pm)?;
        let pm = ops::layernorm_new(dev, &pm, &self.score_mlp.2, DType::F32)?.to_f32_vec(dev)?;
        let mut pooled = vec![0f32; D];
        let nvalid = valid.iter().filter(|v| **v != 0).count().max(1) as f32;
        for (t, v) in valid.iter().enumerate() {
            if *v != 0 {
                for c in 0..D {
                    pooled[c] += pm[t * D + c];
                }
            }
        }
        for v in pooled.iter_mut() {
            *v /= nvalid;
        }
        let pproj = self.score_prompt_proj.apply(&pooled);
        let hproj = ops::linear_new(dev, &hs, &self.score_hs_proj, Act::None, DType::F32)?.to_f32_vec(dev)?;
        let scale = 1.0 / (D as f32).sqrt();
        let pres_p = sigmoid(presence_logit);
        let mut probs = Vec::with_capacity(NQ);
        let mut logits = Vec::with_capacity(NQ);
        for q in 0..NQ {
            let mut s = 0f32;
            for c in 0..D {
                s += hproj[q * D + c] * pproj[c];
            }
            let s = (s * scale).clamp(-12.0, 12.0);
            let p = sigmoid(s) * pres_p;
            probs.push(p);
            let pc = p.clamp(0.0, 1.0);
            logits.push((pc.max(1e-3) / (1.0 - pc).max(1e-3)).ln().clamp(-10.0, 10.0));
        }
        let bv = boxes_t.to_f32_vec(dev)?;
        let boxes = (0..NQ).map(|q| [bv[q * 4], bv[q * 4 + 1], bv[q * 4 + 2], bv[q * 4 + 3]]).collect();

        // ---- segmentation head: prompt cross-attention on the encoder output, pixel decoder, instance embeddings
        let mn = ops::layernorm_new(dev, &memory, &self.seg_cross_norm, DType::BF16)?;
        let enc = self.seg_cross.forward(dev, &mn, &prompt_b, &prompt_b, Some(&kmask), None, Some(&memory))?;
        let mut prev = ops::to_bf16(dev, &enc)?;
        let mut side = SIDE;
        for (i, feat) in [&img.fpn1, &img.fpn0].iter().enumerate() {
            side *= 2;
            let x = Tensor::new(dev, DType::BF16, &[side * side, D])?;
            ops::upadd2(dev, feat, &prev, side, side, &x)?;
            let y = Tensor::new(dev, DType::BF16, &[side * side, D])?;
            ops::conv2d(dev, &x, side, side, &self.pix_convs[i], 1, 1, &y)?;
            ops::groupnorm(dev, &y, 8, &self.pix_norms[i], Act::Relu, &x)?;
            prev = x;
        }
        let inst = ops::linear_new(dev, &prev, &self.inst_head, Act::None, DType::BF16)?;
        Ok(Detections { probs, logits, presence_logit, boxes, hs, inst, pix: prev, memory })
    }

    /// Mask logits f32 [n, 288, 288] for the given queries.
    pub fn masks(&self, det: &Detections, queries: &[usize]) -> Result<Tensor> {
        let dev = &*self.dev;
        let n = queries.len().max(1);
        let sel = Tensor::new(dev, DType::BF16, &[n, D])?;
        for (i, q) in queries.iter().enumerate() {
            dev.dtod(sel.ptr + (i * D * 2) as u64, det.hs.ptr + (q * D * 2) as u64, D * 2)?;
        }
        let me = self.mask_embed.forward(dev, &sel)?;
        let me_b = ops::to_bf16(dev, &me)?;
        let p = det.inst.numel() / D;
        let out = Tensor::new(dev, DType::F32, &[n, p])?;
        ops::gemm(dev, &me_b, &det.inst, None, Act::None, Epi::Store, &out)?;
        Ok(out)
    }

    /// Semantic segmentation logits f32 [288*288] (one map for the whole prompt).
    pub fn semantic(&self, det: &Detections) -> Result<Vec<f32>> {
        let dev = &*self.dev;
        let o = ops::linear_new(dev, &det.pix, &self.sem_head, Act::None, DType::F32)?.to_f32_vec(dev)?;
        Ok(o.chunks_exact(8).map(|c| c[0]).collect())
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
