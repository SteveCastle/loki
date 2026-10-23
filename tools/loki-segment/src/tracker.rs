//! The SAM 3.1 video tracker (`tracker.model.*`): multiplexed memory encoder, decoupled RoPE memory attention, the
//! multiplex mask decoder (16 object slots x 3 candidates) and object pointers. Follows
//! `sam3/model/video_tracking_multiplex.py` (`_prepare_memory_conditioned_features`, `_forward_sam_heads`,
//! `_encode_new_memory`) for one bucket of up to 16 objects.
//!
//! Propagation runs over a *pass*: an ordered list of frames starting at the prompted (conditioning) frame. Memory
//! selection works in pass steps, so forward, backward and wrap-around (loop) passes share one code path.
use crate::cuda::{Arg, Device};
use crate::interactive::{dense_pe, SamOutput, TwoWay, NO_OBJ_SCORE};
use crate::ops::{self, Act, Epi};
use crate::tensor::{DType, Tensor};
use crate::weights::{Conv, HostLinear, HostMlp, Linear, Norm, Weights};
use anyhow::{ensure, Result};
use std::sync::Arc;

pub const SLOTS: usize = 16;
const D: usize = 256;
const SIDE: usize = 72;
const HW: usize = SIDE * SIDE;
const LOW: usize = 288;
const HI: usize = 1008;
const MASKIN: usize = 1152;
const NUM_MASKMEM: usize = 7;
const MAX_PTRS: usize = 16;

/// Per-frame image features the tracker needs (from the propagation neck).
#[derive(Clone)]
pub struct FrameFeats {
    /// level 72 bf16 [5184, 256]
    pub feat: Tensor,
    /// sam_mask_decoder.conv_s0(level 288) bf16 [288^2, 32], conv_s1(level 144) bf16 [144^2, 64]
    pub s0: Tensor,
    pub s1: Tensor,
}

/// What a tracked frame leaves in the memory bank.
#[derive(Clone)]
pub struct FrameMem {
    pub maskmem: Tensor,
    pub feat: Tensor,
    /// multiplexed object pointers [16 * 256] (padding slots zero)
    pub ptr: Vec<f32>,
}

/// Per-object result on one frame: low-res logits (device, f32 [288*288]) and the object score logit.
pub struct ObjFrame {
    pub logits: Tensor,
    pub score: f32,
}

struct CxBlock {
    dw_w: Tensor,
    dw_b: Tensor,
    norm: Norm,
    pw1: Linear,
    pw2: Linear,
}

struct MemLayer {
    sa_q: Linear,
    sa_k: Linear,
    sa_v: Linear,
    sa_out: Linear,
    ca_q: Linear,
    ca_k: Linear,
    ca_v: Linear,
    ca_out: Linear,
    img_q: Linear,
    img_k: Linear,
    lin1: Linear,
    lin2: Linear,
    norm1: Norm,
    norm2: Norm,
    norm3: Norm,
}

pub struct Tracker {
    dev: Arc<Device>,
    // memory encoder
    down: Vec<(Conv, Norm)>,
    down_last: Linear,
    pix_proj: Linear,
    fuser: Vec<CxBlock>,
    no_obj_spatial: Vec<f32>,
    resample: (Tensor, Tensor, Tensor, usize),
    // memory attention
    layers: Vec<MemLayer>,
    mem_norm: Norm,
    rope: Tensor,
    maskmem_tpos: Vec<f32>,
    // multiplex decoder
    transformer: TwoWay,
    out_tokens: Vec<f32>,
    valid_embed: Vec<f32>,
    invalid_embed: Vec<f32>,
    dc1: Linear,
    ln1: Norm,
    dc2: Linear,
    conv_s0: Linear,
    conv_s1: Linear,
    hyper: Vec<(Linear, Linear, Linear)>,
    iou_head: HostMlp,
    obj_head: HostMlp,
    pe: Tensor,
    // pointers
    obj_ptr_proj: HostMlp,
    inter_ptr_proj: HostMlp,
    no_obj_ptr: HostLinear,
    tpos_proj: HostLinear,
    pos72: Tensor,
    pos01: Tensor,
}

/// compute_axial_cis(dim, 72, 72): dim/4 frequencies for x then y, (cos, sin) per pair.
fn axial_rope(dim: usize, side: usize) -> Vec<f32> {
    let nf = dim / 4;
    let freqs: Vec<f32> = (0..nf).map(|i| 1.0 / 10000f32.powf((4 * i) as f32 / dim as f32)).collect();
    let mut out = Vec::with_capacity(side * side * dim);
    for t in 0..side * side {
        let (tx, ty) = ((t % side) as f32, (t / side) as f32);
        for f in &freqs {
            out.push((tx * f).cos());
            out.push((tx * f).sin());
        }
        for f in &freqs {
            out.push((ty * f).cos());
            out.push((ty * f).sin());
        }
    }
    out
}

/// Antialiased-bilinear filter table (start, count, weights[kmax]) for in -> out along one axis.
fn aa_table(n_in: usize, n_out: usize) -> (Vec<i32>, Vec<i32>, Vec<f32>, usize) {
    let scale = n_in as f64 / n_out as f64;
    let fs = scale.max(1.0);
    let support = fs;
    let mut rows = Vec::new();
    let mut kmax = 0;
    for o in 0..n_out {
        let center = (o as f64 + 0.5) * scale;
        let lo = (center - support + 0.5).floor().max(0.0) as usize;
        let hi = ((center + support + 0.5).floor() as usize).min(n_in);
        let mut w: Vec<f64> = (lo..hi).map(|x| (1.0 - ((x as f64 - center + 0.5) / fs).abs()).max(0.0)).collect();
        let s: f64 = w.iter().sum();
        if s > 0.0 {
            w.iter_mut().for_each(|v| *v /= s);
        }
        kmax = kmax.max(w.len());
        rows.push((lo, w));
    }
    let mut start = Vec::new();
    let mut cnt = Vec::new();
    let mut wt = vec![0f32; n_out * kmax];
    for (o, (lo, w)) in rows.iter().enumerate() {
        start.push(*lo as i32);
        cnt.push(w.len() as i32);
        for (k, v) in w.iter().enumerate() {
            wt[o * kmax + k] = *v as f32;
        }
    }
    (start, cnt, wt, kmax)
}

/// get_1d_sine_pe(pos, dim): [sin(pos/dim_t) for dim/2 | cos(...)]
fn sine_1d(pos: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0f32; dim];
    for i in 0..half {
        let dt = 10000f32.powf((2 * (i / 2)) as f32 / half as f32);
        out[i] = (pos / dt).sin();
        out[half + i] = (pos / dt).cos();
    }
    out
}

impl Tracker {
    pub fn load(w: &Weights, t: &str) -> Result<Tracker> {
        let dev = w.dev.clone();
        let mb = format!("{t}.maskmem_backbone");
        let mut down = Vec::new();
        for i in 0..4 {
            down.push((w.conv(&format!("{mb}.mask_downsampler.encoder.{}", i * 3))?, w.norm(&format!("{mb}.mask_downsampler.encoder.{}", i * 3 + 1), 1e-6)?));
        }
        let conv1x1 = |name: &str| -> Result<Linear> {
            let s = w.shape(&format!("{name}.weight"))?;
            w.linear_from_host(&w.host(&format!("{name}.weight"))?, Some(&w.host(&format!("{name}.bias"))?), s[0], s[1])
        };
        let down_last = conv1x1(&format!("{mb}.mask_downsampler.encoder.12"))?;
        let pix_proj = conv1x1(&format!("{mb}.pix_feat_proj"))?;
        let mut fuser = Vec::new();
        for i in 0..2 {
            let p = format!("{mb}.fuser.layers.{i}");
            let gamma = w.host(&format!("{p}.gamma"))?;
            let mut w2 = w.host(&format!("{p}.pwconv2.weight"))?;
            let mut b2 = w.host(&format!("{p}.pwconv2.bias"))?;
            let k2 = w2.len() / D;
            for o in 0..D {
                for k in 0..k2 {
                    w2[o * k2 + k] *= gamma[o];
                }
                b2[o] *= gamma[o];
            }
            fuser.push(CxBlock {
                dw_w: w.f32(&format!("{p}.dwconv.weight"))?.reshape(&[D, 49]),
                dw_b: w.f32(&format!("{p}.dwconv.bias"))?,
                norm: w.norm(&format!("{p}.norm"), 1e-6)?,
                pw1: w.linear(&format!("{p}.pwconv1"))?,
                pw2: w.linear_from_host(&w2, Some(&b2), D, k2)?,
            });
        }
        let (st, cn, wt, kmax) = aa_table(HI, MASKIN);
        let resample = (
            Tensor::from_buf(dev.upload(&st)?, DType::F32, &[st.len()]),
            Tensor::from_buf(dev.upload(&cn)?, DType::F32, &[cn.len()]),
            Tensor::from_f32(&dev, &wt, &[wt.len()])?,
            kmax,
        );
        let mut layers = Vec::new();
        for i in 0..4 {
            let p = format!("{t}.transformer.encoder.layers.{i}");
            let l = |n: &str| w.linear(&format!("{p}.{n}"));
            layers.push(MemLayer {
                sa_q: l("self_attn_q_proj")?,
                sa_k: l("self_attn_k_proj")?,
                sa_v: l("self_attn_v_proj")?,
                sa_out: l("self_attn_out_proj")?,
                ca_q: l("cross_attn_q_proj")?,
                ca_k: l("cross_attn_k_proj")?,
                ca_v: l("cross_attn_v_proj")?,
                ca_out: l("cross_attn_out_proj")?,
                img_q: l("image_cross_attn_q_proj")?,
                img_k: l("image_cross_attn_k_proj")?,
                lin1: l("linear1")?,
                lin2: l("linear2")?,
                norm1: w.norm(&format!("{p}.norm1"), 1e-5)?,
                norm2: w.norm(&format!("{p}.norm2"), 1e-5)?,
                norm3: w.norm(&format!("{p}.norm3"), 1e-5)?,
            });
        }
        let dp = format!("{t}.sam_mask_decoder");
        let mut out_tokens = w.host(&format!("{dp}.obj_score_token.weight"))?;
        out_tokens.extend(w.host(&format!("{dp}.iou_token.weight"))?);
        out_tokens.extend(w.host(&format!("{dp}.mask_tokens.weight"))?);
        let mut hyper = Vec::new();
        for i in 0..3 {
            let h = format!("{dp}.output_hypernetworks_mlps.{i}.layers");
            hyper.push((w.linear(&format!("{h}.0"))?, w.linear(&format!("{h}.1"))?, w.linear(&format!("{h}.2"))?));
        }
        let gauss = w.host(&format!("{t}.image_pe_layer.positional_encoding_gaussian_matrix"))?;
        Ok(Tracker {
            down,
            down_last,
            pix_proj,
            fuser,
            no_obj_spatial: w.host(&format!("{t}.no_obj_embed_spatial"))?,
            resample,
            layers,
            mem_norm: w.norm(&format!("{t}.transformer.encoder.norm"), 1e-5)?,
            rope: Tensor::from_f32(&dev, &axial_rope(32, SIDE), &[HW, 16, 2])?,
            maskmem_tpos: w.host(&format!("{t}.maskmem_tpos_enc"))?,
            transformer: TwoWay::load(w, &format!("{dp}.transformer"))?,
            out_tokens,
            valid_embed: w.host(&format!("{t}.output_valid_embed"))?,
            invalid_embed: w.host(&format!("{t}.output_invalid_embed"))?,
            dc1: w.conv_t2(&format!("{dp}.output_upscaling.0"))?,
            ln1: w.norm(&format!("{dp}.output_upscaling.1"), 1e-6)?,
            dc2: w.conv_t2(&format!("{dp}.output_upscaling.3"))?,
            conv_s0: conv1x1(&format!("{dp}.conv_s0"))?,
            conv_s1: conv1x1(&format!("{dp}.conv_s1"))?,
            hyper,
            iou_head: w.host_mlp(&format!("{dp}.iou_prediction_head"), 3)?,
            obj_head: w.host_mlp(&format!("{dp}.pred_obj_score_head"), 3)?,
            pe: Tensor::from_f32(&dev, &dense_pe(&gauss, SIDE), &[HW, D])?,
            obj_ptr_proj: w.host_mlp(&format!("{t}.obj_ptr_proj"), 3)?,
            inter_ptr_proj: w.host_mlp(&format!("{t}.interactive_obj_ptr_proj"), 3)?,
            no_obj_ptr: w.host_linear(&format!("{t}.no_obj_ptr_linear"))?,
            tpos_proj: w.host_linear(&format!("{t}.obj_ptr_tpos_proj"))?,
            pos72: Tensor::from_f32(&dev, &crate::vit::sine_pos_2d(SIDE, SIDE), &[HW, D])?,
            pos01: Tensor::from_f32(&dev, &crate::vit::sine_pos_2d(SIDE, SIDE).iter().map(|v| v * 0.1).collect::<Vec<_>>(), &[HW, D])?,
            dev,
        })
    }

    /// Tracker features of a frame from its propagation neck (levels 288, 144, 72).
    pub fn frame_feats(&self, prop: &[Tensor]) -> Result<FrameFeats> {
        let dev = &*self.dev;
        Ok(FrameFeats {
            feat: prop[2].clone(),
            s0: ops::linear_new(dev, &prop[0], &self.conv_s0, Act::None, DType::BF16)?,
            s1: ops::linear_new(dev, &prop[1], &self.conv_s1, Act::None, DType::BF16)?,
        })
    }

    /// Object pointer from a decoder token: proj MLP, then the linear no-object blend.
    fn pointer(&self, token: &[f32], appearing: bool, interactive: bool) -> Vec<f32> {
        let p = if interactive { self.inter_ptr_proj.apply(token) } else { self.obj_ptr_proj.apply(token) };
        if appearing {
            p
        } else {
            self.no_obj_ptr.apply(&p)
        }
    }

    /// The linear no-object blend applied to a pointer whose object is absent.
    pub fn no_obj_pointer(&self, p: &[f32]) -> Vec<f32> {
        self.no_obj_ptr.apply(p)
    }

    /// Conditioning-frame entry for one object from the interactive decoder output.
    pub fn cond_object(&self, o: &SamOutput) -> (Vec<f32>, f32) {
        (self.pointer(&o.token, o.obj_score > 0.0, true), o.obj_score)
    }

    /// Memory for a frame: per-object low-res logits (device [288*288] each, slot order), object score logits,
    /// conditioning flags. `nonoverlap`: apply the per-pixel argmax constraint first (conditioning frames).
    pub fn encode_memory(&self, feats: &FrameFeats, logits: &[&Tensor], scores: &[f32], cond: &[bool], nonoverlap: bool) -> Result<Tensor> {
        let hi = self.hires(logits, nonoverlap)?;
        self.encode_hires(feats, &hi, scores, cond)
    }

    /// Low-res logits (288) of n objects -> bilinear 1008x1008 f32 [n, 1008^2]; optional non-overlap constraint
    /// (per-pixel argmax over all n objects, the others clamped to <= -10).
    pub fn hires(&self, logits: &[&Tensor], nonoverlap: bool) -> Result<Tensor> {
        let dev = &*self.dev;
        let n = logits.len();
        let p_lo = LOW * LOW;
        let lo = Tensor::new(dev, DType::F32, &[n.max(1), p_lo])?;
        for (i, l) in logits.iter().enumerate() {
            dev.dtod(lo.ptr + (i * p_lo * 4) as u64, l.ptr, p_lo * 4)?;
        }
        let p_hi = HI * HI;
        let hi = Tensor::new(dev, DType::F32, &[n.max(1), p_hi])?;
        dev.launch_n("k_resize_logits_f32", n * p_hi, &[Arg::Ptr(lo.ptr), Arg::I32(n as i32), Arg::I32(LOW as i32), Arg::I32(LOW as i32), Arg::Ptr(hi.ptr), Arg::I32(HI as i32), Arg::I32(HI as i32)])?;
        if nonoverlap && n > 1 {
            dev.launch_n("k_nonoverlap", p_hi, &[Arg::Ptr(hi.ptr), Arg::I32(n as i32), Arg::I64(p_hi as i64)])?;
        }
        Ok(hi)
    }

    /// Memory of one bucket from its objects' 1008x1008 logits (rows of `hi`, slot order).
    pub fn encode_hires(&self, feats: &FrameFeats, hi: &Tensor, scores: &[f32], cond: &[bool]) -> Result<Tensor> {
        let dev = &*self.dev;
        let n = scores.len();
        ensure!(n <= SLOTS, "at most {SLOTS} objects per bucket");
        let p_hi = HI * HI;
        let condv: Vec<f32> = cond.iter().map(|c| if *c { 1.0 } else { 0.0 }).collect();
        let cond_t = Tensor::from_f32(dev, &if condv.is_empty() { vec![0.0] } else { condv }, &[n.max(1)])?;
        let ch = 2 * SLOTS;
        let mux = Tensor::new(dev, DType::F32, &[p_hi, ch])?;
        dev.launch_n("k_mux_masks", p_hi * ch, &[Arg::Ptr(hi.ptr), Arg::Ptr(cond_t.ptr), Arg::I32(n as i32), Arg::I64(p_hi as i64), Arg::I32(SLOTS as i32), Arg::Ptr(mux.ptr)])?;
        // antialiased bilinear 1008 -> 1152 (x then y)
        let (st, cn, wt, kmax) = &self.resample;
        let tx = Tensor::new(dev, DType::F32, &[HI * MASKIN, ch])?;
        dev.launch_n(
            "k_resample_axis",
            HI * MASKIN * ch,
            &[Arg::Ptr(mux.ptr), Arg::Ptr(tx.ptr), Arg::I32(HI as i32), Arg::I32(HI as i32), Arg::I32(ch as i32), Arg::I32(1), Arg::I32(MASKIN as i32), Arg::Ptr(st.ptr), Arg::Ptr(cn.ptr), Arg::Ptr(wt.ptr), Arg::I32(*kmax as i32)],
        )?;
        drop(mux);
        let ty = Tensor::new(dev, DType::F32, &[MASKIN * MASKIN, ch])?;
        dev.launch_n(
            "k_resample_axis",
            MASKIN * MASKIN * ch,
            &[Arg::Ptr(tx.ptr), Arg::Ptr(ty.ptr), Arg::I32(HI as i32), Arg::I32(MASKIN as i32), Arg::I32(ch as i32), Arg::I32(0), Arg::I32(MASKIN as i32), Arg::Ptr(st.ptr), Arg::Ptr(cn.ptr), Arg::Ptr(wt.ptr), Arg::I32(*kmax as i32)],
        )?;
        drop(tx);
        let mut x = ops::to_bf16(dev, &ty)?;
        drop(ty);
        // downsampler: 4 x (conv k3 s2 -> LayerNorm2d -> GELU), then 1x1 to 256
        let mut side = MASKIN;
        for (conv, norm) in &self.down {
            let oside = side / 2;
            let y = Tensor::new(dev, DType::BF16, &[oside * oside, conv.co])?;
            ops::conv2d(dev, &x, side, side, conv, 2, 1, &y)?;
            let z = Tensor::new(dev, DType::BF16, &[oside * oside, conv.co])?;
            let rows = oside * oside;
            dev.launch_n(
                "k_ln_warp",
                rows * 32,
                &[Arg::Ptr(y.ptr), Arg::I32(1), Arg::Ptr(z.ptr), Arg::I64(rows as i64), Arg::I32(conv.co as i32), Arg::F32(norm.eps), Arg::Ptr(norm.w.ptr), Arg::Ptr(norm.b.ptr), Arg::I32(1)],
            )?;
            x = z;
            side = oside;
        }
        let memb = ops::linear_new(dev, &x, &self.down_last, Act::None, DType::F32)?;
        let mut h = Tensor::new(dev, DType::F32, &[HW, D])?;
        ops::linear(dev, &feats.feat, &self.pix_proj, Act::None, Epi::AddRes(&memb), &h)?;
        // fuser: 2 ConvNeXt blocks
        for b in &self.fuser {
            let hb = ops::to_bf16(dev, &h)?;
            let dw = Tensor::new(dev, DType::F32, &[HW, D])?;
            dev.launch_n("k_dwconv7", HW * D, &[Arg::Ptr(hb.ptr), Arg::Ptr(b.dw_w.ptr), Arg::Ptr(b.dw_b.ptr), Arg::Ptr(dw.ptr), Arg::I32(SIDE as i32), Arg::I32(SIDE as i32), Arg::I32(D as i32)])?;
            let nb = ops::layernorm_new(dev, &dw, &b.norm, DType::BF16)?;
            let f = ops::linear_new(dev, &nb, &b.pw1, Act::GeluErf, DType::BF16)?;
            let h2 = Tensor::new(dev, DType::F32, &[HW, D])?;
            ops::linear(dev, &f, &b.pw2, Act::None, Epi::AddRes(&h), &h2)?;
            h = h2;
        }
        // no-object spatial embedding: every slot whose (multiplexed) object score is <= 0, padding included
        let mut add = vec![0f32; D];
        for s in 0..SLOTS {
            let sc = if s < n { scores[s] } else { 0.0 };
            if sc <= 0.0 {
                for c in 0..D {
                    add[c] += self.no_obj_spatial[s * D + c];
                }
            }
        }
        let add_t = Tensor::from_f32(dev, &add, &[D])?;
        ops::add_new(dev, &h, Some(&add_t), Act::None, DType::BF16)
    }

    /// Memory-conditioned features of the current frame. `mem`: (frame memory, t_pos) for the mask memory (cond
    /// frames first); `ptrs`: (pointer block [16*256], relative position) for the object pointers.
    pub fn attend_memory(&self, feats: &FrameFeats, mem: &[(&FrameMem, i64)], ptrs: &[(&[f32], usize)], num_frames: usize) -> Result<Tensor> {
        let dev = &*self.dev;
        let nm = mem.len();
        let np = ptrs.len() * SLOTS;
        let nk = nm * HW + np;
        // keys/values inputs: image features (zeros for pointer rows), mask memory / pointers, key positions
        let mem_img = Tensor::zeros(dev, DType::BF16, &[nk.max(1), D])?;
        let mem_mm = Tensor::new(dev, DType::BF16, &[nk.max(1), D])?;
        let mut posh = vec![0f32; np * D];
        let pos_all = Tensor::new(dev, DType::F32, &[nk.max(1), D])?;
        for (j, (m, t_pos)) in mem.iter().enumerate() {
            dev.dtod(mem_img.ptr + (j * HW * D * 2) as u64, m.feat.ptr, HW * D * 2)?;
            dev.dtod(mem_mm.ptr + (j * HW * D * 2) as u64, m.maskmem.ptr, HW * D * 2)?;
            let idx = if *t_pos <= 0 || *t_pos >= NUM_MASKMEM as i64 { NUM_MASKMEM - 1 } else { NUM_MASKMEM - *t_pos as usize - 1 };
            let tpos = Tensor::from_f32(dev, &self.maskmem_tpos[idx * D..(idx + 1) * D], &[D])?;
            let dst = Tensor { buf: pos_all.buf.clone(), ptr: pos_all.ptr + (j * HW * D * 4) as u64, dtype: DType::F32, shape: vec![HW, D] };
            ops::add(dev, &self.pos72, Some(&tpos), Act::None, &dst)?;
        }
        if np > 0 {
            let max_ptrs = num_frames.min(MAX_PTRS).max(2);
            let mut ph = Vec::with_capacity(np * D);
            for (pi, (p, rel)) in ptrs.iter().enumerate() {
                ph.extend_from_slice(p);
                let pe = self.tpos_proj.apply(&sine_1d(*rel as f32 / (max_ptrs - 1) as f32, D));
                for s in 0..SLOTS {
                    posh[(pi * SLOTS + s) * D..(pi * SLOTS + s + 1) * D].copy_from_slice(&pe);
                }
            }
            let pb: Vec<u16> = ph.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
            dev.htod_at(mem_mm.ptr + (nm * HW * D * 2) as u64, crate::weights::as_bytes(&pb))?;
            dev.htod_at(pos_all.ptr + (nm * HW * D * 4) as u64, crate::weights::as_bytes(&posh))?;
        }
        let image = &feats.feat;
        let src = ops::to_f32(dev, image)?;
        let mut out = Tensor::new(dev, DType::F32, &[HW, D])?;
        // output = src + 0.1 * pos
        ops::add(dev, &src, Some(&self.pos01), Act::None, &out)?;
        let h = Tensor::new(dev, DType::BF16, &[HW, D])?;
        for l in &self.layers {
            // self-attention with RoPE on q and k
            ops::layernorm(dev, &out, Some(&l.norm1), 1e-5, &h, None)?;
            let q = ops::linear_new(dev, &h, &l.sa_q, Act::None, DType::BF16)?;
            let k = ops::linear_new(dev, &h, &l.sa_k, Act::None, DType::BF16)?;
            let v = ops::linear_new(dev, &h, &l.sa_v, Act::None, DType::BF16)?;
            self.rope(&q, HW)?;
            self.rope(&k, HW)?;
            let a = ops::mha(dev, &q, &k, &v, 8, None, None)?;
            let o2 = Tensor::new(dev, DType::F32, &[HW, D])?;
            ops::linear(dev, &a, &l.sa_out, Act::None, Epi::AddRes(&out), &o2)?;
            out = o2;
            // cross-attention: q = img_q(image) + ca_q(LN2(out)); k = img_k(mem image) + ca_k(memory) + pos
            ops::layernorm(dev, &out, Some(&l.norm2), 1e-5, &h, None)?;
            let qi = ops::linear_new(dev, image, &l.img_q, Act::None, DType::F32)?;
            let qf = Tensor::new(dev, DType::F32, &[HW, D])?;
            ops::linear(dev, &h, &l.ca_q, Act::None, Epi::AddRes(&qi), &qf)?;
            let q = ops::to_bf16(dev, &qf)?;
            self.rope(&q, HW)?;
            let a = if nk > 0 {
                let ki = Tensor::new(dev, DType::F32, &[nk, D])?;
                ops::linear(dev, &mem_img, &l.img_k, Act::None, Epi::AddRes(&pos_all), &ki)?;
                let kf = Tensor::new(dev, DType::F32, &[nk, D])?;
                ops::linear(dev, &mem_mm, &l.ca_k, Act::None, Epi::AddRes(&ki), &kf)?;
                let k = ops::to_bf16(dev, &kf)?;
                if nm > 0 {
                    self.rope(&k, nm * HW)?;
                }
                let v = ops::linear_new(dev, &mem_mm, &l.ca_v, Act::None, DType::BF16)?;
                ops::mha(dev, &q, &k, &v, 8, None, None)?
            } else {
                Tensor::zeros(dev, DType::BF16, &[HW, D])?
            };
            let o2 = Tensor::new(dev, DType::F32, &[HW, D])?;
            ops::linear(dev, &a, &l.ca_out, Act::None, Epi::AddRes(&out), &o2)?;
            out = o2;
            // FFN (GELU)
            ops::layernorm(dev, &out, Some(&l.norm3), 1e-5, &h, None)?;
            let f = ops::linear_new(dev, &h, &l.lin1, Act::GeluErf, DType::BF16)?;
            let o2 = Tensor::new(dev, DType::F32, &[HW, D])?;
            ops::linear(dev, &f, &l.lin2, Act::None, Epi::AddRes(&out), &o2)?;
            out = o2;
        }
        ops::layernorm_new(dev, &out, &self.mem_norm, DType::F32)
    }

    fn rope(&self, x: &Tensor, rows: usize) -> Result<()> {
        self.dev.launch_n(
            "k_rope_strided",
            rows * 8 * 16,
            &[Arg::Ptr(x.ptr), Arg::I64(D as i64), Arg::I32(rows as i32), Arg::I32(8), Arg::I32(32), Arg::Ptr(self.rope.ptr), Arg::I32(HW as i32)],
        )
    }

    /// Multiplex decoder on memory-conditioned features for `n` objects (slots 0..n): per object the best of the 3
    /// candidates (NO_OBJ_SCORE logits when absent), its score logit and its object pointer.
    ///
    /// `sticky` (stability, not in the stock model): when candidates' predicted IoUs are within `sticky.1` of the best,
    /// take the one that overlaps the object's previous mask (`sticky.0[slot]`, 288x288 logits) the most, instead of
    /// letting a near-tie flip the mask's extent from one frame to the next.
    pub fn decode(&self, feats: &FrameFeats, pix: &Tensor, n: usize, sticky: Option<(&[Vec<f32>], f32)>) -> Result<(Vec<ObjFrame>, Vec<f32>)> {
        let dev = &*self.dev;
        let mut tok = self.out_tokens.clone();
        let base = 2 * SLOTS * D;
        for s in 0..SLOTS {
            let e = if s < n { &self.valid_embed } else { &self.invalid_embed };
            for i in 0..3 {
                for c in 0..D {
                    tok[base + (s * 3 + i) * D + c] += e[s * D + c];
                }
            }
        }
        let nt = tok.len() / D;
        let tokens = Tensor::from_f32(dev, &tok, &[nt, D])?;
        let (hs, keys) = self.transformer.forward(dev, &tokens, pix, &self.pe)?;
        let hsv = hs.to_f32_vec(dev)?;
        let kb = ops::to_bf16(dev, &keys)?;
        let u1 = ops::conv_t2(dev, &kb, SIDE, SIDE, &self.dc1, Act::None)?;
        let u1 = ops::add_new(dev, &u1, Some(&feats.s1), Act::None, DType::F32)?;
        let u1 = ops::layernorm_new(dev, &u1, &self.ln1, DType::F32)?;
        let u1 = ops::add_new(dev, &u1, None, Act::GeluErf, DType::BF16)?;
        let u2 = ops::conv_t2(dev, &u1, 2 * SIDE, 2 * SIDE, &self.dc2, Act::None)?;
        let up = ops::add_new(dev, &u2, Some(&feats.s0), Act::GeluErf, DType::BF16)?;
        // hypernetworks for the 3 candidates of each valid slot
        let nm = n * 3;
        let hyper = Tensor::new(dev, DType::BF16, &[nm.max(1), 32])?;
        for i in 0..3 {
            let mut rows = Vec::with_capacity(n * D);
            for s in 0..n {
                let t = base + (s * 3 + i) * D;
                rows.extend_from_slice(&hsv[t..t + D]);
            }
            if n == 0 {
                break;
            }
            let t = ops::to_bf16(dev, &Tensor::from_f32(dev, &rows, &[n, D])?)?;
            let (a, b, c) = &self.hyper[i];
            let h = ops::linear_new(dev, &t, a, Act::Relu, DType::BF16)?;
            let h = ops::linear_new(dev, &h, b, Act::Relu, DType::BF16)?;
            let h = ops::linear_new(dev, &h, c, Act::None, DType::BF16)?;
            for s in 0..n {
                dev.dtod(hyper.ptr + ((s * 3 + i) * 64) as u64, h.ptr + (s * 64) as u64, 64)?;
            }
        }
        let p = LOW * LOW;
        let masks = Tensor::new(dev, DType::F32, &[nm.max(1), p])?;
        if n > 0 {
            ops::gemm(dev, &hyper, &up, None, Act::None, Epi::Store, &masks)?;
        }
        let mut objs = Vec::new();
        let mut ptr = vec![0f32; SLOTS * D];
        for s in 0..n {
            let ious = self.iou_head.apply(&hsv[(SLOTS + s) * D..(SLOTS + s + 1) * D]);
            let score = self.obj_head.apply(&hsv[s * D..(s + 1) * D])[0];
            let mut best = (0..3).max_by(|a, b| ious[*a].partial_cmp(&ious[*b]).unwrap()).unwrap();
            if let Some((prev, margin)) = sticky {
                let close: Vec<usize> = (0..3).filter(|k| ious[best] - ious[*k] <= margin).collect();
                if close.len() > 1 && s < prev.len() && prev[s].len() == p && prev[s].iter().any(|v| *v > 0.0) {
                    let mut best_iou = -1f32;
                    for k in close {
                        let c = masks.rows(s * 3 + k, 1).to_f32_vec(dev)?;
                        let (mut i, mut u) = (0usize, 0usize);
                        for (a, b) in c.iter().zip(&prev[s]) {
                            let (a, b) = (*a > 0.0, *b > 0.0);
                            i += (a && b) as usize;
                            u += (a || b) as usize;
                        }
                        let iou = if u == 0 { 0.0 } else { i as f32 / u as f32 };
                        if iou > best_iou {
                            best_iou = iou;
                            best = k;
                        }
                    }
                }
            }

            let logits = Tensor::new(dev, DType::F32, &[p])?;
            if score > 0.0 {
                dev.dtod(logits.ptr, masks.ptr + ((s * 3 + best) * p * 4) as u64, p * 4)?;
            } else {
                ops::fill_f32(dev, &logits, NO_OBJ_SCORE)?;
            }
            let t = base + (s * 3 + best) * D;
            let pv = self.pointer(&hsv[t..t + D], score > 0.0, false);
            ptr[s * D..(s + 1) * D].copy_from_slice(&pv);
            objs.push(ObjFrame { logits, score });
        }
        Ok((objs, ptr))
    }
}
