//! MiniMax H3 video VAE (comfy `MiniMaxH3VideoVAE`): 3D causal CNN encoder + ViT3D decoder.
//!
//! Numerics follow ComfyUI's fp16 path (`minimax_h3_video_vae_fp16`): fp16 weights and activations with f32
//! accumulation; the decoder's residual stream is kept in f32 (the reference keeps it in fp16).
//! Device data layouts:
//!   encoder volumes  fp16 channels-last [T, H, W, C]
//!   decoder tokens   f32 residual [B*S, 2048], fp16 GEMM operands
//!   canvas           f32 [F, H, W, 3] raw decoder output (pre de-normalization)
//! fp16 device tensors are tagged `DType::BF16` (2-byte storage); they are only ever touched by vvae kernels.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Context, Result};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const CONV_SMEM: u32 = 4 * (256 + 128) * 32 * 2; // 96 KB
/// Pick the block-tile height: 256 rows (16 warps) unless that leaves fewer than ~2 waves of blocks.
fn tile_cfg(m: usize, n: usize, sms: usize) -> (usize, u32, u32, &'static str) {
    let blocks256 = ((m + 255) / 256) * ((n + 127) / 128);
    if blocks256 >= 2 * sms {
        (256, 512, 4 * (256 + 128) * 32 * 2, "")
    } else {
        (128, 256, 4 * (128 + 128) * 32 * 2, "_s")
    }
}
const DIM: usize = 2048;
const HEADS: usize = 32;
const LAYERS: usize = 36;
const FFN: usize = 8192;

// temporal / spatial chunking constants (reference defaults)
const CLIP_LEN: usize = 17;
const TOKEN_DROP: usize = 3;
const RATIO_T: usize = 4;
const RATIO: usize = 16;
const TILE: usize = 256;
const TILE_OVERLAP_MIN: usize = 64;
const TOKENS_CHUNK: usize = (CLIP_LEN + RATIO_T - 1) / RATIO_T; // 5
const FRAME_PRE_PAD: usize = (RATIO_T - CLIP_LEN % RATIO_T) % RATIO_T; // 3
const TOKEN_OVERLAP: usize = (TOKENS_CHUNK - TOKEN_DROP % TOKENS_CHUNK) % TOKENS_CHUNK; // 2
const FRAME_OVERLAP: usize = if TOKEN_OVERLAP * RATIO_T > FRAME_PRE_PAD { TOKEN_OVERLAP * RATIO_T - FRAME_PRE_PAD } else { 0 }; // 5

/// rope inverse frequencies exactly as the reference holds them (fp16 buffer): 1 / 100^(i/8), i = 0..8
const INV_FREQ: [f32; 8] = [1.0, 0.5625, 0.316162109375, 0.1778564453125, 0.0999755859375, 0.056243896484375, 0.0316162109375, 0.0177764892578125];

static PROF: std::sync::OnceLock<crate::cuda::Profiler> = std::sync::OnceLock::new();
fn prof() -> &'static crate::cuda::Profiler {
    PROF.get_or_init(crate::cuda::Profiler::new)
}
/// Print the per-kernel-category GPU time profile (H3_PROFILE=1).
pub fn profile_report() {
    prof().report();
}

fn rh(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Reference `split_tiles`: (starts, lengths, overlaps) in pixels.
pub fn split_tiles(len: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if TILE >= len {
        return (vec![0], vec![len], vec![]);
    }
    let mut n = (len + TILE - 1) / TILE;
    let (mut overlaps, remaining) = loop {
        let ov = vec![TILE_OVERLAP_MIN; n - 1];
        let rem = (TILE * n) as i64 - (TILE_OVERLAP_MIN * (n - 1)) as i64 - len as i64;
        if rem < 0 {
            n += 1;
        } else {
            break (ov, rem as usize);
        }
    };
    for i in 0..remaining / RATIO {
        overlaps[i % (n - 1)] += RATIO;
    }
    let mut starts = vec![0usize];
    for i in 0..n - 1 {
        let s = starts[i] + TILE - overlaps[i];
        starts.push(s);
    }
    (starts, vec![TILE; n], overlaps)
}

/// Reference `_decode_temporal_chunks`: (pad_tokens, num_chunks)
fn decode_chunks(z_len: usize) -> (usize, usize) {
    let mut pseudo = z_len + TOKEN_DROP;
    let mut pad = (TOKENS_CHUNK - pseudo % TOKENS_CHUNK) % TOKENS_CHUNK;
    pseudo += pad;
    let mut num = pseudo as i64 / TOKENS_CHUNK as i64 - (TOKEN_DROP > 0) as i64;
    if num < 1 {
        pad += TOKENS_CHUNK;
        num += 1;
    }
    (pad, num as usize)
}

fn decode_pad_frames(z_len: usize, pad_tokens: usize) -> usize {
    if pad_tokens == 0 {
        return 0;
    }
    let intra_tail = CLIP_LEN % RATIO_T;
    if intra_tail == 0 {
        return pad_tokens * RATIO_T;
    }
    let before = z_len - pad_tokens;
    (0..pad_tokens).map(|k| if (before + k) % TOKENS_CHUNK == 0 { intra_tail } else { RATIO_T }).sum()
}

/// Reference `_decode_temporal_frame_plan`
fn decode_frame_plan(z_len: usize, num_chunks: usize, pad_tokens: usize) -> usize {
    let chunk_dec = TOKENS_CHUNK * RATIO_T;
    let split_count = (TOKEN_DROP > 0) as usize + 1;
    let mut total = 0usize;
    let mut final_overlap = 0usize;
    for i in 0..num_chunks {
        let ts = i * TOKENS_CHUNK;
        let te = ts + TOKENS_CHUNK + TOKEN_OVERLAP;
        let clip_tok = te.min(z_len).saturating_sub(ts.min(z_len));
        let clip_frames = clip_tok * RATIO_T;
        for j in 0..split_count {
            let fs = j * chunk_dec;
            let fe = (fs + chunk_dec).min(clip_frames);
            let n = (fe as i64 - fs as i64 - FRAME_PRE_PAD as i64).max(0) as usize;
            if j == 0 {
                total += n;
            } else {
                final_overlap = n;
            }
        }
    }
    total += final_overlap;
    total - decode_pad_frames(z_len, pad_tokens)
}

/// Latent frames produced by `encode` for `t` input frames.
pub fn encode_latent_frames(t: usize) -> usize {
    if t == 1 {
        1
    } else {
        (t + CLIP_LEN - 1) / CLIP_LEN * TOKENS_CHUNK - TOKEN_DROP
    }
}

// ---------------------------------------------------------------------------------------------
struct Conv {
    w: Tensor, // fp16 [co, ldw], taps (kt, kh, kw, ci)
    b: Tensor, // f32 [co]
    co: usize,
    ci: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    ldw: usize,
}
struct Gn {
    g: Tensor,
    b: Tensor,
}
struct ResBlock {
    n1: Gn,
    c1: Conv,
    n2: Gn,
    c2: Conv,
    nin: Option<Conv>,
}
struct Level {
    blocks: Vec<ResBlock>,
    down: Option<(Conv, usize, usize)>, // conv, time stride, space stride
}
struct Encoder {
    conv_in: Conv,
    levels: Vec<Level>,
    norm_out: Gn,
    conv_out: Conv,
    quant_w: Vec<f32>, // [48][48]
    quant_b: Vec<f32>,
}

struct Block {
    norm1: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    scale1: Tensor,
    norm2: Tensor,
    w1: Tensor, // interleaved gate/up rows
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    scale2: Tensor,
}
struct Decoder {
    pq_w: Tensor,
    pq_b: Tensor,
    xe_w: Tensor,
    xe_b: Tensor,
    regs: Tensor,
    blocks: Vec<Block>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
}

/// A channels-last fp16 volume.
struct Vol {
    t: Tensor,
    tt: usize,
    h: usize,
    w: usize,
    c: usize,
}

pub struct VideoVae {
    dev: Arc<Device>,
    enc: Encoder,
    dec: Decoder,
    lat_mean: Vec<f32>, // fp16-rounded
    lat_std: Vec<f32>,
    lat_mean_d: Tensor,
    lat_std_d: Tensor,
    pix_mean_d: Tensor,
    pix_std_d: Tensor,
    /// max spatial tiles per decoder call
    pub max_batch: usize,
    /// fp16 accumulation inside each 32-wide K slab (promoted to f32 between slabs) for encoder convs / decoder GEMMs
    pub hacc_enc: bool,
    pub hacc_dec: bool,
}

struct Ld<'a> {
    st: &'a SafeTensors,
    dev: &'a Device,
}
impl<'a> Ld<'a> {
    fn f16_raw(&self, name: &str) -> Result<(Vec<u16>, Vec<usize>)> {
        let info = self.st.info(name)?;
        ensure!(info.dtype == "F16", "{name}: expected F16, got {}", info.dtype);
        let b = self.st.bytes(name)?;
        Ok((b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect(), info.shape.clone()))
    }
    fn f16_dev(&self, name: &str) -> Result<Tensor> {
        let info = self.st.info(name)?;
        ensure!(info.dtype == "F16", "{name}: expected F16");
        let b = self.st.bytes(name)?;
        let t = Tensor::new(self.dev, DType::BF16, &info.shape)?;
        self.dev.htod_at(t.ptr, b)?;
        Ok(t)
    }
    fn f32_dev(&self, name: &str) -> Result<Tensor> {
        let v = self.st.f32s(name)?;
        let n = v.len();
        Tensor::from_f32(self.dev, &v, &[n])
    }
    fn u16_dev(&self, v: &[u16], shape: &[usize]) -> Result<Tensor> {
        let t = Tensor::new(self.dev, DType::BF16, shape)?;
        self.dev.htod(&t.buf, v)?;
        Ok(t)
    }
    fn conv(&self, prefix: &str, ci_pad: usize) -> Result<Conv> {
        let (w, shape) = self.f16_raw(&format!("{prefix}.weight"))?;
        ensure!(shape.len() == 5, "{prefix}: conv weight rank");
        let (co, ci, kt, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let ci_p = ci.max(ci_pad);
        ensure!(ci_p % 8 == 0, "{prefix}: ci {ci_p} not a multiple of 8");
        let taps = kt * kh * kw;
        let k = taps * ci_p;
        let ldw = (k + 31) / 32 * 32;
        let mut out = vec![0u16; co * ldw];
        for o in 0..co {
            for c in 0..ci {
                for tap in 0..taps {
                    out[o * ldw + tap * ci_p + c] = w[(o * ci + c) * taps + tap];
                }
            }
        }
        Ok(Conv {
            w: self.u16_dev(&out, &[co, ldw])?,
            b: self.f32_dev(&format!("{prefix}.bias"))?,
            co,
            ci: ci_p,
            kt,
            kh,
            kw,
            ldw,
        })
    }
    fn gn(&self, prefix: &str) -> Result<Gn> {
        Ok(Gn { g: self.f16_dev(&format!("{prefix}.weight"))?, b: self.f16_dev(&format!("{prefix}.bias"))? })
    }
}

impl VideoVae {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<VideoVae> {
        let st = SafeTensors::open(path)?;
        let ld = Ld { st: &st, dev: &dev };
        dev.set_max_smem("k_vv_conv3d", CONV_SMEM)?;
        dev.set_max_smem("k_vv_gemm", CONV_SMEM)?;
        dev.set_max_smem("k_vv_conv3d_h", CONV_SMEM)?;
        dev.set_max_smem("k_vv_gemm_h", CONV_SMEM)?;
        for k in ["k_vv_conv3d_s", "k_vv_conv3d_h_s", "k_vv_gemm_s", "k_vv_gemm_h_s"] {
            dev.set_max_smem(k, CONV_SMEM)?;
        }

        // ---------------- encoder
        let ch_mult = [1usize, 2, 2, 4, 4, 8];
        let space_down = [2usize, 2, 2, 2, 1, 1];
        let time_down = [1usize, 2, 2, 1, 1, 1];
        let conv_in = ld.conv("encoder.conv_in", 8)?;
        let mut levels = Vec::new();
        for l in 0..6 {
            let mut blocks = Vec::new();
            for b in 0..2 {
                let p = format!("encoder.down.{l}.block.{b}");
                let nin = if st.has(&format!("{p}.nin_shortcut.weight")) { Some(ld.conv(&format!("{p}.nin_shortcut"), 0)?) } else { None };
                blocks.push(ResBlock {
                    n1: ld.gn(&format!("{p}.norm1"))?,
                    c1: ld.conv(&format!("{p}.conv1"), 0)?,
                    n2: ld.gn(&format!("{p}.norm2"))?,
                    c2: ld.conv(&format!("{p}.conv2"), 0)?,
                    nin,
                });
            }
            let down = if space_down[l] * time_down[l] > 1 {
                Some((ld.conv(&format!("encoder.down.{l}.downsample.conv"), 0)?, time_down[l], space_down[l]))
            } else {
                None
            };
            ensure!(blocks[1].c2.co == 128 * ch_mult[l], "encoder level {l} channels");
            levels.push(Level { blocks, down });
        }
        let enc = Encoder {
            conv_in,
            levels,
            norm_out: ld.gn("encoder.norm_out")?,
            conv_out: ld.conv("encoder.conv_out", 0)?,
            quant_w: st.f32s("quant_conv.weight")?,
            quant_b: st.f32s("quant_conv.bias")?,
        };

        // ---------------- decoder
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("decoder.transformer_blocks.{i}");
            // interleave w1 rows: [gate(0..H) | up(H..2H)] -> (gate_j, up_j)
            let (w1, s1) = ld.f16_raw(&format!("{p}.ff.w1.weight"))?;
            ensure!(s1 == vec![2 * FFN, DIM], "{p}.ff.w1 shape");
            let mut w1i = vec![0u16; w1.len()];
            for r in 0..2 * FFN {
                let src = if r % 2 == 0 { r / 2 } else { FFN + r / 2 };
                w1i[r * DIM..(r + 1) * DIM].copy_from_slice(&w1[src * DIM..(src + 1) * DIM]);
            }
            let b1 = st.f32s(&format!("{p}.ff.w1.bias"))?;
            let b1i: Vec<f32> = (0..2 * FFN).map(|r| if r % 2 == 0 { b1[r / 2] } else { b1[FFN + r / 2] }).collect();
            blocks.push(Block {
                norm1: ld.f16_dev(&format!("{p}.norm1.weight"))?,
                qkv_w: ld.f16_dev(&format!("{p}.attn.to_qkv.weight"))?,
                qkv_b: ld.f32_dev(&format!("{p}.attn.to_qkv.bias"))?,
                out_w: ld.f16_dev(&format!("{p}.attn.to_out.weight"))?,
                out_b: ld.f32_dev(&format!("{p}.attn.to_out.bias"))?,
                scale1: ld.f32_dev(&format!("{p}.scale1"))?,
                norm2: ld.f16_dev(&format!("{p}.norm2.weight"))?,
                w1: ld.u16_dev(&w1i, &[2 * FFN, DIM])?,
                b1: Tensor::from_f32(&dev, &b1i, &[2 * FFN])?,
                w2: ld.f16_dev(&format!("{p}.ff.w2.weight"))?,
                b2: ld.f32_dev(&format!("{p}.ff.w2.bias"))?,
                scale2: ld.f32_dev(&format!("{p}.scale2"))?,
            });
        }
        let dec = Decoder {
            pq_w: ld.f32_dev("post_quant_conv.weight")?,
            pq_b: ld.f32_dev("post_quant_conv.bias")?,
            xe_w: ld.f32_dev("decoder.x_embedder.weight")?,
            xe_b: ld.f32_dev("decoder.x_embedder.bias")?,
            regs: ld.f16_dev("decoder.register_tokens")?,
            blocks,
            norm_out_w: ld.f16_dev("decoder.norm_out.weight")?,
            norm_out_b: ld.f16_dev("decoder.norm_out.bias")?,
            proj_w: ld.f16_dev("decoder.proj_out.weight")?,
            proj_b: ld.f32_dev("decoder.proj_out.bias")?,
        };
        let lat_mean = st.f32s("latents_mean")?;
        let lat_std = st.f32s("latents_std")?;
        // imagenet stats as the reference holds them (fp16 buffers)
        let pm: Vec<f32> = [0.485f32, 0.456, 0.406].iter().map(|&v| rh(v)).collect();
        let ps: Vec<f32> = [0.229f32, 0.224, 0.225].iter().map(|&v| rh(v)).collect();
        Ok(VideoVae {
            lat_mean_d: Tensor::from_f32(&dev, &lat_mean, &[24])?,
            lat_std_d: Tensor::from_f32(&dev, &lat_std, &[24])?,
            pix_mean_d: Tensor::from_f32(&dev, &pm, &[3])?,
            pix_std_d: Tensor::from_f32(&dev, &ps, &[3])?,
            lat_mean,
            lat_std,
            dev,
            enc,
            dec,
            max_batch: 8,
            hacc_enc: std::env::var("H3_VVAE_HACC").map(|v| v.contains('e')).unwrap_or(true),
            hacc_dec: std::env::var("H3_VVAE_HACC").map(|v| v.contains('d')).unwrap_or(true),
        })
    }

    /// Output frames of `decode` for `tl` latent frames (reference `decode_output_shape`).
    pub fn decode_frame_count(tl: usize) -> usize {
        if tl <= 1 {
            return 1;
        }
        let (pad, num) = decode_chunks(tl);
        decode_frame_plan(tl + pad, num, pad)
    }

    // =========================================================================================
    // Encoder

    fn conv(&self, x: &Tensor, tp: usize, hp: usize, wp: usize, cv: &Conv, st: usize, ss: usize, single: bool, res: Option<&Tensor>) -> Result<Vol> {
        let kt = if single { 1 } else { cv.kt };
        let woff = if single { (cv.kt - 1) * cv.kh * cv.kw * cv.ci } else { 0 };
        ensure!(tp >= kt && hp >= cv.kh && wp >= cv.kw, "conv: input too small");
        let to = (tp - kt) / st + 1;
        let ho = (hp - cv.kh) / ss + 1;
        let wo = (wp - cv.kw) / ss + 1;
        let out = Tensor::new(&self.dev, DType::BF16, &[to * ho * wo, cv.co])?;
        let m = to * ho * wo;
        let (bm, threads, smem, suf) = tile_cfg(m, cv.co, self.dev.sm_count as usize);
        let grid = (((m + bm - 1) / bm) * ((cv.co + 127) / 128)) as u32;
        // ConvParams (must match the CUDA struct layout)
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct P {
            inp: u64, tp: i32, hp: i32, wp: i32, ci: i32,
            w: u64, ldw: i32, k: i32,
            bias: u64, co: i32, _p0: i32,
            out: u64, to: i32, ho: i32, wo: i32,
            kt: i32, kh: i32, kw: i32, st: i32, sh: i32, sw: i32, _p1: i32,
            res: u64,
        }
        unsafe impl cudarc::driver::DeviceRepr for P {}
        let p = P {
            inp: x.ptr, tp: tp as i32, hp: hp as i32, wp: wp as i32, ci: cv.ci as i32,
            w: cv.w.ptr + (woff * 2) as u64, ldw: cv.ldw as i32, k: (kt * cv.kh * cv.kw * cv.ci) as i32,
            bias: cv.b.ptr, co: cv.co as i32, _p0: 0,
            out: out.ptr, to: to as i32, ho: ho as i32, wo: wo as i32,
            kt: kt as i32, kh: cv.kh as i32, kw: cv.kw as i32, st: st as i32, sh: ss as i32, sw: ss as i32, _p1: 0,
            res: res.map(|r| r.ptr).unwrap_or(0),
        };
        let f = self.dev.func(&format!("{}{}", if self.hacc_enc { "k_vv_conv3d_h" } else { "k_vv_conv3d" }, suf))?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem };
        let mut b = self.dev.stream.launch_builder(&f);
        use cudarc::driver::PushKernelArg;
        b.arg(&p);
        let name = format!("conv{}x{}->{}{}", cv.kt.min(kt), cv.ci, cv.co, if st > 1 || ss > 1 { "/s" } else { "" });
        prof().time(&self.dev, &name, || { unsafe { b.launch(cfg) }.context("k_vv_conv3d")?; Ok(()) })?;
        Ok(Vol { t: out, tt: to, h: ho, w: wo, c: cv.co })
    }

    /// GroupNorm+SiLU (gn Some) or identity (None), then pad -> padded tensor + dims
    fn gn_pad(&self, x: &Vol, gn: Option<&Gn>, front: usize, pt: usize, pb: usize, pl: usize, pr: usize) -> Result<(Tensor, usize, usize, usize)> {
        let dev = &self.dev;
        let mut res = None;
        prof().time(dev, "gn_pad", || { res = Some(self.gn_pad_(x, gn, front, pt, pb, pl, pr)?); Ok(()) })?;
        Ok(res.unwrap())
    }
    fn gn_pad_(&self, x: &Vol, gn: Option<&Gn>, front: usize, pt: usize, pb: usize, pl: usize, pr: usize) -> Result<(Tensor, usize, usize, usize)> {
        let dev = &self.dev;
        let mr = match gn {
            Some(_) => {
                let p = x.h * x.w;
                let stats = Tensor::zeros(dev, DType::F32, &[x.tt * 32 * 2 * 2])?; // doubles
                let ppb = 2048usize;
                let blocks = (p + ppb - 1) / ppb;
                dev.launch("k_vv_gn_stats", (blocks as u32, x.tt as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(x.t.ptr), Arg::I32(p as i32), Arg::I32(x.c as i32), Arg::I32(ppb as i32), Arg::Ptr(stats.ptr)])?;
                let mr = Tensor::new(dev, DType::F32, &[x.tt * 32 * 2])?;
                let cnt = (p * (x.c / 32)) as f64;
                // count passed as double: split into two f32 args is awkward; pass via I64 bits
                let n = x.tt * 32;
                let f = dev.func("k_vv_gn_finalize")?;
                let cfg = cudarc::driver::LaunchConfig { grid_dim: (((n + 255) / 256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
                let mut b = dev.stream.launch_builder(&f);
                use cudarc::driver::PushKernelArg;
                let (sp, mp, ni, eps) = (stats.ptr, mr.ptr, n as i32, 1e-6f32);
                b.arg(&sp).arg(&mp).arg(&ni).arg(&cnt).arg(&eps);
                unsafe { b.launch(cfg) }?;
                Some(mr)
            }
            None => None,
        };
        let (tp, hp, wp) = (x.tt + front, x.h + pt + pb, x.w + pl + pr);
        let out = Tensor::new(dev, DType::BF16, &[tp * hp * wp, x.c])?;
        let n = tp * hp * wp * (x.c / 8);
        dev.launch_n(
            "k_vv_gn_apply_pad",
            n,
            &[
                Arg::Ptr(x.t.ptr), Arg::I32(x.tt as i32), Arg::I32(x.h as i32), Arg::I32(x.w as i32), Arg::I32(x.c as i32),
                Arg::Ptr(mr.as_ref().map(|t| t.ptr).unwrap_or(0)),
                Arg::Ptr(gn.map(|g| g.g.ptr).unwrap_or(0)), Arg::Ptr(gn.map(|g| g.b.ptr).unwrap_or(0)),
                Arg::I32(front as i32), Arg::I32(pt as i32), Arg::I32(pb as i32), Arg::I32(pl as i32), Arg::I32(pr as i32),
                Arg::Ptr(out.ptr),
            ],
        )?;
        Ok((out, tp, hp, wp))
    }

    /// Encoder on one tile of one clip. `pix` holds u8 frames [tsrc, hs, ws, 3]; the clip covers frames
    /// t0..t0+t (indices >= tsrc repeat the last frame). Returns the raw moments (first 24 = mean) as
    /// host f32 [24, Tl, th/16, tw/16] after quant_conv, rounded to fp16 like the reference.
    pub fn encode_tile(&self, pix: &Tensor, tsrc: usize, hs: usize, ws: usize, t0: usize, t: usize, y0: usize, x0: usize, th: usize, tw: usize) -> Result<Vec<f32>> {
        let dev = &self.dev;
        let single = t == 1;
        let front = if single { 0 } else { 2 };
        let (tp, hp, wp) = (t + front, th + 2, tw + 2);
        let inp = Tensor::new(dev, DType::BF16, &[tp * hp * wp, 8])?;
        dev.launch_n(
            "k_vv_pix_in",
            tp * hp * wp,
            &[
                Arg::Ptr(pix.ptr), Arg::I32(tsrc as i32), Arg::I32(hs as i32), Arg::I32(ws as i32), Arg::I32(t0 as i32), Arg::I32(y0 as i32), Arg::I32(x0 as i32),
                Arg::I32(t as i32), Arg::I32(th as i32), Arg::I32(tw as i32), Arg::I32(front as i32), Arg::Ptr(inp.ptr), Arg::Ptr(self.pix_mean_d.ptr), Arg::Ptr(self.pix_std_d.ptr),
            ],
        )?;
        let mut h = self.conv(&inp, tp, hp, wp, &self.enc.conv_in, 1, 1, single, None)?;
        drop(inp);
        for lvl in &self.enc.levels {
            for rb in &lvl.blocks {
                let (p1, tp, hp, wp) = self.gn_pad(&h, Some(&rb.n1), front, 1, 1, 1, 1)?;
                let h1 = self.conv(&p1, tp, hp, wp, &rb.c1, 1, 1, single, None)?;
                drop(p1);
                let sc = match &rb.nin {
                    Some(nin) => Some(self.conv(&h.t, h.tt, h.h, h.w, nin, 1, 1, false, None)?),
                    None => None,
                };
                let (p2, tp, hp, wp) = self.gn_pad(&h1, Some(&rb.n2), front, 1, 1, 1, 1)?;
                drop(h1);
                let res = sc.as_ref().map(|v| &v.t).unwrap_or(&h.t);
                let out = self.conv(&p2, tp, hp, wp, &rb.c2, 1, 1, single, Some(res))?;
                h = out;
            }
            if let Some((cv, ts, ss)) = &lvl.down {
                let (pr, pb) = if *ss == 2 { (1, 1) } else { (0, 0) };
                let (p, tp, hp, wp) = self.gn_pad(&h, None, front, 0, pb, 0, pr)?;
                h = self.conv(&p, tp, hp, wp, cv, *ts, *ss, single, None)?;
            }
        }
        let (p, tp, hp, wp) = self.gn_pad(&h, Some(&self.enc.norm_out), front, 1, 1, 1, 1)?;
        drop(h);
        let o = self.conv(&p, tp, hp, wp, &self.enc.conv_out, 1, 1, single, None)?;
        drop(p);
        let (tl, lh, lw) = (o.tt, o.h, o.w);
        let raw: Vec<u16> = dev.dtoh_at(o.t.ptr, tl * lh * lw * 48)?;
        let raw: Vec<f32> = raw.into_iter().map(|b| f16::from_bits(b).to_f32()).collect();
        // quant_conv (1x1, 48 -> 48), keep the first 24 (mean) channels: [24, tl, lh, lw]
        let np = tl * lh * lw;
        let mut out = vec![0f32; 24 * np];
        for p in 0..np {
            let hp = &raw[p * 48..(p + 1) * 48];
            for o in 0..24 {
                let w = &self.enc.quant_w[o * 48..(o + 1) * 48];
                let mut a = self.enc.quant_b[o];
                for c in 0..48 {
                    a += w[c] * hp[c];
                }
                out[o * np + p] = rh(a);
            }
        }
        Ok(out)
    }

    /// Reference `tiled_encode` of one clip (mean channels only) -> [24, tl, H/16, W/16]
    fn encode_clip(&self, pix: &Tensor, tsrc: usize, h: usize, w: usize, t0: usize, t: usize) -> Result<(Vec<f32>, usize)> {
        let (yi, yl, yo) = split_tiles(h);
        let (xi, xl, xo) = split_tiles(w);
        let mut tiles: Vec<Vec<Vec<f32>>> = Vec::new();
        let mut tl = 0;
        for i in 0..yi.len() {
            let mut row = Vec::new();
            for j in 0..xi.len() {
                let v = self.encode_tile(pix, tsrc, h, w, t0, t, yi[i], xi[j], yl[i], xl[j])?;
                tl = v.len() / (24 * (yl[i] / RATIO) * (xl[j] / RATIO));
                row.push(v);
            }
            tiles.push(row);
        }
        let (hl, wl) = (h / RATIO, w / RATIO);
        let lyo: Vec<usize> = yo.iter().map(|o| o / RATIO).collect();
        let lxo: Vec<usize> = xo.iter().map(|o| o / RATIO).collect();
        let mut out = vec![0f32; 24 * tl * hl * wl];
        let wts = |p: usize, e: usize| -> (f32, f32) {
            let wb = rh(p as f32 / e as f32);
            (rh(1.0 - wb), wb)
        };
        let mut oy = 0;
        for i in 0..yi.len() {
            let th = yl[i] / RATIO;
            let keep_h = if i + 1 < yi.len() { th - lyo[i] } else { th };
            let mut ox = 0;
            let mut keep_w_last = 0;
            for j in 0..xi.len() {
                let tw = xl[j] / RATIO;
                let keep_w = if j + 1 < xi.len() { tw - lxo[j] } else { tw };
                let cur = &tiles[i][j];
                for c in 0..24 {
                    for f in 0..tl {
                        for r in 0..keep_h {
                            for q in 0..keep_w {
                                let idx = |tile_w: usize, tile_h: usize, rr: usize, qq: usize| ((c * tl + f) * tile_h + rr) * tile_w + qq;
                                let mut v = cur[idx(tw, th, r, q)];
                                if i > 0 {
                                    let up = &tiles[i - 1][j];
                                    let uh = yl[i - 1] / RATIO;
                                    let e = lyo[i - 1].min(uh).min(th);
                                    if r < e {
                                        let (wa, wb) = wts(r, e);
                                        v = up[idx(tw, uh, uh - e + r, q)] * wa + v * wb;
                                    }
                                }
                                if j > 0 {
                                    let left = &tiles[i][j - 1];
                                    let lw_ = xl[j - 1] / RATIO;
                                    let e = lxo[j - 1].min(lw_).min(tw);
                                    if q < e {
                                        let (wa, wb) = wts(q, e);
                                        v = left[idx(lw_, th, r, lw_ - e + q)] * wa + v * wb;
                                    }
                                }
                                out[((c * tl + f) * hl + oy + r) * wl + ox + q] = v;
                            }
                        }
                    }
                }
                ox += keep_w;
                keep_w_last = ox;
            }
            ensure!(keep_w_last == wl, "encode tiling width bookkeeping");
            oy += keep_h;
        }
        ensure!(oy == hl, "encode tiling height bookkeeping");
        Ok((out, tl))
    }

    /// frames: host u8 RGB [t,h,w,3] (h,w multiples of 16; t may be 1).
    /// Returns the normalized latent f32 [24, Tl, h/16, w/16] (device).
    pub fn encode(&self, frames: &[u8], t: usize, h: usize, w: usize) -> Result<Tensor> {
        ensure!(frames.len() == t * h * w * 3, "encode: frames size");
        ensure!(h % 16 == 0 && w % 16 == 0 && t >= 1, "encode: h, w must be multiples of 16");
        let (hl, wl) = (h / RATIO, w / RATIO);
        let mut clips: Vec<(Vec<f32>, usize)> = Vec::new();
        if t == 1 {
            let pix = self.dev.upload(frames)?;
            let pix = Tensor::from_buf(pix, DType::U8, &[frames.len()]);
            let (v, tl) = self.encode_clip(&pix, 1, h, w, 0, 1)?;
            // reference takes the last latent frame
            let np = hl * wl;
            let mut last = vec![0f32; 24 * np];
            for c in 0..24 {
                last[c * np..(c + 1) * np].copy_from_slice(&v[(c * tl + tl - 1) * np..(c * tl + tl) * np]);
            }
            clips.push((last, 1));
        } else {
            let nclips = (t + CLIP_LEN - 1) / CLIP_LEN;
            for ci in 0..nclips {
                let f0 = ci * CLIP_LEN;
                let f1 = (f0 + CLIP_LEN).min(t);
                let pix = self.dev.upload(&frames[f0 * h * w * 3..f1 * h * w * 3])?;
                let pix = Tensor::from_buf(pix, DType::U8, &[(f1 - f0) * h * w * 3]);
                clips.push(self.encode_clip(&pix, f1 - f0, h, w, 0, CLIP_LEN)?);
            }
        }
        let total: usize = clips.iter().map(|c| c.1).sum();
        let tl = if t == 1 { 1 } else { total - TOKEN_DROP };
        ensure!(tl == encode_latent_frames(t), "latent frame bookkeeping");
        let np = hl * wl;
        let mut out = vec![0f32; 24 * tl * np];
        let mut f_out = 0;
        for (v, ctl) in &clips {
            for f in 0..*ctl {
                if f_out + f >= tl {
                    break;
                }
                for c in 0..24 {
                    let src = &v[(c * ctl + f) * np..(c * ctl + f + 1) * np];
                    let dst = &mut out[(c * tl + f_out + f) * np..(c * tl + f_out + f + 1) * np];
                    for k in 0..np {
                        dst[k] = (src[k] - self.lat_mean[c]) / self.lat_std[c];
                    }
                }
            }
            f_out += ctl;
        }
        Tensor::from_f32(&self.dev, &out, &[24, tl, hl, wl])
    }

    // =========================================================================================
    // Decoder

    fn gemm(&self, a: &Tensor, m: usize, k: usize, w: &Tensor, bias: &Tensor, mode: i32, out: &Tensor, res: Option<&Tensor>, gate: Option<&Tensor>) -> Result<()> {
        let n = w.shape[0];
        ensure!(w.numel() == n * k, "vv gemm: K mismatch");
        let (bm, threads, smem, suf) = tile_cfg(m, n, self.dev.sm_count as usize);
        let grid = (((m + bm - 1) / bm) * ((n + 127) / 128)) as u32;
        let kname = format!("{}{}", if self.hacc_dec { "k_vv_gemm_h" } else { "k_vv_gemm" }, suf);
        let name = format!("gemm{}x{}", n, k);
        prof().time(&self.dev, &name, || self.dev.launch(
            &kname,
            (grid, 1, 1),
            (threads, 1, 1),
            smem,
            &[
                Arg::Ptr(a.ptr), Arg::Ptr(w.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::I32(k as i32), Arg::Ptr(bias.ptr), Arg::I32(mode),
                Arg::Ptr(res.map(|t| t.ptr).unwrap_or(0)), Arg::Ptr(gate.map(|t| t.ptr).unwrap_or(0)),
            ],
        ))
    }

    /// Rope table f32 [S, 24, 2] (cos, sin), emulating the reference's fp16 token ids and fp16 table.
    fn rope_table(&self, tc: usize, th: usize, tw: usize) -> Result<Tensor> {
        let coords = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let c = rh(i as f32 + 0.5);
                    let c = rh(c / n as f32);
                    let c = rh(2.0 * c);
                    rh(c - 1.0)
                })
                .collect()
        };
        let (ct, cy, cx) = (coords(tc), coords(th), coords(tw));
        let np = tc * th * tw;
        let s = np + 5;
        let mut tab = vec![0f32; s * 48];
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        for t in 0..tc {
            for y in 0..th {
                for x in 0..tw {
                    let tok = (t * th + y) * tw + x;
                    let ids = [ct[t], cy[y], cx[x]];
                    for a in 0..3 {
                        let base = two_pi * ids[a];
                        for k in 0..8 {
                            let ang = base * INV_FREQ[k];
                            let p = a * 8 + k;
                            tab[tok * 48 + p * 2] = rh(ang.cos());
                            tab[tok * 48 + p * 2 + 1] = rh(ang.sin());
                        }
                    }
                }
            }
        }
        for tok in np..s {
            for p in 0..24 {
                tab[tok * 48 + p * 2] = 1.0;
            }
        }
        Tensor::from_f32(&self.dev, &tab, &[s, 48])
    }

    /// ViT3D on `nb` tiles whose embedded tokens are in x (f32 [nb*S, 2048], modified). Returns proj f32 [nb*S, 3072].
    fn vit(&self, x: &Tensor, nb: usize, s: usize, rope: &Tensor) -> Result<Tensor> {
        let dev = &self.dev;
        let m = nb * s;
        let xn = Tensor::new(dev, DType::BF16, &[m, DIM])?;
        let qkv = Tensor::new(dev, DType::BF16, &[m, 3 * DIM])?;
        let att = Tensor::new(dev, DType::BF16, &[m, DIM])?;
        let hb = Tensor::new(dev, DType::BF16, &[m, FFN])?;
        let eps = 1e-5f32;
        let norm = |src: &Tensor, w: &Tensor, b: Option<&Tensor>| -> Result<()> {
            prof().time(dev, "norm", || dev.launch("k_vv_norm_rows", (m as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(src.ptr), Arg::Ptr(xn.ptr), Arg::Ptr(w.ptr), Arg::Ptr(b.map(|t| t.ptr).unwrap_or(0)), Arg::I32(b.is_some() as i32), Arg::F32(eps)]))
        };
        let scale_log2 = 0.125f32 * std::f32::consts::LOG2_E;
        for blk in &self.dec.blocks {
            norm(x, &blk.norm1, None)?;
            self.gemm(&xn, m, DIM, &blk.qkv_w, &blk.qkv_b, 0, &qkv, None, None)?;
            prof().time(dev, "qk_rope", || dev.launch("k_vv_qk_rope", (((m * HEADS + 7) / 8) as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(qkv.ptr), Arg::I32(m as i32), Arg::I32(s as i32), Arg::Ptr(rope.ptr), Arg::F32(eps)]))?;
            prof().time(dev, "flash", || dev.launch("k_vv_flash_d64", (((s + 63) / 64) as u32, HEADS as u32, nb as u32), (128, 1, 1), 0, &[Arg::Ptr(qkv.ptr), Arg::Ptr(att.ptr), Arg::I32(s as i32), Arg::F32(scale_log2)]))?;
            self.gemm(&att, m, DIM, &blk.out_w, &blk.out_b, 2, x, Some(x), Some(&blk.scale1))?;
            norm(x, &blk.norm2, None)?;
            self.gemm(&xn, m, DIM, &blk.w1, &blk.b1, 3, &hb, None, None)?;
            self.gemm(&hb, m, FFN, &blk.w2, &blk.b2, 2, x, Some(x), Some(&blk.scale2))?;
        }
        drop(qkv);
        drop(hb);
        drop(att);
        norm(x, &self.dec.norm_out_w, Some(&self.dec.norm_out_b))?;
        let proj = Tensor::new(dev, DType::F32, &[m, 3072])?;
        self.gemm(&xn, m, DIM, &self.dec.proj_w, &self.dec.proj_b, 1, &proj, None, None)?;
        Ok(proj)
    }

    /// Embed one tile's tokens into x rows [row0, row0+S).
    fn embed(&self, z: &Tensor, tz: usize, hz: usize, wz: usize, t0: usize, tc: usize, y0: usize, x0: usize, th: usize, tw: usize, x: &Tensor, row0: usize) -> Result<()> {
        let s = tc * th * tw + 5;
        self.dev.launch(
            "k_vv_embed",
            (s as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(z.ptr), Arg::I32(tz as i32), Arg::I32(hz as i32), Arg::I32(wz as i32), Arg::I32(t0 as i32), Arg::I32(y0 as i32), Arg::I32(x0 as i32),
                Arg::I32(tc as i32), Arg::I32(th as i32), Arg::I32(tw as i32), Arg::Ptr(self.lat_mean_d.ptr), Arg::Ptr(self.lat_std_d.ptr),
                Arg::Ptr(self.dec.pq_w.ptr), Arg::Ptr(self.dec.pq_b.ptr), Arg::Ptr(self.dec.xe_w.ptr), Arg::Ptr(self.dec.xe_b.ptr), Arg::Ptr(self.dec.regs.ptr),
                Arg::Ptr(x.ptr + (row0 * DIM * 4) as u64),
            ],
        )
    }

    /// Raw decoder output of one tile (debug): z frames [t0, t0+tc) rows/cols in latent units -> f32 [3, 4*tc, th*16, tw*16].
    pub fn decode_tile_raw(&self, z: &Tensor, t0: usize, tc: usize, y0: usize, x0: usize, th: usize, tw: usize) -> Result<Vec<f32>> {
        let (tz, hz, wz) = (z.shape[1], z.shape[2], z.shape[3]);
        let s = tc * th * tw + 5;
        let x = Tensor::new(&self.dev, DType::F32, &[s, DIM])?;
        self.embed(z, tz, hz, wz, t0, tc, y0, x0, th, tw, &x, 0)?;
        let rope = self.rope_table(tc, th, tw)?;
        let proj = self.vit(&x, 1, s, &rope)?;
        let p = proj.to_f32_vec(&self.dev)?;
        let (f, hh, ww) = (4 * tc, th * 16, tw * 16);
        let mut out = vec![0f32; 3 * f * hh * ww];
        for c in 0..3 {
            for fi in 0..f {
                for y in 0..hh {
                    for xx in 0..ww {
                        let tok = ((fi / 4) * th + y / 16) * tw + xx / 16;
                        let col = c * 1024 + (fi % 4) * 256 + (y % 16) * 16 + xx % 16;
                        out[((c * f + fi) * hh + y) * ww + xx] = p[tok * 3072 + col];
                    }
                }
            }
        }
        Ok(out)
    }

    /// Reference `tiled_decode` of latent frames [t0, t0+tc) into canvas f32 [4*tc, H, W, 3].
    fn decode_spatial(&self, z: &Tensor, t0: usize, tc: usize, canvas: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let (tz, hz, wz) = (z.shape[1], z.shape[2], z.shape[3]);
        let (hh, ww) = (hz * RATIO, wz * RATIO);
        let f = tc * RATIO_T;
        let (yi, yl, yo) = split_tiles(hh);
        let (xi, xl, xo) = split_tiles(ww);
        let (th, tw) = (yl[0] / RATIO, xl[0] / RATIO);
        let s = tc * th * tw + 5;
        let rope = self.rope_table(tc, th, tw)?;
        let oy_max = yo.iter().copied().max().unwrap_or(0);
        let strip = if oy_max > 0 { Some(Tensor::new(dev, DType::F32, &[f * oy_max * ww * 3])?) } else { None };
        let tiles: Vec<(usize, usize)> = (0..yi.len()).flat_map(|i| (0..xi.len()).map(move |j| (i, j))).collect();
        // balanced batches of at most max_batch tiles (e.g. 28 tiles -> 4 x 7)
        let nbatches = (tiles.len() + self.max_batch.max(1) - 1) / self.max_batch.max(1);
        let nb_max = (tiles.len() + nbatches - 1) / nbatches;
        for batch in tiles.chunks(nb_max) {
            let nb = batch.len();
            let x = Tensor::new(dev, DType::F32, &[nb * s, DIM])?;
            for (bi, &(i, j)) in batch.iter().enumerate() {
                self.embed(z, tz, hz, wz, t0, tc, yi[i] / RATIO, xi[j] / RATIO, th, tw, &x, bi * s)?;
            }
            let proj = self.vit(&x, nb, s, &rope)?;
            drop(x);
            for (bi, &(i, j)) in batch.iter().enumerate() {
                let oy = if i > 0 { yo[i - 1] } else { 0 };
                let ox = if j > 0 { xo[j - 1] } else { 0 };
                let n = f * yl[i] * xl[j];
                dev.launch_n(
                    "k_vv_place_tile",
                    n,
                    &[
                        Arg::Ptr(proj.ptr + (bi * s * 3072 * 4) as u64), Arg::I32(th as i32), Arg::I32(tw as i32), Arg::I32(f as i32), Arg::Ptr(canvas.ptr),
                        Arg::I32(hh as i32), Arg::I32(ww as i32), Arg::I32(yi[i] as i32), Arg::I32(xi[j] as i32),
                        Arg::Ptr(strip.as_ref().map(|t| t.ptr).unwrap_or(0)), Arg::I32(oy_max as i32), Arg::I32(oy as i32), Arg::I32(ox as i32),
                    ],
                )?;
                if j + 1 == xi.len() && i + 1 < yi.len() {
                    let st = strip.as_ref().unwrap();
                    dev.launch_n(
                        "k_vv_copy_strip",
                        f * yo[i] * ww * 3,
                        &[Arg::Ptr(canvas.ptr), Arg::I32(f as i32), Arg::I32(hh as i32), Arg::I32(ww as i32), Arg::I32(yi[i + 1] as i32), Arg::I32(yo[i] as i32), Arg::Ptr(st.ptr), Arg::I32(oy_max as i32)],
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Finalize canvas frames [src0, src0+n) (first e blended with ovl tail) to u8 and hand them to the sink.
    fn emit(&self, canvas: &Tensor, src0: usize, n: usize, hw: usize, ovl: Option<(&Tensor, usize)>, e: usize, sink: &mut dyn FnMut(&[u8], usize) -> Result<()>) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let out = Tensor::new(&self.dev, DType::U8, &[n * hw * 3])?;
        let (op, ol) = ovl.map(|(t, l)| (t.ptr, l)).unwrap_or((0, 0));
        self.dev.launch_n(
            "k_vv_finalize",
            n * hw * 3,
            &[
                Arg::Ptr(canvas.ptr), Arg::I32(src0 as i32), Arg::I32(n as i32), Arg::I64(hw as i64), Arg::Ptr(op), Arg::I32(ol as i32), Arg::I32(e as i32),
                Arg::Ptr(self.pix_mean_d.ptr), Arg::Ptr(self.pix_std_d.ptr), Arg::Ptr(out.ptr),
            ],
        )?;
        let host: Vec<u8> = self.dev.dtoh_at(out.ptr, n * hw * 3)?;
        sink(&host, n)
    }

    /// z: f32 [24,Tl,hl,wl] normalized latent (device). Streams decoded RGB u8 frames [n,H,W,3] (host) to `sink`
    /// in temporal order as they finish; returns total frames.
    pub fn decode(&self, z: &Tensor, tl: usize, hl: usize, wl: usize, sink: &mut dyn FnMut(&[u8], usize) -> Result<()>) -> Result<usize> {
        ensure!(z.dtype == DType::F32 && z.numel() == 24 * tl * hl * wl, "decode: z must be f32 [24, tl, hl, wl]");
        let z = z.reshape(&[24, tl, hl, wl]);
        let dev = &self.dev;
        let (hh, ww) = (hl * RATIO, wl * RATIO);
        let hw = hh * ww;
        if tl == 1 {
            let canvas = Tensor::new(dev, DType::F32, &[RATIO_T * hw * 3])?;
            self.decode_spatial(&z, 0, 1, &canvas)?;
            self.emit(&canvas, RATIO_T - 1, 1, hw, None, 0, sink)?;
            return Ok(1);
        }
        let total = Self::decode_frame_count(tl);
        let (pad, num_chunks) = decode_chunks(tl);
        let z_len = tl + pad;
        let chunk_dec = TOKENS_CHUNK * RATIO_T;
        let mut written = 0usize;
        let mut ovl: Option<(Tensor, usize)> = None; // dec_overlap frames
        let mut emit_limited = |this: &Self, canvas: &Tensor, src0: usize, n: usize, ovl: Option<(&Tensor, usize)>, e: usize, written: &mut usize| -> Result<()> {
            let n = n.min(total.saturating_sub(*written));
            this.emit(canvas, src0, n, hw, ovl, e.min(n), sink)?;
            *written += n;
            Ok(())
        };
        for i in 0..num_chunks {
            let ts = i * TOKENS_CHUNK;
            let te = (ts + TOKENS_CHUNK + TOKEN_OVERLAP).min(z_len);
            let tc = te - ts;
            let f = tc * RATIO_T;
            let canvas = Tensor::new(dev, DType::F32, &[f * hw * 3])?;
            self.decode_spatial(&z, ts, tc, &canvas)?;
            // j = 0 part
            let fe0 = chunk_dec.min(f);
            let n0 = fe0.saturating_sub(FRAME_PRE_PAD);
            let e = match &ovl {
                Some((_, l)) => (*l).min(n0).min(FRAME_OVERLAP),
                None => 0,
            };
            emit_limited(self, &canvas, FRAME_PRE_PAD, n0, ovl.as_ref().map(|(t, l)| (t, *l)), e, &mut written)?;
            // j = 1 part -> new overlap
            let fs1 = chunk_dec;
            let fe1 = (fs1 + chunk_dec).min(f);
            let n1 = (fe1 as i64 - fs1 as i64 - FRAME_PRE_PAD as i64).max(0) as usize;
            if n1 > 0 {
                let t = Tensor::new(dev, DType::F32, &[n1 * hw * 3])?;
                dev.dtod(t.ptr, canvas.ptr + ((fs1 + FRAME_PRE_PAD) * hw * 3 * 4) as u64, n1 * hw * 3 * 4)?;
                ovl = Some((t, n1));
            } else {
                ovl = Some((Tensor::new(dev, DType::F32, &[1])?, 0));
            }
            drop(canvas);
            if i + 1 == num_chunks {
                if let Some((t, l)) = &ovl {
                    if *l > 0 {
                        emit_limited(self, t, 0, *l, None, 0, &mut written)?;
                    }
                }
            }
        }
        if written != total {
            bail!("decode: wrote {written} frames, expected {total}");
        }
        Ok(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frame_counts() {
        assert_eq!(encode_latent_frames(22), 7);
        assert_eq!(encode_latent_frames(39), 12);
        assert_eq!(encode_latent_frames(1), 1);
        assert_eq!(VideoVae::decode_frame_count(7), 22);
        assert_eq!(VideoVae::decode_frame_count(12), 39);
        assert_eq!(VideoVae::decode_frame_count(1), 1);
    }
    #[test]
    fn tiles() {
        assert_eq!(split_tiles(448), (vec![0, 192], vec![256, 256], vec![64]));
        assert_eq!(split_tiles(256), (vec![0], vec![256], vec![]));
        assert_eq!(split_tiles(384), (vec![0, 128], vec![256, 256], vec![128]));
        assert_eq!(split_tiles(640), (vec![0, 192, 384], vec![256; 3], vec![64, 64]));
    }
}
