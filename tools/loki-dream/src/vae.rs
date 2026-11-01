//! FLUX.2 VAE with the small decoder (`flux2-vae_small.safetensors`, diffusers AutoencoderKLFlux2 keys):
//! encoder 128-256-512-512 / decoder 384-384-192-96, 2 (encoder) / 3 (decoder) ResNet blocks per level, GroupNorm(32,
//! eps 1e-6) + SiLU, single-head mid attention, 32 latent channels at 1/8, then a 2x2 patch (128 channels at 1/16) and a
//! BatchNorm standardization (running mean/var, eps 1e-4) into the DiT's latent space.
//! Channels-last bf16 planes with f32 accumulation (the reference runs the VAE in fp32).
//!
//! The same network (with the full 512-512-256-128 decoder) is the SDXL VAE: `load_sd` reads it from an SDXL checkpoint
//! (`first_stage_model.*`, original LDM names) or a standalone SD VAE file, with 4 latent channels at 1/8 scaled by
//! 0.13025 and no patching or BatchNorm (`encode_sd` / `decode_sd`).
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use cudarc::driver::PushKernelArg;
use std::path::Path;
use std::sync::Arc;

const CONV_SMEM: u32 = 65536;
const GN_EPS: f32 = 1e-6;
pub const BN_EPS: f32 = 1e-4;

/// A channels-last bf16 feature plane [h, w, c].
#[derive(Clone)]
pub struct Plane {
    pub t: Tensor,
    pub h: usize,
    pub w: usize,
    pub c: usize,
}
impl Plane {
    pub fn new(dev: &Device, h: usize, w: usize, c: usize) -> Result<Plane> {
        Ok(Plane { t: Tensor::new(dev, DType::BF16, &[h * w, c])?, h, w, c })
    }
    pub fn npix(&self) -> usize {
        self.h * self.w
    }
}

/// Conv weights: [Co][Kpad] bf16 with (ky, kx, ci) ordering; bias f32 [Co].
pub struct Conv {
    w: Tensor,
    b: Tensor,
    pub co: usize,
    pub ci: usize,
    kh: usize,
    kw: usize,
    kpad: usize,
}

impl Conv {
    /// `prefix.weight` [Co, Ci, kh, kw] (or a Linear [Co, Ci] as a 1x1 conv); `ci_pad` / `co_pad` zero-pad channels.
    /// A causal 3D conv [Co, Ci, kt, kh, kw] applied to a single frame keeps its last temporal slice (the earlier taps
    /// only ever see the zero padding in front of the first frame).
    pub(crate) fn load(l: &mut Loader, prefix: &str, ci_pad: Option<usize>, co_pad: Option<usize>) -> Result<Conv> {
        let wname = format!("{prefix}.weight");
        let info = l.st.info(&wname)?.clone();
        let mut w = l.st.f32s(&wname)?;
        let (co, ci, kh, kw) = match info.shape.len() {
            5 => {
                let (co, ci, kt, kh, kw) = (info.shape[0], info.shape[1], info.shape[2], info.shape[3], info.shape[4]);
                let mut last = Vec::with_capacity(co * ci * kh * kw);
                for oc in 0..co * ci {
                    let base = (oc * kt + kt - 1) * kh * kw;
                    last.extend_from_slice(&w[base..base + kh * kw]);
                }
                w = last;
                (co, ci, kh, kw)
            }
            4 => (info.shape[0], info.shape[1], info.shape[2], info.shape[3]),
            2 => (info.shape[0], info.shape[1], 1, 1),
            _ => anyhow::bail!("{prefix}: unexpected conv weight rank"),
        };
        Self::from_parts(l, prefix, &w, (co, ci, kh, kw), ci_pad, co_pad)
    }
    pub(crate) fn from_parts(l: &mut Loader, prefix: &str, w: &[f32], (co, ci, kh, kw): (usize, usize, usize, usize), ci_pad: Option<usize>, co_pad: Option<usize>) -> Result<Conv> {
        let ci_p = ci_pad.unwrap_or(ci).max(ci);
        let co_p = co_pad.unwrap_or(co).max(co);
        ensure!(ci_p % 8 == 0 && co_p % 8 == 0, "{prefix}: padded channels must be multiples of 8");
        let k = kh * kw * ci_p;
        let kpad = (k + 31) / 32 * 32;
        let mut packed = vec![0u16; co_p * kpad];
        for o in 0..co {
            for i in 0..ci {
                for ky in 0..kh {
                    for kx in 0..kw {
                        packed[o * kpad + (ky * kw + kx) * ci_p + i] = bf16_bits(w[((o * ci + i) * kh + ky) * kw + kx]);
                    }
                }
            }
        }
        let mut bias = vec![0f32; co_p];
        if l.st.has(&format!("{prefix}.bias")) && l.st.info(&format!("{prefix}.bias"))?.shape.iter().product::<usize>() == co {
            bias[..co].copy_from_slice(&l.st.f32s(&format!("{prefix}.bias"))?);
        }
        l.uploaded += packed.len() * 2;
        Ok(Conv { w: Tensor::from_bf16(&l.dev, &packed, &[co_p, kpad])?, b: Tensor::from_f32(&l.dev, &bias, &[co_p])?, co: co_p, ci: ci_p, kh, kw, kpad })
    }
}

/// Implicit-GEMM 2D convolution on channels-last planes (`up2`: read the input as its 2x nearest upsample; `res`: add a
/// residual of the output's shape).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv(dev: &Device, inp: &Plane, c: &Conv, stride: usize, pad: usize, up2: bool, res: Option<&Plane>, out: &Plane) -> Result<()> {
    ensure!(inp.c == c.ci, "conv: input channels {} != {}", inp.c, c.ci);
    ensure!(out.c == c.co, "conv: output channels {} != {}", out.c, c.co);
    let p = ConvParamsRaw {
        inp: inp.t.ptr, ih: inp.h as i32, iw: inp.w as i32, ci: inp.c as i32, _p0: 0,
        w: c.w.ptr, bias: c.b.ptr, co: c.co as i32, kpad: c.kpad as i32,
        out: out.t.ptr, oh: out.h as i32, ow: out.w as i32,
        kh: c.kh as i32, kw: c.kw as i32, stride: stride as i32, pad_t: pad as i32, pad_l: pad as i32, up2: up2 as i32,
        mode: res.is_some() as i32, _p1: 0, res: res.map(|r| r.t.ptr).unwrap_or(0),
    };
    let m = out.npix();
    let grid = (((c.co + 127) / 128) * ((m + 127) / 128)) as u32;
    let f = dev.func("k_conv2d_nhwc")?;
    let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: CONV_SMEM };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&p);
    unsafe { b.launch(cfg) }?;
    Ok(())
}

/// Single-head softmax(q k^T / sqrt(C)) v over all pixels of a fused [q | k | v] plane (3C channels) -> [h, w, C].
pub(crate) fn pixel_attention(dev: &Device, qkv: &Plane, c: usize) -> Result<Plane> {
    let n = qkv.npix();
    let n_pad = (n + 7) / 8 * 8;
    let q = Tensor::new(dev, DType::BF16, &[n, c])?;
    let k = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
    let v = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
    for (i, t) in [&q, &k, &v].iter().enumerate() {
        dev.launch_n("k_slice_channels", n * c, &[Arg::Ptr(qkv.t.ptr), Arg::Ptr(t.ptr), Arg::I64(n as i64), Arg::I32((3 * c) as i32), Arg::I32((i * c) as i32), Arg::I32(c as i32)])?;
    }
    let vt = Tensor::new(dev, DType::BF16, &[c, n_pad])?;
    {
        let f = dev.func("k_transpose_bf16")?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (((c + 31) / 32) as u32, ((n_pad + 31) / 32) as u32, 1), block_dim: (32, 8, 1), shared_mem_bytes: 0 };
        let (r, cc) = (n_pad as i32, c as i32);
        let mut b = dev.stream.launch_builder(&f);
        b.arg(&v.ptr).arg(&vt.ptr).arg(&r).arg(&cc);
        unsafe { b.launch(cfg) }?;
    }
    drop(v);
    let o = Plane::new(dev, qkv.h, qkv.w, c)?;
    let qb = 2048usize.min(n);
    let s = Tensor::new(dev, DType::F32, &[qb, n_pad])?;
    let p = Tensor::new(dev, DType::BF16, &[qb, n_pad])?;
    let scale = 1.0 / (c as f32).sqrt();
    let mut q0 = 0;
    while q0 < n {
        let rows = qb.min(n - q0);
        let sv = s.rows(0, rows);
        let pv = p.rows(0, rows);
        ops::gemm_bf16(dev, &q.rows(q0, rows), &k, None, Act::None, Epi::Store, &sv)?;
        dev.launch("k_softmax_rows", (rows as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(sv.ptr), Arg::Ptr(pv.ptr), Arg::I32(n as i32), Arg::I32(n_pad as i32), Arg::F32(scale)])?;
        ops::gemm_bf16(dev, &pv, &vt, None, Act::None, Epi::Store, &o.t.rows(q0, rows))?;
        q0 += rows;
    }
    drop(s);
    drop(p);
    Ok(o)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParamsRaw {
    inp: u64, ih: i32, iw: i32, ci: i32, _p0: i32,
    w: u64, bias: u64, co: i32, kpad: i32,
    out: u64, oh: i32, ow: i32,
    kh: i32, kw: i32, stride: i32, pad_t: i32, pad_l: i32, up2: i32,
    mode: i32, _p1: i32, res: u64,
}
unsafe impl cudarc::driver::DeviceRepr for ConvParamsRaw {}

struct Norm {
    g: Tensor,
    b: Tensor,
}
struct ResBlock {
    n1: Norm,
    c1: Conv,
    n2: Norm,
    c2: Conv,
    shortcut: Option<Conv>,
}
struct AttnBlock {
    norm: Norm,
    qkv: Conv,
    out: Conv,
}

pub struct Vae {
    dev: Arc<Device>,
    enc_in: Conv,
    enc_down: Vec<(Vec<ResBlock>, Option<Conv>)>,
    enc_mid: (ResBlock, AttnBlock, ResBlock),
    enc_norm: Norm,
    enc_out: Conv,
    quant: Conv,
    post_quant: Conv,
    dec_in: Conv,
    dec_mid: (ResBlock, AttnBlock, ResBlock),
    dec_up: Vec<(Vec<ResBlock>, Option<Conv>)>,
    dec_norm: Norm,
    dec_out: Conv,
    pub bn_mean: Option<Tensor>,
    pub bn_var: Option<Tensor>,
}

/// SDXL's latent scale factor.
pub const SDXL_SCALE: f32 = 0.13025;

/// The diffusers name of an SD VAE tensor stored under its original LDM name (`first_stage_model.` stripped; decoder
/// levels counted from the top, `up_levels` of them); None for tensors that are not the VAE's. Diffusers names pass
/// through unchanged.
pub fn sd_vae_name(k: &str, up_levels: usize) -> Option<String> {
    let k = k.strip_prefix("first_stage_model.").unwrap_or(k);
    if !(k.starts_with("encoder.") || k.starts_with("decoder.") || k.starts_with("quant_conv.") || k.starts_with("post_quant_conv.")) {
        return None;
    }
    if k.contains("_blocks.") || k.contains("conv_norm_out") || k.contains("mid_block") {
        return Some(k.to_string());
    }
    let parts: Vec<&str> = k.split('.').collect();
    let res = |rest: &[&str]| rest.join(".").replace("nin_shortcut", "conv_shortcut");
    let out = match parts.as_slice() {
        [side, "down", i, "block", j, rest @ ..] => format!("{side}.down_blocks.{i}.resnets.{j}.{}", res(rest)),
        [side, "down", i, "downsample", rest @ ..] => format!("{side}.down_blocks.{i}.downsamplers.0.{}", rest.join(".")),
        [side, "up", i, "block", j, rest @ ..] => format!("{side}.up_blocks.{}.resnets.{j}.{}", up_levels - 1 - i.parse::<usize>().ok()?, res(rest)),
        [side, "up", i, "upsample", rest @ ..] => format!("{side}.up_blocks.{}.upsamplers.0.{}", up_levels - 1 - i.parse::<usize>().ok()?, rest.join(".")),
        [side, "mid", "block_1", rest @ ..] => format!("{side}.mid_block.resnets.0.{}", res(rest)),
        [side, "mid", "block_2", rest @ ..] => format!("{side}.mid_block.resnets.1.{}", res(rest)),
        [side, "mid", "attn_1", what, rest @ ..] => {
            let n = match *what {
                "norm" => "group_norm",
                "q" => "to_q",
                "k" => "to_k",
                "v" => "to_v",
                "proj_out" => "to_out.0",
                _ => return None,
            };
            format!("{side}.mid_block.attentions.0.{n}.{}", rest.join("."))
        }
        [side, "norm_out", rest @ ..] => format!("{side}.conv_norm_out.{}", rest.join(".")),
        _ => k.to_string(),
    };
    Some(out)
}

fn norm(l: &mut Loader, p: &str) -> Result<Norm> {
    Ok(Norm { g: l.f32(&format!("{p}.weight"))?, b: l.f32(&format!("{p}.bias"))? })
}

impl Vae {
    pub fn init_kernels(dev: &Device) -> Result<()> {
        dev.set_max_smem("k_conv2d_nhwc", CONV_SMEM)
    }

    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Vae> {
        let st = SafeTensors::open(path)?;
        ensure!(st.has("bn.running_mean") && st.has("decoder.conv_in.weight"), "{} is not a FLUX.2 VAE", path.display());
        Self::from_st(dev, st)
    }

    /// The SD / SDXL VAE of a checkpoint (`first_stage_model.*`) or a standalone SD VAE file (LDM or diffusers names).
    pub fn load_sd(dev: Arc<Device>, path: &Path) -> Result<Vae> {
        let st = SafeTensors::open(path)?;
        let up_levels = st
            .tensors
            .keys()
            .filter_map(|k| k.strip_prefix("first_stage_model.").unwrap_or(k).strip_prefix("decoder.up."))
            .filter_map(|r| r.split('.').next()?.parse::<usize>().ok())
            .max()
            .map(|m| m + 1)
            .unwrap_or(4);
        let st = st.renamed(|k| sd_vae_name(k, up_levels));
        ensure!(st.has("decoder.conv_in.weight") && st.has("post_quant_conv.weight"), "{} has no SD VAE (first_stage_model.*)", st.path);
        ensure!(!st.has("bn.running_mean"), "{} is a FLUX.2 VAE, not an SD VAE", st.path);
        Self::from_st(dev, st)
    }

    fn from_st(dev: Arc<Device>, st: SafeTensors) -> Result<Vae> {
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let res = |l: &mut Loader, p: &str| -> Result<ResBlock> {
            Ok(ResBlock {
                n1: norm(l, &format!("{p}.norm1"))?,
                c1: Conv::load(l, &format!("{p}.conv1"), None, None)?,
                n2: norm(l, &format!("{p}.norm2"))?,
                c2: Conv::load(l, &format!("{p}.conv2"), None, None)?,
                shortcut: if l.st.has(&format!("{p}.conv_shortcut.weight")) { Some(Conv::load(l, &format!("{p}.conv_shortcut"), None, None)?) } else { None },
            })
        };
        let attn = |l: &mut Loader, p: &str| -> Result<AttnBlock> {
            // to_q | to_k | to_v fused into one 1x1 conv
            let mut w = Vec::new();
            let mut b = Vec::new();
            let c = l.st.info(&format!("{p}.to_q.weight"))?.shape[0];
            for n in ["to_q", "to_k", "to_v"] {
                w.extend(l.st.f32s(&format!("{p}.{n}.weight"))?);
                b.extend(l.st.f32s(&format!("{p}.{n}.bias"))?);
            }
            let mut qkv = Conv::from_parts(l, &format!("{p}.to_q"), &w, (3 * c, c, 1, 1), None, None)?;
            qkv.b = Tensor::from_f32(&l.dev, &b, &[3 * c])?;
            Ok(AttnBlock { norm: norm(l, &format!("{p}.group_norm"))?, qkv, out: Conv::load(l, &format!("{p}.to_out.0"), None, None)? })
        };
        let enc_in = Conv::load(&mut l, "encoder.conv_in", Some(8), None)?;
        let mut enc_down = Vec::new();
        let count = |st: &SafeTensors, p: &str| (0..).take_while(|j| st.has(&format!("{p}.resnets.{j}.conv1.weight"))).count();
        for i in 0..4 {
            let p = format!("encoder.down_blocks.{i}");
            let blocks = (0..count(l.st, &p)).map(|j| res(&mut l, &format!("{p}.resnets.{j}"))).collect::<Result<Vec<_>>>()?;
            let down = if l.st.has(&format!("{p}.downsamplers.0.conv.weight")) { Some(Conv::load(&mut l, &format!("{p}.downsamplers.0.conv"), None, None)?) } else { None };
            enc_down.push((blocks, down));
        }
        let enc_mid = (res(&mut l, "encoder.mid_block.resnets.0")?, attn(&mut l, "encoder.mid_block.attentions.0")?, res(&mut l, "encoder.mid_block.resnets.1")?);
        let enc_norm = norm(&mut l, "encoder.conv_norm_out")?;
        let enc_out = Conv::load(&mut l, "encoder.conv_out", None, None)?;
        let quant = Conv::load(&mut l, "quant_conv", None, None)?;
        let post_quant = Conv::load(&mut l, "post_quant_conv", Some(8), Some(8))?;
        let dec_in = Conv::load(&mut l, "decoder.conv_in", Some(8), None)?;
        let dec_mid = (res(&mut l, "decoder.mid_block.resnets.0")?, attn(&mut l, "decoder.mid_block.attentions.0")?, res(&mut l, "decoder.mid_block.resnets.1")?);
        let mut dec_up = Vec::new();
        for i in 0..4 {
            let p = format!("decoder.up_blocks.{i}");
            let blocks = (0..count(l.st, &p)).map(|j| res(&mut l, &format!("{p}.resnets.{j}"))).collect::<Result<Vec<_>>>()?;
            let up = if l.st.has(&format!("{p}.upsamplers.0.conv.weight")) { Some(Conv::load(&mut l, &format!("{p}.upsamplers.0.conv"), None, None)?) } else { None };
            dec_up.push((blocks, up));
        }
        let dec_norm = norm(&mut l, "decoder.conv_norm_out")?;
        let dec_out = Conv::load(&mut l, "decoder.conv_out", None, Some(8))?;
        let bn_mean = l.f32_opt("bn.running_mean")?;
        let bn_var = l.f32_opt("bn.running_var")?;
        crate::info!("  vae: {:.2} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(Vae { dev, enc_in, enc_down, enc_mid, enc_norm, enc_out, quant, post_quant, dec_in, dec_mid, dec_up, dec_norm, dec_out, bn_mean, bn_var })
    }

    // ------------------------------------------------------------------ primitives
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, inp: &Plane, c: &Conv, stride: usize, pad: usize, up2: bool, res: Option<&Plane>, out: &Plane) -> Result<()> {
        conv(&self.dev, inp, c, stride, pad, up2, res, out)
    }
    fn gn(&self, x: &Plane, n: &Norm, silu: bool) -> Result<Plane> {
        let out = Plane::new(&self.dev, x.h, x.w, x.c)?;
        ops::groupnorm(&self.dev, &x.t, x.npix(), x.c, &n.g, &n.b, GN_EPS, silu, &out.t)?;
        Ok(out)
    }

    fn resblock(&self, x: &Plane, rb: &ResBlock) -> Result<Plane> {
        let dev = &self.dev;
        let h = self.gn(x, &rb.n1, true)?;
        let c1 = Plane::new(dev, x.h, x.w, rb.c1.co)?;
        self.conv(&h, &rb.c1, 1, 1, false, None, &c1)?;
        drop(h);
        let h2 = self.gn(&c1, &rb.n2, true)?;
        drop(c1);
        let sc_plane;
        let sc = match &rb.shortcut {
            Some(s) => {
                sc_plane = Plane::new(dev, x.h, x.w, s.co)?;
                self.conv(x, s, 1, 0, false, None, &sc_plane)?;
                &sc_plane
            }
            None => x,
        };
        let out = Plane::new(dev, x.h, x.w, rb.c2.co)?;
        self.conv(&h2, &rb.c2, 1, 1, false, Some(sc), &out)?;
        Ok(out)
    }

    /// Single-head attention over all pixels (scale 1/sqrt(C)), residual added by the output projection.
    fn attnblock(&self, x: &Plane, ab: &AttnBlock) -> Result<Plane> {
        let dev = &self.dev;
        let c = x.c;
        let normed = self.gn(x, &ab.norm, false)?;
        let qkv = Plane::new(dev, x.h, x.w, 3 * c)?;
        self.conv(&normed, &ab.qkv, 1, 0, false, None, &qkv)?;
        drop(normed);
        let o = pixel_attention(dev, &qkv, c)?;
        drop(qkv);
        let out = Plane::new(dev, x.h, x.w, c)?;
        self.conv(&o, &ab.out, 1, 0, false, Some(x), &out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ encoder
    /// `rgb` f32 HWC [h, w, 3] in [0, 1] (h, w multiples of 16) -> normalized DiT latent tokens f32 [h/16 * w/16, 128]
    /// (the distribution's mode, patchified and BN-standardized like diffusers' Flux2 `_encode_vae_image`).
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize) -> Result<Tensor> {
        let dev = &self.dev;
        ensure!(h % 16 == 0 && w % 16 == 0, "vae encode: size must be a multiple of 16");
        let (bn_mean, bn_var) = (self.bn_mean.as_ref().context("not a FLUX.2 VAE")?, self.bn_var.as_ref().context("not a FLUX.2 VAE")?);
        let q = self.moments(rgb, h, w)?;
        let (lh, lw) = (h / 16, w / 16);
        let tok = Tensor::new(dev, DType::F32, &[lh * lw, 128])?;
        dev.launch_n("k_latent_pack", lh * lw * 128, &[Arg::Ptr(q.t.ptr), Arg::I32(q.c as i32), Arg::Ptr(tok.ptr), Arg::I32(lh as i32), Arg::I32(lw as i32), Arg::Ptr(bn_mean.ptr), Arg::Ptr(bn_var.ptr), Arg::F32(BN_EPS)])?;
        Ok(tok)
    }

    /// SD: `rgb` f32 HWC [h, w, 3] in [0, 1] (multiples of 8) -> the scaled latent's mean, f32 HWC [h/8, w/8, 4].
    pub fn encode_sd(&self, rgb: &[f32], h: usize, w: usize) -> Result<Vec<f32>> {
        ensure!(h % 8 == 0 && w % 8 == 0, "vae encode: size must be a multiple of 8");
        let q = self.moments(rgb, h, w)?;
        let v = q.t.to_f32_vec(&self.dev)?;
        let mut out = Vec::with_capacity(q.npix() * 4);
        for p in 0..q.npix() {
            out.extend(v[p * q.c..p * q.c + 4].iter().map(|x| x * SDXL_SCALE));
        }
        Ok(out)
    }

    /// The encoder through quant_conv: the latent distribution's moments [h/8, w/8, 2 * latent channels].
    fn moments(&self, rgb: &[f32], h: usize, w: usize) -> Result<Plane> {
        let dev = &self.dev;
        let rgb_t = Tensor::from_f32(dev, rgb, &[h * w, 3])?;
        let x0 = Plane::new(dev, h, w, 8)?;
        dev.launch_n("k_rgb_to_plane8", h * w, &[Arg::Ptr(rgb_t.ptr), Arg::Ptr(x0.t.ptr), Arg::I64((h * w) as i64)])?;
        drop(rgb_t);
        let mut x = Plane::new(dev, h, w, self.enc_in.co)?;
        self.conv(&x0, &self.enc_in, 1, 1, false, None, &x)?;
        drop(x0);
        for (blocks, down) in &self.enc_down {
            for rb in blocks {
                x = self.resblock(&x, rb)?;
            }
            if let Some(dc) = down {
                // F.pad (0, 1, 0, 1) + 3x3 stride-2 conv: no top/left padding, bottom/right read as zero
                let out = Plane::new(dev, x.h / 2, x.w / 2, dc.co)?;
                self.conv(&x, dc, 2, 0, false, None, &out)?;
                x = out;
            }
        }
        x = self.resblock(&x, &self.enc_mid.0)?;
        x = self.attnblock(&x, &self.enc_mid.1)?;
        x = self.resblock(&x, &self.enc_mid.2)?;
        let hn = self.gn(&x, &self.enc_norm, true)?;
        let moments = Plane::new(dev, x.h, x.w, self.enc_out.co)?;
        self.conv(&hn, &self.enc_out, 1, 1, false, None, &moments)?;
        drop(hn);
        let q = Plane::new(dev, x.h, x.w, self.quant.co)?;
        self.conv(&moments, &self.quant, 1, 0, false, None, &q)?;
        Ok(q)
    }

    // ------------------------------------------------------------------ decoder
    /// Normalized DiT latent tokens f32 [h*w, 128] -> RGB u8 [h*16, w*16, 3] (clamped to [-1, 1] like the pipeline).
    pub fn decode(&self, tok: &Tensor, h: usize, w: usize) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let (bn_mean, bn_var) = (self.bn_mean.as_ref().context("not a FLUX.2 VAE")?, self.bn_var.as_ref().context("not a FLUX.2 VAE")?);
        let z = Plane::new(dev, 2 * h, 2 * w, 32)?;
        dev.launch_n("k_latent_unpack", h * w * 128, &[Arg::Ptr(tok.ptr), Arg::Ptr(z.t.ptr), Arg::I32(h as i32), Arg::I32(w as i32), Arg::Ptr(bn_mean.ptr), Arg::Ptr(bn_var.ptr), Arg::F32(BN_EPS)])?;
        self.decode_plane(z)
    }

    /// SD: a latent f32 HWC [h, w, 4] (scaled, as the UNet samples it) -> RGB u8 [h*8, w*8, 3].
    pub fn decode_sd(&self, lat: &[f32], h: usize, w: usize) -> Result<Vec<u8>> {
        ensure!(lat.len() == h * w * 4, "vae decode: latent size");
        let mut bits = vec![0u16; h * w * 8];
        for p in 0..h * w {
            for c in 0..4 {
                bits[p * 8 + c] = bf16_bits(lat[p * 4 + c] / SDXL_SCALE);
            }
        }
        let z = Plane { t: Tensor::from_bf16(&self.dev, &bits, &[h * w, 8])?, h, w, c: 8 };
        self.decode_plane(z)
    }

    fn decode_plane(&self, z: Plane) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let z2 = Plane::new(dev, z.h, z.w, self.post_quant.co)?;
        self.conv(&z, &self.post_quant, 1, 0, false, None, &z2)?;
        drop(z);
        let mut x = Plane::new(dev, z2.h, z2.w, self.dec_in.co)?;
        self.conv(&z2, &self.dec_in, 1, 1, false, None, &x)?;
        drop(z2);
        x = self.resblock(&x, &self.dec_mid.0)?;
        x = self.attnblock(&x, &self.dec_mid.1)?;
        x = self.resblock(&x, &self.dec_mid.2)?;
        for (blocks, up) in &self.dec_up {
            for rb in blocks {
                x = self.resblock(&x, rb)?;
            }
            if let Some(uc) = up {
                let out = Plane::new(dev, x.h * 2, x.w * 2, uc.co)?;
                self.conv(&x, uc, 1, 1, true, None, &out)?;
                x = out;
            }
        }
        let hn = self.gn(&x, &self.dec_norm, true)?;
        drop(x);
        let o = Plane::new(dev, hn.h, hn.w, 8)?;
        self.conv(&hn, &self.dec_out, 1, 1, false, None, &o)?;
        drop(hn);
        let npix = o.npix();
        let rgb = Tensor::new(dev, DType::U8, &[npix * 3])?;
        dev.launch_n("k_vae_out_to_rgb8", npix, &[Arg::Ptr(o.t.ptr), Arg::Ptr(rgb.ptr), Arg::I64(npix as i64), Arg::I32(8)])?;
        dev.dtoh_at(rgb.ptr, npix * 3)
    }
}
