//! FLUX.2 VAE with the small decoder (`flux2-vae_small.safetensors`, diffusers AutoencoderKLFlux2 keys):
//! encoder 128-256-512-512 / decoder 384-384-192-96, 2 (encoder) / 3 (decoder) ResNet blocks per level, GroupNorm(32,
//! eps 1e-6) + SiLU, single-head mid attention, 32 latent channels at 1/8, then a 2x2 patch (128 channels at 1/16) and a
//! BatchNorm standardization (running mean/var, eps 1e-4) into the DiT's latent space.
//! Channels-last bf16 planes with f32 accumulation (the reference runs the VAE in fp32).
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Result};
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
    co: usize,
    ci: usize,
    kh: usize,
    kw: usize,
    kpad: usize,
}

impl Conv {
    /// `prefix.weight` [Co, Ci, kh, kw] (or a Linear [Co, Ci] as a 1x1 conv); `ci_pad` / `co_pad` zero-pad channels.
    fn load(l: &mut Loader, prefix: &str, ci_pad: Option<usize>, co_pad: Option<usize>) -> Result<Conv> {
        let wname = format!("{prefix}.weight");
        let info = l.st.info(&wname)?.clone();
        let w = l.st.f32s(&wname)?;
        let (co, ci, kh, kw) = match info.shape.len() {
            4 => (info.shape[0], info.shape[1], info.shape[2], info.shape[3]),
            2 => (info.shape[0], info.shape[1], 1, 1),
            _ => anyhow::bail!("{prefix}: unexpected conv weight rank"),
        };
        Self::from_parts(l, prefix, &w, (co, ci, kh, kw), ci_pad, co_pad)
    }
    fn from_parts(l: &mut Loader, prefix: &str, w: &[f32], (co, ci, kh, kw): (usize, usize, usize, usize), ci_pad: Option<usize>, co_pad: Option<usize>) -> Result<Conv> {
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
    pub bn_mean: Tensor,
    pub bn_var: Tensor,
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
        for i in 0..4 {
            let p = format!("encoder.down_blocks.{i}");
            let blocks = (0..2).map(|j| res(&mut l, &format!("{p}.resnets.{j}"))).collect::<Result<Vec<_>>>()?;
            let down = if l.st.has(&format!("{p}.downsamplers.0.conv.weight")) { Some(Conv::load(&mut l, &format!("{p}.downsamplers.0.conv"), None, None)?) } else { None };
            enc_down.push((blocks, down));
        }
        let enc_mid = (res(&mut l, "encoder.mid_block.resnets.0")?, attn(&mut l, "encoder.mid_block.attentions.0")?, res(&mut l, "encoder.mid_block.resnets.1")?);
        let enc_norm = norm(&mut l, "encoder.conv_norm_out")?;
        let enc_out = Conv::load(&mut l, "encoder.conv_out", None, None)?;
        let quant = Conv::load(&mut l, "quant_conv", None, None)?;
        let post_quant = Conv::load(&mut l, "post_quant_conv", None, None)?;
        let dec_in = Conv::load(&mut l, "decoder.conv_in", None, None)?;
        let dec_mid = (res(&mut l, "decoder.mid_block.resnets.0")?, attn(&mut l, "decoder.mid_block.attentions.0")?, res(&mut l, "decoder.mid_block.resnets.1")?);
        let mut dec_up = Vec::new();
        for i in 0..4 {
            let p = format!("decoder.up_blocks.{i}");
            let blocks = (0..3).map(|j| res(&mut l, &format!("{p}.resnets.{j}"))).collect::<Result<Vec<_>>>()?;
            let up = if l.st.has(&format!("{p}.upsamplers.0.conv.weight")) { Some(Conv::load(&mut l, &format!("{p}.upsamplers.0.conv"), None, None)?) } else { None };
            dec_up.push((blocks, up));
        }
        let dec_norm = norm(&mut l, "decoder.conv_norm_out")?;
        let dec_out = Conv::load(&mut l, "decoder.conv_out", None, Some(8))?;
        let bn_mean = l.f32("bn.running_mean")?;
        let bn_var = l.f32("bn.running_var")?;
        crate::info!("  vae: {:.2} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(Vae { dev, enc_in, enc_down, enc_mid, enc_norm, enc_out, quant, post_quant, dec_in, dec_mid, dec_up, dec_norm, dec_out, bn_mean, bn_var })
    }

    // ------------------------------------------------------------------ primitives
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, inp: &Plane, c: &Conv, stride: usize, pad: usize, up2: bool, res: Option<&Plane>, out: &Plane) -> Result<()> {
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
        let f = self.dev.func("k_conv2d_nhwc")?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: CONV_SMEM };
        let mut b = self.dev.stream.launch_builder(&f);
        b.arg(&p);
        unsafe { b.launch(cfg) }?;
        Ok(())
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
        let (n, c) = (x.npix(), x.c);
        let normed = self.gn(x, &ab.norm, false)?;
        let qkv = Plane::new(dev, x.h, x.w, 3 * c)?;
        self.conv(&normed, &ab.qkv, 1, 0, false, None, &qkv)?;
        drop(normed);
        let n_pad = (n + 7) / 8 * 8;
        let q = Tensor::new(dev, DType::BF16, &[n, c])?;
        let k = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
        let v = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
        for (i, t) in [&q, &k, &v].iter().enumerate() {
            dev.launch_n("k_slice_channels", n * c, &[Arg::Ptr(qkv.t.ptr), Arg::Ptr(t.ptr), Arg::I64(n as i64), Arg::I32((3 * c) as i32), Arg::I32((i * c) as i32), Arg::I32(c as i32)])?;
        }
        drop(qkv);
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
        let o = Plane::new(dev, x.h, x.w, c)?;
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
        let (lh, lw) = (h / 16, w / 16);
        let tok = Tensor::new(dev, DType::F32, &[lh * lw, 128])?;
        dev.launch_n("k_latent_pack", lh * lw * 128, &[Arg::Ptr(q.t.ptr), Arg::I32(q.c as i32), Arg::Ptr(tok.ptr), Arg::I32(lh as i32), Arg::I32(lw as i32), Arg::Ptr(self.bn_mean.ptr), Arg::Ptr(self.bn_var.ptr), Arg::F32(BN_EPS)])?;
        Ok(tok)
    }

    // ------------------------------------------------------------------ decoder
    /// Normalized DiT latent tokens f32 [h*w, 128] -> RGB u8 [h*16, w*16, 3] (clamped to [-1, 1] like the pipeline).
    pub fn decode(&self, tok: &Tensor, h: usize, w: usize) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let z = Plane::new(dev, 2 * h, 2 * w, 32)?;
        dev.launch_n("k_latent_unpack", h * w * 128, &[Arg::Ptr(tok.ptr), Arg::Ptr(z.t.ptr), Arg::I32(h as i32), Arg::I32(w as i32), Arg::Ptr(self.bn_mean.ptr), Arg::Ptr(self.bn_var.ptr), Arg::F32(BN_EPS)])?;
        let z2 = Plane::new(dev, z.h, z.w, 32)?;
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
