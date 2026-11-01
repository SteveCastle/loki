//! Qwen-Image VAE (the Wan 2.1 VAE: 16 latent channels at 1/8) for still images, as Krea 2 and Anima use it.
//!
//! The VAE is causal 3D, but a single frame only ever meets the zero padding in front of it: every 3x3x3 convolution
//! reduces to its last temporal slice, the time convolutions of the 3D resamplers are skipped on the first frame, and the
//! mid attention runs per frame. So the image path is a 2D network: RMS_norm (x / ||x|| * sqrt(C) * gamma) + SiLU,
//! residual blocks, `resample` = zero-pad (0,1,0,1) + 3x3 stride-2 conv (down) or nearest 2x + 3x3 conv halving the
//! channels (up), single-head attention in the middle. Weights in the original (ComfyUI) key layout:
//! `encoder.conv1`, `encoder.downsamples.N.{residual.{0,2,3,6},shortcut,resample.1}`, `encoder.middle.{0,1,2}`,
//! `encoder.head.{0,2}`, `conv1` / `conv2` (quant / post-quant), and the decoder's `upsamples`.
//! The DiTs see 2x2-patched, per-channel standardized latents (`(z - mean) / std`, Wan 2.1 statistics): tokens
//! [h/16 * w/16, 64] with channel c*4 + dy*2 + dx.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::vae::{conv, pixel_attention, Conv, Plane};
use crate::weights::Loader;
use anyhow::{ensure, Result};
use std::path::Path;
use std::sync::Arc;

pub const Z: usize = 16;
pub const TOKEN_C: usize = 4 * Z;

pub const LATENTS_MEAN: [f32; 16] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921];
pub const LATENTS_STD: [f32; 16] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160];

struct Res {
    g1: Tensor,
    c1: Conv,
    g2: Tensor,
    c2: Conv,
    shortcut: Option<Conv>,
}
struct Attn {
    g: Tensor,
    qkv: Conv,
    proj: Conv,
}
enum Layer {
    Res(Res),
    Down(Conv),
    Up(Conv),
}
struct Half {
    conv_in: Conv,
    layers: Vec<Layer>,
    mid: (Res, Attn, Res),
    head_g: Tensor,
    head: Conv,
}

pub struct QVae {
    dev: Arc<Device>,
    enc: Half,
    dec: Half,
    quant: Conv,
    post_quant: Conv,
    mean: Tensor,
    std: Tensor,
}

fn gamma(l: &mut Loader, name: &str) -> Result<Tensor> {
    let v = l.st.f32s(name)?;
    let bits: Vec<u16> = v.iter().map(|x| crate::tensor::bf16_bits(*x)).collect();
    Tensor::from_bf16(&l.dev, &bits, &[bits.len()])
}

impl QVae {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<QVae> {
        let st = SafeTensors::open(path)?;
        ensure!(st.has("decoder.upsamples.0.residual.0.gamma") && st.has("conv2.weight"), "{} is not a Qwen-Image / Wan 2.1 VAE (original key layout)", path.display());
        ensure!(st.info("conv2.weight")?.shape[0] == Z, "{}: a {}-channel VAE; Krea 2 and Anima need the 16-channel Qwen-Image VAE", path.display(), st.info("conv2.weight")?.shape[0]);
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let res = |l: &mut Loader, p: &str| -> Result<Res> {
            Ok(Res {
                g1: gamma(l, &format!("{p}.residual.0.gamma"))?,
                c1: Conv::load(l, &format!("{p}.residual.2"), None, None)?,
                g2: gamma(l, &format!("{p}.residual.3.gamma"))?,
                c2: Conv::load(l, &format!("{p}.residual.6"), None, None)?,
                shortcut: if l.st.has(&format!("{p}.shortcut.weight")) { Some(Conv::load(l, &format!("{p}.shortcut"), None, None)?) } else { None },
            })
        };
        let half = |l: &mut Loader, side: &str, list: &str, up: bool| -> Result<Half> {
            let conv_in = Conv::load(l, &format!("{side}.conv1"), if up { None } else { Some(8) }, None)?;
            let mut layers = Vec::new();
            let mut i = 0;
            loop {
                let p = format!("{side}.{list}.{i}");
                if l.st.has(&format!("{p}.residual.0.gamma")) {
                    layers.push(Layer::Res(res(l, &p)?));
                } else if l.st.has(&format!("{p}.resample.1.weight")) {
                    let c = Conv::load(l, &format!("{p}.resample.1"), None, None)?;
                    layers.push(if up { Layer::Up(c) } else { Layer::Down(c) });
                } else {
                    break;
                }
                i += 1;
            }
            let m = format!("{side}.middle");
            let mid = (
                res(l, &format!("{m}.0"))?,
                Attn { g: gamma(l, &format!("{m}.1.norm.gamma"))?, qkv: Conv::load(l, &format!("{m}.1.to_qkv"), None, None)?, proj: Conv::load(l, &format!("{m}.1.proj"), None, None)? },
                res(l, &format!("{m}.2"))?,
            );
            Ok(Half { conv_in, layers, mid, head_g: gamma(l, &format!("{side}.head.0.gamma"))?, head: Conv::load(l, &format!("{side}.head.2"), None, if up { Some(8) } else { None })? })
        };
        let enc = half(&mut l, "encoder", "downsamples", false)?;
        let dec = half(&mut l, "decoder", "upsamples", true)?;
        let quant = Conv::load(&mut l, "conv1", None, None)?;
        let post_quant = Conv::load(&mut l, "conv2", None, None)?;
        let mean = Tensor::from_f32(&dev, &LATENTS_MEAN, &[Z])?;
        let std = Tensor::from_f32(&dev, &LATENTS_STD, &[Z])?;
        crate::info!("  vae: {:.2} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(QVae { dev, enc, dec, quant, post_quant, mean, std })
    }

    fn norm(&self, x: &Plane, g: &Tensor, silu: bool) -> Result<Plane> {
        let out = Plane::new(&self.dev, x.h, x.w, x.c)?;
        self.dev.launch_n("k_vae_rmsnorm", x.npix() * 32, &[Arg::Ptr(x.t.ptr), Arg::Ptr(out.t.ptr), Arg::I64(x.npix() as i64), Arg::I32(x.c as i32), Arg::Ptr(g.ptr), Arg::I32(silu as i32)])?;
        Ok(out)
    }

    fn res(&self, x: &Plane, r: &Res) -> Result<Plane> {
        let dev = &self.dev;
        let h = self.norm(x, &r.g1, true)?;
        let c1 = Plane::new(dev, x.h, x.w, r.c1.co)?;
        conv(dev, &h, &r.c1, 1, 1, false, None, &c1)?;
        drop(h);
        let h2 = self.norm(&c1, &r.g2, true)?;
        drop(c1);
        let sc_plane;
        let sc = match &r.shortcut {
            Some(s) => {
                sc_plane = Plane::new(dev, x.h, x.w, s.co)?;
                conv(dev, x, s, 1, 0, false, None, &sc_plane)?;
                &sc_plane
            }
            None => x,
        };
        let out = Plane::new(dev, x.h, x.w, r.c2.co)?;
        conv(dev, &h2, &r.c2, 1, 1, false, Some(sc), &out)?;
        Ok(out)
    }

    fn attn(&self, x: &Plane, a: &Attn) -> Result<Plane> {
        let dev = &self.dev;
        let n = self.norm(x, &a.g, false)?;
        let qkv = Plane::new(dev, x.h, x.w, 3 * x.c)?;
        conv(dev, &n, &a.qkv, 1, 0, false, None, &qkv)?;
        drop(n);
        let o = pixel_attention(dev, &qkv, x.c)?;
        drop(qkv);
        let out = Plane::new(dev, x.h, x.w, x.c)?;
        conv(dev, &o, &a.proj, 1, 0, false, Some(x), &out)?;
        Ok(out)
    }

    fn run(&self, half: &Half, mut x: Plane) -> Result<Plane> {
        let dev = &self.dev;
        let mut y = Plane::new(dev, x.h, x.w, half.conv_in.co)?;
        conv(dev, &x, &half.conv_in, 1, 1, false, None, &y)?;
        let mut mid_done = false;
        // the decoder runs the middle first, the encoder last
        if std::ptr::eq(half, &self.dec) {
            y = self.res(&y, &half.mid.0)?;
            y = self.attn(&y, &half.mid.1)?;
            y = self.res(&y, &half.mid.2)?;
            mid_done = true;
        }
        x = y;
        for layer in &half.layers {
            x = match layer {
                Layer::Res(r) => self.res(&x, r)?,
                Layer::Down(c) => {
                    let out = Plane::new(dev, x.h / 2, x.w / 2, c.co)?;
                    conv(dev, &x, c, 2, 0, false, None, &out)?;
                    out
                }
                Layer::Up(c) => {
                    let out = Plane::new(dev, x.h * 2, x.w * 2, c.co)?;
                    conv(dev, &x, c, 1, 1, true, None, &out)?;
                    out
                }
            };
        }
        if !mid_done {
            x = self.res(&x, &half.mid.0)?;
            x = self.attn(&x, &half.mid.1)?;
            x = self.res(&x, &half.mid.2)?;
        }
        let h = self.norm(&x, &half.head_g, true)?;
        drop(x);
        let out = Plane::new(dev, h.h, h.w, half.head.co)?;
        conv(dev, &h, &half.head, 1, 1, false, None, &out)?;
        Ok(out)
    }

    /// `rgb` f32 HWC in [0, 1] (h, w multiples of 16) -> normalized tokens f32 [h/16 * w/16, 64] (the posterior mean).
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize) -> Result<Tensor> {
        let dev = &self.dev;
        ensure!(h % 16 == 0 && w % 16 == 0, "vae encode: size must be a multiple of 16");
        let rgb_t = Tensor::from_f32(dev, rgb, &[h * w, 3])?;
        let x0 = Plane::new(dev, h, w, 8)?;
        dev.launch_n("k_rgb_to_plane8", h * w, &[Arg::Ptr(rgb_t.ptr), Arg::Ptr(x0.t.ptr), Arg::I64((h * w) as i64)])?;
        drop(rgb_t);
        let moments = self.run(&self.enc, x0)?;
        let q = Plane::new(dev, moments.h, moments.w, self.quant.co)?;
        conv(dev, &moments, &self.quant, 1, 0, false, None, &q)?;
        let (lh, lw) = (h / 16, w / 16);
        let tok = Tensor::new(dev, DType::F32, &[lh * lw, TOKEN_C])?;
        dev.launch_n("k_qlat_pack", lh * lw * TOKEN_C, &[Arg::Ptr(q.t.ptr), Arg::I32(q.c as i32), Arg::Ptr(tok.ptr), Arg::I32(lh as i32), Arg::I32(lw as i32), Arg::I32(Z as i32), Arg::Ptr(self.mean.ptr), Arg::Ptr(self.std.ptr)])?;
        Ok(tok)
    }

    /// Normalized tokens f32 [h*w, 64] -> RGB u8 [h*16, w*16, 3] (clamped to [-1, 1] like the pipelines).
    pub fn decode(&self, tok: &Tensor, h: usize, w: usize) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let z = Plane::new(dev, 2 * h, 2 * w, Z)?;
        dev.launch_n("k_qlat_unpack", h * w * TOKEN_C, &[Arg::Ptr(tok.ptr), Arg::Ptr(z.t.ptr), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(Z as i32), Arg::I32(Z as i32), Arg::Ptr(self.mean.ptr), Arg::Ptr(self.std.ptr)])?;
        let z2 = Plane::new(dev, z.h, z.w, Z)?;
        conv(dev, &z, &self.post_quant, 1, 0, false, None, &z2)?;
        drop(z);
        let o = self.run(&self.dec, z2)?;
        let npix = o.npix();
        let rgb = Tensor::new(dev, DType::U8, &[npix * 3])?;
        dev.launch_n("k_vae_out_to_rgb8", npix, &[Arg::Ptr(o.t.ptr), Arg::Ptr(rgb.ptr), Arg::I64(npix as i64), Arg::I32(o.c as i32)])?;
        dev.dtoh_at(rgb.ptr, npix * 3)
    }
}
