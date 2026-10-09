//! Qwen Image 2.1 VAE (Wan 2.2 layout, temporal kernel 1, RGBA, 16x spatial, 64 latent channels),
//! implemented on channels-last bf16 planes with implicit-GEMM convolutions.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use cudarc::driver::PushKernelArg;
use std::path::Path;
use std::sync::Arc;

pub const Z: usize = 64;
const CONV_SMEM: u32 = 65536;

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
    /// Rows [y0, y0+rows) as a view (contiguous in NHWC).
    pub fn rows(&self, y0: usize, rows: usize) -> Plane {
        Plane { t: self.t.rows(y0 * self.w, rows * self.w), h: rows, w: self.w, c: self.c }
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
    /// Load `prefix.weight` of shape [Co, Ci, (1,) kh, kw] (3D convs with temporal kernel 1 or 2D) into
    /// (ky, kx, ci) order. `ci_pad`: pad input channels up to this (zeros); `co_pad`: pad output channels.
    fn load(l: &mut Loader, prefix: &str, ci_pad: Option<usize>, co_pad: Option<usize>) -> Result<Conv> {
        let wname = format!("{prefix}.weight");
        let info = l.st.info(&wname)?.clone();
        let w = l.st.f32s(&wname)?;
        let (co, ci, kh, kw) = match info.shape.len() {
            5 => {
                ensure!(info.shape[2] == 1, "{prefix}: temporal kernel must be 1");
                (info.shape[0], info.shape[1], info.shape[3], info.shape[4])
            }
            4 => (info.shape[0], info.shape[1], info.shape[2], info.shape[3]),
            _ => anyhow::bail!("{prefix}: unexpected conv weight rank"),
        };
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
                        let src = ((o * ci + i) * kh + ky) * kw + kx;
                        let dst = o * kpad + (ky * kw + kx) * ci_p + i;
                        packed[dst] = bf16_bits(w[src]);
                    }
                }
            }
        }
        let mut bias = vec![0f32; co_p];
        if l.st.has(&format!("{prefix}.bias")) {
            let b = l.st.f32s(&format!("{prefix}.bias"))?;
            bias[..co].copy_from_slice(&b);
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

struct ResBlock {
    g1: Tensor,
    c1: Conv,
    g2: Tensor,
    c2: Conv,
    shortcut: Option<Conv>,
}
struct AttnBlock {
    gamma: Tensor,
    qkv: Conv,
    proj: Conv,
}

pub struct Vae {
    dev: Arc<Device>,
    // encoder
    enc_conv1: Conv,
    enc_down: Vec<(Vec<ResBlock>, Option<Conv>)>, // per stage: res blocks, downsample conv
    enc_mid: (ResBlock, AttnBlock, ResBlock),
    enc_head_g: Tensor,
    enc_head: Conv,
    conv1: Conv, // 128 -> 64 (mu rows only)
    // decoder
    conv2: Conv,
    dec_conv1: Conv,
    dec_mid: (ResBlock, AttnBlock, ResBlock),
    dec_up: Vec<(Vec<ResBlock>, Option<Conv>)>,
    dec_head_g: Tensor,
    dec_head: Conv,
    pub latents_mean: Tensor,
    pub latents_std: Tensor,
    /// strip budget in bytes for intermediate tensors
    pub strip_bytes: usize,
}


pub const LATENTS_MEAN: [f32; 64] = [
    0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.6510, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.0210, 0.0472, 0.9320, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464,
    -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.3680, 0.4809, -0.4433, 0.3764, 0.5729,
    -2.0595, 1.0960, -1.3260, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304,
];
pub const LATENTS_STD: [f32; 64] = [
    3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.7910, 3.0785, 3.6500, 3.9308, 3.0904, 2.8778, 3.7675, 3.7320, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896,
    3.8513, 3.9323, 3.4719, 3.7498, 4.2830, 3.5694, 4.2467, 3.9037, 3.2947, 5.0770, 3.5075, 3.2700, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168,
    3.5643, 4.0550, 5.5614, 4.2963, 4.4080, 3.4959, 3.8747, 3.7608, 3.5735, 3.1490, 3.7662, 3.6746, 3.4563, 3.8161,
];

impl Vae {
    pub fn init_kernels(dev: &Device) -> Result<()> {
        dev.set_max_smem("k_conv2d_nhwc", CONV_SMEM)
    }

    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Vae> {
        let st = SafeTensors::open(path)?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let load_res = |l: &mut Loader, p: &str| -> Result<ResBlock> {
            Ok(ResBlock {
                g1: gamma(l, &format!("{p}.residual.0.gamma"))?,
                c1: Conv::load(l, &format!("{p}.residual.2"), None, None)?,
                g2: gamma(l, &format!("{p}.residual.3.gamma"))?,
                c2: Conv::load(l, &format!("{p}.residual.6"), None, None)?,
                shortcut: if l.st.has(&format!("{p}.shortcut.weight")) { Some(Conv::load(l, &format!("{p}.shortcut"), None, None)?) } else { None },
            })
        };
        let load_attn = |l: &mut Loader, p: &str| -> Result<AttnBlock> {
            Ok(AttnBlock { gamma: gamma(l, &format!("{p}.norm.gamma"))?, qkv: Conv::load(l, &format!("{p}.to_qkv"), None, None)?, proj: Conv::load(l, &format!("{p}.proj"), None, None)? })
        };
        // ---- encoder: dims [96, 96, 192, 384, 768, 768]
        let enc_conv1 = Conv::load(&mut l, "encoder.conv1", Some(8), None)?;
        let mut enc_down = Vec::new();
        for i in 0..5 {
            let p = format!("encoder.downsamples.{i}");
            let mut blocks = Vec::new();
            for j in 0..2 {
                blocks.push(load_res(&mut l, &format!("{p}.downsamples.{j}"))?);
            }
            let down = if i != 4 { Some(Conv::load(&mut l, &format!("{p}.downsamples.2.resample.1"), None, None)?) } else { None };
            enc_down.push((blocks, down));
        }
        let enc_mid = (load_res(&mut l, "encoder.middle.0")?, load_attn(&mut l, "encoder.middle.1")?, load_res(&mut l, "encoder.middle.2")?);
        let enc_head_g = gamma(&mut l, "encoder.head.0.gamma")?;
        let enc_head = Conv::load(&mut l, "encoder.head.2", None, None)?; // 768 -> 128
        // conv1: 128 -> 128 1x1; keep the first 64 output rows (mu)
        let conv1 = {
            let mut c = Conv::load(&mut l, "conv1", None, None)?;
            c.co = 64;
            c.w = c.w.rows(0, 64);
            c.b = c.b.rows(0, 64);
            c
        };
        // ---- decoder: dims [1152, 1152, 1152, 576, 288, 144]
        let conv2 = Conv::load(&mut l, "conv2", None, None)?;
        let dec_conv1 = Conv::load(&mut l, "decoder.conv1", None, None)?;
        let dec_mid = (load_res(&mut l, "decoder.middle.0")?, load_attn(&mut l, "decoder.middle.1")?, load_res(&mut l, "decoder.middle.2")?);
        let mut dec_up = Vec::new();
        for i in 0..5 {
            let p = format!("decoder.upsamples.{i}");
            let mut blocks = Vec::new();
            for j in 0..3 {
                blocks.push(load_res(&mut l, &format!("{p}.upsamples.{j}"))?);
            }
            let up = if i != 4 { Some(Conv::load(&mut l, &format!("{p}.upsamples.3.resample.1"), None, None)?) } else { None };
            dec_up.push((blocks, up));
        }
        let dec_head_g = gamma(&mut l, "decoder.head.0.gamma")?;
        let dec_head = Conv::load(&mut l, "decoder.head.2", None, Some(8))?; // 144 -> 4 (padded to 8)
        crate::info!("  vae: {:.2} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        let latents_mean = Tensor::from_f32(&dev, &LATENTS_MEAN, &[64])?;
        let latents_std = Tensor::from_f32(&dev, &LATENTS_STD, &[64])?;
        Ok(Vae { dev, enc_conv1, enc_down, enc_mid, enc_head_g, enc_head, conv1, conv2, dec_conv1, dec_mid, dec_up, dec_head_g, dec_head, latents_mean, latents_std, strip_bytes: 384 << 20 })
    }

    // ------------------------------------------------------------------ primitives
    fn conv(&self, inp: &Plane, c: &Conv, stride: usize, pad_t: usize, pad_l: usize, up2: bool, res: Option<&Plane>, out: &Plane) -> Result<()> {
        ensure!(inp.c == c.ci, "conv: input channels {} != {}", inp.c, c.ci);
        ensure!(out.c == c.co, "conv: output channels {} != {}", out.c, c.co);
        let p = ConvParamsRaw {
            inp: inp.t.ptr, ih: inp.h as i32, iw: inp.w as i32, ci: inp.c as i32, _p0: 0,
            w: c.w.ptr, bias: c.b.ptr, co: c.co as i32, kpad: c.kpad as i32,
            out: out.t.ptr, oh: out.h as i32, ow: out.w as i32,
            kh: c.kh as i32, kw: c.kw as i32, stride: stride as i32, pad_t: pad_t as i32, pad_l: pad_l as i32, up2: up2 as i32,
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
    /// 3x3 same conv (pad 1).
    fn conv3(&self, inp: &Plane, c: &Conv, res: Option<&Plane>, out: &Plane) -> Result<()> {
        self.conv(inp, c, 1, 1, 1, false, res, out)
    }
    fn rmsnorm(&self, x: &Plane, gamma: &Tensor, silu: bool, out: &Plane) -> Result<()> {
        let n = x.npix();
        self.dev.launch_n("k_vae_rmsnorm", n * 32, &[Arg::Ptr(x.t.ptr), Arg::Ptr(out.t.ptr), Arg::I64(n as i64), Arg::I32(x.c as i32), Arg::Ptr(gamma.ptr), Arg::I32(silu as i32)])
    }

    /// Residual block on a whole plane, processed in horizontal strips with a 2-row halo so that the
    /// intermediates never exist at full size.
    fn resblock(&self, x: &Plane, rb: &ResBlock) -> Result<Plane> {
        let dev = &self.dev;
        let out = Plane::new(dev, x.h, x.w, rb.c1.co)?;
        let row_bytes = x.w * x.c.max(rb.c1.co) * 2;
        let mut strip = (self.strip_bytes / row_bytes).max(8);
        if strip >= x.h {
            strip = x.h;
        }
        let halo = 2;
        let mut y0 = 0;
        while y0 < x.h {
            let y1 = (y0 + strip).min(x.h);
            let lo = y0.saturating_sub(halo);
            let hi = (y1 + halo).min(x.h);
            let xin = x.rows(lo, hi - lo);
            let h1 = Plane::new(dev, xin.h, xin.w, x.c)?;
            self.rmsnorm(&xin, &rb.g1, true, &h1)?;
            let c1 = Plane::new(dev, xin.h, xin.w, rb.c1.co)?;
            self.conv3(&h1, &rb.c1, None, &c1)?;
            drop(h1);
            let h2 = Plane::new(dev, xin.h, xin.w, rb.c1.co)?;
            self.rmsnorm(&c1, &rb.g2, true, &h2)?;
            drop(c1);
            // second conv only over the valid rows [y0, y1): compute on the strip then copy the valid part.
            // (rows within `halo` of a strip edge that is not an image edge are invalid in c1, but only
            // rows up to 1 away from them are affected in c2; valid rows are those with full halo.)
            let c2 = Plane::new(dev, xin.h, xin.w, rb.c1.co)?;
            // shortcut on the same strip
            let sc_plane;
            let sc_ref: Option<&Plane> = if let Some(sc) = &rb.shortcut {
                sc_plane = Plane::new(dev, xin.h, xin.w, rb.c1.co)?;
                self.conv(&xin, sc, 1, 0, 0, false, None, &sc_plane)?;
                Some(&sc_plane)
            } else {
                Some(&xin)
            };
            self.conv3(&h2, &rb.c2, sc_ref, &c2)?;
            // copy valid rows (y0 - lo .. y1 - lo) into out
            let src = c2.rows(y0 - lo, y1 - y0);
            let dst = out.rows(y0, y1 - y0);
            dev.dtod(dst.t.ptr, src.t.ptr, src.t.bytes())?;
            y0 = y1;
        }
        Ok(out)
    }

    /// Single-head attention block over all pixels (dim = channels).
    fn attnblock(&self, x: &Plane, ab: &AttnBlock) -> Result<Plane> {
        let dev = &self.dev;
        let (n, c) = (x.npix(), x.c);
        let normed = Plane::new(dev, x.h, x.w, c)?;
        self.rmsnorm(x, &ab.gamma, false, &normed)?;
        let qkv = Plane::new(dev, x.h, x.w, 3 * c)?;
        self.conv(&normed, &ab.qkv, 1, 0, 0, false, None, &qkv)?;
        drop(normed);
        // split q, k, v into contiguous [n_pad, c] matrices (keys/values zero-padded to a multiple of 8)
        let n_pad = (n + 7) / 8 * 8;
        let q = Tensor::new(dev, DType::BF16, &[n, c])?;
        let k = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
        let v = Tensor::zeros(dev, DType::BF16, &[n_pad, c])?;
        for (i, t) in [&q, &k, &v].iter().enumerate() {
            dev.launch_n("k_slice_channels", n * c, &[Arg::Ptr(qkv.t.ptr), Arg::Ptr(t.ptr), Arg::I64(n as i64), Arg::I32((3 * c) as i32), Arg::I32((i * c) as i32), Arg::I32(c as i32)])?;
        }
        drop(qkv);
        // V^T [c, n_pad]
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
            let qs = q.rows(q0, rows);
            let sv = s.rows(0, rows);
            let pv = p.rows(0, rows);
            crate::ops::gemm_bf16(dev, &qs, &k, None, crate::ops::Act::None, crate::ops::Epi::Store, &sv)?;
            dev.launch("k_softmax_rows", (rows as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(sv.ptr), Arg::Ptr(pv.ptr), Arg::I32(n as i32), Arg::I32(n_pad as i32), Arg::F32(scale)])?;
            let ov = o.t.rows(q0, rows);
            crate::ops::gemm_bf16(dev, &pv, &vt, None, crate::ops::Act::None, crate::ops::Epi::Store, &ov)?;
            q0 += rows;
        }
        drop(s);
        drop(p);
        let out = Plane::new(dev, x.h, x.w, c)?;
        self.conv(&o, &ab.proj, 1, 0, 0, false, Some(x), &out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ encoder
    /// `rgb`: f32 HWC [h, w, 3] in [0,1], h and w multiples of 16. Returns the latent as token-major f32 [h/16 * w/16, 64]
    /// (raw VAE output, not normalized).
    pub fn encode(&self, rgb: &[f32], h: usize, w: usize) -> Result<Tensor> {
        let dev = &self.dev;
        ensure!(h % 16 == 0 && w % 16 == 0, "vae encode: size must be a multiple of 16");
        let rgb_t = Tensor::from_f32(dev, rgb, &[h * w, 3])?;
        let x0 = Plane::new(dev, h, w, 8)?;
        dev.launch_n("k_rgb_to_vae_in", h * w, &[Arg::Ptr(rgb_t.ptr), Arg::Ptr(x0.t.ptr), Arg::I64((h * w) as i64)])?;
        drop(rgb_t);
        let mut x = Plane::new(dev, h, w, self.enc_conv1.co)?;
        self.conv3(&x0, &self.enc_conv1, None, &x)?;
        drop(x0);
        for (i, (blocks, down)) in self.enc_down.iter().enumerate() {
            let x_in = x.clone();
            for rb in blocks {
                x = self.resblock(&x, rb)?;
            }
            if let Some(dc) = down {
                // ZeroPad2d((0,1,0,1)) + conv3x3 stride 2 -> pad_t = pad_l = 0, bottom/right padding implicit
                let out = Plane::new(dev, x.h / 2, x.w / 2, dc.co)?;
                self.conv(&x, dc, 2, 0, 0, false, None, &out)?;
                x = out;
            }
            // AvgDown3D shortcut
            let mode = if down.is_none() { 2 } else if i == 0 { 0 } else { 1 };
            let n = x.npix() * x.c;
            dev.launch_n("k_avgdown_add", n, &[Arg::Ptr(x.t.ptr), Arg::Ptr(x_in.t.ptr), Arg::I32(x.h as i32), Arg::I32(x.w as i32), Arg::I32(x.c as i32), Arg::I32(x_in.c as i32), Arg::I32(mode)])?;
        }
        x = self.resblock(&x, &self.enc_mid.0)?;
        x = self.attnblock(&x, &self.enc_mid.1)?;
        x = self.resblock(&x, &self.enc_mid.2)?;
        let hn = Plane::new(dev, x.h, x.w, x.c)?;
        self.rmsnorm(&x, &self.enc_head_g, true, &hn)?;
        let moments = Plane::new(dev, x.h, x.w, self.enc_head.co)?;
        self.conv3(&hn, &self.enc_head, None, &moments)?;
        let mu = Plane::new(dev, x.h, x.w, 64)?;
        self.conv(&moments, &self.conv1, 1, 0, 0, false, None, &mu)?;
        let out = Tensor::new(dev, DType::F32, &[mu.npix(), 64])?;
        crate::ops::to_f32(dev, &mu.t, &out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ decoder
    /// `z`: token-major bf16 [h*w, 64] (raw latent, already de-normalized). Returns RGB u8 [h*16, w*16, 3].
    pub fn decode(&self, z: &Tensor, h: usize, w: usize) -> Result<Vec<u8>> {
        let dev = &self.dev;
        let zp = Plane { t: z.clone(), h, w, c: 64 };
        let z2 = Plane::new(dev, h, w, 64)?;
        self.conv(&zp, &self.conv2, 1, 0, 0, false, None, &z2)?;
        let mut x = Plane::new(dev, h, w, self.dec_conv1.co)?;
        self.conv3(&z2, &self.dec_conv1, None, &x)?;
        drop(z2);
        x = self.resblock(&x, &self.dec_mid.0)?;
        x = self.attnblock(&x, &self.dec_mid.1)?;
        x = self.resblock(&x, &self.dec_mid.2)?;
        for (i, (blocks, up)) in self.dec_up.iter().enumerate() {
            let x_in = x.clone();
            for rb in blocks {
                x = self.resblock(&x, rb)?;
            }
            if let Some(uc) = up {
                // nearest 2x upsample fused into a 3x3 conv; processed in strips to bound memory
                let out = Plane::new(dev, x.h * 2, x.w * 2, uc.co)?;
                self.upsample_conv(&x, uc, &out)?;
                x = out;
                // DupUp3D shortcut: mode by stage (see Wan2.2 DupUp3D with first_chunk)
                let mode = match i { 0 | 1 => 0, 2 => 1, _ => 2 };
                let n = x.npix() * x.c;
                dev.launch_n("k_dupup_add", n, &[Arg::Ptr(x.t.ptr), Arg::Ptr(x_in.t.ptr), Arg::I32(x_in.h as i32), Arg::I32(x_in.w as i32), Arg::I32(x_in.c as i32), Arg::I32(x.c as i32), Arg::I32(mode)])?;
            }
            drop(x_in);
        }
        // head: norm + silu + conv -> 8 (RGBA + pad), in strips
        let out_px = x.npix();
        let rgb = Tensor::new(dev, DType::U8, &[out_px * 3])?;
        {
            let row_bytes = x.w * x.c * 2;
            let strip = (self.strip_bytes / row_bytes).max(8).min(x.h);
            let mut y0 = 0;
            while y0 < x.h {
                let y1 = (y0 + strip).min(x.h);
                let lo = y0.saturating_sub(1);
                let hi = (y1 + 1).min(x.h);
                let xin = x.rows(lo, hi - lo);
                let hn = Plane::new(dev, xin.h, xin.w, xin.c)?;
                self.rmsnorm(&xin, &self.dec_head_g, true, &hn)?;
                let o = Plane::new(dev, xin.h, xin.w, 8)?;
                self.conv3(&hn, &self.dec_head, None, &o)?;
                let valid = o.rows(y0 - lo, y1 - y0);
                let npix = valid.npix();
                dev.launch_n("k_vae_out_to_rgb8", npix, &[Arg::Ptr(valid.t.ptr), Arg::Ptr(rgb.ptr + (y0 * x.w * 3) as u64), Arg::I64(npix as i64), Arg::I32(8)])?;
                y0 = y1;
            }
        }
        let bytes: Vec<u8> = dev.dtoh_at(rgb.ptr, out_px * 3)?;
        Ok(bytes)
    }

    fn upsample_conv(&self, x: &Plane, uc: &Conv, out: &Plane) -> Result<()> {
        let dev = &self.dev;
        let row_bytes = out.w * out.c * 2;
        let strip_out = ((self.strip_bytes / row_bytes).max(8) / 2 * 2).min(out.h);
        let mut y0 = 0;
        while y0 < out.h {
            let y1 = (y0 + strip_out).min(out.h);
            // input rows needed: (y0-1)/2 .. (y1+1)/2 -> use halo of 1 input row on each side
            let in_lo = (y0 / 2).saturating_sub(1);
            let in_hi = ((y1 + 1) / 2 + 1).min(x.h);
            let xin = x.rows(in_lo, in_hi - in_lo);
            let o = Plane::new(dev, xin.h * 2, out.w, out.c)?;
            self.conv(&xin, uc, 1, 1, 1, true, None, &o)?;
            let valid = o.rows(y0 - in_lo * 2, y1 - y0);
            let dst = out.rows(y0, y1 - y0);
            dev.dtod(dst.t.ptr, valid.t.ptr, valid.t.bytes())?;
            y0 = y1;
        }
        Ok(())
    }
}

fn gamma(l: &mut Loader, name: &str) -> Result<Tensor> {
    let v = l.st.f32s(name).with_context(|| name.to_string())?;
    let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x)).collect();
    Tensor::from_bf16(&l.dev, &bits, &[bits.len()])
}
