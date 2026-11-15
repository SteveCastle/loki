//! MiniMax H3 latent upscaler: LBH-123-AI's `LatentResizer3D` (Apache-2.0 weights, MIT ComfyUI node
//! `MinimaxH3LatentUpscaler3D`), the learned 24-channel latent resize of the two-pass workflow. It replaces the
//! decode -> pixel upscale -> encode round trip (and the ghosting of a plain bilinear latent resize).
//!
//! conv_in -> in_blocks -> trilinear resize (T unchanged) -> out_blocks -> GroupNorm+SiLU -> conv_out. The node
//! normalizes its input LATENT with the video VAE's `latents_mean/std` and de-normalizes the output, but a ComfyUI H3
//! LATENT is already normalized (the VAE wrapper does it; it is the DiT's latent space, what `VideoVae::encode`
//! returns), so the network sees a doubly normalized latent, and so must we: `(z - mean) / std` in, `* std + mean` out. Blocks: `ResBlockEmb3D` (GN+SiLU, conv 3x3x3, GN modulated by the scale embedding, SiLU,
//! conv 3x3x3, identity skip) and `TemporalConv` (GN+SiLU, depthwise (5,1,1) conv, pointwise conv, residual).
//!
//! Two exact rewrites keep it on the video VAE's kernels: the scale embedding is constant over the volume, so each
//! block's modulation `gn(h)*(1+s)+b` folds into that GroupNorm's affine; and a temporal block's depthwise + pointwise
//! convs fold into one dense (5,1,1) conv. Long latents run in the node's 32-frame temporal chunks (replicate padding,
//! 5-frame overlap, linear blend), so the result matches ComfyUI's `enable_temporal_chunking` (the workflow default).
//! fp16 activations, f32 accumulation, like the node at `precision = fp16`.
//!
//! Host data is only ever uploaded while the stream is idle (at load, and at the start of `upscale` before any launch):
//! `Device::htod` is a synchronous copy while allocations are stream-ordered, so uploading into a buffer reused from one
//! that queued kernels still read would race them.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Context, Result};
use half::f16;
use std::path::Path;
use std::sync::Arc;

const CONV_SMEM: u32 = 4 * (256 + 128) * 32 * 2;
const GN_EPS: f32 = 1e-5;
const GROUPS: usize = 32;
/// ComfyUI node: temporal chunks of 32 latent frames, overlap = the temporal kernel size
const CHUNK: usize = 32;
/// The node's LATENTS_MEAN / LATENTS_STD (the video VAE's latent statistics)
const LAT_MEAN: [f32; 24] = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
    -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607,
    -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];
const LAT_STD: [f32; 24] = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665,
    0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596,
    3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
];

/// A conv packed for k_vv_conv3d: fp16 [co][ldw], taps ordered (kt, kh, kw, ci).
struct Conv {
    w: Tensor,
    b: Tensor,
    co: usize,
    ci: usize,
    kt: usize,
    kh: usize,
    kw: usize,
    ldw: usize,
}

struct Gn {
    g: Vec<f32>,
    b: Vec<f32>,
    /// the same affine on the device (unmodulated)
    gd: Tensor,
    bd: Tensor,
}

struct ResBlock {
    gn1: Gn,
    conv1: Conv,
    /// emb_layers Linear(64, 2C): [2C][64] + [2C]
    emb_w: Vec<f32>,
    emb_b: Vec<f32>,
    gn2: Gn,
    conv2: Conv,
}

struct Temporal {
    gn: Gn,
    conv: Conv,
}

enum Block {
    Res(ResBlock),
    Temporal(Temporal),
}

pub struct Upscaler {
    dev: Arc<Device>,
    channels: usize,
    conv_in: Conv,
    in_blocks: Vec<Block>,
    out_blocks: Vec<Block>,
    norm_out: Gn,
    conv_out: Conv,
    /// embed: Linear(1, 64) -> SiLU -> Linear(64, 64)
    e1_w: Vec<f32>,
    e1_b: Vec<f32>,
    e2_w: Vec<f32>,
    e2_b: Vec<f32>,
    temporal_kernel: usize,
    mean: Tensor,
    stdv: Tensor,
}

fn pack_conv(dev: &Device, w: &[f32], shape: &[usize], bias: Vec<f32>) -> Result<Conv> {
    ensure!(shape.len() == 5, "conv weight rank {}", shape.len());
    let (co, ci, kt, kh, kw) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    ensure!(ci % 8 == 0, "conv ci {ci} not a multiple of 8");
    let taps = kt * kh * kw;
    let ldw = (taps * ci + 31) / 32 * 32;
    let mut out = vec![0u16; co * ldw];
    for o in 0..co {
        for c in 0..ci {
            for tap in 0..taps {
                out[o * ldw + tap * ci + c] = f16::from_f32(w[(o * ci + c) * taps + tap]).to_bits();
            }
        }
    }
    let t = Tensor::new(dev, DType::BF16, &[co, ldw])?; // fp16 storage
    dev.htod(&t.buf, &out)?;
    Ok(Conv { w: t, b: Tensor::from_f32(dev, &bias, &[co])?, co, ci, kt, kh, kw, ldw })
}

impl Upscaler {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Upscaler> {
        let st = SafeTensors::open(path)?;
        // checkpoints saved with a wrapper carry an `upscaler.` prefix (the node strips it)
        let pre = if st.has("upscaler.conv_in.weight") { "upscaler." } else { "" };
        let f = |n: &str| st.f32s(&format!("{pre}{n}")).with_context(|| format!("upscaler tensor {pre}{n}"));
        let shape = |n: &str| -> Result<Vec<usize>> { Ok(st.info(&format!("{pre}{n}"))?.shape.clone()) };
        for k in ["k_vv_conv3d", "k_vv_conv3d_s"] {
            dev.set_max_smem(k, CONV_SMEM)?;
        }
        let conv = |n: &str| -> Result<Conv> { pack_conv(&dev, &f(&format!("{n}.weight"))?, &shape(&format!("{n}.weight"))?, f(&format!("{n}.bias"))?) };
        let gn = |n: &str| -> Result<Gn> {
            let (g, b) = (f(&format!("{n}.weight"))?, f(&format!("{n}.bias"))?);
            Ok(Gn { gd: Tensor::from_f32(&dev, &g, &[g.len()])?, bd: Tensor::from_f32(&dev, &b, &[b.len()])?, g, b })
        };
        let conv_in = conv("conv_in")?;
        let channels = conv_in.co;
        ensure!(channels % GROUPS == 0 && channels % 8 == 0, "upscaler width {channels}");
        ensure!(conv_in.ci == 24, "upscaler expects the 24-channel H3 latent, conv_in takes {}", conv_in.ci);
        let mut temporal_kernel = 0;
        let mut blocks = |side: &str| -> Result<Vec<Block>> {
            let mut out = Vec::new();
            for i in 0.. {
                let p = format!("{side}.{i}");
                if st.has(&format!("{pre}{p}.in_layers.0.weight")) {
                    // ResBlockEmb3D: in_layers = [GN, SiLU, conv], emb_layers = [SiLU, Linear], out_norm, out_layers = [SiLU, Dropout, conv]
                    ensure!(!st.has(&format!("{pre}{p}.skip.weight")), "{p}: projected skip connections are not supported");
                    out.push(Block::Res(ResBlock {
                        gn1: gn(&format!("{p}.in_layers.0"))?,
                        conv1: conv(&format!("{p}.in_layers.2"))?,
                        emb_w: f(&format!("{p}.emb_layers.1.weight"))?,
                        emb_b: f(&format!("{p}.emb_layers.1.bias"))?,
                        gn2: gn(&format!("{p}.out_norm"))?,
                        conv2: conv(&format!("{p}.out_layers.2"))?,
                    }));
                } else if st.has(&format!("{pre}{p}.dwconv.weight")) {
                    // TemporalConv: pw(dw(x)) folded into one (k,1,1) conv: W[o, c, t] = Wpw[o, c] * Wdw[c, t]
                    let dws = shape(&format!("{p}.dwconv.weight"))?; // [C, 1, k, 1, 1]
                    let (c, k) = (dws[0], dws[2]);
                    temporal_kernel = k;
                    let dw = f(&format!("{p}.dwconv.weight"))?;
                    let dwb = f(&format!("{p}.dwconv.bias"))?;
                    let pw = f(&format!("{p}.pwconv.weight"))?; // [C, C, 1, 1, 1]
                    let pwb = f(&format!("{p}.pwconv.bias"))?;
                    let mut w = vec![0f32; c * c * k];
                    let mut b = pwb.clone();
                    for o in 0..c {
                        for ci in 0..c {
                            let p = pw[o * c + ci];
                            for t in 0..k {
                                w[(o * c + ci) * k + t] = p * dw[ci * k + t];
                            }
                            b[o] += p * dwb[ci];
                        }
                    }
                    out.push(Block::Temporal(Temporal { gn: gn(&format!("{p}.norm"))?, conv: pack_conv(&dev, &w, &[c, c, k, 1, 1], b)? }));
                } else if st.has(&format!("{pre}{p}.q.weight")) {
                    anyhow::bail!("{p}: attention blocks are not supported (the node forces them off too)");
                } else {
                    break;
                }
            }
            Ok(out)
        };
        let in_blocks = blocks("in_blocks")?;
        let out_blocks = blocks("out_blocks")?;
        ensure!(!in_blocks.is_empty() && !out_blocks.is_empty(), "{}: not a MiniMax H3 latent upscaler (no in/out blocks)", path.display());
        Ok(Upscaler {
            channels,
            norm_out: gn("norm_out")?,
            conv_out: conv("conv_out")?,
            e1_w: f("embed.0.weight")?,
            e1_b: f("embed.0.bias")?,
            e2_w: f("embed.2.weight")?,
            e2_b: f("embed.2.bias")?,
            conv_in,
            in_blocks,
            out_blocks,
            temporal_kernel,
            mean: Tensor::from_f32(&dev, &LAT_MEAN, &[24])?,
            stdv: Tensor::from_f32(&dev, &LAT_STD, &[24])?,
            dev,
        })
    }

    /// Upscale a normalized latent z f32 [24, T, H, W] to [24, T, ho, wo]. `scale` is the node's effective scale (mean of
    /// the two axis ratios), which conditions the network.
    pub fn upscale(&self, z: &Tensor, ho: usize, wo: usize) -> Result<Tensor> {
        let dev = &self.dev;
        ensure!(z.shape.len() == 4 && z.shape[0] == 24, "upscale: latent shape {:?}", z.shape);
        let (t, h, w) = (z.shape[1], z.shape[2], z.shape[3]);
        ensure!(ho >= h && wo >= w, "the latent upscaler only enlarges ({w}x{h} -> {wo}x{ho})");
        let scale = (wo as f64 / w as f64 + ho as f64 / h as f64) / 2.0;
        dev.sync()?;
        let mods = self.modulations(scale as f32)?;
        let out = Tensor::zeros(dev, DType::F32, &[24, t, ho, wo])?;
        if (ho, wo) == (h, w) {
            dev.dtod(out.ptr, z.ptr, z.bytes())?;
            return Ok(out);
        }
        let overlap = self.temporal_kernel;
        if t <= CHUNK {
            let ones = Tensor::from_f32(dev, &vec![1.0f32; t], &[t])?;
            let seg = self.forward_seg(z, 0, t, ho, wo, &mods)?;
            self.blend(&seg, 0, t, &ones, 0, t, ho, wo, &out)?;
            dev.sync()?;
            return Ok(out);
        }
        // ComfyUI LatentResizer3D.forward: replicate-pad by `overlap` frames at both ends, run 32-frame segments with
        // `overlap` frames of context on each side, keep `overlap` frames past each segment edge, blend linearly.
        let mut wsum = vec![0f32; t];
        let mut start = 0;
        while start < t {
            let seg_start = start;
            let seg_end = (start + CHUNK).min(t);
            let out_start = seg_start.saturating_sub(overlap);
            let out_end = (seg_end + overlap).min(t);
            // window [lo, hi) of x_padded (index p <-> frame p - overlap), exactly as the node takes it: two overlaps of
            // context before out_start, none after out_end
            let lo = out_start.saturating_sub(overlap);
            let hi = (t + 2 * overlap).min(out_end + overlap);
            let seg = self.forward_seg(z, lo as isize - overlap as isize, hi - lo, ho, wo, &mods)?;
            let s0 = out_start + overlap - lo;
            let n_valid = out_end - out_start;
            let mut wv = vec![1f32; n_valid];
            if seg_start > out_start {
                let bl = seg_start - out_start;
                for k in 0..bl {
                    wv[k] = (k + 1) as f32 / (bl + 1) as f32;
                }
            }
            if out_end > seg_end {
                let bl = out_end - seg_end;
                for k in 0..bl {
                    wv[n_valid - bl + k] = (bl - k) as f32 / (bl + 1) as f32;
                }
            }
            for k in 0..n_valid {
                wsum[out_start + k] += wv[k];
            }
            // (stream idle before the upload, see the module doc)
            dev.sync()?;
            let wd = Tensor::from_f32(dev, &wv, &[n_valid])?;
            self.blend(&seg, s0, n_valid, &wd, out_start, t, ho, wo, &out)?;
            dev.sync()?;
            start += CHUNK;
        }
        let ws = Tensor::from_f32(dev, &wsum, &[t])?;
        dev.launch_n("k_lu_div_t", out.numel(), &[Arg::Ptr(out.ptr), Arg::I32(24), Arg::I32(t as i32), Arg::I32((ho * wo) as i32), Arg::Ptr(ws.ptr)])?;
        dev.sync()?;
        Ok(out)
    }

    /// Every ResBlock's second GroupNorm affine with the scale embedding folded in (on the device):
    /// gn2(h) * (1 + s) + b  ==  normalized * (g * (1 + s)) + (beta * (1 + s) + b).
    fn modulations(&self, scale: f32) -> Result<Vec<(Tensor, Tensor)>> {
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let ed = self.e1_b.len();
        let s = scale - 1.0;
        let h: Vec<f32> = (0..ed).map(|i| silu(self.e1_w[i] * s + self.e1_b[i])).collect();
        let emb: Vec<f32> = (0..ed).map(|o| self.e2_b[o] + (0..ed).map(|i| self.e2_w[o * ed + i] * h[i]).sum::<f32>()).collect();
        let se: Vec<f32> = emb.iter().map(|&v| silu(v)).collect();
        let c = self.channels;
        let mut out = Vec::new();
        for b in self.in_blocks.iter().chain(self.out_blocks.iter()) {
            if let Block::Res(r) = b {
                let e: Vec<f32> = (0..2 * c).map(|o| r.emb_b[o] + (0..ed).map(|i| r.emb_w[o * ed + i] * se[i]).sum::<f32>()).collect();
                let (sc, sh) = e.split_at(c);
                let g: Vec<f32> = r.gn2.g.iter().zip(sc).map(|(g, s)| g * (1.0 + s)).collect();
                let bb: Vec<f32> = r.gn2.b.iter().zip(sc).zip(sh).map(|((b, s), sh)| b * (1.0 + s) + sh).collect();
                out.push((Tensor::from_f32(&self.dev, &g, &[c])?, Tensor::from_f32(&self.dev, &bb, &[c])?));
            }
        }
        Ok(out)
    }

    /// `_forward_seg` on frames [t0, t0+n) of z (replicate outside), output fp16 channels-last [n, ho, wo, 24].
    fn forward_seg(&self, z: &Tensor, t0: isize, n: usize, ho: usize, wo: usize, mods: &[(Tensor, Tensor)]) -> Result<Tensor> {
        let dev = &self.dev;
        let (t, h, w) = (z.shape[1], z.shape[2], z.shape[3]);
        let xin = Tensor::new(dev, DType::BF16, &[n * h * w, 24])?;
        dev.launch_n(
            "k_lu_to_cl",
            xin.numel(),
            &[Arg::Ptr(z.ptr), Arg::I32(24), Arg::I32(t as i32), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(t0 as i32), Arg::I32(n as i32), Arg::Ptr(self.mean.ptr), Arg::Ptr(self.stdv.ptr), Arg::Ptr(xin.ptr)],
        )?;
        let mut x = self.conv_block(&xin, n, h, w, 24, None, &self.conv_in, 1, None)?;
        drop(xin);
        let mut mi = 0;
        x = self.run_blocks(x, n, h, w, &self.in_blocks, mods, &mut mi)?;
        let c = self.channels;
        let up = Tensor::new(dev, DType::BF16, &[n * ho * wo, c])?;
        dev.launch_n(
            "k_lu_resize",
            n * ho * wo * c / 8,
            &[Arg::Ptr(x.ptr), Arg::I32(n as i32), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(c as i32), Arg::I32(ho as i32), Arg::I32(wo as i32), Arg::Ptr(up.ptr)],
        )?;
        drop(x);
        let x = self.run_blocks(up, n, ho, wo, &self.out_blocks, mods, &mut mi)?;
        self.conv_block(&x, n, ho, wo, c, Some((&self.norm_out.gd, &self.norm_out.bd)), &self.conv_out, 1, None)
    }

    fn run_blocks(&self, mut x: Tensor, n: usize, h: usize, w: usize, blocks: &[Block], mods: &[(Tensor, Tensor)], mi: &mut usize) -> Result<Tensor> {
        let c = self.channels;
        for b in blocks {
            match b {
                Block::Res(r) => {
                    let hmid = self.conv_block(&x, n, h, w, c, Some((&r.gn1.gd, &r.gn1.bd)), &r.conv1, 1, None)?;
                    let (g, bb) = &mods[*mi];
                    *mi += 1;
                    x = self.conv_block(&hmid, n, h, w, c, Some((g, bb)), &r.conv2, 1, Some(&x))?;
                }
                Block::Temporal(tb) => {
                    let pt = tb.conv.kt / 2;
                    x = self.conv_block(&x, n, h, w, c, Some((&tb.gn.gd, &tb.gn.bd)), &tb.conv, pt, Some(&x))?;
                }
            }
        }
        Ok(x)
    }

    /// [GroupNorm + SiLU] -> zero pad (`pt` frames in time; the conv's spatial half-kernel in space) -> conv (+ residual).
    /// x fp16 [n, h, w, ci] -> [n, h, w, co].
    #[allow(clippy::too_many_arguments)]
    fn conv_block(&self, x: &Tensor, n: usize, h: usize, w: usize, ci: usize, gn: Option<(&Tensor, &Tensor)>, cv: &Conv, pt: usize, res: Option<&Tensor>) -> Result<Tensor> {
        let dev = &self.dev;
        let ps = cv.kh / 2;
        ensure!(cv.ci == ci, "conv expects {} channels, got {ci}", cv.ci);
        let mr = match gn {
            Some(_) => {
                let p = n * h * w;
                let stats = Tensor::zeros(dev, DType::F32, &[GROUPS * 2 * 2])?; // doubles
                let ppb = 2048usize;
                dev.launch("k_vv_gn_stats", (((p + ppb - 1) / ppb) as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x.ptr), Arg::I32(p as i32), Arg::I32(ci as i32), Arg::I32(ppb as i32), Arg::Ptr(stats.ptr)])?;
                let mr = Tensor::new(dev, DType::F32, &[GROUPS * 2])?;
                let f = dev.func("k_vv_gn_finalize")?;
                let cfg = cudarc::driver::LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
                let mut b = dev.stream.launch_builder(&f);
                use cudarc::driver::PushKernelArg;
                let (sp, mp, ni, cnt, eps) = (stats.ptr, mr.ptr, GROUPS as i32, (p * (ci / GROUPS)) as f64, GN_EPS);
                b.arg(&sp).arg(&mp).arg(&ni).arg(&cnt).arg(&eps);
                unsafe { b.launch(cfg) }?;
                Some(mr)
            }
            None => None,
        };
        let (gd, bd) = (gn.map(|g| g.0), gn.map(|g| g.1));
        let (tp, hp, wp) = (n + 2 * pt, h + 2 * ps, w + 2 * ps);
        let padded = Tensor::new(dev, DType::BF16, &[tp * hp * wp, ci])?;
        dev.launch_n(
            "k_lu_gn_pad",
            tp * hp * wp * ci / 8,
            &[
                Arg::Ptr(x.ptr), Arg::I32(n as i32), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(ci as i32),
                Arg::Ptr(mr.as_ref().map_or(0, |t| t.ptr)), Arg::Ptr(gd.map_or(0, |t| t.ptr)), Arg::Ptr(bd.map_or(0, |t| t.ptr)),
                Arg::I32(pt as i32), Arg::I32(ps as i32), Arg::Ptr(padded.ptr),
            ],
        )?;
        let (to, ho, wo) = (tp - cv.kt + 1, hp - cv.kh + 1, wp - cv.kw + 1);
        ensure!((to, ho, wo) == (n, h, w), "conv padding bookkeeping");
        // the residual is read and written element by element by the same thread, so the output may alias it
        let out = match res {
            Some(r) => r.clone(),
            None => Tensor::new(dev, DType::BF16, &[n * h * w, cv.co])?,
        };
        let m = n * h * w;
        let (bm, threads, smem, suf) = crate::vae_video::tile_cfg(m, cv.co, dev.sm_count as usize);
        let grid = (((m + bm - 1) / bm) * ((cv.co + 127) / 128)) as u32;
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
            inp: padded.ptr, tp: tp as i32, hp: hp as i32, wp: wp as i32, ci: ci as i32,
            w: cv.w.ptr, ldw: cv.ldw as i32, k: (cv.kt * cv.kh * cv.kw * ci) as i32,
            bias: cv.b.ptr, co: cv.co as i32, _p0: 0,
            out: out.ptr, to: n as i32, ho: h as i32, wo: w as i32,
            kt: cv.kt as i32, kh: cv.kh as i32, kw: cv.kw as i32, st: 1, sh: 1, sw: 1, _p1: 0,
            res: res.map_or(0, |r| r.ptr),
        };
        let f = dev.func(&format!("k_vv_conv3d{suf}"))?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem };
        let mut b = dev.stream.launch_builder(&f);
        use cudarc::driver::PushKernelArg;
        b.arg(&p);
        unsafe { b.launch(cfg) }.context("k_vv_conv3d (upscaler)")?;
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn blend(&self, seg: &Tensor, s0: usize, n: usize, w: &Tensor, t0: usize, t: usize, ho: usize, wo: usize, acc: &Tensor) -> Result<()> {
        self.dev.launch_n(
            "k_lu_blend",
            n * ho * wo * 24,
            &[
                Arg::Ptr(seg.ptr), Arg::I32(24), Arg::I32(ho as i32), Arg::I32(wo as i32), Arg::I32(s0 as i32), Arg::I32(n as i32), Arg::Ptr(w.ptr),
                Arg::I32(t0 as i32), Arg::I32(t as i32), Arg::Ptr(self.mean.ptr), Arg::Ptr(self.stdv.ptr), Arg::Ptr(acc.ptr),
            ],
        )
    }
}

/// The node's "megapixels" target: the latent's aspect at `mp` * 1024^2 pixels, each axis rounded to `align` px, then
/// to whole latent cells. Returns the latent (w, h).
pub fn megapixel_target(lw: usize, lh: usize, mp: f64, align: usize) -> (usize, usize) {
    let ar = lw as f64 / lh as f64;
    let hp = (mp * 1024.0 * 1024.0 / ar).sqrt();
    let wp = hp * ar;
    let a = align.max(1) as f64;
    let (wa, ha) = ((wp / a).round() * a, (hp / a).round() * a);
    let (wf, hf) = ((wa / 16.0).round(), (ha / 16.0).round());
    ((wf as usize).max(1), (hf as usize).max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn megapixel_targets_match_the_workflow_table() {
        // 640x480 first pass -> 1 MP: 1184x896 (the workflow's size table)
        assert_eq!(megapixel_target(40, 30, 1.0, 32), (74, 56));
        // 736x416 (16:9 0.3 MP) -> 1376x768
        let (w, h) = megapixel_target(46, 26, 1.0, 32);
        assert_eq!((w * 16, h * 16), (1376, 768));
    }
}
