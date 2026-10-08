//! MiniMax H3 audio VAE (comfy/ldm/minimax/audio_vae.py `MiniMaxH3AudioVAE`), fp32 throughout.
//!
//! Encoder: DAC conv stack (Snake1d, residual units, strided downsampling 2,4,4,5,5 => 800 samples / latent frame),
//! AttnProjection posterior head (LayerNorms, causal attention 8x256 with head-mean + adaptive pool to 32, GeGLU MLP),
//! mean_proj, latent normalization. Decoder: dec_in_proj, BigVGAN (conv_pre, 7 ConvTranspose upsamples 5,5,2,2,2,2,2,
//! 3 AMPBlock1 per stage with anti-aliased SnakeBeta, activation_post, conv_post, clamp).
//! Stereo channels are independent mono signals and are batched (B = 2).
//!
//! All convolutions run through one implicit-GEMM fp32 kernel family (`avae_conv_*`) with fused epilogues
//! (bias, residual add, resblock-sum accumulation / 3, and Snake1d of the output for the next consumer), so the
//! encoder has no standalone elementwise kernels at all. The decoder's Activation1d (x2 upsample -> SnakeBeta -> x2
//! downsample) is one kernel per activation; the 2x signal only exists in shared memory. activation_post, conv_post
//! and the clamp are fused into the final kernel.
//! Activations are kept whole-sequence: even 15 s stereo needs only ~150 MB of decoder scratch and ~0.75 GB of
//! encoder scratch (long encoder inputs are processed one stereo channel at a time to halve that).
use crate::cuda::{Arg, Device, Profiler};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Result};
use std::path::Path;
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 32000;
pub const HOP: usize = 800;
const ATTN_SMEM: u32 = ((16 * 256 + 2 * 32 * 257) * 4) as u32;

/// One convolution in GEMM layout: weight [nph][kt*ci][co] (K index = tap * ci + channel; co contiguous, co % 4 == 0).
struct Conv {
    w: Tensor,
    b: Option<Tensor>,
    ci: usize,
    co: usize,
    kt: usize,
    dil: i32,
    pad: i32,
    stride: i32,
    /// transposed conv: number of phases (= stride), else 1
    nph: usize,
}

/// Snake parameters prepared for the kernels: alpha (already exp'd for SnakeBeta) and 1/(beta + 1e-9).
struct Snake {
    a: Tensor,
    inv: Tensor,
}

struct ResUnit {
    s0: Snake,
    c7: Conv,
    s1: Snake,
    c1: Conv,
}
struct EncBlock {
    rus: Vec<ResUnit>,
    snake: Snake,
    down: Conv,
    stride: usize,
}
struct Amp {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
    acts: Vec<Snake>,
}

pub struct AudioVae {
    dev: Arc<Device>,
    // encoder
    conv0: Conv,
    blocks: Vec<EncBlock>,
    enc_snake: Snake,
    enc_final: Conv,
    // posterior head
    ln1_w: Tensor,
    ln1_b: Tensor,
    ln3_w: Tensor,
    ln3_b: Tensor,
    qkv: Conv,
    proj3: Conv,
    tail: Vec<Tensor>, // wa, ba, n2w, n2b, nmw, nmb, w0, b0, w1, b1, w2, b2, wm, bm
    lat_mean: Tensor,
    lat_std: Tensor,
    // decoder
    dec_in: Conv,
    conv_pre: Conv,
    ups: Vec<Conv>,
    resblocks: Vec<Amp>,
    act_post: Snake,
    conv_post: Tensor,
    filt: Tensor,
    /// Process the encoder conv stack one stereo channel at a time above this many samples per channel.
    pub enc_split_samples: usize,
    /// Per-kernel GPU timing when H3_PROFILE is set (report with `prof.report()`).
    pub prof: Profiler,
}

struct Ld<'a> {
    st: &'a SafeTensors,
    dev: &'a Device,
}
impl<'a> Ld<'a> {
    fn host(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let info = self.st.info(name)?;
        ensure!(info.dtype == "F32", "{name}: expected F32, got {}", info.dtype);
        Ok((self.st.f32s(name)?, info.shape.clone()))
    }
    fn t(&self, name: &str) -> Result<Tensor> {
        let (v, s) = self.host(name)?;
        Tensor::from_f32(self.dev, &v, &s)
    }
    fn up(&self, v: &[f32]) -> Result<Tensor> {
        Tensor::from_f32(self.dev, v, &[v.len()])
    }
    fn conv(&self, p: &str, bias: bool, dil: usize, pad: usize, stride: usize) -> Result<Conv> {
        let (w, s) = self.host(&format!("{p}.weight"))?;
        ensure!(s.len() == 3, "{p}: conv weight rank {}", s.len());
        let b = if bias { Some(self.t(&format!("{p}.bias"))?) } else { None };
        // [co][ci][k] -> tap-major, K-outer [k][ci][co]
        let (co, ci, k) = (s[0], s[1], s[2]);
        let mut wt = vec![0f32; w.len()];
        for o in 0..co {
            for i in 0..ci {
                for m in 0..k {
                    wt[(m * ci + i) * co + o] = w[(o * ci + i) * k + m];
                }
            }
        }
        Ok(Conv { w: self.up(&wt)?, b, co: s[0], ci: s[1], kt: s[2], dil: dil as i32, pad: pad as i32, stride: stride as i32, nph: 1 })
    }
    /// nn.Linear [out, in] as a 1x1 conv; optional explicit bias vector.
    fn linear(&self, p: &str, bias: Option<Vec<f32>>) -> Result<Conv> {
        let (w, s) = self.host(&format!("{p}.weight"))?;
        ensure!(s.len() == 2);
        let (co, ci) = (s[0], s[1]);
        let mut wt = vec![0f32; w.len()];
        for o in 0..co {
            for i in 0..ci {
                wt[i * co + o] = w[o * ci + i];
            }
        }
        let w = wt;
        let b = match bias {
            Some(v) => Some(self.up(&v)?),
            None => Some(self.t(&format!("{p}.bias"))?),
        };
        Ok(Conv { w: self.up(&w)?, b, co: s[0], ci: s[1], kt: 1, dil: 1, pad: 0, stride: 1, nph: 1 })
    }
    /// ConvTranspose1d (weight [ci, co, K], stride s, padding p) re-laid out per output phase.
    fn conv_t(&self, p: &str, stride: usize) -> Result<Conv> {
        let (w, sh) = self.host(&format!("{p}.weight"))?;
        let (ci, co, k) = (sh[0], sh[1], sh[2]);
        let pad = (k - stride) / 2;
        let kt = (k + stride - 1) / stride;
        let mut out = vec![0f32; stride * co * ci * kt];
        for ph in 0..stride {
            for o in 0..co {
                for i in 0..ci {
                    for m in 0..kt {
                        let kk = ph + m * stride;
                        if kk < k {
                            out[ph * co * ci * kt + (m * ci + i) * co + o] = w[(i * co + o) * k + kk];
                        }
                    }
                }
            }
        }
        Ok(Conv {
            w: self.up(&out)?,
            b: Some(self.t(&format!("{p}.bias"))?),
            ci,
            co,
            kt,
            dil: -1,
            pad: pad as i32,
            stride: stride as i32,
            nph: stride,
        })
    }
    fn snake1d(&self, name: &str) -> Result<Snake> {
        let (a, _) = self.host(name)?;
        let inv: Vec<f32> = a.iter().map(|&x| 1.0f32 / (x + 1e-9f32)).collect();
        Ok(Snake { a: self.up(&a)?, inv: self.up(&inv)? })
    }
    fn snake_beta(&self, p: &str) -> Result<Snake> {
        let (la, _) = self.host(&format!("{p}.act.alpha"))?;
        let (lb, _) = self.host(&format!("{p}.act.beta"))?;
        let a: Vec<f32> = la.iter().map(|x| x.exp()).collect();
        let inv: Vec<f32> = lb.iter().map(|x| 1.0f32 / (x.exp() + 1e-9f32)).collect();
        Ok(Snake { a: self.up(&a)?, inv: self.up(&inv)? })
    }
}

/// Fused conv epilogue (0 = null pointer).
#[derive(Clone, Copy, Default)]
struct Epi {
    y: u64,
    y2: u64,
    r: u64,
    acc: u64,
    sa: u64,
    si: u64,
    div: f32,
}
impl Epi {
    fn y(y: u64) -> Epi {
        Epi { y, div: 1.0, ..Default::default() }
    }
    fn snake(mut self, y2: u64, s: &Snake) -> Epi {
        self.y2 = y2;
        self.sa = s.a.ptr;
        self.si = s.inv.ptr;
        self
    }
    fn res(mut self, r: u64) -> Epi {
        self.r = r;
        self
    }
}

fn ptr(t: &Tensor) -> u64 {
    t.ptr
}

impl AudioVae {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<AudioVae> {
        let st = SafeTensors::open(path)?;
        ensure!(st.has("pre_block.attn.zero_k_bias"), "{} is not a MiniMax H3 audio VAE", path.display());
        let ld = Ld { st: &st, dev: &dev };

        // ---- encoder
        let conv0 = ld.conv("encoder.block.0", true, 1, 3, 1)?;
        let strides = [2usize, 4, 4, 5, 5];
        let mut blocks = Vec::new();
        for (bi, &s) in strides.iter().enumerate() {
            let p = format!("encoder.block.{}.block", bi + 1);
            let mut rus = Vec::new();
            for (r, d) in [1usize, 3, 9].iter().enumerate() {
                let q = format!("{p}.{r}.block");
                rus.push(ResUnit {
                    s0: ld.snake1d(&format!("{q}.0.alpha"))?,
                    c7: ld.conv(&format!("{q}.1"), true, *d, 3 * d, 1)?,
                    s1: ld.snake1d(&format!("{q}.2.alpha"))?,
                    c1: ld.conv(&format!("{q}.3"), true, 1, 0, 1)?,
                });
            }
            blocks.push(EncBlock {
                rus,
                snake: ld.snake1d(&format!("{p}.3.alpha"))?,
                down: ld.conv(&format!("{p}.4"), true, 1, (s + 1) / 2, s)?,
                stride: s,
            });
        }
        let enc_snake = ld.snake1d("encoder.block.6.alpha")?;
        let enc_final = ld.conv("encoder.block.7", true, 1, 1, 1)?;

        // ---- posterior head
        let (qb, _) = ld.host("pre_block.attn.q_bias")?;
        let (kb, _) = ld.host("pre_block.attn.zero_k_bias")?;
        let (vb, _) = ld.host("pre_block.attn.v_bias")?;
        let qkv_bias: Vec<f32> = qb.iter().chain(kb.iter()).chain(vb.iter()).copied().collect();
        let qkv = ld.linear("pre_block.attn.qkv", Some(qkv_bias))?;
        let proj3 = ld.linear("pre_block.proj", None)?;
        let mut tail = Vec::new();
        for n in [
            "pre_block.attn.proj.weight",
            "pre_block.attn.proj.bias",
            "pre_block.norm2.weight",
            "pre_block.norm2.bias",
            "pre_block.mlp.norm.weight",
            "pre_block.mlp.norm.bias",
            "pre_block.mlp.w0.weight",
            "pre_block.mlp.w0.bias",
            "pre_block.mlp.w1.weight",
            "pre_block.mlp.w1.bias",
            "pre_block.mlp.w2.weight",
            "pre_block.mlp.w2.bias",
        ] {
            tail.push(ld.t(n)?);
        }
        // mean_proj (1x1 conv 32->32) is applied inside the tail kernel
        tail.push(ld.t("mean_proj.weight")?);
        tail.push(ld.t("mean_proj.bias")?);

        // ---- decoder
        let dec_in = ld.conv("dec_in_proj", true, 1, 0, 1)?;
        let conv_pre = ld.conv("decoder.conv_pre", true, 1, 3, 1)?;
        let rates = [5usize, 5, 2, 2, 2, 2, 2];
        let mut ups = Vec::new();
        for (i, &u) in rates.iter().enumerate() {
            ups.push(ld.conv_t(&format!("decoder.ups.{i}.0"), u)?);
        }
        let mut resblocks = Vec::new();
        for i in 0..rates.len() * 3 {
            let k = [3usize, 7, 11][i % 3];
            let p = format!("decoder.resblocks.{i}");
            let mut convs1 = Vec::new();
            let mut convs2 = Vec::new();
            for (j, d) in [1usize, 3, 5].iter().enumerate() {
                convs1.push(ld.conv(&format!("{p}.convs1.{j}"), true, *d, (k * d - d) / 2, 1)?);
                convs2.push(ld.conv(&format!("{p}.convs2.{j}"), true, 1, (k - 1) / 2, 1)?);
            }
            let mut acts = Vec::new();
            for a in 0..6 {
                acts.push(ld.snake_beta(&format!("{p}.activations.{a}"))?);
            }
            resblocks.push(Amp { convs1, convs2, acts });
        }
        let act_post = ld.snake_beta("decoder.activation_post")?;
        let (cp, cps) = ld.host("decoder.conv_post.weight")?;
        ensure!(cps == vec![1, 8, 7], "conv_post shape {cps:?}");
        let conv_post = ld.up(&cp)?;

        // all alias-free filters are the same kaiser-sinc (cutoff 0.25, half-width 0.3, 12 taps); verify
        let mut filt: Option<Vec<f32>> = None;
        let mut names: Vec<&String> = st.tensors.keys().filter(|k| k.ends_with(".filter")).collect();
        names.sort();
        for n in names {
            let (f, _) = ld.host(n)?;
            ensure!(f.len() == 12, "{n}: filter length {}", f.len());
            match &filt {
                None => filt = Some(f),
                Some(g) => {
                    if *g != f {
                        bail!("alias-free filter {n} differs from the others (unsupported)");
                    }
                }
            }
        }
        let filt = ld.up(&filt.ok_or_else(|| anyhow::anyhow!("no alias-free filters found"))?)?;

        dev.set_max_smem("avae_attn", ATTN_SMEM)?;
        Ok(AudioVae {
            ln1_w: ld.t("pre_block.norm1.weight")?,
            ln1_b: ld.t("pre_block.norm1.bias")?,
            ln3_w: ld.t("pre_block.norm3.weight")?,
            ln3_b: ld.t("pre_block.norm3.bias")?,
            lat_mean: ld.t("latents_mean")?,
            lat_std: ld.t("latents_std")?,
            dev: dev.clone(),
            conv0,
            blocks,
            enc_snake,
            enc_final,
            qkv,
            proj3,
            tail,
            dec_in,
            conv_pre,
            ups,
            resblocks,
            act_post,
            conv_post,
            filt,
            enc_split_samples: 30 * SAMPLE_RATE,
            prof: Profiler::new(),
        })
    }

    /// Run a convolution. x: [nb][ci] rows with batch stride `xbs` and channel stride `xcs` (elements),
    /// output [nb][co][lout] contiguous.
    fn conv(&self, c: &Conv, x: u64, xbs: usize, xcs: usize, nb: usize, lin: usize, lout: usize, e: Epi) -> Result<()> {
        let (istride, dil, ipad, ostride, opad) =
            if c.nph > 1 { (1, -1, 0, c.stride, c.pad) } else { (c.stride, c.dil, c.pad, 1, 0) };
        let nq_max = (lout + ostride as usize - 1) / ostride as usize + 1;
        let nblocks = |bm: usize, bn: usize| ((nq_max + bn - 1) / bn) * ((c.co + bm - 1) / bm) * nb * c.nph;
        let (mut name, mut bm, mut bn) = match c.co {
            n if n >= 128 => ("avae_conv_128", 128, 128),
            n if n > 32 => ("avae_conv_64", 64, 128),
            n if n > 16 => ("avae_conv_32", 32, 256),
            n if n > 8 => ("avae_conv_16", 16, 256),
            _ => ("avae_conv_8", 8, 256),
        };
        // small grids (short sequences / deep layers): smaller tiles to fill the SMs
        if c.co >= 64 && nblocks(bm, bn) < 2 * self.dev.sm_count as usize {
            (name, bm, bn) = ("avae_conv_64s", 64, 64);
        }
        let grid = (((nq_max + bn - 1) / bn) as u32, ((c.co + bm - 1) / bm) as u32, (nb * c.nph) as u32);
        let bias = c.b.as_ref().map(ptr).unwrap_or(0);
        let label = if self.prof.enabled() { format!("conv ci{} co{} k{} ph{}", c.ci, c.co, c.kt, c.nph) } else { String::new() };
        self.prof.time(&self.dev, &label, || self.dev.launch(
            name,
            grid,
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(c.w.ptr),
                Arg::Ptr(bias),
                Arg::Ptr(x),
                Arg::I64(xbs as i64),
                Arg::I32(xcs as i32),
                Arg::Ptr(e.y),
                Arg::Ptr(e.y2),
                Arg::Ptr(e.r),
                Arg::Ptr(e.acc),
                Arg::Ptr(e.sa),
                Arg::Ptr(e.si),
                Arg::I32(c.ci as i32),
                Arg::I32(c.co as i32),
                Arg::I32(lin as i32),
                Arg::I32(lout as i32),
                Arg::I32(c.kt as i32),
                Arg::I32(istride),
                Arg::I32(dil),
                Arg::I32(ipad),
                Arg::I32(c.nph as i32),
                Arg::I32(ostride),
                Arg::I32(opad),
                Arg::F32(e.div),
            ],
        ))
    }

    /// Contiguous [nb][c.ci][lin] input.
    fn conv_c(&self, c: &Conv, x: u64, nb: usize, lin: usize, lout: usize, e: Epi) -> Result<()> {
        self.conv(c, x, c.ci * lin, lin, nb, lin, lout, e)
    }

    fn act(&self, s: &Snake, x: u64, y: u64, nb: usize, ch: usize, l: usize) -> Result<()> {
        let label = if self.prof.enabled() { format!("act C{ch}") } else { String::new() };
        self.prof.time(&self.dev, &label, || self.dev.launch(
            "avae_act",
            (((l + 255) / 256) as u32, ch as u32, nb as u32),
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(x),
                Arg::Ptr(y),
                Arg::Ptr(self.filt.ptr),
                Arg::Ptr(s.a.ptr),
                Arg::Ptr(s.inv.ptr),
                Arg::I32(ch as i32),
                Arg::I32(l as i32),
                Arg::I64((ch * l) as i64),
                Arg::I64((ch * l) as i64),
            ],
        ))
    }

    /// Encoder conv stack for `nb` mono signals x [nb][L] (L multiple of 800) -> e_out [nb][2048][L/800].
    fn encode_convs(&self, x: u64, nb: usize, l: usize, e_out: u64) -> Result<()> {
        let dev = &*self.dev;
        let n = 64 * l * nb;
        let xb = Tensor::new(dev, DType::F32, &[n])?;
        let mut sb = Tensor::new(dev, DType::F32, &[n])?;
        let mut ub = Tensor::new(dev, DType::F32, &[n])?;
        // conv0 (1 -> 64, k7): residual stream + Snake1d for the first residual unit
        self.conv_c(&self.conv0, x, nb, l, l, Epi::y(xb.ptr).snake(sb.ptr, &self.blocks[0].rus[0].s0))?;
        let mut li = l;
        let nblk = self.blocks.len();
        for (bi, blk) in self.blocks.iter().enumerate() {
            for (r, ru) in blk.rus.iter().enumerate() {
                // snake(x) in sb -> conv7 -> snake -> ub
                self.conv_c(&ru.c7, sb.ptr, nb, li, li, Epi { div: 1.0, ..Default::default() }.snake(ub.ptr, &ru.s1))?;
                // ub -> conv1x1 + x -> x (residual stream), snake(x) for the next consumer -> sb
                let last = r + 1 == blk.rus.len();
                let next = if last { &blk.snake } else { &blk.rus[r + 1].s0 };
                let e = Epi { y: if last { 0 } else { xb.ptr }, div: 1.0, ..Default::default() }.res(xb.ptr).snake(sb.ptr, next);
                self.conv_c(&ru.c1, ub.ptr, nb, li, li, e)?;
            }
            let lo = li / blk.stride;
            // strided downsampling conv: reads sb, writes the next residual stream (xb) + its snake (ub)
            let e = if bi + 1 < nblk {
                Epi::y(xb.ptr).snake(ub.ptr, &self.blocks[bi + 1].rus[0].s0)
            } else {
                Epi { div: 1.0, ..Default::default() }.snake(ub.ptr, &self.enc_snake)
            };
            self.conv_c(&blk.down, sb.ptr, nb, li, lo, e)?;
            std::mem::swap(&mut sb, &mut ub);
            li = lo;
        }
        // final conv 2048 -> 2048, k3
        self.conv_c(&self.enc_final, sb.ptr, nb, li, li, Epi::y(e_out))?;
        Ok(())
    }

    /// wav: host f32 stereo planar [2, n] at 32 kHz in [-1, 1] (zero-padded on the right to a multiple of 800).
    /// Returns the normalized latent f32 [32, 2, T] (device), T = ceil(n / 800).
    pub fn encode(&self, wav: &[f32], n: usize) -> Result<Tensor> {
        ensure!(wav.len() >= 2 * n && n > 0, "encode: wav has {} samples, expected 2 x {n}", wav.len());
        let dev = &*self.dev;
        let t = (n + HOP - 1) / HOP;
        let l = t * HOP;
        let mut host = vec![0f32; 2 * l];
        host[..n].copy_from_slice(&wav[..n]);
        host[l..l + n].copy_from_slice(&wav[n..2 * n]);
        let x = Tensor::from_f32(dev, &host, &[2, l])?;
        let e = Tensor::new(dev, DType::F32, &[2, 2048, t])?;
        if l > self.enc_split_samples {
            for b in 0..2 {
                self.encode_convs(x.ptr + (b * l * 4) as u64, 1, l, e.ptr + (b * 2048 * t * 4) as u64)?;
            }
        } else {
            self.encode_convs(x.ptr, 2, l, e.ptr)?;
        }
        drop(x);
        self.head(&e, t)
    }

    /// AttnProjection + mean_proj + normalization. e: [2][2048][T] -> [32][2][T].
    fn head(&self, e: &Tensor, t: usize) -> Result<Tensor> {
        let dev = &*self.dev;
        let nb = 2;
        let n1 = Tensor::new(dev, DType::F32, &[nb, 2048, t])?;
        let n3 = Tensor::new(dev, DType::F32, &[nb, 2048, t])?;
        dev.launch(
            "avae_ln2_cm",
            (((t + 31) / 32) as u32, nb as u32, 1),
            (32, 8, 1),
            0,
            &[
                Arg::Ptr(e.ptr),
                Arg::Ptr(n1.ptr),
                Arg::Ptr(n3.ptr),
                Arg::Ptr(self.ln1_w.ptr),
                Arg::Ptr(self.ln1_b.ptr),
                Arg::Ptr(self.ln3_w.ptr),
                Arg::Ptr(self.ln3_b.ptr),
                Arg::I32(2048),
                Arg::I32(t as i32),
                Arg::F32(1e-5),
            ],
        )?;
        let qkv = Tensor::new(dev, DType::F32, &[nb, 6144, t])?;
        self.conv_c(&self.qkv, n1.ptr, nb, t, t, Epi::y(qkv.ptr))?;
        let p = Tensor::new(dev, DType::F32, &[nb, 32, t])?;
        self.conv_c(&self.proj3, n3.ptr, nb, t, t, Epi::y(p.ptr))?;
        drop(n1);
        drop(n3);
        let o = Tensor::new(dev, DType::F32, &[nb, 8, t, 256])?;
        dev.launch(
            "avae_attn",
            (((t + 15) / 16) as u32, 8, nb as u32),
            (256, 1, 1),
            ATTN_SMEM,
            &[Arg::Ptr(qkv.ptr), Arg::Ptr(o.ptr), Arg::I32(8), Arg::I32(t as i32), Arg::F32(1.0 / 16.0)],
        )?;
        let z = Tensor::new(dev, DType::F32, &[32, nb, t])?;
        let mut args = vec![Arg::Ptr(o.ptr), Arg::Ptr(p.ptr), Arg::Ptr(z.ptr)];
        for w in &self.tail {
            args.push(Arg::Ptr(w.ptr));
        }
        args.push(Arg::Ptr(self.lat_mean.ptr));
        args.push(Arg::Ptr(self.lat_std.ptr));
        args.extend_from_slice(&[Arg::I32(8), Arg::I32(t as i32), Arg::I32(nb as i32), Arg::F32(1e-5)]);
        dev.launch("avae_head_tail", (t as u32, nb as u32, 1), (256, 1, 1), 0, &args)?;
        Ok(z)
    }

    /// z: f32 [32, 2, T] normalized latent (device) -> host f32 stereo planar [2, T*800] in [-1, 1] (already clamped).
    pub fn decode(&self, z: &Tensor, t: usize) -> Result<Vec<f32>> {
        let wav = self.decode_dev(z, t)?;
        wav.to_f32_vec(&self.dev)
    }

    /// Same as `decode` but leaves the waveform [2, T*800] on the device.
    pub fn decode_dev(&self, z: &Tensor, t: usize) -> Result<Tensor> {
        ensure!(z.dtype == DType::F32 && z.numel() == 32 * 2 * t, "decode: latent {:?} is not f32 [32,2,{t}]", z.shape);
        let dev = &*self.dev;
        let nb = 2;
        let zz = Tensor::new(dev, DType::F32, &[nb, 32, t])?;
        dev.launch_n(
            "avae_denorm",
            32 * nb * t,
            &[Arg::Ptr(z.ptr), Arg::Ptr(zz.ptr), Arg::Ptr(self.lat_mean.ptr), Arg::Ptr(self.lat_std.ptr), Arg::I32(nb as i32), Arg::I32(t as i32)],
        )?;
        let d0 = Tensor::new(dev, DType::F32, &[nb, 2048, t])?;
        self.conv_c(&self.dec_in, zz.ptr, nb, t, t, Epi::y(d0.ptr))?;
        drop(zz);
        // scratch: 5 buffers sized for the largest stage
        let mut maxn = 1024 * t;
        let mut l = t;
        for u in &self.ups {
            l *= u.stride as usize;
            maxn = maxn.max(u.co * l);
        }
        let bufs: Vec<Tensor> = (0..5).map(|_| Tensor::new(dev, DType::F32, &[nb * maxn])).collect::<Result<_>>()?;
        let (xs, cur, ab, cb, acc) = (bufs[0].ptr, bufs[1].ptr, bufs[2].ptr, bufs[3].ptr, bufs[4].ptr);
        // conv_pre into `acc` (the stage input slot)
        self.conv_c(&self.conv_pre, d0.ptr, nb, t, t, Epi::y(acc))?;
        drop(d0);
        let mut l = t;
        for (i, up) in self.ups.iter().enumerate() {
            let lo = l * up.stride as usize;
            let ch = up.co;
            self.conv_c(up, acc, nb, l, lo, Epi::y(xs))?;
            l = lo;
            for j in 0..3 {
                let rb = &self.resblocks[i * 3 + j];
                let mut x_in = xs;
                for k in 0..3 {
                    self.act(&rb.acts[2 * k], x_in, ab, nb, ch, l)?;
                    self.conv_c(&rb.convs1[k], ab, nb, l, l, Epi::y(cb))?;
                    self.act(&rb.acts[2 * k + 1], cb, ab, nb, ch, l)?;
                    if k < 2 {
                        self.conv_c(&rb.convs2[k], ab, nb, l, l, Epi::y(cur).res(x_in))?;
                        x_in = cur;
                    } else {
                        // xs_sum (+)= block output; last block divides by 3
                        let e = Epi { y: acc, r: x_in, acc: if j > 0 { acc } else { 0 }, div: if j == 2 { 3.0 } else { 1.0 }, ..Default::default() };
                        self.conv_c(&rb.convs2[k], ab, nb, l, l, e)?;
                    }
                }
            }
        }
        let ch = self.ups.last().unwrap().co;
        let out = Tensor::new(dev, DType::F32, &[nb, l])?;
        dev.launch(
            "avae_act_post_out",
            (((l + 255) / 256) as u32, nb as u32, 1),
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(acc),
                Arg::Ptr(out.ptr),
                Arg::Ptr(self.filt.ptr),
                Arg::Ptr(self.act_post.a.ptr),
                Arg::Ptr(self.act_post.inv.ptr),
                Arg::Ptr(self.conv_post.ptr),
                Arg::I32(ch as i32),
                Arg::I32(l as i32),
            ],
        )?;
        Ok(out)
    }
}

/// ComfyUI's VAE.encode crops (centered) the waveform to a multiple of 800 samples before encoding
/// (`vae_encode_crop_pixels`, crop_input=True for this VAE). Returns (start, len) of the kept window;
/// callers that want bit-for-bit Comfy behaviour should encode `wav[start..start+len]` of each channel.
pub fn comfy_crop_window(n: usize) -> (usize, usize) {
    let keep = (n / HOP) * HOP;
    ((n % HOP) / 2, keep)
}

/// Port of comfy.audio.resample (bandlimited sinc, hann window, lowpass_filter_width 6, rolloff 0.99) on the host.
/// x: one channel. Output length ceil(new * len / orig) as torchaudio.
pub fn resample(x: &[f32], orig: usize, new: usize) -> Vec<f32> {
    if orig == new {
        return x.to_vec();
    }
    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = gcd(orig, new);
    let (o, nf) = (orig / g, new / g);
    let lpw = 6.0f64;
    let base = (o.min(nf) as f64) * 0.99;
    let width = (lpw * o as f64 / base).ceil() as usize;
    // kernel [nf][2*width + o]
    let klen = 2 * width + o;
    let mut kern = vec![0f32; nf * klen];
    for p in 0..nf {
        for i in 0..klen {
            // computed in f32 like torch (waveform dtype)
            let idx = (i as f32 - width as f32) / o as f32;
            let mut tt = (-(p as f32) / nf as f32 + idx) * base as f32;
            tt = tt.clamp(-lpw as f32, lpw as f32);
            let w = (tt * std::f32::consts::PI / lpw as f32 / 2.0).cos();
            let w = w * w;
            let tp = tt * std::f32::consts::PI;
            let s = if tp == 0.0 { 1.0 } else { tp.sin() / tp };
            kern[p * klen + i] = s * (w * ((base / o as f64) as f32));
        }
    }
    let len = x.len();
    let padded: Vec<f32> = std::iter::repeat(0f32).take(width).chain(x.iter().copied()).chain(std::iter::repeat(0f32).take(width + o)).collect();
    let nframes = (padded.len() - klen) / o + 1;
    let target = ((nf as f64 * len as f64 / o as f64) as f32).ceil() as usize;
    let mut out = Vec::with_capacity(nframes * nf);
    for f in 0..nframes {
        let seg = &padded[f * o..f * o + klen];
        for p in 0..nf {
            let k = &kern[p * klen..(p + 1) * klen];
            let mut acc = 0f32;
            for i in 0..klen {
                acc += seg[i] * k[i];
            }
            out.push(acc);
        }
    }
    out.truncate(target);
    out
}
