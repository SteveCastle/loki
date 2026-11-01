//! SDXL and SDXL-architecture checkpoints: a single file holding the LDM UNet (`model.diffusion_model.*`), both text
//! encoders (`conditioner.embedders.*`, see clip.rs) and the VAE (`first_stage_model.*`, see vae.rs).
//!
//! The UNet is read from the file's own tensors (levels, attention depths and widths follow the weights): ResBlocks with
//! the timestep + size embedding added after the first conv, SpatialTransformers with linear projections, self- and
//! cross-attention (64-wide heads) and a GEGLU feed-forward, skip connections concatenated on the way up.
//! Sampling is k-diffusion's (ComfyUI's KSampler): the discrete DDPM schedule (scaled-linear betas 0.00085..0.012, 1000
//! steps; zero terminal SNR when the checkpoint says so), eps or v prediction, the normal / karras / sgm_uniform
//! schedulers, and the euler, euler_ancestral and dpmpp_2m samplers. Latents stay on the host between UNet calls (128 x
//! 128 x 4 floats at 1024 px).
use crate::clip::{ClipTokenizer, SdxlText, TextCond};
use crate::cli::{self, Cli, Plan};
use crate::cuda::{Arg, Device};
use crate::genmeta;
use crate::image::Rgb8;
use crate::linear::{self, Lin, Want};
use crate::model::{self, Dense};
use crate::ops::{self, Act, AttnView, Epi};
use crate::rng;
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::vae::{self, Conv, Plane, Vae};
use crate::weights::Loader;
use anyhow::{bail, ensure, Context, Result};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

const P: &str = "model.diffusion_model.";
const GN_RES_EPS: f32 = 1e-5;
const GN_XF_EPS: f32 = 1e-6;
const LN_EPS: f32 = 1e-5;
const HEAD: usize = 64;
/// latent channels, and the conv width they are padded to
const LC: usize = 4;
const LC_PAD: usize = 8;

/// What the UNet predicts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prediction {
    Eps,
    V,
}

/// The checkpoint's prediction type and whether it was trained with zero terminal SNR: the `v_pred` / `ztsnr` marker
/// tensors ComfyUI writes, or the `modelspec.prediction_type` metadata.
pub fn prediction_of(st: &SafeTensors) -> (Prediction, bool) {
    let meta = st.metadata.get("modelspec.prediction_type").map(|s| s.to_lowercase()).unwrap_or_default();
    let v = st.has("v_pred") || meta.starts_with('v');
    let ztsnr = st.has("ztsnr") || meta.contains("zsnr") || meta.contains("ztsnr");
    (if v { Prediction::V } else { Prediction::Eps }, ztsnr)
}

/// Step-distilled SDXL variants (Lightning, Turbo, Hyper, DMD2, LCM), by file name.
pub fn is_distilled(path: &Path) -> bool {
    let n = path.file_name().map(|f| f.to_string_lossy().to_lowercase()).unwrap_or_default();
    ["lightning", "turbo", "hyper", "dmd", "lcm"].iter().any(|k| n.contains(k))
}

// ------------------------------------------------------------------------------------------------ weights
struct Norm {
    g: Tensor,
    b: Tensor,
}
struct Res {
    n1: Norm,
    c1: Conv,
    emb: Dense,
    n2: Norm,
    c2: Conv,
    skip: Option<Conv>,
}
struct Block {
    ln1: (Tensor, Tensor),
    qkv: Tensor,
    o1: Dense,
    ln2: (Tensor, Tensor),
    q2: Tensor,
    kv2: Tensor,
    o2: Dense,
    ln3: (Tensor, Tensor),
    ff1: Dense,
    ff2: Dense,
}
struct Xf {
    norm: Norm,
    proj_in: Dense,
    blocks: Vec<Block>,
    proj_out: Dense,
}
enum Layer {
    Res(Res),
    Xf(Xf),
    Down(Conv),
    Up(Conv),
}

pub struct Unet {
    dev: Arc<Device>,
    conv_in: Conv,
    inputs: Vec<Vec<Layer>>,
    middle: Vec<Layer>,
    outputs: Vec<Vec<Layer>>,
    out_norm: Norm,
    conv_out: Conv,
    time: (Dense, Dense),
    label: (Dense, Dense),
    /// model channels (the timestep embedding is model_channels wide)
    ch: usize,
    pub prediction: Prediction,
    pub ztsnr: bool,
}

fn bf16w(l: &mut Loader, prefix: &str) -> Result<Tensor> {
    match linear::lin(l, prefix, Want::Bf16)? {
        Lin::Bf16(w) => Ok(w),
        _ => bail!("{prefix}: expected a float weight"),
    }
}
fn dense(l: &mut Loader, prefix: &str) -> Result<Dense> {
    Ok(Dense { w: bf16w(l, prefix)?, b: linear::bias(l, prefix)? })
}
fn gn(l: &mut Loader, p: &str) -> Result<Norm> {
    Ok(Norm { g: l.f32(&format!("{p}.weight"))?, b: l.f32(&format!("{p}.bias"))? })
}
fn ln(l: &mut Loader, p: &str) -> Result<(Tensor, Tensor)> {
    Ok((linear::small_bf16(l, &format!("{p}.weight"))?, linear::small_bf16(l, &format!("{p}.bias"))?))
}
fn cat_rows(dev: &Device, parts: &[Tensor]) -> Result<Tensor> {
    let k = parts[0].shape[1];
    let n: usize = parts.iter().map(|t| t.shape[0]).sum();
    let w = Tensor::new(dev, DType::BF16, &[n, k])?;
    let mut off = 0;
    for t in parts {
        ensure!(t.shape[1] == k, "fused weight K mismatch");
        dev.dtod(w.ptr + off as u64, t.ptr, t.bytes())?;
        off += t.bytes();
    }
    Ok(w)
}

fn load_res(l: &mut Loader, p: &str) -> Result<Res> {
    let skip = if l.st.has(&format!("{p}.skip_connection.weight")) { Some(Conv::load(l, &format!("{p}.skip_connection"), None, None)?) } else { None };
    Ok(Res {
        n1: gn(l, &format!("{p}.in_layers.0"))?,
        c1: Conv::load(l, &format!("{p}.in_layers.2"), None, None)?,
        emb: dense(l, &format!("{p}.emb_layers.1"))?,
        n2: gn(l, &format!("{p}.out_layers.0"))?,
        c2: Conv::load(l, &format!("{p}.out_layers.3"), None, None)?,
        skip,
    })
}

fn load_xf(l: &mut Loader, p: &str) -> Result<Xf> {
    let dev = l.dev.clone();
    let n = (0..).take_while(|j| l.st.has(&format!("{p}.transformer_blocks.{j}.norm1.weight"))).count();
    ensure!(n > 0, "{p}: no transformer blocks");
    let mut blocks = Vec::with_capacity(n);
    for j in 0..n {
        let b = format!("{p}.transformer_blocks.{j}");
        let qkv = cat_rows(&dev, &[bf16w(l, &format!("{b}.attn1.to_q"))?, bf16w(l, &format!("{b}.attn1.to_k"))?, bf16w(l, &format!("{b}.attn1.to_v"))?])?;
        let kv2 = cat_rows(&dev, &[bf16w(l, &format!("{b}.attn2.to_k"))?, bf16w(l, &format!("{b}.attn2.to_v"))?])?;
        blocks.push(Block {
            ln1: ln(l, &format!("{b}.norm1"))?,
            qkv,
            o1: dense(l, &format!("{b}.attn1.to_out.0"))?,
            ln2: ln(l, &format!("{b}.norm2"))?,
            q2: bf16w(l, &format!("{b}.attn2.to_q"))?,
            kv2,
            o2: dense(l, &format!("{b}.attn2.to_out.0"))?,
            ln3: ln(l, &format!("{b}.norm3"))?,
            ff1: dense(l, &format!("{b}.ff.net.0.proj"))?,
            ff2: dense(l, &format!("{b}.ff.net.2"))?,
        });
    }
    Ok(Xf { norm: gn(l, &format!("{p}.norm"))?, proj_in: dense(l, &format!("{p}.proj_in"))?, blocks, proj_out: dense(l, &format!("{p}.proj_out"))? })
}

/// The layers of `input_blocks.{i}` / `output_blocks.{i}` / `middle_block`, in order.
fn load_block(l: &mut Loader, p: &str) -> Result<Vec<Layer>> {
    let mut out = Vec::new();
    for k in 0.. {
        let q = format!("{p}.{k}");
        if l.st.has(&format!("{q}.op.weight")) {
            out.push(Layer::Down(Conv::load(l, &format!("{q}.op"), None, None)?));
        } else if l.st.has(&format!("{q}.in_layers.0.weight")) {
            out.push(Layer::Res(load_res(l, &q)?));
        } else if l.st.has(&format!("{q}.proj_in.weight")) {
            out.push(Layer::Xf(load_xf(l, &q)?));
        } else if l.st.has(&format!("{q}.conv.weight")) {
            out.push(Layer::Up(Conv::load(l, &format!("{q}.conv"), None, None)?));
        } else {
            break;
        }
    }
    Ok(out)
}

impl Unet {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Unet> {
        let st = SafeTensors::open(path)?;
        ensure!(st.has(&format!("{P}input_blocks.0.0.weight")) && st.has(&format!("{P}label_emb.0.0.weight")), "{} is not an SDXL checkpoint", path.display());
        let (prediction, ztsnr) = prediction_of(&st);
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let conv_in = Conv::load(&mut l, &format!("{P}input_blocks.0.0"), Some(LC_PAD), None)?;
        let ch = conv_in.co;
        let mut inputs = Vec::new();
        for i in 1.. {
            let b = load_block(&mut l, &format!("{P}input_blocks.{i}"))?;
            if b.is_empty() {
                break;
            }
            inputs.push(b);
        }
        let middle = load_block(&mut l, &format!("{P}middle_block"))?;
        let mut outputs = Vec::new();
        for i in 0.. {
            let b = load_block(&mut l, &format!("{P}output_blocks.{i}"))?;
            if b.is_empty() {
                break;
            }
            outputs.push(b);
        }
        ensure!(outputs.len() == inputs.len() + 1, "{}: {} input blocks but {} output blocks", path.display(), inputs.len() + 1, outputs.len());
        let out_norm = gn(&mut l, &format!("{P}out.0"))?;
        let conv_out = Conv::load(&mut l, &format!("{P}out.2"), None, Some(LC_PAD))?;
        let time = (dense(&mut l, &format!("{P}time_embed.0"))?, dense(&mut l, &format!("{P}time_embed.2"))?);
        let label = (dense(&mut l, &format!("{P}label_emb.0.0"))?, dense(&mut l, &format!("{P}label_emb.0.2"))?);
        ensure!(label.0.w.shape[1] == 2816, "{}: an SDXL UNet with {}-wide size conditioning (the refiner?) is not supported", path.display(), label.0.w.shape[1]);
        crate::info!("  unet: {:.2} GB uploaded in {:.1}s{}", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64(), match (prediction, ztsnr) {
            (Prediction::V, true) => ", v-prediction, zero terminal SNR",
            (Prediction::V, false) => ", v-prediction",
            _ => "",
        });
        drop(l);
        st.drop_in_background();
        Ok(Unet { dev, conv_in, inputs, middle, outputs, out_norm, conv_out, time, label, ch, prediction, ztsnr })
    }

    // ------------------------------------------------------------------ conditioning
    /// Step-independent state of one text at one size: context, cached cross-attention keys/values, size embedding.
    pub fn prepare(&self, c: &TextCond, h: usize, w: usize) -> Result<Branch> {
        let dev = &self.dev;
        let bits: Vec<u16> = c.ctx.iter().map(|&x| bf16_bits(x)).collect();
        let ctx = Tensor::from_bf16(dev, &bits, &[c.tokens, c.ctx.len() / c.tokens])?;
        // y = pooled | emb(orig h, w) | emb(crop top, left) | emb(target h, w), each scalar a 256-wide sinusoid
        let mut y = c.pooled.clone();
        for v in [h as f32, w as f32, 0.0, 0.0, h as f32, w as f32] {
            y.extend(sinusoid(v, 256));
        }
        let yb: Vec<u16> = y.iter().map(|&x| bf16_bits(x)).collect();
        let yt = Tensor::from_bf16(dev, &yb, &[1, yb.len()])?;
        let label = self.label.1.apply(dev, &self.label.0.apply(dev, &yt, Act::Silu)?, Act::None)?;
        let mut kv = Vec::new();
        let mut each = |layers: &[Layer]| -> Result<()> {
            for ly in layers {
                if let Layer::Xf(x) = ly {
                    for b in &x.blocks {
                        let t = Tensor::new(dev, DType::BF16, &[c.tokens, b.kv2.shape[0]])?;
                        ops::gemm_bf16(dev, &ctx, &b.kv2, None, Act::None, Epi::Store, &t)?;
                        kv.push(t);
                    }
                }
            }
            Ok(())
        };
        for b in &self.inputs {
            each(b)?;
        }
        each(&self.middle)?;
        for b in &self.outputs {
            each(b)?;
        }
        Ok(Branch { kv, tokens: c.tokens, label })
    }

    // ------------------------------------------------------------------ forward
    /// One UNet evaluation: `x` the scaled model input bf16 [h*w, 8] (4 channels used), `t` the discrete timestep.
    /// Returns the raw prediction plane [h*w, 8].
    pub fn forward(&self, br: &Branch, x: &Plane, t: f32) -> Result<Plane> {
        let dev = &self.dev;
        let tb: Vec<u16> = sinusoid(t, self.ch).iter().map(|&v| bf16_bits(v)).collect();
        let temb = self.time.1.apply(dev, &self.time.0.apply(dev, &Tensor::from_bf16(dev, &tb, &[1, self.ch])?, Act::Silu)?, Act::None)?;
        ops::add_inplace(dev, &temb, &br.label)?;
        ops::act_inplace(dev, &temb, Act::Silu)?;
        let e = temb;
        let mut kv = br.kv.iter();
        let mut h = Plane::new(dev, x.h, x.w, self.conv_in.co)?;
        vae::conv(dev, x, &self.conv_in, 1, 1, false, None, &h)?;
        let mut hs = vec![h.clone()];
        for b in &self.inputs {
            for ly in b {
                h = self.layer(ly, &h, &e, &mut kv, br.tokens)?;
            }
            hs.push(h.clone());
        }
        for ly in &self.middle {
            h = self.layer(ly, &h, &e, &mut kv, br.tokens)?;
        }
        for b in &self.outputs {
            let skip = hs.pop().context("UNet skip connections")?;
            ensure!(skip.h == h.h && skip.w == h.w, "UNet skip size mismatch ({}x{} vs {}x{})", skip.w, skip.h, h.w, h.h);
            let cat = Plane { t: Tensor::zeros(dev, DType::BF16, &[h.npix(), h.c + skip.c])?, h: h.h, w: h.w, c: h.c + skip.c };
            model::put_cols(dev, &cat.t, 0, &h.t)?;
            model::put_cols(dev, &cat.t, h.c, &skip.t)?;
            h = cat;
            for ly in b {
                h = self.layer(ly, &h, &e, &mut kv, br.tokens)?;
            }
        }
        let n = Plane::new(dev, h.h, h.w, h.c)?;
        ops::groupnorm(dev, &h.t, h.npix(), h.c, &self.out_norm.g, &self.out_norm.b, GN_RES_EPS, true, &n.t)?;
        let out = Plane::new(dev, h.h, h.w, self.conv_out.co)?;
        vae::conv(dev, &n, &self.conv_out, 1, 1, false, None, &out)?;
        Ok(out)
    }

    fn layer<'a>(&self, ly: &Layer, x: &Plane, e: &Tensor, kv: &mut impl Iterator<Item = &'a Tensor>, tokens: usize) -> Result<Plane> {
        let dev = &self.dev;
        match ly {
            Layer::Res(r) => self.res(r, x, e),
            Layer::Xf(xf) => self.xf(xf, x, kv, tokens),
            Layer::Down(c) => {
                let out = Plane::new(dev, x.h.div_ceil(2), x.w.div_ceil(2), c.co)?;
                vae::conv(dev, x, c, 2, 1, false, None, &out)?;
                Ok(out)
            }
            Layer::Up(c) => {
                let out = Plane::new(dev, x.h * 2, x.w * 2, c.co)?;
                vae::conv(dev, x, c, 1, 1, true, None, &out)?;
                Ok(out)
            }
        }
    }

    fn res(&self, r: &Res, x: &Plane, e: &Tensor) -> Result<Plane> {
        let dev = &self.dev;
        let h = Plane::new(dev, x.h, x.w, x.c)?;
        ops::groupnorm(dev, &x.t, x.npix(), x.c, &r.n1.g, &r.n1.b, GN_RES_EPS, true, &h.t)?;
        let c1 = Plane::new(dev, x.h, x.w, r.c1.co)?;
        vae::conv(dev, &h, &r.c1, 1, 1, false, None, &c1)?;
        drop(h);
        let ev = r.emb.apply(dev, e, Act::None)?;
        dev.launch_n("k_add_rowvec_bf16", c1.npix() * c1.c, &[Arg::Ptr(c1.t.ptr), Arg::Ptr(ev.ptr), Arg::I64(c1.npix() as i64), Arg::I32(c1.c as i32)])?;
        let h2 = Plane::new(dev, x.h, x.w, c1.c)?;
        ops::groupnorm(dev, &c1.t, c1.npix(), c1.c, &r.n2.g, &r.n2.b, GN_RES_EPS, true, &h2.t)?;
        drop(c1);
        let sc_plane;
        let sc = match &r.skip {
            Some(s) => {
                sc_plane = Plane::new(dev, x.h, x.w, s.co)?;
                vae::conv(dev, x, s, 1, 0, false, None, &sc_plane)?;
                &sc_plane
            }
            None => x,
        };
        let out = Plane::new(dev, x.h, x.w, r.c2.co)?;
        vae::conv(dev, &h2, &r.c2, 1, 1, false, Some(sc), &out)?;
        Ok(out)
    }

    fn xf<'a>(&self, xf: &Xf, x: &Plane, kv: &mut impl Iterator<Item = &'a Tensor>, tokens: usize) -> Result<Plane> {
        let dev = &self.dev;
        let (n, c) = (x.npix(), x.c);
        let heads = c / HEAD;
        let normed = Tensor::new(dev, DType::BF16, &[n, c])?;
        ops::groupnorm(dev, &x.t, n, c, &xf.norm.g, &xf.norm.b, GN_XF_EPS, false, &normed)?;
        let mut h = xf.proj_in.apply(dev, &normed, Act::None)?;
        let mut other = Tensor::new(dev, DType::BF16, &[n, c])?;
        let att = Tensor::new(dev, DType::BF16, &[n, c])?;
        for b in &xf.blocks {
            // self-attention
            ops::layernorm(dev, &h, Some(&b.ln1.0), Some(&b.ln1.1), LN_EPS, &normed)?;
            let qkv = Tensor::new(dev, DType::BF16, &[n, 3 * c])?;
            ops::gemm_bf16(dev, &normed, &b.qkv, None, Act::None, Epi::Store, &qkv)?;
            let v = |off: usize| AttnView { ptr: qkv.ptr + (off * 2) as u64, tok_stride: 3 * c, head_stride: HEAD, len: n };
            ops::attention(dev, v(0), v(c), v(2 * c), AttnView::contiguous(&att, heads, HEAD), heads, heads, HEAD, false, None)?;
            drop(qkv);
            ops::gemm_bf16(dev, &att, &b.o1.w, b.o1.b.as_ref(), Act::None, Epi::AddRes(&h), &other)?;
            std::mem::swap(&mut h, &mut other);
            // cross-attention over the text
            ops::layernorm(dev, &h, Some(&b.ln2.0), Some(&b.ln2.1), LN_EPS, &normed)?;
            let q = Tensor::new(dev, DType::BF16, &[n, c])?;
            ops::gemm_bf16(dev, &normed, &b.q2, None, Act::None, Epi::Store, &q)?;
            let kvt = kv.next().context("cross-attention keys")?;
            let kview = |off: usize| AttnView { ptr: kvt.ptr + (off * 2) as u64, tok_stride: 2 * c, head_stride: HEAD, len: tokens };
            ops::attention(dev, AttnView::contiguous(&q, heads, HEAD), kview(0), kview(c), AttnView::contiguous(&att, heads, HEAD), heads, heads, HEAD, false, None)?;
            drop(q);
            ops::gemm_bf16(dev, &att, &b.o2.w, b.o2.b.as_ref(), Act::None, Epi::AddRes(&h), &other)?;
            std::mem::swap(&mut h, &mut other);
            // GEGLU feed-forward
            ops::layernorm(dev, &h, Some(&b.ln3.0), Some(&b.ln3.1), LN_EPS, &normed)?;
            let ff = b.ff1.apply(dev, &normed, Act::None)?;
            let inner = ff.shape[1] / 2;
            let g = Tensor::new(dev, DType::BF16, &[n, inner])?;
            dev.launch_n("k_geglu_bf16", n * inner, &[Arg::Ptr(ff.ptr), Arg::Ptr(g.ptr), Arg::I64(n as i64), Arg::I32(inner as i32)])?;
            drop(ff);
            ops::gemm_bf16(dev, &g, &b.ff2.w, b.ff2.b.as_ref(), Act::None, Epi::AddRes(&h), &other)?;
            std::mem::swap(&mut h, &mut other);
        }
        let out = Plane::new(dev, x.h, x.w, c)?;
        ops::gemm_bf16(dev, &h, &xf.proj_out.w, xf.proj_out.b.as_ref(), Act::None, Epi::AddRes(&x.t), &out.t)?;
        Ok(out)
    }
}

/// One text's step-independent UNet state.
pub struct Branch {
    kv: Vec<Tensor>,
    tokens: usize,
    label: Tensor,
}

/// LDM `timestep_embedding`: [cos | sin] of t * exp(-ln(10000) i / (dim/2)).
pub fn sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut e = vec![0f32; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        e[i] = (t * f).cos();
        e[half + i] = (t * f).sin();
    }
    e
}

// ------------------------------------------------------------------------------------------------ schedule
/// ComfyUI's ModelSamplingDiscrete: the 1000 training sigmas of the scaled-linear schedule.
pub struct Discrete {
    pub sigmas: Vec<f32>,
    log_sigmas: Vec<f32>,
}

impl Discrete {
    pub fn new(ztsnr: bool) -> Discrete {
        let n = 1000;
        let (a, b) = (0.00085f64.sqrt(), 0.012f64.sqrt());
        let mut acp = 1.0f64;
        let mut sig: Vec<f64> = (0..n)
            .map(|i| {
                let beta = (a + (b - a) * i as f64 / (n - 1) as f64).powi(2);
                acp *= 1.0 - beta;
                ((1.0 - acp) / acp).sqrt()
            })
            .collect();
        if ztsnr {
            // rescale_zero_terminal_snr_sigmas (computed in float32 by ComfyUI)
            let mut ab: Vec<f64> = sig.iter().map(|s| (1.0 / (s * s + 1.0)).sqrt()).collect();
            let (a0, at) = (ab[0], ab[n - 1]);
            for v in &mut ab {
                *v = (*v - at) * a0 / (a0 - at);
            }
            sig = ab.iter().map(|v| v * v).enumerate().map(|(i, a)| if i == n - 1 { 4.8973451890853435e-08 } else { a }).map(|a| ((1.0 - a) / a).sqrt()).collect();
        }
        let sigmas: Vec<f32> = sig.iter().map(|&s| s as f32).collect();
        let log_sigmas = sigmas.iter().map(|s| s.ln()).collect();
        Discrete { sigmas, log_sigmas }
    }
    pub fn sigma_min(&self) -> f32 {
        self.sigmas[0]
    }
    pub fn sigma_max(&self) -> f32 {
        *self.sigmas.last().unwrap()
    }
    /// The nearest training timestep of a sigma.
    pub fn timestep(&self, sigma: f32) -> f32 {
        let l = sigma.ln();
        let mut best = 0;
        for (i, &ls) in self.log_sigmas.iter().enumerate() {
            if (l - ls).abs() < (l - self.log_sigmas[best]).abs() {
                best = i;
            }
        }
        best as f32
    }
    /// Sigma at a (fractional) timestep, interpolated in log space.
    pub fn sigma(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, (self.sigmas.len() - 1) as f32);
        let (lo, hi) = (t.floor() as usize, t.ceil() as usize);
        let w = t - t.floor();
        ((1.0 - w) * self.log_sigmas[lo] + w * self.log_sigmas[hi]).exp()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Normal,
    Karras,
    SgmUniform,
}
impl Scheduler {
    pub fn parse(s: &str) -> Result<Scheduler> {
        Ok(match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "normal" => Scheduler::Normal,
            "karras" => Scheduler::Karras,
            "sgm_uniform" | "sgm" => Scheduler::SgmUniform,
            o => bail!("unknown --scheduler {o:?} (normal, karras, sgm_uniform)"),
        })
    }
    pub fn id(self) -> &'static str {
        match self {
            Scheduler::Normal => "normal",
            Scheduler::Karras => "karras",
            Scheduler::SgmUniform => "sgm_uniform",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sampler {
    Euler,
    EulerAncestral,
    Dpmpp2m,
}
impl Sampler {
    pub fn parse(s: &str) -> Result<Sampler> {
        Ok(match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "euler" => Sampler::Euler,
            "euler_a" | "euler_ancestral" => Sampler::EulerAncestral,
            "dpmpp_2m" | "dpm++_2m" | "dpmpp2m" => Sampler::Dpmpp2m,
            o => bail!("unknown --sampler {o:?} (euler, euler_ancestral, dpmpp_2m)"),
        })
    }
    pub fn id(self) -> &'static str {
        match self {
            Sampler::Euler => "euler",
            Sampler::EulerAncestral => "euler_ancestral",
            Sampler::Dpmpp2m => "dpmpp_2m",
        }
    }
}

/// `steps` sigmas, then 0 (ComfyUI's `calculate_sigmas`).
pub fn schedule(d: &Discrete, sch: Scheduler, steps: usize) -> Vec<f32> {
    let n = steps.max(1);
    let mut out: Vec<f32> = match sch {
        Scheduler::Karras => {
            let rho = 7.0f64;
            let (lo, hi) = ((d.sigma_min() as f64).powf(1.0 / rho), (d.sigma_max() as f64).powf(1.0 / rho));
            (0..n).map(|i| (hi + i as f64 / (n - 1).max(1) as f64 * (lo - hi)).powf(rho) as f32).collect()
        }
        Scheduler::Normal | Scheduler::SgmUniform => {
            let (start, end) = (d.timestep(d.sigma_max()), d.timestep(d.sigma_min()));
            let k = if sch == Scheduler::SgmUniform { n + 1 } else { n };
            // torch.linspace in float32: start + i * step, the second half counted back from the end
            let lin = |i: usize| -> f32 {
                let step = (end - start) / (k - 1).max(1) as f32;
                if i < k / 2 { start + step * i as f32 } else { end - step * (k - 1 - i) as f32 }
            };
            (0..n).map(|i| d.sigma(lin(i))).collect()
        }
    };
    out.push(0.0);
    out
}

// ------------------------------------------------------------------------------------------------ sampling
/// Everything one image's sampling needs.
pub struct SampleArgs<'a> {
    pub sigmas: &'a [f32],
    pub sampler: Sampler,
    pub cfg: f32,
    pub cond: &'a Branch,
    pub uncond: Option<&'a Branch>,
    /// latent size
    pub h: usize,
    pub w: usize,
    pub seed: u64,
    /// img2img: the encoded image (HWC 4) and the step to start at; inpainting: the latent mask [h*w] (1 = redraw)
    pub x0: Option<&'a [f32]>,
    pub start: usize,
    pub mask: Option<&'a [f32]>,
}

impl Unet {
    /// The denoised estimate at `sigma` (classifier-free guided), as ComfyUI's CFGGuider.
    fn denoise(&self, a: &SampleArgs, d: &Discrete, x: &[f32], sigma: f32, noise: &[f32]) -> Result<Vec<f32>> {
        let npix = a.h * a.w;
        // inpainting: the kept region is the init latent noised to this sigma (KSamplerX0Inpaint)
        let xin: Vec<f32> = match (a.mask, a.x0) {
            (Some(m), Some(x0)) => (0..npix * LC).map(|i| {
                let mm = m[i / LC];
                x[i] * mm + (x0[i] + noise[i] * sigma) * (1.0 - mm)
            }).collect(),
            _ => x.to_vec(),
        };
        let c_in = 1.0 / (sigma * sigma + 1.0).sqrt();
        let mut bits = vec![0u16; npix * LC_PAD];
        for p in 0..npix {
            for c in 0..LC {
                bits[p * LC_PAD + c] = bf16_bits(xin[p * LC + c] * c_in);
            }
        }
        let inp = Plane { t: Tensor::from_bf16(&self.dev, &bits, &[npix, LC_PAD])?, h: a.h, w: a.w, c: LC_PAD };
        let t = d.timestep(sigma);
        let eval = |br: &Branch| -> Result<Vec<f32>> {
            let o = self.forward(br, &inp, t)?;
            let v = o.t.to_f32_vec(&self.dev)?;
            Ok((0..npix * LC).map(|i| v[(i / LC) * LC_PAD + i % LC]).collect())
        };
        let c = eval(a.cond)?;
        let out = match a.uncond {
            Some(u) if a.cfg != 1.0 => {
                let u = eval(u)?;
                c.iter().zip(&u).map(|(c, u)| u + a.cfg * (c - u)).collect()
            }
            _ => c,
        };
        let mut den: Vec<f32> = match self.prediction {
            Prediction::Eps => xin.iter().zip(&out).map(|(x, e)| x - sigma * e).collect(),
            Prediction::V => {
                let s2 = sigma * sigma + 1.0;
                xin.iter().zip(&out).map(|(x, v)| x / s2 - v * sigma / s2.sqrt()).collect()
            }
        };
        if let (Some(m), Some(x0)) = (a.mask, a.x0) {
            for i in 0..npix * LC {
                let mm = m[i / LC];
                den[i] = den[i] * mm + x0[i] * (1.0 - mm);
            }
        }
        Ok(den)
    }

    /// k-diffusion sampling; returns the final latent f32 HWC [h, w, 4].
    pub fn sample(&self, a: &SampleArgs, d: &Discrete, progress: &dyn Fn(usize, usize)) -> Result<Vec<f32>> {
        let (npix, sig) = (a.h * a.w, a.sigmas);
        let n_steps = sig.len() - 1;
        ensure!(a.start < n_steps, "--strength leaves no steps to run with {n_steps} steps");
        // torch.randn([1, 4, h, w]) on the CPU (ComfyUI's noise), to HWC
        let chw = rng::torch_randn(a.seed, LC * npix);
        let noise: Vec<f32> = (0..npix * LC).map(|i| chw[(i % LC) * npix + i / LC]).collect();
        let s0 = sig[a.start];
        let mut x: Vec<f32> = match a.x0 {
            Some(x0) => x0.iter().zip(&noise).map(|(x, n)| x + n * s0).collect(),
            // max denoise: ComfyUI scales the first noise by sqrt(1 + sigma^2)
            None => noise.iter().map(|n| n * (1.0 + s0 * s0).sqrt()).collect(),
        };
        let total = n_steps - a.start;
        let mut old_den: Option<Vec<f32>> = None;
        for (k, i) in (a.start..n_steps).enumerate() {
            crate::worker::check_cancel()?;
            progress(k, total);
            let (s, sn) = (sig[i], sig[i + 1]);
            let den = self.denoise(a, d, &x, s, &noise).with_context(|| format!("step {i}"))?;
            match a.sampler {
                Sampler::Euler => {
                    for j in 0..x.len() {
                        x[j] += (x[j] - den[j]) / s * (sn - s);
                    }
                }
                Sampler::EulerAncestral => {
                    let up = if sn > 0.0 { sn.min((sn * sn * (s * s - sn * sn) / (s * s)).max(0.0).sqrt()) } else { 0.0 };
                    let down = (sn * sn - up * up).max(0.0).sqrt();
                    for j in 0..x.len() {
                        x[j] += (x[j] - den[j]) / s * (down - s);
                    }
                    if sn > 0.0 {
                        let chw = rng::torch_randn(a.seed.wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1)), LC * npix);
                        for j in 0..x.len() {
                            x[j] += chw[(j % LC) * npix + j / LC] * up;
                        }
                    }
                }
                Sampler::Dpmpp2m => {
                    if sn == 0.0 {
                        x.copy_from_slice(&den);
                    } else {
                        let (t, tn) = (-(s.ln()), -(sn.ln()));
                        let h = tn - t;
                        let f = -(-h).exp_m1();
                        let r = sn / s;
                        match (&old_den, i > a.start) {
                            (Some(od), true) => {
                                let r2 = (t - (-(sig[i - 1].ln()))) / h;
                                let (a1, a2) = (1.0 + 1.0 / (2.0 * r2), 1.0 / (2.0 * r2));
                                for j in 0..x.len() {
                                    x[j] = r * x[j] + f * (a1 * den[j] - a2 * od[j]);
                                }
                            }
                            _ => {
                                for j in 0..x.len() {
                                    x[j] = r * x[j] + f * den[j];
                                }
                            }
                        }
                    }
                    old_den = Some(den);
                }
            }
            if k % 4 == 3 {
                self.dev.trim_pool();
            }
        }
        progress(total, total);
        Ok(x)
    }
}

// ------------------------------------------------------------------------------------------------ the run
fn load_unet(dev: &Arc<Device>, p: &Path) -> Result<Rc<Unet>> {
    crate::worker::load(&format!("sdxl unet {}", p.display()), p, || Unet::load(dev.clone(), p))
}
fn load_vae(dev: &Arc<Device>, p: &Path) -> Result<Rc<Vae>> {
    crate::worker::load(&format!("sd vae {}", p.display()), p, || Vae::load_sd(dev.clone(), p))
}

/// Worker warm-up: load the UNet and the VAE.
pub(crate) fn warm(cli: &Cli, ckpt: &Path) -> Result<()> {
    let dev = crate::worker::device()?;
    init_kernels(&dev)?;
    load_vae(&dev, cli.vae.as_deref().unwrap_or(ckpt))?;
    load_unet(&dev, ckpt)?;
    Ok(())
}

fn init_kernels(dev: &Device) -> Result<()> {
    ops::gemm_init(dev)?;
    ops::attn_init(dev)?;
    Vae::init_kernels(dev)
}

/// The sampler and scheduler a run uses: the flags, else euler_ancestral + normal (euler + sgm_uniform for distilled
/// checkpoints).
pub(crate) fn sampling(cli: &Cli, distilled: bool) -> Result<(Sampler, Scheduler)> {
    let sampler = match &cli.sampler {
        Some(s) => Sampler::parse(s)?,
        None if distilled => Sampler::Euler,
        None => Sampler::EulerAncestral,
    };
    let scheduler = match &cli.scheduler {
        Some(s) => Scheduler::parse(s)?,
        None if distilled => Scheduler::SgmUniform,
        None => Scheduler::Normal,
    };
    Ok((sampler, scheduler))
}

pub(crate) fn run(cli: &Cli, plan: &Plan, distilled: bool) -> Result<()> {
    let prog = plan.prog;
    let ((gw, gh), (ow, oh)) = (plan.gen, plan.out);
    let (steps, cfg) = (plan.steps, plan.cfg);
    ensure!(cli.text_encoder.is_none(), "--text-encoder: an SDXL checkpoint carries its own text encoders");
    ensure!(cli.precision.as_deref() != Some("int8"), "--precision int8 is not available for SDXL models (they run in bf16)");
    ensure!(cli.shift.is_none(), "--shift is a flow-matching setting; SDXL models use --scheduler");
    let (sampler, scheduler) = sampling(cli, distilled)?;
    let ckpt = &plan.dit;
    let vae_path = cli.vae.clone().unwrap_or_else(|| ckpt.clone());
    crate::info!("{prog}: SDXL {}{}, {} image(s), {}x{}{}, {} steps, CFG {}, {} / {}", cli::file_stem(ckpt), if distilled { " (distilled)" } else { "" }, plan.jobs.len(), ow, oh, if (gw, gh) != (ow, oh) { format!(" (generated at {gw}x{gh})") } else { String::new() }, steps, cfg, sampler.id(), scheduler.id());
    crate::info!("  checkpoint: {}{}", ckpt.display(), if cli.vae.is_some() { format!("\n  vae: {}", vae_path.display()) } else { String::new() });
    let t_dev = std::time::Instant::now();
    let dev = crate::worker::device()?;
    crate::info!("  gpu: {} MB free of {} MB (cuda init {:.1}s)", dev.free_mem()? >> 20, dev.total_mem >> 20, t_dev.elapsed().as_secs_f64());
    init_kernels(&dev)?;
    let t_all = std::time::Instant::now();

    // ---- phase 1: both text encoders over every prompt and the negative
    let t0 = std::time::Instant::now();
    let mut texts: Vec<String> = plan.prompts.clone();
    if let Some(n) = &plan.negative {
        texts.push(n.clone());
    }
    let conds: Vec<TextCond> = {
        let st = SafeTensors::open(ckpt)?;
        let te = SdxlText::load(dev.clone(), &st).context("SDXL text encoders")?;
        let tok = ClipTokenizer::new()?;
        let c = texts.iter().map(|t| te.encode(&tok, t)).collect::<Result<Vec<_>>>()?;
        drop(te);
        st.drop_in_background();
        c
    };
    crate::info!("  text encoders: {} text(s), {:.1}s", texts.len(), t0.elapsed().as_secs_f64());

    // ---- phase 2: VAE; encode the init image
    let vae = load_vae(&dev, &vae_path)?;
    let (lh, lw) = (gh / 8, gw / 8);
    let x0 = match &plan.init_gen {
        Some(img) => Some(vae.encode_sd(&img.to_f32(), gh, gw).context("vae encode")?),
        None => None,
    };
    let mask_lat = plan.mask_gen.as_ref().map(|m| m.max_down(8).data.clone());

    // ---- phase 3: UNet; sample, decode, write each image as soon as it is ready
    let unet = load_unet(&dev, ckpt)?;
    let disc = Discrete::new(unet.ztsnr);
    let sigmas = schedule(&disc, scheduler, steps);
    let start = if x0.is_some() { crate::sampler::img2img_start(steps, cli.strength) } else { 0 };
    let uncond = match &plan.negative {
        Some(_) => Some(unet.prepare(&conds[plan.prompts.len()], gh, gw)?),
        None => None,
    };
    let mut branches: std::collections::HashMap<usize, Branch> = std::collections::HashMap::new();
    let mut results: Vec<serde_json::Value> = Vec::new();
    for (ji, job) in plan.jobs.iter().enumerate() {
        if !branches.contains_key(&job.prompt) {
            branches.insert(job.prompt, unet.prepare(&conds[job.prompt], gh, gw)?);
        }
        crate::info!("image {}/{}: seed {} {}x{}", ji + 1, plan.jobs.len(), job.seed, ow, oh);
        let t0 = std::time::Instant::now();
        let args = SampleArgs { sigmas: &sigmas, sampler, cfg, cond: &branches[&job.prompt], uncond: uncond.as_ref(), h: lh, w: lw, seed: job.seed, x0: x0.as_deref(), start, mask: mask_lat.as_deref() };
        let (bar, show_bar) = cli::step_progress(cli, steps);
        let lat = unet.sample(&args, &disc, &|n, total| {
            bar.set_length(total as u64);
            bar.set_position(n as u64);
            if !show_bar {
                crate::info!("step {n}/{total}");
            }
        })?;
        bar.finish_and_clear();
        let t_sample = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let rgb = vae.decode_sd(&lat, lh, lw).context("vae decode")?;
        crate::info!("  sampled in {:.1}s, decoded in {:.1}s", t_sample, t1.elapsed().as_secs_f64());
        let mut rec = genmeta::Record::new(prog, env!("CARGO_PKG_VERSION"), &cli::file_stem(ckpt));
        rec.set("base_model", "SDXL")
            .set("architecture", "sdxl")
            .set("prompt", plan.prompts[job.prompt].clone())
            .set("negative", plan.negative.clone())
            .set("seed", job.seed)
            .set("steps", steps)
            .set("cfg", cfg as f64)
            .set("turbo", if distilled { Some(true) } else { None })
            .set("sampler", sampler.id())
            .set("scheduler", scheduler.id())
            .set("prediction", if unet.prediction == Prediction::V { Some("v") } else { None })
            .set("precision", "bf16")
            .set("vae", cli.vae.as_ref().and_then(|v| v.file_name()).map(|f| f.to_string_lossy().into_owned()));
        cli::deliver(cli, plan, job, Rgb8 { w: gw, h: gh, data: rgb }, &mut rec, &mut results)?;
    }
    cli::finish_json(cli, plan, results, distilled);
    drop(unet);
    dev.sync()?;
    crate::info!("done in {:.1}s", t_all.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_schedule_matches_sdxl() {
        let d = Discrete::new(false);
        // ComfyUI / diffusers: sigma_min 0.0292, sigma_max 14.6146
        assert!((d.sigma_min() - 0.029168).abs() < 1e-4, "{}", d.sigma_min());
        assert!((d.sigma_max() - 14.614642).abs() < 1e-3, "{}", d.sigma_max());
        assert_eq!(d.timestep(d.sigma_max()), 999.0);
        assert_eq!(d.timestep(d.sigma(500.0)), 500.0);
        let s = schedule(&d, Scheduler::Normal, 4);
        assert_eq!(s.len(), 5);
        assert!((s[0] - d.sigma_max()).abs() < 1e-3 && s[4] == 0.0);
        let k = schedule(&d, Scheduler::Karras, 10);
        assert!((k[0] - d.sigma_max()).abs() < 1e-3 && (k[9] - d.sigma_min()).abs() < 1e-5);
        let z = Discrete::new(true);
        assert!(z.sigma_max() > 4000.0, "{}", z.sigma_max());
    }

    #[test]
    fn samplers_and_schedulers_parse() {
        assert_eq!(Sampler::parse("euler_a").unwrap(), Sampler::EulerAncestral);
        assert_eq!(Sampler::parse("DPMPP-2M").unwrap(), Sampler::Dpmpp2m);
        assert_eq!(Scheduler::parse("sgm_uniform").unwrap(), Scheduler::SgmUniform);
        assert!(Sampler::parse("ddim").is_err());
    }

    #[test]
    fn distilled_by_name() {
        assert!(is_distilled(Path::new("sdxl_lightning_8step.safetensors")));
        assert!(!is_distilled(Path::new("sd_xl_base_1.0.safetensors")));
    }
}
