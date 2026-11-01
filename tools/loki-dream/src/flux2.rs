//! FLUX.2 klein transformer (4B: dim 3072, 5 double + 20 single blocks; 9B: dim 4096, 8 + 24), from the BFL / ComfyUI
//! single files (`double_blocks.N.img_attn.qkv`, `single_blocks.N.linear1`, ...). Follows diffusers'
//! `Flux2Transformer2DModel`:
//!   - text: Qwen3 hidden states 9, 18, 27 concatenated per token (3 x 2560 or 3 x 4096) -> `txt_in`; image tokens are
//!     the 128-channel FLUX.2 latents (the same latent space and BN standardization as MOSS-Image) -> `img_in`. Sequence
//!     [text, image].
//!   - time: sinusoid(256) of t = sigma * 1000 -> Linear-SiLU-Linear; shared modulations (shift/scale/gate) for the double
//!     blocks' image and text streams and for the single blocks; LayerNorm without affine (eps 1e-6) then
//!     `(1 + scale) * x + shift`; no biases anywhere.
//!   - double block: per-stream qkv, q/k RMSNorm, joint attention over [text, image], per-stream proj and SwiGLU MLP.
//!   - single block: parallel attention + MLP (`linear1` -> [q k v | gate up], `linear2` on [attention | swiglu]).
//!   - rope: 4 axes (t, h, w, l) x 32 dims, theta 2000, interleaved pairs; text token i at (0, 0, 0, i), image token at
//!     (0, row, col, 0).
//!   - output: final LayerNorm modulated by (shift, scale) -> Linear(dim -> 128) on the image tokens.
use crate::cuda::Device;
use crate::linear::{self, Lin};
use crate::model::{self, Dense};
use crate::ops::{self, Act, AttnView, Epi, ModNorm, QkNorm, Rope};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const LATENT: usize = 128;
const EPS: f32 = 1e-6;
const HD: usize = 128;
/// The Qwen3 hidden states klein conditions on.
pub const TAPS: [usize; 3] = [9, 18, 27];
pub const MAX_TEXT: usize = 512;

struct Stream {
    qkv: Lin,
    qn: Tensor,
    kn: Tensor,
    proj: Lin,
    mlp_in: Lin,
    mlp_out: Lin,
}
struct Double {
    img: Stream,
    txt: Stream,
}
struct Single {
    lin1: Lin,
    lin2: Lin,
    qn: Tensor,
    kn: Tensor,
}

pub struct Flux2 {
    dev: Arc<Device>,
    pub int8: bool,
    pub dim: usize,
    pub heads: usize,
    pub mlp: usize,
    pub text_dim: usize,
    img_in: Tensor,
    txt_in: Lin,
    t1: Dense,
    t2: Dense,
    mod_img: Dense,
    mod_txt: Dense,
    mod_single: Dense,
    final_mod: Dense,
    final_linear: Tensor,
    doubles: Vec<Double>,
    singles: Vec<Single>,
}

pub struct Run {
    /// text tokens after `txt_in` [T, dim] bf16
    txt: Tensor,
    /// rope [T + N, 64, 2]
    rope: Tensor,
    pub t_len: usize,
    pub h: usize,
    pub w: usize,
}

pub struct Mod {
    img: Vec<Tensor>,
    txt: Vec<Tensor>,
    single: Vec<Tensor>,
    fin: Vec<Tensor>,
}

impl Flux2 {
    pub fn load(dev: Arc<Device>, path: &Path, int8: bool) -> Result<Flux2> {
        let st = SafeTensors::open(path)?;
        let p = crate::arch::prefix_of(&st, "double_stream_modulation_img.lin.weight").with_context(|| format!("{} is not a FLUX.2 model", path.display()))?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let n = |s: &str| format!("{p}{s}");
        let img_in = linear::weight_bf16(&mut l, &n("img_in"))?;
        let dim = img_in.shape[0];
        ensure!(img_in.shape[1] == LATENT, "FLUX.2: expected {LATENT} latent channels, got {}", img_in.shape[1]);
        let w = model::want(int8);
        let txt_in = linear::lin(&mut l, &n("txt_in"), w)?;
        let mut doubles = Vec::new();
        while st.has(&n(&format!("double_blocks.{}.img_attn.norm.query_norm.scale", doubles.len()))) {
            let i = doubles.len();
            let stream = |l: &mut Loader, s: &str| -> Result<Stream> {
                let b = n(&format!("double_blocks.{i}.{s}"));
                Ok(Stream {
                    qkv: linear::lin(l, &format!("{b}_attn.qkv"), w)?,
                    qn: linear::small_bf16(l, &format!("{b}_attn.norm.query_norm.scale"))?,
                    kn: linear::small_bf16(l, &format!("{b}_attn.norm.key_norm.scale"))?,
                    proj: linear::lin(l, &format!("{b}_attn.proj"), w)?,
                    mlp_in: linear::lin(l, &format!("{b}_mlp.0"), w)?,
                    mlp_out: linear::lin(l, &format!("{b}_mlp.2"), w)?,
                })
            };
            doubles.push(Double { img: stream(&mut l, "img")?, txt: stream(&mut l, "txt")? });
        }
        let mut singles = Vec::new();
        while st.has(&n(&format!("single_blocks.{}.norm.query_norm.scale", singles.len()))) {
            let b = n(&format!("single_blocks.{}", singles.len()));
            singles.push(Single {
                lin1: linear::lin(&mut l, &format!("{b}.linear1"), w)?,
                lin2: linear::lin(&mut l, &format!("{b}.linear2"), w)?,
                qn: linear::small_bf16(&mut l, &format!("{b}.norm.query_norm.scale"))?,
                kn: linear::small_bf16(&mut l, &format!("{b}.norm.key_norm.scale"))?,
            });
        }
        ensure!(!doubles.is_empty() && !singles.is_empty(), "FLUX.2: no transformer blocks found");
        let mlp = doubles[0].img.mlp_in.n() / 2;
        let text_dim = txt_in.k();
        let f = Flux2 {
            int8,
            dim,
            heads: dim / HD,
            mlp,
            text_dim,
            img_in,
            txt_in,
            t1: Dense::load(&mut l, &n("time_in.in_layer"))?,
            t2: Dense::load(&mut l, &n("time_in.out_layer"))?,
            mod_img: Dense::load(&mut l, &n("double_stream_modulation_img.lin"))?,
            mod_txt: Dense::load(&mut l, &n("double_stream_modulation_txt.lin"))?,
            mod_single: Dense::load(&mut l, &n("single_stream_modulation.lin"))?,
            final_mod: Dense::load(&mut l, &n("final_layer.adaLN_modulation.1"))?,
            final_linear: linear::weight_bf16(&mut l, &n("final_layer.linear"))?,
            doubles,
            singles,
            dev,
        };
        ensure!(!st.has(&n("guidance_in.in_layer.weight")), "FLUX.2 dev (guidance-distilled, Mistral text encoder) is not supported; FLUX.2 klein is");
        crate::info!("  dit: FLUX.2 klein, dim {}, {} double + {} single blocks, {} {:.1} GB uploaded in {:.1}s", f.dim, f.doubles.len(), f.singles.len(), if int8 { "int8" } else { "bf16" }, l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(f)
    }

    /// The text encoder hidden size this model expects (txt_in K / 3).
    pub fn te_hidden(&self) -> usize {
        self.text_dim / TAPS.len()
    }

    /// `ctx` [T, 3 * te_hidden] bf16 (the tapped hidden states concatenated per token).
    pub fn prepare(&self, ctx: &Tensor, h: usize, w: usize) -> Result<Run> {
        let dev = &self.dev;
        let t = ctx.shape[0];
        ensure!(ctx.shape[1] == self.text_dim, "FLUX.2: text features {} wide, the model wants {}", ctx.shape[1], self.text_dim);
        let txt = Tensor::new(dev, DType::BF16, &[t, self.dim])?;
        self.txt_in.forward(dev, ctx, None, Epi::Store, &txt)?;
        let freqs: Vec<f64> = (0..16).map(|k| 1.0 / 2000f64.powf((2 * k) as f64 / 32.0)).collect();
        let rope = model::rope_table(dev, t + h * w, |tok, f| {
            let (axis, k) = (f / 16, f % 16);
            let pos = if tok < t {
                [0.0, 0.0, 0.0, tok as f64]
            } else {
                let i = tok - t;
                [0.0, (i / w) as f64, (i % w) as f64, 0.0]
            };
            // diffusers computes the angles in f64 and casts cos/sin to f32
            pos[axis] * freqs[k]
        })?;
        Ok(Run { txt, rope, t_len: t, h, w })
    }
}

impl model::Model for Flux2 {
    type Run = Run;
    type Mod = Mod;
    fn channels(&self) -> usize {
        LATENT
    }

    fn modulation(&self, sigma: f32) -> Result<Mod> {
        let dev = &self.dev;
        // pipeline: t = sigma * 1000 -> bf16; transformer(timestep = t / 1000) -> * 1000, each a bf16 op
        let t = model::round_bf16(model::round_bf16(model::round_bf16(sigma * 1000.0) / 1000.0) * 1000.0);
        let e = model::sinusoid_bf16(dev, t, 256)?;
        let temb = self.t2.apply(dev, &self.t1.apply(dev, &e, Act::Silu)?, Act::None)?;
        let s = model::silu_copy(dev, &temb)?;
        Ok(Mod {
            img: model::split_row(dev, &self.mod_img.apply(dev, &s, Act::None)?, 6)?,
            txt: model::split_row(dev, &self.mod_txt.apply(dev, &s, Act::None)?, 6)?,
            single: model::split_row(dev, &self.mod_single.apply(dev, &s, Act::None)?, 3)?,
            fin: model::split_row(dev, &self.final_mod.apply(dev, &s, Act::None)?, 2)?,
        })
    }

    fn forward(&self, run: &Run, x: &Tensor, m: &Mod, out: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let (t, ni) = (run.t_len, run.h * run.w);
        let s = t + ni;
        let (d, nh, mlp) = (self.dim, self.heads, self.mlp);
        ensure!(x.shape[0] == ni, "latent tokens {} != {}", x.shape[0], ni);
        let h = Tensor::new(dev, DType::BF16, &[s, d])?;
        dev.dtod(h.ptr, run.txt.ptr, run.txt.bytes())?;
        ops::gemm_bf16(dev, x, &self.img_in, None, Act::None, Epi::Store, &h.rows(t, ni))?;
        let n = Tensor::new(dev, DType::BF16, &[s, d])?;
        let a = Tensor::new(dev, DType::BF16, &[s, d])?;
        let o = Tensor::new(dev, DType::BF16, &[s, d])?;
        // ---- double blocks: [shift, scale, gate] x [attention, mlp] per stream
        {
            let qkv = Tensor::new(dev, DType::BF16, &[s, 3 * d])?;
            let ff = Tensor::new(dev, DType::BF16, &[s, 2 * mlp])?;
            let (mi, mt) = (&m.img, &m.txt);
            for b in &self.doubles {
                ops::norm_mod(dev, &h, ModNorm::Layer, EPS, t, (&mt[1], &mt[0]), (&mi[1], &mi[0]), &n)?;
                for (st, r0, rn) in [(&b.txt, 0, t), (&b.img, t, ni)] {
                    st.qkv.forward(dev, &n.rows(r0, rn), None, Epi::Store, &qkv.rows(r0, rn))?;
                    let q = qkv.rows(r0, rn);
                    ops::qk_norm_rope(dev, q.ptr, 3 * d, 0, d, rn, nh, nh, Some(&st.qn), Some(&st.kn), EPS, QkNorm::Hf, Rope::Pairs32, Some(&run.rope.rows(r0, rn)))?;
                }
                let view = |off: usize| AttnView { ptr: qkv.ptr + (off * 2) as u64, tok_stride: 3 * d, head_stride: HD, len: s };
                ops::attention(dev, view(0), view(d), view(2 * d), AttnView::contiguous(&a, nh, HD), nh, nh, HD, false, None)?;
                for (st, r0, rn, mm) in [(&b.txt, 0, t, mt), (&b.img, t, ni, mi)] {
                    st.proj.forward(dev, &a.rows(r0, rn), None, Epi::Store, &o.rows(r0, rn))?;
                    ops::gated_add(dev, &h.rows(r0, rn), &o.rows(r0, rn), &mm[2])?;
                }
                ops::norm_mod(dev, &h, ModNorm::Layer, EPS, t, (&mt[4], &mt[3]), (&mi[4], &mi[3]), &n)?;
                for (st, r0, rn, mm) in [(&b.txt, 0, t, mt), (&b.img, t, ni, mi)] {
                    st.mlp_in.forward(dev, &n.rows(r0, rn), None, Epi::Store, &ff.rows(r0, rn))?;
                    st.mlp_out.forward_swiglu(dev, &ff.rows(r0, rn), Epi::Store, &o.rows(r0, rn))?;
                    ops::gated_add(dev, &h.rows(r0, rn), &o.rows(r0, rn), &mm[5])?;
                }
            }
        }
        // ---- single blocks: one modulation (shift, scale, gate)
        {
            let w1 = 3 * d + 2 * mlp;
            let p = Tensor::new(dev, DType::BF16, &[s, w1])?;
            let c = Tensor::new(dev, DType::BF16, &[s, d + mlp])?;
            let ms = &m.single;
            for b in &self.singles {
                ops::norm_mod(dev, &h, ModNorm::Layer, EPS, s, (&ms[1], &ms[0]), (&ms[1], &ms[0]), &n)?;
                b.lin1.forward(dev, &n, None, Epi::Store, &p)?;
                ops::qk_norm_rope(dev, p.ptr, w1, 0, d, s, nh, nh, Some(&b.qn), Some(&b.kn), EPS, QkNorm::Hf, Rope::Pairs32, Some(&run.rope))?;
                let view = |off: usize| AttnView { ptr: p.ptr + (off * 2) as u64, tok_stride: w1, head_stride: HD, len: s };
                ops::attention(dev, view(0), view(d), view(2 * d), AttnView { ptr: c.ptr, tok_stride: d + mlp, head_stride: HD, len: s }, nh, nh, HD, false, None)?;
                ops::swiglu_strided(dev, p.ptr, w1, 3 * d, c.ptr + (d * 2) as u64, d + mlp, s, mlp)?;
                b.lin2.forward(dev, &c, None, Epi::Store, &o)?;
                ops::gated_add(dev, &h, &o, &ms[2])?;
            }
        }
        // ---- output: (shift, scale) as stored in the BFL file
        let xo = h.rows(t, ni);
        let y = n.rows(0, ni);
        ops::norm_mod(dev, &xo, ModNorm::Layer, EPS, ni, (&m.fin[1], &m.fin[0]), (&m.fin[1], &m.fin[0]), &y)?;
        ops::gemm_bf16(dev, &y, &self.final_linear, None, Act::None, Epi::Store, out)?;
        Ok(())
    }
}
