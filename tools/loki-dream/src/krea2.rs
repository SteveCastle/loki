//! Krea 2 transformer (12B single-stream MMDiT: 28 blocks, dim 6144, 48 query / 12 key-value heads of 128, SwiGLU 16384),
//! from the ComfyUI single files (`blocks.N.attn.wq`, `txtfusion.*`, `tproj.1`, ...). Follows diffusers'
//! `Krea2Transformer2DModel`:
//!   - text: 12 hidden states of Qwen3-VL-4B (layers 2, 5, ..., 35) per token -> text fusion: 2 "layer-wise" blocks attend
//!     across the 12 taps of each token, a Linear(12 -> 1) collapses them, 2 refiner blocks attend across the tokens ->
//!     RMSNorm + Linear-GELU-Linear to the model width. Independent of the timestep: computed once per prompt.
//!   - image: 16-channel Qwen-Image latents in 2x2 patches (64 channels) -> `first`. Sequence [text, image].
//!   - time: sinusoid(256) of t = sigma * 1000 -> Linear-GELU-Linear = temb; one shared modulation GELU(temb) -> Linear
//!     (6 x dim), to which each block adds its own learned table (pre/post scale, shift, gate).
//!   - block: x += gate_a * wo(attn(rope(qk_norm(q, k)), v) * sigmoid(g)); x += gate_f * down(silu(gate) * up), inputs
//!     `(1 + scale) * rms(x) + shift`; every RMSNorm is zero-centred (1 + w, eps 1e-5).
//!   - rope: 3 axes (t, h, w) of 32 / 48 / 48 dims, theta 1000, interleaved pairs; text at (0, 0, 0), image at (0, row, col).
//!   - output: (1 + scale) * rms(x) + shift with temb + the final table -> Linear(dim -> 64) (+ bias).
//! The reference pads the prompt to 512 tokens and masks the padding everywhere; dropping it is exactly equivalent.
use crate::cuda::Device;
use crate::linear::{self, Lin, Want};
use crate::model::{self, Dense};
use crate::ops::{self, Act, AttnView, Epi, ModNorm, QkNorm, RmsMode, Rope};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const LATENT: usize = 64;
const EPS: f32 = 1e-5;
const HD: usize = 128;
/// Qwen3-VL-4B hidden states Krea 2 conditions on (index 0 = embeddings).
pub const TAPS: [usize; 12] = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35];

struct Attn {
    /// [q | k | v]
    qkv: Lin,
    /// the output gate's projection (kept apart: quantizers mark it precision-sensitive)
    gate: Lin,
    qn: Tensor,
    kn: Tensor,
    wo: Lin,
    hq: usize,
    hk: usize,
}
struct Block {
    pre: Tensor,
    post: Tensor,
    attn: Attn,
    gate_up: Lin,
    down: Lin,
    /// the block's modulation table [6, dim] (host, bf16 values)
    table: Vec<f32>,
}

pub struct Krea2 {
    dev: Arc<Device>,
    pub prof: crate::cuda::Profiler,
    pub int8: bool,
    pub dim: usize,
    pub text_dim: usize,
    first: Dense,
    t1: Dense,
    t2: Dense,
    tproj: Dense,
    txt_norm: Tensor,
    txt1: Dense,
    txt2: Dense,
    layerwise: Vec<Block>,
    projector: Tensor,
    refiner: Vec<Block>,
    blocks: Vec<Block>,
    last_norm: Tensor,
    last_linear: Dense,
    last_table: Vec<f32>,
}

pub struct Run {
    /// the projected text tokens [T, dim] (step-independent)
    pub txt: Tensor,
    rope: Tensor,
    pub t_len: usize,
    pub h: usize,
    pub w: usize,
}

pub struct Mod {
    /// per block: prescale, preshift, pregate, postscale, postshift, postgate
    blocks: Vec<Vec<Tensor>>,
    last_scale: Tensor,
    last_shift: Tensor,
}

fn load_block(l: &mut Loader, b: &str, want: Want, table: bool) -> Result<Block> {
    let a = format!("{b}.attn");
    let qkv = linear::lin_cat(l, &[&format!("{a}.wq"), &format!("{a}.wk"), &format!("{a}.wv")], want)?;
    let gate = linear::lin(l, &format!("{a}.gate"), want)?;
    let q = l.st.info(&format!("{a}.wq.weight"))?.shape[0];
    let k = l.st.info(&format!("{a}.wk.weight"))?.shape[0];
    let attn = Attn {
        qkv,
        gate,
        qn: linear::small_bf16(l, &format!("{a}.qknorm.qnorm.scale"))?,
        kn: linear::small_bf16(l, &format!("{a}.qknorm.knorm.scale"))?,
        wo: linear::lin(l, &format!("{a}.wo"), want)?,
        hq: q / HD,
        hk: k / HD,
    };
    Ok(Block {
        pre: linear::small_bf16(l, &format!("{b}.prenorm.scale"))?,
        post: linear::small_bf16(l, &format!("{b}.postnorm.scale"))?,
        attn,
        gate_up: linear::lin_cat(l, &[&format!("{b}.mlp.gate"), &format!("{b}.mlp.up")], want)?,
        down: linear::lin(l, &format!("{b}.mlp.down"), want)?,
        table: if table { l.st.f32s(&format!("{b}.mod.lin"))? } else { Vec::new() },
    })
}

impl Krea2 {
    pub fn load(dev: Arc<Device>, path: &Path, int8: bool) -> Result<Krea2> {
        let st = SafeTensors::open(path)?;
        let p = crate::arch::prefix_of(&st, "txtfusion.projector.weight").with_context(|| format!("{} is not a Krea 2 model", path.display()))?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let n = |s: &str| format!("{p}{s}");
        let w = model::want(int8);
        let mut blocks = Vec::new();
        while st.has(&n(&format!("blocks.{}.mod.lin", blocks.len()))) {
            let b = n(&format!("blocks.{}", blocks.len()));
            blocks.push(load_block(&mut l, &b, w, true)?);
        }
        let fusion = |l: &mut Loader, kind: &str| -> Result<Vec<Block>> {
            let mut v = Vec::new();
            while l.st.has(&n(&format!("txtfusion.{kind}.{}.prenorm.scale", v.len()))) {
                let b = n(&format!("txtfusion.{kind}.{}", v.len()));
                v.push(load_block(l, &b, Want::Bf16, false)?);
            }
            Ok(v)
        };
        let layerwise = fusion(&mut l, "layerwise_blocks")?;
        let refiner = fusion(&mut l, "refiner_blocks")?;
        let first = Dense::load(&mut l, &n("first"))?;
        let dim = first.n();
        let projector = l.f32(&n("txtfusion.projector.weight"))?;
        ensure!(projector.numel() == TAPS.len(), "Krea 2: the text projector mixes {} layers, expected {}", projector.numel(), TAPS.len());
        let k = Krea2 {
            prof: crate::cuda::Profiler::new(),
            int8,
            dim,
            text_dim: linear::small_bf16(&mut l, &n("txtmlp.0.scale"))?.numel(),
            first,
            t1: Dense::load(&mut l, &n("tmlp.0"))?,
            t2: Dense::load(&mut l, &n("tmlp.2"))?,
            tproj: Dense::load(&mut l, &n("tproj.1"))?,
            txt_norm: linear::small_bf16(&mut l, &n("txtmlp.0.scale"))?,
            txt1: Dense::load(&mut l, &n("txtmlp.1"))?,
            txt2: Dense::load(&mut l, &n("txtmlp.3"))?,
            layerwise,
            projector,
            refiner,
            blocks,
            last_norm: linear::small_bf16(&mut l, &n("last.norm.scale"))?,
            last_linear: Dense::load(&mut l, &n("last.linear"))?,
            last_table: l.st.f32s(&n("last.modulation.lin"))?,
            dev,
        };
        ensure!(k.first.w.shape[1] == LATENT, "Krea 2: expected {LATENT} latent channels");
        crate::info!("  dit: Krea 2, {} blocks, dim {}, {} {:.1} GB uploaded in {:.1}s", k.blocks.len(), k.dim, if int8 { "int8" } else { "bf16" }, l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(k)
    }

    /// One pre-norm block (`m`: modulation of a DiT block, None for the text fusion's plain residual blocks).
    /// `rope` and `groups` select the attention: rope over the whole sequence, or attention within groups of tokens.
    #[allow(clippy::too_many_arguments)]
    fn block(&self, x: &Tensor, b: &Block, m: Option<&[Tensor]>, rope: Option<&Tensor>, groups: Option<usize>) -> Result<()> {
        let dev = &self.dev;
        let (s, d) = (x.shape[0], x.shape[1]);
        let a = &b.attn;
        let (hq, hk) = (a.hq, a.hk);
        let w = a.qkv.n();
        let n = Tensor::new(dev, DType::BF16, &[s, d])?;
        let pr = &self.prof;
        pr.time(dev, "norm", || match m {
            Some(m) => ops::norm_mod(dev, x, ModNorm::RmsOnePlus(&b.pre), EPS, s, (&m[0], &m[1]), (&m[0], &m[1]), &n),
            None => ops::rmsnorm_x(dev, x, Some(&b.pre), RmsMode::OnePlus, EPS, &n),
        })?;
        let p = Tensor::new(dev, DType::BF16, &[s, w])?;
        pr.time(dev, if a.qkv.is_int8() { "qkv i8" } else { "qkv bf16" }, || a.qkv.forward(dev, &n, None, Epi::Store, &p))?;
        let g = Tensor::new(dev, DType::BF16, &[s, a.gate.n()])?;
        pr.time(dev, if a.gate.is_int8() { "gate proj i8" } else { "gate proj bf16" }, || a.gate.forward(dev, &n, None, Epi::Store, &g))?;
        let (k_off, v_off) = (hq * HD, (hq + hk) * HD);
        pr.time(dev, "qk norm rope", || ops::qk_norm_rope(dev, p.ptr, w, 0, k_off, s, hq, hk, Some(&a.qn), Some(&a.kn), EPS, QkNorm::OnePlus, if rope.is_some() { Rope::Pairs32 } else { Rope::None }, rope))?;
        let at = Tensor::new(dev, DType::BF16, &[s, hq * HD])?;
        match groups {
            Some(g) => ops::attn_groups(dev, &p, 0, k_off, v_off, hq, g, &at)?,
            None => {
                let view = |off: usize| AttnView { ptr: p.ptr + (off * 2) as u64, tok_stride: w, head_stride: HD, len: s };
                pr.time(dev, "attention", || ops::attention(dev, view(0), view(k_off), view(v_off), AttnView::contiguous(&at, hq, HD), hq, hk, HD, false, None))?;
            }
        }
        pr.time(dev, "gate", || ops::sigmoid_gate(dev, &at, g.ptr, a.gate.n()))?;
        let o = Tensor::new(dev, DType::BF16, &[s, d])?;
        pr.time(dev, if a.wo.is_int8() { "wo i8" } else { "wo bf16" }, || a.wo.forward(dev, &at, None, Epi::Store, &o))?;
        drop((at, p, g));
        match m {
            Some(m) => ops::gated_add(dev, x, &o, &m[2])?,
            None => ops::add_inplace(dev, x, &o)?,
        }
        match m {
            Some(m) => ops::norm_mod(dev, x, ModNorm::RmsOnePlus(&b.post), EPS, s, (&m[3], &m[4]), (&m[3], &m[4]), &n)?,
            None => ops::rmsnorm_x(dev, x, Some(&b.post), RmsMode::OnePlus, EPS, &n)?,
        }
        let f = b.down.k();
        let gu = Tensor::new(dev, DType::BF16, &[s, 2 * f])?;
        pr.time(dev, "gate_up", || b.gate_up.forward(dev, &n, None, Epi::Store, &gu))?;
        pr.time(dev, "swiglu + down", || b.down.forward_swiglu(dev, &gu, Epi::Store, &o))?;
        drop(gu);
        match m {
            Some(m) => ops::gated_add(dev, x, &o, &m[5]),
            None => ops::add_inplace(dev, x, &o),
        }
    }

    /// `taps` [T * 12, text_dim] bf16 (token-major, the 12 hidden states of each token consecutive).
    pub fn prepare(&self, taps: &Tensor, h: usize, w: usize) -> Result<Run> {
        let dev = &self.dev;
        let l = TAPS.len();
        let t = taps.shape[0] / l;
        ensure!(t > 0 && taps.shape[1] == self.text_dim, "Krea 2: bad text features {:?}", taps.shape);
        let x = Tensor::new(dev, DType::BF16, &taps.shape)?;
        dev.dtod(x.ptr, taps.ptr, taps.bytes())?;
        for b in &self.layerwise {
            self.block(&x, b, None, None, Some(l))?;
        }
        let y = Tensor::new(dev, DType::BF16, &[t, self.text_dim])?;
        ops::layer_mix(dev, &x, &self.projector, l, &y)?;
        drop(x);
        for b in &self.refiner {
            self.block(&y, b, None, None, None)?;
        }
        let n = Tensor::new(dev, DType::BF16, &y.shape)?;
        ops::rmsnorm_x(dev, &y, Some(&self.txt_norm), RmsMode::OnePlus, EPS, &n)?;
        let txt = self.txt2.apply(dev, &self.txt1.apply(dev, &n, Act::GeluTanh)?, Act::None)?;
        // rope: (t, h, w) axes of 16 / 24 / 24 frequencies, theta 1000
        let axes = [(0usize, 32usize), (16, 48), (40, 48)];
        let rope = model::rope_table(dev, t + h * w, |tok, f| {
            let (ax, (f0, dim)) = axes.iter().enumerate().rev().find(|(_, (f0, _))| f >= *f0).map(|(i, a)| (i, *a)).unwrap();
            let k = f - f0;
            let freq = 1.0 / 1000f64.powf((2 * k) as f64 / dim as f64);
            let pos = if tok < t { 0.0 } else if ax == 1 { ((tok - t) / w) as f64 } else if ax == 2 { ((tok - t) % w) as f64 } else { 0.0 };
            pos * freq
        })?;
        Ok(Run { txt, rope, t_len: t, h, w })
    }
}

impl model::Model for Krea2 {
    type Run = Run;
    type Mod = Mod;
    fn channels(&self) -> usize {
        LATENT
    }

    fn modulation(&self, sigma: f32) -> Result<Mod> {
        let dev = &self.dev;
        // pipeline: timestep = (t / 1000).to(bf16); time_embed: timestep.float() * 1e3
        let e = model::sinusoid_bf16(dev, model::round_bf16(sigma) * 1000.0, 256)?;
        let temb = self.t2.apply(dev, &self.t1.apply(dev, &e, Act::GeluTanh)?, Act::None)?;
        let g = Tensor::new(dev, DType::BF16, &temb.shape)?;
        dev.dtod(g.ptr, temb.ptr, temb.bytes())?;
        ops::act_inplace(dev, &g, Act::GeluTanh)?;
        let tm = self.tproj.apply(dev, &g, Act::None)?.to_f32_vec(dev)?;
        let te = temb.to_f32_vec(dev)?;
        let d = self.dim;
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let v: Vec<f32> = tm.iter().zip(&b.table).map(|(a, t)| model::round_bf16(a + t)).collect();
            blocks.push((0..6).map(|i| Tensor::from_f32(dev, &v[i * d..(i + 1) * d], &[d])).collect::<Result<Vec<_>>>()?);
        }
        let fin: Vec<f32> = (0..2 * d).map(|i| model::round_bf16(te[i % d] + self.last_table[i])).collect();
        Ok(Mod { blocks, last_scale: Tensor::from_f32(dev, &fin[..d], &[d])?, last_shift: Tensor::from_f32(dev, &fin[d..], &[d])? })
    }

    fn forward(&self, run: &Run, x: &Tensor, m: &Mod, out: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let (t, ni) = (run.t_len, run.h * run.w);
        ensure!(x.shape[0] == ni, "latent tokens {} != {}", x.shape[0], ni);
        let h = Tensor::new(dev, DType::BF16, &[t + ni, self.dim])?;
        dev.dtod(h.ptr, run.txt.ptr, run.txt.bytes())?;
        ops::gemm_bf16(dev, x, &self.first.w, self.first.b.as_ref(), Act::None, Epi::Store, &h.rows(t, ni))?;
        for (b, mm) in self.blocks.iter().zip(&m.blocks) {
            self.block(&h, b, Some(mm), Some(&run.rope), None)?;
        }
        let xo = h.rows(t, ni);
        let y = Tensor::new(dev, DType::BF16, &[ni, self.dim])?;
        ops::norm_mod(dev, &xo, ModNorm::RmsOnePlus(&self.last_norm), EPS, ni, (&m.last_scale, &m.last_shift), (&m.last_scale, &m.last_shift), &y)?;
        ops::gemm_bf16(dev, &y, &self.last_linear.w, self.last_linear.b.as_ref(), Act::None, Epi::Store, out)
    }
}
