//! MOSS-Image transformer (Kromoss weights): 5B single-stream DiT in the FLUX.2 latent space.
//!
//! From pipeline_moss_image.py (`MOSSImageTransformer`), dims from transformer/config.json:
//!   - dim 3072, 24 heads of 128, 42 blocks, FFN 8192 with the soft-capped SiTU gate
//!     situ(h, g) = 4 tanh(g/4) sigmoid(g) * 25 tanh(h/25); no biases in the blocks.
//!   - latent: 128 channels per token (32 FLUX.2 channels x 2x2 patch), one token per 16x16 pixels;
//!     `video_embedder` Linear(128 -> 3072).
//!   - text: Qwen3.5 last hidden state (4096) -> Linear -> 2 refiner blocks (attention over the text only, biased
//!     linears, plain RMSNorms) -> appended after the image tokens; text and image share every block.
//!   - time: sinusoid(256, f64, cos first) of t = sigma * 1000 -> Linear-SiLU-Linear; ONE modulation for all blocks
//!     (`shared_block_modulation`: shift/scale/gate for attention and FFN) and a head modulation (shift/scale).
//!   - block: x += o(attn(rope(qk_norm(qkv(modrms(x)))))) * gate_a ; x += down(situ(up_gate(modrms(x)))) * gate_f
//!   - rope: 4 axes of 16 frequencies (base 10000) over (r, f, h, w); image tokens at (0, T, row - (H-1)/2,
//!     col - (W-1)/2), text token i at (0, i, 0, 0); split-half rotation; the table is cast to bf16.
//!   - output: modrms(final_norm, head) -> Linear(3072 -> 128) on the image tokens.
//!
//! The text refiner does not depend on the timestep, so its output is computed once per prompt (`prepare`).
//! Weights: the ComfyUI single files (fused `proj.qkv`): bf16, or int8 ConvRot (block linears W8A8 with a 256-wide
//! Hadamard rotation; activations are rotated and quantized per token on the fly, fused into the norm kernels).
use crate::cuda::Device;
use crate::lora::{BlockLora, LoraDelta};
use crate::ops::{self, Act, AttnArgs, AttnView, Epi, QuantAct};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, bf16_to_f32, DType, Tensor};
use crate::weights::{Loader, QLinear};
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const DIM: usize = 3072;
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 42;
pub const FFN: usize = 8192;
pub const LATENT: usize = 128;
pub const TEXT_DIM: usize = 4096;
pub const EPS: f32 = 1e-6;

/// A block linear: int8 ConvRot or bf16.
pub enum Lin {
    I8(QLinear),
    Bf16(Tensor),
}
impl Lin {
    pub fn n(&self) -> usize {
        match self {
            Lin::I8(q) => q.n,
            Lin::Bf16(t) => t.shape[0],
        }
    }
}

struct Block {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: Lin,
    o: Lin,
    up_gate: Lin,
    down: Lin,
}

struct TextBlock {
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    qkv: (Tensor, Tensor),
    proj: (Tensor, Tensor),
    up_gate: (Tensor, Tensor), // [ffn_up ; ffn_gate] -> [hidden | gate]
    down: (Tensor, Tensor),
}

pub struct Dit {
    dev: Arc<Device>,
    pub prof: crate::cuda::Profiler,
    pub int8: bool,
    /// attention: false = bf16 flash attention, true = int8/fp8 SageAttention-style kernel
    pub sage: bool,
    video_in: (Tensor, Tensor),
    text_in: (Tensor, Tensor),
    text_blocks: Vec<TextBlock>,
    t1: (Tensor, Tensor),
    t2: (Tensor, Tensor),
    modulation: (Tensor, Tensor),
    head_mod: (Tensor, Tensor),
    blocks: Vec<Block>,
    final_norm: Tensor,
    final_linear: Tensor,
    /// unmerged LoRA deltas per block (empty when no LoRA)
    pub lora: Vec<BlockLora>,
    /// diagnostics: copies of the residual stream after these blocks are pushed to `tapped`
    pub tap_blocks: Vec<usize>,
    pub tapped: std::sync::Mutex<Vec<Tensor>>,
}

/// Step-independent state of one prompt at one latent size.
pub struct Run {
    /// refined text tokens [T, 3072] bf16
    text: Tensor,
    /// rope table [N, 64, 2] (image tokens, then text tokens)
    rope: Tensor,
    pub t_len: usize,
    pub h: usize,
    pub w: usize,
}

/// Per-timestep modulation (f32 tensors holding bf16 values).
pub struct Modulation {
    v: Vec<Tensor>, // shift_a, scale_a, gate_a, shift_f, scale_f, gate_f
    head_shift: Tensor,
    head_scale: Tensor,
    pub t: f32,
}

fn bias(l: &mut Loader, name: &str) -> Result<Tensor> {
    l.f32(name)
}

impl Dit {
    pub fn load(dev: Arc<Device>, path: &Path, force_bf16_attn: Option<bool>) -> Result<Dit> {
        let st = SafeTensors::open(path)?;
        ensure!(st.has("shared_block_modulation.weight") && st.has("layers.0.proj.ffn_up_gate.weight"), "{} is not a MOSS-Image diffusion model", path.display());
        let int8 = st.has("layers.0.proj.qkv.comfy_quant");
        let fused = st.has("layers.0.proj.qkv.weight");
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let lw = |l: &mut Loader, name: &str| -> Result<(Tensor, Tensor)> { Ok((l.bf16(&format!("{name}.weight"))?, bias(l, &format!("{name}.bias"))?)) };
        let video_in = lw(&mut l, "video_embedder")?;
        let text_in = lw(&mut l, "text_embedder.proj")?;
        let mut text_blocks = Vec::new();
        for i in 0..2 {
            let p = format!("text_embedder.blocks.{i}");
            text_blocks.push(TextBlock {
                norm1: l.bf16(&format!("{p}.norm1.weight"))?,
                norm2: l.bf16(&format!("{p}.norm2.weight"))?,
                q_norm: l.bf16(&format!("{p}.q_norm.weight"))?,
                k_norm: l.bf16(&format!("{p}.k_norm.weight"))?,
                qkv: lw(&mut l, &format!("{p}.qkv"))?,
                proj: lw(&mut l, &format!("{p}.proj"))?,
                up_gate: (l.bf16_cat(&[&format!("{p}.ffn_up.weight"), &format!("{p}.ffn_gate.weight")])?, l.f32_cat(&[&format!("{p}.ffn_up.bias"), &format!("{p}.ffn_gate.bias")])?),
                down: lw(&mut l, &format!("{p}.ffn_down"))?,
            });
        }
        let t1 = lw(&mut l, "time_embedding.0")?;
        let t2 = lw(&mut l, "time_embedding.2")?;
        let modulation = lw(&mut l, "shared_block_modulation")?;
        let head_mod = lw(&mut l, "video_head_modulation")?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("layers.{i}");
            let lin = |l: &mut Loader, name: &str| -> Result<Lin> {
                if int8 {
                    Ok(Lin::I8(l.qlinear(&format!("{p}.proj.{name}"))?))
                } else {
                    Ok(Lin::Bf16(l.bf16(&format!("{p}.proj.{name}.weight"))?))
                }
            };
            let qkv = if fused {
                lin(&mut l, "qkv")?
            } else if int8 {
                Lin::I8(l.qlinear_cat(&[&format!("{p}.proj.q"), &format!("{p}.proj.k"), &format!("{p}.proj.v")])?)
            } else {
                Lin::Bf16(l.bf16_cat(&[&format!("{p}.proj.q.weight"), &format!("{p}.proj.k.weight"), &format!("{p}.proj.v.weight")])?)
            };
            blocks.push(Block {
                norm1: l.bf16(&format!("{p}.norm1.weight"))?,
                norm2: l.bf16(&format!("{p}.norm2.weight"))?,
                q_norm: l.bf16(&format!("{p}.q_norm.weight"))?,
                k_norm: l.bf16(&format!("{p}.k_norm.weight"))?,
                qkv,
                o: lin(&mut l, "o")?,
                up_gate: lin(&mut l, "ffn_up_gate")?,
                down: lin(&mut l, "ffn_down")?,
            });
        }
        let final_norm = l.bf16("final_norm_video.weight")?;
        let final_linear = l.bf16("final_linear_video.weight")?;
        crate::info!("  dit: {} {:.1} GB uploaded in {:.1}s", if int8 { "int8 convrot" } else { "bf16" }, l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        let sage = match std::env::var("LOKI_ATTN").ok().as_deref() {
            Some("bf16") => false,
            Some("sage") | Some("int8") => true,
            _ => force_bf16_attn.map(|b| !b).unwrap_or(int8),
        };
        ops::sage_init(&dev)?;
        drop(l);
        st.drop_in_background();
        Ok(Dit { dev, prof: crate::cuda::Profiler::new(), int8, sage, video_in, text_in, text_blocks, t1, t2, modulation, head_mod, blocks, final_norm, final_linear, lora: Vec::new(), tap_blocks: Vec::new(), tapped: std::sync::Mutex::new(Vec::new()) })
    }

    /// Text refiner + rope tables for a prompt's features `context` [T, 4096] bf16 at an h x w latent grid.
    pub fn prepare(&self, context: &Tensor, h: usize, w: usize) -> Result<Run> {
        let dev = &self.dev;
        let t = context.shape[0];
        ensure!(t > 0, "empty text conditioning");
        let rope = rope_table(dev, h, w, t)?;
        let rope_text = rope.rows(h * w, t);
        let x = Tensor::new(dev, DType::BF16, &[t, DIM])?;
        ops::gemm_bf16(dev, context, &self.text_in.0, Some(&self.text_in.1), Act::None, Epi::Store, &x)?;
        for b in &self.text_blocks {
            let y = Tensor::new(dev, DType::BF16, &[t, DIM])?;
            ops::modrms(dev, &x, Some(&b.norm1), None, EPS, &y)?;
            let qkv = Tensor::new(dev, DType::BF16, &[t, 3 * DIM])?;
            ops::gemm_bf16(dev, &y, &b.qkv.0, Some(&b.qkv.1), Act::None, Epi::Store, &qkv)?;
            ops::qk_norm_rope_moss(dev, qkv.ptr, 3 * DIM, t, HEADS, &b.q_norm, &b.k_norm, EPS, &rope_text)?;
            let a = Tensor::new(dev, DType::BF16, &[t, DIM])?;
            self.attention(&qkv, t, &a, false)?;
            let o = Tensor::new(dev, DType::BF16, &[t, DIM])?;
            ops::gemm_bf16(dev, &a, &b.proj.0, Some(&b.proj.1), Act::None, Epi::Store, &o)?;
            ops::add_inplace(dev, &x, &o)?;
            ops::modrms(dev, &x, Some(&b.norm2), None, EPS, &y)?;
            let ug = Tensor::new(dev, DType::BF16, &[t, 2 * FFN])?;
            ops::gemm_bf16(dev, &y, &b.up_gate.0, Some(&b.up_gate.1), Act::None, Epi::Store, &ug)?;
            let s = Tensor::new(dev, DType::BF16, &[t, FFN])?;
            ops::situ(dev, &ug, &s)?;
            ops::gemm_bf16(dev, &s, &b.down.0, Some(&b.down.1), Act::None, Epi::Store, &o)?;
            ops::add_inplace(dev, &x, &o)?;
        }
        Ok(Run { text: x, rope, t_len: t, h, w })
    }

    /// The refined text tokens (diagnostics).
    pub fn run_text<'a>(&self, run: &'a Run) -> &'a Tensor {
        &run.text
    }

    /// Modulation for a timestep (t = sigma * 1000).
    pub fn modulation(&self, t: f32) -> Result<Modulation> {
        let dev = &self.dev;
        // sinusoid in f64 like training: cos first, freq = 10000^(-i/128)
        let mut emb = vec![0u16; 256];
        for i in 0..128 {
            let freq = 10000f64.powf(-(i as f64) / 128.0);
            let ph = (t as f64) * freq;
            emb[i] = bf16_bits(ph.cos() as f32);
            emb[128 + i] = bf16_bits(ph.sin() as f32);
        }
        let e = Tensor::from_bf16(dev, &emb, &[1, 256])?;
        let h1 = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        ops::gemm_bf16(dev, &e, &self.t1.0, Some(&self.t1.1), Act::None, Epi::Store, &h1)?;
        ops::act_inplace(dev, &h1, Act::Silu)?;
        let time = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        ops::gemm_bf16(dev, &h1, &self.t2.0, Some(&self.t2.1), Act::None, Epi::Store, &time)?;
        let m = Tensor::new(dev, DType::BF16, &[1, 6 * DIM])?;
        ops::gemm_bf16(dev, &time, &self.modulation.0, Some(&self.modulation.1), Act::None, Epi::Store, &m)?;
        let hm = Tensor::new(dev, DType::BF16, &[1, 2 * DIM])?;
        ops::gemm_bf16(dev, &time, &self.head_mod.0, Some(&self.head_mod.1), Act::None, Epi::Store, &hm)?;
        let mv = m.to_f32_vec(dev)?;
        let hv = hm.to_f32_vec(dev)?;
        let v = (0..6).map(|i| Tensor::from_f32(dev, &mv[i * DIM..(i + 1) * DIM], &[DIM])).collect::<Result<Vec<_>>>()?;
        Ok(Modulation { v, head_shift: Tensor::from_f32(dev, &hv[..DIM], &[DIM])?, head_scale: Tensor::from_f32(dev, &hv[DIM..], &[DIM])?, t })
    }

    /// Joint attention over `n` tokens of a fused qkv [n, 3*DIM] -> out [n, DIM].
    fn attention(&self, qkv: &Tensor, n: usize, out: &Tensor, allow_sage: bool) -> Result<()> {
        let dev = &self.dev;
        let ts = 3 * DIM;
        let (qp, kp, vp) = (qkv.ptr, qkv.ptr + (DIM * 2) as u64, qkv.ptr + (2 * DIM * 2) as u64);
        if self.sage && allow_sage {
            let kv = ops::quantize_kv(dev, kp, vp, ts, n, HEADS)?;
            let qq = ops::quantize_q(dev, qp, ts, n, HEADS, &kv, None)?;
            ops::sage_attn(dev, &qq, n, HEADS, &kv, None, AttnView::contiguous(out, HEADS, HEAD_DIM), None)
        } else {
            ops::flash_attn(
                dev,
                &AttnArgs {
                    q: AttnView { ptr: qp, tok_stride: ts, head_stride: HEAD_DIM, len: n },
                    k1: AttnView { ptr: kp, tok_stride: ts, head_stride: HEAD_DIM, len: n },
                    v1: AttnView { ptr: vp, tok_stride: ts, head_stride: HEAD_DIM, len: n },
                    k2: AttnView::empty(),
                    v2: AttnView::empty(),
                    out: AttnView::contiguous(out, HEADS, HEAD_DIM),
                    hq: HEADS,
                    hk: HEADS,
                    d: HEAD_DIM,
                    kv_limit: None,
                    causal: false,
                    causal_off: 0,
                },
            )
        }
    }

    /// y = lin(x) (bf16 out). `xq` = the int8-quantized input when the block is int8 (computed by the caller with
    /// the fused norm/activation), `xb` = the same input in bf16 (needed for bf16 weights and for LoRA).
    fn linear(&self, lin: &Lin, xq: Option<(&Tensor, &Tensor)>, xb: Option<&Tensor>, out: &Tensor, lora: Option<&LoraDelta>) -> Result<()> {
        let dev = &self.dev;
        match lin {
            Lin::I8(q) => {
                let (aq, sa) = xq.context("int8 linear without quantized input")?;
                ops::gemm_i8(dev, aq, sa, &q.w, &q.scale, None, Epi::Store, out)?;
            }
            Lin::Bf16(w) => ops::gemm_bf16(dev, xb.context("bf16 linear without bf16 input")?, w, None, Act::None, Epi::Store, out)?,
        }
        if let Some(d) = lora {
            let x = xb.context("LoRA without bf16 input")?;
            d.apply(dev, x, out)?;
        }
        Ok(())
    }

    /// One evaluation: `latent` [h*w, 128] bf16 token-major -> velocity `out` [h*w, 128] bf16. Blocks in `skip` are skipped.
    pub fn forward(&self, run: &Run, latent: &Tensor, m: &Modulation, skip: &[usize], out: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let ni = run.h * run.w;
        let n = ni + run.t_len;
        ensure!(latent.shape[0] == ni, "latent tokens {} != {}", latent.shape[0], ni);
        let x = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        ops::gemm_bf16(dev, latent, &self.video_in.0, Some(&self.video_in.1), Act::None, Epi::Store, &x.rows(0, ni))?;
        dev.dtod(x.ptr + (ni * DIM * 2) as u64, run.text.ptr, run.text.bytes())?;
        let int8 = self.int8;
        let has_lora = !self.lora.is_empty();
        let need_bf16 = !int8 || has_lora;
        let xq = Tensor::new(dev, DType::I8, &[n, if int8 { FFN } else { 1 }])?;
        let xs = Tensor::new(dev, DType::F32, &[n])?;
        let yb = Tensor::new(dev, DType::BF16, &[n, if need_bf16 { DIM } else { 1 }])?;
        let qkv = Tensor::new(dev, DType::BF16, &[n, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        let tmp = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        let ug = Tensor::new(dev, DType::BF16, &[n, 2 * FFN])?;
        let sb = Tensor::new(dev, DType::BF16, &[n, if need_bf16 { FFN } else { 1 }])?;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = [&m.v[0], &m.v[1], &m.v[2], &m.v[3], &m.v[4], &m.v[5]];
        let pr = &self.prof;
        // int8 activation buffer viewed as [n, k] (contiguous) for k = DIM or FFN
        let xq_dim = |k: usize| if int8 { xq.prefix(&[n, k]) } else { xq.clone() };
        for (li, b) in self.blocks.iter().enumerate() {
            if skip.contains(&li) {
                continue;
            }
            let lora = self.lora.get(li);
            // ---- attention
            let q_in = xq_dim(DIM);
            if int8 {
                pr.time(dev, "norm+quant", || ops::modrms_quant(dev, &x, &b.norm1, scale_a, shift_a, EPS, &q_in, &xs))?;
            }
            if need_bf16 {
                ops::modrms(dev, &x, Some(&b.norm1), Some((scale_a, shift_a)), EPS, &yb)?;
            }
            pr.time(dev, "gemm qkv", || self.linear(&b.qkv, Some((&q_in, &xs)), Some(&yb), &qkv, lora.and_then(|l| l.qkv.as_ref())))?;
            pr.time(dev, "qk norm+rope", || ops::qk_norm_rope_moss(dev, qkv.ptr, 3 * DIM, n, HEADS, &b.q_norm, &b.k_norm, EPS, &run.rope))?;
            pr.time(dev, "attention", || self.attention(&qkv, n, &attn, true))?;
            if int8 {
                pr.time(dev, "quant", || ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &q_in, &xs))?;
            }
            let lora_o = lora.and_then(|l| l.o.as_ref());
            if let (Lin::I8(q), None) = (&b.o, lora_o) {
                pr.time(dev, "gemm o", || ops::gemm_i8(dev, &q_in, &xs, &q.w, &q.scale, None, Epi::AddResGatedBf16(&x, gate_a), &x))?;
            } else {
                pr.time(dev, "gemm o", || self.linear(&b.o, Some((&q_in, &xs)), Some(&attn), &tmp, lora_o))?;
                pr.time(dev, "residual", || ops::gated_add(dev, &x, &tmp, gate_a))?;
            }
            // ---- FFN
            if int8 {
                pr.time(dev, "norm+quant", || ops::modrms_quant(dev, &x, &b.norm2, scale_f, shift_f, EPS, &q_in, &xs))?;
            }
            if need_bf16 {
                ops::modrms(dev, &x, Some(&b.norm2), Some((scale_f, shift_f)), EPS, &yb)?;
            }
            pr.time(dev, "gemm up_gate", || self.linear(&b.up_gate, Some((&q_in, &xs)), Some(&yb), &ug, lora.and_then(|l| l.up_gate.as_ref())))?;
            let h_in = xq_dim(FFN);
            if need_bf16 {
                ops::situ(dev, &ug, &sb)?;
                if int8 {
                    ops::quant_rows(dev, &sb, QuantAct::None, None, 0.0, &h_in, &xs)?;
                }
            } else {
                pr.time(dev, "situ+quant", || ops::quant_rows(dev, &ug, QuantAct::Situ, None, 0.0, &h_in, &xs))?;
            }
            let lora_d = lora.and_then(|l| l.down.as_ref());
            if let (Lin::I8(q), None) = (&b.down, lora_d) {
                pr.time(dev, "gemm down", || ops::gemm_i8(dev, &h_in, &xs, &q.w, &q.scale, None, Epi::AddResGatedBf16(&x, gate_f), &x))?;
            } else {
                pr.time(dev, "gemm down", || self.linear(&b.down, Some((&h_in, &xs)), Some(&sb), &tmp, lora_d))?;
                pr.time(dev, "residual", || ops::gated_add(dev, &x, &tmp, gate_f))?;
            }
            if self.tap_blocks.contains(&li) {
                let c = Tensor::new(dev, DType::BF16, &[n, DIM])?;
                dev.dtod(c.ptr, x.ptr, x.bytes())?;
                self.tapped.lock().unwrap().push(c);
            }
        }
        let xo = x.rows(0, ni);
        let y = yb_or_new(dev, &yb, ni)?;
        ops::modrms(dev, &xo, Some(&self.final_norm), Some((&m.head_scale, &m.head_shift)), EPS, &y)?;
        ops::gemm_bf16(dev, &y, &self.final_linear, None, Act::None, Epi::Store, out)?;
        Ok(())
    }
}

fn yb_or_new(dev: &Device, yb: &Tensor, rows: usize) -> Result<Tensor> {
    if yb.shape[1] == DIM {
        Ok(yb.rows(0, rows))
    } else {
        Tensor::new(dev, DType::BF16, &[rows, DIM])
    }
}

/// RoPE table [N, 64, 2] = (cos, sin) in bf16 values: 4 axes (r, f, h, w) x 16 bands, band_i = 1/10000^(i/16).
pub fn rope_table(dev: &Device, h: usize, w: usize, t: usize) -> Result<Tensor> {
    let bands: Vec<f32> = (0..16).map(|i| 1.0f32 / 10000f32.powf(i as f32 / 16.0)).collect();
    let n = h * w + t;
    let mut tab = vec![0f32; n * 128];
    let mut put = |tok: usize, coords: [f32; 4]| {
        for a in 0..4 {
            for b in 0..16 {
                let ph = coords[a] * bands[b];
                let f = a * 16 + b;
                tab[(tok * 64 + f) * 2] = bf16_to_f32(bf16_bits(ph.cos()));
                tab[(tok * 64 + f) * 2 + 1] = bf16_to_f32(bf16_bits(ph.sin()));
            }
        }
    };
    let (ch, cw) = ((h as f32 - 1.0) / 2.0, (w as f32 - 1.0) / 2.0);
    for y in 0..h {
        for x in 0..w {
            put(y * w + x, [0.0, t as f32, y as f32 - ch, x as f32 - cw]);
        }
    }
    for i in 0..t {
        put(h * w + i, [0.0, i as f32, 0.0, 0.0]);
    }
    Tensor::from_f32(dev, &tab, &[n, 64, 2])
}
