//! Qwen3 decoder (dense, GQA, q/k RMSNorm, split-half rope) as the text encoder of Krea 2 (Qwen3-VL-4B's language model),
//! FLUX.2 klein (Qwen3-4B / Qwen3-8B) and Anima (Qwen3-0.6B base).
//!
//! The size is read from the checkpoint (hidden, layers, heads, MLP width); head dim 128 throughout. Rope theta is 5e6
//! for Qwen3-VL (text-only MRoPE positions are equal on all three axes, so it reduces to plain rope) and 1e6 otherwise.
//! Weights are streamed layer by layer as in the Qwen3.5 encoder (bf16 after dequantizing fp8; int8 ConvRot stays int8),
//! and every prompt passes a layer before the next is uploaded. The residual stream is bf16 and the ops round like
//! HF transformers in bf16. Callers ask for hidden states by HF index (`hidden_states[i]`: 0 = embeddings, i = output
//! of layer i - 1) and/or the final-normed `last_hidden_state`. A prompt may carry trailing padding (`valid` < len):
//! padded keys are masked for every query, as HF does with an attention mask on right padding.
use crate::cuda::Device;
use crate::linear::{self, Lin, Want};
use crate::ops::{self, AttnView, Epi, QkNorm, RmsMode, Rope};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const HEAD_DIM: usize = 128;
pub const EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub inter: usize,
    pub vocab: usize,
    pub theta: f32,
    pub vl: bool,
}

pub struct Qwen3 {
    dev: Arc<Device>,
    st: SafeTensors,
    prefix: String,
    pub cfg: Config,
}

/// One tokenized prompt; tokens at index >= `valid` are padding.
pub struct Prompt {
    pub tokens: Vec<u32>,
    pub valid: usize,
}

/// Per-prompt outputs: the requested hidden states (in the order asked) and the final-normed last hidden state.
pub struct Encoded {
    pub taps: Vec<Tensor>,
    pub last: Option<Tensor>,
}

impl Qwen3 {
    pub fn open(dev: Arc<Device>, path: &Path) -> Result<Qwen3> {
        let st = SafeTensors::open(path)?;
        let prefix = ["model.language_model.", "language_model.model.", "model.", ""]
            .iter()
            .find(|p| st.has(&format!("{p}layers.0.input_layernorm.weight")) && st.has(&format!("{p}embed_tokens.weight")))
            .map(|p| p.to_string())
            .with_context(|| format!("{} does not look like a Qwen3 text encoder", path.display()))?;
        ensure!(st.has(&format!("{prefix}layers.0.self_attn.q_norm.weight")) && st.has(&format!("{prefix}layers.0.self_attn.q_proj.weight")), "{}: not a Qwen3 checkpoint (no q_norm / q_proj)", path.display());
        let emb = st.info(&format!("{prefix}embed_tokens.weight"))?.shape.clone();
        let q = st.info(&format!("{prefix}layers.0.self_attn.q_proj.weight"))?.shape[0];
        let k = st.info(&format!("{prefix}layers.0.self_attn.k_proj.weight"))?.shape[0];
        let inter = st.info(&format!("{prefix}layers.0.mlp.gate_proj.weight"))?.shape[0];
        let mut layers = 0;
        while st.has(&format!("{prefix}layers.{layers}.input_layernorm.weight")) {
            layers += 1;
        }
        let vl = st.tensors.keys().any(|k| k.contains("visual.")) || prefix.contains("language_model");
        let cfg = Config { hidden: emb[1], layers, heads: q / HEAD_DIM, kv_heads: k / HEAD_DIM, inter, vocab: emb[0], theta: if vl { 5_000_000.0 } else { 1_000_000.0 }, vl };
        ensure!(cfg.heads % cfg.kv_heads == 0 && q % HEAD_DIM == 0, "{}: unexpected Qwen3 attention shape", path.display());
        Ok(Qwen3 { dev, st, prefix, cfg })
    }

    fn name(&self, s: &str) -> String {
        format!("{}{s}", self.prefix)
    }

    /// Embedding rows for `tokens` as bf16 [L, hidden].
    fn embed(&self, tokens: &[u32]) -> Result<Tensor> {
        let dev = &self.dev;
        let name = self.name("embed_tokens.weight");
        let h = self.cfg.hidden;
        let info = self.st.info(&name)?;
        let x = Tensor::new(dev, DType::BF16, &[tokens.len(), h])?;
        for &t in tokens {
            ensure!((t as usize) < self.cfg.vocab, "token id {t} out of the text encoder's vocabulary");
        }
        match info.dtype.as_str() {
            "BF16" | "F16" | "F32" | "F8_E4M3" => {
                let table = self.st.bytes(&name)?;
                let es = table.len() / (self.cfg.vocab * h);
                let mut bits: Vec<u16> = Vec::with_capacity(tokens.len() * h);
                for &t in tokens {
                    let row = &table[t as usize * h * es..(t as usize + 1) * h * es];
                    match info.dtype.as_str() {
                        "BF16" => bits.extend(row.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))),
                        "F16" => bits.extend(row.chunks_exact(2).map(|c| crate::tensor::bf16_bits(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()))),
                        "F32" => bits.extend(row.chunks_exact(4).map(|c| crate::tensor::bf16_bits(f32::from_le_bytes(c.try_into().unwrap())))),
                        _ => {
                            let scale = ["embed_tokens.weight_scale", "embed_tokens.scale_weight"].iter().map(|s| self.name(s)).find(|n| self.st.has(n)).map(|n| self.st.f32s(&n).map(|v| v[0])).transpose()?.unwrap_or(1.0);
                            bits.extend(row.iter().map(|&b| crate::tensor::bf16_bits(linear::fp8_e4m3_to_f32(b) * scale)))
                        }
                    }
                }
                dev.htod_at(x.ptr, unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len() * 2) })?;
            }
            "I8" => {
                let mut l = Loader::new(&self.st, dev.clone());
                let q = l.qlinear(&self.name("embed_tokens"))?;
                let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
                let idt = Tensor::from_buf(dev.upload(&ids)?, DType::F32, &[ids.len()]);
                let f = Tensor::new(dev, DType::F32, &[tokens.len(), h])?;
                ops::embed_int8(dev, &q.w, &q.scale, &idt, &f)?;
                ops::to_bf16(dev, &f, &x)?;
            }
            other => anyhow::bail!("{name}: unsupported dtype {other}"),
        }
        Ok(x)
    }

    /// Rope table [L, 64, 2] (cos, sin) as HF computes it in f32 and casts to bf16.
    pub fn rope(&self, l: usize) -> Result<Tensor> {
        let inv: Vec<f32> = (0..64).map(|i| 1.0f32 / self.cfg.theta.powf((2 * i) as f32 / HEAD_DIM as f32)).collect();
        let mut tab = vec![0f32; l * 128];
        for p in 0..l {
            for (i, f) in inv.iter().enumerate() {
                let a = p as f32 * f;
                tab[p * 128 + i * 2] = half::bf16::from_f32(a.cos()).to_f32();
                tab[p * 128 + i * 2 + 1] = half::bf16::from_f32(a.sin()).to_f32();
            }
        }
        Tensor::from_f32(&self.dev, &tab, &[l, 64, 2])
    }

    /// Run the prompts through the decoder. `taps`: HF hidden-state indices to return; `last`: also return the
    /// final-normed output. `progress(layer)` is called per layer.
    pub fn encode(&self, prompts: &[Prompt], taps: &[usize], last: bool, progress: &dyn Fn(usize)) -> Result<Vec<Encoded>> {
        let dev = &self.dev;
        let c = &self.cfg;
        ensure!(taps.iter().all(|&t| t <= c.layers), "hidden state index out of range");
        let (hq, hk) = (c.heads, c.kv_heads);
        let qkv_n = (hq + 2 * hk) * HEAD_DIM;
        let mut xs: Vec<Tensor> = prompts.iter().map(|p| self.embed(&p.tokens)).collect::<Result<_>>()?;
        let ropes: Vec<Tensor> = prompts.iter().map(|p| self.rope(p.tokens.len())).collect::<Result<_>>()?;
        let limits: Vec<Option<Tensor>> = prompts
            .iter()
            .map(|p| {
                if p.valid >= p.tokens.len() {
                    return Ok(None);
                }
                let v: Vec<i32> = (0..p.tokens.len()).map(|i| (i + 1).min(p.valid.max(1)) as i32).collect();
                Ok(Some(Tensor::from_buf(dev.upload(&v)?, DType::F32, &[v.len()])))
            })
            .collect::<Result<_>>()?;
        let mut outs: Vec<Encoded> = prompts.iter().map(|_| Encoded { taps: Vec::new(), last: None }).collect();
        let tap = |outs: &mut Vec<Encoded>, xs: &[Tensor], idx: usize| -> Result<()> {
            for _ in taps.iter().filter(|&&t| t == idx) {
                for (o, x) in outs.iter_mut().zip(xs) {
                    let cpy = Tensor::new(dev, DType::BF16, &x.shape)?;
                    dev.dtod(cpy.ptr, x.ptr, x.bytes())?;
                    o.taps.push(cpy);
                }
            }
            Ok(())
        };
        // taps are returned in the order asked: collect per index then reorder at the end
        tap(&mut outs, &xs, 0)?;
        for li in 0..c.layers {
            progress(li);
            let p = |s: &str| self.name(&format!("layers.{li}.{s}"));
            let mut l = Loader::new(&self.st, dev.clone());
            let ln1 = linear::small_bf16(&mut l, &p("input_layernorm.weight"))?;
            let ln2 = linear::small_bf16(&mut l, &p("post_attention_layernorm.weight"))?;
            let qn = linear::small_bf16(&mut l, &p("self_attn.q_norm.weight"))?;
            let kn = linear::small_bf16(&mut l, &p("self_attn.k_norm.weight"))?;
            let qkv = linear::lin_cat(&mut l, &[&p("self_attn.q_proj"), &p("self_attn.k_proj"), &p("self_attn.v_proj")], Want::Bf16)?;
            let wo = linear::lin(&mut l, &p("self_attn.o_proj"), Want::Bf16)?;
            let gate_up = linear::lin_cat(&mut l, &[&p("mlp.gate_proj"), &p("mlp.up_proj")], Want::Bf16)?;
            let down = linear::lin(&mut l, &p("mlp.down_proj"), Want::Bf16)?;
            for ((x, rope), lim) in xs.iter_mut().zip(&ropes).zip(&limits) {
                let n = x.shape[0];
                let h = Tensor::new(dev, DType::BF16, &[n, c.hidden])?;
                ops::rmsnorm_x(dev, x, Some(&ln1), RmsMode::Hf, EPS, &h)?;
                let pr = Tensor::new(dev, DType::BF16, &[n, qkv_n])?;
                qkv.forward(dev, &h, None, Epi::Store, &pr)?;
                ops::qk_norm_rope(dev, pr.ptr, qkv_n, 0, hq * HEAD_DIM, n, hq, hk, Some(&qn), Some(&kn), EPS, QkNorm::Hf, Rope::HalfBf16, Some(rope))?;
                let a = Tensor::new(dev, DType::BF16, &[n, hq * HEAD_DIM])?;
                let view = |off: usize| AttnView { ptr: pr.ptr + (off * 2) as u64, tok_stride: qkv_n, head_stride: HEAD_DIM, len: n };
                ops::attention(dev, view(0), view(hq * HEAD_DIM), view((hq + hk) * HEAD_DIM), AttnView::contiguous(&a, hq, HEAD_DIM), hq, hk, HEAD_DIM, lim.is_none(), lim.as_ref())?;
                let o = Tensor::new(dev, DType::BF16, &[n, c.hidden])?;
                wo.forward(dev, &a, None, Epi::Store, &o)?;
                ops::add_inplace(dev, x, &o)?;
                ops::rmsnorm_x(dev, x, Some(&ln2), RmsMode::Hf, EPS, &h)?;
                mlp(dev, &h, &gate_up, &down, c.inter, x)?;
            }
            drop((qkv, wo, gate_up, down));
            tap(&mut outs, &xs, li + 1)?;
            dev.sync()?;
        }
        if last {
            let mut l = Loader::new(&self.st, dev.clone());
            let norm = linear::small_bf16(&mut l, &self.name("norm.weight"))?;
            for (o, x) in outs.iter_mut().zip(&xs) {
                let y = Tensor::new(dev, DType::BF16, &x.shape)?;
                ops::rmsnorm_x(dev, x, Some(&norm), RmsMode::Hf, EPS, &y)?;
                o.last = Some(y);
            }
        }
        // reorder taps from layer order to the order asked
        let mut order: Vec<usize> = (0..taps.len()).collect();
        order.sort_by_key(|&i| (taps[i], i));
        for o in outs.iter_mut() {
            let mut t: Vec<Option<Tensor>> = vec![None; taps.len()];
            for (k, tens) in order.iter().zip(o.taps.drain(..)) {
                t[*k] = Some(tens);
            }
            o.taps = t.into_iter().map(|x| x.expect("tap")).collect();
        }
        dev.sync()?;
        xs.clear();
        Ok(outs)
    }

    pub fn close(self) {
        self.st.drop_in_background();
    }
}

fn mlp(dev: &Device, h: &Tensor, gate_up: &Lin, down: &Lin, inter: usize, x: &Tensor) -> Result<()> {
    let n = h.shape[0];
    let gu = Tensor::new(dev, DType::BF16, &[n, 2 * inter])?;
    gate_up.forward(dev, h, None, Epi::Store, &gu)?;
    let a = Tensor::new(dev, DType::BF16, &[n, inter])?;
    ops::swiglu_round(dev, &gu, &a)?;
    let y = Tensor::new(dev, DType::BF16, &[n, x.shape[1]])?;
    down.forward(dev, &a, None, Epi::Store, &y)?;
    ops::add_inplace(dev, x, &y)
}
