//! Qwen3 (8B) decoder used as the Qwen Image 2.1 text encoder: int8 ConvRot linears, f32 residual
//! stream, GQA attention, interleaved MRoPE, DeepStack injection.
use crate::cuda::Device;
use crate::ops::{self, AttnArgs, AttnView, Epi, QuantAct};
use crate::tensor::{DType, Tensor};
use crate::weights::{Loader, QLinear};
use anyhow::Result;
use std::sync::Arc;

pub const HIDDEN: usize = 4096;
pub const LAYERS: usize = 36;
pub const HEADS: usize = 32;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const INTER: usize = 12288;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f64 = 5_000_000.0;
pub const ROPE_DIMS: [usize; 3] = [24, 20, 20];
pub const QKV_N: usize = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM; // 6144

struct Layer {
    ln1: Tensor,
    qkv: QLinear,
    q_norm: Tensor,
    k_norm: Tensor,
    o: QLinear,
    ln2: Tensor,
    gate_up: QLinear,
    down: QLinear,
}

pub struct Llm {
    dev: Arc<Device>,
    pub embed_w: Tensor,
    pub embed_scale: Tensor,
    layers: Vec<Layer>,
}

impl Llm {
    pub fn load(l: &mut Loader) -> Result<Llm> {
        let emb = l.qlinear("model.embed_tokens")?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                ln1: l.bf16(&format!("{p}.input_layernorm.weight"))?,
                qkv: l.qlinear_cat(&[&format!("{p}.self_attn.q_proj"), &format!("{p}.self_attn.k_proj"), &format!("{p}.self_attn.v_proj")])?,
                q_norm: l.bf16(&format!("{p}.self_attn.q_norm.weight"))?,
                k_norm: l.bf16(&format!("{p}.self_attn.k_norm.weight"))?,
                o: l.qlinear(&format!("{p}.self_attn.o_proj"))?,
                ln2: l.bf16(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_up: l.qlinear_cat(&[&format!("{p}.mlp.gate_proj"), &format!("{p}.mlp.up_proj")])?,
                down: l.qlinear(&format!("{p}.mlp.down_proj"))?,
            });
        }
        Ok(Llm { dev: l.dev.clone(), embed_w: emb.w, embed_scale: emb.scale, layers })
    }

    /// Token embeddings -> f32 [L, 4096]
    pub fn embed(&self, tokens: &[i32]) -> Result<Tensor> {
        let dev = &self.dev;
        let t = Tensor::from_buf(dev.upload(tokens)?, DType::F32, &[tokens.len()]);
        let out = Tensor::new(dev, DType::F32, &[tokens.len(), HIDDEN])?;
        ops::embed_int8(dev, &self.embed_w, &self.embed_scale, &t, &out)?;
        Ok(out)
    }

    /// Build the interleaved MRoPE table [L, 64, 2] from 3 x L position ids.
    pub fn rope_table(&self, pos: &[Vec<i64>; 3]) -> Result<Tensor> {
        let l = pos[0].len();
        let mut tab = vec![0f32; l * 64 * 2];
        let inv: Vec<f64> = (0..64).map(|f| 1.0 / ROPE_THETA.powf((2 * f) as f64 / HEAD_DIM as f64)).collect();
        for i in 0..l {
            for f in 0..64 {
                let mut axis = 0;
                if f < ROPE_DIMS[1] * 3 && f % 3 == 1 {
                    axis = 1;
                } else if f < ROPE_DIMS[2] * 3 && f % 3 == 2 {
                    axis = 2;
                }
                let ang = pos[axis][i] as f64 * inv[f];
                tab[(i * 64 + f) * 2] = ang.cos() as f32;
                tab[(i * 64 + f) * 2 + 1] = ang.sin() as f32;
            }
        }
        Tensor::from_f32(&self.dev, &tab, &[l, 64, 2])
    }

    /// Full forward. `x` f32 [L, 4096] is updated in place with the output of the last layer
    /// (no final norm, matching hidden_states[-1]). `deepstack`: per image, (row offset, [features])
    /// added to that image's rows after layers 0..n.
    pub fn forward(&self, x: &Tensor, rope: &Tensor, deepstack: &[(usize, &[Tensor])]) -> Result<()> {
        let dev = &self.dev;
        let l = x.shape[0];
        let xq = Tensor::new(dev, DType::I8, &[l, HIDDEN])?;
        let xs = Tensor::new(dev, DType::F32, &[l])?;
        let qkv = Tensor::new(dev, DType::BF16, &[l, QKV_N])?;
        let attn = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        let attn_q = Tensor::new(dev, DType::I8, &[l, HIDDEN])?;
        let gu = Tensor::new(dev, DType::BF16, &[l, 2 * INTER])?;
        let hq = Tensor::new(dev, DType::I8, &[l, INTER])?;
        for (li, layer) in self.layers.iter().enumerate() {
            // attention
            ops::quant_rows(dev, x, QuantAct::RmsNorm, Some(&layer.ln1), EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &layer.qkv.w, &layer.qkv.scale, None, Epi::Store, &qkv)?;
            let q_ptr = qkv.ptr;
            let k_ptr = qkv.ptr + (HEADS * HEAD_DIM * 2) as u64;
            let v_ptr = qkv.ptr + ((HEADS + KV_HEADS) * HEAD_DIM * 2) as u64;
            ops::qk_norm_rope_llm(dev, q_ptr, QKV_N, k_ptr, QKV_N, l, HEADS, KV_HEADS, &layer.q_norm, &layer.k_norm, EPS, rope)?;
            ops::flash_attn(
                dev,
                &AttnArgs {
                    q: AttnView { ptr: q_ptr, tok_stride: QKV_N, head_stride: HEAD_DIM, len: l },
                    k1: AttnView { ptr: k_ptr, tok_stride: QKV_N, head_stride: HEAD_DIM, len: l },
                    v1: AttnView { ptr: v_ptr, tok_stride: QKV_N, head_stride: HEAD_DIM, len: l },
                    k2: AttnView::empty(),
                    v2: AttnView::empty(),
                    out: AttnView::contiguous(&attn, HEADS, HEAD_DIM),
                    hq: HEADS,
                    hk: KV_HEADS,
                    d: HEAD_DIM,
                    kv_limit: None,
                    causal: true,
                    causal_off: 0,
                },
            )?;
            ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &attn_q, &xs)?;
            ops::gemm_i8(dev, &attn_q, &xs, &layer.o.w, &layer.o.scale, None, Epi::AddRes(x), x)?;
            // mlp
            ops::quant_rows(dev, x, QuantAct::RmsNorm, Some(&layer.ln2), EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &layer.gate_up.w, &layer.gate_up.scale, None, Epi::Store, &gu)?;
            ops::quant_rows(dev, &gu, QuantAct::SwiGlu, None, 0.0, &hq, &xs)?;
            ops::gemm_i8(dev, &hq, &xs, &layer.down.w, &layer.down.scale, None, Epi::AddRes(x), x)?;
            for &(pos0, feats) in deepstack {
                if li < feats.len() {
                    ops::add_rows_f32(dev, x, pos0, &feats[li])?;
                }
            }
        }
        Ok(())
    }
}
