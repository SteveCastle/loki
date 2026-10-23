//! Qwen3.5-9B (language model only) as MOSS-Image's text encoder: the chat-templated prompt goes through all 32 layers and
//! the final norm; `last_hidden_state` [L, 4096] is the conditioning (pipeline_moss_image.py `encode_prompt`).
//!
//! Architecture (text_encoder/config.json): hybrid stack, every 4th layer full attention, the rest Gated DeltaNet linear
//! attention; all RMSNorms are zero-centred `(1 + w)` (except the DeltaNet's gated output norm, plain `w`).
//!   - Gated DeltaNet: in_proj_qkv (q 16x128 | k 16x128 | v 32x128) -> depthwise causal conv1d (k=4) + SiLU; q, k
//!     L2-normalized, q / sqrt(128); beta = sigmoid(b), g = -exp(A_log) softplus(a + dt_bias); delta rule
//!     S = S e^g + k (beta (v - S^T k))^T, o = S^T q (value head h uses key head h/2); o -> RMSNorm(w) * silu(z) -> out_proj.
//!   - Gated attention: q_proj gives per head [q | gate] (16 x 2 x 256), k/v 4 heads of 256 (GQA), q/k RMSNorm(1+w), rope
//!     on the first 64 dims (theta 1e7; the text-only MRoPE positions are equal on all three axes, so it is plain rope),
//!     causal softmax(q k^T / 16) v, times sigmoid(gate), o_proj.
//!   - MLP: SwiGLU 12288.
//!
//! The weights are streamed layer by layer (bf16, ~0.44 GB per layer) and every prompt of the batch runs through a layer
//! before the next one is uploaded, so the encoder never holds more than one layer on the GPU. The residual stream is
//! bf16 and every op rounds like the reference (HF transformers in bf16).
use crate::cuda::Device;
use crate::ops::{self, Act, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::tokenizer::Tokenizer;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const HIDDEN: usize = 4096;
pub const LAYERS: usize = 32;
pub const INTER: usize = 12288;
pub const HQ: usize = 16;
pub const HKV: usize = 4;
pub const HD: usize = 256;
pub const LIN_HK: usize = 16;
pub const LIN_HV: usize = 32;
pub const LIN_D: usize = 128;
pub const CONV_DIM: usize = 2 * LIN_HK * LIN_D + LIN_HV * LIN_D; // 8192
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 10_000_000.0;
/// fused linear-attention input projection: [qkv (8192) | z (4096) | b (32) | a (32)]
const LIN_IN: usize = CONV_DIM + LIN_HV * LIN_D + 2 * LIN_HV;
/// fused full-attention input projection: [q+gate (8192) | k (1024) | v (1024)]
const ATT_IN: usize = 2 * HQ * HD + 2 * HKV * HD;

/// The chat template the official pipeline applies (`apply_chat_template(..., add_generation_prompt=True,
/// enable_thinking=True)` on one user turn): the content is trimmed and the assistant turn left on an open think block.
pub fn template(prompt: &str) -> String {
    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n", prompt.trim())
}

pub fn is_full_attention(layer: usize) -> bool {
    layer % 4 == 3
}

pub struct TextEncoder {
    dev: Arc<Device>,
    st: SafeTensors,
    prefix: String,
    pub tok: Tokenizer,
}

/// Per-prompt encoder output.
pub struct Encoded {
    /// last_hidden_state [L, 4096] bf16
    pub hidden: Tensor,
    pub tokens: Vec<u32>,
}

impl TextEncoder {
    pub fn open(dev: Arc<Device>, path: &Path) -> Result<TextEncoder> {
        let st = SafeTensors::open(path)?;
        let prefix = ["model.language_model.", "language_model.", "model."]
            .iter()
            .find(|p| st.has(&format!("{p}layers.0.input_layernorm.weight")))
            .map(|p| p.to_string())
            .with_context(|| format!("{} does not look like a Qwen3.5 text encoder", path.display()))?;
        ensure!(st.has(&format!("{prefix}layers.0.linear_attn.in_proj_qkv.weight")), "{}: not a Qwen3.5 (Gated DeltaNet) checkpoint", path.display());
        let emb = st.info(&format!("{prefix}embed_tokens.weight"))?;
        ensure!(emb.shape == [248320, HIDDEN], "unexpected Qwen3.5 embedding shape {:?} (need the 9B model)", emb.shape);
        Ok(TextEncoder { dev, st, prefix, tok: Tokenizer::new()? })
    }

    fn name(&self, s: &str) -> String {
        format!("{}{s}", self.prefix)
    }

    fn bf16(&self, name: &str) -> Result<Tensor> {
        let info = self.st.info(name)?;
        let shape = info.shape.clone();
        match info.dtype.as_str() {
            "BF16" => {
                let t = Tensor::new(&self.dev, DType::BF16, &shape)?;
                self.dev.htod_at(t.ptr, self.st.bytes(name)?)?;
                Ok(t)
            }
            _ => {
                let v = self.st.f32s(name)?;
                let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x)).collect();
                Tensor::from_bf16(&self.dev, &bits, &shape)
            }
        }
    }
    /// Several bf16 [n_i, K] matrices stacked along N.
    fn bf16_cat(&self, names: &[String]) -> Result<Tensor> {
        let mut n = 0;
        let mut k = 0;
        for nm in names {
            let i = self.st.info(nm)?;
            ensure!(i.dtype == "BF16" && i.shape.len() == 2, "{nm}: expected 2D bf16");
            k = i.shape[1];
            n += i.shape[0];
        }
        let t = Tensor::new(&self.dev, DType::BF16, &[n, k])?;
        let mut off = 0usize;
        for nm in names {
            let b = self.st.bytes(nm)?;
            self.dev.htod_at(t.ptr + off as u64, b)?;
            off += b.len();
        }
        Ok(t)
    }
    /// A small vector as f32 holding bf16-rounded values (the official text_encoder/ export stores these in bf16).
    fn f32_bf16_rounded(&self, name: &str) -> Result<Tensor> {
        let v: Vec<f32> = self.st.f32s(name)?.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect();
        Tensor::from_f32(&self.dev, &v, &[v.len()])
    }

    /// Token ids of a prompt after the chat template.
    pub fn tokens(&self, prompt: &str) -> Vec<u32> {
        self.tok.encode(&template(prompt))
    }

    /// Encode prompts (each unpadded, as the official pipeline does). `progress(layer)` is called per layer.
    pub fn encode(&self, prompts: &[String], progress: &dyn Fn(usize)) -> Result<Vec<Encoded>> {
        Ok(self.encode_taps(prompts, &[], progress)?.0)
    }

    /// `encode`, also returning copies of the residual stream after the layers listed in `taps` (per prompt, in order).
    pub fn encode_taps(&self, prompts: &[String], taps: &[usize], progress: &dyn Fn(usize)) -> Result<(Vec<Encoded>, Vec<Vec<Tensor>>)> {
        let dev = &self.dev;
        let mut tapped: Vec<Vec<Tensor>> = vec![Vec::new(); prompts.len()];
        let t_enc = std::time::Instant::now();
        let mark = |what: &str| if std::env::var("LOKI_TE_TIMING").is_ok() { eprintln!("  te {what} at {:.2}s", t_enc.elapsed().as_secs_f64()) };
        let toks: Vec<Vec<u32>> = prompts.iter().map(|p| self.tokens(p)).collect();
        mark("tokenized");
        for (t, p) in toks.iter().zip(prompts) {
            ensure!(t.len() <= 4096, "prompt too long: {} tokens (the pipeline truncates at 4096): {:.60}...", t.len(), p);
        }
        // embeddings, gathered on the host from the mapped table
        let table = self.st.bytes(&self.name("embed_tokens.weight"))?;
        let mut xs: Vec<Tensor> = Vec::with_capacity(toks.len());
        for t in &toks {
            let mut buf = Vec::with_capacity(t.len() * HIDDEN * 2);
            for &id in t {
                let o = id as usize * HIDDEN * 2;
                buf.extend_from_slice(&table[o..o + HIDDEN * 2]);
            }
            let x = Tensor::new(dev, DType::BF16, &[t.len(), HIDDEN])?;
            dev.htod_at(x.ptr, &buf)?;
            xs.push(x);
        }
        let ropes: Vec<Tensor> = toks.iter().map(|t| rope_table(&self.dev, t.len())).collect::<Result<_>>()?;
        // a helper thread pre-faults the mapped weights a few layers ahead (first-touch page faults are the slow part)
        let stop = std::sync::atomic::AtomicBool::new(false);
        let layer_bytes: Vec<Vec<&[u8]>> = (0..LAYERS)
            .map(|li| {
                let pre = self.name(&format!("layers.{li}."));
                let mut v: Vec<(usize, &[u8])> = self.st.tensors.iter().filter(|(k, _)| k.starts_with(&pre)).map(|(k, t)| (t.data_offsets.0, self.st.bytes(k).unwrap_or(&[]))).collect();
                v.sort_by_key(|x| x.0);
                v.into_iter().map(|x| x.1).collect()
            })
            .collect();
        std::thread::scope(|scope| -> Result<()> {
        let stop = &stop;
        let layer_bytes = &layer_bytes;
        scope.spawn(move || {
            let mut acc = 0u8;
            for layer in layer_bytes.iter() {
                for b in layer {
                    for c in b.chunks(4096) {
                        if stop.load(std::sync::atomic::Ordering::Relaxed) {
                            return;
                        }
                        acc = acc.wrapping_add(unsafe { std::ptr::read_volatile(c.as_ptr()) });
                    }
                }
            }
            std::hint::black_box(acc);
        });
        let res = (|| -> Result<()> {
        mark("layers start");
        let timing = std::env::var("LOKI_TE_TIMING").is_ok();
        let (mut t_up, mut t_run) = (0f64, 0f64);
        for li in 0..LAYERS {
            progress(li);
            let t_layer = std::time::Instant::now();
            let p = |s: &str| self.name(&format!("layers.{li}.{s}"));
            let ln1 = self.bf16(&p("input_layernorm.weight"))?;
            let ln2 = self.bf16(&p("post_attention_layernorm.weight"))?;
            let gate_up = self.bf16_cat(&[p("mlp.gate_proj.weight"), p("mlp.up_proj.weight")])?;
            let down = self.bf16(&p("mlp.down_proj.weight"))?;
            if is_full_attention(li) {
                let w_in = self.bf16_cat(&[p("self_attn.q_proj.weight"), p("self_attn.k_proj.weight"), p("self_attn.v_proj.weight")])?;
                let wo = self.bf16(&p("self_attn.o_proj.weight"))?;
                let qn = self.bf16(&p("self_attn.q_norm.weight"))?;
                let kn = self.bf16(&p("self_attn.k_norm.weight"))?;
                if timing { dev.sync()?; t_up += t_layer.elapsed().as_secs_f64(); }
                for (x, rope) in xs.iter().zip(&ropes) {
                    let l = x.shape[0];
                    let h = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
                    ops::rmsnorm1p(dev, x, &ln1, EPS, &h)?;
                    let pr = Tensor::new(dev, DType::BF16, &[l, ATT_IN])?;
                    ops::gemm_bf16(dev, &h, &w_in, None, Act::None, Epi::Store, &pr)?;
                    let qo = Tensor::new(dev, DType::BF16, &[l, HQ, HD])?;
                    let ko = Tensor::new(dev, DType::BF16, &[l, HKV, HD])?;
                    ops::qk_norm_rope_q35(dev, &pr, 2 * HQ * HD, HQ, HKV, &qn, &kn, EPS, rope, &qo, &ko)?;
                    let a = Tensor::new(dev, DType::BF16, &[l, HQ * HD])?;
                    let v_ptr = pr.ptr + ((2 * HQ * HD + HKV * HD) * 2) as u64;
                    ops::attn_q35(dev, &qo, &ko, v_ptr, ATT_IN, pr.ptr, ATT_IN, &a, l, HQ, HKV)?;
                    let o = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
                    ops::gemm_bf16(dev, &a, &wo, None, Act::None, Epi::Store, &o)?;
                    ops::add_inplace(dev, x, &o)?;
                    mlp(dev, x, &ln2, &gate_up, &down)?;
                }
            } else {
                let w_in = self.bf16_cat(&[p("linear_attn.in_proj_qkv.weight"), p("linear_attn.in_proj_z.weight"), p("linear_attn.in_proj_b.weight"), p("linear_attn.in_proj_a.weight")])?;
                let conv_w = self.bf16(&p("linear_attn.conv1d.weight"))?;
                let a_log = self.f32_bf16_rounded(&p("linear_attn.A_log"))?;
                let dt_bias = self.f32_bf16_rounded(&p("linear_attn.dt_bias"))?;
                let norm_w = self.bf16(&p("linear_attn.norm.weight"))?;
                let wo = self.bf16(&p("linear_attn.out_proj.weight"))?;
                if timing { dev.sync()?; t_up += t_layer.elapsed().as_secs_f64(); }
                for x in &xs {
                    let l = x.shape[0];
                    let h = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
                    ops::rmsnorm1p(dev, x, &ln1, EPS, &h)?;
                    let pr = Tensor::new(dev, DType::BF16, &[l, LIN_IN])?;
                    ops::gemm_bf16(dev, &h, &w_in, None, Act::None, Epi::Store, &pr)?;
                    let conv = Tensor::new(dev, DType::BF16, &[l, CONV_DIM])?;
                    ops::causal_conv_silu(dev, pr.ptr, LIN_IN, &conv_w, &conv, l, CONV_DIM)?;
                    let qf = Tensor::new(dev, DType::F32, &[l, LIN_HK, LIN_D])?;
                    let kf = Tensor::new(dev, DType::F32, &[l, LIN_HK, LIN_D])?;
                    let vf = Tensor::new(dev, DType::F32, &[l, LIN_HV, LIN_D])?;
                    let g = Tensor::new(dev, DType::F32, &[l, LIN_HV])?;
                    let beta = Tensor::new(dev, DType::F32, &[l, LIN_HV])?;
                    let ab = pr.ptr + ((CONV_DIM + LIN_HV * LIN_D) * 2) as u64;
                    ops::gdn_prep(dev, &conv, ab, LIN_IN, l, LIN_HK, LIN_HV, &a_log, &dt_bias, &qf, &kf, &vf, &g, &beta)?;
                    let core = Tensor::new(dev, DType::F32, &[l, LIN_HV, LIN_D])?;
                    ops::gdn_recurrent(dev, &qf, &kf, &vf, &g, &beta, &core, l, LIN_HK, LIN_HV)?;
                    let o = Tensor::new(dev, DType::BF16, &[l, LIN_HV * LIN_D])?;
                    ops::gdn_norm(dev, &core, pr.ptr, LIN_IN, CONV_DIM, &norm_w, EPS, &o, l, LIN_HV)?;
                    let y = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
                    ops::gemm_bf16(dev, &o, &wo, None, Act::None, Epi::Store, &y)?;
                    ops::add_inplace(dev, x, &y)?;
                    mlp(dev, x, &ln2, &gate_up, &down)?;
                }
            }
            if taps.contains(&li) {
                for (i, x) in xs.iter().enumerate() {
                    let c = Tensor::new(dev, DType::BF16, &x.shape)?;
                    dev.dtod(c.ptr, x.ptr, x.bytes())?;
                    tapped[i].push(c);
                }
            }
            if timing { dev.sync()?; t_run += t_layer.elapsed().as_secs_f64(); }
            dev.sync()?; // the layer's weights are freed when this scope ends; keep the host from racing ahead
        }
        if timing { eprintln!("  te timing: upload {:.2}s, total {:.2}s", t_up, t_run); }
        Ok(())
        })();
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        res
        })?;
        mark("layers done");
        let norm = self.bf16(&self.name("norm.weight"))?;
        let mut out = Vec::with_capacity(xs.len());
        for (x, t) in xs.into_iter().zip(toks) {
            let hidden = Tensor::new(dev, DType::BF16, &x.shape)?;
            ops::rmsnorm1p(dev, &x, &norm, EPS, &hidden)?;
            out.push(Encoded { hidden, tokens: t });
        }
        dev.sync()?;
        Ok((out, tapped))
    }

    /// Release the encoder; its file mapping is unmapped in the background.
    pub fn close(self) {
        self.st.drop_in_background();
    }

    /// The mapped checkpoint.
    pub fn file(&self) -> &SafeTensors {
        &self.st
    }
}

fn mlp(dev: &Device, x: &Tensor, ln2: &Tensor, gate_up: &Tensor, down: &Tensor) -> Result<()> {
    let l = x.shape[0];
    let h = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
    ops::rmsnorm1p(dev, x, ln2, EPS, &h)?;
    let gu = Tensor::new(dev, DType::BF16, &[l, 2 * INTER])?;
    ops::gemm_bf16(dev, &h, gate_up, None, Act::None, Epi::Store, &gu)?;
    let a = Tensor::new(dev, DType::BF16, &[l, INTER])?;
    ops::swiglu_round(dev, &gu, &a)?;
    let y = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
    ops::gemm_bf16(dev, &a, down, None, Act::None, Epi::Store, &y)?;
    ops::add_inplace(dev, x, &y)
}

/// Text rope table [L, 32, 2] = (cos, sin) of position * inv_freq (inv_freq = 1 / theta^(2i/64)), rounded to bf16 like HF.
fn rope_table(dev: &Device, l: usize) -> Result<Tensor> {
    let inv: Vec<f32> = (0..32).map(|i| 1.0f32 / ROPE_THETA.powf((2 * i) as f32 / 64.0)).collect();
    let mut tab = vec![0f32; l * 64];
    for p in 0..l {
        for i in 0..32 {
            let f = (p as f32) * inv[i];
            tab[p * 64 + i * 2] = half::bf16::from_f32((f as f64).cos() as f32).to_f32();
            tab[p * 64 + i * 2 + 1] = half::bf16::from_f32((f as f64).sin() as f32).to_f32();
        }
    }
    Tensor::from_f32(dev, &tab, &[l, 32, 2])
}
