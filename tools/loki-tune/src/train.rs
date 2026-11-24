//! One training step of a LoRA on the frozen MOSS-Image DiT: flow-matching loss on one image, gradients for the LoRA
//! weights only.
//!
//! Forward: exactly loki-dream's `Dit::forward` (int8 ConvRot W8A8 or bf16 base linears, bf16 flash attention, the
//! LoRA as two extra GEMMs per linear), keeping the residual stream at every block input. Backward: from the last
//! block to the first, each block is run forward again keeping its intermediates (block-level gradient
//! checkpointing: about 1 GB of block inputs at 1 MP instead of ~20 GB of activations), then differentiated:
//!
//!   x += gate_f * down(situ(up_gate(modrms2(x))))    x += gate_a * o(attn(rope(qknorm(qkv(modrms1(x))))))
//!
//! Base weights are frozen: their input gradients use W^T (dequantized for int8, transposed into a scratch buffer
//! per linear). An int8 linear's activations were rotated by the Hadamard-256 ConvRot before quantization, so its
//! input gradient is rotated back (H is symmetric and its own inverse); the quantizer is a straight-through estimator.
//! Text tokens share every block with the image tokens (and so carry gradient), but only image tokens are in the loss.
use crate::gpu;
use crate::lora::{Kind, LoraSet};
use anyhow::{Context, Result};
use loki_dream::cuda::Device;
use loki_dream::dit::{Dit, Lin, Modulation, Run, DIM, EPS, FFN, HEADS, LATENT};
use loki_dream::ops::{self, Act, Epi, QuantAct};
use loki_dream::tensor::{DType, Tensor};

/// Per-model state the steps share: the transposed output projection.
pub struct Trainer {
    final_t: Tensor,
}

/// A block's intermediates for its backward pass.
struct Saved {
    y1: Tensor,
    qkv_pre: Tensor,
    qkv: Tensor,
    attn: Tensor,
    lse: Tensor,
    x_mid: Tensor,
    y2: Tensor,
    ug: Tensor,
    s: Tensor,
    mid: [Tensor; 4],
}

pub struct StepOut {
    pub loss: f32,
}

fn copy(dev: &Device, t: &Tensor) -> Result<Tensor> {
    let c = Tensor::new(dev, t.dtype, &t.shape)?;
    dev.dtod(c.ptr, t.ptr, t.bytes())?;
    Ok(c)
}

/// y = base(x) (bf16): int8 from the quantized input, bf16 from the bf16 input.
pub fn base_fwd(dev: &Device, lin: &Lin, xq: Option<(&Tensor, &Tensor)>, xb: &Tensor, out: &Tensor) -> Result<()> {
    match lin {
        Lin::I8(q) => {
            let (aq, sa) = xq.context("int8 linear without its quantized input")?;
            ops::gemm_i8(dev, aq, sa, &q.w, &q.scale, None, Epi::Store, out)
        }
        Lin::Bf16(w) => ops::gemm_bf16(dev, xb, w, None, Act::None, Epi::Store, out),
    }
}

/// dx (f32 [M, K]) = dy (bf16 [M, N]) . W, rotated back for an int8 ConvRot linear.
pub fn base_bwd(dev: &Device, lin: &Lin, dy: &Tensor, dx: &Tensor) -> Result<()> {
    let (w, scale, i8) = match lin {
        Lin::I8(q) => (&q.w, Some(&q.scale), true),
        Lin::Bf16(w) => (w, None, false),
    };
    let (n, k) = (w.shape[0], w.numel() / w.shape[0]);
    let wt = Tensor::new(dev, DType::BF16, &[k, n])?;
    gpu::transpose_w(dev, w, scale, &wt)?;
    ops::gemm_bf16(dev, dy, &wt, None, Act::None, Epi::Store, dx)?;
    if i8 {
        gpu::fwht256(dev, dx)?;
    }
    Ok(())
}

impl Trainer {
    pub fn new(dev: &Device, dit: &Dit) -> Result<Trainer> {
        let final_t = Tensor::new(dev, DType::BF16, &[DIM, LATENT])?;
        gpu::transpose_w(dev, &dit.final_linear, None, &final_t)?;
        Ok(Trainer { final_t })
    }

    /// out += LoRA(x); returns mid = x A^T.
    fn lora_fwd(&self, dev: &Device, lora: &LoraSet, layer: usize, kind: Kind, x: &Tensor, out: &Tensor) -> Result<Tensor> {
        let (l, m) = lora.get(layer, kind);
        let mid = Tensor::new(dev, DType::BF16, &[x.shape[0], l.r])?;
        ops::gemm_bf16(dev, x, &m.down, None, Act::None, Epi::Store, &mid)?;
        ops::gemm_bf16(dev, &mid, &m.up, None, Act::None, Epi::AddRes(out), out)?;
        Ok(mid)
    }

    /// LoRA weight gradients (accumulated into the set's buffers) and dx += its input gradient.
    #[allow(clippy::too_many_arguments)]
    fn lora_bwd(&self, dev: &Device, lora: &LoraSet, layer: usize, kind: Kind, x: &Tensor, mid: &Tensor, dy: &Tensor, dx: &Tensor) -> Result<()> {
        let (l, m) = lora.get(layer, kind);
        let (gd, gu) = lora.grad_ptrs(l);
        gpu::gemm_tn(dev, dy, mid, gu, lora.scale())?;
        let dmid = Tensor::new(dev, DType::BF16, &[x.shape[0], l.r])?;
        ops::gemm_bf16(dev, dy, &m.up_t, None, Act::None, Epi::Store, &dmid)?;
        gpu::gemm_tn(dev, &dmid, x, gd, 1.0)?;
        ops::gemm_bf16(dev, &dmid, &m.down_t, None, Act::None, Epi::AddRes(dx), dx)
    }

    /// One block forward on x [n, DIM] bf16 in place; with `save`, the intermediates its backward needs.
    fn block_fwd(&self, dev: &Device, dit: &Dit, lora: &LoraSet, li: usize, run: &Run, m: &Modulation, x: &Tensor, save: bool) -> Result<Option<Saved>> {
        let b = &dit.blocks[li];
        let n = x.shape[0];
        let int8 = dit.int8;
        let [shift_a, scale_a, gate_a, shift_f, scale_f, gate_f] = [&m.v[0], &m.v[1], &m.v[2], &m.v[3], &m.v[4], &m.v[5]];
        let xq = Tensor::new(dev, DType::I8, &[n, if int8 { FFN } else { 1 }])?;
        let xs = Tensor::new(dev, DType::F32, &[n])?;
        let q_of = |k: usize| if int8 { Some(xq.prefix(&[n, k])) } else { None };
        // ---- attention
        let y1 = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        ops::modrms(dev, x, Some(&b.norm1), Some((scale_a, shift_a)), EPS, &y1)?;
        let qd = q_of(DIM);
        if let Some(q) = &qd {
            ops::modrms_quant(dev, x, &b.norm1, scale_a, shift_a, EPS, q, &xs)?;
        }
        let qkv = Tensor::new(dev, DType::BF16, &[n, 3 * DIM])?;
        base_fwd(dev, &b.qkv, qd.as_ref().map(|q| (q, &xs)), &y1, &qkv)?;
        let mid_qkv = self.lora_fwd(dev, lora, li, Kind::Qkv, &y1, &qkv)?;
        let qkv_pre = if save { Some(copy(dev, &qkv)?) } else { None };
        ops::qk_norm_rope_moss(dev, qkv.ptr, 3 * DIM, n, HEADS, &b.q_norm, &b.k_norm, EPS, &run.rope)?;
        let attn = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        let lse = Tensor::new(dev, DType::F32, &[HEADS, n])?;
        gpu::attn_fwd(dev, &qkv, n, &attn, &lse)?;
        if let Some(q) = &qd {
            ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, q, &xs)?;
        }
        let tmp = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        base_fwd(dev, &b.o, qd.as_ref().map(|q| (q, &xs)), &attn, &tmp)?;
        let mid_o = self.lora_fwd(dev, lora, li, Kind::O, &attn, &tmp)?;
        ops::gated_add(dev, x, &tmp, gate_a)?;
        let x_mid = if save { Some(copy(dev, x)?) } else { None };
        // ---- FFN
        let y2 = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        ops::modrms(dev, x, Some(&b.norm2), Some((scale_f, shift_f)), EPS, &y2)?;
        if let Some(q) = &qd {
            ops::modrms_quant(dev, x, &b.norm2, scale_f, shift_f, EPS, q, &xs)?;
        }
        let ug = Tensor::new(dev, DType::BF16, &[n, 2 * FFN])?;
        base_fwd(dev, &b.up_gate, qd.as_ref().map(|q| (q, &xs)), &y2, &ug)?;
        let mid_ug = self.lora_fwd(dev, lora, li, Kind::UpGate, &y2, &ug)?;
        let s = Tensor::new(dev, DType::BF16, &[n, FFN])?;
        ops::situ(dev, &ug, &s)?;
        let qf = q_of(FFN);
        if let Some(q) = &qf {
            ops::quant_rows(dev, &s, QuantAct::None, None, 0.0, q, &xs)?;
        }
        base_fwd(dev, &b.down, qf.as_ref().map(|q| (q, &xs)), &s, &tmp)?;
        let mid_d = self.lora_fwd(dev, lora, li, Kind::Down, &s, &tmp)?;
        ops::gated_add(dev, x, &tmp, gate_f)?;
        if !save {
            return Ok(None);
        }
        Ok(Some(Saved { y1, qkv_pre: qkv_pre.unwrap(), qkv, attn, lse, x_mid: x_mid.unwrap(), y2, ug, s, mid: [mid_qkv, mid_o, mid_ug, mid_d] }))
    }

    /// Backward through block `li`: dx (f32 [n, DIM]) holds the gradient at the block output on entry, at its input on exit.
    #[allow(clippy::too_many_arguments)]
    fn block_bwd(&self, dev: &Device, dit: &Dit, lora: &LoraSet, li: usize, run: &Run, m: &Modulation, x_in: &Tensor, sv: &Saved, dx: &Tensor) -> Result<()> {
        let b = &dit.blocks[li];
        let n = x_in.shape[0];
        let [_, scale_a, gate_a, _, scale_f, gate_f] = [&m.v[0], &m.v[1], &m.v[2], &m.v[3], &m.v[4], &m.v[5]];
        // ---- FFN branch: x_out = x_mid + gate_f * down(s)
        let d_f = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        gpu::scale_cols_bf16(dev, dx, gate_f, &d_f)?;
        let ds = Tensor::new(dev, DType::F32, &[n, FFN])?;
        base_bwd(dev, &b.down, &d_f, &ds)?;
        self.lora_bwd(dev, lora, li, Kind::Down, &sv.s, &sv.mid[3], &d_f, &ds)?;
        drop(d_f);
        let dug = Tensor::new(dev, DType::BF16, &[n, 2 * FFN])?;
        gpu::situ_bwd(dev, &sv.ug, &ds, &dug)?;
        drop(ds);
        let dy = Tensor::new(dev, DType::F32, &[n, DIM])?;
        base_bwd(dev, &b.up_gate, &dug, &dy)?;
        self.lora_bwd(dev, lora, li, Kind::UpGate, &sv.y2, &sv.mid[2], &dug, &dy)?;
        drop(dug);
        gpu::modrms_bwd(dev, &sv.x_mid, &b.norm2, Some(scale_f), &dy, dx, EPS)?;
        // ---- attention branch: x_mid = x_in + gate_a * o(attn)
        let d_a = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        gpu::scale_cols_bf16(dev, dx, gate_a, &d_a)?;
        let dattn = Tensor::new(dev, DType::F32, &[n, DIM])?;
        base_bwd(dev, &b.o, &d_a, &dattn)?;
        self.lora_bwd(dev, lora, li, Kind::O, &sv.attn, &sv.mid[1], &d_a, &dattn)?;
        drop(d_a);
        let dattn_b = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        ops::to_bf16(dev, &dattn, &dattn_b)?;
        drop(dattn);
        let dqkv = Tensor::new(dev, DType::BF16, &[n, 3 * DIM])?;
        gpu::attn_bwd(dev, &sv.qkv, &sv.attn, &sv.lse, &dattn_b, n, &dqkv)?;
        drop(dattn_b);
        gpu::qk_norm_rope_bwd(dev, &sv.qkv_pre, &dqkv, 3 * DIM, n, HEADS, &b.q_norm, &b.k_norm, EPS, &run.rope)?;
        base_bwd(dev, &b.qkv, &dqkv, &dy)?;
        self.lora_bwd(dev, lora, li, Kind::Qkv, &sv.y1, &sv.mid[0], &dqkv, &dy)?;
        gpu::modrms_bwd(dev, x_in, &b.norm1, Some(scale_a), &dy, dx, EPS)
    }

    /// The model input for a noisy latent: video_embedder(latent) then the refined text tokens, [n, DIM] bf16.
    fn embed(&self, dev: &Device, dit: &Dit, run: &Run, xt: &Tensor) -> Result<Tensor> {
        let ni = run.h * run.w;
        let n = ni + run.t_len;
        let x = Tensor::new(dev, DType::BF16, &[n, DIM])?;
        ops::gemm_bf16(dev, xt, &dit.video_in.0, Some(&dit.video_in.1), Act::None, Epi::Store, &x.rows(0, ni))?;
        dev.dtod(x.ptr + (ni * DIM * 2) as u64, run.text.ptr, run.text.bytes())?;
        Ok(x)
    }

    /// Velocity prediction for a noisy latent (no gradients): the training forward, for checks.
    pub fn predict(&self, dev: &Device, dit: &Dit, lora: &LoraSet, run: &Run, xt: &Tensor, sigma: f32) -> Result<Tensor> {
        let m = dit.modulation(loki_dream::sampler::timestep(sigma))?;
        let x = self.embed(dev, dit, run, xt)?;
        for li in 0..dit.blocks.len() {
            self.block_fwd(dev, dit, lora, li, run, &m, &x, false)?;
        }
        self.head(dev, dit, run, &m, &x)
    }

    fn head(&self, dev: &Device, dit: &Dit, run: &Run, m: &Modulation, x: &Tensor) -> Result<Tensor> {
        let ni = run.h * run.w;
        let y = Tensor::new(dev, DType::BF16, &[ni, DIM])?;
        ops::modrms(dev, &x.rows(0, ni), Some(&dit.final_norm), Some((&m.head_scale, &m.head_shift)), EPS, &y)?;
        let v = Tensor::new(dev, DType::BF16, &[ni, LATENT])?;
        ops::gemm_bf16(dev, &y, &dit.final_linear, None, Act::None, Epi::Store, &v)?;
        Ok(v)
    }

    /// Loss and LoRA gradients (accumulated) for one image: x0 = its normalized latent f32 [h*w, 128], noise f32 of the
    /// same shape, sigma in (0, 1]; the model sees x_t = (1 - sigma) x0 + sigma noise and learns v = noise - x0.
    pub fn step(&self, dev: &Device, dit: &Dit, lora: &LoraSet, run: &Run, x0: &Tensor, noise: &Tensor, sigma: f32) -> Result<StepOut> {
        let ni = run.h * run.w;
        let m = dit.modulation(loki_dream::sampler::timestep(sigma))?;
        let xt = Tensor::new(dev, DType::BF16, &[ni, LATENT])?;
        gpu::noise_mix_bf16(dev, x0, noise, sigma, &xt)?;
        let x = self.embed(dev, dit, run, &xt)?;
        drop(xt);
        // forward, keeping every block input
        let nb = dit.blocks.len();
        let mut inputs: Vec<Tensor> = Vec::with_capacity(nb);
        for li in 0..nb {
            inputs.push(copy(dev, &x)?);
            self.block_fwd(dev, dit, lora, li, run, &m, &x, false)?;
        }
        let v = self.head(dev, dit, run, &m, &x)?;
        let dv = Tensor::new(dev, DType::BF16, &[ni, LATENT])?;
        let loss = gpu::mse(dev, &v, x0, noise, &dv)?;
        drop(v);
        // head backward
        let n = x.shape[0];
        let dx = Tensor::zeros(dev, DType::F32, &[n, DIM])?;
        {
            let dy = Tensor::new(dev, DType::F32, &[ni, DIM])?;
            ops::gemm_bf16(dev, &dv, &self.final_t, None, Act::None, Epi::Store, &dy)?;
            gpu::modrms_bwd(dev, &x.rows(0, ni), &dit.final_norm, Some(&m.head_scale), &dy, &dx.rows(0, ni), EPS)?;
        }
        drop(x);
        // blocks, last to first: recompute the block keeping its intermediates, then differentiate it
        for li in (0..nb).rev() {
            let x_in = inputs.pop().unwrap();
            let xw = copy(dev, &x_in)?;
            let sv = self.block_fwd(dev, dit, lora, li, run, &m, &xw, true)?.unwrap();
            drop(xw);
            self.block_bwd(dev, dit, lora, li, run, &m, &x_in, &sv, &dx).with_context(|| format!("backward of block {li}"))?;
        }
        Ok(StepOut { loss })
    }
}
