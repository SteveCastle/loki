//! The CUDA device with loki-tune's training kernels loaded next to loki-dream's, and launch wrappers for them.
use anyhow::{ensure, Result};
use loki_dream::cuda::{Arg, Device};
use loki_dream::tensor::{DType, Tensor};
use std::sync::Arc;

const FATBINS: &[(&str, &[u8])] = &[
    ("train", include_bytes!(concat!(env!("OUT_DIR"), "/train.fatbin"))),
    ("attn_train", include_bytes!(concat!(env!("OUT_DIR"), "/attn_train.fatbin"))),
];

const ROWB: u32 = 256;
const ATTN_FWD_SMEM: u32 = 128 * ROWB + 4 * 64 * ROWB;
const ATTN_DKDV_SMEM: u32 = 2 * 128 * ROWB + 4 * 32 * ROWB + 4 * 32 * 4;
const ATTN_DQ_SMEM: u32 = 2 * 128 * ROWB + 4 * 32 * ROWB;

pub fn device() -> Result<Arc<Device>> {
    let dev = Device::with_modules(0, FATBINS)?;
    loki_dream::ops::gemm_init(&dev)?;
    loki_dream::ops::attn_init(&dev)?;
    loki_dream::vae::Vae::init_kernels(&dev)?;
    dev.set_max_smem("k_attn_fwd_lse", ATTN_FWD_SMEM)?;
    dev.set_max_smem("k_attn_bwd_dkdv", ATTN_DKDV_SMEM)?;
    dev.set_max_smem("k_attn_bwd_dq", ATTN_DQ_SMEM)?;
    Ok(dev)
}

fn rows_cols(t: &Tensor) -> (usize, usize) {
    let c = *t.shape.last().unwrap();
    (t.numel() / c, c)
}

/// dx += modrms_backward(x, dy)   (x bf16 [M, K], dy / dx f32 [M, K]; `scale` = the modulation scale, None for a plain norm)
pub fn modrms_bwd(dev: &Device, x: &Tensor, w: &Tensor, scale: Option<&Tensor>, dy: &Tensor, dx: &Tensor, eps: f32) -> Result<()> {
    let (m, k) = rows_cols(x);
    ensure!(dy.dtype == DType::F32 && dx.dtype == DType::F32 && dy.numel() == m * k && dx.numel() == m * k, "modrms_bwd shapes");
    dev.launch("k_modrms_bwd", (m as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x.ptr), Arg::Ptr(w.ptr), Arg::Ptr(scale.map(|s| s.ptr).unwrap_or(0)), Arg::Ptr(dy.ptr), Arg::Ptr(dx.ptr), Arg::I32(m as i32), Arg::I32(k as i32), Arg::F32(eps)])
}

/// In place on the q/k columns of `g` [M, ts]: gradient w.r.t. the rotated, normed q/k -> w.r.t. the projection `pre`.
#[allow(clippy::too_many_arguments)]
pub fn qk_norm_rope_bwd(dev: &Device, pre: &Tensor, g: &Tensor, ts: usize, m: usize, h: usize, wq: &Tensor, wk: &Tensor, eps: f32, rope: &Tensor) -> Result<()> {
    dev.launch_n("k_qk_norm_rope_bwd", m * h * 2 * 32, &[Arg::Ptr(pre.ptr), Arg::Ptr(g.ptr), Arg::I64(ts as i64), Arg::I32(m as i32), Arg::I32(h as i32), Arg::Ptr(wq.ptr), Arg::Ptr(wk.ptr), Arg::F32(eps), Arg::Ptr(rope.ptr)])
}

/// dug [M, 2F] bf16 from ds [M, F] f32 through situ(ug)
pub fn situ_bwd(dev: &Device, ug: &Tensor, ds: &Tensor, dug: &Tensor) -> Result<()> {
    let (m, f2) = rows_cols(ug);
    ensure!(ds.dtype == DType::F32 && ds.numel() == m * f2 / 2 && dug.numel() == m * f2, "situ_bwd shapes");
    dev.launch_n("k_situ_bwd", m * f2 / 2, &[Arg::Ptr(ug.ptr), Arg::Ptr(ds.ptr), Arg::Ptr(dug.ptr), Arg::I64(m as i64), Arg::I32((f2 / 2) as i32)])
}

/// out bf16 = x f32 * g[col]
pub fn scale_cols_bf16(dev: &Device, x: &Tensor, g: &Tensor, out: &Tensor) -> Result<()> {
    let (m, n) = rows_cols(x);
    ensure!(out.numel() == m * n && out.dtype == DType::BF16);
    dev.launch_n("k_scale_cols_bf16", m * n, &[Arg::Ptr(x.ptr), Arg::Ptr(g.ptr), Arg::Ptr(out.ptr), Arg::I64(m as i64), Arg::I32(n as i32)])
}

/// Hadamard-256 per group, in place on f32 rows (ConvRot's rotation is its own inverse).
pub fn fwht256(dev: &Device, x: &Tensor) -> Result<()> {
    let (m, k) = rows_cols(x);
    ensure!(x.dtype == DType::F32 && k % 256 == 0);
    dev.launch("k_fwht256_f32", (m as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x.ptr), Arg::I32(m as i32), Arg::I32(k as i32)])
}

/// out [K, N] bf16 = W^T for a weight [N, K] (bf16, or int8 with per-row `scale`).
pub fn transpose_w(dev: &Device, w: &Tensor, scale: Option<&Tensor>, out: &Tensor) -> Result<()> {
    let (n, k) = (w.shape[0], w.numel() / w.shape[0]);
    ensure!(out.numel() == n * k && out.dtype == DType::BF16);
    let i8 = w.dtype == DType::I8;
    ensure!(i8 == scale.is_some(), "transpose_w: int8 weights need their scale");
    let grid = (k.div_ceil(32) as u32, n.div_ceil(32) as u32, 1);
    dev.launch("k_transpose_w", grid, (32, 8, 1), 0, &[Arg::Ptr(w.ptr), Arg::I32(i8 as i32), Arg::Ptr(scale.map(|s| s.ptr).unwrap_or(0)), Arg::Ptr(out.ptr), Arg::I32(n as i32), Arg::I32(k as i32)])
}

/// bf16(src * s): [R, C] as is or transposed to [C, R]. `src` may be a view into the flat parameter buffer.
pub fn lora_cast(dev: &Device, src: u64, out: &Tensor, r: usize, c: usize, s: f32, transpose: bool) -> Result<()> {
    ensure!(out.numel() == r * c && out.dtype == DType::BF16);
    dev.launch_n("k_lora_cast", r * c, &[Arg::Ptr(src), Arg::Ptr(out.ptr), Arg::I32(r as i32), Arg::I32(c as i32), Arg::F32(s), Arg::I32(transpose as i32)])
}

pub fn mask_blockdiag(dev: &Device, g: u64, dm: usize, r: usize) -> Result<()> {
    dev.launch_n("k_mask_blockdiag", 9 * dm * r, &[Arg::Ptr(g), Arg::I32(dm as i32), Arg::I32(r as i32)])
}

/// C[P, Q] (f32, at `c`) += alpha * U^T V  (U bf16 [M, P], V bf16 [M, Q], both contiguous rows)
pub fn gemm_tn(dev: &Device, u: &Tensor, v: &Tensor, c: u64, alpha: f32) -> Result<()> {
    let (m, p) = rows_cols(u);
    let (m2, q) = rows_cols(v);
    ensure!(m == m2 && u.dtype == DType::BF16 && v.dtype == DType::BF16, "gemm_tn shapes {:?} {:?}", u.shape, v.shape);
    let grid = (q.div_ceil(64) as u32, p.div_ceil(64) as u32, 1);
    dev.launch("k_gemm_tn", grid, (128, 1, 1), 0, &[Arg::Ptr(u.ptr), Arg::I64(p as i64), Arg::Ptr(v.ptr), Arg::I64(q as i64), Arg::Ptr(c), Arg::I32(m as i32), Arg::I32(p as i32), Arg::I32(q as i32), Arg::F32(alpha)])
}

pub fn noise_mix_bf16(dev: &Device, x0: &Tensor, eps: &Tensor, s: f32, out: &Tensor) -> Result<()> {
    let n = x0.numel();
    dev.launch_n("k_noise_mix_bf16", n, &[Arg::Ptr(x0.ptr), Arg::Ptr(eps.ptr), Arg::F32(s), Arg::Ptr(out.ptr), Arg::I64(n as i64)])
}

/// MSE(v, eps - x0) over all elements: writes the gradient (bf16) and returns the loss.
pub fn mse(dev: &Device, v: &Tensor, x0: &Tensor, eps: &Tensor, grad: &Tensor) -> Result<f32> {
    let n = v.numel();
    let blocks = n.div_ceil(256);
    let partial = Tensor::new(dev, DType::F32, &[blocks])?;
    dev.launch_n("k_mse", n, &[Arg::Ptr(v.ptr), Arg::Ptr(x0.ptr), Arg::Ptr(eps.ptr), Arg::Ptr(grad.ptr), Arg::Ptr(partial.ptr), Arg::I64(n as i64)])?;
    let p = partial.to_f32_vec(dev)?;
    Ok((p.iter().map(|&x| x as f64).sum::<f64>() / n as f64) as f32)
}

/// sum of squares of an f32 buffer (`n` values at `x`)
pub fn sumsq(dev: &Device, x: u64, n: usize) -> Result<f64> {
    let blocks = n.div_ceil(256).min(4096);
    let partial = Tensor::new(dev, DType::F32, &[blocks])?;
    dev.launch("k_sumsq", (blocks as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x), Arg::Ptr(partial.ptr), Arg::I64(n as i64)])?;
    Ok(partial.to_f32_vec(dev)?.iter().map(|&x| x as f64).sum())
}

#[allow(clippy::too_many_arguments)]
pub fn adamw(dev: &Device, p: u64, g: u64, m: u64, v: u64, n: usize, lr: f32, b1: f32, b2: f32, eps: f32, wd: f32, step: usize, gscale: f32) -> Result<()> {
    let bc1 = 1.0 - b1.powi(step as i32);
    let bc2 = 1.0 - b2.powi(step as i32);
    dev.launch_n("k_adamw", n, &[Arg::Ptr(p), Arg::Ptr(g), Arg::Ptr(m), Arg::Ptr(v), Arg::I64(n as i64), Arg::F32(lr), Arg::F32(b1), Arg::F32(b2), Arg::F32(eps), Arg::F32(wd), Arg::F32(bc1), Arg::F32(bc2), Arg::F32(gscale)])
}

// ---------------------------------------------------------------------------------------------------------- attention
pub const HEADS: usize = 24;
pub const HEAD_DIM: usize = 128;

fn scales() -> (f32, f32) {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    (scale, scale * std::f32::consts::LOG2_E)
}

/// Joint attention over n tokens of a fused qkv [n, 3*DIM] -> out [n, DIM] bf16 and lse2 [H, n] f32.
pub fn attn_fwd(dev: &Device, qkv: &Tensor, n: usize, out: &Tensor, lse2: &Tensor) -> Result<()> {
    let dim = HEADS * HEAD_DIM;
    let ts = 3 * dim;
    let (_, sl2) = scales();
    let grid = (n.div_ceil(128) as u32, HEADS as u32, 1);
    dev.launch(
        "k_attn_fwd_lse",
        grid,
        (256, 1, 1),
        ATTN_FWD_SMEM,
        &[Arg::Ptr(qkv.ptr), Arg::Ptr(qkv.ptr + (dim * 2) as u64), Arg::Ptr(qkv.ptr + (2 * dim * 2) as u64), Arg::I64(ts as i64), Arg::Ptr(out.ptr), Arg::I64(dim as i64), Arg::Ptr(lse2.ptr), Arg::I32(n as i32), Arg::F32(sl2)],
    )
}

/// Attention backward: from the forward's qkv, out, lse2 and the output gradient `dout` [n, DIM] bf16,
/// write dq | dk | dv into `dqkv` [n, 3*DIM] bf16.
pub fn attn_bwd(dev: &Device, qkv: &Tensor, out: &Tensor, lse2: &Tensor, dout: &Tensor, n: usize, dqkv: &Tensor) -> Result<()> {
    let dim = HEADS * HEAD_DIM;
    let ts = 3 * dim;
    let (scale, sl2) = scales();
    let dsum = Tensor::new(dev, DType::F32, &[HEADS, n])?;
    dev.launch_n("k_attn_bwd_prep", n * HEADS * 32, &[Arg::Ptr(dout.ptr), Arg::I64(dim as i64), Arg::Ptr(out.ptr), Arg::I64(dim as i64), Arg::Ptr(dsum.ptr), Arg::I32(n as i32), Arg::I32(HEADS as i32)])?;
    let (q, k, v) = (qkv.ptr, qkv.ptr + (dim * 2) as u64, qkv.ptr + (2 * dim * 2) as u64);
    let grid = (n.div_ceil(128) as u32, HEADS as u32, 1);
    dev.launch(
        "k_attn_bwd_dkdv",
        grid,
        (256, 1, 1),
        ATTN_DKDV_SMEM,
        &[
            Arg::Ptr(q),
            Arg::Ptr(k),
            Arg::Ptr(v),
            Arg::I64(ts as i64),
            Arg::Ptr(dout.ptr),
            Arg::I64(dim as i64),
            Arg::Ptr(lse2.ptr),
            Arg::Ptr(dsum.ptr),
            Arg::Ptr(dqkv.ptr + (dim * 2) as u64),
            Arg::Ptr(dqkv.ptr + (2 * dim * 2) as u64),
            Arg::I64(ts as i64),
            Arg::I32(n as i32),
            Arg::F32(sl2),
            Arg::F32(scale),
        ],
    )?;
    dev.launch(
        "k_attn_bwd_dq",
        grid,
        (256, 1, 1),
        ATTN_DQ_SMEM,
        &[Arg::Ptr(q), Arg::Ptr(k), Arg::Ptr(v), Arg::I64(ts as i64), Arg::Ptr(dout.ptr), Arg::I64(dim as i64), Arg::Ptr(lse2.ptr), Arg::Ptr(dsum.ptr), Arg::Ptr(dqkv.ptr), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::F32(sl2), Arg::F32(scale)],
    )
}
