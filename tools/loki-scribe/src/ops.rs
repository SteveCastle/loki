//! Rust-side wrappers around the CUDA kernels.
use crate::cuda::{Arg, Device};
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Result};

pub const GEMM_SMEM: u32 = 65536;
pub const ATTN_PREFILL_SMEM: u32 = 2 * 32 * 256 * 4;
pub const VIS_ATTN_SMEM: u32 = 2 * 4 * 64 * 176;

/// Epilogue residual mode of the bf16 GEMM.
#[derive(Clone, Copy)]
pub enum Epi<'a> {
    /// out = v
    Store,
    /// out = res + v
    AddRes(&'a Tensor),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    GeluTanh,
    Silu,
    GeluErf,
}
impl Act {
    fn code(self) -> i32 {
        match self {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::Silu => 2,
            Act::GeluErf => 3,
        }
    }
}

pub fn init(dev: &Device) -> Result<()> {
    for k in ["k_gemm_bf16_bf16", "k_gemm_bf16_f32"] {
        dev.set_max_smem(k, GEMM_SMEM)?;
    }
    dev.set_max_smem("k_attn_prefill", ATTN_PREFILL_SMEM)?;
    dev.set_max_smem("k_attn_split_d72", VIS_ATTN_SMEM)?;
    Ok(())
}

/// bf16 GEMM: out[M,N] = act(a[M,K] . w[N,K]^T + bias) with epilogue. bias is f32 [N].
pub fn gemm_bf16(dev: &Device, a: &Tensor, w: &Tensor, bias: Option<&Tensor>, act: Act, epi: Epi, out: &Tensor) -> Result<()> {
    let m = a.shape[0];
    let k: usize = a.shape[1..].iter().product();
    let n = w.shape[0];
    ensure!(a.dtype == DType::BF16 && w.dtype == DType::BF16, "gemm_bf16 dtypes");
    ensure!(w.numel() / n == k, "gemm_bf16: K mismatch {} vs {}", w.numel() / n, k);
    ensure!(k % 8 == 0 && n % 8 == 0, "gemm_bf16: K%8, N%8 required (K={k}, N={n})");
    ensure!(out.numel() == m * n, "gemm_bf16: out shape {:?} vs {m}x{n}", out.shape);
    let (mode, res) = match epi {
        Epi::Store => (0, 0u64),
        Epi::AddRes(r) => (1, r.ptr),
    };
    let kname = match out.dtype {
        DType::BF16 => "k_gemm_bf16_bf16",
        DType::F32 => "k_gemm_bf16_f32",
        _ => anyhow::bail!("gemm_bf16: bad out dtype"),
    };
    let grid = ((((n + 127) / 128) * ((m + 127) / 128)) as u32, 1, 1);
    dev.launch(
        kname,
        grid,
        (256, 1, 1),
        GEMM_SMEM,
        &[
            Arg::Ptr(a.ptr),
            Arg::Ptr(w.ptr),
            Arg::Ptr(out.ptr),
            Arg::I32(m as i32),
            Arg::I32(n as i32),
            Arg::I32(k as i32),
            Arg::Ptr(bias.map(|b| b.ptr).unwrap_or(0)),
            Arg::I32(act.code()),
            Arg::I32(mode),
            Arg::Ptr(res),
            Arg::Ptr(0),
        ],
    )
}

pub fn to_bf16(dev: &Device, x: &Tensor, out: &Tensor) -> Result<()> {
    ensure!(x.dtype == DType::F32 && out.dtype == DType::BF16 && x.numel() <= out.numel());
    dev.launch_n("k_f32_to_bf16", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(x.numel() as i64)])
}

pub fn act_inplace(dev: &Device, x: &Tensor, act: Act) -> Result<()> {
    let k = match x.dtype {
        DType::F32 => "k_act_f32",
        DType::BF16 => "k_act_bf16",
        _ => anyhow::bail!("act dtype"),
    };
    dev.launch_n(k, x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(x.ptr), Arg::I64(x.numel() as i64), Arg::I32(act.code())])
}

/// RMSNorm over rows of x f32 [M, N] with f32 weight; writes f32 and/or bf16 outputs.
pub fn rmsnorm(dev: &Device, x: &Tensor, m: usize, n: usize, w: &Tensor, eps: f32, yf: Option<&Tensor>, yb: Option<&Tensor>) -> Result<()> {
    dev.launch(
        "k_rmsnorm_w32",
        (m as u32, 1, 1),
        (256, 1, 1),
        0,
        &[Arg::Ptr(x.ptr), Arg::Ptr(yf.map(|t| t.ptr).unwrap_or(0)), Arg::Ptr(yb.map(|t| t.ptr).unwrap_or(0)), Arg::I32(m as i32), Arg::I32(n as i32), Arg::F32(eps), Arg::Ptr(w.ptr)],
    )
}

/// LayerNorm with f32 weight and bias: x f32 [M, N] -> y f32.
pub fn layernorm(dev: &Device, x: &Tensor, m: usize, n: usize, w: &Tensor, b: &Tensor, eps: f32, y: &Tensor) -> Result<()> {
    dev.launch("k_layernorm_w32", (m as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x.ptr), Arg::Ptr(y.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::F32(eps), Arg::Ptr(w.ptr), Arg::Ptr(b.ptr)])
}

pub fn silu_mul_f32(dev: &Device, g: &Tensor, u: &Tensor, h: &Tensor, n: usize) -> Result<()> {
    dev.launch_n("k_silu_mul_f32", n, &[Arg::Ptr(g.ptr), Arg::Ptr(u.ptr), Arg::Ptr(h.ptr), Arg::I64(n as i64)])
}
pub fn silu_mul_bf16(dev: &Device, g: &Tensor, u: &Tensor, h: &Tensor, n: usize) -> Result<()> {
    dev.launch_n("k_silu_mul_bf16", n, &[Arg::Ptr(g.ptr), Arg::Ptr(u.ptr), Arg::Ptr(h.ptr), Arg::I64(n as i64)])
}

/// A view of `rows` rows of a contiguous buffer reinterpreted with a new row width (scratch reuse).
pub fn view(t: &Tensor, dtype: DType, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    assert!(n * dtype.size() <= t.buf.len - (t.ptr - t.buf.ptr()) as usize, "view {shape:?} {dtype:?} exceeds buffer");
    Tensor { buf: t.buf.clone(), ptr: t.ptr, dtype, shape: shape.to_vec() }
}

/// A view at an element offset.
pub fn view_at(t: &Tensor, elem_off: usize, dtype: DType, shape: &[usize]) -> Tensor {
    let ptr = t.ptr + (elem_off * t.dtype.size()) as u64;
    let n: usize = shape.iter().product();
    assert!((ptr - t.buf.ptr()) as usize + n * dtype.size() <= t.buf.len, "view_at {shape:?} exceeds buffer");
    Tensor { buf: t.buf.clone(), ptr, dtype, shape: shape.to_vec() }
}
