//! Rust-side wrappers around the CUDA kernels.
use crate::cuda::{Arg, Device};
use crate::tensor::{DType, Tensor};
use crate::weights::{Conv, Linear, Norm};
use anyhow::{bail, ensure, Result};
use cudarc::driver::PushKernelArg;

pub const GEMM_SMEM: u32 = 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    GeluTanh,
    Silu,
    GeluErf,
    Relu,
    Sigmoid,
}
impl Act {
    pub fn code(self) -> i32 {
        match self {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::Silu => 2,
            Act::GeluErf => 3,
            Act::Relu => 4,
            Act::Sigmoid => 5,
        }
    }
}

/// GEMM epilogue residual mode.
#[derive(Clone, Copy)]
pub enum Epi<'a> {
    Store,
    /// out = res + v  (res has the output's dtype; may alias out)
    AddRes(&'a Tensor),
}

pub fn init(dev: &Device) -> Result<()> {
    for k in ["k_gemm_bf16_bf16", "k_gemm_bf16_f32", "k_conv2d_nhwc"] {
        dev.set_max_smem(k, GEMM_SMEM)?;
    }
    Ok(())
}

/// out[M,N] = act(a[M,K] . w[N,K]^T + bias) (+ res). a bf16 contiguous; out bf16 or f32.
pub fn gemm(dev: &Device, a: &Tensor, w: &Tensor, bias: Option<&Tensor>, act: Act, epi: Epi, out: &Tensor) -> Result<()> {
    let k = *w.shape.last().unwrap();
    let n = w.numel() / k;
    let m = a.numel() / k;
    ensure!(a.dtype == DType::BF16 && w.dtype == DType::BF16, "gemm dtypes {:?} {:?}", a.dtype, w.dtype);
    ensure!(a.numel() == m * k, "gemm: a {:?} not a multiple of K={k}", a.shape);
    ensure!(k % 8 == 0 && n % 8 == 0, "gemm: K%8, N%8 required (K={k}, N={n})");
    ensure!(out.numel() == m * n, "gemm: out {:?} vs {m}x{n}", out.shape);
    let (mode, res) = match epi {
        Epi::Store => (0, 0u64),
        Epi::AddRes(r) => {
            ensure!(r.dtype == out.dtype && r.numel() == out.numel(), "gemm residual shape/dtype");
            (1, r.ptr)
        }
    };
    let kname = match out.dtype {
        DType::BF16 => "k_gemm_bf16_bf16",
        DType::F32 => "k_gemm_bf16_f32",
        _ => bail!("gemm: bad out dtype"),
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

/// Linear layer: out = act(x W^T + b) (+ res).
pub fn linear(dev: &Device, x: &Tensor, l: &Linear, act: Act, epi: Epi, out: &Tensor) -> Result<()> {
    gemm(dev, x, &l.w, l.b.as_ref(), act, epi, out)
}
/// Linear into a fresh tensor of `dtype`.
pub fn linear_new(dev: &Device, x: &Tensor, l: &Linear, act: Act, dtype: DType) -> Result<Tensor> {
    let m = x.numel() / l.k;
    let out = Tensor::new(dev, dtype, &[m, l.n])?;
    linear(dev, x, l, act, Epi::Store, &out)?;
    Ok(out)
}

fn dcode(t: DType) -> Result<i32> {
    match t {
        DType::F32 => Ok(0),
        DType::BF16 => Ok(1),
        _ => bail!("unsupported dtype {t:?}"),
    }
}

/// out = act(a + b) elementwise; b is broadcast with period `b.numel()` when smaller than a.
pub fn add(dev: &Device, a: &Tensor, b: Option<&Tensor>, act: Act, out: &Tensor) -> Result<()> {
    ensure!(out.numel() == a.numel(), "add: out size");
    let (bp, bt, period) = match b {
        Some(b) => {
            ensure!(a.numel() % b.numel() == 0, "add: b {:?} does not tile a {:?}", b.shape, a.shape);
            (b.ptr, dcode(b.dtype)?, if b.numel() == a.numel() { 0 } else { b.numel() as i64 })
        }
        None => (0, 0, 0),
    };
    dev.launch_n(
        "k_add_any",
        a.numel(),
        &[
            Arg::Ptr(a.ptr),
            Arg::I32(dcode(a.dtype)?),
            Arg::Ptr(bp),
            Arg::I32(bt),
            Arg::I64(period),
            Arg::Ptr(out.ptr),
            Arg::I32(dcode(out.dtype)?),
            Arg::I64(a.numel() as i64),
            Arg::I32(act.code()),
        ],
    )
}
pub fn add_new(dev: &Device, a: &Tensor, b: Option<&Tensor>, act: Act, dtype: DType) -> Result<Tensor> {
    let out = Tensor::new(dev, dtype, &a.shape)?;
    add(dev, a, b, act, &out)?;
    Ok(out)
}
pub fn fill_f32(dev: &Device, t: &Tensor, v: f32) -> Result<()> {
    ensure!(t.dtype == DType::F32, "fill_f32 dtype");
    dev.launch_n("k_fill_f32", t.numel(), &[Arg::Ptr(t.ptr), Arg::F32(v), Arg::I64(t.numel() as i64)])
}
pub fn to_bf16(dev: &Device, a: &Tensor) -> Result<Tensor> {
    add_new(dev, a, None, Act::None, DType::BF16)
}
pub fn to_f32(dev: &Device, a: &Tensor) -> Result<Tensor> {
    add_new(dev, a, None, Act::None, DType::F32)
}

/// LayerNorm over the last dim. Optional second output y2 = y + pos (pos [M,N] or [N]).
pub fn layernorm(dev: &Device, x: &Tensor, norm: Option<&Norm>, eps: f32, out: &Tensor, pos: Option<(&Tensor, &Tensor)>) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    ensure!(out.numel() == x.numel(), "layernorm out size");
    let k = match (x.dtype, out.dtype) {
        (DType::F32, DType::BF16) => "k_layernorm_f32_bf16",
        (DType::F32, DType::F32) => "k_layernorm_f32_f32",
        (DType::BF16, DType::BF16) => "k_layernorm_bf16_bf16",
        (DType::BF16, DType::F32) => "k_layernorm_bf16_f32",
        _ => bail!("layernorm dtypes"),
    };
    let (pp, prow, y2) = match pos {
        Some((p, y2)) => {
            ensure!(y2.dtype == out.dtype && y2.numel() == out.numel());
            (p.ptr, if p.numel() == n { 1 } else { 0 }, y2.ptr)
        }
        None => (0, 0, 0),
    };
    let eps = norm.map(|n| n.eps).unwrap_or(eps);
    dev.launch(
        k,
        (m as u32, 1, 1),
        (256, 1, 1),
        0,
        &[
            Arg::Ptr(x.ptr),
            Arg::Ptr(out.ptr),
            Arg::I32(m as i32),
            Arg::I32(n as i32),
            Arg::F32(eps),
            Arg::Ptr(norm.map(|t| t.w.ptr).unwrap_or(0)),
            Arg::Ptr(norm.map(|t| t.b.ptr).unwrap_or(0)),
            Arg::Ptr(pp),
            Arg::I32(prow),
            Arg::Ptr(y2),
        ],
    )
}
pub fn layernorm_new(dev: &Device, x: &Tensor, norm: &Norm, dtype: DType) -> Result<Tensor> {
    let out = Tensor::new(dev, dtype, &x.shape)?;
    layernorm(dev, x, Some(norm), norm.eps, &out, None)?;
    Ok(out)
}

/// GroupNorm (NHWC bf16 [P, C], G groups) + activation.
pub fn groupnorm(dev: &Device, x: &Tensor, groups: usize, norm: &Norm, act: Act, out: &Tensor) -> Result<()> {
    let c = *x.shape.last().unwrap();
    let p = x.numel() / c;
    ensure!(x.dtype == DType::BF16 && out.dtype == DType::BF16 && groups <= 32);
    let stats = dev.alloc_zeros(groups * 2 * 8)?;
    dev.launch("k_groupnorm_stats", (512, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x.ptr), Arg::I64(p as i64), Arg::I32(c as i32), Arg::I32(groups as i32), Arg::Ptr(stats.ptr())])?;
    dev.launch_n(
        "k_groupnorm_apply",
        p * c,
        &[
            Arg::Ptr(x.ptr),
            Arg::Ptr(out.ptr),
            Arg::I64(p as i64),
            Arg::I32(c as i32),
            Arg::I32(groups as i32),
            Arg::Ptr(stats.ptr()),
            Arg::F32(norm.eps),
            Arg::Ptr(norm.w.ptr),
            Arg::Ptr(norm.b.ptr),
            Arg::I32(act.code()),
        ],
    )
}

/// Conv2d (NHWC bf16 in [IH*IW, Ci] -> out [OH*OW, Co]), stride / padding symmetric.
pub fn conv2d(dev: &Device, x: &Tensor, ih: usize, iw: usize, c: &Conv, stride: usize, pad: usize, out: &Tensor) -> Result<(usize, usize)> {
    ensure!(x.dtype == DType::BF16 && out.dtype == DType::BF16, "conv2d dtypes");
    ensure!(c.ci % 8 == 0, "conv2d: Ci % 8 required (Ci={})", c.ci);
    ensure!(x.numel() == ih * iw * c.ci, "conv2d input {:?} vs {ih}x{iw}x{}", x.shape, c.ci);
    let oh = (ih + 2 * pad - c.kh) / stride + 1;
    let ow = (iw + 2 * pad - c.kw) / stride + 1;
    ensure!(out.numel() == oh * ow * c.co, "conv2d out size");
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ConvParams {
        inp: u64,
        ih: i32,
        iw: i32,
        ci: i32,
        _p0: i32,
        w: u64,
        bias: u64,
        co: i32,
        kpad: i32,
        out: u64,
        oh: i32,
        ow: i32,
        kh: i32,
        kw: i32,
        stride: i32,
        pad_t: i32,
        pad_l: i32,
        up2: i32,
        mode: i32,
        _p1: i32,
        res: u64,
    }
    unsafe impl cudarc::driver::DeviceRepr for ConvParams {}
    let p = ConvParams {
        inp: x.ptr,
        ih: ih as i32,
        iw: iw as i32,
        ci: c.ci as i32,
        _p0: 0,
        w: c.w.ptr,
        bias: c.b.as_ref().map(|b| b.ptr).unwrap_or(0),
        co: c.co as i32,
        kpad: c.kpad as i32,
        out: out.ptr,
        oh: oh as i32,
        ow: ow as i32,
        kh: c.kh as i32,
        kw: c.kw as i32,
        stride: stride as i32,
        pad_t: pad as i32,
        pad_l: pad as i32,
        up2: 0,
        mode: 0,
        _p1: 0,
        res: 0,
    };
    let m = oh * ow;
    let grid = ((((c.co + 127) / 128) * ((m + 127) / 128)) as u32, 1, 1);
    let f = dev.func("k_conv2d_nhwc")?;
    let cfg = cudarc::driver::LaunchConfig { grid_dim: grid, block_dim: (256, 1, 1), shared_mem_bytes: GEMM_SMEM };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&p);
    unsafe { b.launch(cfg) }?;
    Ok((oh, ow))
}

/// ConvTranspose2d(k=2, s=2) on NHWC bf16 [H*W, Ci] -> [(2H)*(2W), Co] via GEMM + pixel shuffle; optional act (GeluErf).
pub fn conv_t2(dev: &Device, x: &Tensor, h: usize, w: usize, l: &Linear, act: Act) -> Result<Tensor> {
    let co = l.n / 4;
    let g = linear_new(dev, x, l, Act::None, DType::BF16)?;
    let out = Tensor::new(dev, DType::BF16, &[4 * h * w, co])?;
    dev.launch_n("k_pixel_shuffle2", h * w * 4 * co, &[Arg::Ptr(g.ptr), Arg::Ptr(out.ptr), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(co as i32), Arg::I32(act.code())])?;
    Ok(out)
}

/// out = a + nearest_up2(b) (bf16 NHWC; out H x W x C, b (H/2) x (W/2) x C)
pub fn upadd2(dev: &Device, a: &Tensor, b: &Tensor, h: usize, w: usize, out: &Tensor) -> Result<()> {
    let c = *a.shape.last().unwrap();
    ensure!(a.numel() == h * w * c && b.numel() == (h / 2) * (w / 2) * c);
    dev.launch_n("k_upadd2", h * w * c, &[Arg::Ptr(a.ptr), Arg::Ptr(b.ptr), Arg::Ptr(out.ptr), Arg::I32(h as i32), Arg::I32(w as i32), Arg::I32(c as i32)])
}

// =====================================================================================
// Attention
// =====================================================================================

/// One side of an attention call: a token-major bf16 tensor with explicit strides (in elements).
#[derive(Clone, Copy, Debug)]
pub struct AttnView {
    pub ptr: u64,
    pub tok_stride: usize,
    pub head_stride: usize,
    pub len: usize,
    /// batch stride (elements), used with `batch > 1`
    pub batch_stride: usize,
}
impl AttnView {
    /// Contiguous [len, H*D] tensor (or a column slice at element offset `col`, with row stride `ts`).
    pub fn new(t: &Tensor, col: usize, ts: usize, d: usize, len: usize) -> AttnView {
        AttnView { ptr: t.ptr + (col * 2) as u64, tok_stride: ts, head_stride: d, len, batch_stride: 0 }
    }
}

pub struct AttnArgs<'a> {
    pub q: AttnView,
    pub k: AttnView,
    pub v: AttnView,
    pub out: AttnView,
    pub heads: usize,
    pub d: usize,
    pub batch: usize,
    pub causal: bool,
    /// u8 [len_k], 1 = visible
    pub kmask: Option<&'a Tensor>,
    /// f32 bias, (tensor, head stride, query stride) in elements; key index is the innermost
    pub bias: Option<(&'a Tensor, usize, usize)>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParamsRaw {
    q: u64,
    q_ts: i64,
    q_hs: i32,
    _p0: i32,
    q_bs: i64,
    k: u64,
    v: u64,
    kv_ts: i64,
    kv_hs: i32,
    len: i32,
    kv_bs: i64,
    o: u64,
    o_ts: i64,
    o_hs: i32,
    _p1: i32,
    o_bs: i64,
    nq: i32,
    hq: i32,
    hk: i32,
    d_real: i32,
    kmask: u64,
    bias: u64,
    bias_hs: i64,
    bias_qs: i64,
    causal: i32,
    scale_log2: f32,
}
unsafe impl cudarc::driver::DeviceRepr for AttnParamsRaw {}

pub fn attn_init(dev: &Device) -> Result<()> {
    dev.set_max_smem("k_flash_attn_d128", 128 * 256 + 4 * 64 * 256)?;
    dev.set_max_smem("k_flash_attn_d64", 128 * 128 + 4 * 64 * 128)?;
    dev.set_max_smem("k_flash_attn_d32", 128 * 64 + 4 * 64 * 64)?;
    Ok(())
}

pub fn attention(dev: &Device, a: &AttnArgs) -> Result<()> {
    ensure!(a.k.tok_stride == a.v.tok_stride && a.k.head_stride == a.v.head_stride && a.k.batch_stride == a.v.batch_stride, "attention: K and V must share strides");
    ensure!(a.d % 8 == 0 && a.d <= 128, "attention: head dim {} unsupported", a.d);
    let (kname, rowb) = if a.d > 64 {
        ("k_flash_attn_d128", 256)
    } else if a.d > 32 {
        ("k_flash_attn_d64", 128)
    } else {
        ("k_flash_attn_d32", 64)
    };
    let smem = 128 * rowb + 4 * 64 * rowb;
    let p = AttnParamsRaw {
        q: a.q.ptr,
        q_ts: a.q.tok_stride as i64,
        q_hs: a.q.head_stride as i32,
        _p0: 0,
        q_bs: a.q.batch_stride as i64,
        k: a.k.ptr,
        v: a.v.ptr,
        kv_ts: a.k.tok_stride as i64,
        kv_hs: a.k.head_stride as i32,
        len: a.k.len as i32,
        kv_bs: a.k.batch_stride as i64,
        o: a.out.ptr,
        o_ts: a.out.tok_stride as i64,
        o_hs: a.out.head_stride as i32,
        _p1: 0,
        o_bs: a.out.batch_stride as i64,
        nq: a.q.len as i32,
        hq: a.heads as i32,
        hk: a.heads as i32,
        d_real: a.d as i32,
        kmask: a.kmask.map(|t| t.ptr).unwrap_or(0),
        bias: a.bias.map(|b| b.0.ptr).unwrap_or(0),
        bias_hs: a.bias.map(|b| b.1 as i64).unwrap_or(0),
        bias_qs: a.bias.map(|b| b.2 as i64).unwrap_or(0),
        causal: a.causal as i32,
        scale_log2: (1.0 / (a.d as f32).sqrt()) * std::f32::consts::LOG2_E,
    };
    let f = dev.func(kname)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((a.q.len + 127) / 128) as u32, a.heads as u32, a.batch.max(1) as u32),
        block_dim: (256, 1, 1),
        shared_mem_bytes: smem as u32,
    };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&p);
    unsafe { b.launch(cfg) }?;
    Ok(())
}

/// Standard multi-head attention over contiguous bf16 [nq, E] / [nk, E] inputs (already projected).
pub fn mha(dev: &Device, q: &Tensor, k: &Tensor, v: &Tensor, heads: usize, kmask: Option<&Tensor>, bias: Option<(&Tensor, usize, usize)>) -> Result<Tensor> {
    let e = *q.shape.last().unwrap();
    let nq = q.numel() / e;
    let nk = k.numel() / e;
    let d = e / heads;
    let out = Tensor::new(dev, DType::BF16, &[nq, e])?;
    attention(
        dev,
        &AttnArgs {
            q: AttnView::new(q, 0, e, d, nq),
            k: AttnView::new(k, 0, e, d, nk),
            v: AttnView::new(v, 0, e, d, nk),
            out: AttnView::new(&out, 0, e, d, nq),
            heads,
            d,
            batch: 1,
            causal: false,
            kmask,
            bias,
        },
    )?;
    Ok(out)
}
