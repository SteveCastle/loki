//! Rust-side wrappers around the CUDA kernels.
use crate::cuda::{Arg, Device};
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Result};
use cudarc::driver::PushKernelArg;

pub const GEMM_SMEM: u32 = 65536;

/// Input activation folded into the int8 quantizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantAct {
    None,
    GeluTanh,
    SwiGlu,
    RmsNorm,
}

/// Epilogue residual mode shared by both GEMM kernels.
#[derive(Clone, Copy)]
pub enum Epi<'a> {
    /// out = v
    Store,
    /// out = res + v
    AddRes(&'a Tensor),
    /// out = res + v * gate[n]
    AddResGated(&'a Tensor, &'a Tensor),
    /// out[m][n/2] = silu(v[n]) * v[n+1] (weights interleaved gate/up; output width N/2)
    SwiGluPairs,
}

pub const GEMM_SMEM_W: u32 = 3 * (128 * 64 + 256 * 64);
/// Use the 128x256 int8 tile when N is large enough to fill it well.
pub fn gemm_i8_wide(m: usize, n: usize) -> bool {
    n % 256 == 0 && n >= 2048 && m >= 1024
}

pub fn gemm_init(dev: &Device) -> Result<()> {
    for k in ["k_gemm_i8_bf16", "k_gemm_i8_f32", "k_gemm_bf16_bf16", "k_gemm_bf16_f32"] {
        dev.set_max_smem(k, GEMM_SMEM)?;
    }
    for k in ["k_gemm_i8_bf16_w", "k_gemm_i8_f32_w"] {
        dev.set_max_smem(k, GEMM_SMEM_W)?;
    }
    for k in ["k_quant_rows_f32", "k_quant_rows_bf16"] {
        dev.set_max_smem(k, 96 * 1024)?;
    }
    Ok(())
}

/// Quantize rows: `x` [M, Kin] (f32 or bf16) -> (int8 [M, K], scale f32 [M]).
/// For SwiGlu K = Kin/2. `w`/`eps` only used for RmsNorm.
pub fn quant_rows(dev: &Device, x: &Tensor, act: QuantAct, w: Option<&Tensor>, eps: f32, out_q: &Tensor, out_s: &Tensor) -> Result<()> {
    let m = x.shape[0];
    let kin: usize = x.shape[1..].iter().product();
    let k = if act == QuantAct::SwiGlu { kin / 2 } else { kin };
    ensure!(k % 256 == 0, "quant_rows: K={k} must be a multiple of 256");
    ensure!(out_q.dtype == DType::I8 && out_q.numel() == m * k, "quant_rows out_q shape");
    ensure!(out_s.numel() >= m, "quant_rows out_s shape");
    let act_code = match act {
        QuantAct::None => 0,
        QuantAct::GeluTanh => 1,
        QuantAct::SwiGlu => 2,
        QuantAct::RmsNorm => 3,
    };
    let kname = match x.dtype {
        DType::F32 => "k_quant_rows_f32",
        DType::BF16 => "k_quant_rows_bf16",
        _ => anyhow::bail!("quant_rows: unsupported dtype"),
    };
    let smem = (k * 4) as u32;
    ensure!(smem <= 96 * 1024, "quant_rows: K={k} too large");
    dev.launch(
        kname,
        (m as u32, 1, 1),
        (256, 1, 1),
        smem,
        &[
            Arg::Ptr(x.ptr),
            Arg::I32(m as i32),
            Arg::I32(kin as i32),
            Arg::I32(act_code),
            Arg::Ptr(w.map(|t| t.ptr).unwrap_or(0)),
            Arg::F32(eps),
            Arg::Ptr(out_q.ptr),
            Arg::Ptr(out_s.ptr),
        ],
    )
}

/// AdaLN fused into the int8 quantizer: q = quant(rotate(LN(x) * (1 + scale)))
pub fn adaln_quant(dev: &Device, x: &Tensor, scale: &Tensor, eps: f32, out_q: &Tensor, out_s: &Tensor) -> Result<()> {
    let m = x.shape[0];
    let k: usize = x.shape[1..].iter().product();
    ensure!(x.dtype == DType::BF16 && k % 256 == 0);
    let smem = (k * 4) as u32;
    dev.launch("k_adaln_quant_bf16", (m as u32, 1, 1), (256, 1, 1), smem, &[Arg::Ptr(x.ptr), Arg::I32(m as i32), Arg::I32(k as i32), Arg::F32(eps), Arg::Ptr(scale.ptr), Arg::Ptr(out_q.ptr), Arg::Ptr(out_s.ptr)])
}

/// Gather int8 rows: out[i] = w[idx[i]]
pub fn gather_rows_i8(dev: &Device, w: &Tensor, idx: &Tensor, out: &Tensor) -> Result<()> {
    let k = w.shape[1];
    let rows = idx.numel();
    dev.launch_n("k_gather_rows_i8", rows * (k / 16), &[Arg::Ptr(w.ptr), Arg::Ptr(idx.ptr), Arg::Ptr(out.ptr), Arg::I32(rows as i32), Arg::I32(k as i32)])
}

/// int8 GEMM: out[M,N] = (aq[M,K] . wq[N,K]^T) * sa[m] * sw[n] (+bias) with epilogue.
pub fn gemm_i8(dev: &Device, aq: &Tensor, sa: &Tensor, wq: &Tensor, sw: &Tensor, bias: Option<&Tensor>, epi: Epi, out: &Tensor) -> Result<()> {
    let m = aq.shape[0];
    let k = aq.shape[1];
    let n = wq.shape[0];
    ensure!(wq.shape[1] == k, "gemm_i8: K mismatch {} vs {}", wq.shape[1], k);
    ensure!(k % 16 == 0 && n % 8 == 0, "gemm_i8: K%16, N%8 required (K={k}, N={n})");
    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
        Epi::SwiGluPairs => (3, 0u64, 0u64),
    };
    ensure!(out.numel() == if mode == 3 { m * n / 2 } else { m * n }, "gemm_i8: out shape");
    let wide = gemm_i8_wide(m, n) && std::env::var("H3_GEMM_NARROW").is_err();
    let kname = match (out.dtype, wide) {
        (DType::BF16, false) => "k_gemm_i8_bf16",
        (DType::F32, false) => "k_gemm_i8_f32",
        (DType::BF16, true) => "k_gemm_i8_bf16_w",
        (DType::F32, true) => "k_gemm_i8_f32_w",
        _ => anyhow::bail!("gemm_i8: bad out dtype"),
    };
    let bn = if wide { 256 } else { 128 };
    let grid = ((((n + bn - 1) / bn) * ((m + 127) / 128)) as u32, 1, 1);
    dev.launch(
        kname,
        grid,
        (256, 1, 1),
        if wide { GEMM_SMEM_W } else { GEMM_SMEM },
        &[
            Arg::Ptr(aq.ptr),
            Arg::Ptr(wq.ptr),
            Arg::Ptr(out.ptr),
            Arg::I32(m as i32),
            Arg::I32(n as i32),
            Arg::I32(k as i32),
            Arg::Ptr(sa.ptr),
            Arg::Ptr(sw.ptr),
            Arg::Ptr(bias.map(|b| b.ptr).unwrap_or(0)),
            Arg::I32(mode),
            Arg::Ptr(res),
            Arg::Ptr(gate),
        ],
    )
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

/// bf16 GEMM: out[M,N] = act(a[M,K] . w[N,K]^T + bias) with epilogue. bias is f32 [N].
pub fn gemm_bf16(dev: &Device, a: &Tensor, w: &Tensor, bias: Option<&Tensor>, act: Act, epi: Epi, out: &Tensor) -> Result<()> {
    let m = a.shape[0];
    let k: usize = a.shape[1..].iter().product();
    let n = w.shape[0];
    ensure!(a.dtype == DType::BF16 && w.dtype == DType::BF16, "gemm_bf16 dtypes");
    ensure!(w.numel() / n == k, "gemm_bf16: K mismatch {} vs {}", w.numel() / n, k);
    ensure!(k % 8 == 0 && n % 8 == 0, "gemm_bf16: K%8, N%8 required (K={k}, N={n})");
    ensure!(out.numel() == m * n, "gemm_bf16: out shape {:?} vs {m}x{n}", out.shape);
    let (mode, res, gate) = match epi {
        Epi::Store => (0, 0u64, 0u64),
        Epi::AddRes(r) => (1, r.ptr, 0u64),
        Epi::AddResGated(r, g) => (2, r.ptr, g.ptr),
        Epi::SwiGluPairs => anyhow::bail!("gemm_bf16: SwiGluPairs unsupported"),
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
            Arg::Ptr(gate),
        ],
    )
}

pub fn to_bf16(dev: &Device, x: &Tensor, out: &Tensor) -> Result<()> {
    ensure!(x.dtype == DType::F32 && out.dtype == DType::BF16 && x.numel() == out.numel());
    dev.launch_n("k_f32_to_bf16", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(x.numel() as i64)])
}
pub fn to_f32(dev: &Device, x: &Tensor, out: &Tensor) -> Result<()> {
    ensure!(x.dtype == DType::BF16 && out.dtype == DType::F32 && x.numel() == out.numel());
    dev.launch_n("k_bf16_to_f32", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(x.numel() as i64)])
}
pub fn fill(dev: &Device, x: &Tensor, v: f32) -> Result<()> {
    let k = match x.dtype {
        DType::F32 => "k_fill_f32",
        DType::BF16 => "k_fill_bf16",
        _ => anyhow::bail!("fill dtype"),
    };
    dev.launch_n(k, x.numel(), &[Arg::Ptr(x.ptr), Arg::F32(v), Arg::I64(x.numel() as i64)])
}
pub fn act_inplace(dev: &Device, x: &Tensor, act: Act) -> Result<()> {
    let k = match x.dtype {
        DType::F32 => "k_act_f32",
        DType::BF16 => "k_act_bf16",
        _ => anyhow::bail!("act dtype"),
    };
    dev.launch_n(k, x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(x.ptr), Arg::I64(x.numel() as i64), Arg::I32(act.code())])
}
pub fn add_inplace(dev: &Device, x: &Tensor, y: &Tensor) -> Result<()> {
    ensure!(x.numel() == y.numel() && x.dtype == y.dtype);
    let k = match x.dtype {
        DType::F32 => "k_add_f32",
        DType::BF16 => "k_add_bf16",
        _ => anyhow::bail!("add dtype"),
    };
    dev.launch_n(k, x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(y.ptr), Arg::I64(x.numel() as i64)])
}
/// x[m,n] += y[m,n] * g[n]  (bf16 x,y; f32 gate [n])
pub fn addcmul_inplace(dev: &Device, x: &Tensor, y: &Tensor, g: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    dev.launch_n("k_addcmul_bf16", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(y.ptr), Arg::Ptr(g.ptr), Arg::I64(m as i64), Arg::I32(n as i32)])
}
pub fn add_bias_f32(dev: &Device, x: &Tensor, b: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    dev.launch_n("k_add_bias_f32", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(b.ptr), Arg::I32(m as i32), Arg::I32(n as i32)])
}

/// LayerNorm with optional affine (w, b bf16). x: [M, N] f32/bf16 -> out f32/bf16.
pub fn layernorm(dev: &Device, x: &Tensor, w: Option<&Tensor>, b: Option<&Tensor>, eps: f32, out: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    let k = match (x.dtype, out.dtype) {
        (DType::BF16, DType::BF16) => "k_layernorm_bf16_bf16",
        (DType::F32, DType::BF16) => "k_layernorm_f32_bf16",
        (DType::F32, DType::F32) => "k_layernorm_f32_f32",
        _ => anyhow::bail!("layernorm dtypes"),
    };
    dev.launch(
        k,
        (m as u32, 1, 1),
        (256, 1, 1),
        0,
        &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::F32(eps), Arg::Ptr(w.map(|t| t.ptr).unwrap_or(0)), Arg::Ptr(b.map(|t| t.ptr).unwrap_or(0))],
    )
}
/// AdaLN: LN(x)*(1+scale). scale0 (f32 [N]) applies to rows < prefix_len, scale1 to the rest.
pub fn adaln(dev: &Device, x: &Tensor, scale0: &Tensor, scale1: &Tensor, prefix_len: usize, eps: f32, out: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    ensure!(x.dtype == DType::BF16 && out.dtype == DType::BF16);
    dev.launch(
        "k_adaln_bf16",
        (m as u32, 1, 1),
        (256, 1, 1),
        0,
        &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::F32(eps), Arg::Ptr(scale0.ptr), Arg::Ptr(scale1.ptr), Arg::I32(prefix_len as i32)],
    )
}
pub fn rmsnorm(dev: &Device, x: &Tensor, w: Option<&Tensor>, add_one: bool, eps: f32, out: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let m = x.numel() / n;
    let k = match (x.dtype, out.dtype) {
        (DType::F32, DType::F32) => "k_rmsnorm_f32_f32",
        (DType::F32, DType::BF16) => "k_rmsnorm_f32_bf16",
        (DType::BF16, DType::BF16) => "k_rmsnorm_bf16_bf16",
        _ => anyhow::bail!("rmsnorm dtypes"),
    };
    dev.launch(
        k,
        (m as u32, 1, 1),
        (256, 1, 1),
        0,
        &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::F32(eps), Arg::Ptr(w.map(|t| t.ptr).unwrap_or(0)), Arg::I32(add_one as i32)],
    )
}
/// DiT fused per-head RMSNorm + interleaved rope, in place, on q and k with token strides (elements).
pub fn rms_rope_dit(dev: &Device, q: u64, q_ts: usize, k: u64, k_ts: usize, m: usize, h: usize, wq: &Tensor, wk: &Tensor, eps: f32, rope: &Tensor) -> Result<()> {
    let warps = m * h * 2;
    dev.launch_n("k_rms_rope_dit", warps * 32, &[Arg::Ptr(q), Arg::I64(q_ts as i64), Arg::Ptr(k), Arg::I64(k_ts as i64), Arg::I32(m as i32), Arg::I32(h as i32), Arg::Ptr(wq.ptr), Arg::Ptr(wk.ptr), Arg::F32(eps), Arg::Ptr(rope.ptr)])
}
/// Qwen3 fused q/k RMSNorm + split-half rope, in place. q: [M, Hq, 128] (token stride q_ts elements),
/// k: [M, Hk, 128] (token stride k_ts); rope [M, 64, 2] f32.
pub fn qk_norm_rope_llm(dev: &Device, q: u64, q_ts: usize, k: u64, k_ts: usize, m: usize, hq: usize, hk: usize, wq: &Tensor, wk: &Tensor, eps: f32, rope: &Tensor) -> Result<()> {
    let warps = m * (hq + hk);
    dev.launch_n(
        "k_qk_norm_rope_llm",
        warps * 32,
        &[Arg::Ptr(q), Arg::I64(q_ts as i64), Arg::Ptr(k), Arg::I64(k_ts as i64), Arg::I32(m as i32), Arg::I32(hq as i32), Arg::I32(hk as i32), Arg::Ptr(wq.ptr), Arg::Ptr(wk.ptr), Arg::F32(eps), Arg::Ptr(rope.ptr)],
    )
}
/// x[tok,:] += sum_j w[tok][j] * table[idx[tok][j], :]
pub fn vision_pos_embed(dev: &Device, x: &Tensor, table: &Tensor, idx: &Tensor, w: &Tensor) -> Result<()> {
    let m = x.shape[0];
    let n = x.shape[1];
    dev.launch_n("k_vision_pos_embed", m * n, &[Arg::Ptr(x.ptr), Arg::Ptr(table.ptr), Arg::Ptr(idx.ptr), Arg::Ptr(w.ptr), Arg::I32(m as i32), Arg::I32(n as i32)])
}
/// Vision rope in place on packed qkv [M, 3, H, D]; rope [M, D/2, 2] f32.
pub fn rope_vision(dev: &Device, qkv: &Tensor, h: usize, d: usize, rope: &Tensor) -> Result<()> {
    let m = qkv.shape[0];
    let n = m * 2 * h * (d / 2);
    dev.launch_n("k_rope_vision", n, &[Arg::Ptr(qkv.ptr), Arg::I32(m as i32), Arg::I32(h as i32), Arg::I32(d as i32), Arg::Ptr(rope.ptr)])
}
/// Embedding lookup from an int8 convrot table -> f32 [L, K].
pub fn embed_int8(dev: &Device, table: &Tensor, scale: &Tensor, tokens: &Tensor, out: &Tensor) -> Result<()> {
    let l = tokens.numel();
    let k = table.shape[1];
    ensure!(out.numel() == l * k && out.dtype == DType::F32);
    dev.launch("k_embed_int8_convrot", (l as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(table.ptr), Arg::Ptr(scale.ptr), Arg::Ptr(tokens.ptr), Arg::Ptr(out.ptr), Arg::I32(k as i32)])
}
/// x[pos0 + i, :] += ds[i, :]   (f32)
pub fn add_rows_f32(dev: &Device, x: &Tensor, pos0: usize, ds: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let rows = ds.numel() / n;
    dev.launch_n("k_add_rows_f32", rows * n, &[Arg::Ptr(x.ptr), Arg::I32(n as i32), Arg::I32(pos0 as i32), Arg::Ptr(ds.ptr), Arg::I32(rows as i32)])
}
pub fn gather_rows_f32(dev: &Device, x: &Tensor, idx: &Tensor, out: &Tensor) -> Result<()> {
    let n = *x.shape.last().unwrap();
    let rows = idx.numel();
    ensure!(out.numel() == rows * n);
    dev.launch_n("k_gather_rows_f32", rows * n, &[Arg::Ptr(x.ptr), Arg::Ptr(idx.ptr), Arg::Ptr(out.ptr), Arg::I32(rows as i32), Arg::I32(n as i32)])
}
pub fn swiglu(dev: &Device, x: &Tensor, out: &Tensor) -> Result<()> {
    let h2 = *x.shape.last().unwrap();
    let m = x.numel() / h2;
    ensure!(out.numel() == m * h2 / 2);
    dev.launch_n("k_swiglu_bf16", m * h2 / 2, &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(m as i64), Arg::I32((h2 / 2) as i32)])
}
/// Vision patchify: img f32 [Hp, Wp, 3] normalized -> bf16 [Ntok, 1536]
pub fn vision_patchify(dev: &Device, img: &Tensor, hp: usize, wp: usize, out: &Tensor) -> Result<()> {
    let ntok = (hp / 16) * (wp / 16);
    ensure!(out.numel() == ntok * 1536);
    dev.launch_n("k_vision_patchify", ntok * 1536, &[Arg::Ptr(img.ptr), Arg::Ptr(out.ptr), Arg::I32(hp as i32), Arg::I32(wp as i32)])
}

/// One side of an attention call: a token-major bf16 tensor with explicit strides (in elements).
#[derive(Clone, Copy, Debug)]
pub struct AttnView {
    pub ptr: u64,
    pub tok_stride: usize,
    pub head_stride: usize,
    pub len: usize,
}
impl AttnView {
    /// Contiguous [len, H, D] tensor.
    pub fn contiguous(t: &Tensor, heads: usize, d: usize) -> AttnView {
        assert_eq!(t.numel(), t.shape[0] * heads * d, "AttnView::contiguous shape {:?}", t.shape);
        AttnView { ptr: t.ptr, tok_stride: heads * d, head_stride: d, len: t.shape[0] }
    }
    pub fn empty() -> AttnView {
        AttnView { ptr: 0, tok_stride: 0, head_stride: 0, len: 0 }
    }
}

pub struct AttnArgs<'a> {
    pub q: AttnView,
    pub k1: AttnView,
    pub v1: AttnView,
    pub k2: AttnView,
    pub v2: AttnView,
    pub out: AttnView,
    pub hq: usize,
    pub hk: usize,
    pub d: usize,
    /// per-query key limit (i32 [nq]); None -> causal flag or full
    pub kv_limit: Option<&'a Tensor>,
    pub causal: bool,
    pub causal_off: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParamsRaw {
    q: u64, q_ts: i64, q_hs: i32, _p0: i32,
    k1: u64, v1: u64, kv1_ts: i64, kv1_hs: i32, len1: i32,
    k2: u64, v2: u64, kv2_ts: i64, kv2_hs: i32, len2: i32,
    o: u64, o_ts: i64, o_hs: i32,
    nq: i32, hq: i32, hk: i32, d_real: i32, _p1: i32,
    kv_limit: u64, causal: i32, causal_off: i32,
    scale_log2: f32, _p2: i32,
}
unsafe impl cudarc::driver::DeviceRepr for AttnParamsRaw {}

pub fn attn_init(dev: &Device) -> Result<()> {
    dev.set_max_smem("k_flash_attn_d128", 128 * 256 + 4 * 64 * 256)?;
    dev.set_max_smem("k_flash_attn_d80", 128 * 176 + 4 * 64 * 176)?;
    Ok(())
}

pub fn flash_attn(dev: &Device, a: &AttnArgs) -> Result<()> {
    ensure!(a.d == 128 || a.d == 72, "flash_attn: head dim {} unsupported", a.d);
    ensure!(a.hq % a.hk == 0);
    let (kname, smem) = if a.d == 128 { ("k_flash_attn_d128", 128 * 256 + 4 * 64 * 256) } else { ("k_flash_attn_d80", 128 * 176 + 4 * 64 * 176) };
    let nq = a.q.len;
    let p = AttnParamsRaw {
        q: a.q.ptr, q_ts: a.q.tok_stride as i64, q_hs: a.q.head_stride as i32, _p0: 0,
        k1: a.k1.ptr, v1: a.v1.ptr, kv1_ts: a.k1.tok_stride as i64, kv1_hs: a.k1.head_stride as i32, len1: a.k1.len as i32,
        k2: a.k2.ptr, v2: a.v2.ptr, kv2_ts: a.k2.tok_stride as i64, kv2_hs: a.k2.head_stride as i32, len2: a.k2.len as i32,
        o: a.out.ptr, o_ts: a.out.tok_stride as i64, o_hs: a.out.head_stride as i32,
        nq: nq as i32, hq: a.hq as i32, hk: a.hk as i32, d_real: a.d as i32, _p1: 0,
        kv_limit: a.kv_limit.map(|t| t.ptr).unwrap_or(0), causal: a.causal as i32, causal_off: a.causal_off as i32,
        scale_log2: (1.0 / (a.d as f32).sqrt()) * std::f32::consts::LOG2_E, _p2: 0,
    };
    let f = dev.func(kname)?;
    let cfg = cudarc::driver::LaunchConfig { grid_dim: (((nq + 127) / 128) as u32, a.hq as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem as u32 };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&p);
    unsafe { b.launch(cfg) }?;
    Ok(())
}

// =====================================================================================
// Sage-style low-precision attention (DiT, head dim 128)
// =====================================================================================

/// Quantized keys/values of one segment, as produced by `quantize_kv`.
pub struct QuantKv {
    pub k8: Tensor,    // int8 [n, H, 128]
    pub sk: Tensor,    // f32 [n, H]
    pub vt: Tensor,    // u8 (e4m3) [H, 128, n_pad]
    pub sv: Tensor,    // f32 [n_pad/64, H]
    pub mean_k: Tensor, // f32 [H*128]
    pub mean_v: Tensor, // f32 [H*128]
    pub n: usize,
    pub n_pad: usize,
}

pub struct QuantQ {
    pub q8: Tensor,   // int8 [n, H, 128]
    pub sq: Tensor,   // f32 [n, H]
    pub corr1: Tensor, // f32 [n, H]
    pub corr2: Option<Tensor>,
}

/// Key tile / V quantization group size.
pub const SAGE_BKV: usize = 128;
pub const SAGE_SMEM: u32 = (128 * 128 + 2 * (SAGE_BKV * 128 + 128 * (SAGE_BKV + 16) + SAGE_BKV * 4)) as u32;
const SAGE_KERNEL: &str = if SAGE_BKV == 128 { "k_sage_attn_128" } else { "k_sage_attn_64" };

pub fn sage_init(dev: &Device) -> Result<()> {
    dev.set_max_smem(SAGE_KERNEL, SAGE_SMEM)
}

/// Column means of a [n, H*128] bf16 matrix with token stride `ts` -> f32 [H*128]
pub fn col_mean(dev: &Device, x: u64, ts: usize, n: usize, cols: usize, out: &Tensor) -> Result<()> {
    dev.memset_at(out.ptr, cols * 4)?;
    let blocks = ((n + 63) / 64) as u32;
    dev.launch("k_colsum_bf16", (blocks, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(x), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(cols as i32), Arg::Ptr(out.ptr)])?;
    let s = 1.0 / n as f32;
    dev.launch_n("k_scale_f32", cols, &[Arg::Ptr(out.ptr), Arg::F32(s), Arg::I32(cols as i32)])
}

/// Quantize K and V (bf16, token stride ts, n tokens, H heads) of one segment.
pub fn quantize_kv(dev: &Device, k: u64, v: u64, ts: usize, n: usize, h: usize) -> Result<QuantKv> {
    let cols = h * 128;
    let n_pad = (n + SAGE_BKV - 1) / SAGE_BKV * SAGE_BKV;
    let mean_k = Tensor::new(dev, DType::F32, &[cols])?;
    let mean_v = Tensor::new(dev, DType::F32, &[cols])?;
    col_mean(dev, k, ts, n, cols, &mean_k)?;
    col_mean(dev, v, ts, n, cols, &mean_v)?;
    let k8 = Tensor::new(dev, DType::I8, &[n, h, 128])?;
    let sk = Tensor::new(dev, DType::F32, &[n, h])?;
    dev.launch_n("k_quant_qk_int8", n * h * 32, &[Arg::Ptr(k), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(mean_k.ptr), Arg::Ptr(k8.ptr), Arg::Ptr(sk.ptr)])?;
    let vt = Tensor::new(dev, DType::U8, &[h, 128, n_pad])?;
    let sv = Tensor::new(dev, DType::F32, &[n_pad / SAGE_BKV, h])?;
    dev.launch("k_quant_v_fp8", ((n_pad / SAGE_BKV) as u32, h as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(v), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(mean_v.ptr), Arg::Ptr(vt.ptr), Arg::Ptr(sv.ptr), Arg::I32(SAGE_BKV as i32)])?;
    Ok(QuantKv { k8, sk, vt, sv, mean_k, mean_v, n, n_pad })
}

/// Quantize Q (bf16, token stride ts) and compute the per-segment corrections Q . mean_k.
pub fn quantize_q(dev: &Device, q: u64, ts: usize, n: usize, h: usize, seg1: &QuantKv, seg2: Option<&QuantKv>) -> Result<QuantQ> {
    let q8 = Tensor::new(dev, DType::I8, &[n, h, 128])?;
    let sq = Tensor::new(dev, DType::F32, &[n, h])?;
    dev.launch_n("k_quant_qk_int8", n * h * 32, &[Arg::Ptr(q), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(0), Arg::Ptr(q8.ptr), Arg::Ptr(sq.ptr)])?;
    let corr1 = Tensor::new(dev, DType::F32, &[n, h])?;
    dev.launch_n("k_q_corr", n * h * 32, &[Arg::Ptr(q), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(seg1.mean_k.ptr), Arg::Ptr(corr1.ptr)])?;
    let corr2 = match seg2 {
        Some(s2) => {
            let c = Tensor::new(dev, DType::F32, &[n, h])?;
            dev.launch_n("k_q_corr", n * h * 32, &[Arg::Ptr(q), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(h as i32), Arg::Ptr(s2.mean_k.ptr), Arg::Ptr(c.ptr)])?;
            Some(c)
        }
        None => None,
    };
    Ok(QuantQ { q8, sq, corr1, corr2 })
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SageParamsRaw {
    q: u64, sq: u64, corr1: u64, corr2: u64,
    k1: u64, sk1: u64, v1: u64, sv1: u64, mv1: u64, len1: i32, len1_pad: i32,
    k2: u64, sk2: u64, v2: u64, sv2: u64, mv2: u64, len2: i32, _p0: i32,
    o: u64, o_ts: i64, o_hs: i32,
    nq: i32, h: i32, _p1: i32,
    kv_limit: u64,
    scale_log2: f32, _p2: i32,
}
unsafe impl cudarc::driver::DeviceRepr for SageParamsRaw {}

pub fn sage_attn(dev: &Device, q: &QuantQ, nq: usize, h: usize, seg1: &QuantKv, seg2: Option<&QuantKv>, out: AttnView, kv_limit: Option<&Tensor>) -> Result<()> {
    let p = SageParamsRaw {
        q: q.q8.ptr, sq: q.sq.ptr, corr1: q.corr1.ptr, corr2: q.corr2.as_ref().map(|t| t.ptr).unwrap_or(0),
        k1: seg1.k8.ptr, sk1: seg1.sk.ptr, v1: seg1.vt.ptr, sv1: seg1.sv.ptr, mv1: seg1.mean_v.ptr, len1: seg1.n as i32, len1_pad: seg1.n_pad as i32,
        k2: seg2.map(|s| s.k8.ptr).unwrap_or(0), sk2: seg2.map(|s| s.sk.ptr).unwrap_or(0), v2: seg2.map(|s| s.vt.ptr).unwrap_or(0), sv2: seg2.map(|s| s.sv.ptr).unwrap_or(0), mv2: seg2.map(|s| s.mean_v.ptr).unwrap_or(0), len2: seg2.map(|s| s.n as i32).unwrap_or(0), _p0: 0,
        o: out.ptr, o_ts: out.tok_stride as i64, o_hs: out.head_stride as i32,
        nq: nq as i32, h: h as i32, _p1: 0,
        kv_limit: kv_limit.map(|t| t.ptr).unwrap_or(0),
        scale_log2: (1.0 / (128f32).sqrt()) * std::f32::consts::LOG2_E, _p2: 0,
    };
    let f = dev.func(SAGE_KERNEL)?;
    let cfg = cudarc::driver::LaunchConfig { grid_dim: (((nq + 127) / 128) as u32, h as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: SAGE_SMEM };
    let mut b = dev.stream.launch_builder(&f);
    b.arg(&p);
    unsafe { b.launch(cfg) }?;
    Ok(())
}
