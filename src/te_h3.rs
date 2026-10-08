//! MiniMax H3 text/vision conditioning encoder: Qwen3-VL-32B truncated to 50 layers (ComfyUI
//! `comfy/text_encoders/minimax.py`, `qwen3vl.py`, `qwen35.py` vision tower, `llama.py` Qwen3VL_32BConfig).
//!
//! * Presentation (no chat template): per reference item in request order
//!   image -> "<Picture i>: " <|vision_start|> [vision] <|vision_end|>; audio -> "<Audio j>: ";
//!   video -> "<Video k>: " then per 2-frame block "<T.T seconds>" + vision block; then the prompt.
//!   Every text piece is BPE-tokenized on its own.
//! * Decoder linears are NVFP4 and stay packed on the GPU (≈13 GB); they are expanded on the fly inside
//!   a fused weight-only GEMM (`kernels/nvfp4.cu`) — no bf16 copy of any weight is ever materialized.
//!   AWQ `pre_quant_scale` of o_proj / down_proj is applied to the activations (fused into SwiGLU for down).
//! * f32 residual stream, bf16 GEMM operands, fused q/k RMSNorm + interleaved MRoPE, flash attention (GQA 64/8).
//! * The vision tower (27 bf16 blocks, DeepStack after vision blocks 8/16/24, injected after decoder layers 0..2)
//!   runs per vision block; the patch embedding consumes the f32 pixels exactly (hi/lo bf16 split).
//! * Output: hidden state after layer 50 (no final norm) as bf16 [L, 5120] + per-token tags
//!   (0 = vision block incl. its start/end tokens, 1 = text).
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, AttnArgs, AttnView, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::tokenizer::{self, Tokenizer};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const HIDDEN: usize = 5120;
pub const LAYERS: usize = 50;
pub const HEADS: usize = 64;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const INTER: usize = 25600;
pub const EPS: f32 = 1e-6;
pub const ROPE_THETA: f32 = 5_000_000.0;
pub const ROPE_DIMS: [usize; 3] = [24, 20, 20];
pub const QKV_N: usize = HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM; // 10240
const Q_DIM: usize = HEADS * HEAD_DIM; // 8192

pub const VISION_START: u32 = tokenizer::VISION_START;
pub const VISION_END: u32 = tokenizer::VISION_END;
pub const PAD: u32 = tokenizer::ENDOFTEXT;

// ------------------------------------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------------------------------------

/// One reference item, in request order.
pub enum RefItem {
    /// Reference image (already resized to the canvas by the caller), RGB8 HWC.
    Image { w: usize, h: usize, rgb: Vec<u8> },
    /// A reference audio track: only its "<Audio j>: " label enters the encoder.
    Audio,
    /// Reference video frames sampled at 2 fps (w, h, rgb), with their timestamps in seconds.
    Video { frames: Vec<(usize, usize, Vec<u8>)>, timestamps: Vec<f32> },
}

pub struct Conditioning {
    /// [L, 5120] bf16: raw hidden state after decoder layer 50.
    pub context: Tensor,
    /// [L] 0 = vision block (incl. <|vision_start|>/<|vision_end|>), 1 = text.
    pub tags: Vec<u8>,
    /// [L] token id per position, -1 for vision embeddings.
    pub token_ids: Vec<i32>,
}

/// Per-stage timings of the last `encode` call (seconds, device-synchronized).
#[derive(Default, Clone, Debug)]
pub struct EncodeTiming {
    pub vision: f64,
    pub decoder: f64,
    pub total: f64,
}

/// Per-category GPU timing (enabled with LOKI_PROFILE=1; synchronizes around every timed op).
pub static PROF: std::sync::LazyLock<crate::cuda::Profiler> = std::sync::LazyLock::new(crate::cuda::Profiler::new);

// ------------------------------------------------------------------------------------------------
// NVFP4 linear
// ------------------------------------------------------------------------------------------------

/// NVFP4 weight-only linear (possibly several checkpoint linears concatenated along N).
pub struct Fp4Linear {
    /// u8 [N, K/2] packed e2m1, even element in the high nibble
    pub w: Tensor,
    /// u8 (e4m3) block scales in the cuBLAS tiled layout, [N, K/16]
    pub s: Tensor,
    /// f32 [N] per-row tensor scale (weight_scale_2 of the source linear)
    pub ts: Tensor,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fp4Out {
    /// out bf16 = v
    StoreBf16,
    /// out f32 += v (split-K with atomics when the grid is small)
    AddF32,
}

pub fn fp4_init(dev: &Device) -> Result<()> {
    for (k, bm) in [("k_fp4_gemm_128_bf16", 128), ("k_fp4_gemm_128_addf32", 128), ("k_fp4_gemm_128_atomf32", 128), ("k_fp4_gemm_64_bf16", 64), ("k_fp4_gemm_64_addf32", 64), ("k_fp4_gemm_64_atomf32", 64)] {
        dev.set_max_smem(k, fp4_smem(bm))?;
    }
    dev.set_max_smem("k_attn_split_d72", 2 * 4 * 64 * 176)?;
    Ok(())
}
fn fp4_smem(bm: usize) -> u32 {
    (4 * (bm * 64 * 2 + 4096 + 512)) as u32
}

/// out[M, ldc](cols 0..N) (+)= a[M, K] . W^T, a bf16 row-major (row stride `lda` elements).
pub fn fp4_gemm(dev: &Device, a: u64, lda: usize, m: usize, lin: &Fp4Linear, out: u64, ldc: usize, mode: Fp4Out) -> Result<()> {
    let (n, k) = (lin.n, lin.k);
    ensure!(n % 128 == 0 && k % 64 == 0, "fp4_gemm: N%128 / K%64 required (N={n}, K={k})");
    if m == 0 {
        return Ok(());
    }
    let sms = dev.sm_count.max(1) as usize;
    let nt = n / 128;
    let tiles128 = (m + 127) / 128 * nt;
    let bm = if std::env::var("LOKI_FP4_BM").map(|v| v == "64").unwrap_or(false) || (m <= 64) || tiles128 < 2 * sms { 64 } else { 128 };
    let mt = (m + bm - 1) / bm;
    let blocks = mt * nt;
    let mut splits = 1usize;
    if mode == Fp4Out::AddF32 && blocks < sms {
        // split K so that the grid covers the GPU; partial sums are added atomically into the residual
        let kt = k / 64;
        splits = ((2 * sms + blocks - 1) / blocks).min(kt / 4).max(1);
    }
    let ktiles = k / 64;
    let k_split = (ktiles + splits - 1) / splits * 64;
    let splits = (k + k_split - 1) / k_split;
    let kname = match (bm, mode, splits > 1) {
        (128, Fp4Out::StoreBf16, _) => "k_fp4_gemm_128_bf16",
        (64, Fp4Out::StoreBf16, _) => "k_fp4_gemm_64_bf16",
        (128, Fp4Out::AddF32, false) => "k_fp4_gemm_128_addf32",
        (64, Fp4Out::AddF32, false) => "k_fp4_gemm_64_addf32",
        (128, Fp4Out::AddF32, true) => "k_fp4_gemm_128_atomf32",
        (_, Fp4Out::AddF32, true) => "k_fp4_gemm_64_atomf32",
        _ => unreachable!(),
    };
    dev.launch(
        kname,
        (blocks as u32, 1, splits as u32),
        (256, 1, 1),
        fp4_smem(bm),
        &[
            Arg::Ptr(a),
            Arg::I32(lda as i32),
            Arg::Ptr(lin.w.ptr),
            Arg::Ptr(lin.s.ptr),
            Arg::Ptr(lin.ts.ptr),
            Arg::Ptr(out),
            Arg::I32(ldc as i32),
            Arg::I32(m as i32),
            Arg::I32(n as i32),
            Arg::I32(k as i32),
            Arg::I32(k_split as i32),
        ],
    )
}

/// Dequantize a whole NVFP4 linear to bf16 [N, K] (tests / debugging).
pub fn fp4_dequant(dev: &Device, lin: &Fp4Linear) -> Result<Tensor> {
    let out = Tensor::new(dev, DType::BF16, &[lin.n, lin.k])?;
    dev.launch_n("k_nvfp4_dequant", lin.n * lin.k / 16, &[Arg::Ptr(lin.w.ptr), Arg::Ptr(lin.s.ptr), Arg::Ptr(lin.ts.ptr), Arg::Ptr(out.ptr), Arg::I32(lin.n as i32), Arg::I32(lin.k as i32)])?;
    Ok(out)
}

/// Load NVFP4 linears `prefixes` (same K) concatenated along N.
pub fn load_fp4_cat(st: &SafeTensors, dev: &Device, prefixes: &[&str], uploaded: &mut usize) -> Result<Fp4Linear> {
    let mut n_total = 0;
    let mut k = 0;
    for p in prefixes {
        let meta: serde_json::Value = serde_json::from_slice(st.bytes(&format!("{p}.comfy_quant"))?)?;
        ensure!(meta["format"] == "nvfp4", "{p}: expected nvfp4, got {}", meta["format"]);
        let wi = st.info(&format!("{p}.weight"))?;
        ensure!(wi.dtype == "U8" && wi.shape.len() == 2, "{p}: bad packed weight");
        let kk = wi.shape[1] * 2;
        if k == 0 {
            k = kk;
        }
        ensure!(kk == k, "{p}: K mismatch");
        ensure!(wi.shape[0] % 128 == 0, "{p}: N must be a multiple of 128");
        let si = st.info(&format!("{p}.weight_scale"))?;
        ensure!(si.shape == vec![wi.shape[0], k / 16], "{p}: block scale shape {:?}", si.shape);
        n_total += wi.shape[0];
    }
    let w = Tensor::new(dev, DType::U8, &[n_total, k / 2])?;
    let s = Tensor::new(dev, DType::U8, &[n_total, k / 16])?;
    let mut ts = Vec::with_capacity(n_total);
    let mut row = 0;
    for p in prefixes {
        let wb = st.bytes(&format!("{p}.weight"))?;
        let sb = st.bytes(&format!("{p}.weight_scale"))?;
        let rows = wb.len() / (k / 2);
        dev.htod_at(w.ptr + (row * k / 2) as u64, wb)?;
        dev.htod_at(s.ptr + (row * k / 16) as u64, sb)?;
        let t = st.f32s(&format!("{p}.weight_scale_2"))?;
        ensure!(t.len() == 1);
        ts.extend(std::iter::repeat(t[0]).take(rows));
        row += rows;
        *uploaded += wb.len() + sb.len();
    }
    let ts = Tensor::from_f32(dev, &ts, &[n_total])?;
    Ok(Fp4Linear { w, s, ts, n: n_total, k })
}

// ------------------------------------------------------------------------------------------------
// Tokenization
// ------------------------------------------------------------------------------------------------

/// Added tokens of the Qwen2.5 tokenizer not handled by `Tokenizer`, plus the MiniMax H3 extras.
const EXTRA_TOKENS: &[(&str, u32)] = &[
    ("<tool_call>", 151657),
    ("</tool_call>", 151658),
    ("<|fim_prefix|>", 151659),
    ("<|fim_middle|>", 151660),
    ("<|fim_suffix|>", 151661),
    ("<|fim_pad|>", 151662),
    ("<|repo_name|>", 151663),
    ("<|file_sep|>", 151664),
    ("<tool_response>", 151665),
    ("</tool_response>", 151666),
    ("<think>", 151667),
    ("</think>", 151668),
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
];

/// Tokenize one text piece the way ComfyUI's SDTokenizer does with weights disabled
/// (`\(` / `\)` unescaped, special and added tokens recognized literally, plain BPE elsewhere).
pub fn tokenize_piece(tok: &Tokenizer, text: &str) -> Vec<u32> {
    let text = text.replace("\\)", ")").replace("\\(", "(");
    let mut out = Vec::new();
    let mut rest: &str = &text;
    while !rest.is_empty() {
        let mut best: Option<(usize, usize, u32)> = None;
        for (s, id) in EXTRA_TOKENS {
            if let Some(pos) = rest.find(s) {
                let better = match best {
                    None => true,
                    Some((bp, bl, _)) => pos < bp || (pos == bp && s.len() > bl),
                };
                if better {
                    best = Some((pos, s.len(), *id));
                }
            }
        }
        match best {
            Some((pos, len, id)) => {
                if pos > 0 {
                    out.extend(tok.encode(&rest[..pos]));
                }
                out.push(id);
                rest = &rest[pos + len..];
            }
            None => {
                out.extend(tok.encode(rest));
                rest = "";
            }
        }
    }
    out
}

/// Python's "%.1f" for the block timestamps (round half to even on the exact binary value).
pub fn fmt_1f(x: f64) -> String {
    let r = round_half_even(x * 10.0);
    let neg = r < 0.0;
    let a = r.abs() as i64;
    format!("{}{}.{}", if neg { "-" } else { "" }, a / 10, a % 10)
}

// ------------------------------------------------------------------------------------------------
// Vision preprocessing (process_qwen2vl_images / process_video_block), on the host
// ------------------------------------------------------------------------------------------------

const PATCH: usize = 16;
const MERGE: usize = 2;
const FACTOR: usize = PATCH * MERGE;
const MIN_PIXELS: usize = 3136;
const MAX_PIXELS: usize = 12_845_056;

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Target size of the vision input (h_bar, w_bar) for an h x w image.
pub fn vision_size(h: usize, w: usize) -> (usize, usize) {
    let f = FACTOR as f64;
    let mut hb = (round_half_even(h as f64 / f) as usize) * FACTOR;
    let mut wb = (round_half_even(w as f64 / f) as usize) * FACTOR;
    if hb * wb > MAX_PIXELS {
        let beta = ((h * w) as f64 / MAX_PIXELS as f64).sqrt();
        hb = FACTOR.max(((h as f64 / beta / f).floor() as usize) * FACTOR);
        wb = FACTOR.max(((w as f64 / beta / f).floor() as usize) * FACTOR);
    } else if hb * wb < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / (h * w) as f64).sqrt();
        hb = ((h as f64 * beta / f).ceil() as usize) * FACTOR;
        wb = ((w as f64 * beta / f).ceil() as usize) * FACTOR;
    }
    (hb, wb)
}

/// RGB8 -> normalized f32 CHW at (hb, wb): bilinear (align_corners=False, no antialias) then (x-0.5)/0.5.
fn vision_pixels(w: usize, h: usize, rgb: &[u8], hb: usize, wb: usize) -> Vec<f32> {
    let src: Vec<f32> = rgb.iter().map(|&v| v as f32 / 255.0).collect();
    let mut out = vec![0f32; 3 * hb * wb];
    if hb == h && wb == w {
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    out[(c * hb + y) * wb + x] = (src[(y * w + x) * 3 + c] - 0.5) / 0.5;
                }
            }
        }
        return out;
    }
    let sy = h as f32 / hb as f32;
    let sx = w as f32 / wb as f32;
    for oy in 0..hb {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy as usize).min(h - 1);
        let y1 = if y0 < h - 1 { y0 + 1 } else { y0 };
        let ly = fy - y0 as f32;
        for ox in 0..wb {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx as usize).min(w - 1);
            let x1 = if x0 < w - 1 { x0 + 1 } else { x0 };
            let lx = fx - x0 as f32;
            for c in 0..3 {
                let p = |yy: usize, xx: usize| src[(yy * w + xx) * 3 + c];
                let v = (1.0 - ly) * ((1.0 - lx) * p(y0, x0) + lx * p(y0, x1)) + ly * ((1.0 - lx) * p(y1, x0) + lx * p(y1, x1));
                out[(c * hb + oy) * wb + ox] = (v - 0.5) / 0.5;
            }
        }
    }
    out
}

/// Two normalized CHW frames (identical for still images) -> patches f32 [gh*gw, 1536] in merge-window
/// token order (block_row, block_col, intra_row, intra_col) with features (c, t, py, px).
fn patchify(f0: &[f32], f1: &[f32], hb: usize, wb: usize) -> Vec<f32> {
    let mut out = patchify_exact(f0, f1, hb, wb);
    // ComfyUI's fp32 Conv3d patch embedding runs through cuDNN with TF32 allowed: the pixels enter the tensor
    // cores rounded to TF32 (10-bit mantissa, round-to-nearest-away). Reproducing that rounding brings the whole
    // vision tower to ~1e-4 of the reference (vs ~1e-2 without). LOKI_TF32_PATCH=off|trunc for experiments.
    let mode = std::env::var("LOKI_TF32_PATCH").unwrap_or_default();
    if mode != "off" {
        for v in out.iter_mut() {
            let b = v.to_bits();
            *v = f32::from_bits(if mode == "trunc" { b & !0x1FFF } else { b.wrapping_add(0x1000) & !0x1FFF });
        }
    }
    out
}

fn patchify_exact(f0: &[f32], f1: &[f32], hb: usize, wb: usize) -> Vec<f32> {
    let (gh, gw) = (hb / PATCH, wb / PATCH);
    let n = gh * gw;
    let mut out = vec![0f32; n * 1536];
    for tok in 0..n {
        let (r, c) = token_rc(tok, gw);
        let o = &mut out[tok * 1536..(tok + 1) * 1536];
        for ch in 0..3 {
            for (t, fr) in [f0, f1].iter().enumerate() {
                for py in 0..PATCH {
                    let src = &fr[(ch * hb + r * PATCH + py) * wb + c * PATCH..][..PATCH];
                    o[ch * 512 + t * 256 + py * 16..][..16].copy_from_slice(src);
                }
            }
        }
    }
    out
}

/// Patch (row, col) of token `t` in merge-window order for a grid of width `gw` patches.
pub fn token_rc(t: usize, gw: usize) -> (usize, usize) {
    let mw = gw / 2;
    let br = t / (mw * 4);
    let r2 = t % (mw * 4);
    let bc = r2 / 4;
    let r3 = r2 % 4;
    (br * 2 + r3 / 2, bc * 2 + r3 % 2)
}

// ------------------------------------------------------------------------------------------------
// Vision tower
// ------------------------------------------------------------------------------------------------

pub const V_HIDDEN: usize = 1152;
pub const V_HEADS: usize = 16;
pub const V_HEAD_DIM: usize = 72;
pub const V_INTER: usize = 4304;
pub const V_DEPTH: usize = 27;
pub const V_DEEPSTACK: [usize; 3] = [8, 16, 24];
pub const V_MERGE_DIM: usize = V_HIDDEN * 4;
pub const V_POS_GRID: usize = 48;

struct VBlock {
    ln1_w: Tensor,
    ln1_b: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

struct Merger {
    norm_w: Tensor,
    norm_b: Tensor,
    norm_dim: usize,
    fc1_w: Tensor,
    fc1_b: Tensor,
    fc2_w: Tensor,
    fc2_b: Tensor,
}

struct VisionTower {
    patch_w: Tensor,
    patch_b: Tensor,
    /// 48x48 position table (bf16)
    pos_table: Tensor,
    blocks: Vec<VBlock>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

struct VisionOut {
    merged: Tensor,         // f32 [n/4, 5120]
    deepstack: Vec<Tensor>, // 3 x f32 [n/4, 5120]
    grid_h: usize,          // patches
    grid_w: usize,
}

impl VisionTower {
    fn load(l: &mut Loader) -> Result<VisionTower> {
        let p = "visual";
        let patch_w = l.bf16(&format!("{p}.patch_embed.proj.weight"))?.reshape(&[V_HIDDEN, 1536]);
        let patch_b = l.f32(&format!("{p}.patch_embed.proj.bias"))?;
        let pos_table = l.bf16(&format!("{p}.pos_embed.weight"))?;
        ensure!(pos_table.numel() == V_POS_GRID * V_POS_GRID * V_HIDDEN, "pos_embed shape");
        let mut blocks = Vec::with_capacity(V_DEPTH);
        for i in 0..V_DEPTH {
            let b = format!("{p}.blocks.{i}");
            blocks.push(VBlock {
                ln1_w: l.bf16(&format!("{b}.norm1.weight"))?,
                ln1_b: l.bf16(&format!("{b}.norm1.bias"))?,
                qkv_w: l.bf16(&format!("{b}.attn.qkv.weight"))?,
                qkv_b: l.f32(&format!("{b}.attn.qkv.bias"))?,
                proj_w: l.bf16(&format!("{b}.attn.proj.weight"))?,
                proj_b: l.f32(&format!("{b}.attn.proj.bias"))?,
                ln2_w: l.bf16(&format!("{b}.norm2.weight"))?,
                ln2_b: l.bf16(&format!("{b}.norm2.bias"))?,
                fc1_w: l.bf16(&format!("{b}.mlp.linear_fc1.weight"))?,
                fc1_b: l.f32(&format!("{b}.mlp.linear_fc1.bias"))?,
                fc2_w: l.bf16(&format!("{b}.mlp.linear_fc2.weight"))?,
                fc2_b: l.f32(&format!("{b}.mlp.linear_fc2.bias"))?,
            });
        }
        let load_merger = |l: &mut Loader, pre: &str, norm_dim: usize| -> Result<Merger> {
            Ok(Merger {
                norm_w: l.bf16(&format!("{pre}.norm.weight"))?,
                norm_b: l.bf16(&format!("{pre}.norm.bias"))?,
                norm_dim,
                fc1_w: l.bf16(&format!("{pre}.linear_fc1.weight"))?,
                fc1_b: l.f32(&format!("{pre}.linear_fc1.bias"))?,
                fc2_w: l.bf16(&format!("{pre}.linear_fc2.weight"))?,
                fc2_b: l.f32(&format!("{pre}.linear_fc2.bias"))?,
            })
        };
        let merger = load_merger(l, &format!("{p}.merger"), V_HIDDEN)?;
        let mut deepstack = Vec::new();
        for i in 0..3 {
            deepstack.push(load_merger(l, &format!("{p}.deepstack_merger_list.{i}"), V_MERGE_DIM)?);
        }
        Ok(VisionTower { patch_w, patch_b, pos_table, blocks, merger, deepstack })
    }

    /// x += interpolated position embeddings (merge-window order), reproducing ComfyUI's bf16 math
    /// (fast_pos_embed_interpolate: f32 linspace, bf16 weights, bf16 products, sequential bf16 sums).
    fn add_pos_embed(&self, dev: &Device, x: &Tensor, gh: usize, gw: usize) -> Result<()> {
        let g = V_POS_GRID;
        let linspace = |steps: usize| -> Vec<f32> {
            let (start, end) = (0f32, (g - 1) as f32);
            if steps == 1 {
                return vec![start];
            }
            let step = (end - start) / (steps - 1) as f32;
            let half = steps / 2;
            (0..steps).map(|i| if i < half { start + step * i as f32 } else { end - step * (steps - 1 - i) as f32 }).collect()
        };
        let hs = linspace(gh);
        let ws = linspace(gw);
        let bf = |x: f32| half::bf16::from_f32(x).to_f32();
        let n = gh * gw;
        let mut idx = vec![0i32; n * 4];
        let mut wts = vec![0f32; n * 4];
        for tok in 0..n {
            let (r, c) = token_rc(tok, gw);
            let (hi, wi) = (hs[r], ws[c]);
            let (hf, wf) = (hi as usize, wi as usize);
            let hc = (hf + 1).min(g - 1);
            let wc = (wf + 1).min(g - 1);
            let dh = hi - hf as f32;
            let dw = wi - wf as f32;
            idx[tok * 4..tok * 4 + 4].copy_from_slice(&[(hf * g + wf) as i32, (hf * g + wc) as i32, (hc * g + wf) as i32, (hc * g + wc) as i32]);
            wts[tok * 4..tok * 4 + 4].copy_from_slice(&[bf((1.0 - dh) * (1.0 - dw)), bf((1.0 - dh) * dw), bf(dh * (1.0 - dw)), bf(dh * dw)]);
        }
        let idx_t = Tensor::from_buf(dev.upload(&idx)?, DType::F32, &[n * 4]);
        let w_t = Tensor::from_f32(dev, &wts, &[n * 4])?;
        dev.launch_n("k_pos_embed_bf16emu", n * V_HIDDEN, &[Arg::Ptr(x.ptr), Arg::Ptr(self.pos_table.ptr), Arg::Ptr(idx_t.ptr), Arg::Ptr(w_t.ptr), Arg::I32(n as i32), Arg::I32(V_HIDDEN as i32)])
    }

    /// `patches`: f32 [n, 1536] (merge-window order), grid gh x gw patches.
    fn forward(&self, dev: &Device, patches: &[f32], gh: usize, gw: usize) -> Result<VisionOut> {
        let n = gh * gw;
        ensure!(patches.len() == n * 1536);
        // ---- patch embedding on the exact f32 pixels: x = hi.W + lo.W + b
        let x = Tensor::new(dev, DType::F32, &[n, V_HIDDEN])?;
        {
            let pf = Tensor::from_f32(dev, patches, &[n, 1536])?;
            let hi = Tensor::new(dev, DType::BF16, &[n, 1536])?;
            let lo = Tensor::new(dev, DType::BF16, &[n, 1536])?;
            dev.launch_n("k_split_hilo_bf16", n * 1536, &[Arg::Ptr(pf.ptr), Arg::Ptr(hi.ptr), Arg::Ptr(lo.ptr), Arg::I64((n * 1536) as i64)])?;
            ops::gemm_bf16(dev, &hi, &self.patch_w, Some(&self.patch_b), Act::None, Epi::Store, &x)?;
            ops::gemm_bf16(dev, &lo, &self.patch_w, None, Act::None, Epi::AddRes(&x), &x)?;
            self.add_pos_embed(dev, &x, gh, gw)?;
        }
        // ---- 2D rope table [n, 36, 2]: 18 row frequencies then 18 column frequencies
        let rope = {
            let mut tab = vec![0f32; n * 36 * 2];
            let inv: Vec<f32> = (0..18).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / 36.0)).collect();
            for t in 0..n {
                let (r, c) = token_rc(t, gw);
                for i in 0..18 {
                    let ar = r as f32 * inv[i];
                    let ac = c as f32 * inv[i];
                    tab[(t * 36 + i) * 2] = ar.cos();
                    tab[(t * 36 + i) * 2 + 1] = ar.sin();
                    tab[(t * 36 + 18 + i) * 2] = ac.cos();
                    tab[(t * 36 + 18 + i) * 2 + 1] = ac.sin();
                }
            }
            Tensor::from_f32(dev, &tab, &[n, 36, 2])?
        };

        // Comfy runs the tower in fp32 (bf16 weights cast up). Every GEMM here therefore uses a hi/lo bf16 split
        // of its f32 activations (~16 significant bits, f32 accumulate), with f32 attention and f32 intermediates.
        let sc = Scratch { hi: Tensor::new(dev, DType::BF16, &[n * V_MERGE_DIM])?, lo: Tensor::new(dev, DType::BF16, &[n * V_MERGE_DIM])? };
        let ln_out = Tensor::new(dev, DType::F32, &[n, V_HIDDEN])?;
        let qkv = Tensor::new(dev, DType::F32, &[n, 3 * V_HIDDEN])?;
        let attn_out = Tensor::new(dev, DType::F32, &[n, V_HIDDEN])?;
        let fc1 = Tensor::new(dev, DType::F32, &[n, V_INTER])?;
        let mut deep = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            ops::layernorm(dev, &x, Some(&b.ln1_w), Some(&b.ln1_b), 1e-6, &ln_out)?;
            lin_f32(dev, &sc, &ln_out, &b.qkv_w, Some(&b.qkv_b), &qkv, false)?;
            dev.launch_n("k_rope_vision_f32", n * 2 * V_HEADS * (V_HEAD_DIM / 2), &[Arg::Ptr(qkv.ptr), Arg::I32(n as i32), Arg::I32(V_HEADS as i32), Arg::I32(V_HEAD_DIM as i32), Arg::Ptr(rope.ptr)])?;
            let scale_log2 = (1.0 / (V_HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;
            PROF.time(dev, "vis_attn", || {
                if std::env::var("LOKI_VIS_ATTN").map(|v| v == "simt").unwrap_or(false) {
                    return dev.launch("k_attn_f32_d72", (((n + 63) / 64) as u32, V_HEADS as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(qkv.ptr), Arg::Ptr(attn_out.ptr), Arg::I32(n as i32), Arg::I32(V_HEADS as i32), Arg::F32(scale_log2)]);
                }
                dev.launch_n("k_split_hilo_bf16", n * 3 * V_HIDDEN, &[Arg::Ptr(qkv.ptr), Arg::Ptr(sc.hi.ptr), Arg::Ptr(sc.lo.ptr), Arg::I64((n * 3 * V_HIDDEN) as i64)])?;
                dev.launch("k_attn_split_d72", (((n + 127) / 128) as u32, V_HEADS as u32, 1), (256, 1, 1), 2 * 4 * 64 * 176, &[Arg::Ptr(sc.hi.ptr), Arg::Ptr(sc.lo.ptr), Arg::Ptr(attn_out.ptr), Arg::I32(n as i32), Arg::I32(V_HEADS as i32), Arg::F32(scale_log2)])?;
                Ok(())
            })?;
            PROF.time(dev, "vis_lin", || {
                lin_f32(dev, &sc, &attn_out, &b.proj_w, Some(&b.proj_b), &x, true)?;
                ops::layernorm(dev, &x, Some(&b.ln2_w), Some(&b.ln2_b), 1e-6, &ln_out)?;
                lin_f32(dev, &sc, &ln_out, &b.fc1_w, Some(&b.fc1_b), &fc1, false)?;
                ops::act_inplace(dev, &fc1, Act::GeluTanh)?;
                lin_f32(dev, &sc, &fc1, &b.fc2_w, Some(&b.fc2_b), &x, true)
            })?;
            if let Some(di) = V_DEEPSTACK.iter().position(|&d| d == bi) {
                deep.push(self.merge(dev, &sc, &self.deepstack[di], &x, n)?);
            }
        }
        let merged = self.merge(dev, &sc, &self.merger, &x, n)?;
        Ok(VisionOut { merged, deepstack: deep, grid_h: gh, grid_w: gw })
    }

    /// LayerNorm (over norm_dim) -> view [n/4, 4608] -> fc1 -> GELU(erf) -> fc2 -> f32 [n/4, 5120]
    fn merge(&self, dev: &Device, sc: &Scratch, m: &Merger, x: &Tensor, n: usize) -> Result<Tensor> {
        let rows = n / 4;
        let ln = Tensor::new(dev, DType::F32, &[n, V_HIDDEN])?;
        if m.norm_dim == V_HIDDEN {
            ops::layernorm(dev, x, Some(&m.norm_w), Some(&m.norm_b), 1e-6, &ln)?;
        } else {
            ops::layernorm(dev, &x.reshape(&[rows, V_MERGE_DIM]), Some(&m.norm_w), Some(&m.norm_b), 1e-6, &ln.reshape(&[rows, V_MERGE_DIM]))?;
        }
        let lnv = ln.reshape(&[rows, V_MERGE_DIM]);
        let h = Tensor::new(dev, DType::F32, &[rows, V_MERGE_DIM])?;
        lin_f32(dev, sc, &lnv, &m.fc1_w, Some(&m.fc1_b), &h, false)?;
        ops::act_inplace(dev, &h, Act::GeluErf)?;
        let out = Tensor::new(dev, DType::F32, &[rows, HIDDEN])?;
        lin_f32(dev, sc, &h, &m.fc2_w, Some(&m.fc2_b), &out, false)?;
        Ok(out)
    }
}

/// hi/lo bf16 staging buffers for `lin_f32`.
struct Scratch {
    hi: Tensor,
    lo: Tensor,
}

/// out (+)= a . w^T + b with f32 activations `a` [M, K] split into bf16 hi + lo (two tensor-core GEMMs,
/// f32 accumulate): ~16-bit activation precision against exact bf16 weights.
fn lin_f32(dev: &Device, sc: &Scratch, a: &Tensor, w: &Tensor, b: Option<&Tensor>, out: &Tensor, add: bool) -> Result<()> {
    let m = a.shape[0];
    let k: usize = a.shape[1..].iter().product();
    ensure!(a.dtype == DType::F32 && out.dtype == DType::F32 && m * k <= sc.hi.numel());
    let hi = Tensor { buf: sc.hi.buf.clone(), ptr: sc.hi.ptr, dtype: DType::BF16, shape: vec![m, k] };
    let lo = Tensor { buf: sc.lo.buf.clone(), ptr: sc.lo.ptr, dtype: DType::BF16, shape: vec![m, k] };
    dev.launch_n("k_split_hilo_bf16", m * k, &[Arg::Ptr(a.ptr), Arg::Ptr(hi.ptr), Arg::Ptr(lo.ptr), Arg::I64((m * k) as i64)])?;
    ops::gemm_bf16(dev, &hi, w, b, Act::None, if add { Epi::AddRes(out) } else { Epi::Store }, out)?;
    ops::gemm_bf16(dev, &lo, w, None, Act::None, Epi::AddRes(out), out)
}

// ------------------------------------------------------------------------------------------------
// Decoder
// ------------------------------------------------------------------------------------------------

struct Layer {
    ln1: Tensor,
    qkv: Fp4Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    o: Fp4Linear,
    o_pqs: Tensor,
    ln2: Tensor,
    gate_up: Fp4Linear,
    down: Fp4Linear,
    down_pqs: Tensor,
}

/// Interleaved MRoPE table [L, 64, 2] (cos, sin) computed in f32 like `precompute_freqs_cis`.
pub fn mrope_table(pos: &[Vec<i64>; 3]) -> Vec<f32> {
    let l = pos[0].len();
    let inv: Vec<f32> = (0..64).map(|f| 1.0 / ROPE_THETA.powf((2 * f) as f32 / HEAD_DIM as f32)).collect();
    let mut tab = vec![0f32; l * 128];
    for i in 0..l {
        for f in 0..64 {
            let axis = if f < ROPE_DIMS[1] * 3 && f % 3 == 1 {
                1
            } else if f < ROPE_DIMS[2] * 3 && f % 3 == 2 {
                2
            } else {
                0
            };
            let ang = inv[f] * pos[axis][i] as f32;
            tab[(i * 64 + f) * 2] = ang.cos();
            tab[(i * 64 + f) * 2 + 1] = ang.sin();
        }
    }
    tab
}

// ------------------------------------------------------------------------------------------------
// Encoder
// ------------------------------------------------------------------------------------------------

enum Seg {
    Text(Vec<u32>),
    /// index into the vision-block list
    Vision(usize),
}

/// A vision block: two normalized CHW frames at (hb, wb).
struct VisionBlockIn {
    f0: Vec<f32>,
    f1: Option<Vec<f32>>,
    hb: usize,
    wb: usize,
}

pub struct TextEncoder {
    dev: Arc<Device>,
    st: SafeTensors,
    vision: VisionTower,
    layers: Vec<Layer>,
    tok: Tokenizer,
    /// host-side int8 embedding table rows are gathered on the CPU (saves 0.8 GB of VRAM)
    embed_name: String,
    embed_scale: Vec<f32>,
    pub last_timing: std::sync::Mutex<EncodeTiming>,
}

impl TextEncoder {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Self> {
        let t0 = std::time::Instant::now();
        ops::gemm_init(&dev)?;
        ops::attn_init(&dev)?;
        fp4_init(&dev)?;
        let st = SafeTensors::open(path)?;
        let mut uploaded = 0usize;
        let vision = {
            let mut l = Loader::new(&st, dev.clone());
            let v = VisionTower::load(&mut l).context("loading vision tower")?;
            uploaded += l.uploaded;
            v
        };
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("model.layers.{i}");
            let mut l = Loader::new(&st, dev.clone());
            let a = format!("{p}.self_attn");
            let m = format!("{p}.mlp");
            let layer = Layer {
                ln1: l.bf16(&format!("{p}.input_layernorm.weight"))?,
                qkv: load_fp4_cat(&st, &dev, &[&format!("{a}.q_proj"), &format!("{a}.k_proj"), &format!("{a}.v_proj")], &mut uploaded)?,
                q_norm: l.bf16(&format!("{a}.q_norm.weight"))?,
                k_norm: l.bf16(&format!("{a}.k_norm.weight"))?,
                o: load_fp4_cat(&st, &dev, &[&format!("{a}.o_proj")], &mut uploaded)?,
                o_pqs: l.bf16(&format!("{a}.o_proj.pre_quant_scale"))?,
                ln2: l.bf16(&format!("{p}.post_attention_layernorm.weight"))?,
                gate_up: load_fp4_cat(&st, &dev, &[&format!("{m}.gate_proj"), &format!("{m}.up_proj")], &mut uploaded)?,
                down: load_fp4_cat(&st, &dev, &[&format!("{m}.down_proj")], &mut uploaded)?,
                down_pqs: l.bf16(&format!("{m}.down_proj.pre_quant_scale"))?,
            };
            ensure!(layer.qkv.n == QKV_N && layer.qkv.k == HIDDEN && layer.o.k == Q_DIM && layer.gate_up.n == 2 * INTER && layer.down.k == INTER, "layer {i}: unexpected shapes");
            uploaded += l.uploaded;
            layers.push(layer);
        }
        ensure!(!st.has(&format!("model.layers.{LAYERS}.input_layernorm.weight")), "checkpoint has more than {LAYERS} layers");
        let embed_name = "model.embed_tokens.weight".to_string();
        let einfo = st.info(&embed_name)?;
        ensure!(einfo.dtype == "I8" && einfo.shape[1] == HIDDEN, "embed_tokens: expected int8 [V, {HIDDEN}]");
        let meta: serde_json::Value = serde_json::from_slice(st.bytes("model.embed_tokens.comfy_quant")?)?;
        ensure!(meta["format"] == "int8_tensorwise" && meta.get("convrot").map(|v| v != true).unwrap_or(true), "embed_tokens: unsupported quant {meta}");
        let embed_scale = st.f32s("model.embed_tokens.weight_scale")?;
        ensure!(embed_scale.len() == einfo.shape[0]);
        dev.sync()?;
        crate::info!("  text encoder: {:.2} GB uploaded in {:.1}s", uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        Ok(TextEncoder { dev, st, vision, layers, tok: Tokenizer::new()?, embed_name, embed_scale, last_timing: Default::default() })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.dev
    }

    /// Token / vision-block layout of the presentation (MiniMaxH3Tokenizer.tokenize_with_weights).
    fn layout(&self, prompt: &str, items: &[RefItem]) -> Result<(Vec<Seg>, Vec<VisionBlockIn>)> {
        let mut segs = Vec::new();
        let mut vis = Vec::new();
        let tok = &self.tok;
        let add_text = |segs: &mut Vec<Seg>, s: &str| {
            if !s.is_empty() {
                segs.push(Seg::Text(tokenize_piece(tok, s)));
            }
        };
        let (mut ni, mut na, mut nv) = (0, 0, 0);
        for it in items {
            match it {
                RefItem::Image { w, h, rgb } => {
                    ensure!(rgb.len() == w * h * 3, "image buffer size mismatch");
                    ni += 1;
                    add_text(&mut segs, &format!("<Picture {ni}>: "));
                    let (hb, wb) = vision_size(*h, *w);
                    vis.push(VisionBlockIn { f0: vision_pixels(*w, *h, rgb, hb, wb), f1: None, hb, wb });
                    segs.push(Seg::Vision(vis.len() - 1));
                }
                RefItem::Audio => {
                    na += 1;
                    add_text(&mut segs, &format!("<Audio {na}>: "));
                }
                RefItem::Video { frames, timestamps } => {
                    ensure!(!frames.is_empty(), "video reference without frames");
                    nv += 1;
                    let mut fr: Vec<&(usize, usize, Vec<u8>)> = frames.iter().collect();
                    let mut ts: Vec<f64> = if timestamps.len() == frames.len() { timestamps.iter().map(|&t| t as f64).collect() } else { (0..frames.len()).map(|i| i as f64 / 2.0).collect() };
                    if fr.len() % 2 == 1 {
                        fr.push(*fr.last().unwrap());
                        ts.push(*ts.last().unwrap());
                    }
                    add_text(&mut segs, &format!("<Video {nv}>: "));
                    for i in (0..fr.len()).step_by(2) {
                        // Python: (ts[i] + ts[i+1]) / 2.0 with float32-derived python floats
                        let bt = (ts[i] + ts[i + 1]) / 2.0;
                        add_text(&mut segs, &format!("<{} seconds>", fmt_1f(bt)));
                        let (w, h, a) = fr[i];
                        let (w2, h2, b) = fr[i + 1];
                        ensure!(w == w2 && h == h2, "video frames must share one size");
                        ensure!(a.len() == w * h * 3 && b.len() == w * h * 3, "frame buffer size mismatch");
                        let (hb, wb) = vision_size(*h, *w);
                        vis.push(VisionBlockIn { f0: vision_pixels(*w, *h, a, hb, wb), f1: Some(vision_pixels(*w, *h, b, hb, wb)), hb, wb });
                        segs.push(Seg::Vision(vis.len() - 1));
                    }
                }
            }
        }
        add_text(&mut segs, prompt);
        if segs.is_empty() {
            segs.push(Seg::Text(vec![PAD]));
        }
        Ok((segs, vis))
    }

    /// Token ids of the presentation with each vision block collapsed to one `-1` entry (for checks).
    pub fn tokenize(&self, prompt: &str, items: &[RefItem]) -> Result<Vec<i64>> {
        let (segs, _) = self.layout(prompt, items)?;
        let mut out = Vec::new();
        for s in &segs {
            match s {
                Seg::Text(t) => out.extend(t.iter().map(|&v| v as i64)),
                Seg::Vision(_) => out.extend([VISION_START as i64, -1, VISION_END as i64]),
            }
        }
        Ok(out)
    }

    /// Embedding rows (f32) of `ids` gathered from the int8 table on the host: q * scale[row].
    fn embed_rows(&self, ids: &[u32], out: &mut [f32]) -> Result<()> {
        let table = self.st.bytes(&self.embed_name)?;
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            ensure!(id < self.embed_scale.len(), "token id {id} out of range");
            let row = &table[id * HIDDEN..(id + 1) * HIDDEN];
            let s = self.embed_scale[id];
            for (o, &q) in out[i * HIDDEN..(i + 1) * HIDDEN].iter_mut().zip(row) {
                *o = (q as i8) as f32 * s;
            }
        }
        Ok(())
    }

    pub fn encode(&self, prompt: &str, items: &[RefItem]) -> Result<Conditioning> {
        let dev = &self.dev;
        let t0 = std::time::Instant::now();
        let (segs, vis_in) = self.layout(prompt, items)?;
        // ---- vision tower per block
        let mut vis = Vec::with_capacity(vis_in.len());
        for v in &vis_in {
            let patches = patchify(&v.f0, v.f1.as_ref().unwrap_or(&v.f0), v.hb, v.wb);
            vis.push(self.vision.forward(dev, &patches, v.hb / PATCH, v.wb / PATCH).context("vision tower")?);
        }
        dev.sync()?;
        let t_vis = t0.elapsed().as_secs_f64();
        // ---- sequence: ids, tags, MRoPE positions (qwen2vl_mrope_position_ids), embeddings
        let mut ids: Vec<i32> = Vec::new();
        let mut tags: Vec<u8> = Vec::new();
        let mut pos: [Vec<i64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
        let mut vis_rows: Vec<(usize, usize)> = Vec::new(); // (row, block index)
        let mut offset: i64 = 0;
        for s in &segs {
            match s {
                Seg::Text(t) => {
                    for &id in t {
                        let i = ids.len() as i64;
                        ids.push(id as i32);
                        tags.push(1);
                        for a in pos.iter_mut() {
                            a.push(i + offset);
                        }
                    }
                }
                Seg::Vision(bi) => {
                    let v = &vis[*bi];
                    // <|vision_start|>
                    let i = ids.len() as i64;
                    ids.push(VISION_START as i32);
                    tags.push(0);
                    for a in pos.iter_mut() {
                        a.push(i + offset);
                    }
                    let start = ids.len();
                    let n = v.merged.shape[0];
                    let (mh, mw) = (v.grid_h / 2, v.grid_w / 2);
                    let base = start as i64 + offset;
                    for j in 0..n {
                        ids.push(-1);
                        tags.push(0);
                        pos[0].push(base);
                        pos[1].push(base + (j / mw) as i64);
                        pos[2].push(base + (j % mw) as i64);
                    }
                    vis_rows.push((start, *bi));
                    offset += mh.max(mw) as i64 - n as i64;
                    let i = ids.len() as i64;
                    ids.push(VISION_END as i32);
                    tags.push(0);
                    for a in pos.iter_mut() {
                        a.push(i + offset);
                    }
                }
            }
        }
        let l = ids.len();
        let mut host = vec![0f32; l * HIDDEN];
        {
            // gather text rows in runs
            let mut i = 0;
            while i < l {
                if ids[i] < 0 {
                    i += 1;
                    continue;
                }
                let mut j = i;
                while j < l && ids[j] >= 0 {
                    j += 1;
                }
                let run: Vec<u32> = ids[i..j].iter().map(|&v| v as u32).collect();
                self.embed_rows(&run, &mut host[i * HIDDEN..j * HIDDEN])?;
                i = j;
            }
        }
        let x = Tensor::from_f32(dev, &host, &[l, HIDDEN])?;
        drop(host);
        for &(row, bi) in &vis_rows {
            let m = &vis[bi].merged;
            dev.dtod(x.ptr + (row * HIDDEN * 4) as u64, m.ptr, m.bytes())?;
        }
        let rope = Tensor::from_f32(dev, &mrope_table(&pos), &[l, 64, 2])?;
        let deep: Vec<(usize, &[Tensor])> = vis_rows.iter().map(|&(row, bi)| (row, vis[bi].deepstack.as_slice())).collect();
        self.forward(&x, &rope, &deep)?;
        drop(deep);
        drop(vis);
        let context = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        ops::to_bf16(dev, &x, &context)?;
        dev.sync()?;
        let total = t0.elapsed().as_secs_f64();
        *self.last_timing.lock().unwrap() = EncodeTiming { vision: t_vis, decoder: total - t_vis, total };
        Ok(Conditioning { context, tags, token_ids: ids })
    }

    /// Decoder forward over x f32 [L, 5120] in place (hidden state after layer 50, no final norm).
    fn forward(&self, x: &Tensor, rope: &Tensor, deepstack: &[(usize, &[Tensor])]) -> Result<()> {
        self.forward_layers(x, rope, deepstack, LAYERS, |_, _| Ok(()))
    }

    /// Forward through the first `n_layers` layers, calling `hook(layer_index, x)` after each.
    fn forward_layers<F: FnMut(usize, &Tensor) -> Result<()>>(&self, x: &Tensor, rope: &Tensor, deepstack: &[(usize, &[Tensor])], n_layers: usize, mut hook: F) -> Result<()> {
        let dev = &*self.dev;
        let l = x.shape[0];
        let xn = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        let qkv = Tensor::new(dev, DType::BF16, &[l, QKV_N])?;
        let attn = Tensor::new(dev, DType::BF16, &[l, Q_DIM])?;
        // MLP in row chunks so the [rows, 2*INTER] activations stay bounded for long sequences
        let chunk = l.min(4096);
        let gu = Tensor::new(dev, DType::BF16, &[chunk, 2 * INTER])?;
        let h = Tensor::new(dev, DType::BF16, &[chunk, INTER])?;
        for (li, layer) in self.layers.iter().take(n_layers).enumerate() {
            // ---- attention
            ops::rmsnorm(dev, x, Some(&layer.ln1), false, EPS, &xn)?;
            fp4_gemm(dev, xn.ptr, HIDDEN, l, &layer.qkv, qkv.ptr, QKV_N, Fp4Out::StoreBf16)?;
            let q_ptr = qkv.ptr;
            let k_ptr = qkv.ptr + (Q_DIM * 2) as u64;
            let v_ptr = qkv.ptr + ((Q_DIM + KV_HEADS * HEAD_DIM) * 2) as u64;
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
            // AWQ smoothing of the o_proj input
            dev.launch_n("k_mul_cols_bf16", l * Q_DIM / 2, &[Arg::Ptr(attn.ptr), Arg::Ptr(layer.o_pqs.ptr), Arg::I64(l as i64), Arg::I32(Q_DIM as i32)])?;
            fp4_gemm(dev, attn.ptr, Q_DIM, l, &layer.o, x.ptr, HIDDEN, Fp4Out::AddF32)?;
            // ---- MLP
            ops::rmsnorm(dev, x, Some(&layer.ln2), false, EPS, &xn)?;
            let mut r0 = 0;
            while r0 < l {
                let rows = chunk.min(l - r0);
                fp4_gemm(dev, xn.ptr + (r0 * HIDDEN * 2) as u64, HIDDEN, rows, &layer.gate_up, gu.ptr, 2 * INTER, Fp4Out::StoreBf16)?;
                dev.launch_n("k_swiglu_pqs_bf16", rows * INTER / 2, &[Arg::Ptr(gu.ptr), Arg::Ptr(layer.down_pqs.ptr), Arg::Ptr(h.ptr), Arg::I64(rows as i64), Arg::I32(INTER as i32)])?;
                fp4_gemm(dev, h.ptr, INTER, rows, &layer.down, x.ptr + (r0 * HIDDEN * 4) as u64, HIDDEN, Fp4Out::AddF32)?;
                r0 += rows;
            }
            // ---- DeepStack: visual features of vision blocks 8/16/24 added after decoder layers 0/1/2
            for &(row0, feats) in deepstack {
                if li < feats.len() {
                    ops::add_rows_f32(dev, x, row0, &feats[li])?;
                }
            }
            hook(li, x)?;
        }
        Ok(())
    }

}

impl TextEncoder {
    /// Debug: vision-tower outputs (merged, 3 deepstack) as host f32 for every vision block of `items`.
    pub fn vision_features(&self, items: &[RefItem]) -> Result<Vec<(Vec<f32>, Vec<Vec<f32>>)>> {
        let (_, vis_in) = self.layout("", items)?;
        let mut out = Vec::new();
        for v in &vis_in {
            let patches = patchify(&v.f0, v.f1.as_ref().unwrap_or(&v.f0), v.hb, v.wb);
            let o = self.vision.forward(&self.dev, &patches, v.hb / PATCH, v.wb / PATCH)?;
            let d = o.deepstack.iter().map(|t| t.to_f32_vec(&self.dev)).collect::<Result<Vec<_>>>()?;
            out.push((o.merged.to_f32_vec(&self.dev)?, d));
        }
        Ok(out)
    }

    /// Self-test + micro-benchmark of the fused NVFP4 GEMM on layer `li`'s linears: compares against
    /// dequantize-to-bf16 + bf16 GEMM and returns (name, M, rel_err, fused_ms, dequant+gemm_ms) rows.
    pub fn fp4_selftest(&self, li: usize, ms: &[usize]) -> Result<Vec<(String, usize, f64, f64, f64)>> {
        let dev = &*self.dev;
        let layer = &self.layers[li];
        let mut rows = Vec::new();
        let mut seed = 0x1234_5678u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        };
        for (name, lin) in [("qkv", &layer.qkv), ("o", &layer.o), ("gate_up", &layer.gate_up), ("down", &layer.down)] {
            let wdq = fp4_dequant(dev, lin)?;
            for &m in ms {
                let av: Vec<u16> = (0..m * lin.k).map(|_| half::bf16::from_f32(rnd()).to_bits()).collect();
                let a = Tensor::from_bf16(dev, &av, &[m, lin.k])?;
                let o1 = Tensor::zeros(dev, DType::F32, &[m, lin.n])?;
                let o2 = Tensor::zeros(dev, DType::F32, &[m, lin.n])?;
                fp4_gemm(dev, a.ptr, lin.k, m, lin, o1.ptr, lin.n, Fp4Out::AddF32)?;
                ops::gemm_bf16(dev, &a, &wdq, None, Act::None, Epi::Store, &o2)?;
                let v1 = o1.to_f32_vec(dev)?;
                let v2 = o2.to_f32_vec(dev)?;
                let (mut num, mut den) = (0f64, 0f64);
                for (p, q) in v1.iter().zip(&v2) {
                    num += ((p - q) as f64).powi(2);
                    den += (*q as f64).powi(2);
                }
                let rel = (num / den.max(1e-30)).sqrt();
                // timing
                let reps = 20;
                let ob = Tensor::new(dev, DType::BF16, &[m, lin.n])?;
                dev.sync()?;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    fp4_gemm(dev, a.ptr, lin.k, m, lin, ob.ptr, lin.n, Fp4Out::StoreBf16)?;
                }
                dev.sync()?;
                let t_fused = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    let w = fp4_dequant(dev, lin)?;
                    ops::gemm_bf16(dev, &a, &w, None, Act::None, Epi::Store, &ob)?;
                }
                dev.sync()?;
                let t_sep = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
                rows.push((name.to_string(), m, rel, t_fused, t_sep));
            }
        }
        Ok(rows)
    }
}

/// Free everything the encoder holds on the GPU and hand the pooled memory back to the driver
/// (call after dropping the encoder, before loading the DiT).
pub fn release_device_memory(dev: &Device) -> Result<()> {
    dev.sync()?;
    unsafe {
        use cudarc::driver::sys;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut d: sys::CUdevice = 0;
        if sys::cuCtxGetDevice(&mut d) == sys::CUresult::CUDA_SUCCESS && sys::cuDeviceGetDefaultMemPool(&mut pool, d) == sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemPoolTrimTo(pool, 0);
        }
    }
    Ok(())
}
