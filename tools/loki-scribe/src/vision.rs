//! Qwen3-VL-style vision tower from a GGUF mmproj (`clip.projector_type = qwen3vl_merger`, no DeepStack): a 27-block
//! ViT (1152 wide, 16 heads of 72, 16 px patches, 2D rope + learned 48x48 position table interpolated to the grid),
//! then a 2x2 patch merger (LayerNorm, 4608 -> 4608 -> GELU -> n_embd) giving one decoder embedding per 32x32 pixels.
//! Ported from loki-reshoot's conditioning encoder; activations run in ~f32 (hi/lo bf16 splits on the tensor cores).
use crate::cuda::{Arg, Device};
use crate::gguf::Gguf;
use crate::ops::{self, Act, Epi};
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Context, Result};
use std::sync::Arc;

const PATCH: usize = 16;
const MERGE: usize = 2;
pub const FACTOR: usize = PATCH * MERGE;

struct Block {
    ln1_w: Tensor,
    ln1_b: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    o_w: Tensor,
    o_b: Tensor,
    ln2_w: Tensor,
    ln2_b: Tensor,
    up_w: Tensor,
    up_b: Tensor,
    down_w: Tensor,
    down_b: Tensor,
}

pub struct Vision {
    dev: Arc<Device>,
    hidden: usize,
    heads: usize,
    head_dim: usize,
    inter: usize,
    pos_grid: usize,
    pub out_dim: usize,
    eps: f32,
    patch_w: Tensor, // bf16 [hidden, 3*16*16] (both temporal kernels summed: a still image repeats its frame)
    patch_b: Tensor,
    pos: Tensor, // f32 [grid*grid, hidden]
    blocks: Vec<Block>,
    post_w: Tensor,
    post_b: Tensor,
    mm0_w: Tensor,
    mm0_b: Tensor,
    mm2_w: Tensor,
    mm2_b: Tensor,
    pub vram_bytes: usize,
}

/// An image ready for the tower: normalized CHW f32 at (h, w), both multiples of 32.
pub struct Prepared {
    pub w: usize,
    pub h: usize,
    pub chw: Vec<f32>,
}

impl Prepared {
    /// Merged tokens (decoder embeddings) along y and x.
    pub fn grid(&self) -> (usize, usize) {
        (self.h / FACTOR, self.w / FACTOR)
    }
    pub fn tokens(&self) -> usize {
        let (gy, gx) = self.grid();
        gy * gx
    }
}

fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 {
        2.0 * (x / 2.0).round()
    } else {
        r
    }
}

/// Qwen2-VL smart_resize: both sides multiples of 32, pixel count within [min_tokens, max_tokens] * 32^2.
pub fn target_size(w: usize, h: usize, min_tokens: usize, max_tokens: usize) -> (usize, usize) {
    let f = FACTOR as f64;
    let (min_px, max_px) = ((min_tokens * FACTOR * FACTOR) as f64, (max_tokens * FACTOR * FACTOR) as f64);
    let (hf, wf) = (h as f64, w as f64);
    let mut hb = round_half_even(hf / f) * f;
    let mut wb = round_half_even(wf / f) * f;
    if hb * wb > max_px {
        let beta = (hf * wf / max_px).sqrt();
        hb = (hf / beta / f).floor() * f;
        wb = (wf / beta / f).floor() * f;
    } else if hb * wb < min_px {
        let beta = (min_px / (hf * wf)).sqrt();
        hb = (hf * beta / f).ceil() * f;
        wb = (wf * beta / f).ceil() * f;
    }
    ((wb as usize).max(FACTOR), (hb as usize).max(FACTOR))
}

/// Decode and resize an image (bicubic), normalized to [-1, 1].
pub fn prepare(img: &image::DynamicImage, min_tokens: usize, max_tokens: usize) -> Prepared {
    let rgb = img.to_rgb8();
    let (w0, h0) = (rgb.width() as usize, rgb.height() as usize);
    let (w, h) = target_size(w0, h0, min_tokens, max_tokens);
    let rs = if (w, h) == (w0, h0) { rgb } else { image::imageops::resize(&rgb, w as u32, h as u32, image::imageops::FilterType::CatmullRom) };
    let mut chw = vec![0f32; 3 * w * h];
    for (i, p) in rs.pixels().enumerate() {
        for c in 0..3 {
            chw[c * w * h + i] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    Prepared { w, h, chw }
}

/// Patch (row, col) of token `t` in merge-window order for a grid `gw` patches wide.
fn token_rc(t: usize, gw: usize) -> (usize, usize) {
    let mw = gw / 2;
    let br = t / (mw * 4);
    let r2 = t % (mw * 4);
    let bc = r2 / 4;
    let r3 = r2 % 4;
    (br * 2 + r3 / 2, bc * 2 + r3 % 2)
}

fn f16_or_f32_to_bf16(dev: &Device, g: &Gguf, name: &str, shape: &[usize]) -> Result<Tensor> {
    let v = g.f32(name)?;
    ensure!(v.len() == shape.iter().product::<usize>(), "{name}: {} values, want {:?}", v.len(), shape);
    let b: Vec<u16> = v.iter().map(|x| half::bf16::from_f32(*x).to_bits()).collect();
    Tensor::from_bf16(dev, &b, shape)
}

fn f32_t(dev: &Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let v = g.f32(name)?;
    let n = v.len();
    Tensor::from_f32(dev, &v, &[n])
}

impl Vision {
    pub fn load(dev: Arc<Device>, g: &Gguf) -> Result<Vision> {
        let proj = g.s("clip.projector_type").unwrap_or("");
        ensure!(proj == "qwen3vl_merger", "{}: projector type {proj:?} unsupported (want qwen3vl_merger)", g.path.display());
        let hidden = g.u("clip.vision.embedding_length")?;
        let heads = g.u("clip.vision.attention.head_count")?;
        let inter = g.u("clip.vision.feed_forward_length")?;
        let depth = g.u("clip.vision.block_count")?;
        let out_dim = g.u("clip.vision.projection_dim")?;
        let eps = g.f("clip.vision.attention.layer_norm_epsilon").unwrap_or(1e-6) as f32;
        ensure!(g.u("clip.vision.patch_size")? == PATCH && g.u("clip.vision.spatial_merge_size").unwrap_or(2) == MERGE, "patch/merge size unsupported");
        let head_dim = hidden / heads;
        ensure!(head_dim == 72, "vision head dim {head_dim} unsupported (want 72)");
        if let Some(ds) = g.get("clip.vision.is_deepstack_layers").and_then(|v| v.as_arr()) {
            ensure!(ds.iter().all(|v| matches!(v, crate::gguf::Value::Bool(false))), "DeepStack vision towers (Qwen3-VL) are not supported");
        }
        // patch embedding: two temporal kernels [hidden][3][16][16]; a still image feeds both the same frame
        let w0 = g.f32("v.patch_embd.weight")?;
        let w1 = if g.has("v.patch_embd.weight.1") { g.f32("v.patch_embd.weight.1")? } else { vec![0f32; w0.len()] };
        ensure!(w0.len() == hidden * 3 * PATCH * PATCH, "patch embedding shape");
        let pw: Vec<u16> = w0.iter().zip(&w1).map(|(a, b)| half::bf16::from_f32(a + b).to_bits()).collect();
        let patch_w = Tensor::from_bf16(&dev, &pw, &[hidden, 3 * PATCH * PATCH])?;
        let patch_b = f32_t(&dev, g, "v.patch_embd.bias")?;
        let pos = f32_t(&dev, g, "v.position_embd.weight")?;
        let npos = pos.numel() / hidden;
        let pos_grid = (npos as f64).sqrt() as usize;
        ensure!(pos_grid * pos_grid == npos, "position table is not square");
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let p = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(Block {
                ln1_w: f32_t(&dev, g, &p("ln1.weight"))?,
                ln1_b: f32_t(&dev, g, &p("ln1.bias"))?,
                qkv_w: f16_or_f32_to_bf16(&dev, g, &p("attn_qkv.weight"), &[3 * hidden, hidden])?,
                qkv_b: f32_t(&dev, g, &p("attn_qkv.bias"))?,
                o_w: f16_or_f32_to_bf16(&dev, g, &p("attn_out.weight"), &[hidden, hidden])?,
                o_b: f32_t(&dev, g, &p("attn_out.bias"))?,
                ln2_w: f32_t(&dev, g, &p("ln2.weight"))?,
                ln2_b: f32_t(&dev, g, &p("ln2.bias"))?,
                up_w: f16_or_f32_to_bf16(&dev, g, &p("ffn_up.weight"), &[inter, hidden])?,
                up_b: f32_t(&dev, g, &p("ffn_up.bias"))?,
                down_w: f16_or_f32_to_bf16(&dev, g, &p("ffn_down.weight"), &[hidden, inter])?,
                down_b: f32_t(&dev, g, &p("ffn_down.bias"))?,
            });
        }
        let md = hidden * 4;
        let v = Vision {
            hidden,
            heads,
            head_dim,
            inter,
            pos_grid,
            out_dim,
            eps,
            patch_w,
            patch_b,
            pos,
            blocks,
            post_w: f32_t(&dev, g, "v.post_ln.weight")?,
            post_b: f32_t(&dev, g, "v.post_ln.bias")?,
            mm0_w: f16_or_f32_to_bf16(&dev, g, "mm.0.weight", &[md, md]).context("mm.0")?,
            mm0_b: f32_t(&dev, g, "mm.0.bias")?,
            mm2_w: f16_or_f32_to_bf16(&dev, g, "mm.2.weight", &[out_dim, md]).context("mm.2")?,
            mm2_b: f32_t(&dev, g, "mm.2.bias")?,
            vram_bytes: 0,
            dev,
        };
        let vram = g.tensor_bytes() + v.pos.bytes();
        Ok(Vision { vram_bytes: vram, ..v })
    }

    /// Encode one prepared image -> decoder embeddings f32 [tokens, out_dim] (host), in row-major merged-grid order.
    pub fn encode(&self, img: &Prepared) -> Result<Vec<f32>> {
        let dev = &*self.dev;
        let (gh, gw) = (img.h / PATCH, img.w / PATCH);
        let n = gh * gw;
        let (c, hd, nh) = (self.hidden, self.head_dim, self.heads);
        let pf = PATCH * PATCH * 3;
        // patches in merge-window order, features (c, py, px)
        let mut patches = vec![0f32; n * pf];
        for t in 0..n {
            let (r, col) = token_rc(t, gw);
            let o = &mut patches[t * pf..(t + 1) * pf];
            for ch in 0..3 {
                for py in 0..PATCH {
                    let src = &img.chw[(ch * img.h + r * PATCH + py) * img.w + col * PATCH..][..PATCH];
                    o[ch * 256 + py * 16..][..16].copy_from_slice(src);
                }
            }
        }
        let scratch_k = (3 * c).max(self.inter).max(4 * c);
        let hi = Tensor::new(dev, DType::BF16, &[n * scratch_k])?;
        let lo = Tensor::new(dev, DType::BF16, &[n * scratch_k])?;
        let x = Tensor::new(dev, DType::F32, &[n, c])?;
        {
            let p = Tensor::from_f32(dev, &patches, &[n, pf])?;
            self.lin(&hi, &lo, &p, &self.patch_w, Some(&self.patch_b), &x, false)?;
        }
        self.add_pos(&x, gh, gw)?;
        // 2D rope table [n, hd/2, 2]: hd/4 row frequencies then hd/4 column frequencies
        let rope = {
            let q = hd / 4;
            let mut tab = vec![0f32; n * (hd / 2) * 2];
            let inv: Vec<f32> = (0..q).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / (hd / 2) as f32)).collect();
            for t in 0..n {
                let (r, col) = token_rc(t, gw);
                for i in 0..q {
                    let (ar, ac) = (r as f32 * inv[i], col as f32 * inv[i]);
                    tab[(t * (hd / 2) + i) * 2] = ar.cos();
                    tab[(t * (hd / 2) + i) * 2 + 1] = ar.sin();
                    tab[(t * (hd / 2) + q + i) * 2] = ac.cos();
                    tab[(t * (hd / 2) + q + i) * 2 + 1] = ac.sin();
                }
            }
            Tensor::from_f32(dev, &tab, &[n, hd / 2, 2])?
        };
        let ln = Tensor::new(dev, DType::F32, &[n, c])?;
        let qkv = Tensor::new(dev, DType::F32, &[n, 3 * c])?;
        let att = Tensor::new(dev, DType::F32, &[n, c])?;
        let up = Tensor::new(dev, DType::F32, &[n, self.inter])?;
        let scale_log2 = (1.0 / (hd as f32).sqrt()) * std::f32::consts::LOG2_E;
        for b in &self.blocks {
            ops::layernorm(dev, &x, n, c, &b.ln1_w, &b.ln1_b, self.eps, &ln)?;
            self.lin(&hi, &lo, &ln, &b.qkv_w, Some(&b.qkv_b), &qkv, false)?;
            dev.launch_n("k_rope_vision_f32", n * 2 * nh * (hd / 2), &[Arg::Ptr(qkv.ptr), Arg::I32(n as i32), Arg::I32(nh as i32), Arg::I32(hd as i32), Arg::Ptr(rope.ptr)])?;
            dev.launch_n("k_split_hilo_bf16", n * 3 * c, &[Arg::Ptr(qkv.ptr), Arg::Ptr(hi.ptr), Arg::Ptr(lo.ptr), Arg::I64((n * 3 * c) as i64)])?;
            dev.launch("k_attn_split_d72", (((n + 127) / 128) as u32, nh as u32, 1), (256, 1, 1), ops::VIS_ATTN_SMEM, &[Arg::Ptr(hi.ptr), Arg::Ptr(lo.ptr), Arg::Ptr(att.ptr), Arg::I32(n as i32), Arg::I32(nh as i32), Arg::F32(scale_log2)])?;
            self.lin(&hi, &lo, &att, &b.o_w, Some(&b.o_b), &x, true)?;
            ops::layernorm(dev, &x, n, c, &b.ln2_w, &b.ln2_b, self.eps, &ln)?;
            self.lin(&hi, &lo, &ln, &b.up_w, Some(&b.up_b), &up, false)?;
            ops::act_inplace(dev, &up, Act::GeluTanh)?;
            self.lin(&hi, &lo, &up, &b.down_w, Some(&b.down_b), &x, true)?;
        }
        // merger: LayerNorm (per patch) -> [n/4, 4c] -> fc -> GELU -> fc
        ops::layernorm(dev, &x, n, c, &self.post_w, &self.post_b, self.eps, &ln)?;
        let rows = n / 4;
        let lnv = ln.reshape(&[rows, 4 * c]);
        let h = Tensor::new(dev, DType::F32, &[rows, 4 * c])?;
        self.lin(&hi, &lo, &lnv, &self.mm0_w, Some(&self.mm0_b), &h, false)?;
        ops::act_inplace(dev, &h, Act::GeluErf)?;
        let out = Tensor::new(dev, DType::F32, &[rows, self.out_dim])?;
        self.lin(&hi, &lo, &h, &self.mm2_w, Some(&self.mm2_b), &out, false)?;
        // merge windows are already in row-major merged-grid order
        out.to_f32_vec(dev)
    }

    /// x += bilinear (align_corners) interpolation of the position table onto the gh x gw patch grid.
    fn add_pos(&self, x: &Tensor, gh: usize, gw: usize) -> Result<()> {
        let g = self.pos_grid;
        let lin = |steps: usize| -> Vec<f32> {
            if steps == 1 {
                return vec![0.0];
            }
            (0..steps).map(|i| i as f32 * (g - 1) as f32 / (steps - 1) as f32).collect()
        };
        let (hs, ws) = (lin(gh), lin(gw));
        let n = gh * gw;
        let mut idx = vec![0i32; n * 4];
        let mut wt = vec![0f32; n * 4];
        for t in 0..n {
            let (r, c) = token_rc(t, gw);
            let (hi, wi) = (hs[r], ws[c]);
            let (hf, wf) = (hi.floor() as usize, wi.floor() as usize);
            let (hc, wc) = ((hf + 1).min(g - 1), (wf + 1).min(g - 1));
            let (dh, dw) = (hi - hf as f32, wi - wf as f32);
            idx[t * 4..t * 4 + 4].copy_from_slice(&[(hf * g + wf) as i32, (hf * g + wc) as i32, (hc * g + wf) as i32, (hc * g + wc) as i32]);
            wt[t * 4..t * 4 + 4].copy_from_slice(&[(1.0 - dh) * (1.0 - dw), (1.0 - dh) * dw, dh * (1.0 - dw), dh * dw]);
        }
        let it = Tensor::from_buf(self.dev.upload(&idx)?, DType::I32, &[n * 4]);
        let wtt = Tensor::from_f32(&self.dev, &wt, &[n * 4])?;
        self.dev.launch_n("k_vis_pos_embed", n * self.hidden, &[Arg::Ptr(x.ptr), Arg::Ptr(self.pos.ptr), Arg::Ptr(it.ptr), Arg::Ptr(wtt.ptr), Arg::I32(n as i32), Arg::I32(self.hidden as i32)])
    }

    /// out (+)= a . w^T + b with f32 activations split into bf16 hi + lo (two tensor-core GEMMs, f32 accumulate).
    fn lin(&self, hi: &Tensor, lo: &Tensor, a: &Tensor, w: &Tensor, b: Option<&Tensor>, out: &Tensor, add: bool) -> Result<()> {
        let dev = &*self.dev;
        let m = a.shape[0];
        let k: usize = a.shape[1..].iter().product();
        let h = Tensor { buf: hi.buf.clone(), ptr: hi.ptr, dtype: DType::BF16, shape: vec![m, k] };
        let l = Tensor { buf: lo.buf.clone(), ptr: lo.ptr, dtype: DType::BF16, shape: vec![m, k] };
        dev.launch_n("k_split_hilo_bf16", m * k, &[Arg::Ptr(a.ptr), Arg::Ptr(h.ptr), Arg::Ptr(l.ptr), Arg::I64((m * k) as i64)])?;
        ops::gemm_bf16(dev, &h, w, b, Act::None, if add { Epi::AddRes(out) } else { Epi::Store }, out)?;
        ops::gemm_bf16(dev, &l, w, None, Act::None, Epi::AddRes(out), out)
    }
}
