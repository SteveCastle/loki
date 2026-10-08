//! Qwen3-VL vision tower (Qwen3.5-style ViT with DeepStack), bf16 GEMMs with an f32 residual stream.
use crate::cuda::Device;
use crate::ops::{self, Act, AttnArgs, AttnView, Epi};
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;
use anyhow::Result;
use std::sync::Arc;

pub const HIDDEN: usize = 1152;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 72;
pub const INTER: usize = 4304;
pub const DEPTH: usize = 27;
pub const DEEPSTACK: [usize; 3] = [8, 16, 24];
pub const OUT_HIDDEN: usize = 4096;
pub const MERGE_DIM: usize = HIDDEN * 4;
pub const POS_GRID: usize = 48;

struct Block {
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

pub struct VisionTower {
    dev: Arc<Device>,
    patch_w: Tensor,
    patch_b: Tensor,
    pos_table: Tensor,
    blocks: Vec<Block>,
    merger: Merger,
    deepstack: Vec<Merger>,
}

pub struct VisionOutput {
    /// merged tokens [N/4, 4096] f32
    pub merged: Tensor,
    /// deepstack features, 3 x [N/4, 4096] f32
    pub deepstack: Vec<Tensor>,
    pub grid_h: usize,
    pub grid_w: usize,
}

impl VisionTower {
    pub fn load(l: &mut Loader) -> Result<VisionTower> {
        let p = "model.visual";
        let patch_w = l.bf16(&format!("{p}.patch_embed.proj.weight"))?.reshape(&[HIDDEN, 3 * 2 * 16 * 16]);
        let patch_b = l.f32(&format!("{p}.patch_embed.proj.bias"))?;
        let pos_table = l.bf16(&format!("{p}.pos_embed.weight"))?;
        let mut blocks = Vec::with_capacity(DEPTH);
        for i in 0..DEPTH {
            let b = format!("{p}.blocks.{i}");
            blocks.push(Block {
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
        let merger = load_merger(l, &format!("{p}.merger"), HIDDEN)?;
        let mut deepstack = Vec::new();
        for i in 0..3 {
            deepstack.push(load_merger(l, &format!("{p}.deepstack_merger_list.{i}"), MERGE_DIM)?);
        }
        Ok(VisionTower { dev: l.dev.clone(), patch_w, patch_b, pos_table, blocks, merger, deepstack })
    }

    /// `img`: f32 HWC in [0,1], size (hp, wp) multiples of 32.
    pub fn forward(&self, img: &[f32], hp: usize, wp: usize) -> Result<VisionOutput> {
        let dev = &self.dev;
        assert!(hp % 32 == 0 && wp % 32 == 0, "vision input must be a multiple of 32");
        let (gh, gw) = (hp / 16, wp / 16);
        let n = gh * gw;
        // normalize to [-1, 1] (mean 0.5, std 0.5)
        let norm: Vec<f32> = img.iter().map(|v| (v - 0.5) / 0.5).collect();
        let img_t = Tensor::from_f32(dev, &norm, &[hp, wp, 3])?;
        let patches = Tensor::new(dev, DType::BF16, &[n, 1536])?;
        ops::vision_patchify(dev, &img_t, hp, wp, &patches)?;
        drop(img_t);
        // patch embed -> x f32 [n, 1152]
        let x = Tensor::new(dev, DType::F32, &[n, HIDDEN])?;
        ops::gemm_bf16(dev, &patches, &self.patch_w, Some(&self.patch_b), Act::None, Epi::Store, &x)?;
        drop(patches);
        // position embedding (bilinear interpolation of the 48x48 table), in merge-window token order
        {
            let mut idx = vec![0i32; n * 4];
            let mut wts = vec![0f32; n * 4];
            let lin = |size: usize, i: usize| -> f32 { if size == 1 { 0.0 } else { i as f32 * (POS_GRID - 1) as f32 / (size - 1) as f32 } };
            for t in 0..n {
                let (r, c) = token_rc(t, gw);
                let hi = lin(gh, r);
                let wi = lin(gw, c);
                let hf = hi.floor() as usize;
                let wf = wi.floor() as usize;
                let hc = (hf + 1).min(POS_GRID - 1);
                let wc = (wf + 1).min(POS_GRID - 1);
                let dh = hi - hf as f32;
                let dw = wi - wf as f32;
                idx[t * 4] = (hf * POS_GRID + wf) as i32;
                idx[t * 4 + 1] = (hf * POS_GRID + wc) as i32;
                idx[t * 4 + 2] = (hc * POS_GRID + wf) as i32;
                idx[t * 4 + 3] = (hc * POS_GRID + wc) as i32;
                wts[t * 4] = (1.0 - dh) * (1.0 - dw);
                wts[t * 4 + 1] = (1.0 - dh) * dw;
                wts[t * 4 + 2] = dh * (1.0 - dw);
                wts[t * 4 + 3] = dh * dw;
            }
            let idx_t = Tensor::from_buf(dev.upload(&idx)?, DType::F32, &[n * 4]);
            let w_t = Tensor::from_f32(dev, &wts, &[n * 4])?;
            ops::vision_pos_embed(dev, &x, &self.pos_table, &idx_t, &w_t)?;
        }
        // rope table [n, 36, 2]
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

        let ln_out = Tensor::new(dev, DType::BF16, &[n, HIDDEN])?;
        let qkv = Tensor::new(dev, DType::BF16, &[n, 3 * HIDDEN])?;
        let attn_out = Tensor::new(dev, DType::BF16, &[n, HIDDEN])?;
        let fc1 = Tensor::new(dev, DType::BF16, &[n, INTER])?;
        let mut deep = Vec::new();
        for (bi, b) in self.blocks.iter().enumerate() {
            ops::layernorm(dev, &x, Some(&b.ln1_w), Some(&b.ln1_b), 1e-6, &ln_out)?;
            ops::gemm_bf16(dev, &ln_out, &b.qkv_w, Some(&b.qkv_b), Act::None, Epi::Store, &qkv)?;
            ops::rope_vision(dev, &qkv, HEADS, HEAD_DIM, &rope)?;
            let view = |off: usize| AttnView { ptr: qkv.ptr + (off * 2) as u64, tok_stride: 3 * HIDDEN, head_stride: HEAD_DIM, len: n };
            ops::flash_attn(
                dev,
                &AttnArgs {
                    q: view(0),
                    k1: view(HIDDEN),
                    v1: view(2 * HIDDEN),
                    k2: AttnView::empty(),
                    v2: AttnView::empty(),
                    out: AttnView::contiguous(&attn_out, HEADS, HEAD_DIM),
                    hq: HEADS,
                    hk: HEADS,
                    d: HEAD_DIM,
                    kv_limit: None,
                    causal: false,
                    causal_off: 0,
                },
            )?;
            ops::gemm_bf16(dev, &attn_out, &b.proj_w, Some(&b.proj_b), Act::None, Epi::AddRes(&x), &x)?;
            ops::layernorm(dev, &x, Some(&b.ln2_w), Some(&b.ln2_b), 1e-6, &ln_out)?;
            ops::gemm_bf16(dev, &ln_out, &b.fc1_w, Some(&b.fc1_b), Act::GeluTanh, Epi::Store, &fc1)?;
            ops::gemm_bf16(dev, &fc1, &b.fc2_w, Some(&b.fc2_b), Act::None, Epi::AddRes(&x), &x)?;
            if let Some(di) = DEEPSTACK.iter().position(|&d| d == bi) {
                deep.push(self.merge(&self.deepstack[di], &x, n)?);
            }
        }
        let merged = self.merge(&self.merger, &x, n)?;
        Ok(VisionOutput { merged, deepstack: deep, grid_h: gh, grid_w: gw })
    }

    /// LayerNorm (over norm_dim) -> view [n/4, 4608] -> fc1 -> GELU(erf) -> fc2 -> f32 [n/4, 4096]
    fn merge(&self, m: &Merger, x: &Tensor, n: usize) -> Result<Tensor> {
        let dev = &self.dev;
        let rows = n / 4;
        let ln = Tensor::new(dev, DType::BF16, &[n, HIDDEN])?;
        if m.norm_dim == HIDDEN {
            ops::layernorm(dev, x, Some(&m.norm_w), Some(&m.norm_b), 1e-6, &ln)?;
        } else {
            let xv = x.reshape(&[rows, MERGE_DIM]);
            let lnv = ln.reshape(&[rows, MERGE_DIM]);
            ops::layernorm(dev, &xv, Some(&m.norm_w), Some(&m.norm_b), 1e-6, &lnv)?;
        }
        let lnv = ln.reshape(&[rows, MERGE_DIM]);
        let h = Tensor::new(dev, DType::BF16, &[rows, MERGE_DIM])?;
        ops::gemm_bf16(dev, &lnv, &m.fc1_w, Some(&m.fc1_b), Act::GeluErf, Epi::Store, &h)?;
        let out = Tensor::new(dev, DType::F32, &[rows, OUT_HIDDEN])?;
        ops::gemm_bf16(dev, &h, &m.fc2_w, Some(&m.fc2_b), Act::None, Epi::Store, &out)?;
        Ok(out)
    }
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
