//! The SAM 3 / 3.1 vision trunk: ViT-L/14 at 1008 px (72x72 tokens, 1024 wide, 32 blocks, 16 heads), windowed
//! attention (24x24) except blocks 7/15/23/31, 2-D axial RoPE (window-local; global blocks interpolated to the
//! 24x24 pretraining grid), tiled absolute position embedding, ln_pre, no ln_post.
//!
//! Tokens stay in window-major order through the whole trunk (see `kernels/sam.cu`), so a windowed block is a batch
//! of 9 contiguous attention problems and a global block is one 5184-token problem with per-token angles.
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, AttnArgs, AttnView, Epi};
use crate::tensor::{DType, Tensor};
use crate::weights::{Linear, Norm, Weights};
use anyhow::Result;
use std::sync::Arc;

pub const IMG: usize = 1008;
pub const GRID: usize = 72;
pub const NTOK: usize = GRID * GRID;
pub const DIM: usize = 1024;
const HEADS: usize = 16;
const HD: usize = 64;
const WIN: usize = 24;
const GLOBAL: [usize; 4] = [7, 15, 23, 31];
const KP: usize = 592; // 3*14*14 = 588 padded to a multiple of 8

struct Block {
    norm1: Norm,
    qkv: Linear,
    proj: Linear,
    norm2: Norm,
    fc1: Linear,
    fc2: Linear,
    window: bool,
}

pub struct Vit {
    dev: Arc<Device>,
    patch: Linear,
    pos: Tensor,
    ln_pre: Norm,
    blocks: Vec<Block>,
    rope_win: Tensor,
    rope_glob: Tensor,
}

/// compute_axial_cis(dim=64): 16 frequencies for x then 16 for y, as (cos, sin) pairs per token.
fn axial_rope(positions: &[(f32, f32)]) -> Vec<f32> {
    let theta = 10000f32;
    let freqs: Vec<f32> = (0..16).map(|i| 1.0 / theta.powf((4 * i) as f32 / HD as f32)).collect();
    let mut out = Vec::with_capacity(positions.len() * 64);
    for &(tx, ty) in positions {
        for f in &freqs {
            let a = tx * f;
            out.push(a.cos());
            out.push(a.sin());
        }
        for f in &freqs {
            let a = ty * f;
            out.push(a.cos());
            out.push(a.sin());
        }
    }
    out
}

/// Window-major token index -> (y, x) on the 72x72 grid.
pub fn win_yx(t: usize) -> (usize, usize) {
    let (win, l) = (t / 576, t % 576);
    ((win / 3) * WIN + l / WIN, (win % 3) * WIN + l % WIN)
}

impl Vit {
    pub fn load(w: &Weights, prefix: &str) -> Result<Vit> {
        let dev = w.dev.clone();
        let pw = w.host(&format!("{prefix}.patch_embed.proj.weight"))?; // [1024, 3, 14, 14]
        let mut pp = vec![0f32; DIM * KP];
        for o in 0..DIM {
            pp[o * KP..o * KP + 588].copy_from_slice(&pw[o * 588..(o + 1) * 588]);
        }
        let patch = w.linear_from_host(&pp, None, DIM, KP)?;
        let pos = w.f32(&format!("{prefix}.pos_embed"))?.reshape(&[577, DIM]);
        let ln_pre = w.norm(&format!("{prefix}.ln_pre"), 1e-5)?;
        let mut blocks = Vec::new();
        for i in 0..32 {
            let p = format!("{prefix}.blocks.{i}");
            blocks.push(Block {
                norm1: w.norm(&format!("{p}.norm1"), 1e-5)?,
                qkv: w.linear(&format!("{p}.attn.qkv"))?,
                proj: w.linear(&format!("{p}.attn.proj"))?,
                norm2: w.norm(&format!("{p}.norm2"), 1e-5)?,
                fc1: w.linear(&format!("{p}.mlp.fc1"))?,
                fc2: w.linear(&format!("{p}.mlp.fc2"))?,
                window: !GLOBAL.contains(&i),
            });
        }
        let win_pos: Vec<(f32, f32)> = (0..WIN * WIN).map(|t| ((t % WIN) as f32, (t / WIN) as f32)).collect();
        let scale = WIN as f32 / GRID as f32;
        let glob_pos: Vec<(f32, f32)> = (0..NTOK)
            .map(|t| {
                let (y, x) = win_yx(t);
                (x as f32 * scale, y as f32 * scale)
            })
            .collect();
        let rope_win = Tensor::from_f32(&dev, &axial_rope(&win_pos), &[WIN * WIN, 32, 2])?;
        let rope_glob = Tensor::from_f32(&dev, &axial_rope(&glob_pos), &[NTOK, 32, 2])?;
        Ok(Vit { dev, patch, pos, ln_pre, blocks, rope_win, rope_glob })
    }

    /// img: f32 [3, 1008, 1008], normalized to [-1, 1]. Returns the trunk output bf16 [72*72, 1024] in raster order
    /// (and f32 in window order when `keep_f32`).
    pub fn forward(&self, img: &Tensor) -> Result<Tensor> {
        let dev = &*self.dev;
        let patches = Tensor::new(dev, DType::BF16, &[NTOK, KP])?;
        dev.launch_n("k_vit_patchify", NTOK * KP, &[Arg::Ptr(img.ptr), Arg::Ptr(patches.ptr), Arg::I32(IMG as i32), Arg::I32(KP as i32)])?;
        let x0 = ops::linear_new(dev, &patches, &self.patch, Act::None, DType::F32)?;
        drop(patches);
        dev.launch_n("k_vit_add_pos", NTOK * DIM, &[Arg::Ptr(x0.ptr), Arg::Ptr(self.pos.ptr), Arg::I32(DIM as i32)])?;
        let x = Tensor::new(dev, DType::F32, &[NTOK, DIM])?;
        ops::layernorm(dev, &x0, Some(&self.ln_pre), 1e-5, &x, None)?;
        drop(x0);
        let h = Tensor::new(dev, DType::BF16, &[NTOK, DIM])?;
        let qkv = Tensor::new(dev, DType::BF16, &[NTOK, 3 * DIM])?;
        let att = Tensor::new(dev, DType::BF16, &[NTOK, DIM])?;
        let mlp = Tensor::new(dev, DType::BF16, &[NTOK, self.blocks[0].fc1.n])?;
        for b in &self.blocks {
            ops::layernorm(dev, &x, Some(&b.norm1), 1e-5, &h, None)?;
            ops::linear(dev, &h, &b.qkv, Act::None, Epi::Store, &qkv)?;
            let (rope, period) = if b.window { (&self.rope_win, WIN * WIN) } else { (&self.rope_glob, NTOK) };
            dev.launch_n(
                "k_rope_qkv",
                NTOK * 2 * HEADS * (HD / 2),
                &[Arg::Ptr(qkv.ptr), Arg::I32(NTOK as i32), Arg::I32(HEADS as i32), Arg::I32(HD as i32), Arg::Ptr(rope.ptr), Arg::I32(period as i32)],
            )?;
            let (len, batch) = if b.window { (WIN * WIN, 9) } else { (NTOK, 1) };
            let mut q = AttnView::new(&qkv, 0, 3 * DIM, HD, len);
            let mut k = AttnView::new(&qkv, DIM, 3 * DIM, HD, len);
            let mut v = AttnView::new(&qkv, 2 * DIM, 3 * DIM, HD, len);
            let mut o = AttnView::new(&att, 0, DIM, HD, len);
            q.batch_stride = len * 3 * DIM;
            k.batch_stride = len * 3 * DIM;
            v.batch_stride = len * 3 * DIM;
            o.batch_stride = len * DIM;
            ops::attention(dev, &AttnArgs { q, k, v, out: o, heads: HEADS, d: HD, batch, causal: false, kmask: None, bias: None })?;
            ops::linear(dev, &att, &b.proj, Act::None, Epi::AddRes(&x), &x)?;
            ops::layernorm(dev, &x, Some(&b.norm2), 1e-5, &h, None)?;
            ops::linear(dev, &h, &b.fc1, Act::GeluErf, Epi::Store, &mlp)?;
            ops::linear(dev, &mlp, &b.fc2, Act::None, Epi::AddRes(&x), &x)?;
        }
        let out = Tensor::new(dev, DType::BF16, &[NTOK, DIM])?;
        dev.launch_n("k_vit_unwindow", NTOK * DIM, &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I32(DIM as i32)])?;
        Ok(out)
    }
}

/// One SimpleFPN head of the TriHead neck: scale 4 (two transposed convs with GELU), 2 (one), or 1, then a 1x1 conv
/// and a 3x3 conv, all to 256 channels.
pub struct FpnHead {
    pub scale: usize,
    dconv0: Option<Linear>,
    dconv1: Option<Linear>,
    conv1: Linear,
    conv3: crate::weights::Conv,
}

impl FpnHead {
    pub fn load(w: &Weights, prefix: &str, scale: usize) -> Result<FpnHead> {
        let (dconv0, dconv1) = match scale {
            4 => (Some(w.conv_t2(&format!("{prefix}.dconv_2x2_0"))?), Some(w.conv_t2(&format!("{prefix}.dconv_2x2_1"))?)),
            2 => (Some(w.conv_t2(&format!("{prefix}.dconv_2x2"))?), None),
            _ => (None, None),
        };
        let s = w.shape(&format!("{prefix}.conv_1x1.weight"))?;
        let c1 = w.host(&format!("{prefix}.conv_1x1.weight"))?;
        let b1 = w.host(&format!("{prefix}.conv_1x1.bias"))?;
        let conv1 = w.linear_from_host(&c1, Some(&b1), s[0], s[1])?;
        let conv3 = w.conv(&format!("{prefix}.conv_3x3"))?;
        Ok(FpnHead { scale, dconv0, dconv1, conv1, conv3 })
    }
    /// trunk bf16 [72*72, 1024] raster -> bf16 [(72*scale)^2, 256]
    pub fn forward(&self, dev: &Device, trunk: &Tensor) -> Result<Tensor> {
        let mut x = trunk.clone();
        let mut side = GRID;
        if let Some(d0) = &self.dconv0 {
            x = ops::conv_t2(dev, &x, side, side, d0, if self.dconv1.is_some() { Act::GeluErf } else { Act::None })?;
            side *= 2;
        }
        if let Some(d1) = &self.dconv1 {
            x = ops::conv_t2(dev, &x, side, side, d1, Act::None)?;
            side *= 2;
        }
        let y = ops::linear_new(dev, &x, &self.conv1, Act::None, DType::BF16)?;
        let out = Tensor::new(dev, DType::BF16, &[side * side, 256])?;
        ops::conv2d(dev, &y, side, side, &self.conv3, 1, 1, &out)?;
        Ok(out)
    }
}

/// The three FPN levels (288, 144, 72) of one neck.
pub struct Neck {
    pub heads: Vec<FpnHead>,
}
impl Neck {
    pub fn load(w: &Weights, prefix: &str) -> Result<Neck> {
        Ok(Neck { heads: vec![FpnHead::load(w, &format!("{prefix}.0"), 4)?, FpnHead::load(w, &format!("{prefix}.1"), 2)?, FpnHead::load(w, &format!("{prefix}.2"), 1)?] })
    }
    pub fn forward(&self, dev: &Device, trunk: &Tensor) -> Result<Vec<Tensor>> {
        self.heads.iter().map(|h| h.forward(dev, trunk)).collect()
    }
}

/// DETR-style normalized sine position encoding of a HxW grid (official PositionEmbeddingSine(256)):
/// [H*W, 256] = (pos_y[128], pos_x[128]), y_embed = (i+1)/(H+eps)*2pi.
pub fn sine_pos_2d(h: usize, w: usize) -> Vec<f32> {
    let npf = 128;
    let scale = 2.0 * std::f64::consts::PI;
    let dim_t: Vec<f64> = (0..npf).map(|i| 10000f64.powf((2 * (i / 2)) as f64 / npf as f64)).collect();
    let mut out = vec![0f32; h * w * 256];
    for y in 0..h {
        for x in 0..w {
            let ye = (y as f64 + 1.0) / (h as f64 + 1e-6) * scale;
            let xe = (x as f64 + 1.0) / (w as f64 + 1e-6) * scale;
            let o = &mut out[(y * w + x) * 256..(y * w + x + 1) * 256];
            for i in 0..npf {
                let (py, px) = (ye / dim_t[i], xe / dim_t[i]);
                o[i] = if i % 2 == 0 { py.sin() } else { py.cos() } as f32;
                o[npf + i] = if i % 2 == 0 { px.sin() } else { px.cos() } as f32;
            }
        }
    }
    out
}
