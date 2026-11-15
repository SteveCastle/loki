//! MiniMax-H3-Image VAE (comfy `MiniMaxH3ImageVAE`): the H3 video VAE's 3D causal CNN encoder run on one frame, and a
//! single-frame 2D ViT decoder (24 layers, 1024 wide, 16 heads x 64, 2D rope over the first 32 head dims, 16x16 pixels
//! per latent token).
//!
//! Numerics follow ComfyUI's fp16 path like `vae_video`: fp16 weights and activations with f32 accumulation, the
//! decoder's residual stream in f32. Both directions tile like the reference (256 px tiles, >= 64 px overlaps, linear
//! blends). `encode` returns the reference's posterior sample (mean + std * N(0, 1) drawn like torch on the CPU with
//! seed 42), normalized by the H3 latent statistics, which is what the DiT is conditioned on.
use crate::cuda::{Arg, Device};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::vae_video::{rh, split_tiles, tile_cfg, VideoVae, CONV_SMEM, INV_FREQ};
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

const DIM: usize = 1024;
const HEADS: usize = 16;
const FFN: usize = 4096;
const RATIO: usize = 16;
const PROJ: usize = 3 * RATIO * RATIO; // 768
/// rope pairs: 2 axes x 8 frequencies
const PAIRS: usize = 16;
/// the reference's posterior-sample seed (comfy `IMAGE_ENCODE_SEED`)
pub const ENCODE_SEED: u64 = 42;

struct Block {
    norm1: Tensor,
    qkv_w: Tensor,
    qkv_b: Tensor,
    out_w: Tensor,
    out_b: Tensor,
    scale1: Tensor,
    norm2: Tensor,
    w1: Tensor, // interleaved gate/up rows
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    scale2: Tensor,
}

pub struct ImageVae {
    dev: Arc<Device>,
    /// the encoder (and the latent / pixel statistics)
    enc: VideoVae,
    pq_w: Tensor,
    pq_b: Tensor,
    xe_w: Tensor,
    xe_b: Tensor,
    regs: Tensor,
    blocks: Vec<Block>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    /// max spatial tiles per decoder call (a 256 px tile is 261 tokens)
    pub max_batch: usize,
    /// fp16 accumulation inside each 32-wide K slab (as the video decoder)
    pub hacc: bool,
}

fn f16_dev(st: &SafeTensors, dev: &Device, name: &str) -> Result<Tensor> {
    let info = st.info(name)?;
    ensure!(info.dtype == "F16", "{name}: expected F16, got {}", info.dtype);
    let t = Tensor::new(dev, DType::BF16, &info.shape)?;
    dev.htod_at(t.ptr, st.bytes(name)?)?;
    Ok(t)
}
fn f32_dev(st: &SafeTensors, dev: &Device, name: &str) -> Result<Tensor> {
    let v = st.f32s(name)?;
    let n = v.len();
    Tensor::from_f32(dev, &v, &[n])
}

/// Whether a VAE file is the MiniMax-H3-Image one (its 1024-wide single-frame ViT decoder).
pub fn is_image_vae(st: &SafeTensors) -> bool {
    st.info("decoder.x_embedder.weight").map(|i| i.shape.first() == Some(&DIM)).unwrap_or(false)
        && st.info("decoder.proj_out.weight").map(|i| i.shape.first() == Some(&PROJ)).unwrap_or(false)
}

impl ImageVae {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<ImageVae> {
        let t0 = std::time::Instant::now();
        let st = SafeTensors::open(path)?;
        ensure!(is_image_vae(&st), "{} is not the MiniMax-H3-Image VAE (no 1024-wide single-frame decoder)", path.display());
        let enc = VideoVae::load_opts(dev.clone(), path, false).context("image VAE encoder")?;
        for k in ["k_vv_gemm", "k_vv_gemm_h", "k_vv_gemm_s", "k_vv_gemm_h_s"] {
            dev.set_max_smem(k, CONV_SMEM)?;
        }
        let layers = (0..).take_while(|i| st.has(&format!("decoder.transformer_blocks.{i}.scale1"))).count();
        ensure!(layers > 0, "image VAE without decoder blocks");
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = format!("decoder.transformer_blocks.{i}");
            let w1n = format!("{p}.ff.w1.weight");
            let info = st.info(&w1n)?;
            ensure!(info.dtype == "F16" && info.shape == vec![2 * FFN, DIM], "{w1n}: expected F16 [{}, {DIM}]", 2 * FFN);
            // interleave w1 rows: [gate(0..F) | up(F..2F)] -> (gate_j, up_j) for the fused SwiGLU epilogue
            let w1: Vec<u16> = st.bytes(&w1n)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            let mut w1i = vec![0u16; w1.len()];
            for r in 0..2 * FFN {
                let src = if r % 2 == 0 { r / 2 } else { FFN + r / 2 };
                w1i[r * DIM..(r + 1) * DIM].copy_from_slice(&w1[src * DIM..(src + 1) * DIM]);
            }
            let w1t = Tensor::new(&dev, DType::BF16, &[2 * FFN, DIM])?;
            dev.htod(&w1t.buf, &w1i)?;
            let b1 = st.f32s(&format!("{p}.ff.w1.bias"))?;
            let b1i: Vec<f32> = (0..2 * FFN).map(|r| if r % 2 == 0 { b1[r / 2] } else { b1[FFN + r / 2] }).collect();
            blocks.push(Block {
                norm1: f16_dev(&st, &dev, &format!("{p}.norm1.weight"))?,
                qkv_w: f16_dev(&st, &dev, &format!("{p}.attn.to_qkv.weight"))?,
                qkv_b: f32_dev(&st, &dev, &format!("{p}.attn.to_qkv.bias"))?,
                out_w: f16_dev(&st, &dev, &format!("{p}.attn.to_out.weight"))?,
                out_b: f32_dev(&st, &dev, &format!("{p}.attn.to_out.bias"))?,
                scale1: f32_dev(&st, &dev, &format!("{p}.scale1"))?,
                norm2: f16_dev(&st, &dev, &format!("{p}.norm2.weight"))?,
                w1: w1t,
                b1: Tensor::from_f32(&dev, &b1i, &[2 * FFN])?,
                w2: f16_dev(&st, &dev, &format!("{p}.ff.w2.weight"))?,
                b2: f32_dev(&st, &dev, &format!("{p}.ff.w2.bias"))?,
                scale2: f32_dev(&st, &dev, &format!("{p}.scale2"))?,
            });
        }
        let vae = ImageVae {
            pq_w: f32_dev(&st, &dev, "post_quant_conv.weight")?,
            pq_b: f32_dev(&st, &dev, "post_quant_conv.bias")?,
            xe_w: f32_dev(&st, &dev, "decoder.x_embedder.weight")?,
            xe_b: f32_dev(&st, &dev, "decoder.x_embedder.bias")?,
            regs: f16_dev(&st, &dev, "decoder.register_tokens")?,
            blocks,
            norm_out_w: f16_dev(&st, &dev, "decoder.norm_out.weight")?,
            norm_out_b: f16_dev(&st, &dev, "decoder.norm_out.bias")?,
            proj_w: f16_dev(&st, &dev, "decoder.proj_out.weight")?,
            proj_b: f32_dev(&st, &dev, "decoder.proj_out.bias")?,
            max_batch: 32,
            hacc: std::env::var("LOKI_VVAE_HACC").map(|v| v.contains('d')).unwrap_or(true),
            enc,
            dev,
        };
        crate::info!("  image vae: {layers} decoder layers loaded in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(vae)
    }

    // ------------------------------------------------------------------ encoder

    /// Raw tiled moments f32 [48, h/16, w/16] (mean | logvar, fp16-rounded) of an RGB8 image [h, w, 3].
    pub fn moments(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<f32>> {
        ensure!(rgb.len() == h * w * 3, "encode: image buffer size");
        ensure!(h % 16 == 0 && w % 16 == 0, "encode: image size must be a multiple of 16, got {w}x{h}");
        let pix = Tensor::from_buf(self.dev.upload(rgb)?, DType::U8, &[rgb.len()]);
        let (m, tl) = self.enc.encode_clip_n(&pix, 1, h, w, 0, 1, 48)?;
        ensure!(tl == 1, "image encode produced {tl} latent frames");
        Ok(m)
    }

    /// The DiT's view of an image: normalized posterior sample f32 [24, 1, h/16, w/16] (device).
    pub fn encode(&self, rgb: &[u8], h: usize, w: usize) -> Result<Tensor> {
        let m = self.moments(rgb, h, w)?;
        let np = (h / RATIO) * (w / RATIO);
        let noise = crate::rng::torch_randn(ENCODE_SEED, 24 * np);
        let mut z = vec![0f32; 24 * np];
        for c in 0..24 {
            for p in 0..np {
                let mean = m[c * np + p];
                let logvar = m[(24 + c) * np + p].clamp(-30.0, 20.0);
                let s = mean + (0.5 * logvar).exp() * noise[c * np + p];
                z[c * np + p] = (s - self.enc.lat_mean[c]) / self.enc.lat_std[c];
            }
        }
        Tensor::from_f32(&self.dev, &z, &[24, 1, h / RATIO, w / RATIO])
    }

    // ------------------------------------------------------------------ decoder

    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, a: &Tensor, m: usize, k: usize, w: &Tensor, bias: &Tensor, mode: i32, out: &Tensor, res: Option<&Tensor>, gate: Option<&Tensor>) -> Result<()> {
        let n = w.shape[0];
        ensure!(w.numel() == n * k, "iv gemm: K mismatch");
        let (bm, threads, smem, suf) = tile_cfg(m, n, self.dev.sm_count as usize);
        let grid = (((m + bm - 1) / bm) * ((n + 127) / 128)) as u32;
        let kname = format!("{}{}", if self.hacc { "k_vv_gemm_h" } else { "k_vv_gemm" }, suf);
        self.dev.launch(
            &kname,
            (grid, 1, 1),
            (threads, 1, 1),
            smem,
            &[
                Arg::Ptr(a.ptr), Arg::Ptr(w.ptr), Arg::Ptr(out.ptr), Arg::I32(m as i32), Arg::I32(n as i32), Arg::I32(k as i32), Arg::Ptr(bias.ptr), Arg::I32(mode),
                Arg::Ptr(res.map(|t| t.ptr).unwrap_or(0)), Arg::Ptr(gate.map(|t| t.ptr).unwrap_or(0)),
            ],
        )
    }

    /// Rope table f32 [S, 16, 2] (cos, sin) over (y, x) token ids, emulating the reference's fp16 ids and table; the 5
    /// suffix tokens (registers + zero token) sit at id 0 (no rotation).
    fn rope_table(&self, th: usize, tw: usize) -> Result<Tensor> {
        let coords = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let c = rh(i as f32 + 0.5);
                    let c = rh(c / n as f32);
                    let c = rh(2.0 * c);
                    rh(c - 1.0)
                })
                .collect()
        };
        let (cy, cx) = (coords(th), coords(tw));
        let np = th * tw;
        let s = np + 5;
        let mut tab = vec![0f32; s * PAIRS * 2];
        let two_pi = (2.0 * std::f64::consts::PI) as f32;
        for y in 0..th {
            for x in 0..tw {
                let tok = y * tw + x;
                for (a, id) in [cy[y], cx[x]].into_iter().enumerate() {
                    let base = two_pi * id;
                    for k in 0..8 {
                        let ang = base * INV_FREQ[k];
                        let p = a * 8 + k;
                        tab[(tok * PAIRS + p) * 2] = rh(ang.cos());
                        tab[(tok * PAIRS + p) * 2 + 1] = rh(ang.sin());
                    }
                }
            }
        }
        for tok in np..s {
            for p in 0..PAIRS {
                tab[(tok * PAIRS + p) * 2] = 1.0;
            }
        }
        Tensor::from_f32(&self.dev, &tab, &[s, PAIRS * 2])
    }

    /// The ViT on `nb` tiles of `s` tokens each, embedded in x (f32 [nb*s, 1024], modified). Returns proj f32 [nb*s, 768].
    fn vit(&self, x: &Tensor, nb: usize, s: usize, rope: &Tensor) -> Result<Tensor> {
        let dev = &self.dev;
        let m = nb * s;
        let xn = Tensor::new(dev, DType::BF16, &[m, DIM])?;
        let qkv = Tensor::new(dev, DType::BF16, &[m, 3 * DIM])?;
        let att = Tensor::new(dev, DType::BF16, &[m, DIM])?;
        let hb = Tensor::new(dev, DType::BF16, &[m, FFN])?;
        let eps = 1e-5f32;
        let norm = |src: &Tensor, w: &Tensor, b: Option<&Tensor>| -> Result<()> {
            dev.launch("k_iv_norm_rows", (m as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(src.ptr), Arg::Ptr(xn.ptr), Arg::Ptr(w.ptr), Arg::Ptr(b.map(|t| t.ptr).unwrap_or(0)), Arg::I32(b.is_some() as i32), Arg::F32(eps)])
        };
        let scale_log2 = 0.125f32 * std::f32::consts::LOG2_E;
        for blk in &self.blocks {
            norm(x, &blk.norm1, None)?;
            self.gemm(&xn, m, DIM, &blk.qkv_w, &blk.qkv_b, 0, &qkv, None, None)?;
            dev.launch("k_iv_qk_rope", (((m * HEADS + 7) / 8) as u32, 1, 1), (256, 1, 1), 0, &[Arg::Ptr(qkv.ptr), Arg::I32(m as i32), Arg::I32(s as i32), Arg::Ptr(rope.ptr), Arg::F32(eps)])?;
            dev.launch("k_iv_flash_d64", (((s + 63) / 64) as u32, HEADS as u32, nb as u32), (128, 1, 1), 0, &[Arg::Ptr(qkv.ptr), Arg::Ptr(att.ptr), Arg::I32(s as i32), Arg::F32(scale_log2)])?;
            self.gemm(&att, m, DIM, &blk.out_w, &blk.out_b, 2, x, Some(x), Some(&blk.scale1))?;
            norm(x, &blk.norm2, None)?;
            self.gemm(&xn, m, DIM, &blk.w1, &blk.b1, 3, &hb, None, None)?;
            self.gemm(&hb, m, FFN, &blk.w2, &blk.b2, 2, x, Some(x), Some(&blk.scale2))?;
        }
        drop((qkv, hb, att));
        norm(x, &self.norm_out_w, Some(&self.norm_out_b))?;
        let proj = Tensor::new(dev, DType::F32, &[m, PROJ])?;
        self.gemm(&xn, m, DIM, &self.proj_w, &self.proj_b, 1, &proj, None, None)?;
        Ok(proj)
    }

    /// z: f32 [24, (1,) hl, wl] normalized latent (device) -> RGB8 [hl*16, wl*16, 3] (host). Reference `tiled_decode`.
    pub fn decode(&self, z: &Tensor, hl: usize, wl: usize) -> Result<Vec<u8>> {
        ensure!(z.dtype == DType::F32 && z.numel() == 24 * hl * wl, "decode: z must be f32 [24, {hl}, {wl}], got {:?}", z.shape);
        let dev = &self.dev;
        let (hh, ww) = (hl * RATIO, wl * RATIO);
        let canvas = Tensor::new(dev, DType::F32, &[hh * ww * 3])?;
        let (yi, yl, yo) = split_tiles(hh);
        let (xi, xl, xo) = split_tiles(ww);
        let (th, tw) = (yl[0] / RATIO, xl[0] / RATIO);
        let s = th * tw + 5;
        let rope = self.rope_table(th, tw)?;
        let oy_max = yo.iter().copied().max().unwrap_or(0);
        let strip = if oy_max > 0 { Some(Tensor::new(dev, DType::F32, &[oy_max * ww * 3])?) } else { None };
        let tiles: Vec<(usize, usize)> = (0..yi.len()).flat_map(|i| (0..xi.len()).map(move |j| (i, j))).collect();
        // balanced batches of at most max_batch tiles; tiles are placed in row-major order, as the blends need
        let nbatches = tiles.len().div_ceil(self.max_batch.max(1));
        let nb_max = tiles.len().div_ceil(nbatches);
        for batch in tiles.chunks(nb_max) {
            let nb = batch.len();
            let x = Tensor::new(dev, DType::F32, &[nb * s, DIM])?;
            for (bi, &(i, j)) in batch.iter().enumerate() {
                dev.launch(
                    "k_iv_embed",
                    (s as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &[
                        Arg::Ptr(z.ptr), Arg::I32(hl as i32), Arg::I32(wl as i32), Arg::I32((yi[i] / RATIO) as i32), Arg::I32((xi[j] / RATIO) as i32), Arg::I32(th as i32), Arg::I32(tw as i32),
                        Arg::Ptr(self.enc.lat_mean_d.ptr), Arg::Ptr(self.enc.lat_std_d.ptr), Arg::Ptr(self.pq_w.ptr), Arg::Ptr(self.pq_b.ptr), Arg::Ptr(self.xe_w.ptr), Arg::Ptr(self.xe_b.ptr),
                        Arg::Ptr(self.regs.ptr), Arg::Ptr(x.ptr + (bi * s * DIM * 4) as u64),
                    ],
                )?;
            }
            let proj = self.vit(&x, nb, s, &rope)?;
            drop(x);
            for (bi, &(i, j)) in batch.iter().enumerate() {
                let oy = if i > 0 { yo[i - 1] } else { 0 };
                let ox = if j > 0 { xo[j - 1] } else { 0 };
                dev.launch_n(
                    "k_iv_place_tile",
                    yl[i] * xl[j],
                    &[
                        Arg::Ptr(proj.ptr + (bi * s * PROJ * 4) as u64), Arg::I32(th as i32), Arg::I32(tw as i32), Arg::Ptr(canvas.ptr), Arg::I32(hh as i32), Arg::I32(ww as i32),
                        Arg::I32(yi[i] as i32), Arg::I32(xi[j] as i32), Arg::Ptr(strip.as_ref().map(|t| t.ptr).unwrap_or(0)), Arg::I32(oy_max as i32), Arg::I32(oy as i32), Arg::I32(ox as i32),
                    ],
                )?;
                if j + 1 == xi.len() && i + 1 < yi.len() {
                    let st = strip.as_ref().unwrap();
                    dev.launch_n(
                        "k_vv_copy_strip",
                        yo[i] * ww * 3,
                        &[Arg::Ptr(canvas.ptr), Arg::I32(1), Arg::I32(hh as i32), Arg::I32(ww as i32), Arg::I32(yi[i + 1] as i32), Arg::I32(yo[i] as i32), Arg::Ptr(st.ptr), Arg::I32(oy_max as i32)],
                    )?;
                }
            }
        }
        let hw = hh * ww;
        let out = Tensor::new(dev, DType::U8, &[hw * 3])?;
        dev.launch_n(
            "k_vv_finalize",
            hw * 3,
            &[
                Arg::Ptr(canvas.ptr), Arg::I32(0), Arg::I32(1), Arg::I64(hw as i64), Arg::Ptr(0), Arg::I32(0), Arg::I32(0),
                Arg::Ptr(self.enc.pix_mean_d.ptr), Arg::Ptr(self.enc.pix_std_d.ptr), Arg::Ptr(out.ptr),
            ],
        )?;
        let host: Vec<u8> = dev.dtoh_at(out.ptr, hw * 3)?;
        Ok(host)
    }

    /// Decode to f32 pixels in [0, 1] (debug / validation): the canvas before u8 rounding.
    pub fn decode_f32(&self, z: &Tensor, hl: usize, wl: usize) -> Result<Vec<f32>> {
        Ok(self.decode(z, hl, wl)?.into_iter().map(|v| v as f32 / 255.0).collect())
    }
}
