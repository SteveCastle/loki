//! Video Sparse Attention (VSA) for FastH3, the way ComfyUI's BlockSparseAttention node runs it in `vsa` mode
//! (`comfy_extras/nodes_sparse_attention.py` on `comfy_kitchen.sol_attn`): see `kernels/vsa.cu` for the algorithm.
//!
//! The run keeps its packed sequence in the tile order for the whole forward (every per-token op is order-free), so only
//! the embedding, the final layer and the attention know about it: each non-video segment is zero-padded to whole
//! 64-row tiles, the target video is cut into 4x4x4 latent-patch cubes (live rows first, pad rows last).
use crate::cuda::{Arg, Device};
use crate::tensor::{DType, Tensor};
use anyhow::{ensure, Result};
use cudarc::driver::PushKernelArg;

pub const TILE: usize = 64;
const CUBE: usize = 4;
/// Fraction of the non-sink key tiles each query tile keeps (FastH3 is trained at 10%).
pub const KEEP: f64 = 0.10;
/// Shorter sequences run dense (ComfyUI's default).
pub const MIN_TOKENS: usize = 12288;

/// Tiling of a packed sequence: `src[p]` = original row of padded row `p` (-1 = pad), `inv` the reverse.
#[derive(Debug, Clone)]
pub struct Plan {
    pub src: Vec<i32>,
    pub inv: Vec<i32>,
    /// live rows per tile
    pub block_len: Vec<i32>,
    /// tiles before the video cubes (all non-video segments): the always-exact sinks
    pub sink_tiles: usize,
}

impl Plan {
    /// `segments`: row counts of the non-video segments in sequence order (text, references, target audio), followed
    /// by the target video of `lt x h2 x w2` latent patches in (t, h, w) raster order.
    pub fn new(segments: &[usize], lt: usize, h2: usize, w2: usize) -> Plan {
        let mut src: Vec<i32> = Vec::new();
        let mut row = 0i32;
        for &n in segments {
            let tiles = n.div_ceil(TILE);
            for i in 0..tiles * TILE {
                src.push(if i < n { row + i as i32 } else { -1 });
            }
            row += n as i32;
        }
        let sink_tiles = src.len() / TILE;
        let (pt, ph, pw) = (lt.div_ceil(CUBE) * CUBE, h2.div_ceil(CUBE) * CUBE, w2.div_ceil(CUBE) * CUBE);
        for ct in (0..pt).step_by(CUBE) {
            for ch in (0..ph).step_by(CUBE) {
                for cw in (0..pw).step_by(CUBE) {
                    let mut live = Vec::with_capacity(TILE);
                    for t in ct..ct + CUBE {
                        for h in ch..ch + CUBE {
                            for w in cw..cw + CUBE {
                                if t < lt && h < h2 && w < w2 {
                                    live.push(row + ((t * h2 + h) * w2 + w) as i32);
                                }
                            }
                        }
                    }
                    let n = live.len();
                    src.extend(live);
                    src.extend(std::iter::repeat(-1).take(TILE - n));
                }
            }
        }
        let total = row as usize + lt * h2 * w2;
        let mut inv = vec![-1i32; total];
        for (p, &s) in src.iter().enumerate() {
            if s >= 0 {
                inv[s as usize] = p as i32;
            }
        }
        let block_len = src.chunks(TILE).map(|c| c.iter().filter(|&&s| s >= 0).count() as i32).collect();
        Plan { src, inv, block_len, sink_tiles }
    }
    pub fn tiles(&self) -> usize {
        self.block_len.len()
    }
}

/// Key tiles a query tile keeps by score among `n` non-sink tiles: round(ratio * n) clamped to [1, n-1] (0 when n <= 1).
pub fn topk_count(n: usize, ratio: f64) -> usize {
    if n <= 1 {
        return 0;
    }
    ((ratio * n as f64).round_ties_even() as usize).clamp(1, n - 1)
}

/// Per-run VSA state: the tiling tables on the device and the per-batch scratch.
pub struct VsaRun {
    pub tiles: usize,
    pub sink_tiles: usize,
    pub kk: usize,
    /// live rows per tile (i32)
    pub blen: Tensor,
    /// target video row -> padded row (i32)
    pub vmap: Tensor,
    /// pad rows (i32), kept zero in x
    pub dead: Tensor,
    pub n_dead: usize,
    /// sigma at or below which a step runs sparse (ComfyUI's start_percent; default every step)
    pub max_sigma: f32,
    gate: Tensor,  // bf16 [abatch, INNER]
    kmean: Tensor, // f32 [tiles, H, 128]
    vmean: Tensor, // f32 [tiles, H, 128]
    qmean: Tensor, // f32 [abatch/64, H, 128]
    oc: Tensor,    // f32 [abatch/64, H, 128]
    bits: Tensor,  // u32 [abatch/64, H, W]
    list: Tensor,  // i32 [abatch/256, H, tiles]
    cnt: Tensor,   // i32 [abatch/256, H]
}

fn i32_tensor(dev: &Device, v: &[i32]) -> Result<Tensor> {
    Ok(Tensor::from_buf(dev.upload(v)?, DType::F32, &[v.len().max(1)])) // i32 payload
}

impl VsaRun {
    /// Device bytes `new` allocates for `tiles` tiles and an attention batch of `abatch` rows.
    pub fn bytes(tiles: usize, abatch: usize, heads: usize) -> usize {
        let qt = abatch / TILE;
        abatch * heads * 128 * 2 + 2 * tiles * heads * 128 * 4 + 2 * qt * heads * 128 * 4 + qt * heads * tiles.div_ceil(32) * 4 + (abatch / 256) * heads * (tiles + 1) * 4
    }

    pub fn new(dev: &Device, plan: &Plan, v0: usize, n_video: usize, abatch: usize, heads: usize, max_sigma: f32) -> Result<VsaRun> {
        ensure!(abatch % 256 == 0, "VSA needs an attention batch that is a multiple of 256 rows");
        let tiles = plan.tiles();
        let vmap: Vec<i32> = (0..n_video).map(|i| plan.inv[v0 + i]).collect();
        let dead: Vec<i32> = plan.src.iter().enumerate().filter(|(_, &s)| s < 0).map(|(p, _)| p as i32).collect();
        let qt = abatch / TILE;
        let inner = heads * 128;
        Ok(VsaRun {
            tiles,
            sink_tiles: plan.sink_tiles,
            kk: topk_count(tiles - plan.sink_tiles, KEEP),
            blen: i32_tensor(dev, &plan.block_len)?,
            vmap: i32_tensor(dev, &vmap)?,
            n_dead: dead.len(),
            dead: i32_tensor(dev, &dead)?,
            max_sigma,
            gate: Tensor::new(dev, DType::BF16, &[abatch, inner])?,
            kmean: Tensor::new(dev, DType::F32, &[tiles, inner])?,
            vmean: Tensor::new(dev, DType::F32, &[tiles, inner])?,
            qmean: Tensor::new(dev, DType::F32, &[qt, inner])?,
            oc: Tensor::new(dev, DType::F32, &[qt, inner])?,
            bits: Tensor::new(dev, DType::F32, &[qt * heads * tiles.div_ceil(32)])?,
            list: Tensor::new(dev, DType::F32, &[(abatch / 256) * heads * tiles])?,
            cnt: Tensor::new(dev, DType::F32, &[(abatch / 256) * heads])?,
        })
    }

    /// Gate output rows for the batch row offset `r` (bf16, row stride INNER).
    pub fn gate_rows(&self, r: usize, inner: usize) -> u64 {
        self.gate.ptr + (r * inner * 2) as u64
    }

    /// Zero the pad rows of x.
    pub fn zero_dead(&self, dev: &Device, x: u64, hidden: usize) -> Result<()> {
        if self.n_dead == 0 {
            return Ok(());
        }
        dev.launch_n("k_vsa_zero_rows_bf16", self.n_dead * hidden / 8, &[Arg::Ptr(x), Arg::Ptr(self.dead.ptr), Arg::I32(self.n_dead as i32), Arg::I32(hidden as i32)])
    }

    /// V tile means for the chunk of `n` rows starting at padded row `r0` (a multiple of 64): bf16 rows at `v`, stride `ts`.
    pub fn v_means(&self, dev: &Device, v: u64, ts: usize, r0: usize, n: usize, heads: usize) -> Result<()> {
        let t0 = r0 / TILE;
        let out = self.vmean.ptr + (t0 * heads * 128 * 4) as u64;
        dev.launch(
            "k_vsa_tile_mean_bf16",
            (n.div_ceil(TILE) as u32, heads as u32, 1),
            (128, 1, 1),
            0,
            &[Arg::Ptr(v), Arg::I64(ts as i64), Arg::I32(n as i32), Arg::I32(heads as i32), Arg::Ptr(self.blen.ptr), Arg::I32(t0 as i32), Arg::Ptr(out)],
        )
    }

    /// K tile means over the whole sequence from the int8 keys (rotated and mean-shifted: the scores they give differ
    /// from the true ones by a per-query constant, which neither the top-k nor the softmax sees).
    pub fn k_means(&self, dev: &Device, k8: u64, sk: u64, seq: usize, heads: usize) -> Result<()> {
        dev.launch(
            "k_vsa_tile_mean_i8",
            (self.tiles as u32, heads as u32, 1),
            (128, 1, 1),
            0,
            &[Arg::Ptr(k8), Arg::Ptr(sk), Arg::I32(seq as i32), Arg::I32(heads as i32), Arg::Ptr(self.blen.ptr), Arg::I32(0), Arg::Ptr(self.kmean.ptr)],
        )
    }

    /// Selection + coarse branch for the attention batch of `bn` query rows at padded row `b0` (q8/sq batch-local), then
    /// the per-256-row tile lists for the sparse attention kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn select(&self, dev: &Device, q8: u64, sq: u64, b0: usize, bn: usize, heads: usize, scale: f32) -> Result<()> {
        let ntq = bn.div_ceil(TILE);
        let t0 = b0 / TILE;
        dev.launch(
            "k_vsa_tile_mean_i8",
            (ntq as u32, heads as u32, 1),
            (128, 1, 1),
            0,
            &[Arg::Ptr(q8), Arg::Ptr(sq), Arg::I32(bn as i32), Arg::I32(heads as i32), Arg::Ptr(self.blen.ptr), Arg::I32(t0 as i32), Arg::Ptr(self.qmean.ptr)],
        )?;
        // query tiles per block: up to 8, as many as fit 88 KB of scores (plus 8 KB static: the 99 KB per-block limit)
        let qg = ((88 * 1024) / (self.tiles * 4)).clamp(1, 8);
        dev.launch(
            "k_vsa_select",
            (ntq.div_ceil(qg) as u32, heads as u32, 1),
            (256, 1, 1),
            (qg * self.tiles * 4) as u32,
            &[
                Arg::Ptr(self.qmean.ptr),
                Arg::Ptr(self.kmean.ptr),
                Arg::Ptr(self.vmean.ptr),
                Arg::I32(self.tiles as i32),
                Arg::I32(heads as i32),
                Arg::I32(self.sink_tiles as i32),
                Arg::I32(t0 as i32),
                Arg::I32(self.kk as i32),
                Arg::F32(scale),
                Arg::I32(ntq as i32),
                Arg::I32(qg as i32),
                Arg::Ptr(self.bits.ptr),
                Arg::Ptr(self.oc.ptr),
            ],
        )?;
        dev.launch(
            "k_vsa_lists",
            (bn.div_ceil(256) as u32, heads as u32, 1),
            (32, 1, 1),
            0,
            &[Arg::Ptr(self.bits.ptr), Arg::I32(ntq as i32), Arg::I32(heads as i32), Arg::I32(self.tiles.div_ceil(32) as i32), Arg::I32(self.tiles as i32), Arg::Ptr(self.list.ptr), Arg::Ptr(self.cnt.ptr)],
        )
    }

    /// Sparse fine attention for the batch (same operands as the dense kernel, plus this batch's tile lists).
    #[allow(clippy::too_many_arguments)]
    pub fn attention<P: cudarc::driver::DeviceRepr>(&self, dev: &Device, params: &P, nq: usize, heads: usize, smem: u32) -> Result<()> {
        let f = dev.func("k_h3_attn_v6_sparse")?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (nq.div_ceil(256) as u32, heads as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
        let (list, cnt, blen, cap) = (self.list.ptr, self.cnt.ptr, self.blen.ptr, self.tiles as i32);
        let mut b = dev.stream.launch_builder(&f);
        b.arg(params);
        b.arg(&list);
        b.arg(&cnt);
        b.arg(&blen);
        b.arg(&cap);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// The last `select`'s tile lists, as (entries [blocks][H][tiles], counts [blocks][H]) (for tests).
    pub fn lists(&self, dev: &Device) -> Result<(Vec<i32>, Vec<i32>)> {
        Ok((dev.dtoh_at::<i32>(self.list.ptr, self.list.numel())?, dev.dtoh_at::<i32>(self.cnt.ptr, self.cnt.numel())?))
    }

    /// attn_out[r] += gate[r] * coarse[r / 64] for the batch's `bn` rows.
    pub fn add_coarse(&self, dev: &Device, attn_out: u64, bn: usize, inner: usize) -> Result<()> {
        dev.launch_n("k_vsa_add_coarse", bn * inner / 2, &[Arg::Ptr(attn_out), Arg::Ptr(self.gate.ptr), Arg::Ptr(self.oc.ptr), Arg::I32(bn as i32), Arg::I32(inner as i32)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_pads_segments_and_cubes() {
        // text 70 rows, audio 10 rows, video 5 x 2 x 6 patches
        let p = Plan::new(&[70, 10], 5, 2, 6);
        assert_eq!(p.sink_tiles, 2 + 1);
        // cubes: ceil(5/4)=2 x ceil(2/4)=1 x ceil(6/4)=2
        assert_eq!(p.tiles(), 3 + 4);
        assert_eq!(&p.block_len[..3], &[64, 6, 10]);
        // first cube: t 0..4, h 0..2, w 0..4 -> 32 live rows; the w 4..6 cube: 4*2*2 = 16; t=4 cubes: 1*2*4 = 8 and 4
        assert_eq!(&p.block_len[3..], &[32, 16, 8, 4]);
        // every original row appears exactly once
        let n = 80 + 60;
        let mut seen = vec![0; n];
        for &s in &p.src {
            if s >= 0 {
                seen[s as usize] += 1;
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
        for (r, &q) in p.inv.iter().enumerate() {
            assert_eq!(p.src[q as usize], r as i32);
        }
        // live rows first inside a cube, in (t, h, w) raster order
        let cube0 = &p.src[3 * 64..4 * 64];
        assert_eq!(&cube0[..5], &[80, 81, 82, 83, 86]);
        assert!(cube0[32..].iter().all(|&s| s == -1));
    }

    #[test]
    fn topk_matches_comfy() {
        assert_eq!(topk_count(0, 0.1), 0);
        assert_eq!(topk_count(1, 0.1), 0);
        assert_eq!(topk_count(5, 0.1), 1);
        assert_eq!(topk_count(25, 0.1), 2); // 2.5 rounds to even
        assert_eq!(topk_count(35, 0.1), 4); // 3.5 rounds to even
        assert_eq!(topk_count(449, 0.1), 45);
    }
}
