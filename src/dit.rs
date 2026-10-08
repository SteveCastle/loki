//! Qwen Image 2.1 transformer (32 single-stream blocks, dim 4096, int8 ConvRot linears, bf16 activations)
//! with the step-independent text/reference prefix cached as per-block K/V after the first step.
use crate::cuda::{Arg, Device};
use crate::ops::{self, Act, AttnArgs, AttnView, Epi, QuantAct, QuantKv};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, bf16_to_f32, DType, Tensor};
use crate::weights::{Loader, QLinear};
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const DIM: usize = 4096;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 128;
pub const LAYERS: usize = 32;
pub const MLP: usize = 12288;
pub const EPS: f32 = 1e-6;
pub const AXES: [usize; 3] = [16, 56, 56];

struct Block {
    qkv: QLinear,
    norm_q: Tensor,
    norm_k: Tensor,
    out: QLinear,
    gate_up: QLinear,
    mlp_out: QLinear,
}

pub struct Dit {
    dev: Arc<Device>,
    pub prof: crate::cuda::Profiler,
    /// false: bf16 flash attention with a bf16 cache; true: int8/fp8 attention with a quantized cache
    pub sage: bool,
    img_in: Tensor,
    txt_norm: Tensor, // (w + 1) as bf16
    txt_in1: Tensor,
    txt_in2: Tensor,
    t_lin1: Tensor,
    t_lin2: Tensor,
    modulation: Tensor,
    blocks: Vec<Block>,
    norm_out_lin: Tensor,
    proj_out: Tensor,
}

/// One reference image for `prepare`: its normalized latent [rh*rw, 64] bf16 token-major, spliced into
/// the text context at `slot` (ComfyUI's image_slots; ascending).
pub struct RefLatent {
    pub slot: usize,
    pub latent: Tensor,
    pub rh: usize,
    pub rw: usize,
}

/// Prepared, step-independent state for one sampling run.
pub struct Run {
    /// prefix hidden states before the blocks [P, 4096] bf16 (text with reference tokens spliced in)
    prefix_x: Tensor,
    /// rope tables [P, 64, 2] and [Nt, 64, 2]
    rope_prefix: Tensor,
    rope_target: Tensor,
    kv_limit: Tensor,
    /// per block: K and V for the prefix [P, 4096] bf16 each (bf16 mode)
    cache_k: Vec<Tensor>,
    cache_v: Vec<Tensor>,
    /// per block: quantized prefix K/V (sage mode)
    cache_q: Vec<QuantKv>,
    cached: bool,
    /// modulation for t = 0 (prefix rows): scale1, gate1, scale2, gate2 as f32 [4096]
    mod0: [Tensor; 4],
    temb0: Tensor,
    pub p: usize,
    pub h: usize,
    pub w: usize,
}

struct Modulation {
    scale1: Tensor,
    gate1: Tensor,
    scale2: Tensor,
    gate2: Tensor,
    out_scale: Tensor,
}

impl Dit {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Dit> {
        let st = SafeTensors::open(path)?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let img_in = l.bf16("img_in.weight")?;
        let txt_norm = {
            let v = st.f32s("txt_in.text_norm.weight")?;
            let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x + 1.0)).collect();
            Tensor::from_bf16(&dev, &bits, &[DIM])?
        };
        let txt_in1 = l.bf16("txt_in.in_layer.weight")?;
        let txt_in2 = l.bf16("txt_in.out_layer.weight")?;
        let t_lin1 = l.bf16("time_text_embed.timestep_embedder.linear_1.weight")?;
        let t_lin2 = l.bf16("time_text_embed.timestep_embedder.linear_2.weight")?;
        let modulation = l.bf16("modulation.1.weight")?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let p = format!("transformer_blocks.{i}");
            blocks.push(Block {
                qkv: l.qlinear_cat(&[&format!("{p}.attn.to_q"), &format!("{p}.attn.to_k"), &format!("{p}.attn.to_v")])?,
                norm_q: l.bf16(&format!("{p}.attn.norm_q.weight"))?,
                norm_k: l.bf16(&format!("{p}.attn.norm_k.weight"))?,
                out: l.qlinear(&format!("{p}.attn.to_out.0"))?,
                gate_up: l.qlinear(&format!("{p}.img_mlp.gate_up"))?.interleave_gate_up(&dev)?,
                mlp_out: l.qlinear(&format!("{p}.img_mlp.out"))?,
            });
        }
        let norm_out_lin = l.bf16("norm_out.linear.weight")?;
        let proj_out = l.bf16("proj_out.weight")?;
        crate::info!("  dit: {:.1} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        let sage = std::env::var("LOKI_ATTN").map(|v| v != "bf16").unwrap_or(true);
        ops::sage_init(&dev)?;
        Ok(Dit { dev, prof: crate::cuda::Profiler::new(), sage, img_in, txt_norm, txt_in1, txt_in2, t_lin1, t_lin2, modulation, blocks, norm_out_lin, proj_out })
    }

    /// temb for a timestep: ComfyUI rounds t*1000 and t to bf16, builds a 256-d sinusoid, then
    /// linear_1 -> SiLU -> linear_2 (bf16). Returns [1, 4096] bf16.
    fn temb(&self, sigma: f32) -> Result<Tensor> {
        let dev = &self.dev;
        let t1000 = bf16_to_f32(bf16_bits(sigma * 1000.0));
        let t = bf16_to_f32(bf16_bits(t1000 / 1000.0));
        let tt = t * 1000.0;
        let mut emb = vec![0u16; 256];
        for i in 0..128 {
            let freq = (-(10000f32.ln()) * i as f32 / 128.0).exp();
            let a = tt * freq;
            emb[i] = bf16_bits(a.cos());
            emb[128 + i] = bf16_bits(a.sin());
        }
        let e = Tensor::from_bf16(dev, &emb, &[1, 256])?;
        let h1 = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        ops::gemm_bf16(dev, &e, &self.t_lin1, None, Act::Silu, Epi::Store, &h1)?;
        let out = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        ops::gemm_bf16(dev, &h1, &self.t_lin2, None, Act::None, Epi::Store, &out)?;
        Ok(out)
    }

    /// modulation(SiLU(temb)) -> scale1, tanh(gate1), scale2, tanh(gate2) (f32 [4096] each) and the
    /// norm_out scale.
    fn modulation(&self, temb: &Tensor) -> Result<Modulation> {
        let dev = &self.dev;
        let s = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        dev.dtod(s.ptr, temb.ptr, temb.bytes())?;
        ops::act_inplace(dev, &s, Act::Silu)?;
        let m = Tensor::new(dev, DType::BF16, &[1, 4 * DIM])?;
        ops::gemm_bf16(dev, &s, &self.modulation, None, Act::None, Epi::Store, &m)?;
        let o = Tensor::new(dev, DType::BF16, &[1, DIM])?;
        ops::gemm_bf16(dev, &s, &self.norm_out_lin, None, Act::None, Epi::Store, &o)?;
        let mv = m.to_f32_vec(dev)?;
        let ov = o.to_f32_vec(dev)?;
        let tanh_bf = |v: f32| bf16_to_f32(bf16_bits(v.tanh()));
        let scale1 = Tensor::from_f32(dev, &mv[0..DIM], &[DIM])?;
        let gate1 = Tensor::from_f32(dev, &mv[DIM..2 * DIM].iter().map(|v| tanh_bf(*v)).collect::<Vec<_>>(), &[DIM])?;
        let scale2 = Tensor::from_f32(dev, &mv[2 * DIM..3 * DIM], &[DIM])?;
        let gate2 = Tensor::from_f32(dev, &mv[3 * DIM..4 * DIM].iter().map(|v| tanh_bf(*v)).collect::<Vec<_>>(), &[DIM])?;
        let out_scale = Tensor::from_f32(dev, &ov, &[DIM])?;
        Ok(Modulation { scale1, gate1, scale2, gate2, out_scale })
    }

    /// Prepare a run: `context` [Lc, 4096] bf16 with every reference spliced in at its slot, giving the
    /// sequence (text, reference, ..., text, target) like ComfyUI's build_sequence; target grid (h, w) in
    /// latent pixels. Text tokens attend causally, each reference block attends to everything before its end.
    pub fn prepare(&self, context: &Tensor, refs: &[RefLatent], h: usize, w: usize) -> Result<Run> {
        let dev = &self.dev;
        let lc = context.shape[0];
        let mut last = 0;
        for r in refs {
            ensure!(r.latent.shape[0] == r.rh * r.rw, "reference latent rows do not match its grid");
            ensure!(r.slot >= last && r.slot <= lc, "reference slots must be ascending and within the context");
            last = r.slot;
        }
        // txt_in
        let txt = {
            let n = Tensor::new(dev, DType::BF16, &[lc, DIM])?;
            ops::rmsnorm(dev, context, Some(&self.txt_norm), false, EPS, &n)?;
            let h1 = Tensor::new(dev, DType::BF16, &[lc, DIM])?;
            ops::gemm_bf16(dev, &n, &self.txt_in1, None, Act::GeluTanh, Epi::Store, &h1)?;
            let t = Tensor::new(dev, DType::BF16, &[lc, DIM])?;
            ops::gemm_bf16(dev, &h1, &self.txt_in2, None, Act::None, Epi::Store, &t)?;
            t
        };
        let p = lc + refs.iter().map(|r| r.rh * r.rw).sum::<usize>();
        let prefix_x = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        // prefix rows, rope ids (reference grids centred on the target) and block-causal key limits, segment by segment
        let center = |n: usize, nt: usize, i: usize| -> f64 { i as f64 - (n - n / 2) as f64 + 0.5 * ((n % 2) as f64 - (nt % 2) as f64) };
        let mut ids: Vec<[f64; 3]> = Vec::with_capacity(p + h * w);
        let mut lim: Vec<i32> = Vec::with_capacity(p);
        let (mut pos, mut row, mut txt_at) = (0f64, 0usize, 0usize);
        for seg in refs.iter().map(Some).chain(std::iter::once(None)) {
            let end = seg.map_or(lc, |r| r.slot);
            let n = end - txt_at;
            if n > 0 {
                dev.dtod(prefix_x.ptr + (row * DIM * 2) as u64, txt.ptr + (txt_at * DIM * 2) as u64, n * DIM * 2)?;
                for i in 0..n {
                    ids.push([pos + i as f64; 3]);
                    lim.push((row + i + 1) as i32);
                }
                pos += n as f64;
                row += n;
                txt_at = end;
            }
            match seg {
                Some(r) => {
                    let nr = r.rh * r.rw;
                    ops::gemm_bf16(dev, &r.latent, &self.img_in, None, Act::None, Epi::Store, &prefix_x.rows(row, nr))?;
                    for y in 0..r.rh {
                        for x in 0..r.rw {
                            ids.push([pos, center(r.rh, h, y), center(r.rw, w, x)]);
                        }
                    }
                    lim.extend(std::iter::repeat((row + nr) as i32).take(nr));
                    pos += r.rh.max(r.rw) as f64;
                    row += nr;
                }
                None => {
                    for y in 0..h {
                        for x in 0..w {
                            ids.push([pos, center(h, h, y), center(w, w, x)]);
                        }
                    }
                }
            }
        }
        debug_assert_eq!(row, p);
        let rope_all = rope_table(&ids);
        let rope_prefix = Tensor::from_f32(dev, &rope_all[..p * 128], &[p, 64, 2])?;
        let rope_target = Tensor::from_f32(dev, &rope_all[p * 128..], &[h * w, 64, 2])?;
        let kv_limit = Tensor::from_buf(dev.upload(&lim)?, DType::F32, &[p]);
        // prefix modulation (t = 0)
        let temb0 = self.temb(0.0)?;
        let m0 = self.modulation(&temb0)?;
        let mut cache_k = Vec::with_capacity(LAYERS);
        let mut cache_v = Vec::with_capacity(LAYERS);
        if !self.sage {
            for _ in 0..LAYERS {
                cache_k.push(Tensor::new(dev, DType::BF16, &[p, DIM])?);
                cache_v.push(Tensor::new(dev, DType::BF16, &[p, DIM])?);
            }
        }
        Ok(Run { prefix_x, rope_prefix, rope_target, kv_limit, cache_k, cache_v, cache_q: Vec::new(), cached: false, mod0: [m0.scale1, m0.gate1, m0.scale2, m0.gate2], temb0, p, h, w })
    }

    /// Run the prefix through all blocks once, filling the per-block K/V cache.
    fn fill_cache(&self, run: &mut Run) -> Result<()> {
        let dev = &self.dev;
        let p = run.p;
        let x = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        dev.dtod(x.ptr, run.prefix_x.ptr, run.prefix_x.bytes())?;
        let xq = Tensor::new(dev, DType::I8, &[p, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[p])?;
        let qkv = Tensor::new(dev, DType::BF16, &[p, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[p, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[p, MLP])?;
        let hq = Tensor::new(dev, DType::I8, &[p, MLP])?;
        let [scale1, gate1, scale2, gate2] = &run.mod0;
        for (li, b) in self.blocks.iter().enumerate() {
            ops::adaln_quant(dev, &x, scale1, EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.qkv.w, &b.qkv.scale, None, Epi::Store, &qkv)?;
            let qp = qkv.ptr;
            let kp = qkv.ptr + (DIM * 2) as u64;
            let vp = qkv.ptr + (2 * DIM * 2) as u64;
            ops::rms_rope_dit(dev, qp, 3 * DIM, kp, 3 * DIM, p, HEADS, &b.norm_q, &b.norm_k, EPS, &run.rope_prefix)?;
            if self.sage {
                let kv = ops::quantize_kv(dev, kp, vp, 3 * DIM, p, HEADS)?;
                let qq = ops::quantize_q(dev, qp, 3 * DIM, p, HEADS, &kv, None)?;
                ops::sage_attn(dev, &qq, p, HEADS, &kv, None, AttnView::contiguous(&attn, HEADS, HEAD_DIM), Some(&run.kv_limit))?;
                run.cache_q.push(kv);
            } else {
                // store K/V (strided -> contiguous)
                copy_2d(dev, run.cache_k[li].ptr, DIM * 2, kp, 3 * DIM * 2, DIM * 2, p)?;
                copy_2d(dev, run.cache_v[li].ptr, DIM * 2, vp, 3 * DIM * 2, DIM * 2, p)?;
                ops::flash_attn(
                    dev,
                    &AttnArgs {
                        q: AttnView { ptr: qp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: p },
                        k1: AttnView { ptr: kp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: p },
                        v1: AttnView { ptr: vp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: p },
                        k2: AttnView::empty(),
                        v2: AttnView::empty(),
                        out: AttnView::contiguous(&attn, HEADS, HEAD_DIM),
                        hq: HEADS,
                        hk: HEADS,
                        d: HEAD_DIM,
                        kv_limit: Some(&run.kv_limit),
                        causal: false,
                        causal_off: 0,
                    },
                )?;
            }
            ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.out.w, &b.out.scale, None, Epi::AddResGated(&x, gate1), &x)?;
            ops::adaln_quant(dev, &x, scale2, EPS, &xq, &xs)?;
            ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::SwiGluPairs, &gu)?;
            ops::quant_rows(dev, &gu, QuantAct::None, None, 0.0, &hq, &xs)?;
            ops::gemm_i8(dev, &hq, &xs, &b.mlp_out.w, &b.mlp_out.scale, None, Epi::AddResGated(&x, gate2), &x)?;
        }
        run.cached = true;
        Ok(())
    }

    /// One denoising evaluation: `latent` [Nt, 64] bf16 token-major (normalized space), sigma -> velocity [Nt, 64] bf16.
    pub fn forward(&self, run: &mut Run, latent: &Tensor, sigma: f32, out: &Tensor) -> Result<()> {
        let dev = &self.dev;
        if !run.cached {
            self.fill_cache(run).context("filling prefix cache")?;
        }
        let nt = run.h * run.w;
        let p = run.p;
        let temb = self.temb(sigma)?;
        let m = self.modulation(&temb)?;
        let _ = &run.temb0;
        let x = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        ops::gemm_bf16(dev, latent, &self.img_in, None, Act::None, Epi::Store, &x)?;
        let hbuf = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let xq = Tensor::new(dev, DType::I8, &[nt, DIM])?;
        let xs = Tensor::new(dev, DType::F32, &[nt])?;
        let qkv = Tensor::new(dev, DType::BF16, &[nt, 3 * DIM])?;
        let attn = Tensor::new(dev, DType::BF16, &[nt, DIM])?;
        let gu = Tensor::new(dev, DType::BF16, &[nt, MLP])?;
        let hq = Tensor::new(dev, DType::I8, &[nt, MLP])?;
        let pr = &self.prof;
        for (li, b) in self.blocks.iter().enumerate() {
            pr.time(dev, "adaln+quant", || ops::adaln_quant(dev, &x, &m.scale1, EPS, &xq, &xs))?;
            pr.time(dev, "gemm qkv", || ops::gemm_i8(dev, &xq, &xs, &b.qkv.w, &b.qkv.scale, None, Epi::Store, &qkv))?;
            let qp = qkv.ptr;
            let kp = qkv.ptr + (DIM * 2) as u64;
            let vp = qkv.ptr + (2 * DIM * 2) as u64;
            pr.time(dev, "rms_rope", || ops::rms_rope_dit(dev, qp, 3 * DIM, kp, 3 * DIM, nt, HEADS, &b.norm_q, &b.norm_k, EPS, &run.rope_target))?;
            if self.sage {
                let mut kv2 = None;
                pr.time(dev, "attn quantize", || { kv2 = Some(ops::quantize_kv(dev, kp, vp, 3 * DIM, nt, HEADS)?); Ok(()) })?;
                let kv2 = kv2.unwrap();
                let mut qq = None;
                pr.time(dev, "attn quantize", || { qq = Some(ops::quantize_q(dev, qp, 3 * DIM, nt, HEADS, &run.cache_q[li], Some(&kv2))?); Ok(()) })?;
                let qq = qq.unwrap();
                pr.time(dev, "attention", || ops::sage_attn(dev, &qq, nt, HEADS, &run.cache_q[li], Some(&kv2), AttnView::contiguous(&attn, HEADS, HEAD_DIM), None))?;
            } else {
                pr.time(dev, "attention", || ops::flash_attn(
                    dev,
                    &AttnArgs {
                        q: AttnView { ptr: qp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: nt },
                        k1: AttnView::contiguous(&run.cache_k[li], HEADS, HEAD_DIM),
                        v1: AttnView::contiguous(&run.cache_v[li], HEADS, HEAD_DIM),
                        k2: AttnView { ptr: kp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: nt },
                        v2: AttnView { ptr: vp, tok_stride: 3 * DIM, head_stride: HEAD_DIM, len: nt },
                        out: AttnView::contiguous(&attn, HEADS, HEAD_DIM),
                        hq: HEADS,
                        hk: HEADS,
                        d: HEAD_DIM,
                        kv_limit: None,
                        causal: false,
                        causal_off: 0,
                    },
                ))?;
            }
            let _ = p;
            pr.time(dev, "quant", || ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &xq, &xs))?;
            pr.time(dev, "gemm out", || ops::gemm_i8(dev, &xq, &xs, &b.out.w, &b.out.scale, None, Epi::AddResGated(&x, &m.gate1), &x))?;
            pr.time(dev, "adaln+quant", || ops::adaln_quant(dev, &x, &m.scale2, EPS, &xq, &xs))?;
            pr.time(dev, "gemm gate_up", || ops::gemm_i8(dev, &xq, &xs, &b.gate_up.w, &b.gate_up.scale, None, Epi::SwiGluPairs, &gu))?;
            pr.time(dev, "quant", || ops::quant_rows(dev, &gu, QuantAct::None, None, 0.0, &hq, &xs))?;
            pr.time(dev, "gemm mlp_out", || ops::gemm_i8(dev, &hq, &xs, &b.mlp_out.w, &b.mlp_out.scale, None, Epi::AddResGated(&x, &m.gate2), &x))?;
        }
        ops::adaln(dev, &x, &m.out_scale, &m.out_scale, 0, EPS, &hbuf)?;
        ops::gemm_bf16(dev, &hbuf, &self.proj_out, None, Act::None, Epi::Store, out)?;
        Ok(())
    }
}

/// Flux-style rope table: for each token, 64 (cos, sin) pairs: 8 from axis 0, 28 from axis 1, 28 from axis 2 (theta 10000).
fn rope_table(ids: &[[f64; 3]]) -> Vec<f32> {
    let mut out = vec![0f32; ids.len() * 128];
    for (t, id) in ids.iter().enumerate() {
        let mut j = 0;
        for a in 0..3 {
            let dim = AXES[a];
            for i in 0..dim / 2 {
                let scale = (2 * i) as f64 / dim as f64;
                let omega = 1.0 / 10000f64.powf(scale);
                let ang = (id[a] as f32) as f64 * omega; // ids are f32 in ComfyUI
                let ang = (ang as f32) as f64; // out = einsum in float32 after f64 omega... keep f32 angle
                out[(t * 64 + j) * 2] = ang.cos() as f32;
                out[(t * 64 + j) * 2 + 1] = ang.sin() as f32;
                j += 1;
            }
        }
    }
    out
}

/// Strided device copy of bf16 rows: `rows` rows of `width` bytes (pitches in bytes, all multiples of 16).
pub fn copy_2d(dev: &Device, dst: u64, dst_pitch: usize, src: u64, src_pitch: usize, width: usize, rows: usize) -> Result<()> {
    let w = width / 2;
    dev.launch_n("k_copy_rows_bf16", rows * (w / 8), &[Arg::Ptr(dst), Arg::I64((dst_pitch / 2) as i64), Arg::Ptr(src), Arg::I64((src_pitch / 2) as i64), Arg::I32(w as i32), Arg::I32(rows as i32)])
}

/// Latent helpers on token-major [N, 64] tensors.
pub fn latent_norm_in(dev: &Device, x: &Tensor, mean: &Tensor, std: &Tensor, out: &Tensor) -> Result<()> {
    let n = x.shape[0];
    dev.launch_n("k_latent_norm_in", n * 64, &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(n as i64), Arg::Ptr(mean.ptr), Arg::Ptr(std.ptr)])
}
pub fn latent_norm_out(dev: &Device, x: &Tensor, mean: &Tensor, std: &Tensor, out: &Tensor) -> Result<()> {
    let n = x.shape[0];
    dev.launch_n("k_latent_norm_out", n * 64, &[Arg::Ptr(x.ptr), Arg::Ptr(out.ptr), Arg::I64(n as i64), Arg::Ptr(mean.ptr), Arg::Ptr(std.ptr)])
}
pub fn axpy(dev: &Device, x: &Tensor, v: &Tensor, alpha: f32) -> Result<()> {
    dev.launch_n("k_axpy_f32_bf16", x.numel(), &[Arg::Ptr(x.ptr), Arg::Ptr(v.ptr), Arg::F32(alpha), Arg::I64(x.numel() as i64)])
}
