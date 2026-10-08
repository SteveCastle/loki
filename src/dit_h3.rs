//! MiniMax H3 audio-video DiT (ref2va): 50 single-stream blocks, hidden 5376, 56 heads x 128, int8 ConvRot
//! linears, adaLN-curve time conditioning, bf16 token refiner for the Qwen text states, Sage-style int8/fp8
//! bidirectional attention over the packed [text | refs | audio | video] sequence.
//!
//! Port of ComfyUI `comfy/ldm/minimax/model.py` (MiniMaxH3Model.forward incl. the audio-carry conversion).
//! Long sequences are processed in row chunks (norm/qkv/K-V quantization, then attention + out-proj + MLP per
//! query chunk), so only x (bf16), Q8/K8/V8 for the whole sequence and one chunk of intermediates are resident.
use crate::cuda::{Arg, Device, Profiler};
use crate::ops::{self, AttnArgs, AttnView, Epi, QuantAct};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::{Loader, QLinear};
use anyhow::{ensure, Context, Result};
use cudarc::driver::PushKernelArg;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const HIDDEN: usize = 5376;
pub const HEADS: usize = 56;
pub const HEAD_DIM: usize = 128;
pub const INNER: usize = HEADS * HEAD_DIM; // 7168
pub const FFN: usize = 14336;
pub const LAYERS: usize = 50;
pub const TEXT_DIM: usize = 5120;
pub const T_DIM: usize = 8;
pub const EPS: f32 = 1e-5;
pub const SHIFT_VIDEO: f32 = 12.0;
pub const SHIFT_AUDIO: f32 = 3.0;
pub const VISUAL_COND_TIMESTEP: f64 = 0.999;
pub const AUDIO_COND_TIMESTEP: f64 = 1.0;
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const FRAME_RESCALE: f64 = 5.0 / 3.0;
const MOD_ROWS: usize = 12; // 4 timestep classes x 3 modality tags
/// timestep classes: 0 = video/text (t_v), 1 = target audio (t_a), 2 = visual refs, 3 = audio refs
const CLS_V: i32 = 0;
const CLS_A: i32 = 1;
const CLS_REF_V: i32 = 2;
const CLS_REF_A: i32 = 3;
/// modality tags (ComfyUI seg_tag): video 0, text 1, audio 2
const TAG_VIDEO: i32 = 0;
const TAG_AUDIO: i32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefKind {
    Image,
    Audio,
    Video,
    VideoAudio,
}

/// One reference block of the ref2va payload (order = ComfyUI's ref_blocks order).
pub struct RefBlock {
    pub kind: RefKind,
    /// video latent frames (1 for images)
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    /// audio latent frames (Audio / VideoAudio)
    pub ref_audio_t: usize,
    /// f32 [24, t, h, w] normalized VAE latent (Image / Video / VideoAudio)
    pub video: Option<Tensor>,
    /// f32 [32, 2, rt] (Audio / VideoAudio)
    pub audio: Option<Tensor>,
}

pub struct DitInputs<'a> {
    /// raw Qwen states [L, 5120] (bf16 or f32)
    pub text: &'a Tensor,
    /// per text token: 0 = vision block, 1 = text
    pub text_tags: &'a [u8],
    pub refs: &'a [RefBlock],
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub audio_t: usize,
    pub seed: u64,
    /// visual condition noise augmentation; None => 0.999 (model default), Some(1.0) => no noise
    pub cond_noise_aug: Option<f32>,
}

struct Block {
    q: QLinear,  // rows [0, 7168) of qkv_proj
    kv: QLinear, // rows [7168, 21504) of qkv_proj
    out: QLinear,
    fc1: QLinear, // gate/up interleaved for the fused SwiGLU epilogue
    fc2: QLinear,
    norm1: Tensor,
    norm2: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
}

pub struct Dit {
    dev: Arc<Device>,
    pub prof: Profiler,
    st: SafeTensors,
    blocks: Vec<Block>,
    adaln_w: Tensor, // f16 raw [L, 96768, 8]
    adaln_b: Tensor, // f16 raw [L, 96768]
    fin_adaln_w: Tensor,
    fin_adaln_b: Tensor,
    fin_norm: Tensor,
    vout_w: Tensor,
    vout_b: Tensor,
    aout_w: Tensor,
    aout_b: Tensor,
    vproj_wt: Tensor, // f32 [96, 5376] (transposed)
    vproj_b: Tensor,
    aproj_wt: Tensor, // f32 [32, 5376]
    aproj_b: Tensor,
    cond_w: Tensor, // bf16 [5376, 5120]
    cond_b: Tensor, // f32
    t_table: Vec<f32>, // [1025, 8]
    inv_freq: Vec<f32>,
    pub path: PathBuf,
}

/// Prepared per-run state (layout, tables, refined text, buffers).
pub struct Run {
    pub seq: usize,
    pub s_pad: usize,
    pub text_len: usize,
    /// rows before the target audio (text + references)
    pub prefix: usize,
    pub a0: usize,
    pub audio_t: usize,
    pub v0: usize,
    pub lt: usize,
    pub lh: usize,
    pub lw: usize,
    pub chunk: usize,
    prefix_x: Tensor,
    modrow: Tensor, // i32 [S]
    rope: Tensor,   // bf16 [S, 48, 2]
    x: Tensor,
    /// per chunk: Q int8 [C, H, 128] + scales [C, H]
    q8: Tensor,
    sq: Tensor,
    /// whole sequence: K int8 [S_pad, H, 128] + scales, V fp16 [S_pad, H, 128] + per-64-key scales [S_pad/64, H]
    k8: Tensor,
    sk: Tensor,
    v16: Tensor,
    sv: Tensor,
    mean_k: Tensor,
    mean_v: Tensor,
    big: Tensor,
    xq: Tensor,
    xs: Tensor,
    modtab: Tensor,
    fmod: Tensor,
    /// debug: copy x to the host after these blocks (index, data)
    pub snap_layers: Vec<usize>,
    pub snapshots: Vec<(usize, Vec<f32>)>,
}

impl Run {
    /// Refined text states + embedded references (rows [0, prefix)) as f32 (debug).
    pub fn prefix_states(&self, dev: &Device) -> Result<Vec<f32>> {
        self.prefix_x.to_f32_vec(dev)
    }
    pub fn x_states(&self, dev: &Device) -> Result<Vec<f32>> {
        self.x.to_f32_vec(dev)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TEmbRaw {
    v: [[f32; 8]; 4],
}
unsafe impl cudarc::driver::DeviceRepr for TEmbRaw {}

#[repr(C)]
#[derive(Clone, Copy)]
struct EpiRaw {
    bias: u64,
    mode: i32,
    _p: i32,
    res: u64,
    gate: u64,
    modrow: u64,
    gstride: i64,
}
unsafe impl cudarc::driver::DeviceRepr for EpiRaw {}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnRaw {
    q: u64,
    sq: u64,
    k: u64,
    sk: u64,
    v: u64,
    sv: u64,
    mv: u64,
    o: u64,
    o_ts: i64,
    nq: i32,
    nk: i32,
    h: i32,
    scale_log2: f32,
}
unsafe impl cudarc::driver::DeviceRepr for AttnRaw {}

/// Hadamard-rotate q/k heads before the int8 attention quantization (H3_QK_ROT=0 disables)
fn qk_rotate() -> bool {
    std::env::var("H3_QK_ROT").map(|v| v != "0").unwrap_or(true)
}
fn attn_stages() -> usize {
    std::env::var("H3_ATTN_STAGES").ok().and_then(|v| v.parse().ok()).unwrap_or(3)
}
const ATTN_STAGE_BYTES: usize = 64 * 128 + 64 * 256 + 64 * 4;
fn attn_kernel() -> (&'static str, u32) {
    if attn_stages() == 2 {
        ("k_h3_attn2", (128 * 128 + 2 * ATTN_STAGE_BYTES) as u32)
    } else {
        ("k_h3_attn3", (128 * 128 + 3 * ATTN_STAGE_BYTES) as u32)
    }
}

fn view(t: &Tensor, off_bytes: usize, dtype: DType, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    assert!(off_bytes + n * dtype.size() <= t.buf.len, "view out of range");
    Tensor { buf: t.buf.clone(), ptr: t.buf.ptr() + off_bytes as u64, dtype, shape: shape.to_vec() }
}

fn time_shift_sigma(sigma: f32, from: f32, to: f32) -> f32 {
    let base = sigma / (from + sigma * (1.0 - from));
    to * base / (1.0 + (to - 1.0) * base)
}

/// ComfyUI _axis_from_sqrt_area
fn axis_from_sqrt_area(dim: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let n = dim / 2;
    (0..n).map(|i| (i as f64 * (ratio / n as f64) + (1.0 - ratio) / 2.0) * 32.0).collect()
}
/// (h, w) coordinates of one latent frame's 2x2 patch rows, and the w axis
fn frame_grid(h: usize, w: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let area = ((h * w) as f64).sqrt();
    let ha = axis_from_sqrt_area(h, area);
    let wa = axis_from_sqrt_area(w, area);
    let mut g = Vec::with_capacity(ha.len() * wa.len());
    for &y in &ha {
        for &x in &wa {
            g.push([y, x]);
        }
    }
    (g, wa)
}
fn video_t_spans(n: usize) -> Vec<f64> {
    (0..n).map(|k| FRAME_RESCALE * FRAME_PER_TOKEN[k % 5]).collect()
}
fn video_t_grid(n: usize, origin: f64) -> Vec<f64> {
    let spans = video_t_spans(n);
    let mut out = Vec::with_capacity(n);
    let mut c = 0.0f64;
    for k in 0..n {
        out.push(origin + c);
        c += spans[k];
    }
    out
}
fn span_sum(n: usize) -> f64 {
    video_t_spans(n).iter().fold(0.0, |a, b| a + b)
}
fn audio_grid(pos: &mut Vec<[f64; 3]>, cursor: f64, t: usize, w_low: f64, w_high: f64) {
    for ch in 0..2 {
        for i in 0..t {
            pos.push([cursor + i as f64, 0.0, if ch == 0 { w_low } else { w_high }]);
        }
    }
}
fn video_grid(pos: &mut Vec<[f64; 3]>, vt: usize, frame: &[[f64; 2]], cursor: f64) {
    for t in video_t_grid(vt, cursor) {
        for f in frame {
            pos.push([t, f[0], f[1]]);
        }
    }
}

impl Dit {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<Dit> {
        let st = SafeTensors::open(path)?;
        let t0 = std::time::Instant::now();
        ops::gemm_init(&dev)?;
        ops::attn_init(&dev)?;
        ops::sage_init(&dev)?;
        dev.set_max_smem("k_h3_gemm_i8", ops::GEMM_SMEM)?;
        dev.set_max_smem("k_h3_gemm_i8_w", ops::GEMM_SMEM_W)?;
        dev.set_max_smem("k_h3_norm_mod_quant", (HIDDEN * 4) as u32)?;
        dev.set_max_smem("k_h3_quant_v", 128 * 129 * 4)?;
        dev.set_max_smem("k_h3_attn2", (128 * 128 + 2 * ATTN_STAGE_BYTES) as u32)?;
        dev.set_max_smem("k_h3_attn3", (128 * 128 + 3 * ATTN_STAGE_BYTES) as u32)?;
        let (blocks, adaln_w, adaln_b, uploaded) = {
            let mut l = Loader::new(&st, dev.clone());
            let mut blocks = Vec::with_capacity(LAYERS);
            let no = 18 * HIDDEN;
            let adaln_w = Tensor::new(&dev, DType::U8, &[LAYERS * no * T_DIM * 2])?;
            let adaln_b = Tensor::new(&dev, DType::U8, &[LAYERS * no * 2])?;
            for i in 0..LAYERS {
                let p = format!("blocks.{i}");
                let fc1 = l.qlinear(&format!("{p}.mlp.fc1"))?;
                ensure!(fc1.n == 2 * FFN && fc1.k == HIDDEN, "unexpected fc1 shape");
                let fc1 = fc1.interleave_gate_up(&dev)?;
                let qkv = l.qlinear(&format!("{p}.attn.qkv_proj"))?;
                ensure!(qkv.n == 3 * INNER && qkv.k == HIDDEN, "unexpected qkv shape {}x{}", qkv.n, qkv.k);
                let sub = |r0: usize, n: usize| QLinear {
                    w: Tensor { buf: qkv.w.buf.clone(), ptr: qkv.w.ptr + (r0 * HIDDEN) as u64, dtype: DType::I8, shape: vec![n, HIDDEN] },
                    scale: Tensor { buf: qkv.scale.buf.clone(), ptr: qkv.scale.ptr + (r0 * 4) as u64, dtype: DType::F32, shape: vec![n] },
                    n,
                    k: HIDDEN,
                };
                blocks.push(Block {
                    q: sub(0, INNER),
                    kv: sub(INNER, 2 * INNER),
                    out: l.qlinear(&format!("{p}.attn.out_proj"))?,
                    fc1,
                    fc2: l.qlinear(&format!("{p}.mlp.fc2"))?,
                    norm1: l.bf16(&format!("{p}.norm1.weight"))?,
                    norm2: l.bf16(&format!("{p}.norm2.weight"))?,
                    q_norm: l.bf16(&format!("{p}.attn.q_norm.weight"))?,
                    k_norm: l.bf16(&format!("{p}.attn.k_norm.weight"))?,
                });
                let wi = st.info(&format!("{p}.adaln_proj.linear.weight"))?;
                ensure!(wi.dtype == "F16" && wi.shape == vec![no, T_DIM], "unexpected adaln weight {:?} {:?}", wi.dtype, wi.shape);
                let wb = st.bytes(&format!("{p}.adaln_proj.linear.weight"))?;
                dev.htod_at(adaln_w.ptr + (i * no * T_DIM * 2) as u64, wb)?;
                let bb = st.bytes(&format!("{p}.adaln_proj.linear.bias"))?;
                dev.htod_at(adaln_b.ptr + (i * no * 2) as u64, bb)?;
                l.uploaded += wb.len() + bb.len();
            }
            (blocks, adaln_w, adaln_b, l.uploaded)
        };
        let mut l = Loader::new(&st, dev.clone());
        ensure!(!st.has("final_layer.video_out.weight") || st.info("final_layer.video_out.weight")?.shape[0] == 96, "PDD multi-head final layer is not supported");
        let fin_adaln_w = Tensor::from_buf(dev.upload(st.bytes("final_layer.adaln_proj.linear.weight")?)?, DType::U8, &[2 * HIDDEN * T_DIM * 2]);
        let fin_adaln_b = Tensor::from_buf(dev.upload(st.bytes("final_layer.adaln_proj.linear.bias")?)?, DType::U8, &[2 * HIDDEN * 2]);
        let fin_norm = l.bf16("final_layer.norm.weight")?;
        let vout_w = l.f32("final_layer.video_out.weight")?;
        let vout_b = l.f32("final_layer.video_out.bias")?;
        let aout_w = l.f32("final_layer.audio_out.weight")?;
        let aout_b = l.f32("final_layer.audio_out.bias")?;
        let transpose = |name: &str, k: usize| -> Result<Tensor> {
            let w = st.f32s(name)?; // [N, k]
            let n = w.len() / k;
            let mut t = vec![0f32; w.len()];
            for i in 0..n {
                for j in 0..k {
                    t[j * n + i] = w[i * k + j];
                }
            }
            Tensor::from_f32(&dev, &t, &[k, n])
        };
        let vproj_wt = transpose("video_patch_proj.weight", 96)?;
        let vproj_b = l.f32("video_patch_proj.bias")?;
        let aproj_wt = transpose("audio_patch_proj.weight", 32)?;
        let aproj_b = l.f32("audio_patch_proj.bias")?;
        let cond_w = l.bf16("condition_proj.weight")?;
        let cond_b = l.f32("condition_proj.bias")?;
        let t_table = st.f32s("adaln_t_table")?;
        ensure!(t_table.len() == 1025 * T_DIM, "unexpected adaln_t_table size");
        let inv_freq = st.f32s("rope.inv_freq")?;
        ensure!(inv_freq.len() == 16);
        dev.sync()?;
        eprintln!("  dit: {:.2} GB uploaded in {:.1}s", (uploaded + l.uploaded) as f64 / 1e9, t0.elapsed().as_secs_f64());
        Ok(Dit {
            dev,
            prof: Profiler::new(),
            st,
            blocks,
            adaln_w,
            adaln_b,
            fin_adaln_w,
            fin_adaln_b,
            fin_norm,
            vout_w,
            vout_b,
            aout_w,
            aout_b,
            vproj_wt,
            vproj_b,
            aproj_wt,
            aproj_b,
            cond_w,
            cond_b,
            t_table,
            inv_freq,
            path: path.to_path_buf(),
        })
    }

    // ------------------------------------------------------------------ kernels
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, aq: u64, sa: u64, m: usize, w: &QLinear, mode: i32, res: u64, gate: u64, modrow: u64, out: u64) -> Result<()> {
        let n = w.n;
        let k = w.k;
        ensure!(k % 64 == 0, "gemm: K % 64");
        let wide = ops::gemm_i8_wide(m, n);
        let (kname, bn, smem) = if wide { ("k_h3_gemm_i8_w", 256, ops::GEMM_SMEM_W) } else { ("k_h3_gemm_i8", 128, ops::GEMM_SMEM) };
        let grid = (((n + bn - 1) / bn) * ((m + 127) / 128)) as u32;
        let ep = EpiRaw { bias: 0, mode, _p: 0, res, gate, modrow, gstride: (6 * HIDDEN) as i64 };
        let f = self.dev.func(kname)?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut b = self.dev.stream.launch_builder(&f);
        b.arg(&aq).arg(&w.w.ptr).arg(&out).arg(&mi).arg(&ni).arg(&ki).arg(&sa).arg(&w.scale.ptr).arg(&ep);
        unsafe { b.launch(cfg) }.with_context(|| format!("{kname} m={m} n={n} k={k}"))?;
        Ok(())
    }

    fn launch_temb(&self, name: &str, grid: u32, args: &[u64], ints: &[i32], te: &TEmbRaw, out: u64) -> Result<()> {
        let f = self.dev.func(name)?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        let mut b = self.dev.stream.launch_builder(&f);
        for a in args {
            b.arg(a);
        }
        for i in ints {
            b.arg(i);
        }
        b.arg(te).arg(&out);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Patch-embed rows from a channel-first latent into bf16 rows at `out`.
    #[allow(clippy::too_many_arguments)]
    fn embed(&self, src: &Tensor, video: bool, t: usize, h: usize, w: usize, nrows: usize, in_scale: f32, out: u64) -> Result<()> {
        ensure!(src.dtype == DType::F32, "embed: latent must be f32");
        let (wt, b, kf) = if video { (&self.vproj_wt, &self.vproj_b, 96) } else { (&self.aproj_wt, &self.aproj_b, 32) };
        if nrows == 0 {
            return Ok(());
        }
        self.dev.launch(
            "k_h3_embed",
            (((nrows + 7) / 8) as u32, 1, 1),
            (256, 1, 1),
            0,
            &[
                Arg::Ptr(src.ptr),
                Arg::I32(if video { 0 } else { 1 }),
                Arg::I32(t as i32),
                Arg::I32(h as i32),
                Arg::I32(w as i32),
                Arg::I32(nrows as i32),
                Arg::F32(in_scale),
                Arg::Ptr(wt.ptr),
                Arg::Ptr(b.ptr),
                Arg::I32(kf),
                Arg::I32(HIDDEN as i32),
                Arg::Ptr(out),
            ],
        )
    }

    /// q and k (k at column INNER) per-head RMSNorm (+ rope when `rope` != 0), in place
    fn qk_norm_rope(&self, qkv: u64, ts: usize, m: usize, wq: &Tensor, wk: &Tensor, rope: u64) -> Result<()> {
        self.dev.launch_n(
            "k_h3_qk_norm_rope",
            m * HEADS * 2 * 32,
            &[Arg::Ptr(qkv), Arg::I64(ts as i64), Arg::I32(m as i32), Arg::I32(HEADS as i32), Arg::I32(INNER as i32), Arg::Ptr(wq.ptr), Arg::Ptr(wk.ptr), Arg::F32(EPS), Arg::Ptr(rope), Arg::I32(2), Arg::I32(0)],
        )
    }
    /// one [m, H, 128] tensor (token stride ts): per-head RMSNorm + partial rope, in place
    fn head_norm_rope(&self, x: u64, ts: usize, m: usize, w: &Tensor, rope: u64) -> Result<()> {
        self.dev.launch_n(
            "k_h3_qk_norm_rope",
            m * HEADS * 32,
            &[Arg::Ptr(x), Arg::I64(ts as i64), Arg::I32(m as i32), Arg::I32(HEADS as i32), Arg::I32(0), Arg::Ptr(w.ptr), Arg::Ptr(w.ptr), Arg::F32(EPS), Arg::Ptr(rope), Arg::I32(1), Arg::I32(qk_rotate() as i32)],
        )
    }

    /// Attention of the current chunk's queries (run.q8 / run.sq, n rows) against the whole sequence.
    fn attention(&self, run: &Run, n: usize, out: u64) -> Result<()> {
        let p = AttnRaw {
            q: run.q8.ptr,
            sq: run.sq.ptr,
            k: run.k8.ptr,
            sk: run.sk.ptr,
            v: run.v16.ptr,
            sv: run.sv.ptr,
            mv: run.mean_v.ptr,
            o: out,
            o_ts: INNER as i64,
            nq: n as i32,
            nk: run.seq as i32,
            h: HEADS as i32,
            scale_log2: (1.0 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E,
        };
        let (name, smem) = attn_kernel();
        let f = self.dev.func(name)?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (((n + 127) / 128) as u32, HEADS as u32, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
        let mut b = self.dev.stream.launch_builder(&f);
        b.arg(&p);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    // ------------------------------------------------------------------ token refiner
    fn refine_text(&self, text: &Tensor) -> Result<Tensor> {
        let dev = &self.dev;
        let l = text.shape[0];
        ensure!(text.numel() == l * TEXT_DIM, "text states must be [L, {TEXT_DIM}], got {:?}", text.shape);
        let tb = if text.dtype == DType::BF16 {
            text.reshape(&[l, TEXT_DIM])
        } else {
            let t = Tensor::new(dev, DType::BF16, &[l, TEXT_DIM])?;
            ops::to_bf16(dev, text, &t)?;
            t
        };
        let x = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        ops::gemm_bf16(dev, &tb, &self.cond_w, Some(&self.cond_b), ops::Act::None, Epi::Store, &x)?;
        let n = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        let qkv = Tensor::new(dev, DType::BF16, &[l, 3 * INNER])?;
        let attn = Tensor::new(dev, DType::BF16, &[l, INNER])?;
        let h = Tensor::new(dev, DType::BF16, &[l, 2 * FFN])?;
        let s = Tensor::new(dev, DType::BF16, &[l, FFN])?;
        let mut ld = Loader::new(&self.st, dev.clone());
        for i in 0..2 {
            let p = format!("token_refiner.blocks.{i}");
            let norm1 = ld.bf16(&format!("{p}.norm1.weight"))?;
            let norm2 = ld.bf16(&format!("{p}.norm2.weight"))?;
            let qn = ld.bf16(&format!("{p}.attn.q_norm.weight"))?;
            let kn = ld.bf16(&format!("{p}.attn.k_norm.weight"))?;
            {
                let wqkv = ld.bf16(&format!("{p}.attn.qkv_proj.weight"))?;
                ops::rmsnorm(dev, &x, Some(&norm1), false, EPS, &n)?;
                ops::gemm_bf16(dev, &n, &wqkv, None, ops::Act::None, Epi::Store, &qkv)?;
            }
            self.qk_norm_rope(qkv.ptr, 3 * INNER, l, &qn, &kn, 0)?;
            ops::flash_attn(
                dev,
                &AttnArgs {
                    q: AttnView { ptr: qkv.ptr, tok_stride: 3 * INNER, head_stride: HEAD_DIM, len: l },
                    k1: AttnView { ptr: qkv.ptr + (INNER * 2) as u64, tok_stride: 3 * INNER, head_stride: HEAD_DIM, len: l },
                    v1: AttnView { ptr: qkv.ptr + (2 * INNER * 2) as u64, tok_stride: 3 * INNER, head_stride: HEAD_DIM, len: l },
                    k2: AttnView::empty(),
                    v2: AttnView::empty(),
                    out: AttnView::contiguous(&attn, HEADS, HEAD_DIM),
                    hq: HEADS,
                    hk: HEADS,
                    d: HEAD_DIM,
                    kv_limit: None,
                    causal: false,
                    causal_off: 0,
                },
            )?;
            {
                let wo = ld.bf16(&format!("{p}.attn.out_proj.weight"))?;
                ops::gemm_bf16(dev, &attn, &wo, None, ops::Act::None, Epi::AddRes(&x), &x)?;
            }
            ops::rmsnorm(dev, &x, Some(&norm2), false, EPS, &n)?;
            {
                let w1 = ld.bf16(&format!("{p}.mlp.fc1.weight"))?;
                ops::gemm_bf16(dev, &n, &w1, None, ops::Act::None, Epi::Store, &h)?;
            }
            ops::swiglu(dev, &h, &s)?;
            {
                let w2 = ld.bf16(&format!("{p}.mlp.fc2.weight"))?;
                ops::gemm_bf16(dev, &s, &w2, None, ops::Act::None, Epi::AddRes(&x), &x)?;
            }
            dev.sync()?; // release this block's weights before loading the next
        }
        let fnorm = ld.bf16("token_refiner.final_norm.weight")?;
        let out = Tensor::new(dev, DType::BF16, &[l, HIDDEN])?;
        ops::rmsnorm(dev, &x, Some(&fnorm), false, EPS, &out)?;
        Ok(out)
    }

    // ------------------------------------------------------------------ prepare
    pub fn prepare(&self, inp: &DitInputs) -> Result<Run> {
        let dev = &self.dev;
        let l = inp.text.shape[0];
        ensure!(inp.text_tags.len() == l, "text_tags length {} != text length {l}", inp.text_tags.len());
        let (lt, lh, lw, ta) = (inp.latent_t, inp.latent_h, inp.latent_w, inp.audio_t);
        ensure!(lh % 2 == 0 && lw % 2 == 0, "latent h/w must be even (got {lh}x{lw})");
        ensure!(lt > 0 && ta > 0, "empty target");

        // ---------------- layout (PackedLayout)
        let (frame, w_grid) = frame_grid(lh, lw);
        let target_w = (w_grid[0], *w_grid.last().unwrap());
        let mut pos: Vec<[f64; 3]> = Vec::new();
        let mut modrow: Vec<i32> = Vec::new();
        for i in 0..l {
            pos.push([i as f64, 0.0, 0.0]);
            ensure!(inp.text_tags[i] <= 1, "text tag must be 0 or 1");
            modrow.push(CLS_V * 3 + inp.text_tags[i] as i32);
        }
        let mut cursor = l as f64;
        for r in inp.refs {
            cursor += match r.kind {
                RefKind::Image => 1.0,
                RefKind::Audio => r.ref_audio_t as f64,
                RefKind::Video | RefKind::VideoAudio => (r.ref_audio_t as f64).max(span_sum(r.latent_t)),
            };
        }
        let target_cursor = cursor;
        // (kind is_video, row start, rows, ref index)
        enum Emb {
            Video { r: usize, row: usize },
            Audio { r: usize, row: usize },
        }
        let mut embeds = Vec::new();
        let mut c = l as f64;
        for (ri, r) in inp.refs.iter().enumerate() {
            match r.kind {
                RefKind::Image => {
                    ensure!(r.latent_h % 2 == 0 && r.latent_w % 2 == 0, "ref image latent dims must be even");
                    let (rf, _) = frame_grid(r.latent_h, r.latent_w);
                    embeds.push(Emb::Video { r: ri, row: pos.len() });
                    for f in &rf {
                        pos.push([c, f[0], f[1]]);
                        modrow.push(CLS_REF_V * 3 + TAG_VIDEO);
                    }
                    c += 1.0;
                }
                RefKind::Audio => {
                    let rt = r.ref_audio_t;
                    if rt > 0 {
                        embeds.push(Emb::Audio { r: ri, row: pos.len() });
                        audio_grid(&mut pos, c, rt, target_w.0, target_w.1);
                        modrow.extend(std::iter::repeat(CLS_REF_A * 3 + TAG_AUDIO).take(2 * rt));
                    }
                    c += rt as f64;
                }
                RefKind::Video | RefKind::VideoAudio => {
                    ensure!(r.latent_h % 2 == 0 && r.latent_w % 2 == 0, "ref video latent dims must be even");
                    let rt = r.ref_audio_t;
                    let vt = r.latent_t;
                    let (rf, rw) = frame_grid(r.latent_h, r.latent_w);
                    if rt > 0 {
                        embeds.push(Emb::Audio { r: ri, row: pos.len() });
                        audio_grid(&mut pos, c, rt, rw[0], *rw.last().unwrap());
                        modrow.extend(std::iter::repeat(CLS_REF_A * 3 + TAG_AUDIO).take(2 * rt));
                    }
                    embeds.push(Emb::Video { r: ri, row: pos.len() });
                    let n0 = pos.len();
                    video_grid(&mut pos, vt, &rf, c);
                    modrow.extend(std::iter::repeat(CLS_REF_V * 3 + TAG_VIDEO).take(pos.len() - n0));
                    c += (rt as f64).max(span_sum(vt));
                }
            }
        }
        let prefix = pos.len();
        let a0 = prefix;
        audio_grid(&mut pos, target_cursor, ta, target_w.0, target_w.1);
        modrow.extend(std::iter::repeat(CLS_A * 3 + TAG_AUDIO).take(2 * ta));
        let v0 = pos.len();
        video_grid(&mut pos, lt, &frame, target_cursor);
        modrow.extend(std::iter::repeat(CLS_V * 3 + TAG_VIDEO).take(pos.len() - v0));
        let seq = pos.len();
        let s_pad = (seq + 127) / 128 * 128;

        // ---------------- rope table: bf16 (cos, sin) of pos_f32 * inv_freq for 48 pairs (t,h,w x 16)
        let mut rope = vec![0u16; seq * 96];
        for (i, p) in pos.iter().enumerate() {
            for a in 0..3 {
                let pf = p[a] as f32;
                for f in 0..16 {
                    let ang = pf * self.inv_freq[f];
                    let j = a * 16 + f;
                    rope[(i * 48 + j) * 2] = bf16_bits((ang as f64).cos() as f32);
                    rope[(i * 48 + j) * 2 + 1] = bf16_bits((ang as f64).sin() as f32);
                }
            }
        }
        let rope = Tensor::from_bf16(dev, &rope, &[seq, 48, 2])?;
        let modrow_t = Tensor::from_buf(dev.upload(&modrow)?, DType::F32, &[seq]);

        // ---------------- prefix rows: refined text + reference embeddings
        let prefix_x = Tensor::new(dev, DType::BF16, &[prefix.max(1), HIDDEN])?;
        let text_states = self.refine_text(inp.text).context("token refiner")?;
        dev.dtod(prefix_x.ptr, text_states.ptr, l * HIDDEN * 2)?;
        drop(text_states);
        let aug = inp.cond_noise_aug.unwrap_or(VISUAL_COND_TIMESTEP as f32);
        for e in &embeds {
            match *e {
                Emb::Video { r, row } => {
                    let rb = &inp.refs[r];
                    let lat = rb.video.as_ref().context("reference block without video latent")?;
                    let (vt, h, w) = (if rb.kind == RefKind::Image { 1 } else { rb.latent_t }, rb.latent_h, rb.latent_w);
                    ensure!(lat.numel() == 24 * vt * h * w, "ref video latent shape {:?} != [24,{vt},{h},{w}]", lat.shape);
                    let src = if aug < 1.0 {
                        // every condition restarts the same RNG stream (as ComfyUI does)
                        use rand::SeedableRng;
                        use rand_distr::Distribution;
                        let mut rng = rand::rngs::StdRng::seed_from_u64(inp.seed);
                        let mut v = lat.to_f32_vec(dev)?;
                        for x in v.iter_mut() {
                            let nz: f32 = rand_distr::StandardNormal.sample(&mut rng);
                            *x = aug * *x + (1.0 - aug) * nz;
                        }
                        Tensor::from_f32(dev, &v, &lat.shape)?
                    } else {
                        lat.clone()
                    };
                    self.embed(&src, true, vt, h, w, vt * (h / 2) * (w / 2), 1.0, prefix_x.ptr + (row * HIDDEN * 2) as u64)?;
                }
                Emb::Audio { r, row } => {
                    let rb = &inp.refs[r];
                    let lat = rb.audio.as_ref().context("reference block without audio latent")?;
                    let rt = rb.ref_audio_t;
                    ensure!(lat.numel() == 64 * rt, "ref audio latent shape {:?} != [32,2,{rt}]", lat.shape);
                    self.embed(lat, false, rt, 0, 0, 2 * rt, 1.0, prefix_x.ptr + (row * HIDDEN * 2) as u64)?;
                }
            }
        }

        // ---------------- buffers
        let x = Tensor::new(dev, DType::BF16, &[seq, HIDDEN])?;
        let k8 = Tensor::zeros(dev, DType::I8, &[s_pad, HEADS, HEAD_DIM])?;
        let sk = Tensor::zeros(dev, DType::F32, &[s_pad, HEADS])?;
        let v16 = Tensor::zeros(dev, DType::BF16, &[s_pad, HEADS, HEAD_DIM])?; // fp16 bits
        let sv = Tensor::zeros(dev, DType::F32, &[s_pad / 64, HEADS])?;
        let mean_k = Tensor::zeros(dev, DType::F32, &[INNER])?;
        let mean_v = Tensor::zeros(dev, DType::F32, &[INNER])?;
        let modtab = Tensor::new(dev, DType::F32, &[LAYERS * MOD_ROWS * 6 * HIDDEN])?;
        let fmod = Tensor::new(dev, DType::F32, &[4 * 2 * HIDDEN])?;
        dev.sync()?;
        // chunk rows: as large as memory allows (<= 8192), multiple of 128
        let free = dev.free_mem()?;
        let per_row = 2 * INNER * 2 + FFN + INNER + 4 * HEADS + 4;
        let margin = 512usize << 20;
        let mut chunk = std::env::var("H3_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(8192usize);
        while chunk > 512 && chunk * per_row + margin > free {
            chunk /= 2;
        }
        chunk = chunk.min(s_pad);
        let big = Tensor::new(dev, DType::BF16, &[chunk, 2 * INNER])?;
        let q8 = Tensor::new(dev, DType::I8, &[chunk, HEADS, HEAD_DIM])?;
        let sq = Tensor::new(dev, DType::F32, &[chunk, HEADS])?;
        let xq = Tensor::new(dev, DType::I8, &[chunk, FFN])?;
        let xs = Tensor::new(dev, DType::F32, &[chunk])?;
        Ok(Run {
            seq,
            s_pad,
            text_len: l,
            prefix,
            a0,
            audio_t: ta,
            v0,
            lt,
            lh,
            lw,
            chunk,
            prefix_x,
            modrow: modrow_t,
            rope,
            x,
            q8,
            sq,
            k8,
            sk,
            v16,
            sv,
            mean_k,
            mean_v,
            big,
            xq,
            xs,
            modtab,
            fmod,
            snap_layers: Vec::new(),
            snapshots: Vec::new(),
        })
    }

    fn lerp_temb(&self, t: f32) -> [f32; 8] {
        let pos = t.clamp(0.0, 1.0) * 1024.0;
        let i0 = (pos.floor() as usize).min(1023);
        let w = pos - i0 as f32;
        let mut out = [0f32; 8];
        for k in 0..8 {
            let a = self.t_table[i0 * 8 + k];
            let b = self.t_table[(i0 + 1) * 8 + k];
            out[k] = if w.abs() < 0.5 { w.mul_add(b - a, a) } else { (-(b - a)).mul_add(1.0 - w, b) };
        }
        out
    }

    // ------------------------------------------------------------------ forward
    /// ComfyUI MiniMaxH3Model.forward(x=[video, carried audio], timestep=sigma*1000): writes the (negated)
    /// velocity for the video and for the carried audio variable.
    pub fn forward(&self, run: &mut Run, x_video: &Tensor, x_audio: &Tensor, sigma: f32, out_video: &Tensor, out_audio: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let pr = &self.prof;
        let (lt, lh, lw, ta) = (run.lt, run.lh, run.lw, run.audio_t);
        ensure!(x_video.numel() == 24 * lt * lh * lw && x_video.dtype == DType::F32, "x_video must be f32 [24,{lt},{lh},{lw}]");
        ensure!(x_audio.numel() == 64 * ta && x_audio.dtype == DType::F32, "x_audio must be f32 [32,2,{ta}]");
        ensure!(out_video.numel() == x_video.numel() && out_video.dtype == DType::F32, "out_video shape");
        ensure!(out_audio.numel() == x_audio.numel() && out_audio.dtype == DType::F32, "out_audio shape");
        let seq = run.seq;

        // ---- timesteps (f32 like torch) and adaLN tables
        let s_v = sigma.max(1e-6);
        let s_a = time_shift_sigma(s_v, SHIFT_VIDEO, SHIFT_AUDIO);
        let t_v = 1.0 - s_v;
        let t_a = 1.0 - s_a;
        let cls = [t_v, t_a, (t_v as f64).max(VISUAL_COND_TIMESTEP) as f32, (t_a as f64).max(AUDIO_COND_TIMESTEP) as f32];
        let te = TEmbRaw { v: [self.lerp_temb(cls[0]), self.lerp_temb(cls[1]), self.lerp_temb(cls[2]), self.lerp_temb(cls[3])] };
        pr.time(dev, "adaln tables", || {
            let n = LAYERS * 18 * HIDDEN;
            self.launch_temb("k_h3_mod_table", ((n + 255) / 256) as u32, &[self.adaln_w.ptr, self.adaln_b.ptr], &[LAYERS as i32, HIDDEN as i32], &te, run.modtab.ptr)?;
            self.launch_temb("k_h3_mod_final", ((2 * HIDDEN + 255) / 256) as u32, &[self.fin_adaln_w.ptr, self.fin_adaln_b.ptr], &[HIDDEN as i32], &te, run.fmod.ptr)
        })?;
        let audio_scale = SHIFT_VIDEO / SHIFT_AUDIO;
        let carry = s_a / s_v;

        // ---- embed
        pr.time(dev, "embed", || {
            dev.dtod(run.x.ptr, run.prefix_x.ptr, run.prefix * HIDDEN * 2)?;
            self.embed(x_audio, false, ta, 0, 0, 2 * ta, carry, run.x.ptr + (run.a0 * HIDDEN * 2) as u64)?;
            self.embed(x_video, true, lt, lh, lw, lt * (lh / 2) * (lw / 2), 1.0, run.x.ptr + (run.v0 * HIDDEN * 2) as u64)
        })?;
        if run.snap_layers.contains(&usize::MAX) {
            let v = run.x.to_f32_vec(dev)?;
            run.snapshots.push((usize::MAX, v));
        }

        // ---- blocks
        let c = run.chunk;
        let nchunks = (seq + c - 1) / c;
        let order: Vec<usize> = if nchunks >= 2 { std::iter::once(nchunks - 2).chain(std::iter::once(nchunks - 1)).chain(0..nchunks - 2).collect() } else { vec![0] };
        let row_bytes = HIDDEN * 2;
        let mstride = MOD_ROWS * 6 * HIDDEN * 4;
        let ts_kv = 2 * INNER;
        for (li, b) in self.blocks.iter().enumerate() {
            let modtab = run.modtab.ptr + (li * mstride) as u64;
            let norm_quant = |r0: usize, n: usize, w: &Tensor, k_shift: i32, k_scale: i32| -> Result<()> {
                pr.time(dev, "norm+mod+quant", || {
                    dev.launch(
                        "k_h3_norm_mod_quant",
                        (n as u32, 1, 1),
                        (256, 1, 1),
                        (HIDDEN * 4) as u32,
                        &[
                            Arg::Ptr(run.x.ptr + (r0 * row_bytes) as u64),
                            Arg::I32(n as i32),
                            Arg::I32(HIDDEN as i32),
                            Arg::Ptr(w.ptr),
                            Arg::F32(EPS),
                            Arg::Ptr(run.modrow.ptr + (r0 * 4) as u64),
                            Arg::Ptr(modtab),
                            Arg::I32(k_shift),
                            Arg::I32(k_scale),
                            Arg::Ptr(run.xq.ptr),
                            Arg::Ptr(run.xs.ptr),
                        ],
                    )
                })
            };
            // phase 1 (all chunks): norm1 + modulation + quant -> K|V projection -> k norm + rope -> K8 / V16
            for (oi, &ci) in order.iter().enumerate() {
                let r0 = ci * c;
                let n = c.min(seq - r0);
                norm_quant(r0, n, &b.norm1, 0, 1)?;
                pr.time(dev, "gemm kv", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.kv, 0, 0, 0, 0, run.big.ptr))?;
                let kp = run.big.ptr;
                let vp = run.big.ptr + (INNER * 2) as u64;
                pr.time(dev, "qk norm+rope", || self.head_norm_rope(kp, ts_kv, n, &b.k_norm, run.rope.ptr + (r0 * 96 * 2) as u64))?;
                pr.time(dev, "attn quantize", || {
                    if oi == 0 {
                        // smoothing means from the first processed (full) chunk: any per-channel vector keeps the
                        // single-segment softmax exact (K) / is restored in the output (V)
                        ops::col_mean(dev, kp, ts_kv, n, INNER, &run.mean_k)?;
                        ops::col_mean(dev, vp, ts_kv, n, INNER, &run.mean_v)?;
                    }
                    let tok = (r0 * HEADS) as u64;
                    dev.launch_n("k_quant_qk_int8", n * HEADS * 32, &[Arg::Ptr(kp), Arg::I64(ts_kv as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(run.mean_k.ptr), Arg::Ptr(run.k8.ptr + tok * 128), Arg::Ptr(run.sk.ptr + tok * 4)])?;
                    dev.launch(
                        "k_h3_quant_v16",
                        (((n + 63) / 64) as u32, HEADS as u32, 1),
                        (256, 1, 1),
                        0,
                        &[Arg::Ptr(vp), Arg::I64(ts_kv as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(run.mean_v.ptr), Arg::Ptr(run.v16.ptr), Arg::I32(r0 as i32), Arg::Ptr(run.sv.ptr)],
                    )
                })?;
            }
            // phase 2 (per chunk): norm1 -> Q projection -> q norm + rope -> Q8 -> attention -> out-proj (gated residual)
            //                      -> norm2 + mod + quant -> fc1 (SwiGLU) -> quant -> fc2 (gated residual)
            for ci in 0..nchunks {
                let r0 = ci * c;
                let n = c.min(seq - r0);
                let xr = run.x.ptr + (r0 * row_bytes) as u64;
                let mr = run.modrow.ptr + (r0 * 4) as u64;
                norm_quant(r0, n, &b.norm1, 0, 1)?;
                pr.time(dev, "gemm q", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.q, 0, 0, 0, 0, run.big.ptr))?;
                pr.time(dev, "qk norm+rope", || self.head_norm_rope(run.big.ptr, INNER, n, &b.q_norm, run.rope.ptr + (r0 * 96 * 2) as u64))?;
                if std::env::var("H3_DEBUG_QSTATS").is_ok() && ci == 0 && li % 7 == 0 {
                    let q = view(&run.big, 0, DType::BF16, &[n, INNER]).to_f32_vec(dev)?;
                    let (mut ratio, mut amax_r) = (0f64, 0f64);
                    for h in 0..HEADS {
                        let mut mean = [0f64; 128];
                        let mut ss = 0f64;
                        let mut am = 0f64;
                        for i in 0..n {
                            let mut rowmax = 0f64;
                            let mut rowss = 0f64;
                            for d in 0..128 {
                                let v = q[i * INNER + h * 128 + d] as f64;
                                mean[d] += v / n as f64;
                                ss += v * v;
                                rowss += v * v;
                                rowmax = rowmax.max(v.abs());
                            }
                            am += rowmax / (rowss / 128.0).sqrt() / n as f64;
                        }
                        let mn = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
                        ratio += mn / (ss / n as f64).sqrt() / HEADS as f64;
                        amax_r += am / HEADS as f64;
                    }
                    eprintln!("layer {li}: |mean_q|/rms|q| = {ratio:.3}, row amax/rms = {amax_r:.2}");
                }
                pr.time(dev, "attn quantize", || {
                    dev.launch_n("k_quant_qk_int8", n * HEADS * 32, &[Arg::Ptr(run.big.ptr), Arg::I64(INNER as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(0), Arg::Ptr(run.q8.ptr), Arg::Ptr(run.sq.ptr)])
                })?;
                let attn = view(&run.big, 0, DType::BF16, &[n, INNER]);
                pr.time(dev, "attention", || self.attention(run, n, attn.ptr))?;
                let xq_a = view(&run.xq, 0, DType::I8, &[n, INNER]);
                pr.time(dev, "quant", || ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &xq_a, &run.xs))?;
                pr.time(dev, "gemm out", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.out, 4, xr, modtab + (2 * HIDDEN * 4) as u64, mr, xr))?;
                norm_quant(r0, n, &b.norm2, 3, 4)?;
                pr.time(dev, "gemm fc1", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.fc1, 3, 0, 0, 0, run.big.ptr))?;
                let h = view(&run.big, 0, DType::BF16, &[n, FFN]);
                let xq_h = view(&run.xq, 0, DType::I8, &[n, FFN]);
                pr.time(dev, "quant", || ops::quant_rows(dev, &h, QuantAct::None, None, 0.0, &xq_h, &run.xs))?;
                pr.time(dev, "gemm fc2", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.fc2, 4, xr, modtab + (5 * HIDDEN * 4) as u64, mr, xr))?;
            }
            if run.snap_layers.contains(&li) {
                let v = run.x.to_f32_vec(dev)?;
                run.snapshots.push((li, v));
            }
        }

        // ---- final layer (video rows, then audio rows), chunked through `big` as f32 scratch
        let coef = 1.0 + (audio_scale - 1.0) * s_a;
        pr.time(dev, "final", || {
            for (start, rows, video) in [(run.v0, seq - run.v0, true), (run.a0, 2 * ta, false)] {
                let cls_i = if video { 0 } else { 1 };
                let shift = run.fmod.ptr + ((cls_i * 2) * HIDDEN * 4) as u64;
                let scale = run.fmod.ptr + ((cls_i * 2 + 1) * HIDDEN * 4) as u64;
                let fc = 2 * c; // f32 rows that fit in `big`
                let mut r = 0;
                while r < rows {
                    let n = fc.min(rows - r);
                    dev.launch(
                        "k_h3_final_mod",
                        (n as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &[Arg::Ptr(run.x.ptr + ((start + r) * row_bytes) as u64), Arg::I32(n as i32), Arg::I32(HIDDEN as i32), Arg::Ptr(self.fin_norm.ptr), Arg::F32(EPS), Arg::Ptr(shift), Arg::Ptr(scale), Arg::Ptr(run.big.ptr)],
                    )?;
                    let (w, bias, no, kind, t) = if video { (&self.vout_w, &self.vout_b, 96, 0, lt) } else { (&self.aout_w, &self.aout_b, 32, 1, ta) };
                    let (yp, alpha, beta, out) = if video { (0u64, 0.0f32, 1.0f32, out_video) } else { (x_audio.ptr, 1.0 - audio_scale, coef, out_audio) };
                    dev.launch(
                        "k_h3_head",
                        (((n + 63) / 64) as u32, 1, 1),
                        (256, 1, 1),
                        0,
                        &[Arg::Ptr(run.big.ptr), Arg::I32(n as i32), Arg::I32(HIDDEN as i32), Arg::Ptr(w.ptr), Arg::Ptr(bias.ptr), Arg::I32(no), Arg::I32(kind), Arg::I32(t as i32), Arg::I32(lh as i32), Arg::I32(lw as i32), Arg::I32(r as i32), Arg::Ptr(yp), Arg::F32(carry), Arg::F32(alpha), Arg::F32(beta), Arg::Ptr(out.ptr)],
                    )?;
                    r += n;
                }
            }
            Ok(())
        })?;
        Ok(())
    }
}
