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
    fc1_scale: Tensor, // fc1 per-row scales (gate/up interleaved); the weight lives in Dit::fc1w (evictable)
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
    /// fc1 weights [28672, 5376] int8 (gate/up interleaved), per layer on the device or evicted to host memory
    /// when a very long run needs the VRAM (streamed through Run::fc1_stage then)
    fc1w: std::sync::Mutex<Vec<Fc1W>>,
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
    t_table: Vec<f32>, // [1025, 8]
    inv_freq: Vec<f32>,
    pub path: PathBuf,
}

enum Fc1W {
    Dev(Tensor),
    Host(Vec<u8>),
}
const FC1_BYTES: usize = 2 * FFN * HIDDEN;

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
    /// attention query batch rows (multiple of chunk)
    pub abatch: usize,
    prefix_x: Tensor,
    modrow: Tensor, // i32 [S]
    rope: Tensor,   // bf16 [S, 48, 2]
    x: Tensor,
    /// per attention batch: Q int8 [A, H, 128] + scales [A, H], attention output bf16 [A, 7168]
    q8: Tensor,
    sq: Tensor,
    attn_out: Tensor,
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
    /// staging buffer for evicted fc1 weights (only when some layers are evicted)
    fc1_stage: Option<Tensor>,
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
/// PV precision of the attention kernel: fp8 V (default; half the L2->SM traffic, which bounds this kernel)
/// or fp16 V (H3_ATTN=fp16).
fn attn_fp16() -> bool {
    std::env::var("H3_ATTN").map(|v| v == "fp16").unwrap_or(false)
}
const ATTN_V4_SMEM: u32 = (128 * 128 + 3 * (64 * 144 + 64 * 272 + 64 * 4)) as u32;
const ATTN_V5_SMEM: u32 = (128 * 128 + 4 * (64 * 144 + 128 * 80 + 64 * 4)) as u32;
const ATTN_V6_SMEM: u32 = (256 * 128 + 3 * (64 * 144 + 128 * 80 + 64 * 4)) as u32;
const ATTN_V5G3_SMEM: u32 = (192 * 128 + 3 * (64 * 144 + 128 * 80 + 64 * 4)) as u32;
/// (kernel, dynamic smem, queries per block, threads per block)
fn attn_kernel() -> (&'static str, u32, usize, u32) {
    if attn_fp16() {
        ("k_h3_attn_v4", ATTN_V4_SMEM, 128, 256)
    } else if std::env::var("H3_ATTN_V6").is_ok() {
        ("k_h3_attn_v6", ATTN_V6_SMEM, 256, 256)
    } else if let Ok(k) = std::env::var("H3_ATTN_G2") {
        (match k.as_str() { "nosm" => "k_h3_attn_v5_nosm", "nosm_nold" => "k_h3_attn_v5_nosm_nold", "nold" => "k_h3_attn_v5_nold", "a" => "k_h3_attn_v5_a", "b" => "k_h3_attn_v5_b", _ => "k_h3_attn_v5" }, ATTN_V5_SMEM, 128, 256)
    } else {
        (match std::env::var("H3_ATTN_G3").as_deref() { Ok("a") => "k_h3_attn_v5g3_a", Ok("b") => "k_h3_attn_v5g3_b", _ => "k_h3_attn_v5g3" }, ATTN_V5G3_SMEM, 192, 384)
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Attn8Raw {
    q: u64,
    sq: u64,
    k: u64,
    sk: u64,
    vt: u64,
    s_pad: i64,
    sv: u64,
    mv: u64,
    o: u64,
    o_ts: i64,
    nq: i32,
    nk: i32,
    h: i32,
    scale_log2: f32,
}
unsafe impl cudarc::driver::DeviceRepr for Attn8Raw {}

/// Attention launch: q8/sq for nq queries, K8/sk, V (fp16 [nk_pad][H][128] or fp8 [H][128][nk_pad]) + per-256-key
/// scales, mean_v; out bf16 [nq, 7168].
#[allow(clippy::too_many_arguments)]
fn launch_attention(dev: &Device, q: u64, sq: u64, nq: usize, k: u64, sk: u64, v: u64, s_pad: usize, sv: u64, mv: u64, nk: usize, out: u64) -> Result<()> {
    let scale_log2 = (1.0 / (HEAD_DIM as f32).sqrt()) * std::f32::consts::LOG2_E;
    let (name, smem, bq, threads) = attn_kernel();
    let f = dev.func(name)?;
    let cfg = cudarc::driver::LaunchConfig { grid_dim: (((nq + bq - 1) / bq) as u32, HEADS as u32, 1), block_dim: (threads, 1, 1), shared_mem_bytes: smem };
    let mut b = dev.stream.launch_builder(&f);
    let p16 = AttnRaw { q, sq, k, sk, v, sv, mv, o: out, o_ts: INNER as i64, nq: nq as i32, nk: nk as i32, h: HEADS as i32, scale_log2 };
    let p8 = Attn8Raw { q, sq, k, sk, vt: v, s_pad: s_pad as i64, sv, mv, o: out, o_ts: INNER as i64, nq: nq as i32, nk: nk as i32, h: HEADS as i32, scale_log2 };
    if attn_fp16() {
        b.arg(&p16);
    } else {
        b.arg(&p8);
    }
    unsafe { b.launch(cfg) }?;
    Ok(())
}

/// Stand-alone attention micro-benchmark on random data (no weights): seconds per call for `nq` queries
/// against `nk` keys (56 heads).
pub fn attention_bench(dev: &Device, nq: usize, nk: usize, iters: usize) -> Result<f64> {
    let (name, smem, _, _) = attn_kernel();
    dev.set_max_smem(name, smem)?;
    let nk_pad = (nk + 255) / 256 * 256;
    let mut seed = 12345u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let q8: Vec<i8> = (0..nq * INNER).map(|_| (rnd() % 255) as i32 as i8).collect();
    let k8: Vec<i8> = (0..nk_pad * INNER).map(|_| (rnd() % 255) as i32 as i8).collect();
    let q8 = Tensor::from_i8(dev, &q8, &[nq * INNER])?;
    let k8 = Tensor::from_i8(dev, &k8, &[nk_pad * INNER])?;
    let v = if attn_fp16() {
        let v16: Vec<u16> = (0..nk_pad * INNER).map(|_| half::f16::from_f32((rnd() % 1000) as f32 / 10.0 - 50.0).to_bits()).collect();
        Tensor::from_bf16(dev, &v16, &[nk_pad * INNER])?
    } else {
        let v8: Vec<i8> = (0..nk_pad * INNER).map(|_| ((rnd() % 100) as i32 + 0x20) as i8).collect();
        Tensor::from_i8(dev, &v8, &[nk_pad * INNER])?
    };
    let sq = Tensor::from_f32(dev, &vec![0.002f32; nq * HEADS], &[nq * HEADS])?;
    let sk = Tensor::from_f32(dev, &vec![0.002f32; nk_pad * HEADS], &[nk_pad * HEADS])?;
    let sv = Tensor::from_f32(dev, &vec![0.5f32; nk_pad / 256 * HEADS], &[nk_pad / 256 * HEADS])?;
    let mv = Tensor::zeros(dev, DType::F32, &[INNER])?;
    let out = Tensor::new(dev, DType::BF16, &[nq, INNER])?;
    launch_attention(dev, q8.ptr, sq.ptr, nq, k8.ptr, sk.ptr, v.ptr, nk_pad, sv.ptr, mv.ptr, nk, out.ptr)?;
    dev.sync()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch_attention(dev, q8.ptr, sq.ptr, nq, k8.ptr, sk.ptr, v.ptr, nk_pad, sv.ptr, mv.ptr, nk, out.ptr)?;
    }
    dev.sync()?;
    let o = out.to_f32_vec(dev)?;
    if !o.iter().all(|x| x.is_finite()) {
        eprintln!("warning: attention bench produced non-finite output");
    }
    Ok(t.elapsed().as_secs_f64() / iters as f64)
}

/// Stand-alone int8 GEMM micro-benchmark (random data): seconds per call of [m,k] x [n,k]^T with epilogue `mode`.
pub fn gemm_bench(dev: &Arc<Device>, m: usize, n: usize, k: usize, mode: i32, iters: usize) -> Result<f64> {
    ops::gemm_init(dev)?;
    dev.set_max_smem("k_h3_gemm_i8", ops::GEMM_SMEM)?;
    dev.set_max_smem("k_h3_gemm_i8_w", ops::GEMM_SMEM_W)?;
    dev.set_max_smem("k_h3_gemm_i8_w4", 4 * (128 * 64 + 256 * 64))?;
    let mut seed = 777u32;
    let mut rnd = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    let av: Vec<i8> = (0..m * k).map(|_| (rnd() >> 8) as u8 as i8).collect();
    let wv: Vec<i8> = (0..n * k).map(|_| (rnd() >> 8) as u8 as i8).collect();
    let a = Tensor::from_i8(dev, &av, &[m, k])?;
    let w = QLinear { w: Tensor::from_i8(dev, &wv, &[n, k])?, scale: Tensor::zeros(dev, DType::F32, &[n])?, n, k };
    let sa = Tensor::zeros(dev, DType::F32, &[m])?;
    let out = Tensor::zeros(dev, DType::BF16, &[m, n])?;
    let gate = Tensor::zeros(dev, DType::F32, &[12 * 6 * HIDDEN.max(n)])?;
    let mr = Tensor::zeros(dev, DType::F32, &[m])?;
    let run = |_: ()| -> Result<()> {
        Dit::gemm_raw(dev, a.ptr, sa.ptr, m, &w, mode, out.ptr, gate.ptr, mr.ptr, out.ptr)
    };
    run(())?;
    dev.sync()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        run(())?;
    }
    dev.sync()?;
    Ok(t.elapsed().as_secs_f64() / iters as f64)
}

/// Hand the CUDA default memory pool's cached free blocks back to the driver (the engine keeps freed memory in
/// the pool; after the refiner weights are dropped this makes ~1.5 GB visible to the scratch sizing again).
fn trim_mem_pool(dev: &Device) -> Result<()> {
    use cudarc::driver::sys;
    dev.sync()?;
    unsafe {
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        let mut d: sys::CUdevice = 0;
        if sys::cuCtxGetDevice(&mut d) == sys::CUresult::CUDA_SUCCESS && sys::cuDeviceGetDefaultMemPool(&mut pool, d) == sys::CUresult::CUDA_SUCCESS {
            let _ = sys::cuMemPoolTrimTo(pool, 0);
        }
    }
    Ok(())
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
        dev.set_max_smem("k_h3_gemm_i8_w4", 4 * (128 * 64 + 256 * 64))?;
        dev.set_max_smem("k_h3_norm_mod_quant", (HIDDEN * 4) as u32)?;
        dev.set_max_smem("k_h3_quant_v", 128 * 129 * 4)?;
        let (an, asm, _, _) = attn_kernel();
        dev.set_max_smem(an, asm)?;
        dev.set_max_smem("k_h3_quant_v8", 64 * 129 * 4)?;
        // All block weights live in one exact-size allocation (the stream-ordered pool otherwise over-reserves
        // ~1.7 GB while growing, which matters at ~110k tokens).
        let mut fc1_list: Vec<Fc1W> = Vec::with_capacity(LAYERS);
        let (blocks, adaln_w, adaln_b, uploaded) = {
            let no = 18 * HIDDEN;
            let names = |i: usize| -> Vec<(String, usize)> {
                let p = format!("blocks.{i}");
                vec![
                    (format!("{p}.attn.qkv_proj.weight"), 3 * INNER * HIDDEN),
                    (format!("{p}.attn.qkv_proj.weight_scale"), 3 * INNER * 4),
                    (format!("{p}.attn.out_proj.weight"), HIDDEN * INNER),
                    (format!("{p}.attn.out_proj.weight_scale"), HIDDEN * 4),
                    (format!("{p}.mlp.fc1.weight_scale"), 2 * FFN * 4),
                    (format!("{p}.mlp.fc2.weight"), HIDDEN * FFN),
                    (format!("{p}.mlp.fc2.weight_scale"), HIDDEN * 4),
                    (format!("{p}.norm1.weight"), HIDDEN * 2),
                    (format!("{p}.norm2.weight"), HIDDEN * 2),
                    (format!("{p}.attn.q_norm.weight"), HEAD_DIM * 2),
                    (format!("{p}.attn.k_norm.weight"), HEAD_DIM * 2),
                ]
            };
            let align = |x: usize| (x + 255) / 256 * 256;
            let mut total = align(LAYERS * no * T_DIM * 2) + align(LAYERS * no * 2);
            for i in 0..LAYERS {
                for (name, bytes) in names(i) {
                    let have = st.bytes(&name)?.len();
                    ensure!(have == bytes, "{name}: {have} bytes, expected {bytes}");
                    total += align(bytes);
                }
                let meta: serde_json::Value = serde_json::from_slice(st.bytes(&format!("blocks.{i}.mlp.fc1.comfy_quant"))?)?;
                for q in ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"] {
                    let m: serde_json::Value = serde_json::from_slice(st.bytes(&format!("blocks.{i}.{q}.comfy_quant"))?)?;
                    ensure!(m["format"] == "int8_tensorwise" && m["convrot"] == true && m["convrot_groupsize"] == 256, "blocks.{i}.{q}: unsupported quant format {m}");
                }
                let _ = meta;
            }
            let arena = Arc::new(dev.alloc(total)?);
            let mut off = 0usize;
            let mut uploaded = 0usize;
            let mut take = |dtype: DType, shape: &[usize], data: Option<&[u8]>| -> Result<Tensor> {
                let n: usize = shape.iter().product::<usize>() * dtype.size();
                let t = Tensor { buf: arena.clone(), ptr: arena.ptr() + off as u64, dtype, shape: shape.to_vec() };
                if let Some(d) = data {
                    ensure!(d.len() == n, "arena upload size mismatch");
                    dev.htod_at(t.ptr, d)?;
                    uploaded += d.len();
                }
                off += align(n);
                Ok(t)
            };
            let adaln_w = take(DType::U8, &[LAYERS * no * T_DIM * 2], None)?;
            let adaln_b = take(DType::U8, &[LAYERS * no * 2], None)?;
            let mut blocks = Vec::with_capacity(LAYERS);
            let mut adaln_bytes = 0usize;
            for i in 0..LAYERS {
                let p = format!("blocks.{i}");
                let qlin = |take: &mut dyn FnMut(DType, &[usize], Option<&[u8]>) -> Result<Tensor>, q: &str, n: usize, k: usize| -> Result<QLinear> {
                    let w = take(DType::I8, &[n, k], Some(st.bytes(&format!("{p}.{q}.weight"))?))?;
                    let scale = take(DType::F32, &[n], Some(st.bytes(&format!("{p}.{q}.weight_scale"))?))?;
                    Ok(QLinear { w, scale, n, k })
                };
                let qkv = qlin(&mut take, "attn.qkv_proj", 3 * INNER, HIDDEN)?;
                let out = qlin(&mut take, "attn.out_proj", HIDDEN, INNER)?;
                // fc1: [gate | up] rows interleaved (gate_j, up_j) on the host for the fused SwiGLU epilogue
                let (fc1_scale, fc1_w) = {
                    let (n, k) = (2 * FFN, HIDDEN);
                    let wb = st.bytes(&format!("{p}.mlp.fc1.weight"))?;
                    let sb = st.bytes(&format!("{p}.mlp.fc1.weight_scale"))?;
                    let mut wi = vec![0u8; n * k];
                    let mut si = vec![0u8; n * 4];
                    for r in 0..n {
                        let src = if r % 2 == 0 { r / 2 } else { FFN + r / 2 };
                        wi[r * k..(r + 1) * k].copy_from_slice(&wb[src * k..(src + 1) * k]);
                        si[r * 4..(r + 1) * 4].copy_from_slice(&sb[src * 4..(src + 1) * 4]);
                    }
                    let w = Tensor::new(&dev, DType::I8, &[n, k])?;
                    dev.htod_at(w.ptr, &wi)?;
                    let scale = take(DType::F32, &[n], Some(&si))?;
                    (scale, w)
                };
                fc1_list.push(Fc1W::Dev(fc1_w));
                let fc2 = qlin(&mut take, "mlp.fc2", HIDDEN, FFN)?;
                let norm1 = take(DType::BF16, &[HIDDEN], Some(st.bytes(&format!("{p}.norm1.weight"))?))?;
                let norm2 = take(DType::BF16, &[HIDDEN], Some(st.bytes(&format!("{p}.norm2.weight"))?))?;
                let q_norm = take(DType::BF16, &[HEAD_DIM], Some(st.bytes(&format!("{p}.attn.q_norm.weight"))?))?;
                let k_norm = take(DType::BF16, &[HEAD_DIM], Some(st.bytes(&format!("{p}.attn.k_norm.weight"))?))?;
                let sub = |r0: usize, n: usize| QLinear {
                    w: Tensor { buf: qkv.w.buf.clone(), ptr: qkv.w.ptr + (r0 * HIDDEN) as u64, dtype: DType::I8, shape: vec![n, HIDDEN] },
                    scale: Tensor { buf: qkv.scale.buf.clone(), ptr: qkv.scale.ptr + (r0 * 4) as u64, dtype: DType::F32, shape: vec![n] },
                    n,
                    k: HIDDEN,
                };
                blocks.push(Block { q: sub(0, INNER), kv: sub(INNER, 2 * INNER), out, fc1_scale, fc2, norm1, norm2, q_norm, k_norm });
                let wi = st.info(&format!("{p}.adaln_proj.linear.weight"))?;
                ensure!(wi.dtype == "F16" && wi.shape == vec![no, T_DIM], "unexpected adaln weight {:?} {:?}", wi.dtype, wi.shape);
                let wb = st.bytes(&format!("{p}.adaln_proj.linear.weight"))?;
                dev.htod_at(adaln_w.ptr + (i * no * T_DIM * 2) as u64, wb)?;
                let bb = st.bytes(&format!("{p}.adaln_proj.linear.bias"))?;
                dev.htod_at(adaln_b.ptr + (i * no * 2) as u64, bb)?;
                adaln_bytes += wb.len() + bb.len();
            }
            drop(take);
            (blocks, adaln_w, adaln_b, uploaded + adaln_bytes + LAYERS * FC1_BYTES)
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
        let t_table = st.f32s("adaln_t_table")?;
        ensure!(t_table.len() == 1025 * T_DIM, "unexpected adaln_t_table size");
        let inv_freq = st.f32s("rope.inv_freq")?;
        ensure!(inv_freq.len() == 16);
        trim_mem_pool(&dev)?;
        eprintln!("  dit: {:.2} GB uploaded in {:.1}s, {} MB free", (uploaded + l.uploaded) as f64 / 1e9, t0.elapsed().as_secs_f64(), dev.free_mem()? >> 20);
        Ok(Dit {
            dev,
            prof: Profiler::new(),
            st,
            blocks,
            fc1w: std::sync::Mutex::new(fc1_list),
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
            t_table,
            inv_freq,
            path: path.to_path_buf(),
        })
    }

    // ------------------------------------------------------------------ kernels
    #[allow(clippy::too_many_arguments)]
    fn gemm(&self, aq: u64, sa: u64, m: usize, w: &QLinear, mode: i32, res: u64, gate: u64, modrow: u64, out: u64) -> Result<()> {
        Dit::gemm_raw(&self.dev, aq, sa, m, w, mode, res, gate, modrow, out)
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm_raw(dev: &Device, aq: u64, sa: u64, m: usize, w: &QLinear, mode: i32, res: u64, gate: u64, modrow: u64, out: u64) -> Result<()> {
        let n = w.n;
        let k = w.k;
        ensure!(k % 64 == 0, "gemm: K % 64");
        let wide = ops::gemm_i8_wide(m, n);
        let w4 = std::env::var("H3_GEMM_W4").is_ok();
        let (kname, bn, smem) = if wide && w4 { ("k_h3_gemm_i8_w4", 256, 4 * (128 * 64 + 256 * 64)) } else if wide { ("k_h3_gemm_i8_w", 256, ops::GEMM_SMEM_W) } else { ("k_h3_gemm_i8", 128, ops::GEMM_SMEM) };
        let grid = (((n + bn - 1) / bn) * ((m + 127) / 128)) as u32;
        let ep = EpiRaw { bias: 0, mode, _p: 0, res, gate, modrow, gstride: (6 * HIDDEN) as i64 };
        let f = dev.func(kname)?;
        let cfg = cudarc::driver::LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem };
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut b = dev.stream.launch_builder(&f);
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

    /// Fused per-head RMSNorm + rope + H128 rotation (- mean) + int8 quantization of [m, H, 128] (token stride ts)
    #[allow(clippy::too_many_arguments)]
    fn head_nrq(&self, x: u64, ts: usize, m: usize, w: &Tensor, rope: u64, mean: u64, q8: u64, scale: u64) -> Result<()> {
        self.dev.launch_n(
            "k_h3_head_nrq",
            m * HEADS * 32,
            &[Arg::Ptr(x), Arg::I64(ts as i64), Arg::I32(m as i32), Arg::I32(HEADS as i32), Arg::Ptr(w.ptr), Arg::F32(EPS), Arg::Ptr(rope), Arg::Ptr(mean), Arg::Ptr(q8), Arg::Ptr(scale)],
        )
    }

    /// Attention of the current chunk's queries (run.q8 / run.sq, n rows) against the whole sequence.
    fn attention(&self, run: &Run, n: usize, out: u64) -> Result<()> {
        launch_attention(&self.dev, run.q8.ptr, run.sq.ptr, n, run.k8.ptr, run.sk.ptr, run.v16.ptr, run.s_pad, run.sv.ptr, run.mean_v.ptr, run.seq, out)
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
        {
            // condition_proj + refiner weights are only needed here: loaded per prepare, freed afterwards
            let mut ld = Loader::new(&self.st, dev.clone());
            let cond_w = ld.bf16("condition_proj.weight")?;
            let cond_b = ld.f32("condition_proj.bias")?;
            ops::gemm_bf16(dev, &tb, &cond_w, Some(&cond_b), ops::Act::None, Epi::Store, &x)?;
            dev.sync()?;
        }
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
        let s_pad = (seq + 255) / 256 * 256;

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
        trim_mem_pool(dev)?;
        if std::env::var("H3_DEBUG_MEM").is_ok() {
            eprintln!("  dit prepare: free after refiner {} MB", dev.free_mem()? >> 20);
        }
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
        // V: fp16 [S_pad, H, 128] (H3_ATTN=fp16) or fp8 e4m3 transposed [H, 128, S_pad]
        let v16 = if attn_fp16() { Tensor::zeros(dev, DType::BF16, &[s_pad, HEADS, HEAD_DIM])? } else { Tensor::zeros(dev, DType::U8, &[HEADS, HEAD_DIM, s_pad])? };
        let sv = Tensor::zeros(dev, DType::F32, &[s_pad / 256, HEADS])?;
        let mean_k = Tensor::zeros(dev, DType::F32, &[INNER])?;
        let mean_v = Tensor::zeros(dev, DType::F32, &[INNER])?;
        let modtab = Tensor::new(dev, DType::F32, &[MOD_ROWS * 6 * HIDDEN])?; // built per layer
        let fmod = Tensor::new(dev, DType::F32, &[4 * 2 * HIDDEN])?;
        dev.sync()?;
        // Scratch sizes from the free memory: GEMM/MLP chunk rows `chunk` (big + xq: ~43 KB/row) and the attention
        // query batch `abatch` (Q8 + scales + bf16 output: ~22 KB/row), abatch a multiple of chunk, both multiples of 256.
        let per_c = 2 * INNER * 2 + FFN + 4;
        let per_a = INNER + 4 * HEADS + INNER * 2;
        let margin = 160usize << 20;
        let fc1_stage = self.balance_fc1(8192 * per_a + 4096 * per_c + margin)?;
        trim_mem_pool(dev)?;
        let free = dev.free_mem()?;
        if std::env::var("H3_DEBUG_MEM").is_ok() {
            eprintln!("  dit prepare: seq {seq}, free before scratch {} MB", free >> 20);
        }
        let budget = free.saturating_sub(margin);
        let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
        let (mut abatch, mut chunk) = (8192usize, 8192usize);
        let cands = [(8192usize, 8192usize), (8192, 4096), (8192, 2048), (4096, 2048), (4096, 1024), (2048, 1024), (2048, 512), (1024, 512), (512, 256)];
        for (a, c) in cands {
            abatch = a;
            chunk = c;
            if a * per_a + c * per_c <= budget {
                break;
            }
        }
        if let Some(c) = env("H3_CHUNK") {
            chunk = c;
            abatch = abatch.max(c);
        }
        if let Some(a) = env("H3_ABATCH") {
            abatch = a;
        }
        chunk = ((chunk / 256).max(1) * 256).min(s_pad);
        abatch = ((abatch / chunk).max(1) * chunk).min(s_pad.div_ceil(chunk) * chunk);
        let big = Tensor::new(dev, DType::BF16, &[chunk, 2 * INNER])?;
        let q8 = Tensor::new(dev, DType::I8, &[abatch, HEADS, HEAD_DIM])?;
        let sq = Tensor::new(dev, DType::F32, &[abatch, HEADS])?;
        let attn_out = Tensor::new(dev, DType::BF16, &[abatch, INNER])?;
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
            abatch,
            prefix_x,
            modrow: modrow_t,
            rope,
            x,
            q8,
            sq,
            attn_out,
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
            fc1_stage,
            snap_layers: Vec::new(),
            snapshots: Vec::new(),
        })
    }

    /// Make `want` bytes of device memory available for run scratch by evicting fc1 weights of the last layers
    /// to host memory (or restore evicted layers when memory allows). Returns the staging buffer if any layer
    /// stays evicted.
    fn balance_fc1(&self, want: usize) -> Result<Option<Tensor>> {
        let dev = &self.dev;
        let mut slots = self.fc1w.lock().unwrap();
        trim_mem_pool(dev)?;
        let free = dev.free_mem()?;
        let evicted = slots.iter().filter(|s| matches!(s, Fc1W::Host(_))).count();
        if free < want {
            // each evicted layer frees FC1_BYTES; one staging buffer (FC1_BYTES) is needed once anything is evicted
            let need = want - free + if evicted == 0 { FC1_BYTES } else { 0 };
            let k = need.div_ceil(FC1_BYTES).min(LAYERS);
            let mut done = 0;
            for li in (0..LAYERS).rev() {
                if done == k {
                    break;
                }
                if let Fc1W::Dev(t) = &slots[li] {
                    let mut host = vec![0u8; FC1_BYTES];
                    let v: Vec<u8> = dev.dtoh_at(t.ptr, FC1_BYTES)?;
                    host.copy_from_slice(&v);
                    slots[li] = Fc1W::Host(host);
                    done += 1;
                }
            }
            eprintln!("  dit: evicted fc1 of {done} more layer(s) to host memory ({} total) to fit the run", evicted + done);
        } else if evicted > 0 {
            // restore layers while memory allows (keeping `want` free)
            let mut avail = free - want;
            for li in 0..LAYERS {
                if let Fc1W::Host(bytes) = &slots[li] {
                    // the last restored layer also releases the staging buffer
                    if avail < FC1_BYTES {
                        break;
                    }
                    let t = Tensor::new(dev, DType::I8, &[2 * FFN, HIDDEN])?;
                    dev.htod_at(t.ptr, bytes)?;
                    slots[li] = Fc1W::Dev(t);
                    avail -= FC1_BYTES;
                }
            }
        }
        trim_mem_pool(dev)?;
        let still = slots.iter().filter(|s| matches!(s, Fc1W::Host(_))).count();
        Ok(if still > 0 { Some(Tensor::new(dev, DType::I8, &[2 * FFN, HIDDEN])?) } else { None })
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
        pr.time(dev, "adaln tables", || self.launch_temb("k_h3_mod_final", ((2 * HIDDEN + 255) / 256) as u32, &[self.fin_adaln_w.ptr, self.fin_adaln_b.ptr], &[HIDDEN as i32], &te, run.fmod.ptr))?;
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
        let ts_kv = 2 * INNER;
        for (li, b) in self.blocks.iter().enumerate() {
            // fc1 weight (device resident, or streamed from host into the staging buffer once per layer)
            let fc1 = {
                let slots = self.fc1w.lock().unwrap();
                let w = match &slots[li] {
                    Fc1W::Dev(t) => t.clone(),
                    Fc1W::Host(bytes) => {
                        let stage = run.fc1_stage.as_ref().context("evicted fc1 weights but no staging buffer")?;
                        pr.time(dev, "fc1 upload", || {
                            dev.sync()?; // the previous user of the staging buffer must be done
                            dev.htod_at(stage.ptr, bytes)
                        })?;
                        stage.clone()
                    }
                };
                QLinear { w, scale: b.fc1_scale.clone(), n: 2 * FFN, k: HIDDEN }
            };
            // this layer's adaLN rows: 4 timestep classes x 3 modality tags x (shift, scale, gate) x 2
            let modtab = run.modtab.ptr;
            pr.time(dev, "adaln tables", || {
                let no = 18 * HIDDEN;
                self.launch_temb(
                    "k_h3_mod_table",
                    ((no + 255) / 256) as u32,
                    &[self.adaln_w.ptr + (li * no * T_DIM * 2) as u64, self.adaln_b.ptr + (li * no * 2) as u64],
                    &[1, HIDDEN as i32],
                    &te,
                    modtab,
                )
            })?;
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
                let rope_r = run.rope.ptr + (r0 * 96 * 2) as u64;
                let tok = (r0 * HEADS) as u64;
                pr.time(dev, "qk norm+rope+quant", || {
                    if oi == 0 {
                        // smoothing means from the first processed (full) chunk: any per-channel vector keeps the
                        // single-segment softmax exact (K) / is restored in the output (V)
                        self.head_norm_rope(kp, ts_kv, n, &b.k_norm, rope_r)?;
                        ops::col_mean(dev, kp, ts_kv, n, INNER, &run.mean_k)?;
                        ops::col_mean(dev, vp, ts_kv, n, INNER, &run.mean_v)?;
                        dev.launch_n("k_quant_qk_int8", n * HEADS * 32, &[Arg::Ptr(kp), Arg::I64(ts_kv as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(run.mean_k.ptr), Arg::Ptr(run.k8.ptr + tok * 128), Arg::Ptr(run.sk.ptr + tok * 4)])
                    } else {
                        self.head_nrq(kp, ts_kv, n, &b.k_norm, rope_r, run.mean_k.ptr, run.k8.ptr + tok * 128, run.sk.ptr + tok * 4)
                    }
                })?;
                pr.time(dev, "attn quantize", || {
                    if attn_fp16() {
                        dev.launch(
                            "k_h3_quant_v16",
                            (((n + 255) / 256) as u32, HEADS as u32, 1),
                            (256, 1, 1),
                            0,
                            &[Arg::Ptr(vp), Arg::I64(ts_kv as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(run.mean_v.ptr), Arg::Ptr(run.v16.ptr), Arg::I32(r0 as i32), Arg::Ptr(run.sv.ptr)],
                        )
                    } else {
                        dev.launch(
                            "k_h3_quant_v8",
                            (((n + 255) / 256) as u32, HEADS as u32, 1),
                            (256, 1, 1),
                            64 * 129 * 4,
                            &[Arg::Ptr(vp), Arg::I64(ts_kv as i64), Arg::I32(n as i32), Arg::I32(HEADS as i32), Arg::Ptr(run.mean_v.ptr), Arg::Ptr(run.v16.ptr), Arg::I64(run.s_pad as i64), Arg::I32(r0 as i32), Arg::Ptr(run.sv.ptr)],
                        )
                    }
                })?;
            }
            // phase 2, per attention batch of `abatch` rows (sub-chunks of `chunk` rows):
            //   norm1 -> Q projection -> q norm + rope + rotate + int8 -> attention over the whole sequence ->
            //   per sub-chunk: quant -> out-proj (gated residual) -> norm2 + mod + quant -> fc1 (SwiGLU) -> quant -> fc2 (gated residual)
            let ab = run.abatch;
            for b0 in (0..seq).step_by(ab) {
                let bn = ab.min(seq - b0);
                for r0 in (b0..b0 + bn).step_by(c) {
                    let n = c.min(b0 + bn - r0);
                    norm_quant(r0, n, &b.norm1, 0, 1)?;
                    pr.time(dev, "gemm q", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.q, 0, 0, 0, 0, run.big.ptr))?;
                    let qo = ((r0 - b0) * HEADS) as u64;
                    pr.time(dev, "qk norm+rope+quant", || self.head_nrq(run.big.ptr, INNER, n, &b.q_norm, run.rope.ptr + (r0 * 96 * 2) as u64, 0, run.q8.ptr + qo * 128, run.sq.ptr + qo * 4))?;
                }
                pr.time(dev, "attention", || self.attention(run, bn, run.attn_out.ptr))?;
                for r0 in (b0..b0 + bn).step_by(c) {
                    let n = c.min(b0 + bn - r0);
                    let xr = run.x.ptr + (r0 * row_bytes) as u64;
                    let mr = run.modrow.ptr + (r0 * 4) as u64;
                    let attn = view(&run.attn_out, (r0 - b0) * INNER * 2, DType::BF16, &[n, INNER]);
                    let xq_a = view(&run.xq, 0, DType::I8, &[n, INNER]);
                    pr.time(dev, "quant", || ops::quant_rows(dev, &attn, QuantAct::None, None, 0.0, &xq_a, &run.xs))?;
                    pr.time(dev, "gemm out", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.out, 4, xr, modtab + (2 * HIDDEN * 4) as u64, mr, xr))?;
                    norm_quant(r0, n, &b.norm2, 3, 4)?;
                    pr.time(dev, "gemm fc1", || self.gemm(run.xq.ptr, run.xs.ptr, n, &fc1, 3, 0, 0, 0, run.big.ptr))?;
                    let h = view(&run.big, 0, DType::BF16, &[n, FFN]);
                    let xq_h = view(&run.xq, 0, DType::I8, &[n, FFN]);
                    pr.time(dev, "quant", || ops::quant_rows(dev, &h, QuantAct::None, None, 0.0, &xq_h, &run.xs))?;
                    pr.time(dev, "gemm fc2", || self.gemm(run.xq.ptr, run.xs.ptr, n, &b.fc2, 4, xr, modtab + (5 * HIDDEN * 4) as u64, mr, xr))?;
                }
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
