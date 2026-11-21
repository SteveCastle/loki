//! The Qwen3.5 / Qwen3.8 hybrid text decoder (GGUF architecture "qwen35"), dense FFN variants.
//!
//! Layers alternate three Gated DeltaNet linear-attention blocks with one gated full-attention block
//! (`full_attention_interval`). One forward pass serves both phases:
//!   * prefill: one sequence, a chunk of T rows; weights are dequantized to bf16 per matrix and multiplied on the
//!     tensor cores (`Linear::gemm`);
//!   * decode: up to 8 sequences, one row each; weights stay quantized and are read once for all rows by the q8
//!     dp4a GEMV kernels (`Linear::gemv`), so a batch of sequences costs about as much as one.
//! The residual stream is f32. Each sequence owns its state (`SeqState`): the conv window and the 128x128 delta-rule
//! state of every DeltaNet layer, and a bf16 KV cache of every attention layer, sized for the request.
use crate::cuda::{Arg, Device};
use crate::gguf::{GgmlType, Gguf};
use crate::ops::{self, view, view_at, Epi};
use crate::quant::{self, Act, Linear};
use crate::tensor::{DType, Tensor};
use anyhow::{bail, ensure, Context, Result};
use std::sync::Arc;

/// Per-category GPU timing (LOKI_PROFILE=1; synchronizes around every timed op).
pub static PROF: std::sync::LazyLock<crate::cuda::Profiler> = std::sync::LazyLock::new(crate::cuda::Profiler::new);

/// Rows of the largest prefill chunk (longer prompts are fed in several chunks).
pub const PREFILL_CHUNK: usize = 2048;
/// Sequences decoded together (the GEMV kernels take up to 8 activation rows).
pub const MAX_BATCH: usize = 8;

#[derive(Clone, Debug)]
pub struct Config {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_ff: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub rope_base: f32,
    pub sections: [usize; 3],
    pub eps: f32,
    pub conv_k: usize,
    pub d_state: usize,
    pub n_kgroup: usize,
    pub n_vhead: usize,
    pub d_inner: usize,
    pub full_interval: usize,
    pub vocab: usize,
    pub ctx_train: usize,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        let arch = g.s("general.architecture").unwrap_or("");
        ensure!(arch == "qwen35", "{}: architecture {arch:?} is not supported (loki-scribe runs Qwen3.5/3.8 \"qwen35\" GGUFs)", g.path.display());
        let k = |s: &str| format!("qwen35.{s}");
        let sections = g.get(&k("rope.dimension_sections")).and_then(|v| v.as_arr()).map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as usize).collect::<Vec<_>>()).unwrap_or_default();
        ensure!(sections.len() >= 3, "missing rope.dimension_sections");
        let head_dim = g.u(&k("attention.key_length"))?;
        let c = Config {
            n_layer: g.u(&k("block_count"))?,
            n_embd: g.u(&k("embedding_length"))?,
            n_ff: g.u(&k("feed_forward_length"))?,
            n_head: g.u(&k("attention.head_count"))?,
            n_head_kv: g.u(&k("attention.head_count_kv"))?,
            head_dim,
            rope_dims: g.u(&k("rope.dimension_count"))?,
            rope_base: g.f(&k("rope.freq_base"))? as f32,
            sections: [sections[0], sections[1], sections[2]],
            eps: g.f(&k("attention.layer_norm_rms_epsilon"))? as f32,
            conv_k: g.u(&k("ssm.conv_kernel"))?,
            d_state: g.u(&k("ssm.state_size"))?,
            n_kgroup: g.u(&k("ssm.group_count"))?,
            n_vhead: g.u(&k("ssm.time_step_rank"))?,
            d_inner: g.u(&k("ssm.inner_size"))?,
            full_interval: g.u(&k("full_attention_interval")).unwrap_or(4),
            vocab: g.tensor("token_embd.weight")?.dims[1],
            ctx_train: g.u(&k("context_length")).unwrap_or(262144),
        };
        // the kernels are written for these shapes (every Qwen3.5/3.8 dense size shares them)
        ensure!(c.head_dim == 256 && c.rope_dims == 64, "attention head dim {} / rope dims {} unsupported (want 256 / 64)", c.head_dim, c.rope_dims);
        ensure!(c.d_state == 128 && c.d_inner == c.n_vhead * 128 && c.n_vhead % c.n_kgroup == 0, "DeltaNet shape unsupported");
        ensure!(c.conv_k == 4, "conv kernel {} unsupported", c.conv_k);
        ensure!(c.n_head % c.n_head_kv == 0 && c.n_head / c.n_head_kv <= 6, "GQA group {} unsupported", c.n_head / c.n_head_kv);
        ensure!(c.n_embd % 256 == 0 && c.n_ff % 256 == 0, "hidden sizes must be multiples of 256");
        Ok(c)
    }
    pub fn is_attn(&self, l: usize) -> bool {
        (l + 1) % self.full_interval == 0
    }
    pub fn conv_ch(&self) -> usize {
        2 * self.n_kgroup * self.d_state + self.d_inner
    }
    fn n_attn(&self) -> usize {
        (0..self.n_layer).filter(|&l| self.is_attn(l)).count()
    }
    fn n_gdn(&self) -> usize {
        self.n_layer - self.n_attn()
    }
    /// KV cache bytes per token of one sequence.
    pub fn kv_bytes_per_token(&self) -> usize {
        self.n_attn() * 2 * self.n_head_kv * self.head_dim * 2
    }
    /// Recurrent state bytes of one sequence.
    pub fn state_bytes(&self) -> usize {
        self.n_gdn() * (3 * self.conv_ch() + self.n_vhead * 128 * 128) * 4
    }
}

enum Mixer {
    Attn { q: Linear, k: Linear, v: Linear, o: Linear, q_norm: Tensor, k_norm: Tensor, idx: usize },
    Gdn { qkv: Linear, z: Linear, beta: Linear, alpha: Linear, conv: Tensor, dt: Tensor, a: Tensor, norm: Tensor, out: Linear, idx: usize },
}

struct Layer {
    attn_norm: Tensor,
    post_norm: Tensor,
    mixer: Mixer,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// One sequence's decoder state on the GPU.
pub struct SeqState {
    conv: Tensor, // f32 [n_gdn][3][C]
    ssm: Tensor,  // f32 [n_gdn][H][128][128]
    kv: Tensor,   // bf16 [n_attn][2][cap][HK][256]
    pub cap: usize,
    /// tokens already in the cache
    pub len: usize,
}

impl SeqState {
    pub fn new(dev: &Device, c: &Config, cap: usize) -> Result<SeqState> {
        let conv = Tensor::zeros(dev, DType::F32, &[c.n_gdn() * 3 * c.conv_ch()])?;
        let ssm = Tensor::zeros(dev, DType::F32, &[c.n_gdn() * c.n_vhead * 128 * 128])?;
        let kv = Tensor::new(dev, DType::BF16, &[c.n_attn() * 2 * cap * c.n_head_kv * c.head_dim])?;
        Ok(SeqState { conv, ssm, kv, cap, len: 0 })
    }
    fn conv_ptr(&self, c: &Config, i: usize) -> u64 {
        self.conv.ptr + (i * 3 * c.conv_ch() * 4) as u64
    }
    fn ssm_ptr(&self, c: &Config, i: usize) -> u64 {
        self.ssm.ptr + (i * c.n_vhead * 128 * 128 * 4) as u64
    }
    fn kv_ptr(&self, c: &Config, i: usize, which: usize) -> u64 {
        let per = self.cap * c.n_head_kv * c.head_dim;
        self.kv.ptr + (((i * 2 + which) * per) * 2) as u64
    }
}

/// Scratch buffers sized for the largest chunk.
struct Scratch {
    rows: usize,
    x: Tensor,     // f32 [R, E] residual
    xb: Tensor,    // bf16 [R, max(E, F, d_inner)] GEMM operand
    big1: Tensor,  // f32 [R, max(conv_ch, 2*H*256)]
    big2: Tensor,  // f32 [R, conv_ch]
    zb: Tensor,    // f32 [R, max(d_inner, 2*HK*256)]
    o: Tensor,     // f32 [R, max(d_inner, H*256)]
    qf: Tensor,    // f32 [R, H*256]
    ab: Tensor,    // f32 [2, R, n_vhead]
    gb: Tensor,    // bf16 [R, F]
    ub: Tensor,    // bf16 [R, F]
    rope: Tensor,  // f32 [R, 64]
    deq: Tensor,   // bf16 [max N*K]
    // decode-only (<= MAX_BATCH rows)
    xf: Tensor,  // f32 [8, F]
    gf: Tensor,  // f32 [8, F]
    uf: Tensor,  // f32 [8, F]
    xq: Tensor,  // i8 [8, F]
    xd: Tensor,  // f32 [8, F/32]
    lg: Tensor,  // f32 [8, vocab]
    pm: Tensor,  // f32 [8, nsplit, H]
    pl: Tensor,
    po: Tensor,  // f32 [8, nsplit, H, 256]
    nsplit: usize,
    blob: Tensor, // per-step tables
}

pub struct TextModel {
    pub dev: Arc<Device>,
    pub cfg: Config,
    g: Arc<Gguf>,
    layers: Vec<Layer>,
    out_norm: Tensor,
    output: Linear,
    embd_ty: GgmlType,
    sc: std::cell::RefCell<Option<Scratch>>,
    pub max_ctx: usize,
    pub vram_bytes: usize,
}

fn f32_tensor(dev: &Device, g: &Gguf, name: &str) -> Result<Tensor> {
    let v = g.f32(name)?;
    let n = v.len();
    Tensor::from_f32(dev, &v, &[n])
}

/// Where a linear's input comes from.
enum Src<'a> {
    Gemv(Act<'a>),
    Gemm(Tensor),
}

impl TextModel {
    pub fn load(dev: Arc<Device>, g: Arc<Gguf>, max_ctx: usize, progress: &dyn Fn(usize, usize)) -> Result<TextModel> {
        let cfg = Config::from_gguf(&g)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let (mut ai, mut gi) = (0, 0);
        for l in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{l}.{s}");
            let lin = |s: &str| Linear::load(&dev, &g, &p(s)).with_context(|| format!("loading blk.{l}.{s}"));
            let mixer = if cfg.is_attn(l) {
                let m = Mixer::Attn { q: lin("attn_q.weight")?, k: lin("attn_k.weight")?, v: lin("attn_v.weight")?, o: lin("attn_output.weight")?, q_norm: f32_tensor(&dev, &g, &p("attn_q_norm.weight"))?, k_norm: f32_tensor(&dev, &g, &p("attn_k_norm.weight"))?, idx: ai };
                ai += 1;
                m
            } else {
                if g.has(&p("attn_qkv_gate.weight")) || g.has(&p("ssm_beta_alpha.weight")) {
                    bail!("{}: fused DeltaNet projections (attn_qkv_gate / ssm_beta_alpha) are not supported; use a GGUF with separate attn_qkv / attn_gate / ssm_beta / ssm_alpha", g.path.display());
                }
                let m = Mixer::Gdn {
                    qkv: lin("attn_qkv.weight")?,
                    z: lin("attn_gate.weight")?,
                    beta: lin("ssm_beta.weight")?,
                    alpha: lin("ssm_alpha.weight")?,
                    conv: f32_tensor(&dev, &g, &p("ssm_conv1d.weight"))?,
                    dt: f32_tensor(&dev, &g, &p("ssm_dt.bias"))?,
                    a: f32_tensor(&dev, &g, &p("ssm_a"))?,
                    norm: f32_tensor(&dev, &g, &p("ssm_norm.weight"))?,
                    out: lin("ssm_out.weight")?,
                    idx: gi,
                };
                gi += 1;
                m
            };
            layers.push(Layer {
                attn_norm: f32_tensor(&dev, &g, &p("attn_norm.weight"))?,
                post_norm: f32_tensor(&dev, &g, &p("post_attention_norm.weight"))?,
                mixer,
                gate: lin("ffn_gate.weight")?,
                up: lin("ffn_up.weight")?,
                down: lin("ffn_down.weight")?,
            });
            progress(l + 1, cfg.n_layer);
        }
        let out_norm = f32_tensor(&dev, &g, "output_norm.weight")?;
        let output = if g.has("output.weight") { Linear::load(&dev, &g, "output.weight")? } else { Linear::load(&dev, &g, "token_embd.weight")? };
        let embd_ty = g.tensor("token_embd.weight")?.ty;
        embd_ty.block()?;
        let mut vram = output.bytes();
        for l in &layers {
            vram += l.gate.bytes() + l.up.bytes() + l.down.bytes();
            vram += match &l.mixer {
                Mixer::Attn { q, k, v, o, .. } => q.bytes() + k.bytes() + v.bytes() + o.bytes(),
                Mixer::Gdn { qkv, z, beta, alpha, out, .. } => qkv.bytes() + z.bytes() + beta.bytes() + alpha.bytes() + out.bytes(),
            };
        }
        let max_ctx = max_ctx.min(cfg.ctx_train).max(256);
        Ok(TextModel { dev, cfg, g, layers, out_norm, output, embd_ty, sc: std::cell::RefCell::new(None), max_ctx, vram_bytes: vram })
    }

    fn scratch(&self) -> Result<std::cell::RefMut<'_, Scratch>> {
        let mut sc = self.sc.borrow_mut();
        if sc.is_none() {
            let c = &self.cfg;
            let d = &self.dev;
            let r = PREFILL_CHUNK;
            let (e, f) = (c.n_embd, c.n_ff);
            let h256 = c.n_head * c.head_dim;
            let mut maxnk = self.output.n * 0;
            for l in &self.layers {
                maxnk = maxnk.max(l.gate.n * l.gate.k).max(l.up.n * l.up.k).max(l.down.n * l.down.k);
                match &l.mixer {
                    Mixer::Attn { q, k, v, o, .. } => maxnk = maxnk.max(q.n * q.k).max(k.n * k.k).max(v.n * v.k).max(o.n * o.k),
                    Mixer::Gdn { qkv, z, beta, alpha, out, .. } => maxnk = maxnk.max(qkv.n * qkv.k).max(z.n * z.k).max(beta.n * beta.k).max(alpha.n * alpha.k).max(out.n * out.k),
                }
            }
            let nsplit = (self.max_ctx + 255) / 256;
            *sc = Some(Scratch {
                rows: r,
                x: Tensor::new(d, DType::F32, &[r, e])?,
                xb: Tensor::new(d, DType::BF16, &[r * e.max(f).max(c.d_inner).max(h256)])?,
                big1: Tensor::new(d, DType::F32, &[r * c.conv_ch().max(2 * h256)])?,
                big2: Tensor::new(d, DType::F32, &[r * c.conv_ch()])?,
                zb: Tensor::new(d, DType::F32, &[r * c.d_inner.max(2 * c.n_head_kv * c.head_dim)])?,
                o: Tensor::new(d, DType::F32, &[r * c.d_inner.max(h256)])?,
                qf: Tensor::new(d, DType::F32, &[r * h256])?,
                ab: Tensor::new(d, DType::F32, &[2 * r * c.n_vhead])?,
                gb: Tensor::new(d, DType::BF16, &[r * f])?,
                ub: Tensor::new(d, DType::BF16, &[r * f])?,
                rope: Tensor::new(d, DType::F32, &[r * 64])?,
                deq: Tensor::new(d, DType::BF16, &[maxnk])?,
                xf: Tensor::new(d, DType::F32, &[MAX_BATCH * f.max(e)])?,
                gf: Tensor::new(d, DType::F32, &[MAX_BATCH * f])?,
                uf: Tensor::new(d, DType::F32, &[MAX_BATCH * f])?,
                xq: Tensor::new(d, DType::I8, &[MAX_BATCH * f.max(e)])?,
                xd: Tensor::new(d, DType::F32, &[MAX_BATCH * f.max(e) / 32])?,
                lg: Tensor::new(d, DType::F32, &[MAX_BATCH * self.output.n])?,
                pm: Tensor::new(d, DType::F32, &[MAX_BATCH * nsplit * c.n_head])?,
                pl: Tensor::new(d, DType::F32, &[MAX_BATCH * nsplit * c.n_head])?,
                po: Tensor::new(d, DType::F32, &[MAX_BATCH * nsplit * c.n_head * 256])?,
                nsplit,
                blob: Tensor::new(d, DType::U8, &[1 << 20])?,
            });
        }
        Ok(std::cell::RefMut::map(sc, |s| s.as_mut().unwrap()))
    }

    /// Free the scratch buffers (worker mode between requests keeps the weights only).
    pub fn drop_scratch(&self) {
        *self.sc.borrow_mut() = None;
    }

    /// Bytes of scratch the first forward allocates (an estimate for planning).
    pub fn scratch_bytes(&self) -> usize {
        let c = &self.cfg;
        let r = PREFILL_CHUNK;
        let h256 = c.n_head * c.head_dim;
        r * (c.n_embd * 4 + c.n_ff.max(h256) * 2 * 3 + c.conv_ch().max(2 * h256) * 4 + c.conv_ch() * 4 + c.d_inner * 4 * 2 + h256 * 4) + c.n_ff * c.n_embd * 2
    }

    /// The embedding row of a token (f32 [n_embd]).
    pub fn embed(&self, tok: u32, out: &mut [f32]) -> Result<()> {
        let t = self.g.tensor("token_embd.weight")?;
        let row = quant::dequant_row(self.embd_ty, self.g.data(t), tok as usize, self.cfg.n_embd)?;
        out.copy_from_slice(&row);
        Ok(())
    }

    /// Interleaved M-RoPE (cos, sin) for 32 frequency pairs of one position (t, h, w).
    pub fn rope_row(&self, p: [i64; 3], out: &mut [f32]) {
        let c = &self.cfg;
        let half = c.rope_dims / 2;
        let [s0, s1, s2] = c.sections;
        for f in 0..half {
            let axis = if f % 3 == 1 && f < 3 * s1 {
                1
            } else if f % 3 == 2 && f < 3 * s2 {
                2
            } else if f % 3 == 0 && f < 3 * s0 {
                0
            } else {
                3
            };
            let pos = if axis == 3 { 0.0 } else { p[axis] as f32 };
            let inv = c.rope_base.powf(-2.0 * f as f32 / c.rope_dims as f32);
            let a = pos * inv;
            out[f * 2] = a.cos();
            out[f * 2 + 1] = a.sin();
        }
    }

    /// Run rows through the decoder.
    ///   prefill: `seqs` = [one sequence], `x` = T embedding rows (host f32 [T, E]) at cache positions len..len+T
    ///   decode:  `seqs` = S sequences, `x` = one row each
    /// `pos` holds the M-RoPE position (t, h, w) of every row. Returns the logits of the last row of every sequence
    /// when `want_logits` (f32 [S, vocab]).
    pub fn forward(&self, x_host: &[f32], pos: &[[i64; 3]], seqs: &mut [&mut SeqState], prefill: bool, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let c = self.cfg.clone();
        let dev = &*self.dev;
        let e = c.n_embd;
        let rows = x_host.len() / e;
        let s_n = seqs.len();
        ensure!(rows > 0 && pos.len() == rows);
        let t_n = if prefill { rows } else { 1 };
        if prefill {
            ensure!(s_n == 1, "prefill takes one sequence");
        } else {
            ensure!(rows == s_n && s_n <= MAX_BATCH, "decode takes one row per sequence (<= {MAX_BATCH})");
        }
        for s in seqs.iter() {
            ensure!(s.len + t_n <= s.cap, "sequence exceeds its context ({} + {t_n} > {})", s.len, s.cap);
        }
        let sc = self.scratch()?;
        ensure!(rows <= sc.rows, "chunk of {rows} rows exceeds {}", sc.rows);
        let gemv = rows <= MAX_BATCH;

        // ---- per-step tables, one upload
        let mut blob: Vec<u8> = Vec::new();
        let sect = |blob: &mut Vec<u8>, bytes: &[u8]| -> u64 {
            while blob.len() % 256 != 0 {
                blob.push(0);
            }
            let off = blob.len();
            blob.extend_from_slice(bytes);
            off as u64
        };
        let as_bytes = |v: &[u64]| -> Vec<u8> { v.iter().flat_map(|p| p.to_le_bytes()).collect() };
        let as_bytes32 = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|p| p.to_le_bytes()).collect() };
        let (n_attn, n_gdn) = (c.n_attn(), c.n_gdn());
        let mut conv_p = Vec::with_capacity(n_gdn * s_n);
        let mut ssm_p = Vec::with_capacity(n_gdn * s_n);
        for i in 0..n_gdn {
            for s in seqs.iter() {
                conv_p.push(s.conv_ptr(&c, i));
                ssm_p.push(s.ssm_ptr(&c, i));
            }
        }
        let mut k_p = Vec::with_capacity(n_attn * s_n);
        let mut v_p = Vec::with_capacity(n_attn * s_n);
        for i in 0..n_attn {
            for s in seqs.iter() {
                k_p.push(s.kv_ptr(&c, i, 0));
                v_p.push(s.kv_ptr(&c, i, 1));
            }
        }
        let row_seq: Vec<i32> = (0..rows).map(|r| if prefill { 0 } else { r as i32 }).collect();
        let row_pos: Vec<i32> = (0..rows).map(|r| if prefill { (seqs[0].len + r) as i32 } else { seqs[r].len as i32 }).collect();
        let lens: Vec<i32> = seqs.iter().map(|s| (s.len + t_n) as i32).collect();
        let gather: Vec<i32> = if prefill { vec![(rows - 1) as i32] } else { (0..rows as i32).collect() };
        let o_conv = sect(&mut blob, &as_bytes(&conv_p));
        let o_ssm = sect(&mut blob, &as_bytes(&ssm_p));
        let o_k = sect(&mut blob, &as_bytes(&k_p));
        let o_v = sect(&mut blob, &as_bytes(&v_p));
        let o_rseq = sect(&mut blob, &as_bytes32(&row_seq));
        let o_rpos = sect(&mut blob, &as_bytes32(&row_pos));
        let o_len = sect(&mut blob, &as_bytes32(&lens));
        let o_gat = sect(&mut blob, &as_bytes32(&gather));
        ensure!(blob.len() <= sc.blob.numel(), "step tables too large");
        dev.htod_at(sc.blob.ptr, &blob)?;
        let bp = |o: u64| sc.blob.ptr + o;
        // rope + residual
        let mut rope = vec![0f32; rows * 64];
        for (r, p) in pos.iter().enumerate() {
            self.rope_row(*p, &mut rope[r * 64..r * 64 + 64]);
        }
        dev.htod_at(sc.rope.ptr, bytes_of(&rope))?;
        dev.htod_at(sc.x.ptr, bytes_of(x_host))?;
        let x = view(&sc.x, DType::F32, &[rows, e]);

        for layer in &self.layers {
            // ---------------- mixer
            let a = self.norm_act(&sc, &x, rows, &layer.attn_norm, gemv)?;
            match &layer.mixer {
                Mixer::Gdn { qkv, z, beta, alpha, conv, dt, a: aa, norm, out, idx } => {
                    let cc = c.conv_ch();
                    let hv = c.n_vhead;
                    let qkv_o = view(&sc.big1, DType::F32, &[rows, cc]);
                    let z_o = view(&sc.zb, DType::F32, &[rows, c.d_inner]);
                    let al = view(&sc.ab, DType::F32, &[rows, hv]);
                    let be = view_at(&sc.ab, rows * hv, DType::F32, &[rows, hv]);
                    self.lin(&sc, qkv, &a, rows, &qkv_o, false)?;
                    self.lin(&sc, z, &a, rows, &z_o, false)?;
                    self.lin(&sc, alpha, &a, rows, &al, false)?;
                    self.lin(&sc, beta, &a, rows, &be, false)?;
                    let cv = view(&sc.big2, DType::F32, &[rows, cc]);
                    let states_conv = bp(o_conv) + (idx * s_n * 8) as u64;
                    dev.launch_n("k_conv_silu", s_n * cc, &[Arg::Ptr(qkv_o.ptr), Arg::I32(cc as i32), Arg::Ptr(conv.ptr), Arg::Ptr(states_conv), Arg::Ptr(cv.ptr), Arg::I32(s_n as i32), Arg::I32(t_n as i32), Arg::I32(cc as i32)])?;
                    let qk_heads = 2 * c.n_kgroup;
                    dev.launch_n("k_l2norm_heads", rows * qk_heads * 32, &[Arg::Ptr(cv.ptr), Arg::I32(cc as i32), Arg::I32(rows as i32), Arg::I32(qk_heads as i32), Arg::F32(c.eps)])?;
                    dev.launch_n("k_gdn_gates", rows * hv, &[Arg::Ptr(al.ptr), Arg::Ptr(be.ptr), Arg::Ptr(dt.ptr), Arg::Ptr(aa.ptr), Arg::I32(rows as i32), Arg::I32(hv as i32)])?;
                    let o = view(&sc.o, DType::F32, &[rows, c.d_inner]);
                    let states_ssm = bp(o_ssm) + (idx * s_n * 8) as u64;
                    PROF.time(dev, "gdn", || dev.launch(
                        "k_gdn",
                        (hv as u32, s_n as u32, 1),
                        (128, 1, 1),
                        0,
                        &[Arg::Ptr(cv.ptr), Arg::I32(cc as i32), Arg::Ptr(al.ptr), Arg::Ptr(be.ptr), Arg::Ptr(states_ssm), Arg::Ptr(o.ptr), Arg::I32(t_n as i32), Arg::I32(hv as i32), Arg::I32(c.n_kgroup as i32), Arg::F32(1.0 / (c.d_state as f32).sqrt())],
                    ))?;
                    // gated norm straight into the next GEMM's operand
                    let (yf, yb) = if gemv { (sc.xf.ptr, 0) } else { (0, sc.xb.ptr) };
                    dev.launch_n("k_gated_rmsnorm", rows * hv * 32, &[Arg::Ptr(o.ptr), Arg::Ptr(z_o.ptr), Arg::I32(c.d_inner as i32), Arg::Ptr(norm.ptr), Arg::Ptr(yf), Arg::Ptr(yb), Arg::I32(rows as i32), Arg::I32(hv as i32), Arg::F32(c.eps)])?;
                    let y = self.act_ready(&sc, rows, c.d_inner, gemv)?;
                    self.lin(&sc, out, &y, rows, &x, true)?;
                }
                Mixer::Attn { q, k, v, o, q_norm, k_norm, idx } => {
                    let (hq, hk, hd) = (c.n_head, c.n_head_kv, c.head_dim);
                    let qg = view(&sc.big1, DType::F32, &[rows, 2 * hq * hd]);
                    let kk = view(&sc.zb, DType::F32, &[rows, hk * hd]);
                    let vv = view_at(&sc.zb, rows * hk * hd, DType::F32, &[rows, hk * hd]);
                    self.lin(&sc, q, &a, rows, &qg, false)?;
                    self.lin(&sc, k, &a, rows, &kk, false)?;
                    self.lin(&sc, v, &a, rows, &vv, false)?;
                    let kps = bp(o_k) + (idx * s_n * 8) as u64;
                    let vps = bp(o_v) + (idx * s_n * 8) as u64;
                    dev.launch_n(
                        "k_qk_norm_rope",
                        rows * (hq + hk) * 32,
                        &[Arg::Ptr(qg.ptr), Arg::Ptr(kk.ptr), Arg::Ptr(vv.ptr), Arg::Ptr(q_norm.ptr), Arg::Ptr(k_norm.ptr), Arg::Ptr(sc.rope.ptr), Arg::Ptr(sc.qf.ptr), Arg::Ptr(kps), Arg::Ptr(vps), Arg::Ptr(bp(o_rseq)), Arg::Ptr(bp(o_rpos)), Arg::I32(rows as i32), Arg::I32(hq as i32), Arg::I32(hk as i32), Arg::F32(c.eps)],
                    )?;
                    let ao = view(&sc.o, DType::F32, &[rows, hq * hd]);
                    let scale_log2 = (1.0 / (hd as f32).sqrt()) * std::f32::consts::LOG2_E;
                    if prefill {
                        let s = &seqs[0];
                        dev.launch(
                            "k_attn_prefill",
                            (((rows + 31) / 32) as u32, hq as u32, 1),
                            (256, 1, 1),
                            ops::ATTN_PREFILL_SMEM,
                            &[Arg::Ptr(sc.qf.ptr), Arg::Ptr(s.kv_ptr(&c, *idx, 0)), Arg::Ptr(s.kv_ptr(&c, *idx, 1)), Arg::Ptr(ao.ptr), Arg::I32(rows as i32), Arg::I32(s.len as i32), Arg::I32(hq as i32), Arg::I32(hk as i32), Arg::F32(scale_log2)],
                        )?;
                    } else {
                        let maxlen = lens.iter().copied().max().unwrap_or(1) as usize;
                        let ns = (maxlen + 255) / 256;
                        ensure!(ns <= sc.nsplit, "context {maxlen} exceeds the decode split buffers");
                        dev.launch(
                            "k_attn_decode",
                            (ns as u32, hk as u32, s_n as u32),
                            (128, 1, 1),
                            0,
                            &[Arg::Ptr(sc.qf.ptr), Arg::Ptr(kps), Arg::Ptr(vps), Arg::Ptr(bp(o_len)), Arg::Ptr(sc.pm.ptr), Arg::Ptr(sc.pl.ptr), Arg::Ptr(sc.po.ptr), Arg::I32(ns as i32), Arg::I32(hq as i32), Arg::I32(hk as i32), Arg::F32(scale_log2)],
                        )?;
                        dev.launch("k_attn_decode_combine", (hq as u32, s_n as u32, 1), (256, 1, 1), 0, &[Arg::Ptr(sc.pm.ptr), Arg::Ptr(sc.pl.ptr), Arg::Ptr(sc.po.ptr), Arg::Ptr(bp(o_len)), Arg::Ptr(ao.ptr), Arg::I32(ns as i32), Arg::I32(hq as i32)])?;
                    }
                    dev.launch_n("k_attn_gate", rows * hq * hd, &[Arg::Ptr(ao.ptr), Arg::Ptr(qg.ptr), Arg::I32(rows as i32), Arg::I32(hq as i32)])?;
                    let y = self.from_f32(&sc, &ao, rows, hq * hd, gemv)?;
                    self.lin(&sc, o, &y, rows, &x, true)?;
                }
            }
            // ---------------- FFN
            let a = self.norm_act(&sc, &x, rows, &layer.post_norm, gemv)?;
            let f = c.n_ff;
            if gemv {
                let g = view(&sc.gf, DType::F32, &[rows, f]);
                let u = view(&sc.uf, DType::F32, &[rows, f]);
                self.lin(&sc, &layer.gate, &a, rows, &g, false)?;
                self.lin(&sc, &layer.up, &a, rows, &u, false)?;
                ops::silu_mul_f32(dev, &g, &u, &sc.xf, rows * f)?;
                let h = self.act_ready(&sc, rows, f, true)?;
                self.lin(&sc, &layer.down, &h, rows, &x, true)?;
            } else {
                let g = view(&sc.gb, DType::BF16, &[rows, f]);
                let u = view(&sc.ub, DType::BF16, &[rows, f]);
                self.lin(&sc, &layer.gate, &a, rows, &g, false)?;
                self.lin(&sc, &layer.up, &a, rows, &u, false)?;
                let h = view(&sc.xb, DType::BF16, &[rows, f]);
                ops::silu_mul_bf16(dev, &g, &u, &h, rows * f)?;
                self.lin(&sc, &layer.down, &Src::Gemm(h.clone()), rows, &x, true)?;
            }
        }
        for s in seqs.iter_mut() {
            s.len += t_n;
        }
        if !want_logits {
            return Ok(None);
        }
        // ---- LM head on the last row of each sequence
        let nl = gather.len();
        let last = view(&sc.big2, DType::F32, &[nl, e]);
        dev.launch_n("k_gather_rows", nl * e, &[Arg::Ptr(x.ptr), Arg::Ptr(bp(o_gat)), Arg::Ptr(last.ptr), Arg::I32(nl as i32), Arg::I32(e as i32)])?;
        let a = self.norm_act(&sc, &last, nl, &self.out_norm, true)?;
        let lg = view(&sc.lg, DType::F32, &[nl, self.output.n]);
        self.lin(&sc, &self.output, &a, nl, &lg, false)?;
        Ok(Some(dev.dtoh_at::<f32>(lg.ptr, nl * self.output.n)?))
    }

    /// RMSNorm of x into the operand a linear takes.
    fn norm_act<'a>(&self, sc: &'a Scratch, x: &Tensor, rows: usize, w: &Tensor, gemv: bool) -> Result<Src<'a>> {
        let n = *x.shape.last().unwrap();
        if gemv {
            ops::rmsnorm(&self.dev, x, rows, n, w, self.cfg.eps, Some(&sc.xf), None)?;
            self.act_ready(sc, rows, n, true)
        } else {
            let xb = view(&sc.xb, DType::BF16, &[rows, n]);
            ops::rmsnorm(&self.dev, x, rows, n, w, self.cfg.eps, None, Some(&xb))?;
            Ok(Src::Gemm(xb))
        }
    }

    /// The operand for rows already written to xf (decode: q8-quantize it) or xb (prefill).
    fn act_ready<'a>(&self, sc: &'a Scratch, rows: usize, n: usize, gemv: bool) -> Result<Src<'a>> {
        if gemv {
            quant::quant_q8(&self.dev, &sc.xf, rows, n, &sc.xq, &sc.xd)?;
            Ok(Src::Gemv(Act { x: &sc.xf, xq: &sc.xq, xd: &sc.xd }))
        } else {
            Ok(Src::Gemm(view(&sc.xb, DType::BF16, &[rows, n])))
        }
    }

    /// Operand from an f32 [rows, n] buffer.
    fn from_f32<'a>(&self, sc: &'a Scratch, src: &Tensor, rows: usize, n: usize, gemv: bool) -> Result<Src<'a>> {
        if gemv {
            self.dev.dtod(sc.xf.ptr, src.ptr, rows * n * 4)?;
        } else {
            ops::to_bf16(&self.dev, &view(src, DType::F32, &[rows * n]), &view(&sc.xb, DType::BF16, &[rows * n]))?;
        }
        self.act_ready(sc, rows, n, gemv)
    }

    fn lin(&self, sc: &Scratch, l: &Linear, src: &Src, rows: usize, out: &Tensor, add: bool) -> Result<()> {
        let cat = match (src, l.n) {
            (Src::Gemv(_), n) if n == self.cfg.vocab => "lm_head",
            (Src::Gemv(_), _) => "gemv",
            (Src::Gemm(_), _) => "gemm",
        };
        PROF.time(&self.dev, cat, || self.lin_inner(sc, l, src, rows, out, add))
    }

    fn lin_inner(&self, sc: &Scratch, l: &Linear, src: &Src, rows: usize, out: &Tensor, add: bool) -> Result<()> {
        match src {
            Src::Gemv(a) => l.gemv(&self.dev, a, rows, out, add),
            Src::Gemm(xb) => {
                let a = view(xb, DType::BF16, &[rows, l.k]);
                let o = view(out, out.dtype, &[rows, l.n]);
                let w = PROF.time_val(&self.dev, "dequant", || l.bf16(&self.dev, &sc.deq))?;
                PROF.time(&self.dev, "gemm_mma", || ops::gemm_bf16(&self.dev, &a, &w, None, ops::Act::None, if add { Epi::AddRes(&o) } else { Epi::Store }, &o))
            }
        }
    }
}

pub fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
