//! Anima (CircleStone Labs / Comfy Org): Cosmos-Predict2 2B (MiniTrainDIT: 28 blocks, dim 2048, 16 heads of 128) with an
//! LLM adapter, from the ComfyUI single files (`net.blocks.N.*`, `net.llm_adapter.*`). Follows ComfyUI's `MiniTrainDIT`
//! and diffusers' `AnimaTextConditioner`:
//!   - text: Qwen3-0.6B `last_hidden_state` of the raw prompt (no chat template) is the *source*; the prompt's T5 token
//!     ids are embedded as the *target*; 6 adapter blocks (self-attention over the target, cross-attention to the source,
//!     GELU MLP; RMSNorms, head dim 64, rope on both sides) -> out_proj -> RMSNorm, zero-padded to 512 tokens. The DiT
//!     cross-attends to those 1024-wide tokens; their keys and values are computed once per prompt for all blocks.
//!   - image: 16-channel Qwen-Image latents in 2x2 patches (input channel c*4 + dy*2 + dx; the zero padding-mask channel
//!     dropped); output rows reordered from Cosmos' (dy, dx, c) to the same layout.
//!   - time: t = sigma; sinusoid(2048) is both the embedding (after an RMSNorm) and the input of an AdaLN-LoRA MLP whose
//!     6144-wide output is added to every modulation (each block: SiLU -> Linear(2048 -> 256) -> Linear(256 -> 6144) for
//!     self-attention, cross-attention and MLP -> shift, scale, gate).
//!   - block: LayerNorm (no affine, eps 1e-6) then `* (1 + scale) + shift`; self-attention with q/k RMSNorm and 3D rope;
//!     cross-attention (no rope); GELU MLP 8192; gated residuals.
//!   - rope: per head, frequencies [t: 22 | h: 21 | w: 21] over split halves, h/w theta 10000 * 4^(42/40) (extrapolation
//!     ratio 4), positions (0, row, col), computed in f32.
use crate::cuda::Device;
use crate::linear::{self, Lin, Want};
use crate::model::{self, Dense};
use crate::ops::{self, Act, AttnView, Epi, ModNorm, QkNorm, RmsMode, Rope};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const LATENT: usize = 64;
const EPS: f32 = 1e-6;
const HD: usize = 128;
pub const CTX_LEN: usize = 512;
const CTX_DIM: usize = 1024;
const AD_HEADS: usize = 16;
const AD_HD: usize = 64;

struct AdaLn {
    a: Tensor,
    b: Tensor,
}
struct Block {
    ada: [AdaLn; 3],
    qkv: Lin,
    sq: Tensor,
    sk: Tensor,
    so: Lin,
    cq: Lin,
    cqn: Tensor,
    ck: Tensor,
    cv: Tensor,
    ckn: Tensor,
    co: Lin,
    l1: Lin,
    l2: Lin,
}
struct AdBlock {
    n_sa: Tensor,
    sa_qkv: Tensor,
    sa_qn: Tensor,
    sa_kn: Tensor,
    sa_o: Tensor,
    n_ca: Tensor,
    ca_q: Tensor,
    ca_kv: Tensor,
    ca_qn: Tensor,
    ca_kn: Tensor,
    ca_o: Tensor,
    n_mlp: Tensor,
    mlp1: Dense,
    mlp2: Dense,
}

pub struct Anima {
    dev: Arc<Device>,
    pub int8: bool,
    dim: usize,
    heads: usize,
    x_in: Tensor,
    t1: Tensor,
    t2: Tensor,
    t_norm: Tensor,
    blocks: Vec<Block>,
    fin_ada: AdaLn,
    fin_linear: Tensor,
    ad_embed: Vec<u8>,
    ad_embed_dtype: String,
    ad_blocks: Vec<AdBlock>,
    ad_out: Dense,
    ad_norm: Tensor,
}

pub struct Run {
    /// per block: cross-attention keys (normed) and values [512, dim]
    kv: Vec<(Tensor, Tensor)>,
    rope: Tensor,
    pub h: usize,
    pub w: usize,
}

pub struct Mod {
    /// per block: [self, cross, mlp] x (shift, scale, gate)
    blocks: Vec<Vec<Tensor>>,
    fin_shift: Tensor,
    fin_scale: Tensor,
}

impl Anima {
    pub fn load(dev: Arc<Device>, path: &Path, int8: bool) -> Result<Anima> {
        let st = SafeTensors::open(path)?;
        let p0 = crate::arch::prefix_of(&st, "net.llm_adapter.embed.weight").map(|p| format!("{p}net.")).or_else(|| crate::arch::prefix_of(&st, "llm_adapter.embed.weight").map(|p| p.to_string())).with_context(|| format!("{} is not an Anima model", path.display()))?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let n = |s: &str| format!("{p0}{s}");
        let w = model::want(int8);
        // x_embedder: [dim, 68] over (c, dy, dx) with the padding-mask channel last (always zero): keep the first 64 columns
        let xw = st.f32s(&n("x_embedder.proj.1.weight"))?;
        let xs = st.info(&n("x_embedder.proj.1.weight"))?.shape.clone();
        let dim = xs[0];
        ensure!(xs[1] == LATENT + 4, "Anima: x_embedder takes {} inputs, expected {}", xs[1], LATENT + 4);
        let bits: Vec<u16> = (0..dim).flat_map(|r| xw[r * xs[1]..r * xs[1] + LATENT].iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>()).collect();
        let x_in = Tensor::from_bf16(&dev, &bits, &[dim, LATENT])?;
        let ada = |l: &mut Loader, p: &str| -> Result<AdaLn> { Ok(AdaLn { a: linear::weight_bf16(l, &format!("{p}.1"))?, b: linear::weight_bf16(l, &format!("{p}.2"))? }) };
        let mut blocks = Vec::new();
        while st.has(&n(&format!("blocks.{}.mlp.layer1.weight", blocks.len()))) {
            let b = n(&format!("blocks.{}", blocks.len()));
            let ck = linear::weight_bf16(&mut l, &format!("{b}.cross_attn.k_proj"))?;
            let cv = linear::weight_bf16(&mut l, &format!("{b}.cross_attn.v_proj"))?;
            blocks.push(Block {
                ada: [ada(&mut l, &format!("{b}.adaln_modulation_self_attn"))?, ada(&mut l, &format!("{b}.adaln_modulation_cross_attn"))?, ada(&mut l, &format!("{b}.adaln_modulation_mlp"))?],
                qkv: linear::lin_cat(&mut l, &[&format!("{b}.self_attn.q_proj"), &format!("{b}.self_attn.k_proj"), &format!("{b}.self_attn.v_proj")], w)?,
                sq: linear::small_bf16(&mut l, &format!("{b}.self_attn.q_norm.weight"))?,
                sk: linear::small_bf16(&mut l, &format!("{b}.self_attn.k_norm.weight"))?,
                so: linear::lin(&mut l, &format!("{b}.self_attn.output_proj"), w)?,
                cq: linear::lin(&mut l, &format!("{b}.cross_attn.q_proj"), w)?,
                cqn: linear::small_bf16(&mut l, &format!("{b}.cross_attn.q_norm.weight"))?,
                ck,
                cv,
                ckn: linear::small_bf16(&mut l, &format!("{b}.cross_attn.k_norm.weight"))?,
                co: linear::lin(&mut l, &format!("{b}.cross_attn.output_proj"), w)?,
                l1: linear::lin(&mut l, &format!("{b}.mlp.layer1"), w)?,
                l2: linear::lin(&mut l, &format!("{b}.mlp.layer2"), w)?,
            });
        }
        ensure!(!blocks.is_empty(), "Anima: no blocks");
        // final linear rows are (dy, dx, c): reorder to (c, dy, dx)
        let fw = st.f32s(&n("final_layer.linear.weight"))?;
        ensure!(fw.len() == LATENT * dim, "Anima: unexpected final layer shape");
        let mut fb = vec![0u16; LATENT * dim];
        for c in 0..16 {
            for pq in 0..4 {
                let src = pq * 16 + c;
                let dst = c * 4 + pq;
                for k in 0..dim {
                    fb[dst * dim + k] = bf16_bits(fw[src * dim + k]);
                }
            }
        }
        let fin_linear = Tensor::from_bf16(&dev, &fb, &[LATENT, dim])?;
        // adapter
        let a = |s: &str| n(&format!("llm_adapter.{s}"));
        let mut ad_blocks = Vec::new();
        while st.has(&a(&format!("blocks.{}.norm_mlp.weight", ad_blocks.len()))) {
            let b = a(&format!("blocks.{}", ad_blocks.len()));
            let cat = |l: &mut Loader, names: &[String]| -> Result<Tensor> {
                match linear::lin_cat(l, &names.iter().map(|s| s.as_str()).collect::<Vec<_>>(), Want::Bf16)? {
                    Lin::Bf16(t) => Ok(t),
                    Lin::I8(_) => anyhow::bail!("Anima: int8 adapter weights are not supported"),
                }
            };
            ad_blocks.push(AdBlock {
                n_sa: linear::small_bf16(&mut l, &format!("{b}.norm_self_attn.weight"))?,
                sa_qkv: cat(&mut l, &[format!("{b}.self_attn.q_proj"), format!("{b}.self_attn.k_proj"), format!("{b}.self_attn.v_proj")])?,
                sa_qn: linear::small_bf16(&mut l, &format!("{b}.self_attn.q_norm.weight"))?,
                sa_kn: linear::small_bf16(&mut l, &format!("{b}.self_attn.k_norm.weight"))?,
                sa_o: linear::weight_bf16(&mut l, &format!("{b}.self_attn.o_proj"))?,
                n_ca: linear::small_bf16(&mut l, &format!("{b}.norm_cross_attn.weight"))?,
                ca_q: linear::weight_bf16(&mut l, &format!("{b}.cross_attn.q_proj"))?,
                ca_kv: cat(&mut l, &[format!("{b}.cross_attn.k_proj"), format!("{b}.cross_attn.v_proj")])?,
                ca_qn: linear::small_bf16(&mut l, &format!("{b}.cross_attn.q_norm.weight"))?,
                ca_kn: linear::small_bf16(&mut l, &format!("{b}.cross_attn.k_norm.weight"))?,
                ca_o: linear::weight_bf16(&mut l, &format!("{b}.cross_attn.o_proj"))?,
                n_mlp: linear::small_bf16(&mut l, &format!("{b}.norm_mlp.weight"))?,
                mlp1: Dense::load(&mut l, &format!("{b}.mlp.0"))?,
                mlp2: Dense::load(&mut l, &format!("{b}.mlp.2"))?,
            });
        }
        ensure!(!st.has(&a("in_proj.weight")), "Anima: adapters with an input projection are not supported");
        let emb_name = a("embed.weight");
        let an = Anima {
            int8,
            dim,
            heads: dim / HD,
            x_in,
            t1: linear::weight_bf16(&mut l, &n("t_embedder.1.linear_1"))?,
            t2: linear::weight_bf16(&mut l, &n("t_embedder.1.linear_2"))?,
            t_norm: linear::small_bf16(&mut l, &n("t_embedding_norm.weight"))?,
            blocks,
            fin_ada: ada(&mut l, &n("final_layer.adaln_modulation"))?,
            fin_linear,
            ad_embed: st.bytes(&emb_name)?.to_vec(),
            ad_embed_dtype: st.info(&emb_name)?.dtype.clone(),
            ad_blocks,
            ad_out: Dense::load(&mut l, &a("out_proj"))?,
            ad_norm: linear::small_bf16(&mut l, &a("norm.weight"))?,
            dev,
        };
        ensure!(an.ad_embed_dtype == "BF16" || an.ad_embed_dtype == "F16" || an.ad_embed_dtype == "F32", "Anima: adapter embedding dtype {}", an.ad_embed_dtype);
        crate::info!("  dit: Anima, {} blocks, dim {}, {} {:.1} GB uploaded in {:.1}s", an.blocks.len(), an.dim, if int8 { "int8" } else { "bf16" }, l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        drop(l);
        st.drop_in_background();
        Ok(an)
    }

    fn embed_t5(&self, ids: &[u32]) -> Result<Tensor> {
        let es = match self.ad_embed_dtype.as_str() {
            "F32" => 4,
            _ => 2,
        };
        let vocab = self.ad_embed.len() / (CTX_DIM * es);
        let mut bits: Vec<u16> = Vec::with_capacity(ids.len() * CTX_DIM);
        for &id in ids {
            ensure!((id as usize) < vocab, "T5 token {id} outside the adapter's embedding");
            let row = &self.ad_embed[id as usize * CTX_DIM * es..(id as usize + 1) * CTX_DIM * es];
            match self.ad_embed_dtype.as_str() {
                "BF16" => bits.extend(row.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]))),
                "F16" => bits.extend(row.chunks_exact(2).map(|c| bf16_bits(half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32()))),
                _ => bits.extend(row.chunks_exact(4).map(|c| bf16_bits(f32::from_le_bytes(c.try_into().unwrap())))),
            }
        }
        Tensor::from_bf16(&self.dev, &bits, &[ids.len(), CTX_DIM])
    }

    /// The adapter's rope table [n, 32, 2] (theta 10000, head dim 64), cos/sin rounded to bf16 as the reference casts them.
    fn ad_rope(&self, n: usize) -> Result<Tensor> {
        let inv: Vec<f32> = (0..32).map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / AD_HD as f32)).collect();
        let mut tab = vec![0f32; n * 64];
        for p in 0..n {
            for (i, f) in inv.iter().enumerate() {
                let a = p as f32 * f;
                tab[p * 64 + i * 2] = model::round_bf16(a.cos());
                tab[p * 64 + i * 2 + 1] = model::round_bf16(a.sin());
            }
        }
        Tensor::from_f32(&self.dev, &tab, &[n, 32, 2])
    }

    /// LLM adapter: Qwen3 states `src` [S, 1024] (rows >= `src_valid` masked; zeroed by the caller) and T5 ids -> context
    /// [512, 1024] bf16.
    pub fn adapter(&self, src: &Tensor, src_valid: usize, t5: &[u32]) -> Result<Tensor> {
        let dev = &self.dev;
        let (s, l) = (src.shape[0], t5.len());
        let x = self.embed_t5(t5)?;
        let rope_t = self.ad_rope(l)?;
        let rope_s = self.ad_rope(s)?;
        // an empty prompt masks every source token: the cross-attention then contributes nothing (as in the reference)
        let lim = if src_valid < s { Some(Tensor::from_buf(dev.upload(&vec![src_valid as i32; l])?, DType::F32, &[l])) } else { None };
        let n = Tensor::new(dev, DType::BF16, &[l, CTX_DIM])?;
        let a = Tensor::new(dev, DType::BF16, &[l, CTX_DIM])?;
        let o = Tensor::new(dev, DType::BF16, &[l, CTX_DIM])?;
        let hd = AD_HD;
        for b in &self.ad_blocks {
            // self-attention over the target tokens
            ops::rmsnorm_x(dev, &x, Some(&b.n_sa), RmsMode::Hf, EPS, &n)?;
            let qkv = Tensor::new(dev, DType::BF16, &[l, 3 * CTX_DIM])?;
            ops::gemm_bf16(dev, &n, &b.sa_qkv, None, Act::None, Epi::Store, &qkv)?;
            ops::qk_norm_rope_d(dev, qkv.ptr, 3 * CTX_DIM, 0, CTX_DIM, l, AD_HEADS, AD_HEADS, hd, Some(&b.sa_qn), Some(&b.sa_kn), EPS, QkNorm::Hf, Rope::HalfBf16, Some(&rope_t))?;
            let view = |t: &Tensor, off: usize, ts: usize, len: usize| AttnView { ptr: t.ptr + (off * 2) as u64, tok_stride: ts, head_stride: hd, len };
            ops::attention(dev, view(&qkv, 0, 3 * CTX_DIM, l), view(&qkv, CTX_DIM, 3 * CTX_DIM, l), view(&qkv, 2 * CTX_DIM, 3 * CTX_DIM, l), AttnView::contiguous(&a, AD_HEADS, hd), AD_HEADS, AD_HEADS, hd, false, None)?;
            ops::gemm_bf16(dev, &a, &b.sa_o, None, Act::None, Epi::Store, &o)?;
            ops::add_inplace(dev, &x, &o)?;
            // cross-attention to the Qwen3 states (q with the target's rope, k with the source's)
            if src_valid > 0 {
                ops::rmsnorm_x(dev, &x, Some(&b.n_ca), RmsMode::Hf, EPS, &n)?;
                let q = Tensor::new(dev, DType::BF16, &[l, CTX_DIM])?;
                ops::gemm_bf16(dev, &n, &b.ca_q, None, Act::None, Epi::Store, &q)?;
                ops::qk_norm_rope_d(dev, q.ptr, CTX_DIM, 0, 0, l, AD_HEADS, 0, hd, Some(&b.ca_qn), Some(&b.ca_qn), EPS, QkNorm::Hf, Rope::HalfBf16, Some(&rope_t))?;
                let kv = Tensor::new(dev, DType::BF16, &[s, 2 * CTX_DIM])?;
                ops::gemm_bf16(dev, src, &b.ca_kv, None, Act::None, Epi::Store, &kv)?;
                ops::qk_norm_rope_d(dev, kv.ptr, 2 * CTX_DIM, 0, 0, s, 0, AD_HEADS, hd, Some(&b.ca_kn), Some(&b.ca_kn), EPS, QkNorm::Hf, Rope::HalfBf16, Some(&rope_s))?;
                ops::attention(dev, view(&q, 0, CTX_DIM, l), view(&kv, 0, 2 * CTX_DIM, s), view(&kv, CTX_DIM, 2 * CTX_DIM, s), AttnView::contiguous(&a, AD_HEADS, hd), AD_HEADS, AD_HEADS, hd, false, lim.as_ref())?;
                ops::gemm_bf16(dev, &a, &b.ca_o, None, Act::None, Epi::Store, &o)?;
                ops::add_inplace(dev, &x, &o)?;
            }
            // MLP
            ops::rmsnorm_x(dev, &x, Some(&b.n_mlp), RmsMode::Hf, EPS, &n)?;
            let hdn = b.mlp1.apply(dev, &n, Act::GeluErf)?;
            ops::gemm_bf16(dev, &hdn, &b.mlp2.w, b.mlp2.b.as_ref(), Act::None, Epi::Store, &o)?;
            ops::add_inplace(dev, &x, &o)?;
        }
        let y = self.ad_out.apply(dev, &x, Act::None)?;
        let ctx = Tensor::zeros(dev, DType::BF16, &[l.max(CTX_LEN), CTX_DIM])?;
        ops::rmsnorm_x(dev, &y, Some(&self.ad_norm), RmsMode::Hf, EPS, &ctx.rows(0, l))?;
        Ok(ctx)
    }

    /// Cross-attention keys / values of a context for every block, and the image rope.
    pub fn prepare(&self, ctx: &Tensor, h: usize, w: usize) -> Result<Run> {
        let dev = &self.dev;
        let n = ctx.shape[0];
        let mut kv = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let k = Tensor::new(dev, DType::BF16, &[n, self.dim])?;
            let v = Tensor::new(dev, DType::BF16, &[n, self.dim])?;
            ops::gemm_bf16(dev, ctx, &b.ck, None, Act::None, Epi::Store, &k)?;
            ops::gemm_bf16(dev, ctx, &b.cv, None, Act::None, Epi::Store, &v)?;
            ops::qk_norm_rope(dev, k.ptr, self.dim, 0, 0, n, 0, self.heads, Some(&b.ckn), Some(&b.ckn), EPS, QkNorm::Hf, Rope::None, None)?;
            kv.push((k, v));
        }
        let hw_theta = 10000f64 * 4f64.powf(42.0 / 40.0);
        let rope = model::rope_table(dev, h * w, |tok, f| {
            let (row, col) = ((tok / w) as f32, (tok % w) as f32);
            if f < 22 {
                0.0
            } else if f < 43 {
                let k = (f - 22) as f32;
                (row * (1.0 / (hw_theta as f32).powf(2.0 * k / 42.0))) as f64
            } else {
                let k = (f - 43) as f32;
                (col * (1.0 / (hw_theta as f32).powf(2.0 * k / 42.0))) as f64
            }
        })?;
        Ok(Run { kv, rope, h, w })
    }
}

impl model::Model for Anima {
    type Run = Run;
    type Mod = Mod;
    fn channels(&self) -> usize {
        LATENT
    }

    fn modulation(&self, sigma: f32) -> Result<Mod> {
        let dev = &self.dev;
        let d = self.dim;
        // pipeline: timestep = t (= sigma * 1000) in bf16, / 1000 in bf16
        let t = model::round_bf16(model::round_bf16(sigma * 1000.0) / 1000.0);
        let e = model::sinusoid_bf16(dev, t, d)?;
        let h1 = Tensor::new(dev, DType::BF16, &[1, d])?;
        ops::gemm_bf16(dev, &e, &self.t1, None, Act::None, Epi::Store, &h1)?;
        ops::act_inplace(dev, &h1, Act::Silu)?;
        let lora = Tensor::new(dev, DType::BF16, &[1, 3 * d])?;
        ops::gemm_bf16(dev, &h1, &self.t2, None, Act::None, Epi::Store, &lora)?;
        let emb = Tensor::new(dev, DType::BF16, &[1, d])?;
        ops::rmsnorm_x(dev, &e, Some(&self.t_norm), RmsMode::Hf, EPS, &emb)?;
        let se = model::silu_copy(dev, &emb)?;
        let lv = lora.to_f32_vec(dev)?;
        let mid = Tensor::new(dev, DType::BF16, &[1, 256])?;
        let apply = |a: &AdaLn, n_out: usize| -> Result<Vec<f32>> {
            ops::gemm_bf16(dev, &se, &a.a, None, Act::None, Epi::Store, &mid)?;
            let o = Tensor::new(dev, DType::BF16, &[1, n_out])?;
            ops::gemm_bf16(dev, &mid, &a.b, None, Act::None, Epi::Store, &o)?;
            Ok(o.to_f32_vec(dev)?.iter().zip(&lv).map(|(x, y)| model::round_bf16(x + y)).collect())
        };
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for b in &self.blocks {
            let mut v = Vec::with_capacity(9);
            for a in &b.ada {
                let m = apply(a, 3 * d)?;
                for i in 0..3 {
                    v.push(Tensor::from_f32(dev, &m[i * d..(i + 1) * d], &[d])?);
                }
            }
            blocks.push(v);
        }
        let f = apply(&self.fin_ada, 2 * d)?;
        Ok(Mod { blocks, fin_shift: Tensor::from_f32(dev, &f[..d], &[d])?, fin_scale: Tensor::from_f32(dev, &f[d..], &[d])? })
    }

    fn forward(&self, run: &Run, x: &Tensor, m: &Mod, out: &Tensor) -> Result<()> {
        let dev = &self.dev;
        let ni = run.h * run.w;
        let (d, nh) = (self.dim, self.heads);
        ensure!(x.shape[0] == ni, "latent tokens {} != {}", x.shape[0], ni);
        let h = Tensor::new(dev, DType::BF16, &[ni, d])?;
        ops::gemm_bf16(dev, x, &self.x_in, None, Act::None, Epi::Store, &h)?;
        let n = Tensor::new(dev, DType::BF16, &[ni, d])?;
        let qkv = Tensor::new(dev, DType::BF16, &[ni, 3 * d])?;
        let a = Tensor::new(dev, DType::BF16, &[ni, d])?;
        let o = Tensor::new(dev, DType::BF16, &[ni, d])?;
        let ff = Tensor::new(dev, DType::BF16, &[ni, self.blocks[0].l1.n()])?;
        for ((b, mm), (k, v)) in self.blocks.iter().zip(&m.blocks).zip(&run.kv) {
            // self-attention
            ops::norm_mod(dev, &h, ModNorm::Layer, EPS, ni, (&mm[1], &mm[0]), (&mm[1], &mm[0]), &n)?;
            b.qkv.forward(dev, &n, None, Epi::Store, &qkv)?;
            ops::qk_norm_rope(dev, qkv.ptr, 3 * d, 0, d, ni, nh, nh, Some(&b.sq), Some(&b.sk), EPS, QkNorm::Hf, Rope::Half32, Some(&run.rope))?;
            let view = |off: usize| AttnView { ptr: qkv.ptr + (off * 2) as u64, tok_stride: 3 * d, head_stride: HD, len: ni };
            ops::attention(dev, view(0), view(d), view(2 * d), AttnView::contiguous(&a, nh, HD), nh, nh, HD, false, None)?;
            b.so.forward(dev, &a, None, Epi::Store, &o)?;
            ops::gated_add(dev, &h, &o, &mm[2])?;
            // cross-attention
            ops::norm_mod(dev, &h, ModNorm::Layer, EPS, ni, (&mm[4], &mm[3]), (&mm[4], &mm[3]), &n)?;
            let q = qkv.prefix(&[ni, d]);
            b.cq.forward(dev, &n, None, Epi::Store, &q)?;
            ops::qk_norm_rope(dev, q.ptr, d, 0, 0, ni, nh, 0, Some(&b.cqn), Some(&b.cqn), EPS, QkNorm::Hf, Rope::None, None)?;
            ops::attention(dev, AttnView::contiguous(&q, nh, HD), AttnView::contiguous(k, nh, HD), AttnView::contiguous(v, nh, HD), AttnView::contiguous(&a, nh, HD), nh, nh, HD, false, None)?;
            b.co.forward(dev, &a, None, Epi::Store, &o)?;
            ops::gated_add(dev, &h, &o, &mm[5])?;
            // MLP
            ops::norm_mod(dev, &h, ModNorm::Layer, EPS, ni, (&mm[7], &mm[6]), (&mm[7], &mm[6]), &n)?;
            b.l1.forward(dev, &n, None, Epi::Store, &ff)?;
            ops::act_inplace(dev, &ff, Act::GeluErf)?;
            b.l2.forward(dev, &ff, None, Epi::Store, &o)?;
            ops::gated_add(dev, &h, &o, &mm[8])?;
        }
        ops::norm_mod(dev, &h, ModNorm::Layer, EPS, ni, (&m.fin_scale, &m.fin_shift), (&m.fin_scale, &m.fin_shift), &n)?;
        ops::gemm_bf16(dev, &n, &self.fin_linear, None, Act::None, Epi::Store, out)
    }
}
