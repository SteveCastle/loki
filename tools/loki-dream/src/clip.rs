//! SDXL's text side: the CLIP BPE tokenizer, prompt weighting and 75-token chunking (ComfyUI's rules), and the two
//! text encoders packed in an SDXL checkpoint, CLIP ViT-L/14 (HF layout, quick GELU) and OpenCLIP ViT-bigG/14 (OpenCLIP
//! layout, GELU, text projection).
//!
//! Conditioning is the penultimate hidden state of both (no final LayerNorm), concatenated to [77 * chunks, 768 + 1280];
//! the pooled vector is bigG's final-LayerNorm'd end-of-text state of the first chunk through its text projection.
//! `(text)` weights by 1.1 per level, `(text:1.3)` sets a weight, `\(` / `\)` are literal; a weighted token's state is
//! moved away from the empty prompt's state at that position: z' = z_empty + w (z - z_empty).
use crate::cuda::Device;
use crate::linear::{self, Want};
use crate::ops::{self, Act, AttnView, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::weights::Loader;
use anyhow::{bail, ensure, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

const VOCAB_JSON: &str = include_str!("../assets/clip/vocab.json");
const MERGES_TXT: &str = include_str!("../assets/clip/merges.txt");

pub const BOS: u32 = 49406;
pub const EOS: u32 = 49407;
/// tokens per chunk, start and end included
pub const CHUNK: usize = 77;
/// ComfyUI keeps a word of fewer tokens than this whole in one chunk
const MAX_WORD: usize = 8;

// ------------------------------------------------------------------------------------------------ tokenizer
pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_enc: [char; 256],
    cache: RefCell<HashMap<String, Vec<u32>>>,
}

impl ClipTokenizer {
    pub fn new() -> Result<ClipTokenizer> {
        let vocab: HashMap<String, u32> = serde_json::from_str(VOCAB_JSON).context("CLIP vocabulary")?;
        let mut ranks = HashMap::new();
        for (i, line) in MERGES_TXT.lines().skip(1).enumerate() {
            if let Some((a, b)) = line.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i);
            }
        }
        Ok(ClipTokenizer { vocab, ranks, byte_enc: crate::tokenizer::bytes_to_unicode(), cache: RefCell::new(HashMap::new()) })
    }

    /// Token ids of `text`, without the start and end tokens.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut out = Vec::new();
        for piece in pretokenize(&clean) {
            if piece == "<|startoftext|>" {
                out.push(BOS);
                continue;
            }
            if piece == "<|endoftext|>" {
                out.push(EOS);
                continue;
            }
            if let Some(ids) = self.cache.borrow().get(&piece) {
                out.extend(ids);
                continue;
            }
            let mapped: String = piece.bytes().map(|b| self.byte_enc[b as usize]).collect();
            let ids: Vec<u32> = self.bpe(&mapped).iter().filter_map(|t| self.vocab.get(t).copied()).collect();
            out.extend(&ids);
            self.cache.borrow_mut().insert(piece, ids);
        }
        out
    }

    /// BPE over one pre-token: its characters, the last one marked as a word end (`</w>`).
    fn bpe(&self, word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let mut parts: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = parts.last_mut() {
            last.push_str("</w>");
        }
        while parts.len() > 1 {
            let best = (0..parts.len() - 1).filter_map(|i| self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())).map(|&r| (r, i))).min();
            let Some((_, _)) = best else { break };
            let (a, b) = {
                let i = best.unwrap().1;
                (parts[i].clone(), parts[i + 1].clone())
            };
            // merge every occurrence of the pair, left to right
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        parts
    }
}

/// CLIP's split: `<|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+`.
fn pretokenize(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let rest = |i: usize| c[i..].iter().collect::<String>();
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }
        if ch == '<' {
            let r = rest(i);
            if let Some(t) = ["<|startoftext|>", "<|endoftext|>"].iter().find(|t| r.starts_with(*t)) {
                out.push(t.to_string());
                i += t.chars().count();
                continue;
            }
        }
        if ch == '\'' {
            let r = rest(i);
            if let Some(t) = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"].iter().find(|t| r.starts_with(*t)) {
                out.push(t.to_string());
                i += t.len();
                continue;
            }
        }
        let start = i;
        if ch.is_alphabetic() {
            while i < c.len() && c[i].is_alphabetic() {
                i += 1;
            }
        } else if ch.is_numeric() {
            i += 1;
        } else {
            while i < c.len() && !c[i].is_whitespace() && !c[i].is_alphabetic() && !c[i].is_numeric() {
                i += 1;
            }
        }
        out.push(c[start..i].iter().collect());
    }
    out
}

// ------------------------------------------------------------------------------------------------ prompt weighting
const ESC_CLOSE: &str = "\u{0}\u{1}";
const ESC_OPEN: &str = "\u{0}\u{2}";

/// ComfyUI's `parse_parentheses`: top-level `( ... )` groups and the text between them.
fn parse_parentheses(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut level: i32 = 0;
    for ch in s.chars() {
        match ch {
            '(' => {
                if level == 0 {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    cur.push('(');
                } else {
                    cur.push('(');
                }
                level += 1;
            }
            ')' => {
                level -= 1;
                if level == 0 {
                    cur.push(')');
                    out.push(std::mem::take(&mut cur));
                } else {
                    cur.push(')');
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn token_weights(s: &str, weight: f32) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    for x in parse_parentheses(s) {
        if x.len() >= 2 && x.starts_with('(') && x.ends_with(')') {
            let mut inner = x[1..x.len() - 1].to_string();
            let mut w = weight * 1.1;
            if let Some(p) = inner.rfind(':') {
                if p > 0 {
                    if let Ok(v) = inner[p + 1..].trim().parse::<f32>() {
                        w = v;
                        inner.truncate(p);
                    }
                }
            }
            out.extend(token_weights(&inner, w));
        } else {
            out.push((x, weight));
        }
    }
    out
}

/// The prompt as (text, weight) segments, escapes resolved.
pub fn weighted_segments(text: &str) -> Vec<(String, f32)> {
    let escaped = text.replace("\\)", ESC_CLOSE).replace("\\(", ESC_OPEN);
    token_weights(&escaped, 1.0).into_iter().map(|(t, w)| (t.replace(ESC_CLOSE, ")").replace(ESC_OPEN, "("), w)).collect()
}

/// One 77-token chunk: ids (None = padding) and per-token weights.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    pub ids: Vec<Option<u32>>,
    pub weights: Vec<f32>,
}
impl Chunk {
    /// The ids with padding filled in (CLIP-L pads with end-of-text, bigG with 0).
    pub fn filled(&self, pad: u32) -> Vec<u32> {
        self.ids.iter().map(|t| t.unwrap_or(pad)).collect()
    }
}

/// ComfyUI's `tokenize_with_weights` (SDXL: max length 77, padded): words are tokenized one by one; a word that no
/// longer fits starts the next chunk unless it is 8 tokens or longer, then it is split.
pub fn chunks(tok: &ClipTokenizer, text: &str) -> Vec<Chunk> {
    let mut words: Vec<Vec<(u32, f32)>> = Vec::new();
    for (seg, w) in weighted_segments(text) {
        for word in seg.replace('\n', " ").split(' ').filter(|x| !x.is_empty()) {
            let t: Vec<(u32, f32)> = tok.encode(word).into_iter().map(|id| (id, w)).collect();
            if !t.is_empty() {
                words.push(t);
            }
        }
    }
    let mut out = Vec::new();
    let mut batch: Vec<(Option<u32>, f32)> = vec![(Some(BOS), 1.0)];
    let close = |batch: &mut Vec<(Option<u32>, f32)>, out: &mut Vec<Chunk>| {
        batch.push((Some(EOS), 1.0));
        batch.resize(CHUNK, (None, 1.0));
        out.push(Chunk { ids: batch.iter().map(|x| x.0).collect(), weights: batch.iter().map(|x| x.1).collect() });
        *batch = vec![(Some(BOS), 1.0)];
    };
    for mut g in words {
        let large = g.len() >= MAX_WORD;
        while !g.is_empty() {
            if g.len() + batch.len() > CHUNK - 1 {
                let room = CHUNK - batch.len() - 1;
                if large {
                    batch.extend(g.drain(..room).map(|(t, w)| (Some(t), w)));
                }
                close(&mut batch, &mut out);
            } else {
                batch.extend(g.drain(..).map(|(t, w)| (Some(t), w)));
            }
        }
    }
    close(&mut batch, &mut out);
    out
}

// ------------------------------------------------------------------------------------------------ encoders
struct Layer {
    ln1: (Tensor, Tensor),
    qkv: Tensor,
    qkv_b: Tensor,
    out: Tensor,
    out_b: Tensor,
    ln2: (Tensor, Tensor),
    fc1: Tensor,
    fc1_b: Tensor,
    fc2: Tensor,
    fc2_b: Tensor,
}

/// One CLIP text transformer.
pub struct ClipEncoder {
    dev: Arc<Device>,
    pub dim: usize,
    heads: usize,
    layers: Vec<Layer>,
    tok_emb: Tensor,
    pos_emb: Tensor,
    quick_gelu: bool,
    /// final LayerNorm and projection [out, in] for the pooled output (bigG)
    pool: Option<((Tensor, Tensor), Tensor)>,
    /// padding token
    pub pad: u32,
}

fn bf16w(l: &mut Loader, prefix: &str) -> Result<Tensor> {
    match linear::lin(l, prefix, Want::Bf16)? {
        linear::Lin::Bf16(w) => Ok(w),
        _ => bail!("{prefix}: expected a float weight"),
    }
}

fn ln(l: &mut Loader, prefix: &str) -> Result<(Tensor, Tensor)> {
    Ok((linear::small_bf16(l, &format!("{prefix}.weight"))?, linear::small_bf16(l, &format!("{prefix}.bias"))?))
}

fn f32t(l: &mut Loader, name: &str) -> Result<Tensor> {
    l.f32(name)
}

impl ClipEncoder {
    /// CLIP ViT-L/14 text model in HF layout under `p` (`...text_model.`): 12 layers, 768 wide, quick GELU.
    /// Only the layers up to the penultimate output are loaded.
    pub fn load_hf(dev: Arc<Device>, st: &SafeTensors, p: &str, pad: u32, pool: bool, proj: Option<&str>) -> Result<ClipEncoder> {
        let mut l = Loader::new(st, dev.clone());
        let tok_emb = linear::small_bf16(&mut l, &format!("{p}embeddings.token_embedding.weight"))?;
        let pos_emb = linear::small_bf16(&mut l, &format!("{p}embeddings.position_embedding.weight"))?;
        let dim = tok_emb.shape[1];
        let n = (0..).take_while(|i| st.has(&format!("{p}encoder.layers.{i}.layer_norm1.weight"))).count();
        ensure!(n > 1, "{p}: no CLIP text layers");
        let keep = if pool { n } else { n - 1 };
        let mut layers = Vec::with_capacity(keep);
        for i in 0..keep {
            let q = format!("{p}encoder.layers.{i}.");
            let qkv = match linear::lin_cat(&mut l, &[&format!("{q}self_attn.q_proj"), &format!("{q}self_attn.k_proj"), &format!("{q}self_attn.v_proj")], Want::Bf16)? {
                linear::Lin::Bf16(w) => w,
                _ => bail!("{q}: expected float weights"),
            };
            let qkv_b = linear::bias_cat(&mut l, &[&format!("{q}self_attn.q_proj"), &format!("{q}self_attn.k_proj"), &format!("{q}self_attn.v_proj")], &[dim, dim, dim])?.context("CLIP q/k/v bias")?;
            layers.push(Layer {
                ln1: ln(&mut l, &format!("{q}layer_norm1"))?,
                qkv,
                qkv_b,
                out: bf16w(&mut l, &format!("{q}self_attn.out_proj"))?,
                out_b: f32t(&mut l, &format!("{q}self_attn.out_proj.bias"))?,
                ln2: ln(&mut l, &format!("{q}layer_norm2"))?,
                fc1: bf16w(&mut l, &format!("{q}mlp.fc1"))?,
                fc1_b: f32t(&mut l, &format!("{q}mlp.fc1.bias"))?,
                fc2: bf16w(&mut l, &format!("{q}mlp.fc2"))?,
                fc2_b: f32t(&mut l, &format!("{q}mlp.fc2.bias"))?,
            });
        }
        let pool = if pool {
            let proj = proj.context("pooled output needs a text projection")?;
            Some((ln(&mut l, &format!("{p}final_layer_norm"))?, bf16w(&mut l, proj)?))
        } else {
            None
        };
        let quick_gelu = dim == 768;
        Ok(ClipEncoder { dev, dim, heads: dim / 64, layers, tok_emb, pos_emb, quick_gelu, pool, pad })
    }

    /// OpenCLIP ViT-bigG/14 text model under `p` (`...model.`): 32 layers, 1280 wide, GELU, `text_projection` [in, out].
    pub fn load_openclip(dev: Arc<Device>, st: &SafeTensors, p: &str, pad: u32) -> Result<ClipEncoder> {
        let mut l = Loader::new(st, dev.clone());
        let tok_emb = linear::small_bf16(&mut l, &format!("{p}token_embedding.weight"))?;
        let pos_emb = linear::small_bf16(&mut l, &format!("{p}positional_embedding"))?;
        let dim = tok_emb.shape[1];
        let n = (0..).take_while(|i| st.has(&format!("{p}transformer.resblocks.{i}.ln_1.weight"))).count();
        ensure!(n > 1, "{p}: no OpenCLIP text layers");
        let mut layers = Vec::with_capacity(n);
        for i in 0..n {
            let q = format!("{p}transformer.resblocks.{i}.");
            let qkv_name = format!("{q}attn.in_proj_weight");
            let qkv = {
                let v = st.f32s(&qkv_name)?;
                let bits: Vec<u16> = v.iter().map(|&x| crate::tensor::bf16_bits(x)).collect();
                l.uploaded += bits.len() * 2;
                Tensor::from_bf16(&dev, &bits, &[3 * dim, dim])?
            };
            layers.push(Layer {
                ln1: ln(&mut l, &format!("{q}ln_1"))?,
                qkv,
                qkv_b: f32t(&mut l, &format!("{q}attn.in_proj_bias"))?,
                out: bf16w(&mut l, &format!("{q}attn.out_proj"))?,
                out_b: f32t(&mut l, &format!("{q}attn.out_proj.bias"))?,
                ln2: ln(&mut l, &format!("{q}ln_2"))?,
                fc1: bf16w(&mut l, &format!("{q}mlp.c_fc"))?,
                fc1_b: f32t(&mut l, &format!("{q}mlp.c_fc.bias"))?,
                fc2: bf16w(&mut l, &format!("{q}mlp.c_proj"))?,
                fc2_b: f32t(&mut l, &format!("{q}mlp.c_proj.bias"))?,
            });
        }
        // x @ text_projection: stored [in, out], transposed to the [out, in] the GEMM takes
        let pname = format!("{p}text_projection");
        let pinfo = st.info(&pname)?.clone();
        ensure!(pinfo.shape.len() == 2, "{pname}: expected a matrix");
        let (pi, po) = (pinfo.shape[0], pinfo.shape[1]);
        let v = st.f32s(&pname)?;
        let mut t = vec![0u16; pi * po];
        for i in 0..pi {
            for o in 0..po {
                t[o * pi + i] = crate::tensor::bf16_bits(v[i * po + o]);
            }
        }
        let proj = Tensor::from_bf16(&dev, &t, &[po, pi])?;
        let pool = Some((ln(&mut l, &format!("{p}ln_final"))?, proj));
        Ok(ClipEncoder { dev, dim, heads: dim / 64, layers, tok_emb, pos_emb, quick_gelu: false, pool, pad })
    }

    /// Penultimate hidden states of every chunk f32 [chunks * 77, dim], and the pooled output of the first chunk.
    pub fn encode(&self, chunks: &[Vec<u32>]) -> Result<(Vec<f32>, Option<Vec<f32>>)> {
        let dev = &self.dev;
        let (d, nc) = (self.dim, chunks.len());
        let t = nc * CHUNK;
        let ids: Vec<i32> = chunks.iter().flatten().map(|&x| x as i32).collect();
        ensure!(ids.len() == t, "CLIP: every chunk must hold {CHUNK} tokens");
        let ids_t = Tensor::from_buf(dev.upload(&ids)?, DType::F32, &[t]);
        let x = Tensor::new(dev, DType::BF16, &[t, d])?;
        dev.launch_n("k_embed_bf16", t * d, &[crate::cuda::Arg::Ptr(self.tok_emb.ptr), crate::cuda::Arg::Ptr(ids_t.ptr), crate::cuda::Arg::Ptr(x.ptr), crate::cuda::Arg::I32(t as i32), crate::cuda::Arg::I32(d as i32)])?;
        for c in 0..nc {
            ops::add_inplace(dev, &x.rows(c * CHUNK, CHUNK), &self.pos_emb.rows(0, CHUNK))?;
        }
        let normed = Tensor::new(dev, DType::BF16, &[t, d])?;
        let qkv = Tensor::new(dev, DType::BF16, &[t, 3 * d])?;
        let att = Tensor::new(dev, DType::BF16, &[t, d])?;
        let x2 = Tensor::new(dev, DType::BF16, &[t, d])?;
        let hid = Tensor::new(dev, DType::BF16, &[t, self.layers[0].fc1.shape[0]])?;
        let n = self.layers.len();
        let mut penult: Option<Vec<f32>> = None;
        let (mut cur, mut other) = (x, x2);
        for (i, ly) in self.layers.iter().enumerate() {
            // the penultimate state: the input of the last layer of the full model
            if (self.pool.is_some() && i == n - 1) || (self.pool.is_none() && i == n) {
                penult = Some(cur.to_f32_vec(dev)?);
            }
            ops::layernorm(dev, &cur, Some(&ly.ln1.0), Some(&ly.ln1.1), 1e-5, &normed)?;
            ops::gemm_bf16(dev, &normed, &ly.qkv, Some(&ly.qkv_b), Act::None, Epi::Store, &qkv)?;
            for c in 0..nc {
                let base = qkv.ptr + (c * CHUNK * 3 * d * 2) as u64;
                let view = |off: usize| AttnView { ptr: base + (off * 2) as u64, tok_stride: 3 * d, head_stride: 64, len: CHUNK };
                let o = AttnView { ptr: att.ptr + (c * CHUNK * d * 2) as u64, tok_stride: d, head_stride: 64, len: CHUNK };
                ops::attention(dev, view(0), view(d), view(2 * d), o, self.heads, self.heads, 64, true, None)?;
            }
            ops::gemm_bf16(dev, &att, &ly.out, Some(&ly.out_b), Act::None, Epi::AddRes(&cur), &other)?;
            std::mem::swap(&mut cur, &mut other);
            ops::layernorm(dev, &cur, Some(&ly.ln2.0), Some(&ly.ln2.1), 1e-5, &normed)?;
            if self.quick_gelu {
                ops::gemm_bf16(dev, &normed, &ly.fc1, Some(&ly.fc1_b), Act::None, Epi::Store, &hid)?;
                dev.launch_n("k_quick_gelu_bf16", hid.numel(), &[crate::cuda::Arg::Ptr(hid.ptr), crate::cuda::Arg::I64(hid.numel() as i64)])?;
            } else {
                ops::gemm_bf16(dev, &normed, &ly.fc1, Some(&ly.fc1_b), Act::GeluErf, Epi::Store, &hid)?;
            }
            ops::gemm_bf16(dev, &hid, &ly.fc2, Some(&ly.fc2_b), Act::None, Epi::AddRes(&cur), &other)?;
            std::mem::swap(&mut cur, &mut other);
        }
        let penult = match penult {
            Some(p) => p,
            None => cur.to_f32_vec(dev)?,
        };
        let pooled = match &self.pool {
            Some(((w, b), proj)) => {
                // end-of-text: the first occurrence of the largest id (torch argmax)
                let first = &chunks[0];
                let eos = first.iter().enumerate().fold(0, |best, (i, &v)| if v > first[best] { i } else { best });
                let row = cur.rows(eos, 1);
                let nrow = Tensor::new(dev, DType::BF16, &[1, d])?;
                ops::layernorm(dev, &row, Some(w), Some(b), 1e-5, &nrow)?;
                let out = Tensor::new(dev, DType::F32, &[1, proj.shape[0]])?;
                ops::gemm_bf16(dev, &nrow, proj, None, Act::None, Epi::Store, &out)?;
                Some(out.to_f32_vec(dev)?)
            }
            None => None,
        };
        Ok((penult, pooled))
    }
}

/// Both SDXL text encoders of a checkpoint.
pub struct SdxlText {
    pub l: ClipEncoder,
    pub g: ClipEncoder,
}

/// SDXL conditioning of one text: cross-attention context f32 [chunks * 77, 2048] and the pooled vector [1280].
pub struct TextCond {
    pub ctx: Vec<f32>,
    pub tokens: usize,
    pub pooled: Vec<f32>,
}

impl SdxlText {
    pub fn load(dev: Arc<Device>, st: &SafeTensors) -> Result<SdxlText> {
        let lp = "conditioner.embedders.0.transformer.text_model.";
        ensure!(st.has(&format!("{lp}embeddings.token_embedding.weight")), "{}: no CLIP-L text encoder (conditioner.embedders.0) in this SDXL checkpoint", st.path);
        let l = ClipEncoder::load_hf(dev.clone(), st, lp, EOS, false, None)?;
        let g = if st.has("conditioner.embedders.1.model.token_embedding.weight") {
            ClipEncoder::load_openclip(dev, st, "conditioner.embedders.1.model.", 0)?
        } else if st.has("conditioner.embedders.1.transformer.text_model.embeddings.token_embedding.weight") {
            ClipEncoder::load_hf(dev, st, "conditioner.embedders.1.transformer.text_model.", 0, true, Some("conditioner.embedders.1.transformer.text_projection"))?
        } else {
            bail!("{}: no OpenCLIP bigG text encoder (conditioner.embedders.1) in this SDXL checkpoint", st.path)
        };
        ensure!(l.dim == 768 && g.dim == 1280, "unexpected SDXL text encoder widths {} / {}", l.dim, g.dim);
        Ok(SdxlText { l, g })
    }

    pub fn encode(&self, tok: &ClipTokenizer, text: &str) -> Result<TextCond> {
        let ch = chunks(tok, text);
        let weighted = ch.iter().any(|c| c.weights.iter().any(|&w| w != 1.0));
        let mut all = ch.clone();
        if weighted {
            all.push(chunks(tok, "").remove(0));
        }
        let nc = ch.len();
        let enc = |e: &ClipEncoder| -> Result<(Vec<f32>, Option<Vec<f32>>)> {
            let ids: Vec<Vec<u32>> = all.iter().map(|c| c.filled(e.pad)).collect();
            let (mut z, pooled) = e.encode(&ids)?;
            if weighted {
                let d = e.dim;
                let empty = z[nc * CHUNK * d..].to_vec();
                for (k, c) in ch.iter().enumerate() {
                    for (j, &w) in c.weights.iter().enumerate() {
                        if w != 1.0 {
                            let row = &mut z[(k * CHUNK + j) * d..(k * CHUNK + j + 1) * d];
                            for (v, e0) in row.iter_mut().zip(&empty[j * d..(j + 1) * d]) {
                                *v = (*v - e0) * w + e0;
                            }
                        }
                    }
                }
                z.truncate(nc * CHUNK * d);
            }
            Ok((z, pooled))
        };
        let (zl, _) = enc(&self.l)?;
        let (zg, pooled) = enc(&self.g)?;
        let t = nc * CHUNK;
        let (dl, dg) = (self.l.dim, self.g.dim);
        let mut ctx = vec![0f32; t * (dl + dg)];
        for r in 0..t {
            ctx[r * (dl + dg)..r * (dl + dg) + dl].copy_from_slice(&zl[r * dl..(r + 1) * dl]);
            ctx[r * (dl + dg) + dl..(r + 1) * (dl + dg)].copy_from_slice(&zg[r * dg..(r + 1) * dg]);
        }
        Ok(TextCond { ctx, tokens: t, pooled: pooled.context("pooled text embedding")? })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_like_clip() {
        let t = ClipTokenizer::new().unwrap();
        // openai/clip-vit-large-patch14: "a photo of a cat" -> 320 1125 539 320 2368
        assert_eq!(t.encode("a photo of a cat"), vec![320, 1125, 539, 320, 2368]);
        assert_eq!(t.encode("A  Photo"), t.encode("a photo"));
        assert_eq!(pretokenize("don't 42x, ok!"), vec!["don", "'t", "4", "2", "x", ",", "ok", "!"]);
    }

    #[test]
    fn parses_weights() {
        assert_eq!(weighted_segments("a (red:1.5) car"), vec![("a ".to_string(), 1.0), ("red".to_string(), 1.5), (" car".to_string(), 1.0)]);
        let w = weighted_segments("((big)) dog");
        assert!((w[0].1 - 1.21).abs() < 1e-5 && w[0].0 == "big");
        assert_eq!(weighted_segments("smile \\(face\\)"), vec![("smile (face)".to_string(), 1.0)]);
    }

    #[test]
    fn chunks_at_75_tokens() {
        let t = ClipTokenizer::new().unwrap();
        let c = chunks(&t, "");
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].ids[0], Some(BOS));
        assert_eq!(c[0].ids[1], Some(EOS));
        assert_eq!(c[0].ids[2], None);
        assert_eq!(c[0].filled(0)[2], 0);
        let long = vec!["cat"; 100].join(" ");
        let c = chunks(&t, &long);
        assert_eq!(c.len(), 2);
        assert!(c.iter().all(|k| k.ids.len() == CHUNK));
        assert_eq!(c[0].ids[76], Some(EOS));
    }
}
