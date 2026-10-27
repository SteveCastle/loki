//! Concept-prompt text path: the CLIP BPE tokenizer (open_clip `SimpleTokenizer`, vocabulary embedded) and the
//! SAM 3 text encoder (24-layer causal transformer, width 1024, context 32) followed by the 1024 -> 256 resizer.
use crate::cuda::Device;
use crate::ops::{self, Act, AttnArgs, AttnView, Epi};
use crate::tensor::{DType, Tensor};
use crate::weights::{Linear, Norm, Weights};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;

pub const CTX: usize = 32;
const WIDTH: usize = 1024;
const HEADS: usize = 16;
const BPE_GZ: &[u8] = include_bytes!("../assets/bpe_simple_vocab_16e6.txt.gz");

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(String, String), usize>,
    byte_enc: Vec<char>,
    pat: regex::Regex,
    pub sot: u32,
    pub eot: u32,
}

/// open_clip's bytes_to_unicode: (byte -> char table, chars in the mapping's insertion order, which is the order
/// of the first 256 vocabulary entries).
fn bytes_to_unicode() -> (Vec<char>, Vec<char>) {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec![' '; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    (table, cs.iter().map(|c| char::from_u32(*c).unwrap()).collect())
}

impl Tokenizer {
    pub fn new() -> Result<Tokenizer> {
        let mut text = String::new();
        flate2::read::GzDecoder::new(BPE_GZ).read_to_string(&mut text).context("decompressing the BPE vocabulary")?;
        let lines: Vec<&str> = text.split('\n').collect();
        let merges: Vec<(String, String)> = lines[1..49152 - 256 - 2 + 1]
            .iter()
            .map(|l| {
                let mut it = l.split_whitespace();
                (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
            })
            .collect();
        let (byte_enc, ordered) = bytes_to_unicode();
        let mut vocab: Vec<String> = ordered.iter().map(|c| c.to_string()).collect();
        vocab.extend(ordered.iter().map(|c| format!("{c}</w>")));
        for (a, b) in &merges {
            vocab.push(format!("{a}{b}"));
        }
        vocab.push("<start_of_text>".into());
        vocab.push("<end_of_text>".into());
        let encoder: HashMap<String, u32> = vocab.into_iter().enumerate().map(|(i, s)| (s, i as u32)).collect();
        let ranks = merges.into_iter().enumerate().map(|(i, m)| (m, i)).collect();
        let pat = regex::Regex::new(r"(?i)<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+")?;
        let sot = encoder["<start_of_text>"];
        let eot = encoder["<end_of_text>"];
        Ok(Tokenizer { encoder, ranks, byte_enc, pat, sot, eot })
    }

    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        if let Some(last) = word.last_mut() {
            last.push_str("</w>");
        }
        if word.len() < 2 {
            return word;
        }
        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..word.len() - 1 {
                if let Some(&r) = self.ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.map(|b| r < b.0).unwrap_or(true) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, bi)) = best else { break };
            // merge every occurrence of the best pair, left to right (as the reference does)
            let (first, second) = (word[bi].clone(), word[bi + 1].clone());
            let mut nw = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    nw.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    nw.push(word[i].clone());
                    i += 1;
                }
            }
            word = nw;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// Token ids of `text` without sot/eot ("lower" cleaning: whitespace collapsed, lowercase).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let clean = text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let mut out = Vec::new();
        for m in self.pat.find_iter(&clean) {
            let t: String = m.as_str().bytes().map(|b| self.byte_enc[b as usize]).collect();
            for piece in self.bpe(&t) {
                if let Some(id) = self.encoder.get(&piece) {
                    out.push(*id);
                }
            }
        }
        out
    }

    /// [sot] + ids + [eot], truncated to the context (last token forced to eot), zero padded.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut t = vec![self.sot];
        t.extend(self.encode(text));
        t.push(self.eot);
        if t.len() > CTX {
            t.truncate(CTX);
            t[CTX - 1] = self.eot;
        }
        t.resize(CTX, 0);
        t
    }
}

struct Block {
    ln1: Norm,
    qkv: Linear,
    out: Linear,
    ln2: Norm,
    fc: Linear,
    proj: Linear,
}

pub struct TextEncoder {
    dev: Arc<Device>,
    tok_emb: Vec<half::f16>,
    pos: Vec<f32>,
    blocks: Vec<Block>,
    ln_final: Norm,
    resizer: Linear,
    pub tokenizer: Tokenizer,
}

/// Encoded concept prompt: features f32 [32, 256] and the validity of each of the 32 positions.
#[derive(Clone)]
pub struct TextFeatures {
    pub feats: Tensor,
    pub valid: Vec<bool>,
    pub tokens: Vec<u32>,
}

impl TextEncoder {
    pub fn load(w: &Weights, prefix: &str) -> Result<TextEncoder> {
        let p = format!("{prefix}.encoder");
        let emb_name = format!("{p}.token_embedding.weight");
        let bytes = w.st.bytes(&emb_name)?;
        anyhow::ensure!(w.st.info(&emb_name)?.dtype == "F16", "token embedding must be F16");
        let tok_emb: Vec<half::f16> = bytes.chunks_exact(2).map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]]))).collect();
        let pos = w.host(&format!("{p}.positional_embedding"))?;
        let mut blocks = Vec::new();
        for i in 0..24 {
            let b = format!("{p}.transformer.resblocks.{i}");
            blocks.push(Block {
                ln1: w.norm(&format!("{b}.ln_1"), 1e-5)?,
                qkv: w.in_proj(&format!("{b}.attn"), None)?,
                out: w.linear(&format!("{b}.attn.out_proj"))?,
                ln2: w.norm(&format!("{b}.ln_2"), 1e-5)?,
                fc: w.linear(&format!("{b}.mlp.c_fc"))?,
                proj: w.linear(&format!("{b}.mlp.c_proj"))?,
            });
        }
        Ok(TextEncoder {
            dev: w.dev.clone(),
            tok_emb,
            pos,
            blocks,
            ln_final: w.norm(&format!("{p}.ln_final"), 1e-5)?,
            resizer: w.linear(&format!("{prefix}.resizer"))?,
            tokenizer: Tokenizer::new()?,
        })
    }

    pub fn encode(&self, text: &str) -> Result<TextFeatures> {
        let dev = &*self.dev;
        let tokens = self.tokenizer.tokenize(text);
        let mut x0 = vec![0f32; CTX * WIDTH];
        for (i, &t) in tokens.iter().enumerate() {
            for c in 0..WIDTH {
                x0[i * WIDTH + c] = self.tok_emb[t as usize * WIDTH + c].to_f32() + self.pos[i * WIDTH + c];
            }
        }
        let x = Tensor::from_f32(dev, &x0, &[CTX, WIDTH])?;
        let h = Tensor::new(dev, DType::BF16, &[CTX, WIDTH])?;
        let qkv = Tensor::new(dev, DType::BF16, &[CTX, 3 * WIDTH])?;
        let att = Tensor::new(dev, DType::BF16, &[CTX, WIDTH])?;
        let mlp = Tensor::new(dev, DType::BF16, &[CTX, 4 * WIDTH])?;
        let hd = WIDTH / HEADS;
        for b in &self.blocks {
            ops::layernorm(dev, &x, Some(&b.ln1), 1e-5, &h, None)?;
            ops::linear(dev, &h, &b.qkv, Act::None, Epi::Store, &qkv)?;
            ops::attention(
                dev,
                &AttnArgs {
                    q: AttnView::new(&qkv, 0, 3 * WIDTH, hd, CTX),
                    k: AttnView::new(&qkv, WIDTH, 3 * WIDTH, hd, CTX),
                    v: AttnView::new(&qkv, 2 * WIDTH, 3 * WIDTH, hd, CTX),
                    out: AttnView::new(&att, 0, WIDTH, hd, CTX),
                    heads: HEADS,
                    d: hd,
                    batch: 1,
                    causal: true,
                    kmask: None,
                    bias: None,
                },
            )?;
            ops::linear(dev, &att, &b.out, Act::None, Epi::AddRes(&x), &x)?;
            ops::layernorm(dev, &x, Some(&b.ln2), 1e-5, &h, None)?;
            ops::linear(dev, &h, &b.fc, Act::GeluErf, Epi::Store, &mlp)?;
            ops::linear(dev, &mlp, &b.proj, Act::None, Epi::AddRes(&x), &x)?;
        }
        ops::layernorm(dev, &x, Some(&self.ln_final), 1e-5, &h, None)?;
        let feats = ops::linear_new(dev, &h, &self.resizer, Act::None, DType::F32)?;
        let valid = tokens.iter().map(|t| *t != 0).collect();
        Ok(TextFeatures { feats, valid, tokens })
    }
}
