//! Byte-level BPE (GPT-2 style) read from the GGUF vocabulary (`tokenizer.ggml.tokens` / `merges`), with the Qwen3.5
//! pre-tokenizer (`tokenizer.ggml.pre = qwen35`), implemented by hand because it needs a lookahead:
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! Text passed to `encode` is plain: special tokens are only ever added by id (chat.rs), never parsed out of user text.
use crate::gguf::Gguf;
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::sync::LazyLock;

pub struct Tokenizer {
    /// (left, right) -> (rank, merged id)
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// token id -> raw bytes (None for control tokens)
    bytes: Vec<Option<Vec<u8>>>,
    byte_tok: [u32; 256],
    pub specials: HashMap<String, u32>,
    pub eos: u32,
}

fn bytes_to_unicode() -> [char; 256] {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.iter().zip(cs.iter()) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

impl Tokenizer {
    pub fn from_gguf(g: &Gguf) -> Result<Tokenizer> {
        let model = g.s("tokenizer.ggml.model").unwrap_or("");
        if model != "gpt2" {
            bail!("tokenizer model {model:?} unsupported (want gpt2 byte-level BPE)");
        }
        let pre = g.s("tokenizer.ggml.pre").unwrap_or("");
        if pre != "qwen35" {
            crate::info!("warning: pre-tokenizer {pre:?}; using the qwen35 rules");
        }
        let toks = g.get("tokenizer.ggml.tokens").and_then(|v| v.as_arr()).context("missing tokenizer.ggml.tokens")?;
        let types = g.get("tokenizer.ggml.token_type").and_then(|v| v.as_arr());
        let merges = g.get("tokenizer.ggml.merges").and_then(|v| v.as_arr()).context("missing tokenizer.ggml.merges")?;
        let b2u = bytes_to_unicode();
        let mut u2b: HashMap<char, u8> = HashMap::new();
        for (b, c) in b2u.iter().enumerate() {
            u2b.insert(*c, b as u8);
        }
        let mut encoder = HashMap::with_capacity(toks.len());
        let mut bytes = Vec::with_capacity(toks.len());
        let mut specials = HashMap::new();
        for (i, t) in toks.iter().enumerate() {
            let s = t.as_str().unwrap_or("").to_string();
            let ty = types.and_then(|a| a.get(i)).and_then(|v| v.as_u64()).unwrap_or(1);
            if ty == 3 || ty == 4 {
                specials.insert(s.clone(), i as u32);
            }
            bytes.push(if ty == 3 || ty == 5 {
                None
            } else if ty == 4 {
                Some(s.as_bytes().to_vec())
            } else {
                Some(s.chars().map(|c| u2b.get(&c).copied().unwrap_or(b'?')).collect())
            });
            encoder.insert(s, i as u32);
        }
        let mut mm = HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let Some(m) = m.as_str() else { continue };
            let Some((a, b)) = m.split_once(' ') else { continue };
            let (Some(&ia), Some(&ib)) = (encoder.get(a), encoder.get(b)) else { continue };
            if let Some(&im) = encoder.get(&format!("{a}{b}")) {
                mm.insert((ia, ib), (rank as u32, im));
            }
        }
        let mut byte_tok = [0u32; 256];
        for b in 0..256 {
            byte_tok[b] = *encoder.get(&b2u[b].to_string()).with_context(|| format!("byte token {b} missing"))?;
        }
        let eos = g.get("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()).map(|v| v as u32).or_else(|| specials.get("<|im_end|>").copied()).context("no eos token")?;
        Ok(Tokenizer { merges: mm, bytes, byte_tok, specials, eos })
    }

    pub fn special(&self, s: &str) -> Result<u32> {
        self.specials.get(s).copied().with_context(|| format!("token {s} missing from the vocabulary"))
    }

    pub fn vocab_size(&self) -> usize {
        self.bytes.len()
    }

    /// Encode plain text.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in pretokenize(text) {
            let mut ids: Vec<u32> = piece.bytes().map(|b| self.byte_tok[b as usize]).collect();
            while ids.len() >= 2 {
                let mut best: Option<(u32, usize, u32)> = None;
                for i in 0..ids.len() - 1 {
                    if let Some(&(r, m)) = self.merges.get(&(ids[i], ids[i + 1])) {
                        if best.map(|b| r < b.0).unwrap_or(true) {
                            best = Some((r, i, m));
                        }
                    }
                }
                let Some((_, i, m)) = best else { break };
                ids[i] = m;
                ids.remove(i + 1);
            }
            out.extend(ids);
        }
        out
    }

    /// Raw bytes of a token (empty for control tokens).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).and_then(|b| b.as_deref()).unwrap_or(&[])
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut b = Vec::new();
        for &i in ids {
            b.extend_from_slice(self.token_bytes(i));
        }
        String::from_utf8_lossy(&b).into_owned()
    }
}

// ---------------------------------------------------------------- pre-tokenizer

static RE_L: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"^\p{L}$").unwrap());
static RE_M: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"^\p{M}$").unwrap());
static RE_N: LazyLock<regex::Regex> = LazyLock::new(|| regex::Regex::new(r"^\p{N}$").unwrap());

#[derive(Clone, Copy)]
struct Cls {
    l: bool,
    m: bool,
    n: bool,
    ws: bool,
    nl: bool,
}

fn classify(c: char) -> Cls {
    if c.is_ascii() {
        return Cls { l: c.is_ascii_alphabetic(), m: false, n: c.is_ascii_digit(), ws: c.is_ascii_whitespace() || c == '\u{0b}', nl: c == '\r' || c == '\n' };
    }
    let mut buf = [0u8; 4];
    let s: &str = c.encode_utf8(&mut buf);
    Cls { l: RE_L.is_match(s), m: RE_M.is_match(s), n: RE_N.is_match(s), ws: c.is_whitespace(), nl: false }
}

/// Split text into pre-tokens following the qwen35 pattern (leftmost alternative wins, as in the regex).
pub fn pretokenize(text: &str) -> Vec<&str> {
    let idx: Vec<(usize, char)> = text.char_indices().collect();
    let cls: Vec<Cls> = idx.iter().map(|(_, c)| classify(*c)).collect();
    let n = idx.len();
    let at = |i: usize| if i < n { idx[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = idx[i].1;
        let end = (|| {
            // 1. contractions
            if c == '\'' && i + 1 < n {
                let l1 = idx[i + 1].1.to_ascii_lowercase();
                if matches!(l1, 's' | 't' | 'm' | 'd') {
                    return i + 2;
                }
                if i + 2 < n {
                    let l2 = idx[i + 2].1.to_ascii_lowercase();
                    if (l1 == 'r' && l2 == 'e') || (l1 == 'v' && l2 == 'e') || (l1 == 'l' && l2 == 'l') {
                        return i + 3;
                    }
                }
            }
            // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
            let lm = |k: usize| k < n && (cls[k].l || cls[k].m);
            let prefix_ok = !(cls[i].nl || cls[i].l || cls[i].n);
            if prefix_ok && lm(i + 1) {
                let mut j = i + 1;
                while lm(j) {
                    j += 1;
                }
                return j;
            }
            if lm(i) {
                let mut j = i;
                while lm(j) {
                    j += 1;
                }
                return j;
            }
            // 3. \p{N}
            if cls[i].n {
                return i + 1;
            }
            // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
            let punct = |k: usize| k < n && !(cls[k].ws || cls[k].l || cls[k].m || cls[k].n);
            let start = if c == ' ' && punct(i + 1) { Some(i + 1) } else if punct(i) { Some(i) } else { None };
            if let Some(mut j) = start {
                while punct(j) {
                    j += 1;
                }
                while j < n && cls[j].nl {
                    j += 1;
                }
                return j;
            }
            // whitespace run
            let mut e = i;
            while e < n && cls[e].ws {
                e += 1;
            }
            if e > i {
                // 5. \s*[\r\n]+ : up to and including the last newline of the run
                if let Some(last_nl) = (i..e).rev().find(|&k| cls[k].nl) {
                    return last_nl + 1;
                }
                // 6. \s+(?!\S)
                if e == n {
                    return e;
                }
                if e - i > 1 {
                    return e - 1;
                }
                // 7. \s+
                return e;
            }
            i + 1
        })();
        out.push(&text[at(i)..at(end)]);
        i = end;
    }
    out
}
