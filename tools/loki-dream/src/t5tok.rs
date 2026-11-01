//! The T5 tokenizer (SentencePiece Unigram, 32k pieces) that Anima's LLM adapter uses for its query tokens.
//!
//! HF `tokenizer.json` of `t5_tokenizer/`: the SentencePiece precompiled charsmap normalizer (spm_precompiled, the same
//! grapheme-wise replacement the `tokenizers` crate does), whitespace split + Metaspace (every word prefixed with `▁`),
//! Unigram Viterbi over the piece scores (unknown characters get the minimum score - 10 and consecutive unknowns fuse),
//! then `</s>` appended. Vocabulary, scores and charsmap are embedded (assets/t5/).
use anyhow::{Context, Result};
use std::collections::HashMap;

const TOKENIZER_JSON: &str = include_str!("../assets/t5/tokenizer.json");
pub const EOS: u32 = 1;

pub struct T5Tokenizer {
    pieces: HashMap<String, (u32, f64)>,
    max_len: usize,
    min_score: f64,
    unk: u32,
    norm: spm_precompiled::Precompiled,
}

#[derive(serde::Deserialize)]
struct Asset {
    vocab: Vec<(String, f64)>,
    unk_id: u32,
    precompiled_charsmap: String,
}

impl T5Tokenizer {
    pub fn new() -> Result<T5Tokenizer> {
        let a: Asset = serde_json::from_str(TOKENIZER_JSON).context("T5 tokenizer asset")?;
        // deserialized from a borrowed string: spm_precompiled reads the base64 field as &str
        let js = format!("{{\"type\":\"Precompiled\",\"precompiled_charsmap\":\"{}\"}}", a.precompiled_charsmap);
        let norm: spm_precompiled::Precompiled = serde_json::from_str(&js).context("T5 charsmap")?;
        let min_score = a.vocab.iter().map(|v| v.1).fold(f64::INFINITY, f64::min);
        let max_len = a.vocab.iter().map(|v| v.0.len()).max().unwrap_or(1);
        let pieces = a.vocab.into_iter().enumerate().map(|(i, (p, s))| (p, (i as u32, s))).collect();
        Ok(T5Tokenizer { pieces, max_len, min_score, unk: a.unk_id, norm })
    }

    /// Token ids of `text`, `</s>` included.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalized = self.norm.normalize_string(text);
        let mut out = Vec::new();
        for word in normalized.split_whitespace() {
            self.viterbi(&format!("\u{2581}{word}"), &mut out);
        }
        out.push(EOS);
        out
    }

    fn viterbi(&self, s: &str, out: &mut Vec<u32>) {
        let b = s.as_bytes();
        let n = b.len();
        // best[i] = (score, id, start) of the best path ending at byte i
        let mut best: Vec<Option<(f64, u32, usize)>> = vec![None; n + 1];
        best[0] = Some((0.0, 0, 0));
        let unk_score = self.min_score - 10.0;
        let mut i = 0;
        while i < n {
            let here = best[i].map(|x| x.0).unwrap_or(f64::NEG_INFINITY);
            let ch_len = s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            let mut single = false;
            let mut end = i;
            for c in s[i..].chars() {
                end += c.len_utf8();
                if end - i > self.max_len {
                    break;
                }
                if let Some(&(id, score)) = self.pieces.get(&s[i..end]) {
                    let cand = here + score;
                    if best[end].map(|x| cand > x.0).unwrap_or(true) {
                        best[end] = Some((cand, id, i));
                    }
                    if end - i == ch_len {
                        single = true;
                    }
                }
            }
            if !single {
                let cand = here + unk_score;
                if best[i + ch_len].map(|x| cand > x.0).unwrap_or(true) {
                    best[i + ch_len] = Some((cand, self.unk, i));
                }
            }
            i += ch_len;
        }
        let mut ids = Vec::new();
        let mut e = n;
        while e > 0 {
            let (_, id, st) = best[e].expect("unigram lattice has a gap");
            ids.push(id);
            e = st;
        }
        ids.reverse();
        let mut prev_unk = false;
        for id in ids {
            let unk = id == self.unk;
            if !(unk && prev_unk) {
                out.push(id);
            }
            prev_unk = unk;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixtures from the Hugging Face T5TokenizerFast (Anima's t5_tokenizer).
    #[test]
    fn matches_huggingface() {
        let t = T5Tokenizer::new().unwrap();
        for (text, ids) in serde_json::from_str::<Vec<(String, Vec<u32>)>>(include_str!("../tests/t5_cases.json")).unwrap() {
            assert_eq!(t.encode(&text), ids, "T5 tokenization mismatch for {text:?}");
        }
    }
}
