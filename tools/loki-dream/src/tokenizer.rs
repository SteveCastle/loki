//! Qwen3.5 byte-level BPE tokenizer (248k vocabulary), the text side of MOSS-Image / Kromoss.
//!
//! HF `tokenizer.json`: NFC normalizer, then the Qwen2 split pattern
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! (implemented by hand in `pretokenize`), byte-level mapping and BPE merges. Every added token (`<|im_start|>`, `<think>`,
//! ...) is matched literally before the split, as the HF tokenizer does. Vocabulary, merges and added tokens are embedded
//! (assets/, extracted from lodestones/Kromoss `tokenizer/tokenizer.json`).
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use unicode_normalization::UnicodeNormalization;

const VOCAB_JSON: &str = include_str!("../assets/vocab.json");
const MERGES_TXT: &str = include_str!("../assets/merges.txt");
const ADDED_JSON: &str = include_str!("../assets/added_tokens.json");

pub const ENDOFTEXT: u32 = 248044;
pub const IM_START: u32 = 248045;
pub const IM_END: u32 = 248046;

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    decoder: Vec<String>,
    ranks: HashMap<(u32, u32), (u32, u32)>, // (a, b) -> (rank, merged id)
    byte_to_unicode: [char; 256],
    special: Vec<(String, u32)>,
}

fn bytes_to_unicode() -> [char; 256] {
    // GPT-2 byte<->unicode mapping
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

#[derive(serde::Deserialize)]
struct Added {
    id: u32,
    content: String,
}

impl Tokenizer {
    pub fn new() -> Result<Tokenizer> {
        let encoder: HashMap<String, u32> = serde_json::from_str(VOCAB_JSON)?;
        let mut decoder = vec![String::new(); encoder.values().copied().max().unwrap_or(0) as usize + 1];
        for (k, v) in &encoder {
            decoder[*v as usize] = k.clone();
        }
        let mut ranks = HashMap::new();
        let mut rank = 0u32;
        for (i, line) in MERGES_TXT.lines().enumerate() {
            if line.starts_with("#version") || line.is_empty() {
                continue;
            }
            let (a, b) = line.split_once(' ').ok_or_else(|| anyhow!("bad merge line {i}"))?;
            let ia = *encoder.get(a).ok_or_else(|| anyhow!("merge token {a} not in vocab"))?;
            let ib = *encoder.get(b).ok_or_else(|| anyhow!("merge token {b} not in vocab"))?;
            let m = *encoder.get(&format!("{a}{b}")).ok_or_else(|| anyhow!("merged token {a}{b} not in vocab"))?;
            ranks.insert((ia, ib), (rank, m));
            rank += 1;
        }
        let added: Vec<Added> = serde_json::from_str(ADDED_JSON)?;
        let mut special: Vec<(String, u32)> = added.into_iter().map(|a| (a.content, a.id)).collect();
        // longest first so that a token that is a prefix of another never wins
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        Ok(Tokenizer { encoder, decoder, ranks, byte_to_unicode: bytes_to_unicode(), special })
    }

    /// Encode text (NFC-normalized), recognizing the added tokens literally.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            let mut best: Option<(usize, usize, u32)> = None;
            for (s, id) in &self.special {
                if let Some(pos) = rest.find(s.as_str()) {
                    if best.map(|b| pos < b.0).unwrap_or(true) {
                        best = Some((pos, s.len(), *id));
                    }
                }
            }
            match best {
                Some((pos, len, id)) => {
                    self.encode_plain(&rest[..pos], &mut out);
                    out.push(id);
                    rest = &rest[pos + len..];
                }
                None => {
                    self.encode_plain(rest, &mut out);
                    rest = "";
                }
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let text: String = text.nfc().collect();
        for piece in pretokenize(&text) {
            let mut ids: Vec<u32> = piece
                .bytes()
                .map(|b| *self.encoder.get(&self.byte_to_unicode[b as usize].to_string()).expect("byte token missing"))
                .collect();
            loop {
                if ids.len() < 2 {
                    break;
                }
                let mut best = (u32::MAX, usize::MAX, 0u32);
                for i in 0..ids.len() - 1 {
                    if let Some(&(r, m)) = self.ranks.get(&(ids[i], ids[i + 1])) {
                        if r < best.0 {
                            best = (r, i, m);
                        }
                    }
                }
                if best.1 == usize::MAX {
                    break;
                }
                ids[best.1] = best.2;
                ids.remove(best.1 + 1);
            }
            out.extend(ids);
        }
    }

    /// Token id -> its vocabulary string (byte-level mapped), for diagnostics.
    pub fn token_str(&self, id: u32) -> &str {
        self.decoder.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// The Qwen2 pre-tokenization regex, by hand.
pub fn pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        // 1. contractions (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if c == '\'' && i + 1 < n {
            let l1 = chars[i + 1].to_ascii_lowercase();
            if matches!(l1, 's' | 't' | 'm' | 'd') {
                out.push(chars[i..i + 2].iter().collect());
                i += 2;
                continue;
            }
            if i + 2 < n {
                let l2 = chars[i + 2].to_ascii_lowercase();
                if (l1 == 'r' && l2 == 'e') || (l1 == 'v' && l2 == 'e') || (l1 == 'l' && l2 == 'l') {
                    out.push(chars[i..i + 3].iter().collect());
                    i += 3;
                    continue;
                }
            }
        }
        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        {
            let mut j = i;
            if !(c == '\r' || c == '\n' || is_letter(c) || is_number(c)) {
                j += 1;
            }
            if j < n && is_letter(chars[j]) {
                while j < n && is_letter(chars[j]) {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        // 3. \p{N}
        if is_number(c) {
            out.push(c.to_string());
            i += 1;
            continue;
        }
        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            let mut j = i;
            if chars[j] == ' ' {
                j += 1;
            }
            if j < n && !(chars[j].is_whitespace() || is_letter(chars[j]) || is_number(chars[j])) {
                while j < n && !(chars[j].is_whitespace() || is_letter(chars[j]) || is_number(chars[j])) {
                    j += 1;
                }
                while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        // 5. \s*[\r\n]+
        {
            let mut j = i;
            while j < n && chars[j].is_whitespace() && chars[j] != '\r' && chars[j] != '\n' {
                j += 1;
            }
            if j < n && (chars[j] == '\r' || chars[j] == '\n') {
                while j < n && (chars[j] == '\r' || chars[j] == '\n') {
                    j += 1;
                }
                out.push(chars[i..j].iter().collect());
                i = j;
                continue;
            }
        }
        // 6. \s+(?!\S)  and 7. \s+
        if c.is_whitespace() {
            let mut j = i;
            while j < n && chars[j].is_whitespace() {
                j += 1;
            }
            if j < n && j - i > 1 {
                j -= 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        out.push(c.to_string());
        i += 1;
    }
    out
}
