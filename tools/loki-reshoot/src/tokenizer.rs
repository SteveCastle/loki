//! Qwen2 byte-level BPE tokenizer (GPT-2 style), with the Qwen2.5 pre-tokenization pattern:
//!   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//! Vocabulary and merges are embedded in the binary.
use anyhow::{anyhow, Result};
use std::collections::HashMap;

const VOCAB_JSON: &str = include_str!("../assets/vocab.json");
const MERGES_TXT: &str = include_str!("../assets/merges.txt");

pub const IM_START: u32 = 151644;
pub const IM_END: u32 = 151645;
pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
pub const IMAGE_PAD: u32 = 151655;
pub const ENDOFTEXT: u32 = 151643;

pub struct Tokenizer {
    encoder: HashMap<String, u32>,
    ranks: HashMap<(u32, u32), u32>,
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

impl Tokenizer {
    pub fn new() -> Result<Tokenizer> {
        let encoder: HashMap<String, u32> = serde_json::from_str(VOCAB_JSON)?;
        let mut ranks = HashMap::new();
        for (i, line) in MERGES_TXT.lines().enumerate() {
            if line.starts_with("#version") || line.trim().is_empty() {
                continue;
            }
            let mut it = line.split(' ');
            let a = it.next().ok_or_else(|| anyhow!("bad merge line {i}"))?;
            let b = it.next().ok_or_else(|| anyhow!("bad merge line {i}"))?;
            let ia = *encoder.get(a).ok_or_else(|| anyhow!("merge token {a} not in vocab"))?;
            let ib = *encoder.get(b).ok_or_else(|| anyhow!("merge token {b} not in vocab"))?;
            ranks.insert((ia, ib), i as u32);
        }
        let special = vec![
            ("<|endoftext|>".to_string(), ENDOFTEXT),
            ("<|im_start|>".to_string(), IM_START),
            ("<|im_end|>".to_string(), IM_END),
            ("<|object_ref_start|>".to_string(), 151646),
            ("<|object_ref_end|>".to_string(), 151647),
            ("<|box_start|>".to_string(), 151648),
            ("<|box_end|>".to_string(), 151649),
            ("<|quad_start|>".to_string(), 151650),
            ("<|quad_end|>".to_string(), 151651),
            ("<|vision_start|>".to_string(), VISION_START),
            ("<|vision_end|>".to_string(), VISION_END),
            ("<|vision_pad|>".to_string(), 151654),
            ("<|image_pad|>".to_string(), IMAGE_PAD),
            ("<|video_pad|>".to_string(), 151656),
        ];
        Ok(Tokenizer { encoder, ranks, byte_to_unicode: bytes_to_unicode(), special })
    }

    /// Encode text, recognizing the special tokens literally.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // find earliest special token
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
        for piece in pretokenize(text) {
            let mapped: String = piece.bytes().map(|b| self.byte_to_unicode[b as usize]).collect();
            // initial symbols: one per unicode char of the mapped string
            let mut ids: Vec<u32> = mapped.chars().map(|c| *self.encoder.get(&c.to_string()).expect("byte token missing")).collect();
            // merge loop
            loop {
                if ids.len() < 2 {
                    break;
                }
                let mut best_rank = u32::MAX;
                let mut best_i = usize::MAX;
                for i in 0..ids.len() - 1 {
                    if let Some(r) = self.ranks.get(&(ids[i], ids[i + 1])) {
                        if *r < best_rank {
                            best_rank = *r;
                            best_i = i;
                        }
                    }
                }
                if best_i == usize::MAX {
                    break;
                }
                let merged = self.merge_id(ids[best_i], ids[best_i + 1]);
                ids[best_i] = merged;
                ids.remove(best_i + 1);
            }
            out.extend(ids);
        }
    }

    fn merge_id(&self, a: u32, b: u32) -> u32 {
        // token string of the merge = concat of the two token strings
        let sa = self.decode_token(a);
        let sb = self.decode_token(b);
        *self.encoder.get(&format!("{sa}{sb}")).expect("merged token missing from vocab")
    }

    fn decode_token(&self, id: u32) -> String {
        // slow path only used during merges; cache would be faster but vocab lookups are fine for short prompts
        self.decoder().get(&id).cloned().unwrap_or_default()
    }

    fn decoder(&self) -> &HashMap<u32, String> {
        use std::sync::OnceLock;
        static DEC: OnceLock<HashMap<u32, String>> = OnceLock::new();
        DEC.get_or_init(|| self.encoder.iter().map(|(k, v)| (*v, k.clone())).collect())
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}
fn is_number(c: char) -> bool {
    c.is_numeric()
}

/// Implements the Qwen2 pre-tokenization regex by hand.
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
            // \s+(?!\S): if followed by non-space, leave the last whitespace char for the next token
            if j < n && j - i > 1 {
                j -= 1;
            }
            out.push(chars[i..j].iter().collect());
            i = j;
            continue;
        }
        // fallback: single char (should not happen)
        out.push(c.to_string());
        i += 1;
    }
    out
}
