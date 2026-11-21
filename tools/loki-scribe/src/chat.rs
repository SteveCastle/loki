//! The Qwen3.5/3.8 chat template (as shipped in the GGUF's `tokenizer.chat_template`) for one user turn:
//!
//!   [<|im_start|>system\n{reasoning instructions}\n\n{system}<|im_end|>\n]
//!   <|im_start|>user\n{<|vision_start|><|image_pad|>*N<|vision_end|> per image}{text}<|im_end|>\n
//!   <|im_start|>assistant\n<think>\n            (thinking on)
//!   <|im_start|>assistant\n<think>\n\n</think>\n\n   (thinking off)
//!
//! Plain text between special tokens is BPE-encoded as one piece, exactly as when the rendered template is tokenized.
use crate::tokenizer::Tokenizer;
use anyhow::Result;

/// How hard the model thinks before answering (the template's `reasoning_effort`), or no thinking at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Think {
    Off,
    Low,
    Medium,
    High,
}

impl Think {
    pub fn parse(s: &str) -> Option<Think> {
        Some(match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "no" | "false" => Think::Off,
            "low" => Think::Low,
            "medium" => Think::Medium,
            // the template's default effort when thinking is enabled
            "high" | "xhigh" | "on" | "yes" | "true" => Think::High,
            _ => return None,
        })
    }
    fn instructions(self) -> &'static str {
        match self {
            Think::High => "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.",
            Think::Low => "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.",
            _ => "",
        }
    }
}

/// One element of the prompt sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    Tok(u32),
    /// image `i` of the request: its vision embeddings go here
    Image(usize),
}

struct Builder<'a> {
    tok: &'a Tokenizer,
    out: Vec<Piece>,
    text: String,
}

impl Builder<'_> {
    fn text(&mut self, s: &str) {
        self.text.push_str(s);
    }
    fn flush(&mut self) {
        if !self.text.is_empty() {
            let t = std::mem::take(&mut self.text);
            self.out.extend(self.tok.encode(&t).into_iter().map(Piece::Tok));
        }
    }
    fn special(&mut self, s: &str) -> Result<()> {
        self.flush();
        self.out.push(Piece::Tok(self.tok.special(s)?));
        Ok(())
    }
}

/// Build the prompt for `n_images` images followed by `user`.
pub fn prompt(tok: &Tokenizer, system: Option<&str>, n_images: usize, user: &str, think: Think) -> Result<Vec<Piece>> {
    let mut b = Builder { tok, out: Vec::new(), text: String::new() };
    let instr = think.instructions();
    let sys = system.map(str::trim).filter(|s| !s.is_empty());
    if sys.is_some() || (think != Think::Off && !instr.is_empty()) {
        b.special("<|im_start|>")?;
        b.text("system\n");
        match sys {
            Some(s) if think != Think::Off && !instr.is_empty() => {
                b.text(instr);
                b.text("\n\n");
                b.text(s);
            }
            Some(s) => b.text(s),
            None => b.text(instr),
        }
        b.special("<|im_end|>")?;
        b.text("\n");
    }
    b.special("<|im_start|>")?;
    b.text("user\n");
    for i in 0..n_images {
        b.special("<|vision_start|>")?;
        b.out.push(Piece::Image(i));
        b.special("<|vision_end|>")?;
    }
    b.text(user.trim());
    b.special("<|im_end|>")?;
    b.text("\n");
    b.special("<|im_start|>")?;
    b.text("assistant\n");
    b.special("<think>")?;
    if think == Think::Off {
        b.text("\n\n");
        b.special("</think>")?;
        b.text("\n\n");
    } else {
        b.text("\n");
    }
    b.flush();
    Ok(b.out)
}

/// The prompt as text, images shown as `<|image_pad|>` x their token count (for --show-prompt).
pub fn render(tok: &Tokenizer, pieces: &[Piece], image_tokens: &[usize]) -> String {
    let mut s = String::new();
    let rev: std::collections::HashMap<u32, &str> = tok.specials.iter().map(|(k, v)| (*v, k.as_str())).collect();
    let mut run: Vec<u32> = Vec::new();
    let flush = |run: &mut Vec<u32>, s: &mut String| {
        s.push_str(&tok.decode(run));
        run.clear();
    };
    for p in pieces {
        match p {
            Piece::Tok(t) => {
                if let Some(name) = rev.get(t) {
                    flush(&mut run, &mut s);
                    s.push_str(name);
                } else {
                    run.push(*t);
                }
            }
            Piece::Image(i) => {
                flush(&mut run, &mut s);
                s.push_str(&format!("<|image_pad|>x{}", image_tokens.get(*i).copied().unwrap_or(0)));
            }
        }
    }
    flush(&mut run, &mut s);
    s
}
