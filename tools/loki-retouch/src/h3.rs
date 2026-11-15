//! MiniMax H3 Image as loki-retouch's second editing model (`--model h3`): loki-reshoot's H3 engine
//! (`loki_reshoot::h3image`) driven with this tool's residency cache, cancellation and progress output.
//!
//! The input is `<Picture 1>` (the picture the edit changes) and every --ref adds `<Picture 2>`, `<Picture 3>`, ... (up
//! to 16 in all). Prompts written for Qwen Image's `<imageN>` tags are translated, so presets and habits carry over.
use anyhow::Result;
use loki_reshoot::h3image::Host;
use std::any::Any;
use std::path::Path;
use std::rc::Rc;

/// The model names `--model` takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Qwen,
    H3,
}

impl Model {
    pub fn parse(s: &str) -> Result<Model> {
        Ok(match s.to_ascii_lowercase().replace(['-', '_', '.', ' '], "").as_str() {
            "qwen" | "qwenimage" | "qwenimage21" | "qwen21" => Model::Qwen,
            "h3" | "h3image" | "minimax" | "minimaxh3" | "minimaxh3image" => Model::H3,
            o => anyhow::bail!("unknown --model {o:?} (qwen = Qwen Image 2.1, h3 = MiniMax H3 Image)"),
        })
    }
}

/// Whether a diffusion model file is MiniMax H3 Image (`--dit FILE` then picks the H3 engine by itself).
pub fn is_h3_file(p: &Path) -> bool {
    loki_reshoot::safetensors::SafeTensors::open(p).map(|st| loki_reshoot::h3image::is_h3_image(&|n| st.has(n))).unwrap_or(false)
}

/// `<imageN>` (Qwen Image's reference tags, any case) -> `<Picture N>` (MiniMax H3's).
pub fn picture_tags(prompt: &str) -> String {
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    while let Some(i) = rest.to_ascii_lowercase().find("<image") {
        let tail = &rest[i + 6..];
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && tail[digits.len()..].starts_with('>') {
            out.push_str(&rest[..i]);
            out.push_str(&format!("<Picture {digits}>"));
            rest = &tail[digits.len() + 1..];
        } else {
            out.push_str(&rest[..i + 6]);
            rest = tail;
        }
    }
    out.push_str(rest);
    out
}

/// This tool's worker cache, cancel flag and progress output as the H3 engine's host.
pub struct RetouchHost {
    pub bar: indicatif::ProgressBar,
    pub show_bar: bool,
}

impl Host for RetouchHost {
    fn load_any(&self, key: &str, file: &Path, load: &mut dyn FnMut() -> Result<Rc<dyn Any>>) -> Result<Rc<dyn Any>> {
        // the cache stores the engine's model behind one more Rc
        crate::worker::load(key, file, || load()).map(|r: Rc<Rc<dyn Any>>| (*r).clone())
    }
    fn check_cancel(&self) -> Result<()> {
        crate::worker::check_cancel()
    }
    fn progress(&self, i: usize, n: usize) {
        self.bar.set_position(i as u64);
        if !self.show_bar {
            crate::info!("step {i}/{n}"); // machine-readable progress for callers that capture stderr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_reference_tags() {
        assert_eq!(picture_tags("put the jacket of <image2> on <image1>"), "put the jacket of <Picture 2> on <Picture 1>");
        assert_eq!(picture_tags("<IMAGE12> and <image> and <imagex>"), "<Picture 12> and <image> and <imagex>");
        assert_eq!(picture_tags("no tags"), "no tags");
    }

    #[test]
    fn parses_model_names() {
        assert_eq!(Model::parse("H3").unwrap(), Model::H3);
        assert_eq!(Model::parse("MiniMax-H3-Image").unwrap(), Model::H3);
        assert_eq!(Model::parse("qwen-image-2.1").unwrap(), Model::Qwen);
        assert!(Model::parse("flux").is_err());
    }
}
