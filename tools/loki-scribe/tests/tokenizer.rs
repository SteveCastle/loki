//! The tokenizer against llama.cpp's reference cases for the qwen35 vocabulary (GPU-free): a vocab-only GGUF and the
//! inputs / expected ids its own tokenizer test uses, fetched once into cargo's test tmpdir.
use loki_scribe::gguf::Gguf;
use loki_scribe::tokenizer::Tokenizer;
use std::path::{Path, PathBuf};

const BASE: &str = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/models/ggml-vocab-qwen35.gguf";

static FETCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fetch(suffix: &str) -> PathBuf {
    let _one_at_a_time = FETCH.lock().unwrap_or_else(|e| e.into_inner());
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("vocab");
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join(format!("ggml-vocab-qwen35.gguf{suffix}"));
    if !dest.exists() {
        let resp = ureq::get(&format!("{BASE}{suffix}")).call().unwrap_or_else(|e| panic!("fetching the vocab fixture{suffix}: {e}"));
        let mut bytes = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader(), &mut bytes).unwrap();
        std::fs::write(&dest, bytes).unwrap();
    }
    dest
}

#[test]
fn matches_llama_cpp_reference() {
    let g = Gguf::open(&fetch("")).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    let inp = std::fs::read_to_string(fetch(".inp")).unwrap();
    let out = std::fs::read_to_string(fetch(".out")).unwrap();
    let cases: Vec<&str> = inp.split("\n__ggml_vocab_test__\n").collect();
    let want: Vec<&str> = out.lines().collect();
    assert!(want.len() >= 40, "fixture has {} cases", want.len());
    let mut bad = Vec::new();
    for (i, w) in want.iter().enumerate() {
        let got = tok.encode(cases[i]).iter().map(|t| t.to_string()).collect::<Vec<_>>().join(" ");
        if got.trim() != w.trim() {
            bad.push(format!("case {i} {:?}\n  got  {got}\n  want {w}", cases[i]));
        }
    }
    assert!(bad.is_empty(), "{} of {} cases differ:\n{}", bad.len(), want.len(), bad.join("\n"));
}

#[test]
fn decode_round_trips() {
    let g = Gguf::open(&fetch("")).unwrap();
    let tok = Tokenizer::from_gguf(&g).unwrap();
    for s in ["Hello, world!", "naïve café — 日本語 🦙\n\n  tabs\tand spaces  ", "MEN WALK ON MOON"] {
        assert_eq!(tok.decode(&tok.encode(s)), s);
    }
}
