use loki_dream::tokenizer::Tokenizer;

/// Fixtures from the Hugging Face tokenizer of lodestones/Kromoss (ref/ref_dream.py tokens), chat template included.
#[test]
fn tokenizer_matches_huggingface() {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!("tokenizer_cases.json")).unwrap();
    let tok = Tokenizer::new().unwrap();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.encode(text), want, "tokenization mismatch for {text:?}");
    }
}

#[test]
fn template_matches_the_official_chat_template() {
    assert_eq!(
        loki_dream::text_encoder::template("  a cat \n"),
        "<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}

/// Fixtures from the Hugging Face Qwen3 tokenizer (Anima / FLUX.2 klein / Krea 2 ship the same vocabulary), raw text.
#[test]
fn qwen3_tokenizer_matches_huggingface() {
    let cases: Vec<(String, Vec<u32>)> = serde_json::from_str(include_str!("qwen3_cases.json")).unwrap();
    let tok = Tokenizer::qwen3().unwrap();
    for (text, want) in cases {
        assert_eq!(tok.encode(&text), want, "tokenization mismatch for {text:?}");
    }
}
