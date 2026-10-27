//! The CLIP BPE tokenizer against cases produced by the official SAM 3 tokenizer (ref env, `tokenizer_ve.py`).
use loki_segment::text::Tokenizer;

#[test]
fn matches_the_reference_tokenizer() {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!("tokenizer_cases.json")).unwrap();
    let tok = Tokenizer::new().unwrap();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        assert_eq!(tok.tokenize(text), want, "tokenizing {text:?}");
    }
}

