use h3ref2va::tokenizer::Tokenizer;

#[test]
fn tokenizer_matches_huggingface() {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!("tokenizer_cases.json")).unwrap();
    let tok = Tokenizer::new().unwrap();
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let got = tok.encode(text);
        assert_eq!(got, want, "tokenization mismatch for {text:?}");
    }
}
