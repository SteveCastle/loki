//! CPU-side checks of the MiniMax H3 conditioning presentation (no GPU needed).
use loki_reshoot::te_h3::{fmt_1f, mrope_table, token_rc, tokenize_piece, vision_size};
use loki_reshoot::tokenizer::Tokenizer;

#[test]
fn timestamps_format_like_python() {
    // Python: "%.1f" % x rounds half to even on the exact binary value
    assert_eq!(fmt_1f(0.25), "0.2");
    assert_eq!(fmt_1f(1.25), "1.2");
    assert_eq!(fmt_1f(0.75), "0.8");
    assert_eq!(fmt_1f(2.0), "2.0");
    assert_eq!(fmt_1f(10.75), "10.8");
}

#[test]
fn vision_sizes_follow_process_qwen2vl_images() {
    assert_eq!(vision_size(192, 320), (192, 320));
    // round(170/32)=5 -> 160, round(250/32)=8 -> 256
    assert_eq!(vision_size(170, 250), (160, 256));
    // banker's rounding: 336/32 = 10.5 -> 10
    assert_eq!(vision_size(336, 336), (320, 320));
    // min_pixels: tiny images are scaled up
    let (h, w) = vision_size(20, 20);
    assert!(h * w >= 3136 && h % 32 == 0 && w % 32 == 0);
    // max_pixels cap
    let (h, w) = vision_size(4000, 6000);
    assert!(h * w <= 12_845_056 && h % 32 == 0 && w % 32 == 0);
}

#[test]
fn merge_window_order() {
    // grid width 4 patches -> 2 merge windows per row
    assert_eq!(token_rc(0, 4), (0, 0));
    assert_eq!(token_rc(1, 4), (0, 1));
    assert_eq!(token_rc(2, 4), (1, 0));
    assert_eq!(token_rc(3, 4), (1, 1));
    assert_eq!(token_rc(4, 4), (0, 2));
    assert_eq!(token_rc(8, 4), (2, 0));
}

#[test]
fn labels_and_extra_tokens() {
    let tok = Tokenizer::new().unwrap();
    // values from ComfyUI's MiniMaxH3Tokenizer (ref_out/te/*/tokens.json)
    assert_eq!(tokenize_piece(&tok, "<Audio 1>: "), vec![65406, 220, 16, 26818, 220]);
    assert_eq!(tokenize_piece(&tok, "<0.2 seconds>"), vec![27, 15, 13, 17, 6486, 29]);
    let t = tokenize_piece(&tok, "a <d> b");
    assert!(t.contains(&151669), "{t:?}");
    assert_eq!(tokenize_piece(&tok, "<Picture 1>: "), vec![21604, 3826, 220, 16, 26818, 220]);
    // escaped parentheses lose their backslash
    assert_eq!(tokenize_piece(&tok, "\\(x\\)"), tokenize_piece(&tok, "(x)"));
}

#[test]
fn mrope_text_positions_are_plain_rope() {
    let pos = [vec![0i64, 1, 2], vec![0, 1, 2], vec![0, 1, 2]];
    let t = mrope_table(&pos);
    assert_eq!(t.len(), 3 * 128);
    // position 0 -> cos 1, sin 0
    assert!((t[0] - 1.0).abs() < 1e-7 && t[1].abs() < 1e-7);
    // frequency 0 at position 2: angle 2
    assert!((t[2 * 128] - 2f32.cos()).abs() < 1e-6);
}
