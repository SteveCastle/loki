//! Host-side helpers of the audio VAE module (no GPU needed). GPU accuracy is checked by `avae_check`
//! against ComfyUI dumps (ref/ref_avae.py).
use loki_reshoot::vae_audio::{comfy_crop_window, resample};

#[test]
fn crop_window_matches_comfy() {
    // vae_encode_crop_pixels: keep (n // 800) * 800 samples starting at (n % 800) // 2
    assert_eq!(comfy_crop_window(32000), (0, 32000));
    assert_eq!(comfy_crop_window(32799), (399, 32000));
    assert_eq!(comfy_crop_window(801), (0, 800));
}

#[test]
fn resample_lengths_and_dc() {
    let x = vec![0.5f32; 44100];
    let y = resample(&x, 44100, 32000);
    assert_eq!(y.len(), 32000);
    // interior of a DC signal stays DC (sinc kernel sums to ~1)
    for v in &y[100..31900] {
        assert!((v - 0.5).abs() < 2e-3, "{v}");
    }
    assert_eq!(resample(&x, 32000, 32000), x);
    assert_eq!(resample(&vec![0.0f32; 48000], 48000, 32000).len(), 32000);
}
