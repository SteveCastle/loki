//! Video VAE frame bookkeeping on the GPU: decode streams exactly `decode_frame_count(tl)` frames for every
//! latent length, and encode produces the reference latent frame count. Skipped when the model is missing.
//! Run through the GPU lock: bash tools/gpu.sh cargo test --release --test vvae_frames
use loki_reshoot::cuda::Device;
use loki_reshoot::tensor::Tensor;
use loki_reshoot::vae_video::{encode_latent_frames, VideoVae};
use std::path::PathBuf;

#[test]
fn vvae_frame_counts_gpu() {
    let model = PathBuf::from(std::env::var("LOKI_VVAE").unwrap_or_else(|_| r"C:\Users\steph\bin\models\minimax_h3_video_vae_fp16.safetensors".into()));
    if !model.exists() {
        eprintln!("skipping: {} not found", model.display());
        return;
    }
    let dev = Device::new(0).unwrap();
    let vae = VideoVae::load(dev.clone(), &model).unwrap();
    // reference counts from comfy decode_output_shape (computed with the Comfy code for tl = 1..13)
    let expect = [(1usize, 1usize), (2, 5), (3, 9), (4, 13), (5, 17), (6, 18), (7, 22), (8, 26), (9, 30), (10, 34), (11, 35), (12, 39), (13, 43)];
    for &(tl, frames) in &expect {
        assert_eq!(VideoVae::decode_frame_count(tl), frames, "frame count tl={tl}");
    }
    let (hl, wl) = (4, 6);
    for tl in [1usize, 2, 3, 6, 7, 8, 12] {
        let z: Vec<f32> = (0..24 * tl * hl * wl).map(|i| ((i * 7919) % 200) as f32 / 100.0 - 1.0).collect();
        let zt = Tensor::from_f32(&dev, &z, &[24, tl, hl, wl]).unwrap();
        let mut got = 0usize;
        let mut bytes = 0usize;
        let n = vae
            .decode(&zt, tl, hl, wl, &mut |px, n| {
                got += n;
                bytes += px.len();
                Ok(())
            })
            .unwrap();
        assert_eq!(n, VideoVae::decode_frame_count(tl));
        assert_eq!(got, n);
        assert_eq!(bytes, n * hl * 16 * wl * 16 * 3);
    }
    for t in [1usize, 5, 17, 22] {
        let px: Vec<u8> = (0..t * 64 * 96 * 3).map(|i| (i % 251) as u8).collect();
        let lat = vae.encode(&px, t, 64, 96).unwrap();
        assert_eq!(lat.shape, vec![24, encode_latent_frames(t), 4, 6], "encode t={t}");
    }
}
