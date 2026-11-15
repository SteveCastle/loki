//! Latent upscaler against the ComfyUI node's own network (ref/ref_upscaler.py, fp32 on CPU): every
//! LOKI_UPSCALER_REF directory (a path list; one with T > 32 exercises the temporal chunking) must match within fp16
//! tolerance. Skipped when the reference or the weights are missing.
//! Run through the GPU lock: LOKI_UPSCALER_REF=<dir>[;<dir>] bash tools/gpu.sh cargo test --release --test upscaler
use loki_reshoot::cuda::Device;
use loki_reshoot::tensor::Tensor;
use loki_reshoot::upscaler::Upscaler;
use std::path::PathBuf;

fn read_f32(p: &std::path::Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
fn upscaler_matches_reference_gpu() {
    let Some(refs) = std::env::var_os("LOKI_UPSCALER_REF") else {
        eprintln!("skipping: LOKI_UPSCALER_REF not set");
        return;
    };
    let weights = PathBuf::from(std::env::var("LOKI_UPSCALER").unwrap_or_else(|_| r"C:\Users\steph\bin\models\minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors".into()));
    if !weights.exists() {
        eprintln!("skipping: {} not found", weights.display());
        return;
    }
    let dev = Device::new(0).unwrap();
    let up = Upscaler::load(dev.clone(), &weights).unwrap();
    for dir in std::env::split_paths(&refs) {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let g = |k: &str| meta[k].as_u64().unwrap() as usize;
        let (t, h, w, ho, wo) = (g("T"), g("H"), g("W"), g("HO"), g("WO"));
        let z = read_f32(&dir.join("latent.bin"));
        let want = read_f32(&dir.join("out.bin"));
        let zt = Tensor::from_f32(&dev, &z, &[24, t, h, w]).unwrap();
        let got = up.upscale(&zt, ho, wo).unwrap().to_f32_vec(&dev).unwrap();
        assert_eq!(got.len(), want.len());
        let (mut num, mut den, mut max) = (0f64, 0f64, 0f32);
        for (a, b) in got.iter().zip(&want) {
            num += ((a - b) as f64).powi(2);
            den += (*b as f64).powi(2);
            max = max.max((a - b).abs());
        }
        let rel = (num / den).sqrt();
        // per-frame error, to catch chunk-seam mistakes that a global norm would average away
        let hw = ho * wo;
        let mut worst = (0usize, 0f64);
        for f in 0..t {
            let (mut n, mut d) = (0f64, 0f64);
            for c in 0..24 {
                for k in 0..hw {
                    let i = (c * t + f) * hw + k;
                    n += ((got[i] - want[i]) as f64).powi(2);
                    d += (want[i] as f64).powi(2);
                }
            }
            let r = (n / d).sqrt();
            if r > worst.1 {
                worst = (f, r);
            }
        }
        eprintln!("{}: {t}x{h}x{w} -> {ho}x{wo}: rel L2 {rel:.5}, max abs {max:.4}, worst frame {} ({:.5})", dir.display(), worst.0, worst.1);
        assert!(rel < 0.01 && worst.1 < 0.02, "upscaler mismatch vs reference");
    }
}
