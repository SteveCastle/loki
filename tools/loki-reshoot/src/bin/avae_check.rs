//! Compare the Rust MiniMax H3 audio VAE against the ComfyUI reference dumps (ref/ref_avae.py) and benchmark it.
//! Usage: avae_check [ref_dir] [model]
//!   defaults: C:\Users\steph\dev\loki\tools\loki-reshoot\ref_out\avae, C:\Users\steph\bin\models\minimax_h3_audio_vae_fp32.safetensors
use anyhow::Result;
use loki_reshoot::cuda::Device;
use loki_reshoot::tensor::Tensor;
use loki_reshoot::vae_audio::AudioVae;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

/// (max-abs, rel-L2, SNR dB)
fn cmp(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        let d = *a as f64 - *b as f64;
        num += d * d;
        den += (*b as f64) * (*b as f64);
        mx = mx.max(d.abs());
    }
    let rel = (num / den.max(1e-30)).sqrt();
    (mx, rel, -20.0 * rel.max(1e-12).log10())
}
fn report(name: &str, got: &[f32], want: &[f32]) -> f64 {
    let (mx, rel, snr) = cmp(got, want);
    println!("  {name:<44} max-abs {mx:.3e}  rel-L2 {rel:.3e}  SNR {snr:6.2} dB");
    rel
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ref_dir = PathBuf::from(args.get(1).cloned().unwrap_or(r"C:\Users\steph\dev\loki\tools\loki-reshoot\ref_out\avae".into()));
    let model = PathBuf::from(args.get(2).cloned().unwrap_or(r"C:\Users\steph\bin\models\minimax_h3_audio_vae_fp32.safetensors".into()));
    let dev = Device::new(0)?;
    let free0 = dev.free_mem()?;
    let t0 = Instant::now();
    let mut vae = AudioVae::load(dev.clone(), &model)?;
    dev.sync()?;
    let free_loaded = dev.free_mem()?;
    println!("loaded in {:.2}s, weights {} MB", t0.elapsed().as_secs_f64(), (free0 - free_loaded) >> 20);

    if std::env::var("LOKI_PROFILE").is_ok() {
        // profile mode: stereo encode + decode (AVAE_PROF_SECS, default 5 s), per-kernel GPU time
        let secs: usize = std::env::var("AVAE_PROF_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let n = secs * 32000;
        let wav: Vec<f32> = (0..2 * n).map(|i| 0.3 * ((i % n) as f32 * 0.031).sin()).collect();
        let z = vae.encode(&wav, n)?;
        let _ = vae.decode(&z, n / 800)?;
        let mut vae = vae;
        vae.prof = loki_reshoot::cuda::Profiler::new();
        let z = vae.encode(&wav, n)?;
        let _ = vae.decode(&z, n / 800)?;
        vae.prof.report();
        return Ok(());
    }
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(ref_dir.join("meta.json"))?)?;
    let mut worst = 0f64;
    for (name, c) in meta["cases"].as_object().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let t = (n + 799) / 800;
        println!("== {name}: {n} samples/ch ({:.2} s), T = {t}", n as f64 / 32000.0);
        let wav = read_f32(&ref_dir.join(format!("{name}_wav.bin")));
        let z = vae.encode(&wav, n)?;
        let zh = z.to_f32_vec(&dev)?;
        let lat32 = read_f32(&ref_dir.join(format!("{name}_fp32_latent.bin")));
        let lattf = read_f32(&ref_dir.join(format!("{name}_tf32_latent.bin")));
        worst = worst.max(report("encode latent vs comfy fp32", &zh, &lat32));
        report("encode latent vs comfy default (tf32 convs)", &zh, &lattf);
        report("  (comfy tf32 vs comfy fp32 latent)", &lattf, &lat32);

        let dec32 = read_f32(&ref_dir.join(format!("{name}_fp32_decoded.bin")));
        let dectf = read_f32(&ref_dir.join(format!("{name}_tf32_decoded.bin")));
        let zref = Tensor::from_f32(&dev, &lat32, &[32, 2, t])?;
        let d = vae.decode(&zref, t)?;
        worst = worst.max(report("decode(ref fp32 latent) vs comfy fp32", &d, &dec32));
        let zref_tf = Tensor::from_f32(&dev, &lattf, &[32, 2, t])?;
        let dtf = vae.decode(&zref_tf, t)?;
        report("decode(ref tf32 latent) vs comfy default", &dtf, &dectf);
        report("  (comfy tf32 vs comfy fp32 decoded)", &dectf, &dec32);
        let d2 = vae.decode(&z, t)?;
        report("end-to-end decode(encode(wav)) vs comfy fp32", &d2, &dec32);
        let (_, _, snr) = cmp(&d2[..2 * n], &wav);
        let (_, _, snr_ref) = cmp(&dec32, &wav);
        println!("  roundtrip SNR vs input: rust {snr:.2} dB, comfy fp32 {snr_ref:.2} dB");
        // long-input path: encoder conv stack one stereo channel at a time
        let mut v2 = vae;
        let saved = v2.enc_split_samples;
        v2.enc_split_samples = 0;
        let zs = v2.encode(&wav, n)?.to_f32_vec(&dev)?;
        v2.enc_split_samples = saved;
        vae = v2;
        report("encode (per-channel split path) vs batched", &zs, &zh);
    }
    let rin = ref_dir.join("resample_in_44100.bin");
    if rin.exists() {
        let x = read_f32(&rin);
        let want = read_f32(&ref_dir.join("resample_out_32000.bin"));
        let got = loki_reshoot::vae_audio::resample(&x, 44100, 32000);
        println!("== resample 44.1k -> 32k (host port of comfy.audio.resample): len {} vs {}", got.len(), want.len());
        if got.len() == want.len() {
            report("resample vs comfy", &got, &want);
        }
    }
    println!("worst rel-L2 vs strict fp32 reference: {worst:.3e}");

    // ---- timings (synthetic input, stereo)
    println!("== timings (stereo, after warm-up)");
    for secs in [1usize, 5, 15] {
        let n = secs * 32000;
        let t = n / 800;
        let wav: Vec<f32> = (0..2 * n).map(|i| (0.3 * ((i % n) as f32 * 0.031).sin() + 0.1 * ((i * 7919 % 1000) as f32 / 500.0 - 1.0)) as f32).collect();
        let z = vae.encode(&wav, n)?; // warm-up
        let _ = vae.decode(&z, t)?;
        dev.sync()?;
        let reps = 3;
        let t1 = Instant::now();
        for _ in 0..reps {
            let _z = vae.encode(&wav, n)?;
        }
        dev.sync()?;
        let te = t1.elapsed().as_secs_f64() / reps as f64;
        let t2 = Instant::now();
        for _ in 0..reps {
            let _w = vae.decode_dev(&z, t)?;
        }
        dev.sync()?;
        let td = t2.elapsed().as_secs_f64() / reps as f64;
        let t3 = Instant::now();
        let _w = vae.decode(&z, t)?;
        let tdh = t3.elapsed().as_secs_f64();
        let used = free_loaded.saturating_sub(dev.free_mem()?);
        println!(
            "  {secs:>2} s: encode {:7.1} ms   decode {:7.1} ms (device)  {:7.1} ms (incl. D2H)   pool high-water above weights {} MB",
            te * 1e3,
            td * 1e3,
            tdh * 1e3,
            used >> 20
        );
    }
    vae.prof.report();
    Ok(())
}
