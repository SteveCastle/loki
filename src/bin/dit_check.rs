//! Compare the Rust VAE encode/decode and one DiT step against the ComfyUI reference (ref/ref_dit.py).
//! Usage: dit_check <te_ref_dir> <dit_ref_dir>
use anyhow::Result;
use loki_retouch::cuda::Device;
use loki_retouch::dit::{self, Dit};
use loki_retouch::image::Rgb8;
use loki_retouch::ops;
use loki_retouch::tensor::{bf16_bits, DType, Tensor};
use loki_retouch::vae::Vae;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}
fn chw_to_hwc(x: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for p in 0..hw {
        for ch in 0..c {
            out[p * c + ch] = x[ch * hw + p];
        }
    }
    out
}
fn stats(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut num = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut maxd = 0f64;
    for (a, b) in got.iter().zip(want) {
        num += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        maxd = maxd.max((*a as f64 - *b as f64).abs());
    }
    let rel_rms = (got.iter().zip(want).map(|(a, b)| ((a - b) * (a - b)) as f64).sum::<f64>() / nb).sqrt();
    println!("{name}: cosine {:.6}  rel-rms {:.4}  max-abs-diff {:.4}  (ref rms {:.4})", num / (na.sqrt() * nb.sqrt()), rel_rms, maxd, (nb / want.len() as f64).sqrt());
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let te_dir = Path::new(&args[1]);
    let dit_dir = Path::new(&args[2]);
    let home = std::env::var("USERPROFILE").unwrap();
    let dev = Device::new(0)?;
    ops::gemm_init(&dev)?;
    ops::attn_init(&dev)?;
    Vae::init_kernels(&dev)?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dit_dir.join("meta.json"))?)?;
    let rl_shape: Vec<usize> = meta["ref_latent_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let x_shape: Vec<usize> = meta["x_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let sigma = meta["sigma"].as_f64().unwrap() as f32;
    let slot = meta["slot"].as_u64().unwrap() as usize;
    let (rh, rw) = (rl_shape[2], rl_shape[3]);
    let (h, w) = (x_shape[2], x_shape[3]);

    // ---- VAE encode
    let vae = Vae::load(dev.clone(), Path::new(&format!("{home}/dev/loki-retouch/models/qwen_image_2.1_vae_bf16.safetensors")))?;
    let img = Rgb8::load(&te_dir.join("resized.png"))?;
    let t0 = std::time::Instant::now();
    let lat = vae.encode(&img.to_f32(), img.h, img.w)?; // [rh*rw, 64] f32 token-major
    dev.sync()?;
    println!("vae encode {}x{} took {:.2}s", img.w, img.h, t0.elapsed().as_secs_f64());
    let lat_hwc = lat.to_f32_vec(&dev)?;
    let ref_chw = read_f32(&dit_dir.join("ref_latent.bin"));
    let ref_hwc = chw_to_hwc(&ref_chw, 64, rh * rw);
    stats("vae encode", &lat_hwc, &ref_hwc);

    // ---- DiT one step using the reference context and reference latent
    let te_meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(te_dir.join("meta.json"))?)?;
    let lc = te_meta["shape"][0].as_u64().unwrap() as usize;
    let ctx = read_f32(&te_dir.join("context.bin"));
    let ctx_bf: Vec<u16> = ctx.iter().map(|v| bf16_bits(*v)).collect();
    let context = Tensor::from_bf16(&dev, &ctx_bf, &[lc, 4096])?;
    let ref_lat_t = Tensor::from_f32(&dev, &ref_hwc, &[rh * rw, 64])?;
    let ref_norm = Tensor::new(&dev, DType::BF16, &[rh * rw, 64])?;
    dit::latent_norm_in(&dev, &ref_lat_t, &vae.latents_mean, &vae.latents_std, &ref_norm)?;
    let model = Dit::load(dev.clone(), Path::new(&format!("{home}/dev/loki-retouch/models/qwen_image_2.1_int8_convrot.safetensors")))?;
    let mut run = model.prepare(&context, &[dit::RefLatent { slot, latent: ref_norm, rh, rw }], h, w)?;
    let x_chw = read_f32(&dit_dir.join("x.bin"));
    let x_hwc = chw_to_hwc(&x_chw, 64, h * w);
    let x_bf: Vec<u16> = x_hwc.iter().map(|v| bf16_bits(*v)).collect();
    let xt = Tensor::from_bf16(&dev, &x_bf, &[h * w, 64])?;
    let v = Tensor::new(&dev, DType::BF16, &[h * w, 64])?;
    let t0 = std::time::Instant::now();
    model.forward(&mut run, &xt, sigma, &v)?;
    dev.sync()?;
    println!("dit step (incl. prefix) took {:.2}s", t0.elapsed().as_secs_f64());
    let t0 = std::time::Instant::now();
    model.forward(&mut run, &xt, sigma, &v)?;
    dev.sync()?;
    println!("dit cached step took {:.2}s", t0.elapsed().as_secs_f64());
    let vv = v.to_f32_vec(&dev)?;
    let denoised: Vec<f32> = x_hwc.iter().zip(&vv).map(|(x, v)| x - v * sigma).collect();
    let ref_den_chw = read_f32(&dit_dir.join("denoised.bin"));
    let ref_den = chw_to_hwc(&ref_den_chw, 64, h * w);
    stats("dit denoised", &denoised, &ref_den);
    drop(run);
    drop(model);

    // ---- VAE decode of the reference denoised latent
    let den_t = Tensor::from_f32(&dev, &ref_den, &[h * w, 64])?;
    let den_bf = Tensor::new(&dev, DType::BF16, &[h * w, 64])?;
    ops::to_bf16(&dev, &den_t, &den_bf)?;
    let t0 = std::time::Instant::now();
    let rgb = vae.decode(&den_bf, h, w)?;
    dev.sync()?;
    println!("vae decode {}x{} took {:.2}s", w * 16, h * 16, t0.elapsed().as_secs_f64());
    let ref_dec_all = read_f32(&dit_dir.join("decoded.bin"));
    let chans = ref_dec_all.len() / (w * 16 * h * 16);
    let mut ref_dec = Vec::with_capacity(w * 16 * h * 16 * 3);
    for p in 0..w * 16 * h * 16 {
        for c in 0..3 {
            ref_dec.push(ref_dec_all[p * chans + c]);
        }
    }
    let got: Vec<f32> = rgb.iter().map(|v| *v as f32 / 255.0).collect();
    stats("vae decode", &got, &ref_dec);
    Rgb8 { w: w * 16, h: h * 16, data: rgb }.save_png(&dit_dir.join("decoded_rust.png"))?;
    println!("wrote {}", dit_dir.join("decoded_rust.png").display());
    Ok(())
}
