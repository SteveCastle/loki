//! Validate the MiniMax-H3-Image VAE against ComfyUI reference dumps (ref/ref_ivae.py).
//! Usage: ivae_check [ref_dir] [case,case,...] [--png out_dir]
use anyhow::{Context, Result};
use loki_reshoot::cuda::Device;
use loki_reshoot::tensor::Tensor;
use loki_reshoot::vae_image::ImageVae;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// (max abs error, relative L2 error)
fn stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut max = 0f32;
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        max = max.max((x - y).abs());
        num += ((x - y) as f64).powi(2);
        den += (*y as f64).powi(2);
    }
    (max, (num / den.max(1e-30)).sqrt() as f32)
}

/// PSNR (dB) and max abs difference of u8 pixels against reference pixels in [0, 1]
fn psnr_u8(a: &[u8], b: &[f32]) -> (f64, f32) {
    let mut se = 0f64;
    let mut mx = 0f32;
    for (x, y) in a.iter().zip(b) {
        let d = *x as f32 - y * 255.0;
        se += (d as f64) * (d as f64);
        mx = mx.max(d.abs());
    }
    let mse = se / a.len() as f64;
    (10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10(), mx)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ref_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "ref_out/ivae".into()));
    let cases: Vec<String> = args.get(2).filter(|a| !a.starts_with("--")).map(|s| s.split(',').map(|x| x.to_string()).collect()).unwrap_or_else(|| vec!["small".into(), "tiled".into()]);
    let png_dir = args.iter().position(|a| a == "--png").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let model = PathBuf::from(std::env::var("LOKI_IVAE").unwrap_or_else(|_| r"C:\Users\steph\bin\models\minimax_h3_image_vae_fp16.safetensors".into()));
    let dev = Device::new(0)?;
    let t0 = Instant::now();
    let vae = ImageVae::load(dev.clone(), &model)?;
    println!("loaded in {:.1}s", t0.elapsed().as_secs_f64());
    for case in &cases {
        let d = ref_dir.join(case);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("meta.json"))?)?;
        let (w, h) = (meta["w"].as_u64().unwrap() as usize, meta["h"].as_u64().unwrap() as usize);
        let px = std::fs::read(d.join("pixels.u8"))?;
        let (hl, wl) = (h / 16, w / 16);
        // encoder moments, then the normalized posterior sample
        let t = Instant::now();
        let m = vae.moments(&px, h, w)?;
        let rm = read_f32(&d.join("moments.bin"))?;
        let np = hl * wl;
        let (mmax, mrel) = stats(&m[..24 * np], &rm[..24 * np]);
        let (lmax, lrel) = stats(&m[24 * np..], &rm[24 * np..]);
        let z = vae.encode(&px, h, w)?;
        dev.sync()?;
        let zv = z.to_f32_vec(&dev)?;
        let rz = read_f32(&d.join("latent.bin"))?;
        let (zmax, zrel) = stats(&zv, &rz);
        println!("{case} {w}x{h}: encode {:.2}s  mean max {mmax:.4} rel {mrel:.5} | logvar max {lmax:.4} rel {lrel:.5} | latent max {zmax:.4} rel {zrel:.5}", t.elapsed().as_secs_f64());
        // decode the reference latent (isolates the decoder)
        let rzt = Tensor::from_f32(&dev, &rz, &[24, 1, hl, wl])?;
        let t = Instant::now();
        let img = vae.decode(&rzt, hl, wl)?;
        let rd = read_f32(&d.join("decoded.bin"))?;
        let (psnr, mx) = psnr_u8(&img, &rd);
        println!("{case}: decode {:.2}s  PSNR vs reference {psnr:.2} dB, max |diff| {mx:.1}/255", t.elapsed().as_secs_f64());
        if let Some(pd) = &png_dir {
            std::fs::create_dir_all(pd)?;
            image::RgbImage::from_raw(w as u32, h as u32, img).context("png")?.save(pd.join(format!("{case}.png")))?;
        }
    }
    Ok(())
}
