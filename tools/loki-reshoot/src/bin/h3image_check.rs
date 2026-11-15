//! MiniMax-H3-Image end to end with the plain command-line host (compare with ref/ref_e2e_image.py).
//!
//!   h3image_check --prompt TEXT [--width W --height H] [--seed N] [--steps N] [--image FILE ...] --out FILE.png [--compare REF.png]
use anyhow::{Context, Result};
use loki_reshoot::h3image::{self, Paths, Plain, Request};
use loki_reshoot::image::Rgb8;
use std::path::PathBuf;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let prompt = get("--prompt").context("--prompt")?;
    let images: Vec<Rgb8> = args.windows(2).filter(|w| w[0] == "--image").map(|w| Rgb8::load(std::path::Path::new(&w[1]))).collect::<Result<_>>()?;
    let (w, h) = match (get("--width"), get("--height")) {
        (Some(w), Some(h)) => (w.parse()?, h.parse()?),
        _ => images.first().map(|i| h3image::ref_vae_size(i.w, i.h)).unwrap_or((h3image::DEFAULT_SIDE, h3image::DEFAULT_SIDE)),
    };
    let req = Request {
        prompt,
        images,
        width: w,
        height: h,
        steps: get("--steps").map(|s| s.parse()).transpose()?.unwrap_or(h3image::DEFAULT_STEPS),
        seed: get("--seed").map(|s| s.parse()).transpose()?.unwrap_or(1),
        init: None,
    };
    let paths = Paths::resolve(None, None, None)?;
    let dev = h3image::device(false)?;
    let out = h3image::generate(&dev, &Plain, &paths, &req)?;
    println!("sigmas {:?}", out.sigmas);
    let p = PathBuf::from(get("--out").context("--out")?);
    out.image.save_png(&p)?;
    if let Some(r) = get("--compare") {
        let r = Rgb8::load(std::path::Path::new(&r))?;
        anyhow::ensure!((r.w, r.h) == (out.image.w, out.image.h), "reference is {}x{}", r.w, r.h);
        let se: f64 = r.data.iter().zip(&out.image.data).map(|(a, b)| (*a as f64 - *b as f64).powi(2)).sum();
        let mse = se / r.data.len() as f64;
        println!("PSNR vs reference: {:.2} dB", 10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10());
    }
    println!("wrote {}", p.display());
    Ok(())
}
