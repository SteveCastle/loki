//! MiniMax-H3-Image DiT validation against ComfyUI dumps (ref/ref_dit_image.py).
//!
//!   dit_image_check [ref_dir]                   compare with ref_out/dit_image (default)
//!   dit_image_check --bench <W> <H> [refs] [n]  synthetic run at W x H px with `refs` reference images of that size
use anyhow::{Context, Result};
use loki_reshoot::cuda::Device;
use loki_reshoot::dit_h3::{Dit, DitInputs, RefBlock, RefKind, HIDDEN};
use loki_reshoot::tensor::{DType, Tensor};
use std::path::{Path, PathBuf};

fn model() -> PathBuf {
    PathBuf::from(std::env::var("LOKI_H3I_DIT").unwrap_or_else(|_| r"C:\Users\steph\bin\models\minimax_h3_image_pruned_int8_convrot.safetensors".into()))
}

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn stats(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let (mut d2, mut b2, mut a2, mut ab, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        d2 += (x - y) * (x - y);
        b2 += y * y;
        a2 += x * x;
        ab += x * y;
        mx = mx.max((x - y).abs());
    }
    println!(
        "  {name:<20} max_abs {mx:9.4}  rel_l2 {:.4e}  cos {:.6}  (ref rms {:.4}, ours rms {:.4})",
        (d2 / b2.max(1e-30)).sqrt(),
        ab / (a2.sqrt() * b2.sqrt()).max(1e-30),
        (b2 / b.len() as f64).sqrt(),
        (a2 / a.len() as f64).sqrt()
    );
}

fn check(dir: &Path) -> Result<()> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let dev = Device::new(0)?;
    let t0 = std::time::Instant::now();
    let dit = Dit::load(dev.clone(), &model())?;
    eprintln!("load {:.1}s, image model {}, PDD grid {:?}", t0.elapsed().as_secs_f64(), dit.is_image(), dit.pdd_grid());
    let u = |v: &serde_json::Value| v.as_u64().unwrap() as usize;
    let (l, lh, lw) = (u(&meta["text_len"]), u(&meta["latent_h"]), u(&meta["latent_w"]));
    let text = Tensor::from_f32(&dev, &read_f32(&dir.join("text.bin"))?, &[l, 5120])?;
    let tags = std::fs::read(dir.join("tags.bin"))?;
    let mut refs = Vec::new();
    for (i, r) in meta["refs"].as_array().unwrap().iter().enumerate() {
        let (h, w) = (u(&r["latent_h"]), u(&r["latent_w"]));
        let z = Tensor::from_f32(&dev, &read_f32(&dir.join(format!("ref{i}.bin")))?, &[24, 1, h, w])?;
        refs.push(RefBlock { kind: RefKind::Image, latent_t: 1, latent_h: h, latent_w: w, ref_audio_t: 0, video: Some(z), audio: None });
    }
    let inp = DitInputs { text: &text, text_tags: &tags, guides: &[], refs: &refs, latent_t: 1, latent_h: lh, latent_w: lw, audio_t: 0, seed: 0, cond_noise_aug: Some(1.0) };
    let mut run = dit.prepare(&inp)?;
    dev.sync()?;
    eprintln!("seq {} (ref {})", run.seq, meta["seq_len"]);
    run.sigmas = meta["schedule"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
    let x = Tensor::from_f32(&dev, &read_f32(&dir.join("x_in.bin"))?, &[24, 1, lh, lw])?;
    let out = Tensor::zeros(&dev, DType::F32, &[24, 1, lh, lw])?;
    let snap_blocks: Vec<usize> = meta["snap_blocks"].as_array().unwrap().iter().map(u).collect();
    for s in meta["evals"].as_array().unwrap() {
        let sigma = s.as_f64().unwrap();
        let sname = format!("{sigma:.6}");
        run.snap_layers = std::iter::once(usize::MAX).chain(snap_blocks.iter().copied()).collect();
        run.snapshots.clear();
        dit.forward_image(&mut run, &x, sigma as f32, &out)?;
        dev.sync()?;
        println!("sigma {sigma}:");
        for (li, v) in &run.snapshots {
            let tag = if *li == usize::MAX { "in".to_string() } else { format!("b{li}") };
            let p = dir.join(format!("h_{sname}_{tag}.bin"));
            if p.exists() {
                let r = read_f32(&p)?;
                stats(&format!("h {tag}"), v, &r);
                for (nm, a, b) in [("text", 0, l), ("refs", l, run.v0), ("image", run.v0, run.seq)] {
                    stats(&format!("   {tag} {nm}"), &v[a * HIDDEN..b * HIDDEN], &r[a * HIDDEN..b * HIDDEN]);
                }
            }
        }
        stats("out", &out.to_f32_vec(&dev)?, &read_f32(&dir.join(format!("out_{sname}.bin")))?);
    }
    Ok(())
}

fn bench(w: usize, h: usize, nrefs: usize, n: usize) -> Result<()> {
    use rand::{Rng, SeedableRng};
    let dev = Device::new(0)?;
    eprintln!("  device: {} MB free of {} MB before loading", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let dit = Dit::load(dev.clone(), &model())?;
    let (lh, lw) = (h / 16, w / 16);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut randn = |k: usize| -> Vec<f32> { (0..k).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect() };
    let l = 300;
    let text = Tensor::from_f32(&dev, &randn(l * 5120), &[l, 5120])?;
    let tags: Vec<u8> = (0..l).map(|i| if (20..120).contains(&i) { 0 } else { 1 }).collect();
    let mut refs = Vec::new();
    for _ in 0..nrefs {
        refs.push(RefBlock { kind: RefKind::Image, latent_t: 1, latent_h: lh, latent_w: lw, ref_audio_t: 0, video: Some(Tensor::from_f32(&dev, &randn(24 * lh * lw), &[24, 1, lh, lw])?), audio: None });
    }
    let inp = DitInputs { text: &text, text_tags: &tags, guides: &[], refs: &refs, latent_t: 1, latent_h: lh, latent_w: lw, audio_t: 0, seed: 0, cond_noise_aug: None };
    let t1 = std::time::Instant::now();
    let mut run = dit.prepare(&inp)?;
    run.sigmas = vec![1.0, 0.9677, 0.8877, 0.75, 0.5336, 0.2326, 0.0];
    dev.sync()?;
    println!("{w}x{h} + {nrefs} refs: seq {} tokens, chunk {} abatch {}, prepare {:.2}s, free {} MB", run.seq, run.chunk, run.abatch, t1.elapsed().as_secs_f64(), dev.free_mem()? >> 20);
    let x = Tensor::from_f32(&dev, &randn(24 * lh * lw), &[24, 1, lh, lw])?;
    let out = Tensor::zeros(&dev, DType::F32, &[24, 1, lh, lw])?;
    for i in 0..n {
        let t = std::time::Instant::now();
        let s = run.sigmas[i % 6] as f32;
        dit.forward_image(&mut run, &x, s, &out)?;
        dev.sync()?;
        println!("  forward {i}: {:.3}s", t.elapsed().as_secs_f64());
    }
    dit.prof.report();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--bench" {
        let p = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
        return bench(p(2, 2048), p(3, 2048), p(4, 0), p(5, 3));
    }
    check(&PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "ref_out/dit_image".into())))
}
