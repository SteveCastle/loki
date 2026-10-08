//! DiT validation against ComfyUI dumps (ref/ref_dit.py) and synthetic benchmarks.
//!
//!   dit_check [ref_dir]                         compare with ref_out/dit (default)
//!   dit_check --bench <W> <H> <frames> [n]      synthetic run at W x H px, `frames` frames, n forwards
use anyhow::{Context, Result};
use loki_reshoot::cuda::Device;
use loki_reshoot::dit_h3::{Dit, DitInputs, RefBlock, RefKind, HIDDEN, LAYERS};
use loki_reshoot::tensor::{DType, Tensor};
use std::path::{Path, PathBuf};

const MODEL: &str = r"C:\Users\steph\bin\models\minimax_h3_ref2va_pruned_int8_convrot.safetensors";

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn stats(name: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{name}: length mismatch");
    let (mut d2, mut b2, mut a2, mut ab, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    let mut nan = 0;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() {
            nan += 1;
            continue;
        }
        let (x, y) = (*x as f64, *y as f64);
        d2 += (x - y) * (x - y);
        b2 += y * y;
        a2 += x * x;
        ab += x * y;
        mx = mx.max((x - y).abs());
    }
    println!(
        "  {name:<24} max_abs {mx:9.4}  rel_l2 {:.4e}  cos {:.6}  (ref rms {:.4}, ours rms {:.4}){}",
        (d2 / b2.max(1e-30)).sqrt(),
        ab / (a2.sqrt() * b2.sqrt()).max(1e-30),
        (b2 / b.len() as f64).sqrt(),
        (a2 / a.len() as f64).sqrt(),
        if nan > 0 { format!("  NONFINITE {nan}") } else { String::new() }
    );
}

fn check(dir: &Path) -> Result<()> {
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("meta.json"))?)?;
    let dev = Device::new(0)?;
    let t0 = std::time::Instant::now();
    let dit = Dit::load(dev.clone(), Path::new(MODEL))?;
    eprintln!("load {:.1}s", t0.elapsed().as_secs_f64());
    let l = meta["text_len"].as_u64().unwrap() as usize;
    let (lt, lh, lw, ta) = (meta["latent_t"].as_u64().unwrap() as usize, meta["latent_h"].as_u64().unwrap() as usize, meta["latent_w"].as_u64().unwrap() as usize, meta["audio_t"].as_u64().unwrap() as usize);
    let text = Tensor::from_f32(&dev, &read_f32(&dir.join("text.bin"))?, &[l, 5120])?;
    let tags = std::fs::read(dir.join("tags.bin"))?;
    let mut refs = Vec::new();
    for (i, r) in meta["refs"].as_array().unwrap().iter().enumerate() {
        let kind = match r["kind"].as_str().unwrap() {
            "image" => RefKind::Image,
            "audio" => RefKind::Audio,
            "video" => RefKind::Video,
            "video_audio" => RefKind::VideoAudio,
            k => anyhow::bail!("unknown ref kind {k}"),
        };
        let u = |k: &str| r[k].as_u64().unwrap_or(0) as usize;
        let video = if r.get("video_shape").is_some() {
            let shape: Vec<usize> = r["video_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            Some(Tensor::from_f32(&dev, &read_f32(&dir.join(format!("ref{i}_video.bin")))?, &shape)?)
        } else {
            None
        };
        let audio = if r.get("audio_shape").is_some() {
            let shape: Vec<usize> = r["audio_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            Some(Tensor::from_f32(&dev, &read_f32(&dir.join(format!("ref{i}_audio.bin")))?, &shape)?)
        } else {
            None
        };
        refs.push(RefBlock { kind, latent_t: u("latent_t"), latent_h: u("latent_h"), latent_w: u("latent_w"), ref_audio_t: u("ref_audio_t"), video, audio });
    }
    let inp = DitInputs { text: &text, text_tags: &tags, refs: &refs, latent_t: lt, latent_h: lh, latent_w: lw, audio_t: ta, seed: 0, cond_noise_aug: Some(1.0) };
    let t1 = std::time::Instant::now();
    let mut run = dit.prepare(&inp)?;
    dev.sync()?;
    eprintln!("prepare {:.2}s  seq {} (ref {}), chunk {}", t1.elapsed().as_secs_f64(), run.seq, meta["seq_len"], run.chunk);
    let ts = read_f32(&dir.join("text_states.bin"))?;
    let ours = run.prefix_states(&dev)?;
    stats("text_states", &ours[..l * HIDDEN], &ts);

    let xv = Tensor::from_f32(&dev, &read_f32(&dir.join("video_in.bin"))?, &[24, lt, lh, lw])?;
    let xa = Tensor::from_f32(&dev, &read_f32(&dir.join("audio_in.bin"))?, &[32, 2, ta])?;
    let ov = Tensor::zeros(&dev, DType::F32, &[24, lt, lh, lw])?;
    let oa = Tensor::zeros(&dev, DType::F32, &[32, 2, ta])?;
    let snap_blocks: Vec<usize> = meta["snap_blocks"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    for s in meta["sigmas"].as_array().unwrap() {
        let sigma = s.as_f64().unwrap() as f32;
        let sname = format!("{}", s.as_f64().unwrap());
        run.snap_layers = std::iter::once(usize::MAX).chain(snap_blocks.iter().copied()).collect();
        run.snapshots.clear();
        let t2 = std::time::Instant::now();
        dit.forward(&mut run, &xv, &xa, sigma, &ov, &oa)?;
        dev.sync()?;
        println!("sigma {sigma}: forward {:.3}s (with debug snapshots)", t2.elapsed().as_secs_f64());
        for (li, v) in &run.snapshots {
            let tag = if *li == usize::MAX { "in".to_string() } else { format!("b{li}") };
            let p = dir.join(format!("h_{sname}_{tag}.bin"));
            if p.exists() {
                let r = read_f32(&p)?;
                stats(&format!("h {tag}"), v, &r);
                for (nm, a, b) in [("text", 0, l), ("refs", l, run.a0), ("audio", run.a0, run.v0), ("video", run.v0, run.seq)] {
                    if b > a {
                        stats(&format!("   {tag} {nm}"), &v[a * HIDDEN..b * HIDDEN], &r[a * HIDDEN..b * HIDDEN]);
                    }
                }
            }
        }
        stats("out_video", &ov.to_f32_vec(&dev)?, &read_f32(&dir.join(format!("out_video_{sname}.bin")))?);
        stats("out_audio", &oa.to_f32_vec(&dev)?, &read_f32(&dir.join(format!("out_audio_{sname}.bin")))?);
        // timing without snapshots
        run.snap_layers.clear();
        dev.sync()?;
        let t3 = std::time::Instant::now();
        dit.forward(&mut run, &xv, &xa, sigma, &ov, &oa)?;
        dev.sync()?;
        println!("  forward {:.3}s", t3.elapsed().as_secs_f64());
    }
    dit.prof.report();
    Ok(())
}

fn bench(w: usize, h: usize, frames: usize, n: usize) -> Result<()> {
    use rand::{Rng, SeedableRng};
    let dev = Device::new(0)?;
    eprintln!("  device: {} MB free of {} MB before loading", dev.free_mem()? >> 20, dev.total_mem >> 20);
    let dit = Dit::load(dev.clone(), Path::new(MODEL))?;
    let align = |mut f: usize| {
        while f % 17 != 5 {
            f += 1;
        }
        f
    };
    let fc = align(frames.max(5));
    let lt = if fc <= 5 { 2 } else { ((fc - 5) / 17) * 5 + 2 };
    let ta = ((fc as f64 / 24.0) * 40.0).round() as usize;
    let (lh, lw) = (h / 16, w / 16);
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut randn = |k: usize| -> Vec<f32> { (0..k).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect() };
    let l = 300;
    let text = Tensor::from_f32(&dev, &randn(l * 5120), &[l, 5120])?;
    let tags: Vec<u8> = (0..l).map(|i| if (20..120).contains(&i) { 0 } else { 1 }).collect();
    let refs = vec![RefBlock { kind: RefKind::Image, latent_t: 1, latent_h: lh, latent_w: lw, ref_audio_t: 0, video: Some(Tensor::from_f32(&dev, &randn(24 * lh * lw), &[24, 1, lh, lw])?), audio: None }];
    let inp = DitInputs { text: &text, text_tags: &tags, refs: &refs, latent_t: lt, latent_h: lh, latent_w: lw, audio_t: ta, seed: 0, cond_noise_aug: None };
    let t1 = std::time::Instant::now();
    let mut run = dit.prepare(&inp)?;
    dev.sync()?;
    println!("{w}x{h} {fc} frames: latent {lt}x{lh}x{lw}, audio {ta}, seq {} tokens, chunk {} abatch {}, prepare {:.2}s, free {} MB", run.seq, run.chunk, run.abatch, t1.elapsed().as_secs_f64(), dev.free_mem()? >> 20);
    let xv = Tensor::from_f32(&dev, &randn(24 * lt * lh * lw), &[24, lt, lh, lw])?;
    let xa = Tensor::from_f32(&dev, &randn(64 * ta), &[32, 2, ta])?;
    let ov = Tensor::zeros(&dev, DType::F32, &[24, lt, lh, lw])?;
    let oa = Tensor::zeros(&dev, DType::F32, &[32, 2, ta])?;
    for i in 0..n {
        let t = std::time::Instant::now();
        dit.forward(&mut run, &xv, &xa, 0.9 - 0.2 * i as f32, &ov, &oa)?;
        dev.sync()?;
        println!("  forward {i}: {:.3}s", t.elapsed().as_secs_f64());
    }
    let v = ov.to_f32_vec(&dev)?;
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    println!("  out rms {:.4} nonfinite {nonfinite}; layers {LAYERS}", (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt());
    dit.prof.report();
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "--mma" {
        return mma_bench();
    }
    if args.len() > 1 && args[1] == "--alloctest" {
        let dev = Device::new(0)?;
        let f0 = dev.free_mem()?;
        let mut v = Vec::new();
        for _ in 0..400 {
            v.push(Tensor::new(&dev, DType::F32, &[64])?);
        }
        dev.sync()?;
        let f1 = dev.free_mem()?;
        let mut w = Vec::new();
        for _ in 0..20 {
            w.push(Tensor::new(&dev, DType::I8, &[115_605_504])?);
        }
        dev.sync()?;
        let f2 = dev.free_mem()?;
        println!("400 tiny allocs: {} MB; 20 x 115.6MB allocs: {} MB (payload {} MB)", (f0 - f1) >> 20, (f1 - f2) >> 20, (20 * 115_605_504usize) >> 20);
        drop(w);
        drop(v);
        dev.sync()?;
        let g0 = dev.free_mem()?;
        let big = dev.alloc(19_370_000_000)?;
        dev.sync()?;
        let g1 = dev.free_mem()?;
        println!("one 19.37 GB alloc: free {} -> {} MB (delta {} MB, payload {} MB)", g0 >> 20, g1 >> 20, (g0 - g1) >> 20, 19_370_000_000usize >> 20);
        // touch it with a memset to force residency
        dev.memset_at(big.ptr(), big.len)?;
        dev.sync()?;
        println!("after memset: free {} MB", dev.free_mem()? >> 20);
        return Ok(());
    }
    if args.len() > 1 && args[1] == "--gemm" {
        let dev = Device::new(0)?;
        let p = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        let (m, iters) = (p(2, 8192), p(3, 50));
        for (name, n, k, mode) in [("qkv/kv", 14336usize, 5376usize, 0), ("q", 7168, 5376, 0), ("out", 5376, 7168, 4), ("fc1", 28672, 5376, 3), ("fc2", 5376, 14336, 4)] {
            let dt = loki_reshoot::dit_h3::gemm_bench(&dev, m, n, k, mode, iters)?;
            println!("gemm {name:<6} m {m} n {n} k {k}: {:.3} ms  {:.1} TOPS", dt * 1e3, 2.0 * m as f64 * n as f64 * k as f64 / dt / 1e12);
        }
        return Ok(());
    }
    if args.len() > 1 && args[1] == "--attn" {
        let dev = Device::new(0)?;
        let nk: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32768);
        let nq: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(8192);
        let dt = loki_reshoot::dit_h3::attention_bench(&dev, nq, nk, std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(30))?;
        let flops = 4.0 * nq as f64 * nk as f64 * 7168.0;
        println!("attention nq {nq} nk {nk}: {:.2} ms  {:.1} TOPS", dt * 1e3, flops / dt / 1e12);
        return Ok(());
    }
    if args.len() > 1 && args[1] == "--bench" {
        let p = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
        return bench(p(2, 1344), p(3, 768), p(4, 124), p(5, 2));
    }
    let dir = args.get(1).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Users\steph\dev\loki\tools\loki-reshoot\ref_out\dit"));
    check(&dir)
}

#[allow(dead_code)]
pub fn mma_bench() -> Result<()> {
    use loki_reshoot::cuda::Arg;
    let dev = Device::new(0)?;
    let out = Tensor::zeros(&dev, DType::F32, &[4])?;
    let iters = 65536;
    for (mode, name, macs) in [(0, "s8 m16n8k32", 4096.0), (1, "e4m3 f32acc m16n8k32", 4096.0), (2, "f16 f16acc m16n8k16", 2048.0), (3, "bf16 f32acc m16n8k16", 2048.0), (4, "f16 f32acc m16n8k16", 2048.0), (5, "mixed s8:f16 1:2 (interleaved)", 0.0), (6, "mixed s8:f16 5:11 (blocks)", 0.0), (7, "mixed s8:e4m3 1:1", 0.0)] {
        let blocks = dev.sm_count as u32 * 4;
        dev.launch("k_h3_mma_bench", (blocks, 1, 1), (128, 1, 1), 0, &[Arg::I32(mode), Arg::I32(16), Arg::Ptr(out.ptr)])?;
        dev.sync()?;
        let t = std::time::Instant::now();
        dev.launch("k_h3_mma_bench", (blocks, 1, 1), (128, 1, 1), 0, &[Arg::I32(mode), Arg::I32(iters), Arg::Ptr(out.ptr)])?;
        dev.sync()?;
        let dt = t.elapsed().as_secs_f64();
        let per16 = match mode { 5 => 6.0 * 4096.0 + 10.0 * 2048.0, 6 => 5.0 * 4096.0 + 11.0 * 2048.0, 7 => 16.0 * 4096.0, _ => 16.0 * macs };
        let ops = blocks as f64 * 4.0 * iters as f64 * per16 * 2.0;
        // ideal time if each instruction type ran at its own peak (s8 660, f16 333)
        let ideal = match mode { 5 => (6.0 * 4096.0 * 2.0 / 660e12 + 10.0 * 2048.0 * 2.0 / 333e12) * blocks as f64 * 4.0 * iters as f64, 6 => (5.0 * 4096.0 * 2.0 / 660e12 + 11.0 * 2048.0 * 2.0 / 333e12) * blocks as f64 * 4.0 * iters as f64, 7 => (8.0 * 4096.0 * 2.0 / 660e12 + 8.0 * 4096.0 * 2.0 / 333e12) * blocks as f64 * 4.0 * iters as f64, _ => 0.0 };
        println!("{name:<32} {:.1} TOPS  ({:.0}% of ideal mix)", ops / dt / 1e12, if ideal > 0.0 { 100.0 * ideal / dt } else { 0.0 });
    }
    Ok(())
}
