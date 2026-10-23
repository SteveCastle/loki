//! Stage checks of the native engine against the reference dumps of ref/ref_dream.py (the official pipeline code).
//!
//!   dream_check sched DIR                 scheduler sigmas
//!   dream_check te    DIR                 Qwen3.5 hidden states (layers 0, 1, 3, 4, 16, 31 and last_hidden_state)
//!   dream_check dit   DIR [--bf16|--int8] one transformer forward (text refiner, blocks 0/21/41, cond out, uncond skip 22)
//!   dream_check vae   DIR                 VAE decode (and encode when the dump has one)
//!   dream_check full  DIR TAG [--bf16|--int8] [--lora FILE] whole sampler from the dumped noise -> latent + PNG PSNR
//!
//! Models are found like loki-dream finds them. Prints one metrics row per tensor: cosine, max |diff|, mean |diff|,
//! relative L2, and the reference's mean |x| for scale.
use anyhow::{bail, Context, Result};
use loki_dream::cuda::Device;
use loki_dream::dit::{Dit, LATENT};
use loki_dream::models;
use loki_dream::pipeline::{self, Inputs, Settings};
use loki_dream::tensor::{bf16_bits, DType, Tensor};
use loki_dream::text_encoder::TextEncoder;
use loki_dream::vae::Vae;
use std::path::{Path, PathBuf};

struct Dump {
    dir: PathBuf,
    index: serde_json::Value,
}
impl Dump {
    fn open(dir: &Path) -> Result<Dump> {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).context("index.json")?)?;
        Ok(Dump { dir: dir.to_path_buf(), index })
    }
    fn has(&self, name: &str) -> bool {
        self.index.get(name).is_some()
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.index[name]["shape"].as_array().with_context(|| format!("{name} not in dump"))?.iter().map(|v| v.as_u64().unwrap() as usize).collect())
    }
    fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let b = std::fs::read(self.dir.join(format!("{name}.bin"))).with_context(|| format!("reading {name}.bin"))?;
        Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn json(&self, file: &str) -> Result<serde_json::Value> {
        Ok(serde_json::from_str(&std::fs::read_to_string(self.dir.join(file))?)?)
    }
}

fn metrics(name: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        println!("{name:<28} LENGTH MISMATCH {} vs {}", got.len(), want.len());
        return;
    }
    let (mut dot, mut ng, mut nw, mut maxd, mut sumd, mut nd, mut mw) = (0f64, 0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        let (a, b) = (a as f64, b as f64);
        dot += a * b;
        ng += a * a;
        nw += b * b;
        let d = (a - b).abs();
        maxd = maxd.max(d);
        sumd += d;
        nd += d * d;
        mw += b.abs();
    }
    let n = got.len() as f64;
    println!("{name:<28} cos {:.6}  max|d| {:<10.4} mean|d| {:<10.5} relL2 {:<9.5} ref mean|x| {:.4}", dot / (ng.sqrt() * nw.sqrt()).max(1e-30), maxd, sumd / n, (nd / nw.max(1e-30)).sqrt(), mw / n);
}

fn bf16_tensor(dev: &Device, v: &[f32], shape: &[usize]) -> Result<Tensor> {
    let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x)).collect();
    Tensor::from_bf16(dev, &bits, shape)
}

/// [C, h, w] -> token-major [h*w, C]
fn chw_to_tokens(v: &[f32], c: usize, hw: usize) -> Vec<f32> {
    let mut o = vec![0f32; v.len()];
    for ch in 0..c {
        for p in 0..hw {
            o[p * c + ch] = v[ch * hw + p];
        }
    }
    o
}

fn dit_path(args: &[String]) -> Result<PathBuf> {
    if args.iter().any(|a| a == "--int8") {
        models::resolve(None, &models::DIT_INT8)
    } else {
        models::resolve(None, &models::DIT_BF16)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: dream_check sched|te|dit|vae|full DIR [...]");
    }
    if args[0] == "upload" {
        return upload(&args);
    }
    let d = Dump::open(Path::new(&args[1]))?;
    match args[0].as_str() {
        "sched" => {
            for n in [1, 4, 8, 20, 28, 50] {
                let want = d.f32(&format!("sigmas_{n}"))?;
                let got = loki_dream::sampler::sigmas(n, 5.0);
                let exact = got == want;
                println!("sigmas_{n:<3} {} steps  exact {}  got {:?}", got.len() - 1, exact, &got[..got.len().min(6)]);
                if !exact {
                    metrics(&format!("sigmas_{n}"), &got, &want);
                }
            }
        }
        "te" => {
            let dev = Device::new(0)?;
            loki_dream::ops::gemm_init(&dev)?;
            let te = TextEncoder::open(dev.clone(), &models::resolve(None, &models::TEXT_ENCODER)?)?;
            let meta = d.json("te.json")?;
            let prompts: Vec<String> = meta.as_array().unwrap().iter().map(|m| m["prompt"].as_str().unwrap().to_string()).collect();
            let taps: Vec<usize> = (0..31).collect();
            let t0 = std::time::Instant::now();
            let (enc, tapped) = te.encode_taps(&prompts, &taps, &|_| {})?;
            println!("text encoder: {} prompts in {:.2}s", prompts.len(), t0.elapsed().as_secs_f64());
            for (i, e) in enc.iter().enumerate() {
                let ids: Vec<u32> = d.f32(&format!("te{i}_ids")).map(|v| v.iter().map(|x| x.to_bits()).collect()).unwrap_or_default();
                let ids_ok = ids.is_empty() || ids == e.tokens;
                println!("-- prompt {i} ({} tokens, ids match: {ids_ok})", e.tokens.len());
                for (k, &l) in taps.iter().enumerate() {
                    let v = tapped[i][k].to_f32_vec(&dev)?;
                    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                    std::fs::create_dir_all(d.dir.join("native"))?;
                    std::fs::write(d.dir.join("native").join(format!("te{i}_h{l}.bin")), bytes)?;
                    metrics(&format!("te{i} layer {l}"), &v, &d.f32(&format!("te{i}_h{l}"))?);
                    if d.has(&format!("te{i}_h{l}_32")) {
                        let t = d.f32(&format!("te{i}_h{l}_32"))?;
                        metrics(&format!("te{i} layer {l} native/fp32"), &v, &t);
                        metrics(&format!("te{i} layer {l} ref/fp32"), &d.f32(&format!("te{i}_h{l}"))?, &t);
                    }
                }
                let mine = e.hidden.to_f32_vec(&dev)?;
                metrics(&format!("te{i} last_hidden_state"), &mine, &d.f32(&format!("te{i}_last"))?);
                if d.has(&format!("te{i}_last32")) {
                    let truth = d.f32(&format!("te{i}_last32"))?;
                    metrics(&format!("te{i} native vs fp32"), &mine, &truth);
                    metrics(&format!("te{i} reference vs fp32"), &d.f32(&format!("te{i}_last"))?, &truth);
                }
            }
        }
        "dit" => {
            let dev = Device::new(0)?;
            loki_dream::ops::gemm_init(&dev)?;
            loki_dream::ops::attn_init(&dev)?;
            let meta = d.json("dit.json")?;
            let (w, h) = (meta["w"].as_u64().unwrap() as usize / 16, meta["h"].as_u64().unwrap() as usize / 16);
            let t = meta["t"].as_f64().unwrap() as f32;
            let neg = meta["neg_index"].as_u64().unwrap() as usize;
            let mut dit = Dit::load(dev.clone(), &dit_path(&args)?, if args.iter().any(|a| a == "--sage") { Some(false) } else { Some(true) })?;
            dit.tap_blocks = vec![0, 21, 41];
            let ctx = |i: usize| -> Result<Tensor> {
                let s = d.shape(&format!("te{i}_last"))?;
                bf16_tensor(&dev, &d.f32(&format!("te{i}_last"))?, &s)
            };
            let lat = chw_to_tokens(&d.f32("dit_latent")?, LATENT, h * w);
            let latb = bf16_tensor(&dev, &lat, &[h * w, LATENT])?;
            let cond = dit.prepare(&ctx(0)?, h, w)?;
            metrics("text refiner out", &dit.run_text(&cond).to_f32_vec(&dev)?, &d.f32("dit_txt")?);
            let m = dit.modulation(t)?;
            let out = Tensor::new(&dev, DType::BF16, &[h * w, LATENT])?;
            let t0 = std::time::Instant::now();
            dit.forward(&cond, &latb, &m, &[], &out)?;
            dev.sync()?;
            println!("forward {:.3}s", t0.elapsed().as_secs_f64());
            let tapped: Vec<Tensor> = std::mem::take(&mut *dit.tapped.lock().unwrap());
            for (k, b) in [0, 21, 41].iter().enumerate() {
                metrics(&format!("block {b} out"), &tapped[k].to_f32_vec(&dev)?, &d.f32(&format!("dit_blk{b}"))?);
            }
            metrics("velocity (cond)", &out.to_f32_vec(&dev)?, &chw_to_tokens(&d.f32("dit_out_cond")?, LATENT, h * w));
            dit.tap_blocks.clear();
            let unc = dit.prepare(&ctx(neg)?, h, w)?;
            dit.forward(&unc, &latb, &m, &[22], &out)?;
            metrics("velocity (uncond, skip 22)", &out.to_f32_vec(&dev)?, &chw_to_tokens(&d.f32("dit_out_uncond_skip22")?, LATENT, h * w));
        }
        "vae" => {
            let dev = Device::new(0)?;
            loki_dream::ops::gemm_init(&dev)?;
            Vae::init_kernels(&dev)?;
            let vae = Vae::load(dev.clone(), &models::resolve(None, &models::VAE)?)?;
            let zs = d.shape("vae_z")?;
            let (h, w) = (zs[1], zs[2]);
            let z = Tensor::from_f32(&dev, &chw_to_tokens(&d.f32("vae_z")?, LATENT, h * w), &[h * w, LATENT])?;
            let t0 = std::time::Instant::now();
            let rgb = vae.decode(&z, h, w)?;
            println!("decode {}x{} in {:.3}s", w * 16, h * 16, t0.elapsed().as_secs_f64());
            let want = d.f32("vae_dec")?; // [3, H, W] in [-1, 1]
            let hw = h * w * 256;
            let want_hwc = chw_to_tokens(&want, 3, hw);
            let want_u8: Vec<f32> = want_hwc.iter().map(|v| ((v + 1.0) * 127.5).round().clamp(0.0, 255.0)).collect();
            let got: Vec<f32> = rgb.iter().map(|v| *v as f32).collect();
            metrics("vae decode (0..255)", &got, &want_u8);
            println!("vae decode PSNR {:.2} dB", psnr(&got, &want_u8));
            if d.has("vae_enc_in") {
                let s = d.shape("vae_enc_in")?; // [3, H, W] in [-1, 1]
                let (ih, iw) = (s[1], s[2]);
                let img = chw_to_tokens(&d.f32("vae_enc_in")?, 3, ih * iw).iter().map(|v| (v + 1.0) * 0.5).collect::<Vec<f32>>();
                let tok = vae.encode(&img, ih, iw)?;
                metrics("vae encode (normalized)", &tok.to_f32_vec(&dev)?, &chw_to_tokens(&d.f32("vae_enc")?, LATENT, ih * iw / 256));
            }
        }
        "full" => {
            let tag = args.get(2).context("full DIR TAG")?.clone();
            let dev = Device::new(0)?;
            loki_dream::ops::gemm_init(&dev)?;
            loki_dream::ops::attn_init(&dev)?;
            Vae::init_kernels(&dev)?;
            let meta = d.json(&format!("{tag}.json"))?;
            let (w, h) = (meta["w"].as_u64().unwrap() as usize / 16, meta["h"].as_u64().unwrap() as usize / 16);
            let mut dit = Dit::load(dev.clone(), &dit_path(&args)?, None)?;
            if let Some(i) = args.iter().position(|a| a == "--lora") {
                dit.lora = loki_dream::lora::load(&dev, &[loki_dream::lora::LoraSpec::parse(&args[i + 1])?], loki_dream::dit::LAYERS)?;
            }
            let ctx = |i: usize| -> Result<Tensor> {
                let s = d.shape(&format!("te{i}_last"))?;
                bf16_tensor(&dev, &d.f32(&format!("te{i}_last"))?, &s)
            };
            let cfg = meta["cfg"].as_f64().unwrap() as f32;
            let cond = dit.prepare(&ctx(0)?, h, w)?;
            let unc = if cfg != 1.0 { Some(dit.prepare(&ctx(meta["neg_index"].as_u64().unwrap() as usize)?, h, w)?) } else { None };
            let s = Settings { steps: meta["steps"].as_u64().unwrap() as usize, cfg, shift: 5.0, seed: 0, skip_layers: if meta["no_skip"].as_bool().unwrap_or(false) { vec![] } else { vec![22] }, strength: 1.0 };
            let noise = chw_to_tokens(&d.f32(&format!("{tag}_noise"))?, LATENT, h * w);
            let seeded = pipeline::noise_tokens(meta["seed"].as_u64().unwrap(), h, w);
            metrics("noise (native torch.randn)", &seeded, &noise);
            let t0 = std::time::Instant::now();
            let lat = pipeline::sample(&dev, &dit, &s, &Inputs { cond: &cond, uncond: unc.as_ref(), x0: None, mask: None, noise: Some(noise) }, &|_, _| {})?;
            dev.sync()?;
            println!("sampled {} steps in {:.2}s", s.steps, t0.elapsed().as_secs_f64());
            metrics("final latent", &lat.to_f32_vec(&dev)?, &chw_to_tokens(&d.f32(&format!("{tag}_latent"))?, LATENT, h * w));
            drop(dit);
            let vae = Vae::load(dev.clone(), &models::resolve(None, &models::VAE)?)?;
            let rgb = vae.decode(&lat, h, w)?;
            let img = loki_dream::image::Rgb8 { w: w * 16, h: h * 16, data: rgb };
            let out = d.dir.join(format!("{tag}_native.png"));
            img.save_png(&out, &[])?;
            let refimg = loki_dream::image::Rgb8::load(&d.dir.join(format!("{tag}.png")))?;
            let a: Vec<f32> = img.data.iter().map(|v| *v as f32).collect();
            let b: Vec<f32> = refimg.data.iter().map(|v| *v as f32).collect();
            println!("image PSNR vs reference {:.2} dB  ({})", psnr(&a, &b), out.display());
        }
        o => bail!("unknown check {o}"),
    }
    Ok(())
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn upload(args: &[String]) -> Result<()> {
    {
            // read throughput of a model file: mmap touch vs buffered reads vs mmap -> GPU
            let path = PathBuf::from(&args[1]);
            let st = loki_dream::safetensors::SafeTensors::open(&path)?;
            let names: Vec<String> = st.tensors.keys().cloned().collect();
            let t0 = std::time::Instant::now();
            let mut sum = 0u64;
            let mut total = 0usize;
            for n in &names {
                let b = st.bytes(n)?;
                total += b.len();
                for c in b.chunks(4096) { sum = sum.wrapping_add(c[0] as u64); }
            }
            let dt = t0.elapsed().as_secs_f64();
            println!("mmap touch: {:.1} GB in {:.2}s = {:.1} GB/s ({sum})", total as f64 / 1e9, dt, total as f64 / 1e9 / dt);
            let dev = Device::new(0)?;
            let buf = dev.alloc(1 << 30)?;
            let t0 = std::time::Instant::now();
            for n in &names {
                let b = st.bytes(n)?;
                for c in b.chunks(1 << 30) { dev.htod_at(buf.ptr(), c)?; }
            }
            dev.sync()?;
            let dt = t0.elapsed().as_secs_f64();
            println!("mmap -> gpu: {:.2}s = {:.1} GB/s", dt, total as f64 / 1e9 / dt);
        }
    Ok(())
}
