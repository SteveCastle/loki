//! Stage checks of the Krea 2 / FLUX.2 klein / Anima paths against the dumps of ref/ref_arch.py.
//!
//!   arch_check te   ARCH DIR --te FILE [--dit FILE]          text conditioning (taps / Qwen3 states / adapter output)
//!   arch_check dit  ARCH DIR --dit FILE [--int8|--bf16]      one transformer forward from the reference conditioning
//!   arch_check vae  ARCH DIR --vae FILE                      decode (and encode when dumped)
//!
//! Prints cosine, max |diff|, mean |diff|, relative L2 and the reference's mean |x| per tensor.
use anyhow::{bail, Context, Result};
use loki_dream::anima::Anima;
use loki_dream::arch::Arch;
use loki_dream::cuda::Device;
use loki_dream::engines::{self, Cond};
use loki_dream::flux2::Flux2;
use loki_dream::krea2::Krea2;
use loki_dream::model::Model;
use loki_dream::qvae::{self, QVae};
use loki_dream::tensor::{bf16_bits, DType, Tensor};
use loki_dream::vae::Vae;
use std::path::{Path, PathBuf};

struct Dump {
    dir: PathBuf,
    index: serde_json::Value,
}
impl Dump {
    fn open(dir: &Path) -> Result<Dump> {
        Ok(Dump { dir: dir.to_path_buf(), index: serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).context("index.json")?)? })
    }
    fn has(&self, n: &str) -> bool {
        self.index.get(n).is_some()
    }
    fn shape(&self, n: &str) -> Result<Vec<usize>> {
        Ok(self.index[n]["shape"].as_array().with_context(|| format!("{n} not in dump"))?.iter().map(|v| v.as_u64().unwrap() as usize).collect())
    }
    fn f32(&self, n: &str) -> Result<Vec<f32>> {
        let b = std::fs::read(self.dir.join(format!("{n}.bin"))).with_context(|| format!("reading {n}.bin"))?;
        let int = self.index[n]["dtype"] == "i32";
        Ok(b.chunks_exact(4).map(|c| if int { i32::from_le_bytes(c.try_into().unwrap()) as f32 } else { f32::from_le_bytes(c.try_into().unwrap()) }).collect())
    }
    fn json(&self, f: &str) -> Result<serde_json::Value> {
        Ok(serde_json::from_str(&std::fs::read_to_string(self.dir.join(f))?)?)
    }
}

fn metrics(name: &str, got: &[f32], want: &[f32]) {
    if got.len() != want.len() {
        println!("{name:<34} LENGTH MISMATCH {} vs {}", got.len(), want.len());
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
    println!("{name:<34} cos {:.6}  max|d| {:<10.4} mean|d| {:<10.5} relL2 {:<9.5} ref mean|x| {:.4}", dot / (ng.sqrt() * nw.sqrt()).max(1e-30), maxd, sumd / n, (nd / nw.max(1e-30)).sqrt(), mw / n);
}

fn psnr(a: &[f32], b: &[f32]) -> f64 {
    let mse: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.len() as f64;
    10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10()
}

fn bf16_tensor(dev: &Device, v: &[f32], shape: &[usize]) -> Result<Tensor> {
    let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x)).collect();
    Tensor::from_bf16(dev, &bits, shape)
}

fn arg(args: &[String], name: &str) -> Option<PathBuf> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(PathBuf::from)
}

/// [16, 2h, 2w] (CHW) -> tokens [h*w, 64] (c*4 + dy*2 + dx)
fn pack16(v: &[f32], h: usize, w: usize) -> Vec<f32> {
    let (hh, ww) = (2 * h, 2 * w);
    let mut out = vec![0f32; v.len()];
    for c in 0..16 {
        for y in 0..hh {
            for x in 0..ww {
                out[((y / 2) * w + x / 2) * 64 + c * 4 + (y % 2) * 2 + x % 2] = v[(c * hh + y) * ww + x];
            }
        }
    }
    out
}

/// `arch_check sdxl te|unet|vae DIR --ckpt FILE`: the SDXL stages against ref/ref_sdxl.py.
fn sdxl(args: &[String]) -> Result<()> {
    use loki_dream::clip::{chunks, ClipTokenizer, SdxlText, TextCond};
    use loki_dream::safetensors::SafeTensors;
    use loki_dream::sdxl::Unet;
    use loki_dream::vae::Plane;
    let d = Dump::open(Path::new(&args[1]))?;
    let ckpt = arg(args, "--ckpt").context("--ckpt FILE")?;
    let dev = Device::new(0)?;
    loki_dream::ops::gemm_init(&dev)?;
    loki_dream::ops::attn_init(&dev)?;
    Vae::init_kernels(&dev)?;
    match args[0].as_str() {
        "te" => {
            let prompts: Vec<String> = serde_json::from_value(d.json("prompts.json")?)?;
            let st = SafeTensors::open(&ckpt)?;
            let te = SdxlText::load(dev.clone(), &st)?;
            let tok = ClipTokenizer::new()?;
            for (i, p) in prompts.iter().enumerate() {
                let c = &chunks(&tok, p)[0];
                let ids_l: Vec<f32> = c.filled(loki_dream::clip::EOS).iter().map(|&x| x as f32).collect();
                let ids_g: Vec<f32> = c.filled(0).iter().map(|&x| x as f32).collect();
                println!("prompt {i}: {p:?}");
                println!("  ids CLIP-L {}  ids bigG {}", if ids_l == d.f32(&format!("te{i}_ids_l"))? { "identical" } else { "DIFFER" }, if ids_g == d.f32(&format!("te{i}_ids_g"))? { "identical" } else { "DIFFER" });
                let tc = te.encode(&tok, p)?;
                let want = d.f32(&format!("te{i}_ctx"))?;
                let (l, g): (Vec<f32>, Vec<f32>) = (tc.ctx.chunks(2048).flat_map(|r| r[..768].to_vec()).collect(), tc.ctx.chunks(2048).flat_map(|r| r[768..].to_vec()).collect());
                metrics("  context CLIP-L", &l, &want.chunks(2048).flat_map(|r| r[..768].to_vec()).collect::<Vec<_>>());
                metrics("  context bigG", &g, &want.chunks(2048).flat_map(|r| r[768..].to_vec()).collect::<Vec<_>>());
                metrics("  pooled", &tc.pooled, &d.f32(&format!("te{i}_pooled"))?);
            }
        }
        "unet" => {
            let meta = d.json("unet.json")?;
            let (w, h) = (meta["size"][0].as_u64().unwrap() as usize, meta["size"][1].as_u64().unwrap() as usize);
            let t = meta["t"].as_f64().unwrap() as f32;
            let (lh, lw) = (h / 8, w / 8);
            let unet = Unet::load(dev.clone(), &ckpt)?;
            let tc = TextCond { ctx: d.f32("te0_ctx")?, tokens: 77, pooled: d.f32("te0_pooled")? };
            let br = unet.prepare(&tc, h, w)?;
            let x = d.f32("unet_x")?;
            let n = lh * lw;
            let mut bits = vec![0u16; n * 8];
            for p in 0..n {
                for c in 0..4 {
                    bits[p * 8 + c] = bf16_bits(x[c * n + p]);
                }
            }
            let inp = Plane { t: Tensor::from_bf16(&dev, &bits, &[n, 8])?, h: lh, w: lw, c: 8 };
            let t0 = std::time::Instant::now();
            let o = unet.forward(&br, &inp, t)?;
            dev.sync()?;
            let el = t0.elapsed().as_secs_f64();
            let v = o.t.to_f32_vec(&dev)?;
            let got: Vec<f32> = (0..4 * n).map(|i| v[(i % n) * 8 + i / n]).collect();
            metrics(&format!("eps {w}x{h} t={t} ({el:.2}s)"), &got, &d.f32("unet_eps")?);
        }
        "vae" => {
            let meta = d.json("vae.json")?;
            let (w, h) = (meta["size"][0].as_u64().unwrap() as usize, meta["size"][1].as_u64().unwrap() as usize);
            let (lh, lw) = (h / 8, w / 8);
            let vae = Vae::load_sd(dev.clone(), &ckpt)?;
            let lat = d.f32("vae_lat")?;
            let n = lh * lw;
            let hwc: Vec<f32> = (0..4 * n).map(|i| lat[(i % 4) * n + i / 4]).collect();
            let rgb = vae.decode_sd(&hwc, lh, lw)?;
            let got: Vec<f32> = rgb.iter().map(|&b| b as f32).collect();
            let want = d.f32("vae_rgb")?;
            println!("decode PSNR {:.2} dB", psnr(&got, &want));
            let img: Vec<f32> = want.iter().map(|v| v / 255.0).collect();
            let enc = vae.encode_sd(&img, h, w)?;
            let we = d.f32("vae_enc")?;
            metrics("encode (latent mean)", &enc, &(0..4 * n).map(|i| we[(i % 4) * n + i / 4]).collect::<Vec<_>>());
        }
        o => bail!("arch_check sdxl: unknown stage {o}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(|a| a == "sdxl").unwrap_or(false) {
        return sdxl(&args[1..]);
    }
    if args.len() < 3 {
        bail!("usage: arch_check te|dit|vae ARCH DIR [--te FILE] [--dit FILE] [--vae FILE] [--int8|--bf16]");
    }
    let a = Arch::parse(&args[1])?;
    let d = Dump::open(Path::new(&args[2]))?;
    let dev = Device::new(0)?;
    loki_dream::ops::gemm_init(&dev)?;
    loki_dream::ops::attn_init(&dev)?;
    Vae::init_kernels(&dev)?;
    let int8 = args.iter().any(|x| x == "--int8");
    match args[0].as_str() {
        "te" => {
            let te = arg(&args, "--te").context("--te FILE")?;
            let prompts: Vec<String> = serde_json::from_value(d.json("prompts.json")?)?;
            let t0 = std::time::Instant::now();
            let conds = engines::encode_texts(&dev, a, &te, &prompts, true)?;
            println!("text encoder: {} texts in {:.2}s", prompts.len(), t0.elapsed().as_secs_f64());
            let anima = match (a, arg(&args, "--dit")) {
                (Arch::Anima, Some(p)) => Some(Anima::load(dev.clone(), &p, false)?),
                _ => None,
            };
            for (i, c) in conds.iter().enumerate() {
                match c {
                    Cond::Flux2(x) => {
                        let got = x.to_f32_vec(&dev)?;
                        let want = d.f32(&format!("te{i}_ctx"))?;
                        metrics(&format!("te{i} ctx (512 rows)"), &got, &want);
                        let valid = loki_dream::tokenizer::Tokenizer::qwen3()?.encode(&engines::klein_template(&prompts[i])).len();
                        std::fs::write(d.dir.join(format!("native_te{i}_ctx.bin")), got.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
                        let w = got.len() / 512;
                        metrics(&format!("te{i} ctx ({valid} real rows)"), &got[..valid * w], &want[..valid * w]);
                        metrics(&format!("te{i} ctx (padding rows)"), &got[valid * w..], &want[valid * w..]);
                        let hd = w / 3;
                        for (k, l) in [(0usize, 9usize), (1, 18), (2, 27)] {
                            let pick = |v: &[f32]| -> Vec<f32> { v[..valid * w].chunks(w).flat_map(|r| r[k * hd..(k + 1) * hd].to_vec()).collect() };
                            metrics(&format!("te{i} hidden {l} (real rows)"), &pick(&got), &pick(&want));
                        }
                    }
                    Cond::Krea2(x) => {
                        let got = x.to_f32_vec(&dev)?;
                        let mask = d.f32(&format!("te{i}_mask"))?;
                        let s = d.shape(&format!("te{i}_taps"))?; // [L, 12, 2560]
                        let row = s[1] * s[2];
                        let want: Vec<f32> = d.f32(&format!("te{i}_taps"))?.chunks(row).zip(&mask).filter(|(_, m)| **m > 0.5).flat_map(|(r, _)| r.to_vec()).collect();
                        metrics(&format!("te{i} taps (valid rows)"), &got, &want);
                        let per = row / s[1];
                        for (k, l) in [(0usize, 2usize), (5, 17), (11, 35)] {
                            let pick = |v: &[f32]| -> Vec<f32> { v.chunks(row).flat_map(|r| r[k * per..(k + 1) * per].to_vec()).collect() };
                            metrics(&format!("te{i} hidden {l}"), &pick(&got), &pick(&want));
                        }
                    }
                    Cond::Anima(src, valid, t5) => {
                        let got = src.to_f32_vec(&dev)?;
                        std::fs::write(d.dir.join(format!("native_te{i}_qwen.bin")), got.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
                        metrics(&format!("te{i} qwen last ({valid} valid)"), &got, &d.f32(&format!("te{i}_qwen"))?);
                        let want_t5: Vec<u32> = d.f32(&format!("te{i}_t5"))?.iter().map(|v| *v as u32).collect();
                        println!("te{i} T5 ids match: {} ({} ids)", *t5 == want_t5, t5.len());
                        if let Some(m) = &anima {
                            let ctx = m.adapter(src, *valid, t5)?;
                            let cv = ctx.to_f32_vec(&dev)?;
                            metrics(&format!("te{i} adapter ctx"), &cv, &d.f32(&format!("te{i}_ctx"))?);
                            let qs = d.shape(&format!("te{i}_qwen"))?;
                            let refsrc = bf16_tensor(&dev, &d.f32(&format!("te{i}_qwen"))?, &qs)?;
                            let ctx2 = m.adapter(&refsrc, *valid, &want_t5)?;
                            metrics(&format!("te{i} adapter ctx (ref input)"), &ctx2.to_f32_vec(&dev)?, &d.f32(&format!("te{i}_ctx"))?);
                        }
                    }
                }
            }
        }
        "dit" => {
            let dit = arg(&args, "--dit").context("--dit FILE")?;
            let meta = d.json("dit.json")?;
            let (w, h) = (meta["size"][0].as_u64().unwrap() as usize, meta["size"][1].as_u64().unwrap() as usize);
            let sigma = meta["sigma"].as_f64().unwrap() as f32;
            let (lh, lw) = (h / 16, w / 16);
            let n_texts = serde_json::from_value::<Vec<String>>(d.json("prompts.json")?)?.len();
            let t0 = std::time::Instant::now();
            match a {
                Arch::Flux2 => {
                    let m = Flux2::load(dev.clone(), &dit, int8)?;
                    let x = bf16_tensor(&dev, &d.f32("dit_x")?, &[lh * lw, 128])?;
                    let md = m.modulation(sigma)?;
                    for i in 0..n_texts {
                        let s = d.shape(&format!("te{i}_ctx"))?;
                        let run = m.prepare(&bf16_tensor(&dev, &d.f32(&format!("te{i}_ctx"))?, &s)?, lh, lw)?;
                        let out = Tensor::new(&dev, DType::BF16, &[lh * lw, 128])?;
                        m.forward(&run, &x, &md, &out)?;
                        metrics(&format!("velocity {i}"), &out.to_f32_vec(&dev)?, &d.f32(&format!("dit_v{i}"))?);
                    }
                }
                Arch::Krea2 => {
                    let m = Krea2::load(dev.clone(), &dit, int8)?;
                    let x = bf16_tensor(&dev, &d.f32("dit_x")?, &[lh * lw, 64])?;
                    let md = m.modulation(sigma)?;
                    for i in 0..n_texts {
                        let mask = d.f32(&format!("te{i}_mask"))?;
                        let s = d.shape(&format!("te{i}_taps"))?;
                        let row = s[1] * s[2];
                        let v: Vec<f32> = d.f32(&format!("te{i}_taps"))?.chunks(row).zip(&mask).filter(|(_, m)| **m > 0.5).flat_map(|(r, _)| r.to_vec()).collect();
                        let t = v.len() / row;
                        let run = m.prepare(&bf16_tensor(&dev, &v, &[t * s[1], s[2]])?, lh, lw)?;
                        if d.has(&format!("dit_txt{i}")) {
                            metrics(&format!("text tokens {i}"), &run.txt.to_f32_vec(&dev)?, &d.f32(&format!("dit_txt{i}"))?);
                        }
                        let out = Tensor::new(&dev, DType::BF16, &[lh * lw, 64])?;
                        m.forward(&run, &x, &md, &out)?;
                        let got = out.to_f32_vec(&dev)?;
                        metrics(&format!("velocity {i} ({t} text tokens)"), &got, &d.f32(&format!("dit_v{i}"))?);
                        if d.has(&format!("dit_v{i}_32")) {
                            let truth = d.f32(&format!("dit_v{i}_32"))?;
                            metrics(&format!("velocity {i} native vs fp32"), &got, &truth);
                            metrics(&format!("velocity {i} reference vs fp32"), &d.f32(&format!("dit_v{i}"))?, &truth);
                        }
                    }
                }
                Arch::Anima => {
                    let m = Anima::load(dev.clone(), &dit, int8)?;
                    let x = bf16_tensor(&dev, &pack16(&d.f32("dit_x5")?, lh, lw), &[lh * lw, 64])?;
                    let md = m.modulation(sigma)?;
                    for i in 0..n_texts {
                        let s = d.shape(&format!("te{i}_ctx"))?;
                        let run = m.prepare(&bf16_tensor(&dev, &d.f32(&format!("te{i}_ctx"))?, &s)?, lh, lw)?;
                        let out = Tensor::new(&dev, DType::BF16, &[lh * lw, 64])?;
                        m.forward(&run, &x, &md, &out)?;
                        metrics(&format!("velocity {i}"), &out.to_f32_vec(&dev)?, &pack16(&d.f32(&format!("dit_v{i}"))?, lh, lw));
                    }
                }
                Arch::Moss => bail!("use dream_check for MOSS-Image"),
                Arch::Sdxl => bail!("SDXL is checked with the sdxl command"),
                Arch::H3Image => bail!("MiniMax H3 Image is checked in loki-reshoot (dit_image_check, ivae_check, h3image_check)"),
            }
            dev.sync()?;
            println!("load + forwards {:.2}s", t0.elapsed().as_secs_f64());
        }
        "vae" => {
            let vp = arg(&args, "--vae").context("--vae FILE")?;
            let zs = d.shape("vae_z")?; // [C, H/8, W/8]
            let (c, zh, zw) = (zs[0], zs[1], zs[2]);
            let (lh, lw) = (zh / 2, zw / 2);
            let z = d.f32("vae_z")?;
            let want = d.f32("vae_img")?; // [3, H, W] in [-1, 1]
            let npix = zh * zw * 64;
            let want_u8: Vec<f32> = (0..npix).flat_map(|p| (0..3).map(move |ch| (p, ch))).map(|(p, ch)| ((want[ch * npix + p] + 1.0) * 127.5).round().clamp(0.0, 255.0)).collect();
            let rgb = if a == Arch::Flux2 {
                let v = Vae::load(dev.clone(), &vp)?;
                let mean = v.bn_mean.as_ref().context("bn")?.to_f32_vec(&dev)?;
                let var = v.bn_var.as_ref().context("bn")?.to_f32_vec(&dev)?;
                let mut tok = vec![0f32; lh * lw * 128];
                for ci in 0..c {
                    for y in 0..zh {
                        for x in 0..zw {
                            let ch = ci * 4 + (y % 2) * 2 + x % 2;
                            tok[((y / 2) * lw + x / 2) * 128 + ch] = (z[(ci * zh + y) * zw + x] - mean[ch]) / (var[ch] + loki_dream::vae::BN_EPS).sqrt();
                        }
                    }
                }
                v.decode(&Tensor::from_f32(&dev, &tok, &[lh * lw, 128])?, lh, lw)?
            } else {
                let v = QVae::load(dev.clone(), &vp)?;
                let mut zn = z.clone();
                for ci in 0..16 {
                    for p in 0..zh * zw {
                        zn[ci * zh * zw + p] = (z[ci * zh * zw + p] - qvae::LATENTS_MEAN[ci]) / qvae::LATENTS_STD[ci];
                    }
                }
                v.decode(&Tensor::from_f32(&dev, &pack16(&zn, lh, lw), &[lh * lw, 64])?, lh, lw)?
            };
            let got: Vec<f32> = rgb.iter().map(|v| *v as f32).collect();
            metrics("vae decode (0..255)", &got, &want_u8);
            println!("vae decode PSNR {:.2} dB", psnr(&got, &want_u8));
            if a != Arch::Flux2 && d.shape("vae_in").is_ok() {
                let s = d.shape("vae_in")?; // [3, H, W] in [-1, 1]
                let (h, w) = (s[1], s[2]);
                let x = d.f32("vae_in")?;
                let rgb: Vec<f32> = (0..h * w).flat_map(|p| (0..3).map(move |c| (p, c))).map(|(p, c)| (x[c * h * w + p] + 1.0) * 0.5).collect();
                let v = QVae::load(dev.clone(), &vp)?;
                let tok = v.encode(&rgb, h, w)?.to_f32_vec(&dev)?;
                let mut zn = d.f32("vae_enc")?; // [16, H/8, W/8]
                for ci in 0..16 {
                    for p in 0..h * w / 64 {
                        zn[ci * h * w / 64 + p] = (zn[ci * h * w / 64 + p] - qvae::LATENTS_MEAN[ci]) / qvae::LATENTS_STD[ci];
                    }
                }
                metrics("vae encode (normalized tokens)", &tok, &pack16(&zn, h / 16, w / 16));
            }
        }
        "te-layers" => {
            // every hidden state of the first prompt (FLUX.2 template, padded to 512) -> DIR/native_h{i}.bin
            let te = loki_dream::qwen3::Qwen3::open(dev.clone(), &arg(&args, "--te").context("--te FILE")?)?;
            let prompts: Vec<String> = serde_json::from_value(d.json("prompts.json")?)?;
            let mut ids = loki_dream::tokenizer::Tokenizer::qwen3()?.encode(&engines::klein_template(&prompts[0]));
            let valid = ids.len();
            ids.resize(512, loki_dream::tokenizer::qwen3::ENDOFTEXT);
            let taps: Vec<usize> = (0..=te.cfg.layers).collect();
            let out = te.encode(&[loki_dream::qwen3::Prompt { tokens: ids, valid }], &taps, false, &|_| {})?;
            for (i, t) in out[0].taps.iter().enumerate() {
                std::fs::write(d.dir.join(format!("native_h{i}.bin")), t.to_f32_vec(&dev)?.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>())?;
            }
            println!("wrote {} hidden states ({valid} real tokens)", taps.len());
            let r = te.rope(512)?.to_f32_vec(&dev)?;
            println!("rope row 511 cos f0..3 {:?} f30..31 {:?}", [r[511 * 128], r[511 * 128 + 2], r[511 * 128 + 4], r[511 * 128 + 6]], [r[511 * 128 + 60], r[511 * 128 + 62]]);
        }
        o => bail!("unknown check {o}"),
    }
    Ok(())
}
