//! Validate the MiniMax H3 conditioning encoder against ComfyUI reference dumps (ref/ref_te.py).
//!
//!   te_check [--ref DIR] [--model FILE] [--selftest] [--bench] [case ...]
use anyhow::{Context, Result};
use h3ref2va::cuda::Device;
use h3ref2va::image::Rgb8;
use h3ref2va::te_h3::{self, RefItem, TextEncoder};
use std::path::{Path, PathBuf};

fn load_item(v: &serde_json::Value) -> Result<RefItem> {
    Ok(match v["type"].as_str().unwrap_or("") {
        "image" => {
            let im = Rgb8::load(Path::new(v["path"].as_str().context("image path")?))?;
            RefItem::Image { w: im.w, h: im.h, rgb: im.data }
        }
        "audio" => RefItem::Audio,
        "video" => {
            let mut frames = Vec::new();
            for p in v["frames"].as_array().context("frames")? {
                let im = Rgb8::load(Path::new(p.as_str().unwrap()))?;
                frames.push((im.w, im.h, im.data));
            }
            let timestamps = v["timestamps"].as_array().map(|a| a.iter().map(|t| t.as_f64().unwrap() as f32).collect()).unwrap_or_default();
            RefItem::Video { frames, timestamps }
        }
        t => anyhow::bail!("unknown item type {t}"),
    })
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
        v
    };
    let ref_dir = PathBuf::from(take("--ref").unwrap_or_else(|| r"C:\Users\steph\dev\h3ref2va\ref_out\te".into()));
    let model = PathBuf::from(take("--model").unwrap_or_else(|| r"C:\Users\steph\dev\h3ref2va\models\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors".into()));
    let vision_dir = take("--vision");
    let vision_img = take("--vision-img");
    let long = take("--long");
    let selftest =args.iter().any(|a| a == "--selftest");
    let bench = args.iter().any(|a| a == "--bench");
    args.retain(|a| !a.starts_with("--"));
    let cases: Vec<String> = if args.is_empty() { vec!["text".into(), "img1".into(), "img2".into(), "video".into(), "odd".into()] } else { args };

    let dev = Device::new(0)?;
    let free0 = dev.free_mem()?;
    let t0 = std::time::Instant::now();
    let te = TextEncoder::load(dev.clone(), &model)?;
    let free1 = dev.free_mem()?;
    println!("load {:.1}s, resident {:.2} GB", t0.elapsed().as_secs_f64(), (free0 - free1) as f64 / 1e9);

    if selftest {
        for (name, m, rel, tf, ts) in te.fp4_selftest(0, &[117, 512, 2048, 3496])? {
            let fl = 2.0 * m as f64 * match name.as_str() { "qkv" => 10240.0 * 5120.0, "o" => 5120.0 * 8192.0, "gate_up" => 51200.0 * 5120.0, _ => 5120.0 * 25600.0 };
            println!("fp4 {name:<8} M={m:<5} rel_err {rel:.2e}  fused {tf:7.3} ms ({:5.1} TFLOPS)  dequant+gemm {ts:7.3} ms", fl / tf / 1e9);
        }
    }

    if let Some(vd) = vision_dir {
        // compare vision-tower outputs against ref_te_vision.py dumps: <dir> holds merged.bin, deep{0,1,2}.bin of <dir>/../inputs/<img>
        let img = Rgb8::load(Path::new(&vision_img.clone().context("--vision-img")?))?;
        let feats = te.vision_features(&[RefItem::Image { w: img.w, h: img.h, rgb: img.data }])?;
        let (m, d) = &feats[0];
        let rd = |n: &str| -> Result<Vec<f32>> { Ok(std::fs::read(Path::new(&vd).join(n))?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()) };
        let rel = |a: &[f32], b: &[f32]| -> f64 {
            let (mut n, mut dd) = (0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                n += ((x - y) as f64).powi(2);
                dd += (*y as f64).powi(2);
            }
            (n / dd).sqrt()
        };
        println!("vision merged rel {:.3e}", rel(m, &rd("merged.bin")?));
        for i in 0..3 {
            println!("vision deep{i} rel {:.3e}", rel(&d[i], &rd(&format!("deep{i}.bin"))?));
        }
    }

    if let Some(spec) = long {
        // synthetic long-sequence benchmark: "<imgs>x<W>x<H>[,v<frames>x<W>x<H>]"
        let mut items = Vec::new();
        let mut seed = 7u32;
        let mut noise = |w: usize, h: usize| -> Vec<u8> {
            (0..w * h * 3)
                .map(|i| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (((i % (w * 3)) as u32 * 255 / (w as u32 * 3)) as u8).wrapping_add((seed >> 28) as u8)
                })
                .collect()
        };
        for part in spec.split(',') {
            let v: Vec<usize> = part.trim_start_matches('v').split('x').map(|s| s.parse().unwrap()).collect();
            if part.starts_with('v') {
                let frames = (0..v[0]).map(|_| (v[1], v[2], noise(v[1], v[2]))).collect();
                items.push(RefItem::Audio);
                items.push(RefItem::Video { frames, timestamps: (0..v[0]).map(|i| i as f32 / 2.0).collect() });
            } else {
                for _ in 0..v[0] {
                    items.push(RefItem::Image { w: v[1], h: v[2], rgb: noise(v[1], v[2]) });
                }
            }
        }
        let prompt = "A cinematic shot of a lighthouse on a cliff during a storm, waves crashing, a man in a yellow raincoat shouts over the wind. ".repeat(4);
        for it in 0..3 {
            let f = dev.free_mem()?;
            let c = te.encode(&prompt, &items)?;
            let t = te.last_timing.lock().unwrap().clone();
            println!("long[{it}] L={} vision tokens {} time {:.3}s (vision {:.3}s, decoder {:.3}s)  activations ~{:.2} GB", c.tags.len(), c.tags.iter().filter(|&&t| t == 0).count(), t.total, t.vision, t.decoder, (f as f64 - dev.free_mem()? as f64) / 1e9);
        }
    }

    let mut all_ok = true;
    for case in &cases {
        let dir = ref_dir.join(case);
        let cj: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("case.json")).with_context(|| format!("{case}: case.json"))?)?;
        let prompt = cj["prompt"].as_str().unwrap_or("").to_string();
        let items: Vec<RefItem> = cj["items"].as_array().cloned().unwrap_or_default().iter().map(load_item).collect::<Result<_>>()?;
        // tokens
        let ref_tok: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(dir.join("tokens.json"))?)?;
        let ref_tok: Vec<i64> = ref_tok.iter().map(|v| v.as_i64().unwrap_or(-1)).collect();
        let my_tok = te.tokenize(&prompt, &items)?;
        let tok_ok = ref_tok == my_tok;
        if !tok_ok {
            println!("  {case}: TOKENS DIFFER\n    ref {:?}\n    got {:?}", ref_tok, my_tok);
        }
        // encode
        let cond = te.encode(&prompt, &items)?;
        if bench {
            for _ in 0..2 {
                te.encode(&prompt, &items)?;
            }
        }
        let tim = te.last_timing.lock().unwrap().clone();
        let got = cond.context.to_f32_vec(&dev)?;
        let rf: Vec<f32> = std::fs::read(dir.join("context.bin"))?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let ref_tags: Vec<u8> = serde_json::from_slice::<Vec<u8>>(&std::fs::read(dir.join("tags.json"))?)?;
        let tags_ok = ref_tags == cond.tags;
        let l = cond.tags.len();
        if rf.len() != got.len() {
            println!("  {case}: SHAPE MISMATCH ref {} rows vs {} rows", rf.len() / te_h3::HIDDEN, l);
            all_ok = false;
            continue;
        }
        let stats = |sel: &dyn Fn(usize) -> bool| -> (f64, f64, f64) {
            let (mut num, mut den, mut dot, mut ng, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
            for r in 0..l {
                if !sel(r) {
                    continue;
                }
                for c in 0..te_h3::HIDDEN {
                    let (a, b) = (got[r * te_h3::HIDDEN + c] as f64, rf[r * te_h3::HIDDEN + c] as f64);
                    num += (a - b) * (a - b);
                    den += b * b;
                    dot += a * b;
                    ng += a * a;
                    mx = mx.max((a - b).abs());
                }
            }
            ((num / den.max(1e-30)).sqrt(), dot / (ng.sqrt() * den.sqrt()).max(1e-30), mx)
        };
        let (rel, cos, mx) = stats(&|_| true);
        let (rel_t, _, _) = stats(&|r| cond.tags[r] == 1);
        let (rel_v, _, _) = stats(&|r| cond.tags[r] == 0);
        // worst row
        let mut worst = (0usize, 0f64);
        for r in 0..l {
            let (mut num, mut den) = (0f64, 0f64);
            for c in 0..te_h3::HIDDEN {
                let (a, b) = (got[r * te_h3::HIDDEN + c] as f64, rf[r * te_h3::HIDDEN + c] as f64);
                num += (a - b) * (a - b);
                den += b * b;
            }
            let e = (num / den.max(1e-30)).sqrt();
            if e > worst.1 {
                worst = (r, e);
            }
        }
        let ok = tok_ok && tags_ok && rel < 3e-2;
        all_ok &= ok;
        println!(
            "{case:<6} L={l:<5} tokens {} tags {}  rel_l2 {rel:.3e} (text {rel_t:.3e}, vision {rel_v:.3e})  cos {cos:.6}  max_abs {mx:.3e}  worst row {} ({:.3e})  time {:.3}s (vision {:.3}s, decoder {:.3}s)  {}",
            if tok_ok { "OK" } else { "DIFF" },
            if tags_ok { "OK" } else { "DIFF" },
            worst.0,
            worst.1,
            tim.total,
            tim.vision,
            tim.decoder,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    te_h3::PROF.report();
    let free2 = dev.free_mem()?;
    println!("peak VRAM (pool high-water incl. weights) {:.2} GB", (free0 - free2) as f64 / 1e9);
    drop(te);
    te_h3::release_device_memory(&dev)?;
    let free3 = dev.free_mem()?;
    println!("after drop + trim: {:.2} GB still held", (free0 as f64 - free3 as f64) / 1e9);
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
