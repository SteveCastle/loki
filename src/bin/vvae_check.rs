//! Validate the video VAE against ComfyUI reference dumps (ref/ref_vvae.py).
//! Usage: vvae_check [ref_dir] [case,case,...] [--roundtrip] [--png out_dir]
use anyhow::{Context, Result};
use loki_reshoot::cuda::Device;
use loki_reshoot::tensor::Tensor;
use loki_reshoot::vae_video::VideoVae;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn read_f32(p: &Path) -> Result<Vec<f32>> {
    let b = std::fs::read(p).with_context(|| format!("reading {}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

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

fn psnr_u8(a: &[u8], b_f: &[f32]) -> (f64, f32) {
    let mut se = 0f64;
    let mut mx = 0f32;
    for (x, y) in a.iter().zip(b_f) {
        let d = *x as f32 - y;
        se += (d as f64) * (d as f64);
        mx = mx.max(d.abs());
    }
    let mse = se / a.len() as f64;
    (10.0 * (255.0f64 * 255.0 / mse.max(1e-12)).log10(), mx)
}

fn used(dev: &Device) -> f64 {
    let free = dev.free_mem().unwrap_or(0);
    (dev.total_mem - free) as f64 / (1u64 << 30) as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ref_dir = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| r"C:\Users\steph\dev\loki-reshoot\ref_out\vvae".into()));
    let cases: Vec<String> = args.get(2).map(|s| s.split(',').map(|x| x.to_string()).collect()).unwrap_or_else(|| vec!["v448".into(), "v448_39".into(), "v640".into(), "img448".into()]);
    let roundtrip = args.iter().any(|a| a == "--roundtrip");
    let png_dir = args.iter().position(|a| a == "--png").and_then(|i| args.get(i + 1)).map(PathBuf::from);
    let debug = args.iter().any(|a| a == "--debug");
    let model = PathBuf::from(std::env::var("LOKI_VVAE").unwrap_or_else(|_| r"C:\Users\steph\dev\loki-reshoot\models\minimax_h3_video_vae_fp16.safetensors".into()));

    let dev = Device::new(0)?;
    let base = used(&dev);
    let t0 = Instant::now();
    let vae = VideoVae::load(dev.clone(), &model)?;
    dev.sync()?;
    println!("loaded in {:.2}s, weights {:.2} GB", t0.elapsed().as_secs_f64(), used(&dev) - base);
    let after_load = used(&dev);

    if let Some(spec) = args.iter().position(|a| a == "--bench").and_then(|i| args.get(i + 1)) {
        let v: Vec<usize> = spec.split('x').map(|s| s.parse().unwrap()).collect();
        let (w, h, t) = (v[0], v[1], v[2]);
        let mut pix = vec![0u8; t * h * w * 3];
        for f in 0..t {
            for y in 0..h {
                for x in 0..w {
                    let i = ((f * h + y) * w + x) * 3;
                    let fx = (x as f32 + 3.0 * f as f32) / 37.0;
                    let fy = y as f32 / 23.0;
                    pix[i] = (128.0 + 100.0 * (fx.sin() * fy.cos())) as u8;
                    pix[i + 1] = (128.0 + 100.0 * ((fx * 0.7 + fy * 1.3).sin())) as u8;
                    pix[i + 2] = ((x * 255 / w + f * 4) % 256) as u8;
                }
            }
        }
        for rep in 0..2 {
            dev.sync()?;
            let t0 = Instant::now();
            let lat = vae.encode(&pix, t, h, w)?;
            dev.sync()?;
            let te = t0.elapsed().as_secs_f64();
            let (tl, hl, wl) = (lat.shape[1], lat.shape[2], lat.shape[3]);
            let t0 = Instant::now();
            let mut out: Vec<u8> = Vec::new();
            let n = vae.decode(&lat, tl, hl, wl, &mut |px, _| {
                out.extend_from_slice(px);
                Ok(())
            })?;
            let td = t0.elapsed().as_secs_f64();
            let n_cmp = out.len().min(pix.len());
            let pixf: Vec<f32> = pix[..n_cmp].iter().map(|&v| v as f32).collect();
            println!("bench {w}x{h}x{t} (rep {rep}): encode {te:.2}s -> [24,{tl},{hl},{wl}], decode {td:.2}s -> {n} frames, roundtrip PSNR {:.2} dB, pool peak {:.2} GB", psnr_u8(&out[..n_cmp], &pixf).0, used(&dev) - after_load);
        }
        dev.sync()?;
        loki_reshoot::vae_video::profile_report();
        return Ok(());
    }

    for case in &cases {
        let d = ref_dir.join(case);
        let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("meta.json"))?)?;
        let (t, h, w) = (meta["t"].as_u64().unwrap() as usize, meta["h"].as_u64().unwrap() as usize, meta["w"].as_u64().unwrap() as usize);
        let ls: Vec<usize> = meta["latent_shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let (tl, hl, wl) = (ls[1], ls[2], ls[3]);
        let pix = std::fs::read(d.join("pixels.u8"))?;
        let ref_lat = read_f32(&d.join("latent.bin"))?;
        let ref_dec = read_f32(&d.join("decoded.bin"))?;
        let ref_frames = meta["decoded_shape"][0].as_u64().unwrap() as usize;
        println!("== {case}: {t}x{h}x{w} -> latent [24,{tl},{hl},{wl}] -> {ref_frames} frames");

        if debug && d.join("dbg_enc_out.bin").exists() {
            let r = read_f32(&d.join("dbg_enc_out.bin"))?;
            let sh = &meta["dbg_enc_out"];
            let (dt, dh, dw) = (sh[2].as_u64().unwrap() as usize, sh[3].as_u64().unwrap() as usize, sh[4].as_u64().unwrap() as usize);
            let tclip = t.min(17);
            let pd = Tensor::from_buf(dev.upload(&pix[..tclip * h * w * 3])?, loki_reshoot::tensor::DType::U8, &[tclip * h * w * 3]);
            let mine = vae.encode_tile(&pd, tclip, h, w, 0, if t == 1 { 1 } else { 17 }, 0, 0, dh * 16, dw * 16)?;
            let n = 24 * dt * dh * dw;
            let (mx, rl) = stats(&mine[..n], &r[..n]);
            println!("  dbg tile0 moments(mean): max {mx:.4e} rel {rl:.4e}");
            let zr = read_f32(&d.join("dbg_dec_in.bin"))?;
            let sh = &meta["dbg_dec_in"];
            let (zt, zh, zw) = (sh[2].as_u64().unwrap() as usize, sh[3].as_u64().unwrap() as usize, sh[4].as_u64().unwrap() as usize);
            // dbg_dec_in is the de-normalized latent: re-normalize for decode_tile_raw
            let mut zn = zr[..24 * zt * zh * zw].to_vec();
            let lm = [0.858090341091156f32, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264];
            let lsd = [1.2223774194717407f32, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523];
            let np = zt * zh * zw;
            for c in 0..24 {
                let (m, s) = (half::f16::from_f32(lm[c]).to_f32(), half::f16::from_f32(lsd[c]).to_f32());
                for v in &mut zn[c * np..(c + 1) * np] {
                    *v = (*v - m) / s;
                }
            }
            let zt_ = Tensor::from_f32(&dev, &zn, &[24, zt, zh, zw])?;
            let raw = vae.decode_tile_raw(&zt_, 0, zt, 0, 0, zh, zw)?;
            let rr = read_f32(&d.join("dbg_dec_out.bin"))?;
            let (mx, rl) = stats(&raw, &rr[..raw.len()]);
            println!("  dbg tile0 decoder raw: max {mx:.4e} rel {rl:.4e}");
        }

        // ---- encode
        dev.sync()?;
        let t0 = Instant::now();
        let lat = vae.encode(&pix, t, h, w)?;
        dev.sync()?;
        let t_enc = t0.elapsed().as_secs_f64();
        let mem_enc = used(&dev) - after_load;
        let mine = lat.to_f32_vec(&dev)?;
        anyhow::ensure!(lat.shape == vec![24, tl, hl, wl], "latent shape {:?}", lat.shape);
        let (mx, rl) = stats(&mine, &ref_lat);
        println!("  encode: {t_enc:.3}s  latent max-abs {mx:.4e} rel-L2 {rl:.4e}  (comfy {:.3}s)  pool {:.2} GB", meta["t_enc"].as_f64().unwrap_or(0.0), mem_enc);

        // ---- decode the reference latent
        let zref = Tensor::from_f32(&dev, &ref_lat, &[24, tl, hl, wl])?;
        let mut frames: Vec<u8> = Vec::new();
        dev.sync()?;
        let t0 = Instant::now();
        let n = vae.decode(&zref, tl, hl, wl, &mut |px, _n| {
            frames.extend_from_slice(px);
            Ok(())
        })?;
        let t_dec = t0.elapsed().as_secs_f64();
        let mem_dec = used(&dev) - after_load;
        anyhow::ensure!(n == ref_frames && n == VideoVae::decode_frame_count(tl), "frame count {n} vs ref {ref_frames}");
        let ref255: Vec<f32> = ref_dec.iter().map(|v| v * 255.0).collect();
        let (p, mxd) = psnr_u8(&frames, &ref255);
        println!("  decode: {t_dec:.3}s  {n} frames  PSNR vs comfy {p:.2} dB  max-abs {mxd:.1}/255  (comfy {:.3}s)  pool {:.2} GB", meta["t_dec"].as_f64().unwrap_or(0.0), mem_dec);
        // per-frame PSNR spread
        let fs = h * w * 3;
        let pf: Vec<f64> = (0..n).map(|f| psnr_u8(&frames[f * fs..(f + 1) * fs], &ref255[f * fs..(f + 1) * fs]).0).collect();
        let worst = pf.iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  decode per-frame PSNR min {worst:.2} dB");

        // ---- roundtrip
        let mut rt_frames: Vec<u8> = Vec::new();
        if roundtrip || png_dir.is_some() {
            vae.decode(&lat, tl, hl, wl, &mut |px, _n| {
                rt_frames.extend_from_slice(px);
                Ok(())
            })?;
            let n_cmp = rt_frames.len().min(pix.len());
            let pixf: Vec<f32> = pix[..n_cmp].iter().map(|&v| v as f32).collect();
            let (p_rt, _) = psnr_u8(&rt_frames[..n_cmp], &pixf);
            let comfy_rt = psnr_u8(&pix[..n_cmp], &ref255[..n_cmp]).0;
            println!("  roundtrip PSNR vs input: ours {p_rt:.2} dB, comfy {comfy_rt:.2} dB");
        }
        if let Some(dir) = &png_dir {
            std::fs::create_dir_all(dir)?;
            // rows: input / comfy / ours (ref latent) / ours roundtrip; columns: up to 4 frames
            let picks: Vec<usize> = if n == 1 { vec![0] } else { vec![0, n / 3, 2 * n / 3, n - 1] };
            let cols = picks.len();
            let rows = 4;
            let mut img = vec![0u8; rows * h * cols * w * 3];
            let fs = h * w * 3;
            for (ci, &f) in picks.iter().enumerate() {
                for r in 0..rows {
                    for y in 0..h {
                        for x in 0..w {
                            for c in 0..3 {
                                let i = f * fs + (y * w + x) * 3 + c;
                                let v = match r {
                                    0 => pix[i],
                                    1 => (ref_dec[i] * 255.0).round().clamp(0.0, 255.0) as u8,
                                    2 => frames[i],
                                    _ => *rt_frames.get(i).unwrap_or(&0),
                                };
                                img[((r * h + y) * cols * w + ci * w + x) * 3 + c] = v;
                            }
                        }
                    }
                }
            }
            let out = loki_reshoot::image::Rgb8 { w: cols * w, h: rows * h, data: img };
            out.save_png(&dir.join(format!("{case}.png")))?;
        }
    }
    Ok(())
}
