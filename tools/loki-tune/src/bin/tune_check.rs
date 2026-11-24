//! Gradient checks for loki-tune's backward kernels and for a whole training step.
//!
//!   tune_check ops                      every backward kernel against an f64 CPU reference / central differences
//!   tune_check step --model FILE [--tokens 16x16] [--rank 16]
//!                                       directional derivative of the full-model loss along its gradient (and along
//!                                       random directions per linear type): finite differences vs <grad, d>
use anyhow::{ensure, Context, Result};
use loki_dream::cuda::Device;
use loki_dream::tensor::{bf16_bits, bf16_to_f32, DType, Tensor};
use loki_tune::gpu;
use loki_tune::lora::{Kind, LoraSet, Rng};
use std::path::PathBuf;

fn rb(x: f64) -> f32 {
    bf16_to_f32(bf16_bits(x as f32))
}
fn randv(rng: &mut Rng, n: usize, s: f64) -> Vec<f32> {
    (0..n).map(|_| rb(rng.normal() * s)).collect()
}
fn up_bf16(dev: &Device, v: &[f32], shape: &[usize]) -> Result<Tensor> {
    let bits: Vec<u16> = v.iter().map(|x| bf16_bits(*x)).collect();
    Tensor::from_bf16(dev, &bits, shape)
}
fn rel(a: &[f32], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - y).powi(2)).sum();
    let den: f64 = b.iter().map(|y| y * y).sum::<f64>().max(1e-30);
    (num / den).sqrt()
}
fn report(name: &str, e: f64, tol: f64, fails: &mut usize) {
    let ok = e <= tol;
    if !ok {
        *fails += 1;
    }
    println!("  {:<34} rel err {:.2e}  {}", name, e, if ok { "ok" } else { "FAIL" });
}

// ------------------------------------------------------------------------------------------------ ops
fn check_attention(dev: &Device, fails: &mut usize) -> Result<()> {
    let (h, d, n) = (gpu::HEADS, gpu::HEAD_DIM, 300usize);
    let dim = h * d;
    let mut rng = Rng::new(1);
    let qkv = randv(&mut rng, n * 3 * dim, 1.0);
    let dout = randv(&mut rng, n * dim, 1.0);
    let qkv_t = up_bf16(dev, &qkv, &[n, 3 * dim])?;
    let dout_t = up_bf16(dev, &dout, &[n, dim])?;
    let o_t = Tensor::new(dev, DType::BF16, &[n, dim])?;
    let lse_t = Tensor::new(dev, DType::F32, &[h, n])?;
    gpu::attn_fwd(dev, &qkv_t, n, &o_t, &lse_t)?;
    let dqkv_t = Tensor::new(dev, DType::BF16, &[n, 3 * dim])?;
    gpu::attn_bwd(dev, &qkv_t, &o_t, &lse_t, &dout_t, n, &dqkv_t)?;
    let (o, lse, dqkv) = (o_t.to_f32_vec(dev)?, lse_t.to_f32_vec(dev)?, dqkv_t.to_f32_vec(dev)?);
    let scale = 1.0 / (d as f64).sqrt();
    let at = |i: usize, part: usize, hh: usize, k: usize| qkv[i * 3 * dim + part * dim + hh * d + k] as f64;
    let (mut ro, mut rl, mut rdq) = (vec![0f64; n * dim], vec![0f64; h * n], vec![0f64; n * 3 * dim]);
    for hh in 0..h {
        let mut p = vec![0f64; n * n];
        for i in 0..n {
            let mut mx = f64::MIN;
            for j in 0..n {
                let s: f64 = (0..d).map(|k| at(i, 0, hh, k) * at(j, 1, hh, k)).sum::<f64>() * scale;
                p[i * n + j] = s;
                mx = mx.max(s);
            }
            let sum: f64 = (0..n).map(|j| (p[i * n + j] - mx).exp()).sum();
            rl[hh * n + i] = (mx + sum.ln()) * std::f64::consts::LOG2_E;
            for j in 0..n {
                p[i * n + j] = (p[i * n + j] - mx).exp() / sum;
            }
            for k in 0..d {
                ro[i * dim + hh * d + k] = (0..n).map(|j| p[i * n + j] * at(j, 2, hh, k)).sum();
            }
        }
        for i in 0..n {
            let dsum: f64 = (0..d).map(|k| dout[i * dim + hh * d + k] as f64 * ro[i * dim + hh * d + k]).sum();
            for j in 0..n {
                let dp: f64 = (0..d).map(|k| dout[i * dim + hh * d + k] as f64 * at(j, 2, hh, k)).sum();
                let ds = p[i * n + j] * (dp - dsum);
                for k in 0..d {
                    rdq[i * 3 * dim + hh * d + k] += scale * ds * at(j, 1, hh, k);
                    rdq[j * 3 * dim + dim + hh * d + k] += scale * ds * at(i, 0, hh, k);
                    rdq[j * 3 * dim + 2 * dim + hh * d + k] += p[i * n + j] * dout[i * dim + hh * d + k] as f64;
                }
            }
        }
    }
    report("attention out", rel(&o, &ro), 1e-2, fails);
    report("attention lse", rel(&lse, &rl), 1e-4, fails);
    for (part, name) in ["dq", "dk", "dv"].iter().enumerate() {
        let a: Vec<f32> = (0..n).flat_map(|i| dqkv[i * 3 * dim + part * dim..i * 3 * dim + (part + 1) * dim].to_vec()).collect();
        let b: Vec<f64> = (0..n).flat_map(|i| rdq[i * 3 * dim + part * dim..i * 3 * dim + (part + 1) * dim].to_vec()).collect();
        report(&format!("attention {name}"), rel(&a, &b), 2e-2, fails);
    }
    Ok(())
}

/// central differences of L(x) = sum(dy * f(x)) for every coordinate
fn numgrad(x: &[f64], f: &dyn Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut xs = x.to_vec();
    (0..x.len())
        .map(|i| {
            let e = 1e-4 * x[i].abs().max(1.0);
            xs[i] = x[i] + e;
            let a = f(&xs);
            xs[i] = x[i] - e;
            let b = f(&xs);
            xs[i] = x[i];
            (a - b) / (2.0 * e)
        })
        .collect()
}

fn check_modrms(dev: &Device, fails: &mut usize) -> Result<()> {
    let (m, k, eps) = (3usize, 512usize, 1e-6f64);
    let mut rng = Rng::new(2);
    let x = randv(&mut rng, m * k, 1.0);
    let w = randv(&mut rng, k, 0.5);
    let sc: Vec<f32> = (0..k).map(|_| rng.normal() as f32 * 0.3).collect();
    let dy: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
    for modulate in [true, false] {
        let dx = Tensor::zeros(dev, DType::F32, &[m, k])?;
        let sct = Tensor::from_f32(dev, &sc, &[k])?;
        gpu::modrms_bwd(dev, &up_bf16(dev, &x, &[m, k])?, &up_bf16(dev, &w, &[k])?, if modulate { Some(&sct) } else { None }, &Tensor::from_f32(dev, &dy, &[m, k])?, &dx, eps as f32)?;
        let got = dx.to_f32_vec(dev)?;
        let xd: Vec<f64> = x.iter().map(|v| *v as f64).collect();
        let f = |xs: &[f64]| -> f64 {
            let mut l = 0.0;
            for r in 0..m {
                let row = &xs[r * k..(r + 1) * k];
                let rr = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / k as f64 + eps).sqrt();
                for i in 0..k {
                    let c = w[i] as f64 * if modulate { 1.0 + sc[i] as f64 } else { 1.0 };
                    l += dy[r * k + i] as f64 * row[i] * rr * c;
                }
            }
            l
        };
        report(if modulate { "modulated rmsnorm backward" } else { "rmsnorm backward" }, rel(&got, &numgrad(&xd, &f)), 1e-4, fails);
    }
    Ok(())
}

fn check_qk_norm_rope(dev: &Device, fails: &mut usize) -> Result<()> {
    let (m, h, eps) = (3usize, 2usize, 1e-6f64);
    let ts = 3 * h * 128;
    let mut rng = Rng::new(3);
    let pre = randv(&mut rng, m * ts, 1.0);
    let g = randv(&mut rng, m * ts, 1.0);
    let wq = randv(&mut rng, 128, 0.5);
    let wk = randv(&mut rng, 128, 0.5);
    let mut rope = vec![0f32; m * 128];
    for i in 0..m * 64 {
        let a = rng.uniform() * 6.28;
        rope[i * 2] = rb(a.cos());
        rope[i * 2 + 1] = rb(a.sin());
    }
    let gt = up_bf16(dev, &g, &[m, ts])?;
    gpu::qk_norm_rope_bwd(dev, &up_bf16(dev, &pre, &[m, ts])?, &gt, ts, m, h, &up_bf16(dev, &wq, &[128])?, &up_bf16(dev, &wk, &[128])?, eps as f32, &Tensor::from_f32(dev, &rope, &[m, 64, 2])?)?;
    let got = gt.to_f32_vec(dev)?;
    let xd: Vec<f64> = pre.iter().map(|v| *v as f64).collect();
    let f = |xs: &[f64]| -> f64 {
        let mut l = 0.0;
        for t in 0..m {
            for which in 0..2 {
                for hh in 0..h {
                    let base = t * ts + which * h * 128 + hh * 128;
                    let w = if which == 0 { &wq } else { &wk };
                    let v = &xs[base..base + 128];
                    let r = 1.0 / (v.iter().map(|a| a * a).sum::<f64>() / 128.0 + eps).sqrt();
                    let nn: Vec<f64> = (0..128).map(|i| v[i] * r * w[i] as f64).collect();
                    for i in 0..64 {
                        let (c, s) = (rope[(t * 64 + i) * 2] as f64, rope[(t * 64 + i) * 2 + 1] as f64);
                        let oa = nn[i] * c - nn[i + 64] * s;
                        let ob = nn[i + 64] * c + nn[i] * s;
                        l += g[base + i] as f64 * oa + g[base + i + 64] as f64 * ob;
                    }
                }
            }
        }
        l
    };
    let num = numgrad(&xd, &f);
    // q and k columns only (v is untouched)
    let pick = |v: &[f32]| -> Vec<f32> { (0..m).flat_map(|t| v[t * ts..t * ts + 2 * h * 128].to_vec()).collect() };
    let pickd = |v: &[f64]| -> Vec<f64> { (0..m).flat_map(|t| v[t * ts..t * ts + 2 * h * 128].to_vec()).collect() };
    report("qk norm + rope backward", rel(&pick(&got), &pickd(&num)), 1e-2, fails);
    let vcols: Vec<f32> = (0..m).flat_map(|t| got[t * ts + 2 * h * 128..(t + 1) * ts].to_vec()).collect();
    let vref: Vec<f64> = (0..m).flat_map(|t| g[t * ts + 2 * h * 128..(t + 1) * ts].iter().map(|x| *x as f64).collect::<Vec<_>>()).collect();
    report("qk norm + rope leaves v alone", rel(&vcols, &vref), 0.0, fails);
    Ok(())
}

fn check_situ(dev: &Device, fails: &mut usize) -> Result<()> {
    let (m, f) = (4usize, 256usize);
    let mut rng = Rng::new(4);
    let ug = randv(&mut rng, m * 2 * f, 3.0);
    let ds: Vec<f32> = (0..m * f).map(|_| rng.normal() as f32).collect();
    let dug = Tensor::new(dev, DType::BF16, &[m, 2 * f])?;
    gpu::situ_bwd(dev, &up_bf16(dev, &ug, &[m, 2 * f])?, &Tensor::from_f32(dev, &ds, &[m, f])?, &dug)?;
    let got = dug.to_f32_vec(dev)?;
    let xd: Vec<f64> = ug.iter().map(|v| *v as f64).collect();
    let fun = |xs: &[f64]| -> f64 {
        let mut l = 0.0;
        for r in 0..m {
            for i in 0..f {
                let (hh, g) = (xs[r * 2 * f + i], xs[r * 2 * f + f + i]);
                let left = 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp());
                l += ds[r * f + i] as f64 * left * 25.0 * (hh / 25.0).tanh();
            }
        }
        l
    };
    report("situ backward", rel(&got, &numgrad(&xd, &fun)), 1e-2, fails);
    Ok(())
}

fn check_gemm_tn(dev: &Device, fails: &mut usize) -> Result<()> {
    let (m, p, q) = (300usize, 48usize, 200usize);
    let mut rng = Rng::new(5);
    let u = randv(&mut rng, m * p, 1.0);
    let v = randv(&mut rng, m * q, 1.0);
    let c0: Vec<f32> = (0..p * q).map(|_| rng.normal() as f32).collect();
    let c = Tensor::from_f32(dev, &c0, &[p, q])?;
    gpu::gemm_tn(dev, &up_bf16(dev, &u, &[m, p])?, &up_bf16(dev, &v, &[m, q])?, c.ptr, 0.5)?;
    let got = c.to_f32_vec(dev)?;
    let want: Vec<f64> = (0..p * q).map(|i| c0[i] as f64 + 0.5 * (0..m).map(|r| u[r * p + i / q] as f64 * v[r * q + i % q] as f64).sum::<f64>()).collect();
    report("gemm U^T V (LoRA weight grads)", rel(&got, &want), 1e-5, fails);
    Ok(())
}

fn check_fwht_transpose(dev: &Device, fails: &mut usize) -> Result<()> {
    let mut rng = Rng::new(6);
    let (m, k) = (3usize, 512usize);
    let x: Vec<f32> = (0..m * k).map(|_| rng.normal() as f32).collect();
    let t = Tensor::from_f32(dev, &x, &[m, k])?;
    gpu::fwht256(dev, &t)?;
    let want: Vec<f64> = loki_dream::reference::hadamard256_rows(&x, k).iter().map(|v| *v as f64).collect();
    report("hadamard-256", rel(&t.to_f32_vec(dev)?, &want), 1e-5, fails);
    let (n, kk) = (70usize, 40usize);
    let w = randv(&mut rng, n * kk, 1.0);
    let out = Tensor::new(dev, DType::BF16, &[kk, n])?;
    gpu::transpose_w(dev, &up_bf16(dev, &w, &[n, kk])?, None, &out)?;
    let want: Vec<f64> = (0..kk * n).map(|i| w[(i % n) * kk + i / n] as f64).collect();
    report("transpose bf16", rel(&out.to_f32_vec(dev)?, &want), 0.0, fails);
    let q: Vec<i8> = (0..n * kk).map(|_| (rng.below(255) as i32 - 127) as i8).collect();
    let s: Vec<f32> = (0..n).map(|_| rng.uniform() as f32 * 0.01).collect();
    gpu::transpose_w(dev, &Tensor::from_i8(dev, &q, &[n, kk])?, Some(&Tensor::from_f32(dev, &s, &[n])?), &out)?;
    let want: Vec<f64> = (0..kk * n).map(|i| rb(q[(i % n) * kk + i / n] as f64 * s[i % n] as f64) as f64).collect();
    report("transpose int8 dequant", rel(&out.to_f32_vec(dev)?, &want), 1e-6, fails);
    Ok(())
}

// ------------------------------------------------------------------------------------------------ whole step
fn dump_f32(dir: &std::path::Path, name: &str, v: &[f32]) -> Result<()> {
    std::fs::write(dir.join(format!("{name}.bin")), v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())?;
    Ok(())
}

fn check_step(dev: &std::sync::Arc<Device>, model: &PathBuf, hw: (usize, usize), rank: usize, dump: Option<&std::path::Path>, fails: &mut usize) -> Result<()> {
    use loki_dream::dit::{Dit, LATENT};
    let dit = Dit::load(dev.clone(), model, Some(true)).with_context(|| format!("loading {}", model.display()))?;
    println!("  model: {} ({})", model.display(), if dit.int8 { "int8" } else { "bf16" });
    let (h, w) = hw;
    let mut rng = Rng::new(9);
    let ctx = randv(&mut rng, 24 * 4096, 0.5);
    let run = dit.prepare(&up_bf16(dev, &ctx, &[24, 4096])?, h, w)?;
    let x0 = Tensor::from_f32(dev, &(0..h * w * LATENT).map(|_| rng.normal() as f32).collect::<Vec<_>>(), &[h * w, LATENT])?;
    let noise = Tensor::from_f32(dev, &(0..h * w * LATENT).map(|_| rng.normal() as f32).collect::<Vec<_>>(), &[h * w, LATENT])?;
    let sigma = 0.6f32;
    let mut lora = LoraSet::new(dev, rank, rank as f32, 3)?;
    // non-zero up weights so every LoRA matrix has a gradient
    let mut p = lora.p.to_f32_vec(dev)?;
    for l in &lora.lins {
        for (i, x) in p[l.up..l.up + l.n * l.r].iter_mut().enumerate() {
            let (row, col) = (i / l.r, i % l.r);
            if l.kind != Kind::Qkv || row / loki_dream::dit::DIM == col / rank {
                *x = (rng.normal() * 0.01) as f32;
            }
        }
    }
    let set = |lora: &mut LoraSet, v: &[f32]| -> Result<()> {
        dev.htod_at(lora.p.ptr, unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) })?;
        lora.cast(dev)
    };
    set(&mut lora, &p)?;
    let tr = loki_tune::train::Trainer::new(dev, &dit)?;
    let t0 = std::time::Instant::now();
    lora.zero_grads(dev)?;
    let out = tr.step(dev, &dit, &lora, &run, &x0, &noise, sigma)?;
    lora.mask_grads(dev)?;
    dev.sync()?;
    println!("  step: loss {:.5} in {:.2}s ({} tokens)", out.loss, t0.elapsed().as_secs_f64(), h * w + 24);
    let g = lora.g.to_f32_vec(dev)?;
    if let Some(dir) = dump {
        // inputs, LoRA and gradients for ref/ref_tune.py
        std::fs::create_dir_all(dir)?;
        dump_f32(dir, "ctx", &ctx)?;
        dump_f32(dir, "x0", &x0.to_f32_vec(dev)?)?;
        dump_f32(dir, "noise", &noise.to_f32_vec(dev)?)?;
        std::fs::write(dir.join("step.json"), serde_json::json!({"h": h, "w": w, "sigma": sigma, "rank": rank, "alpha": rank, "text_len": 24, "loss": out.loss}).to_string())?;
        let none = Default::default();
        lora.export(dev, &lora.p, false, &dir.join("lora.safetensors"), &none)?;
        lora.export(dev, &lora.g, false, &dir.join("grads.safetensors"), &none)?;
        println!("  dumped to {}", dir.display());
    }
    let loss_at = |lora: &mut LoraSet, v: &[f32]| -> Result<f64> {
        set(lora, v)?;
        let pred = tr.predict(dev, &dit, lora, &run, &{
            let xt = Tensor::new(dev, DType::BF16, &[h * w, LATENT])?;
            gpu::noise_mix_bf16(dev, &x0, &noise, sigma, &xt)?;
            xt
        }, sigma)?;
        let tmp = Tensor::new(dev, DType::BF16, &[h * w, LATENT])?;
        Ok(gpu::mse(dev, &pred, &x0, &noise, &tmp)? as f64)
    };
    // directions: the whole gradient, then each linear type / the first and last block alone
    let mut dirs: Vec<(String, Vec<f32>)> = vec![("all (gradient direction)".into(), g.clone())];
    for kind in Kind::ALL {
        let mut d = vec![0f32; g.len()];
        for l in lora.lins.iter().filter(|l| l.kind == kind) {
            for i in (l.down..l.down + l.r * l.k).chain(l.up..l.up + l.n * l.r) {
                d[i] = g[i];
            }
        }
        dirs.push((format!("{kind:?} only"), d));
    }
    for layer in [0usize, 41] {
        let mut d = vec![0f32; g.len()];
        for l in lora.lins.iter().filter(|l| l.layer == layer) {
            for i in (l.down..l.down + l.r * l.k).chain(l.up..l.up + l.n * l.r) {
                d[i] = g[i];
            }
        }
        dirs.push((format!("block {layer} only"), d));
    }
    for (name, d) in dirs {
        let gd: f64 = g.iter().zip(&d).map(|(a, b)| *a as f64 * *b as f64).sum();
        ensure!(gd > 0.0, "{name}: zero gradient");
        // steps that change the loss by ~2e-2 and ~8e-2 (well above the bf16 rounding noise of the forward);
        // the error reported is the smaller one (curvature grows with the step, noise shrinks)
        let mut best = f64::MAX;
        let mut fds = Vec::new();
        for target in [2e-2, 8e-2] {
            let eps = target / gd;
            let pp: Vec<f32> = p.iter().zip(&d).map(|(a, b)| a + (eps * *b as f64) as f32).collect();
            let pm: Vec<f32> = p.iter().zip(&d).map(|(a, b)| a - (eps * *b as f64) as f32).collect();
            let fd = (loss_at(&mut lora, &pp)? - loss_at(&mut lora, &pm)?) / (2.0 * eps);
            best = best.min((fd - gd).abs() / gd);
            fds.push(fd);
        }
        let e = best;
        let fd = fds[0];
        report(&format!("step d/de loss: {name}"), e, if dit.int8 { 0.15 } else { 0.05 }, fails);
        println!("      finite differences {:.4e} / {:.4e} vs gradient {gd:.4e}", fds[0], fds[1]);
        let _ = fd;
    }
    Ok(())
}

/// int8 ConvRot linears against the same layers in bf16: forward output and input gradient.
fn check_int8(dev: &std::sync::Arc<Device>, fails: &mut usize) -> Result<()> {
    use loki_dream::dit::{Dit, Lin};
    use loki_dream::ops::{self, QuantAct};
    let bf = Dit::load(dev.clone(), &loki_dream::models::resolve_model(std::path::Path::new("kromoss_bf16"))?, Some(true))?;
    let i8 = Dit::load(dev.clone(), &loki_dream::models::resolve_model(std::path::Path::new("kromoss_int8_convrot"))?, Some(true))?;
    let mut rng = Rng::new(11);
    let m = 64usize;
    for (li, name) in [(0usize, "qkv"), (0, "o"), (0, "up_gate"), (0, "down"), (20, "qkv"), (41, "down")] {
        let pick = |d: &Dit| -> *const Lin {
            let b = &d.blocks[li];
            match name {
                "qkv" => &b.qkv,
                "o" => &b.o,
                "up_gate" => &b.up_gate,
                _ => &b.down,
            }
        };
        let (lb, lq) = unsafe { (&*pick(&bf), &*pick(&i8)) };
        let (n, k) = (lb.n(), match lb { Lin::Bf16(w) => w.shape[1], _ => 0 });
        let x = up_bf16(dev, &randv(&mut rng, m * k, 1.0), &[m, k])?;
        let xq = Tensor::new(dev, DType::I8, &[m, k])?;
        let xs = Tensor::new(dev, DType::F32, &[m])?;
        ops::quant_rows(dev, &x, QuantAct::None, None, 0.0, &xq, &xs)?;
        let yb = Tensor::new(dev, DType::BF16, &[m, n])?;
        let yq = Tensor::new(dev, DType::BF16, &[m, n])?;
        loki_tune::train::base_fwd(dev, lb, None, &x, &yb)?;
        loki_tune::train::base_fwd(dev, lq, Some((&xq, &xs)), &x, &yq)?;
        let want: Vec<f64> = yb.to_f32_vec(dev)?.iter().map(|v| *v as f64).collect();
        report(&format!("int8 forward block {li} {name}"), rel(&yq.to_f32_vec(dev)?, &want), 3e-2, fails);
        let dy = up_bf16(dev, &randv(&mut rng, m * n, 1.0), &[m, n])?;
        let gb = Tensor::new(dev, DType::F32, &[m, k])?;
        let gq = Tensor::new(dev, DType::F32, &[m, k])?;
        loki_tune::train::base_bwd(dev, lb, &dy, &gb)?;
        loki_tune::train::base_bwd(dev, lq, &dy, &gq)?;
        let want: Vec<f64> = gb.to_f32_vec(dev)?.iter().map(|v| *v as f64).collect();
        report(&format!("int8 input grad block {li} {name}"), rel(&gq.to_f32_vec(dev)?, &want), 3e-2, fails);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dev = gpu::device()?;
    let mut fails = 0usize;
    match args.first().map(String::as_str) {
        Some("ops") | None => {
            println!("backward kernels:");
            check_attention(&dev, &mut fails)?;
            check_modrms(&dev, &mut fails)?;
            check_qk_norm_rope(&dev, &mut fails)?;
            check_situ(&dev, &mut fails)?;
            check_gemm_tn(&dev, &mut fails)?;
            check_fwht_transpose(&dev, &mut fails)?;
        }
        Some("int8") => {
            println!("int8 ConvRot linears vs bf16:");
            check_int8(&dev, &mut fails)?;
        }
        Some("step") => {
            let mut model = None;
            let mut hw = (16usize, 16usize);
            let mut rank = 16usize;
            let mut dump = None;
            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--model" => model = args.get(i + 1).map(PathBuf::from),
                    "--tokens" => {
                        let (a, b) = args[i + 1].split_once('x').context("--tokens HxW")?;
                        hw = (a.parse()?, b.parse()?);
                    }
                    "--rank" => rank = args[i + 1].parse()?,
                    "--dump" => dump = args.get(i + 1).map(PathBuf::from),
                    o => anyhow::bail!("unknown argument {o}"),
                }
                i += 2;
            }
            let model = loki_dream::models::resolve_model(&model.context("--model FILE")?)?;
            println!("full training step:");
            check_step(&dev, &model, hw, rank, dump.as_deref(), &mut fails)?;
        }
        Some(o) => anyhow::bail!("unknown check {o} (ops | step)"),
    }
    ensure!(fails == 0, "{fails} check(s) failed");
    println!("all checks passed");
    Ok(())
}
