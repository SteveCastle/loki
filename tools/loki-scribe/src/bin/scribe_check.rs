//! Developer checks for loki-scribe (not shipped): tokenizer ids, quantized GEMV/dequant kernels against the CPU
//! reference, and per-layer hidden-state dumps for comparison with an independent implementation.
//!   scribe_check tokenize MODEL.gguf TEXT
//!   scribe_check quant MODEL.gguf TENSOR [M]
//!   scribe_check logits MODEL.gguf TOKEN_IDS(comma separated) [TOPK]
use anyhow::{bail, Context, Result};
use loki_scribe::cuda::Device;
use loki_scribe::gguf::Gguf;
use loki_scribe::quant::{self, Act, Linear};
use loki_scribe::tensor::{DType, Tensor};
use loki_scribe::tokenizer::Tokenizer;
use std::path::Path;
use std::sync::Arc;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    match a.get(1).map(String::as_str) {
        Some("tokenize") => {
            let g = Gguf::open(Path::new(&a[2]))?;
            let t = Tokenizer::from_gguf(&g)?;
            let text = a[3].replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r");
            let ids = t.encode(&text);
            println!("{}", ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","));
            Ok(())
        }
        Some("quant") => quant_check(&a[2], &a[3], a.get(4).and_then(|m| m.parse().ok()).unwrap_or(3)),
        Some("bench") => bench(&a[2], &a[3..]),
        Some("logits") => logits(&a[2], &a[3], a.get(4).and_then(|m| m.parse().ok()).unwrap_or(10)),
        _ => bail!("usage: scribe_check tokenize|quant|logits ..."),
    }
}

/// GEMV (q8 activations) and dequant+bf16 GEMM of one weight against an f64 CPU reference.
fn quant_check(model: &str, name: &str, m: usize) -> Result<()> {
    let dev = Device::new(0)?;
    loki_scribe::ops::init(&dev)?;
    let g = Gguf::open(Path::new(model))?;
    let t = g.tensor(name)?;
    let (n, k) = t.rows_cols();
    println!("{name}: {} [{n} x {k}]", t.ty.name());
    let w = quant::dequant_cpu(t.ty, g.data(t), n * k)?;
    let lin = Linear::load(&dev, &g, name)?;
    let mut x = vec![0f32; m * k];
    let mut s = 12345u64;
    for v in x.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *v = ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0;
    }
    let nn = n.min(4096);
    let mut refv = vec![0f64; m * nn];
    for mi in 0..m {
        for r in 0..nn {
            let mut acc = 0f64;
            for c in 0..k {
                acc += x[mi * k + c] as f64 * w[r * k + c] as f64;
            }
            refv[mi * nn + r] = acc;
        }
    }
    let report = |what: &str, got: &[f32], ld: usize| {
        let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
        for mi in 0..m {
            for r in 0..nn {
                let d = got[mi * ld + r] as f64 - refv[mi * nn + r];
                num += d * d;
                den += refv[mi * nn + r] * refv[mi * nn + r];
                mx = mx.max(d.abs());
            }
        }
        println!("  {what:<8} rel_rms {:.2e}  max_abs {:.3e}", (num / den).sqrt(), mx);
    };
    let xt = Tensor::from_f32(&dev, &x, &[m, k])?;
    let xq = Tensor::new(&dev, DType::I8, &[m * k])?;
    let xd = Tensor::new(&dev, DType::F32, &[m * k / 32])?;
    quant::quant_q8(&dev, &xt, m, k, &xq, &xd)?;
    let out = Tensor::new(&dev, DType::F32, &[m, n])?;
    lin.gemv(&dev, &Act { x: &xt, xq: &xq, xd: &xd }, m, &out, false)?;
    report("gemv", &out.to_f32_vec(&dev)?, n);
    let deq = Tensor::new(&dev, DType::BF16, &[n * k])?;
    let wb = lin.bf16(&dev, &deq)?;
    let wv = wb.to_f32_vec(&dev)?;
    let mut werr = 0f64;
    for i in 0..(nn * k) {
        werr = werr.max(((wv[i] - w[i]) / (w[i].abs() + 1e-3)).abs() as f64);
    }
    println!("  dequant  max rel {:.2e} (bf16 rounding ~4e-3)", werr);
    let xb_v: Vec<u16> = x.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
    let xb = Tensor::from_bf16(&dev, &xb_v, &[m, k])?;
    lin.gemm(&dev, &xb, &out, loki_scribe::ops::Epi::Store, &deq)?;
    report("gemm", &out.to_f32_vec(&dev)?, n);
    Ok(())
}

/// Run a token sequence through the model (prefill) and print the top logits of the last position, then
/// greedy-continue a few tokens through the decode path.
fn logits(model: &str, ids: &str, topk: usize) -> Result<()> {
    let dev = Device::new(0)?;
    loki_scribe::ops::init(&dev)?;
    let g = Arc::new(Gguf::open(Path::new(model))?);
    let tok = Tokenizer::from_gguf(&g)?;
    let m = loki_scribe::model::TextModel::load(dev.clone(), g.clone(), 4096, &|_, _| {})?;
    let ids: Vec<u32> = ids.split(',').map(|s| s.trim().parse().context("token id")).collect::<Result<_>>()?;
    let e = m.cfg.n_embd;
    let mut st = loki_scribe::model::SeqState::new(&dev, &m.cfg, 4096)?;
    let mut x = vec![0f32; ids.len() * e];
    for (i, t) in ids.iter().enumerate() {
        m.embed(*t, &mut x[i * e..(i + 1) * e])?;
    }
    let pos: Vec<[i64; 3]> = (0..ids.len() as i64).map(|p| [p, p, p]).collect();
    let t0 = std::time::Instant::now();
    let lg = m.forward(&x, &pos, &mut [&mut st], true, true)?.unwrap();
    dev.sync()?;
    println!("prefill {} tokens: {:.3}s", ids.len(), t0.elapsed().as_secs_f64());
    let show = |lg: &[f32]| {
        let mut idx: Vec<usize> = (0..lg.len()).collect();
        idx.sort_by(|a, b| lg[*b].partial_cmp(&lg[*a]).unwrap());
        let mx = lg[idx[0]];
        let z: f64 = lg.iter().map(|v| ((v - mx) as f64).exp()).sum();
        for &i in idx.iter().take(topk) {
            println!("  {:>7} {:>9.4}  p={:.4}  {:?}", i, lg[i], ((lg[i] - mx) as f64).exp() / z, tok.decode(&[i as u32]));
        }
        idx[0] as u32
    };
    let mut next = show(&lg);
    let mut out = vec![next];
    let mut p = ids.len() as i64;
    let t1 = std::time::Instant::now();
    for _ in 0..16 {
        let mut x = vec![0f32; e];
        m.embed(next, &mut x)?;
        let lg = m.forward(&x, &[[p, p, p]], &mut [&mut st], false, true)?.unwrap();
        p += 1;
        let mut best = 0;
        for i in 0..lg.len() {
            if lg[i] > lg[best] {
                best = i;
            }
        }
        next = best as u32;
        out.push(next);
    }
    dev.sync()?;
    println!("decode 16 tokens: {:.1} ms/token", t1.elapsed().as_secs_f64() * 1000.0 / 16.0);
    println!("greedy: {:?}", tok.decode(&out));
    println!("ids: {:?}", out);
    Ok(())
}

/// GEMV bandwidth per weight and batch size.
fn bench(model: &str, names: &[String]) -> Result<()> {
    let dev = Device::new(0)?;
    loki_scribe::ops::init(&dev)?;
    let g = Gguf::open(Path::new(model))?;
    for name in names {
        let lin = Linear::load(&dev, &g, name)?;
        let (n, k) = (lin.n, lin.k);
        let xt = Tensor::zeros(&dev, DType::F32, &[8, k])?;
        let xq = Tensor::zeros(&dev, DType::I8, &[8 * k])?;
        let xd = Tensor::zeros(&dev, DType::F32, &[8 * k / 32])?;
        let out = Tensor::new(&dev, DType::F32, &[8, n])?;
        let mut line = format!("{name:<28} {:?} {n}x{k} {:.1} MB:", lin.kind, lin.bytes() as f64 / 1e6);
        for m in [1usize, 2, 4, 8] {
            let a = Act { x: &xt, xq: &xq, xd: &xd };
            lin.gemv(&dev, &a, m, &out, false)?;
            dev.sync()?;
            let reps = 50;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                lin.gemv(&dev, &a, m, &out, false)?;
            }
            dev.sync()?;
            let s = t.elapsed().as_secs_f64() / reps as f64;
            line += &format!("  M{m} {:.0}us {:.0}GB/s", s * 1e6, lin.bytes() as f64 / s / 1e9);
        }
        println!("{line}");
    }
    Ok(())
}
