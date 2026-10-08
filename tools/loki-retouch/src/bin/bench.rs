//! Kernel micro-benchmarks: `cargo run --release --bin bench`
use anyhow::Result;
use loki_retouch::cuda::Device;
use loki_retouch::ops::{self, Act, AttnArgs, AttnView, Epi, QuantAct};
use loki_retouch::tensor::{DType, Tensor};
use std::time::Instant;

fn time<F: FnMut() -> Result<()>>(dev: &Device, iters: usize, mut f: F) -> Result<f64> {
    f()?;
    dev.sync()?;
    let t0 = Instant::now();
    for _ in 0..iters {
        f()?;
    }
    dev.sync()?;
    Ok(t0.elapsed().as_secs_f64() / iters as f64)
}

fn main() -> Result<()> {
    let dev = Device::new(0)?;
    ops::gemm_init(&dev)?;
    ops::attn_init(&dev)?;
    let m = 32768usize;
    for (n, k) in [(4096usize, 4096usize), (24576, 4096), (4096, 12288), (1024, 4096)] {
        let a = Tensor::zeros(&dev, DType::I8, &[m, k])?;
        let b = Tensor::zeros(&dev, DType::I8, &[n, k])?;
        let sa = Tensor::zeros(&dev, DType::F32, &[m])?;
        let sb = Tensor::zeros(&dev, DType::F32, &[n])?;
        let out = Tensor::new(&dev, DType::BF16, &[m, n])?;
        let t = time(&dev, 5, || ops::gemm_i8(&dev, &a, &sa, &b, &sb, None, Epi::Store, &out))?;
        let tops = 2.0 * m as f64 * n as f64 * k as f64 / t / 1e12;
        println!("int8 gemm {m}x{n}x{k}: {:.2} ms  {:.0} TOPS", t * 1e3, tops);
    }
    {
        let (n, k) = (24576usize, 4096usize);
        let a = Tensor::zeros(&dev, DType::I8, &[m, k])?;
        let b = Tensor::zeros(&dev, DType::I8, &[n, k])?;
        let sa = Tensor::zeros(&dev, DType::F32, &[m])?;
        let sb = Tensor::zeros(&dev, DType::F32, &[n])?;
        let out = Tensor::new(&dev, DType::BF16, &[m, n / 2])?;
        let t = time(&dev, 5, || ops::gemm_i8(&dev, &a, &sa, &b, &sb, None, Epi::SwiGluPairs, &out))?;
        println!("int8 gemm {m}x{n}x{k} swiglu-pairs epilogue: {:.2} ms  {:.0} TOPS", t * 1e3, 2.0 * m as f64 * n as f64 * k as f64 / t / 1e12);
    }
    {
        let (n, k) = (4096usize, 4096usize);
        let a = Tensor::zeros(&dev, DType::BF16, &[m, k])?;
        let b = Tensor::zeros(&dev, DType::BF16, &[n, k])?;
        let out = Tensor::new(&dev, DType::BF16, &[m, n])?;
        let t = time(&dev, 5, || ops::gemm_bf16(&dev, &a, &b, None, Act::None, Epi::Store, &out))?;
        println!("bf16 gemm {m}x{n}x{k}: {:.2} ms  {:.0} TFLOPS", t * 1e3, 2.0 * m as f64 * n as f64 * k as f64 / t / 1e12);
    }
    {
        let x = Tensor::zeros(&dev, DType::BF16, &[m, 24576])?;
        let q = Tensor::new(&dev, DType::I8, &[m, 12288])?;
        let s = Tensor::new(&dev, DType::F32, &[m])?;
        let t = time(&dev, 5, || ops::quant_rows(&dev, &x, QuantAct::SwiGlu, None, 0.0, &q, &s))?;
        println!("quant swiglu {m}x24576: {:.2} ms ({:.0} GB/s read)", t * 1e3, (m * 24576 * 2) as f64 / t / 1e9);
        let x2 = Tensor::zeros(&dev, DType::BF16, &[m, 4096])?;
        let q2 = Tensor::new(&dev, DType::I8, &[m, 4096])?;
        let t = time(&dev, 5, || ops::quant_rows(&dev, &x2, QuantAct::None, None, 0.0, &q2, &s))?;
        println!("quant none {m}x4096: {:.2} ms", t * 1e3);
    }
    {
        let (nq, nk, h, d) = (32400usize, 39400usize, 32usize, 128usize);
        let q = Tensor::zeros(&dev, DType::BF16, &[nq, h, d])?;
        let k = Tensor::zeros(&dev, DType::BF16, &[nk, h, d])?;
        let v = Tensor::zeros(&dev, DType::BF16, &[nk, h, d])?;
        let o = Tensor::new(&dev, DType::BF16, &[nq, h, d])?;
        let args = AttnArgs {
            q: AttnView::contiguous(&q, h, d),
            k1: AttnView::contiguous(&k, h, d),
            v1: AttnView::contiguous(&v, h, d),
            k2: AttnView::empty(),
            v2: AttnView::empty(),
            out: AttnView::contiguous(&o, h, d),
            hq: h,
            hk: h,
            d,
            kv_limit: None,
            causal: false,
            causal_off: 0,
        };
        let t = time(&dev, 3, || ops::flash_attn(&dev, &args))?;
        let flops = 4.0 * nq as f64 * nk as f64 * d as f64 * h as f64;
        println!("attention {nq}x{nk} h{h} d{d}: {:.1} ms  {:.0} TFLOPS", t * 1e3, flops / t / 1e12);
        // per-step DiT estimate: 32 blocks
        println!("  -> x32 blocks = {:.2} s per step", t * 32.0);
        // sage attention on the same shapes, prefix 7000 + target
        ops::sage_init(&dev)?;
        let n1 = 7000usize;
        let n2 = nk - n1;
        let kv1 = ops::quantize_kv(&dev, k.ptr, v.ptr, h * d, n1, h)?;
        let kv2 = ops::quantize_kv(&dev, k.rows(n1, n2).ptr, v.rows(n1, n2).ptr, h * d, n2, h)?;
        let qq = ops::quantize_q(&dev, q.ptr, h * d, nq, h, &kv1, Some(&kv2))?;
        let t = time(&dev, 3, || ops::sage_attn(&dev, &qq, nq, h, &kv1, Some(&kv2), AttnView::contiguous(&o, h, d), None))?;
        println!("sage attention {nq}x{nk} h{h} d{d}: {:.1} ms  {:.0} TFLOPS-equiv", t * 1e3, flops / t / 1e12);
        println!("  -> x32 blocks = {:.2} s per step", t * 32.0);
        let tq = time(&dev, 3, || {
            let kv2 = ops::quantize_kv(&dev, k.rows(n1, n2).ptr, v.rows(n1, n2).ptr, h * d, n2, h)?;
            let _ = ops::quantize_q(&dev, q.ptr, h * d, nq, h, &kv1, Some(&kv2))?;
            Ok(())
        })?;
        println!("per-step quantization of target q/k/v: {:.1} ms (x32 = {:.2} s)", tq * 1e3, tq * 32.0);
    }
    Ok(())
}
