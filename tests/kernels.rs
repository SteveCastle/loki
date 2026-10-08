use h3ref2va::cuda::Device;
use h3ref2va::ops::{self, Act, Epi, QuantAct};
use h3ref2va::reference;
use h3ref2va::tensor::{bf16_bits, bf16_to_f32, DType, Tensor};
use rand::{Rng, SeedableRng};

fn dev() -> std::sync::Arc<Device> {
    let d = Device::new(0).unwrap();
    ops::gemm_init(&d).unwrap();
    d
}
fn rnd(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut r = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n).map(|_| r.gen_range(-scale..scale)).collect()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}
fn bf16_round(v: f32) -> f32 {
    bf16_to_f32(bf16_bits(v))
}

#[test]
fn quant_rows_matches_reference() {
    let d = dev();
    for (m, k, act) in [(3usize, 512usize, QuantAct::None), (5, 4096, QuantAct::None), (2, 1024, QuantAct::SwiGlu), (4, 768, QuantAct::GeluTanh), (3, 4096, QuantAct::RmsNorm)] {
        let kin = if act == QuantAct::SwiGlu { k * 2 } else { k };
        let x = rnd(m * kin, 1, 2.0);
        let w = rnd(k, 2, 1.0);
        // CPU reference: activation then hadamard then quant
        let mut pre = vec![0f32; m * k];
        for r in 0..m {
            for i in 0..k {
                let v = match act {
                    QuantAct::None => x[r * kin + i],
                    QuantAct::GeluTanh => {
                        let v = x[r * kin + i];
                        0.5 * v * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v * v * v)).tanh())
                    }
                    QuantAct::SwiGlu => {
                        let g = x[r * kin + i];
                        let u = x[r * kin + k + i];
                        g / (1.0 + (-g).exp()) * u
                    }
                    QuantAct::RmsNorm => {
                        let ss: f32 = (0..k).map(|j| x[r * kin + j] * x[r * kin + j]).sum::<f32>() / k as f32;
                        x[r * kin + i] / (ss + 1e-6).sqrt() * bf16_round(w[i])
                    }
                };
                pre[r * k + i] = v;
            }
        }
        let rot = reference::hadamard256_rows(&pre, k);
        let (q_ref, s_ref) = reference::quant_rows_i8(&rot, k);
        let xt = Tensor::from_f32(&d, &x, &[m, kin]).unwrap();
        let wt = Tensor::from_bf16(&d, &w.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>(), &[k]).unwrap();
        let q = Tensor::new(&d, DType::I8, &[m, k]).unwrap();
        let s = Tensor::new(&d, DType::F32, &[m]).unwrap();
        ops::quant_rows(&d, &xt, act, Some(&wt), 1e-6, &q, &s).unwrap();
        let qg = q.to_f32_vec(&d).unwrap();
        let sg = s.to_f32_vec(&d).unwrap();
        for r in 0..m {
            assert!((sg[r] - s_ref[r]).abs() <= 1e-5 * s_ref[r].abs() + 1e-7, "scale mismatch row {r}: {} vs {}", sg[r], s_ref[r]);
        }
        let mut off_by_more = 0;
        for i in 0..m * k {
            let diff = (qg[i] - q_ref[i] as f32).abs();
            if diff > 1.0 {
                off_by_more += 1;
            }
        }
        assert_eq!(off_by_more, 0, "quant mismatch for act {act:?}");
        let dequant_err: f32 = (0..m * k).map(|i| ((qg[i] - q_ref[i] as f32) * s_ref[i / k]).abs()).fold(0.0, f32::max);
        assert!(dequant_err < 0.05, "dequant err {dequant_err}");
        println!("quant {act:?} m={m} k={k} ok");
    }
}

#[test]
fn gemm_i8_matches_reference() {
    let d = dev();
    for (m, n, k) in [(1usize, 8usize, 64usize), (130, 136, 128), (257, 256, 4096), (64, 1024, 12288), (40, 24, 80), (1030, 2048, 256)] {
        let mut r = rand::rngs::StdRng::seed_from_u64(7);
        let a: Vec<i8> = (0..m * k).map(|_| r.gen_range(-127..=127)).collect();
        let b: Vec<i8> = (0..n * k).map(|_| r.gen_range(-127..=127)).collect();
        let sa: Vec<f32> = (0..m).map(|_| r.gen_range(0.001..0.01)).collect();
        let sb: Vec<f32> = (0..n).map(|_| r.gen_range(0.001..0.01)).collect();
        let bias: Vec<f32> = (0..n).map(|_| r.gen_range(-1.0..1.0)).collect();
        let res: Vec<f32> = (0..m * n).map(|_| r.gen_range(-1.0..1.0)).collect();
        let gate: Vec<f32> = (0..n).map(|_| r.gen_range(-1.0..1.0)).collect();
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for kk in 0..k {
                    acc += a[i * k + kk] as i64 * b[j * k + kk] as i64;
                }
                want[i * n + j] = res[i * n + j] + (acc as f32 * sa[i] * sb[j] + bias[j]) * gate[j];
            }
        }
        let at = Tensor::from_i8(&d, &a, &[m, k]).unwrap();
        let bt = Tensor::from_i8(&d, &b, &[n, k]).unwrap();
        let sat = Tensor::from_f32(&d, &sa, &[m]).unwrap();
        let sbt = Tensor::from_f32(&d, &sb, &[n]).unwrap();
        let biast = Tensor::from_f32(&d, &bias, &[n]).unwrap();
        let rest = Tensor::from_f32(&d, &res, &[m, n]).unwrap();
        let gatet = Tensor::from_f32(&d, &gate, &[n]).unwrap();
        let out = Tensor::new(&d, DType::F32, &[m, n]).unwrap();
        ops::gemm_i8(&d, &at, &sat, &bt, &sbt, Some(&biast), Epi::AddResGated(&rest, &gatet), &out).unwrap();
        let got = out.to_f32_vec(&d).unwrap();
        let err = max_abs_diff(&got, &want);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(err <= 1e-4 * scale + 1e-4, "gemm_i8 m={m} n={n} k={k} err={err} (scale {scale})");
        // bf16 output, plain store
        let outb = Tensor::new(&d, DType::BF16, &[m, n]).unwrap();
        ops::gemm_i8(&d, &at, &sat, &bt, &sbt, None, Epi::Store, &outb).unwrap();
        let gotb = outb.to_f32_vec(&d).unwrap();
        let mut maxrel = 0f32;
        for i in 0..m {
            for j in 0..n {
                let mut acc: i64 = 0;
                for kk in 0..k {
                    acc += a[i * k + kk] as i64 * b[j * k + kk] as i64;
                }
                let w = acc as f32 * sa[i] * sb[j];
                maxrel = maxrel.max((gotb[i * n + j] - w).abs() / (w.abs() + 1.0));
            }
        }
        assert!(maxrel < 1e-2, "bf16 out rel err {maxrel}");
        // swiglu-pairs epilogue: out[m][j] = silu(v[2j]) * v[2j+1]
        if n % 16 == 0 {
            let outs = Tensor::new(&d, DType::BF16, &[m, n / 2]).unwrap();
            ops::gemm_i8(&d, &at, &sat, &bt, &sbt, None, Epi::SwiGluPairs, &outs).unwrap();
            let gots = outs.to_f32_vec(&d).unwrap();
            let mut maxrel = 0f32;
            for i in 0..m {
                for j in 0..n / 2 {
                    let dot = |col: usize| -> f32 {
                        let mut acc: i64 = 0;
                        for kk in 0..k { acc += a[i * k + kk] as i64 * b[col * k + kk] as i64; }
                        acc as f32 * sa[i] * sb[col]
                    };
                    let g = dot(2 * j);
                    let u = dot(2 * j + 1);
                    let w = g / (1.0 + (-g).exp()) * u;
                    maxrel = maxrel.max((gots[i * (n / 2) + j] - w).abs() / (w.abs() + 1.0));
                }
            }
            assert!(maxrel < 1e-2, "swiglu pairs rel err {maxrel}");
        }
        println!("gemm_i8 {m}x{n}x{k} ok err={err}");
    }
}

#[test]
fn gemm_bf16_matches_reference() {
    let d = dev();
    for (m, n, k) in [(1usize, 8usize, 32usize), (130, 136, 96), (300, 256, 1152), (64, 4096, 4096), (70, 72, 4304), (33, 16, 72)] {
        let a = rnd(m * k, 3, 1.0).into_iter().map(bf16_round).collect::<Vec<_>>();
        let b = rnd(n * k, 4, 1.0).into_iter().map(bf16_round).collect::<Vec<_>>();
        let bias = rnd(n, 5, 1.0);
        let res = rnd(m * n, 6, 1.0);
        let mut want = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f64;
                for kk in 0..k {
                    acc += a[i * k + kk] as f64 * b[j * k + kk] as f64;
                }
                let v = acc as f32 + bias[j];
                let v = v / (1.0 + (-v).exp()); // silu
                want[i * n + j] = v + res[i * n + j];
            }
        }
        let at = Tensor::from_bf16(&d, &a.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>(), &[m, k]).unwrap();
        let bt = Tensor::from_bf16(&d, &b.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>(), &[n, k]).unwrap();
        let biast = Tensor::from_f32(&d, &bias, &[n]).unwrap();
        let rest = Tensor::from_f32(&d, &res, &[m, n]).unwrap();
        let out = Tensor::new(&d, DType::F32, &[m, n]).unwrap();
        ops::gemm_bf16(&d, &at, &bt, Some(&biast), Act::Silu, Epi::AddRes(&rest), &out).unwrap();
        let got = out.to_f32_vec(&d).unwrap();
        let err = max_abs_diff(&got, &want);
        let scale = want.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(err <= 2e-3 * scale + 1e-3, "gemm_bf16 m={m} n={n} k={k} err={err} scale={scale}");
        println!("gemm_bf16 {m}x{n}x{k} ok err={err}");
    }
}

#[test]
fn layernorm_rmsnorm_match() {
    let d = dev();
    let (m, n) = (7usize, 4096usize);
    let x = rnd(m * n, 9, 3.0);
    let w = rnd(n, 10, 1.0).into_iter().map(bf16_round).collect::<Vec<_>>();
    let b = rnd(n, 11, 1.0).into_iter().map(bf16_round).collect::<Vec<_>>();
    let xt = Tensor::from_f32(&d, &x, &[m, n]).unwrap();
    let wt = Tensor::from_bf16(&d, &w.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>(), &[n]).unwrap();
    let bt = Tensor::from_bf16(&d, &b.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>(), &[n]).unwrap();
    let out = Tensor::new(&d, DType::F32, &[m, n]).unwrap();
    ops::layernorm(&d, &xt, Some(&wt), Some(&bt), 1e-6, &out).unwrap();
    let got = out.to_f32_vec(&d).unwrap();
    for r in 0..m {
        let row = &x[r * n..(r + 1) * n];
        let mean = row.iter().sum::<f32>() / n as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        for i in 0..n {
            let want = (row[i] - mean) / (var + 1e-6).sqrt() * w[i] + b[i];
            assert!((got[r * n + i] - want).abs() < 1e-3, "ln mismatch {} vs {}", got[r * n + i], want);
        }
    }
    ops::rmsnorm(&d, &xt, Some(&wt), false, 1e-6, &out).unwrap();
    let got = out.to_f32_vec(&d).unwrap();
    for r in 0..m {
        let row = &x[r * n..(r + 1) * n];
        let ss = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
        for i in 0..n {
            let want = row[i] / (ss + 1e-6).sqrt() * w[i];
            assert!((got[r * n + i] - want).abs() < 1e-3, "rms mismatch");
        }
    }
    println!("norms ok");
}

#[test]
fn embedding_unrotates() {
    let d = dev();
    let (v, k) = (10usize, 512usize);
    // build a "weight" row, rotate it like the quantizer does, store int8 + scale, then check lookup recovers it
    let w = rnd(v * k, 12, 1.0);
    let rot = reference::hadamard256_rows(&w, k);
    let (q, s) = reference::quant_rows_i8(&rot, k);
    let table = Tensor::from_i8(&d, &q, &[v, k]).unwrap();
    let scale = Tensor::from_f32(&d, &s, &[v]).unwrap();
    let toks: Vec<i32> = vec![3, 0, 9, 3];
    let tt = Tensor::from_buf(d.upload(&toks).unwrap(), DType::F32, &[toks.len()]);
    let out = Tensor::new(&d, DType::F32, &[toks.len(), k]).unwrap();
    ops::embed_int8(&d, &table, &scale, &tt, &out).unwrap();
    let got = out.to_f32_vec(&d).unwrap();
    for (i, t) in toks.iter().enumerate() {
        let row = &w[*t as usize * k..(*t as usize + 1) * k];
        let err = max_abs_diff(&got[i * k..(i + 1) * k], row);
        assert!(err < 0.02, "embedding err {err}");
    }
    println!("embedding ok");
}

fn cpu_attention(q: &[f32], k: &[f32], v: &[f32], nq: usize, nk: usize, hq: usize, hk: usize, d: usize, limit: &dyn Fn(usize) -> usize) -> Vec<f32> {
    let mut out = vec![0f32; nq * hq * d];
    let scale = 1.0 / (d as f32).sqrt();
    for h in 0..hq {
        let kh = h / (hq / hk);
        for i in 0..nq {
            let lim = limit(i).min(nk);
            let mut s = vec![0f32; lim];
            let mut mx = f32::NEG_INFINITY;
            for j in 0..lim {
                let mut acc = 0f32;
                for e in 0..d {
                    acc += q[(i * hq + h) * d + e] * k[(j * hk + kh) * d + e];
                }
                s[j] = acc * scale;
                mx = mx.max(s[j]);
            }
            let mut sum = 0f32;
            for j in 0..lim {
                s[j] = (s[j] - mx).exp();
                sum += s[j];
            }
            for e in 0..d {
                let mut acc = 0f32;
                for j in 0..lim {
                    acc += s[j] / sum * v[(j * hk + kh) * d + e];
                }
                out[(i * hq + h) * d + e] = acc;
            }
        }
    }
    out
}

#[test]
fn flash_attention_matches_reference() {
    let d = dev();
    ops::attn_init(&d).unwrap();
    use h3ref2va::ops::{AttnArgs, AttnView};
    // (nq, nk, hq, hk, dim, mode) mode: 0 full, 1 causal, 2 kv_limit blocks, 3 two-segment full
    for (nq, nk, hq, hk, dim, mode) in [
        (64usize, 64usize, 2usize, 2usize, 128usize, 0),
        (200, 333, 4, 2, 128, 1),
        (150, 150, 2, 2, 128, 2),
        (130, 300, 2, 1, 128, 3),
        (100, 170, 4, 4, 72, 0),
        (257, 257, 2, 2, 72, 1),
    ] {
        let q: Vec<f32> = rnd(nq * hq * dim, 21, 1.0).into_iter().map(bf16_round).collect();
        let k: Vec<f32> = rnd(nk * hk * dim, 22, 1.0).into_iter().map(bf16_round).collect();
        let v: Vec<f32> = rnd(nk * hk * dim, 23, 1.0).into_iter().map(bf16_round).collect();
        let causal_off = nk as i64 - nq as i64;
        let limits: Vec<i32> = (0..nq).map(|i| if i < 50 { i as i32 + 1 } else { 150 }).collect();
        let want = match mode {
            1 => cpu_attention(&q, &k, &v, nq, nk, hq, hk, dim, &|i| (i as i64 + 1 + causal_off) as usize),
            2 => cpu_attention(&q, &k, &v, nq, nk, hq, hk, dim, &|i| limits[i] as usize),
            _ => cpu_attention(&q, &k, &v, nq, nk, hq, hk, dim, &|_| nk),
        };
        let to_bf = |x: &[f32]| x.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>();
        let qt = Tensor::from_bf16(&d, &to_bf(&q), &[nq, hq, dim]).unwrap();
        let kt = Tensor::from_bf16(&d, &to_bf(&k), &[nk, hk, dim]).unwrap();
        let vt = Tensor::from_bf16(&d, &to_bf(&v), &[nk, hk, dim]).unwrap();
        let ot = Tensor::new(&d, DType::BF16, &[nq, hq, dim]).unwrap();
        let limt = Tensor::from_buf(d.upload(&limits).unwrap(), DType::F32, &[nq]);
        let (k1, v1, k2, v2) = if mode == 3 {
            let split = 100;
            let k1 = AttnView { ptr: kt.ptr, tok_stride: hk * dim, head_stride: dim, len: split };
            let v1 = AttnView { ptr: vt.ptr, tok_stride: hk * dim, head_stride: dim, len: split };
            let k2 = AttnView { ptr: kt.rows(split, nk - split).ptr, tok_stride: hk * dim, head_stride: dim, len: nk - split };
            let v2 = AttnView { ptr: vt.rows(split, nk - split).ptr, tok_stride: hk * dim, head_stride: dim, len: nk - split };
            (k1, v1, k2, v2)
        } else {
            (AttnView::contiguous(&kt, hk, dim), AttnView::contiguous(&vt, hk, dim), AttnView::empty(), AttnView::empty())
        };
        let args = AttnArgs {
            q: AttnView::contiguous(&qt, hq, dim),
            k1, v1, k2, v2,
            out: AttnView::contiguous(&ot, hq, dim),
            hq, hk, d: dim,
            kv_limit: if mode == 2 { Some(&limt) } else { None },
            causal: mode == 1,
            causal_off,
        };
        ops::flash_attn(&d, &args).unwrap();
        let got = ot.to_f32_vec(&d).unwrap();
        let err = max_abs_diff(&got, &want);
        assert!(err < 2e-2, "attention mismatch nq={nq} nk={nk} hq={hq} hk={hk} d={dim} mode={mode}: err={err}");
        println!("attention nq={nq} nk={nk} hq={hq} hk={hk} d={dim} mode={mode} ok err={err:.4}");
    }
}

#[test]
fn sage_attention_close_to_reference() {
    let d = dev();
    ops::attn_init(&d).unwrap();
    ops::sage_init(&d).unwrap();
    use h3ref2va::ops::AttnView;
    // two segments: prefix of 150 keys (padded to 192) + 300 target keys; 2 heads; queries 200 with full attention
    for (nq, n1, n2, hq, mode) in [(200usize, 150usize, 300usize, 2usize, 0), (130, 70, 0, 3, 1), (257, 100, 500, 4, 0)] {
        let dim = 128;
        let nk = n1 + n2;
        // smooth-ish data with channel offsets to exercise the mean subtraction
        let mut q: Vec<f32> = rnd(nq * hq * dim, 31, 1.0);
        let mut k: Vec<f32> = rnd(nk * hq * dim, 32, 1.0);
        let mut v: Vec<f32> = rnd(nk * hq * dim, 33, 1.0);
        for i in 0..nk * hq * dim {
            k[i] += ((i % dim) as f32 / 16.0).sin() * 3.0;
            v[i] += ((i % dim) as f32 / 9.0).cos() * 2.0;
        }
        for x in q.iter_mut() { *x = bf16_round(*x); }
        for x in k.iter_mut() { *x = bf16_round(*x); }
        for x in v.iter_mut() { *x = bf16_round(*x); }
        let limits: Vec<i32> = (0..nq).map(|i| if mode == 1 { (i as i32 + 1).min(nk as i32) } else { nk as i32 }).collect();
        let want = cpu_attention(&q, &k, &v, nq, nk, hq, hq, dim, &|i| limits[i] as usize);
        let to_bf = |x: &[f32]| x.iter().map(|v| bf16_bits(*v)).collect::<Vec<_>>();
        let qt = Tensor::from_bf16(&d, &to_bf(&q), &[nq, hq, dim]).unwrap();
        let kt = Tensor::from_bf16(&d, &to_bf(&k), &[nk, hq, dim]).unwrap();
        let vt = Tensor::from_bf16(&d, &to_bf(&v), &[nk, hq, dim]).unwrap();
        let seg1 = ops::quantize_kv(&d, kt.ptr, vt.ptr, hq * dim, n1, hq).unwrap();
        let seg2 = if n2 > 0 { Some(ops::quantize_kv(&d, kt.rows(n1, n2).ptr, vt.rows(n1, n2).ptr, hq * dim, n2, hq).unwrap()) } else { None };
        let qq = ops::quantize_q(&d, qt.ptr, hq * dim, nq, hq, &seg1, seg2.as_ref()).unwrap();
        let ot = Tensor::new(&d, DType::BF16, &[nq, hq, dim]).unwrap();
        let limt = Tensor::from_buf(d.upload(&limits).unwrap(), DType::F32, &[nq]);
        ops::sage_attn(&d, &qq, nq, hq, &seg1, seg2.as_ref(), AttnView::contiguous(&ot, hq, dim), if mode == 1 { Some(&limt) } else { None }).unwrap();
        let got = ot.to_f32_vec(&d).unwrap();
        let mut num = 0f64; let mut na = 0f64; let mut nb = 0f64;
        for (a, b) in got.iter().zip(&want) { num += (*a as f64) * (*b as f64); na += (*a as f64).powi(2); nb += (*b as f64).powi(2); }
        let cos = num / (na.sqrt() * nb.sqrt());
        let err = max_abs_diff(&got, &want);
        let rms = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();
        println!("sage nq={nq} n1={n1} n2={n2} h={hq} mode={mode}: cosine {cos:.5} max-abs {err:.4} (ref rms {rms:.3})");
        assert!(cos > 0.995, "sage attention too far from reference: cosine {cos}");
        assert!(!got.iter().any(|v| v.is_nan()), "NaN in output");
    }
}
