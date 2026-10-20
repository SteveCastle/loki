//! VSA (FastH3 sparse attention) vs a CPU port of ComfyUI's eager reference (comfy_kitchen backends/eager/sol_attn.py with
//! tail=False, top-k selection, block_len and the gated coarse branch) on random data. Needs a CUDA GPU.
use loki_reshoot::cuda::{Arg, Device};
use loki_reshoot::tensor::{bf16_bits, bf16_to_f32, DType, Tensor};
use loki_reshoot::vsa::{Plan, VsaRun, TILE};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

const H: usize = 2;
const D: usize = 128;
const INNER: usize = H * D;
const ATTN_V6_SMEM: u32 = (256 * 128 + 3 * (64 * 144 + 128 * 80 + 64 * 4)) as u32;

#[repr(C)]
#[derive(Clone, Copy)]
struct Attn8Raw {
    q: u64,
    sq: u64,
    k: u64,
    sk: u64,
    vt: u64,
    s_pad: i64,
    sv: u64,
    mv: u64,
    o: u64,
    o_ts: i64,
    nq: i32,
    nk: i32,
    h: i32,
    scale_log2: f32,
}
unsafe impl cudarc::driver::DeviceRepr for Attn8Raw {}

fn quant_rows(x: &[f32], rows: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * INNER];
    let mut s = vec![0f32; rows * H];
    for r in 0..rows {
        for h in 0..H {
            let o = r * INNER + h * D;
            let amax = x[o..o + D].iter().fold(0f32, |a, v| a.max(v.abs()));
            let sc = amax / 127.0;
            s[r * H + h] = sc;
            for d in 0..D {
                q[o + d] = if sc > 0.0 { (x[o + d] / sc).round().clamp(-128.0, 127.0) as i8 } else { 0 };
            }
        }
    }
    (q, s)
}

fn bytes<T: Copy>(v: &[T]) -> Vec<u8> {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)).to_vec() }
}

#[test]
fn vsa_matches_reference() -> anyhow::Result<()> {
    let dev = match Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no CUDA device ({e}), skipping");
            return Ok(());
        }
    };
    // text 100 rows + target audio 40 rows (sinks), video 12 x 12 x 16 patches (36 cubes, the last t-row of cubes full)
    let (lt, h2, w2) = (12, 12, 16);
    let plan = Plan::new(&[100, 40], lt, h2, w2);
    let seq = plan.src.len();
    let s_pad = seq.div_ceil(256) * 256;
    let tiles = plan.tiles();
    let mut rng = rand::rngs::StdRng::seed_from_u64(7);
    let mut randn = |n: usize, s: f32| -> Vec<f32> { (0..n).map(|_| { let z: f32 = StandardNormal.sample(&mut rng); s * z }).collect::<Vec<f32>>() };
    // pad rows get large keys: any leak into the softmax would show
    let live: Vec<bool> = plan.src.iter().map(|&s| s >= 0).collect();
    let mut q = randn(s_pad * INNER, 1.0);
    let mut k = randn(s_pad * INNER, 1.0);
    let v: Vec<f32> = randn(s_pad * INNER, 1.0).iter().map(|&x| bf16_to_f32(bf16_bits(x))).collect();
    let gate: Vec<f32> = randn(s_pad * INNER, 0.5).iter().map(|&x| bf16_to_f32(bf16_bits(x))).collect();
    let mv = randn(INNER, 0.2);
    // structure so the top-k is not arbitrary: queries and keys of a cube share a random direction per cube
    for t in 0..tiles {
        let dir = randn(INNER, 1.5);
        for r in t * TILE..(t + 1) * TILE {
            for c in 0..INNER {
                q[r * INNER + c] += dir[c];
                k[r * INNER + c] += dir[c];
            }
        }
    }
    for r in 0..seq {
        if !live[r] {
            for c in 0..INNER {
                k[r * INNER + c] *= 20.0;
            }
        }
    }
    let (q8, sq) = quant_rows(&q, s_pad);
    let (k8, sk) = quant_rows(&k, s_pad);
    let qd: Vec<f32> = (0..s_pad * INNER).map(|i| q8[i] as f32 * sq[(i / INNER) * H + (i % INNER) / D]).collect();
    let kd: Vec<f32> = (0..s_pad * INNER).map(|i| k8[i] as f32 * sk[(i / INNER) * H + (i % INNER) / D]).collect();

    // ---- device
    let up = |b: Vec<u8>, dt: DType, n: usize| -> anyhow::Result<Tensor> { Ok(Tensor::from_buf(dev.upload(&b)?, dt, &[n])) };
    let q8_t = up(bytes(&q8), DType::I8, q8.len())?;
    let sq_t = Tensor::from_f32(&dev, &sq, &[sq.len()])?;
    let k8_t = up(bytes(&k8), DType::I8, k8.len())?;
    let sk_t = Tensor::from_f32(&dev, &sk, &[sk.len()])?;
    let v16: Vec<u16> = v.iter().map(|&x| bf16_bits(x)).collect();
    let v_t = Tensor::from_bf16(&dev, &v16, &[v16.len()])?;
    let mv_t = Tensor::from_f32(&dev, &mv, &[INNER])?;
    let vt = Tensor::zeros(&dev, DType::U8, &[H * D * s_pad])?;
    let sv = Tensor::zeros(&dev, DType::F32, &[(s_pad / 256) * H])?;
    dev.launch(
        "k_h3_quant_v8",
        (seq.div_ceil(256) as u32, H as u32, 1),
        (256, 1, 1),
        64 * 129 * 4,
        &[Arg::Ptr(v_t.ptr), Arg::I64(INNER as i64), Arg::I32(seq as i32), Arg::I32(H as i32), Arg::Ptr(mv_t.ptr), Arg::Ptr(vt.ptr), Arg::I64(s_pad as i64), Arg::I32(0), Arg::Ptr(sv.ptr)],
    )?;
    dev.set_max_smem("k_h3_attn_v6_sparse", ATTN_V6_SMEM)?;
    dev.set_max_smem("k_vsa_select", 88 * 1024)?;
    let vsa = VsaRun::new(&dev, &plan, 140, lt * h2 * w2, s_pad, H, f32::INFINITY)?;
    let g16: Vec<u16> = gate.iter().map(|&x| bf16_bits(x)).collect();
    dev.htod_at(vsa.gate_rows(0, INNER), &bytes(&g16[..seq * INNER]))?;
    let scale = 1.0 / (D as f32).sqrt();
    vsa.v_means(&dev, v_t.ptr, INNER, 0, seq, H)?;
    vsa.k_means(&dev, k8_t.ptr, sk_t.ptr, seq, H)?;
    vsa.select(&dev, q8_t.ptr, sq_t.ptr, 0, seq, H, scale)?;
    let out = Tensor::zeros(&dev, DType::BF16, &[seq * INNER])?;
    let p = Attn8Raw {
        q: q8_t.ptr,
        sq: sq_t.ptr,
        k: k8_t.ptr,
        sk: sk_t.ptr,
        vt: vt.ptr,
        s_pad: s_pad as i64,
        sv: sv.ptr,
        mv: mv_t.ptr,
        o: out.ptr,
        o_ts: INNER as i64,
        nq: seq as i32,
        nk: seq as i32,
        h: H as i32,
        scale_log2: scale * std::f32::consts::LOG2_E,
    };
    vsa.attention(&dev, &p, seq, H, ATTN_V6_SMEM)?;
    vsa.add_coarse(&dev, out.ptr, seq, INNER)?;
    dev.sync()?;
    let got = out.to_f32_vec(&dev)?;
    let (list, cnt) = vsa.lists(&dev)?;

    // ---- CPU reference
    let mean = |x: &[f32], t: usize, h: usize| -> Vec<f64> {
        let mut m = vec![0f64; D];
        let len = plan.block_len[t] as usize;
        for r in t * TILE..t * TILE + len {
            for d in 0..D {
                m[d] += x[r * INNER + h * D + d] as f64;
            }
        }
        m.iter().map(|s| s / len as f64).collect()
    };
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let nsink = plan.sink_tiles;
    let kk = vsa.kk;
    assert_eq!(kk, 4); // round(0.1 * 36)
    let (mut sel_diff, mut sel_total) = (0usize, 0usize);
    let (mut worst_cos, mut sum_cos, mut n_cos) = (1f64, 0f64, 0usize);
    for h in 0..H {
        let km: Vec<Vec<f64>> = (0..tiles).map(|t| mean(&kd, t, h)).collect();
        let vm: Vec<Vec<f64>> = (0..tiles).map(|t| mean(&v, t, h)).collect();
        for qt in 0..tiles {
            let qm = mean(&qd, qt, h);
            let sc: Vec<f64> = km.iter().map(|kv| dot(&qm, kv)).collect();
            // selection (reference)
            let mut want = vec![false; tiles];
            if qt < nsink {
                want.iter_mut().for_each(|w| *w = true);
            } else {
                let mut ranked: Vec<f64> = sc[nsink..].to_vec();
                ranked.sort_by(|a, b| b.total_cmp(a));
                let thr = ranked[kk - 1];
                for j in 0..tiles {
                    want[j] = j < nsink || (j as i64 - qt as i64).abs() <= 1 || (j >= nsink && sc[j] >= thr);
                }
            }
            // selection (GPU): block qt/4, mask bit qt%4
            let (b, bit) = (qt / 4, qt % 4);
            let n = cnt[b * H + h] as usize;
            let mut got_sel = vec![false; tiles];
            let entries = &list[(b * H + h) * tiles..(b * H + h) * tiles + n];
            assert!(entries.windows(2).all(|w| (w[0] & 0xFFFFFF) < (w[1] & 0xFFFFFF)), "tile list not ascending");
            for &e in entries {
                if (e >> (24 + bit)) & 1 == 1 {
                    got_sel[(e & 0xFFFFFF) as usize] = true;
                }
            }
            sel_total += tiles;
            sel_diff += want.iter().zip(&got_sel).filter(|(a, b)| a != b).count();
            // coarse
            let mx = sc.iter().fold(f64::MIN, |a, &s| a.max(s * scale as f64));
            let e: Vec<f64> = sc.iter().map(|&s| (s * scale as f64 - mx).exp()).collect();
            let es: f64 = e.iter().sum();
            let oc: Vec<f64> = (0..D).map(|d| (0..tiles).map(|j| e[j] * vm[j][d]).sum::<f64>() / es).collect();
            // fine (with the GPU's selection, to test the kernel arithmetic on its own) + gated coarse, per live query row
            for r in qt * TILE..qt * TILE + plan.block_len[qt] as usize {
                let qr: Vec<f64> = (0..D).map(|d| qd[r * INNER + h * D + d] as f64).collect();
                let mut logits: Vec<(usize, f64)> = Vec::new();
                for j in (0..tiles).filter(|&j| got_sel[j]) {
                    for key in j * TILE..j * TILE + plan.block_len[j] as usize {
                        let kr: Vec<f64> = (0..D).map(|d| kd[key * INNER + h * D + d] as f64).collect();
                        logits.push((key, dot(&qr, &kr) * scale as f64));
                    }
                }
                let m = logits.iter().fold(f64::MIN, |a, &(_, l)| a.max(l));
                let den: f64 = logits.iter().map(|&(_, l)| (l - m).exp()).sum();
                let (mut num, mut nn, mut gg) = (0f64, 0f64, 0f64);
                for d in 0..D {
                    let fine: f64 = logits.iter().map(|&(key, l)| (l - m).exp() * v[key * INNER + h * D + d] as f64).sum::<f64>() / den;
                    let want_o = fine + gate[r * INNER + h * D + d] as f64 * oc[d];
                    let got_o = got[r * INNER + h * D + d] as f64;
                    num += want_o * got_o;
                    nn += want_o * want_o;
                    gg += got_o * got_o;
                }
                let cos = num / (nn.sqrt() * gg.sqrt());
                worst_cos = worst_cos.min(cos);
                sum_cos += cos;
                n_cos += 1;
            }
        }
    }
    let mean_cos = sum_cos / n_cos as f64;
    eprintln!("vsa: {tiles} tiles, selection mismatches {sel_diff}/{sel_total}, output cos mean {mean_cos:.5} worst {worst_cos:.5}");
    assert!(sel_diff * 1000 <= sel_total, "selection differs from the reference in {sel_diff}/{sel_total}");
    assert!(mean_cos > 0.999 && worst_cos > 0.99, "sparse attention output differs from the reference");
    Ok(())
}
