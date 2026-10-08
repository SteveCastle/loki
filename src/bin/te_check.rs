//! Compare the Rust text encoder against a ComfyUI reference dump (ref/ref_te.py).
//! Usage: te_check <ref_dir> [model_path]
use anyhow::{Context, Result};
use loki_retouch::cuda::Device;
use loki_retouch::image::Rgb8;
use loki_retouch::ops;
use loki_retouch::text_encoder::{template, TextEncoder};
use loki_retouch::tokenizer::Tokenizer;
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let ref_dir = Path::new(&args[1]);
    let model = args.get(2).cloned().unwrap_or_else(|| format!("{}/dev/loki-retouch/models/qwen3vl_8b_int8_convrot.safetensors", std::env::var("USERPROFILE").unwrap()));
    let prompt = std::fs::read_to_string(Path::new(file!()).parent().unwrap().join("../../ref/prompt.txt")).context("prompt.txt")?;
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(ref_dir.join("meta.json"))?)?;
    let shape = meta["shape"].as_array().unwrap();
    let (l_ref, d_ref) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
    let slot_ref = meta["image_slots"][0].as_u64().unwrap() as usize;

    // tokenizer check
    let tok = Tokenizer::new()?;
    let text = template(&prompt, 1);
    let mine = tok.encode(&text);
    let ref_toks: Vec<serde_json::Value> = serde_json::from_str(&std::fs::read_to_string(ref_dir.join("tokens.json"))?)?;
    let ref_ids: Vec<i64> = ref_toks.iter().map(|v| v.as_i64().unwrap_or(151655)).collect();
    let mine_i64: Vec<i64> = mine.iter().map(|&v| v as i64).collect();
    if mine_i64 != ref_ids {
        println!("TOKEN MISMATCH: mine {} vs ref {}", mine.len(), ref_ids.len());
        for (i, (a, b)) in mine_i64.iter().zip(ref_ids.iter()).enumerate() {
            if a != b {
                println!("  first diff at {i}: mine {a} ref {b}");
                break;
            }
        }
        if mine_i64.len() != ref_ids.len() {
            println!("  mine tail {:?}", &mine_i64[mine_i64.len().saturating_sub(5)..]);
            println!("  ref tail {:?}", &ref_ids[ref_ids.len().saturating_sub(5)..]);
        }
    } else {
        println!("tokens match ({} tokens)", mine.len());
    }

    let dev = Device::new(0)?;
    ops::gemm_init(&dev)?;
    ops::attn_init(&dev)?;
    let te = TextEncoder::load(dev.clone(), Path::new(&model))?;
    let img = Rgb8::load(&ref_dir.join("resized.png"))?;
    let t0 = std::time::Instant::now();
    let cond = te.encode(&prompt, &[img])?;
    dev.sync()?;
    println!("encode took {:.2}s; context {:?} slot {}", t0.elapsed().as_secs_f64(), cond.context.shape, cond.slots[0]);
    println!("reference: [{l_ref}, {d_ref}] slot {slot_ref}");
    let got = cond.context.to_f32_vec(&dev)?;
    let bytes = std::fs::read(ref_dir.join("context.bin"))?;
    let want: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(want.len(), l_ref * d_ref);
    let l = cond.context.shape[0].min(l_ref);
    let mut num = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut worst_row = (0usize, 0f64);
    for r in 0..l {
        let mut rn = 0f64;
        let mut ra = 0f64;
        let mut rb = 0f64;
        for c in 0..d_ref {
            let a = got[r * d_ref + c] as f64;
            let b = want[r * d_ref + c] as f64;
            rn += a * b;
            ra += a * a;
            rb += b * b;
        }
        let cos = rn / (ra.sqrt() * rb.sqrt() + 1e-12);
        if 1.0 - cos > worst_row.1 {
            worst_row = (r, 1.0 - cos);
        }
        num += rn;
        na += ra;
        nb += rb;
    }
    println!("overall cosine similarity: {:.6}", num / (na.sqrt() * nb.sqrt()));
    println!("worst row {} (1-cos = {:.4})", worst_row.0, worst_row.1);
    println!("rel rms error: {:.4}", (got[..l * d_ref].iter().zip(&want[..l * d_ref]).map(|(a, b)| ((a - b) * (a - b)) as f64).sum::<f64>() / nb).sqrt());
    Ok(())
}
