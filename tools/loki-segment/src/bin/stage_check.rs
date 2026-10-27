//! Compare the native engine stage by stage against a reference dump (ref/ref_image.py).
//!     stage_check REF_DIR [MODEL]
use anyhow::Result;
use loki_segment::cuda::Device;
use loki_segment::dump::{compare, Dump};
use loki_segment::tensor::Tensor;
use loki_segment::vit::{Neck, Vit};
use loki_segment::weights::Weights;
use std::path::PathBuf;
use std::time::Instant;

fn report(name: &str, ours: &[f32], refd: &Dump) -> Result<()> {
    let r = refd.read(name)?;
    if r.len() != ours.len() {
        println!("{name:28} SIZE MISMATCH ours {} ref {}", ours.len(), r.len());
        return Ok(());
    }
    let (cos, rel, mx) = compare(&r, ours);
    let amax = r.iter().fold(0f32, |a, v| a.max(v.abs()));
    println!("{name:28} cos {cos:.6} rel {rel:.2e} maxabs {mx:.4} (|ref|max {amax:.4})");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let refdir = PathBuf::from(&args[1]);
    let model = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "models/sam3.1_multiplex_fp16.safetensors".into()));
    let refd = Dump::open(&refdir)?;
    let dev = Device::new(0)?;
    loki_segment::ops::init(&dev)?;
    loki_segment::ops::attn_init(&dev)?;
    let w = Weights::open(&model, dev.clone())?;
    let t0 = Instant::now();
    let vit = Vit::load(&w, "detector.backbone.vision_backbone.trunk")?;
    let necks = [
        ("sam3", Neck::load(&w, "detector.backbone.vision_backbone.convs")?),
        ("interactive", Neck::load(&w, "detector.backbone.vision_backbone.interactive_convs")?),
        ("propagation", Neck::load(&w, "detector.backbone.vision_backbone.propagation_convs")?),
    ];
    dev.sync()?;
    println!("loaded in {:.2}s ({} MB)", t0.elapsed().as_secs_f64(), w.uploaded.get() >> 20);
    if let Some(ipath) = args.get(3) {
        let m = loki_segment::image::Rgb8::load(std::path::Path::new(ipath))?;
        let px = loki_segment::image::resize_bilinear_aa(&m, 1008, 1008);
        let mut chw = vec![0f32; 3 * 1008 * 1008];
        for i in 0..1008 * 1008 {
            for c in 0..3 {
                chw[c * 1008 * 1008 + i] = (px[i * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
            }
        }
        report("image", &chw, &refd)?;
        let r = refd.read("image")?;
        let ndiff = r.iter().zip(&chw).filter(|(a, b)| (*a - *b).abs() > 1e-3).count();
        println!("image: {ndiff} of {} values differ by more than one 8-bit step", r.len());
    }
    let img = refd.read("image")?;
    let img = Tensor::from_f32(&dev, &img, &[3, 1008, 1008])?;
    for _ in 0..2 {
        let t = Instant::now();
        let tr = vit.forward(&img)?;
        dev.sync()?;
        println!("trunk {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        let _ = tr;
    }
    let trunk = vit.forward(&img)?;
    report("trunk", &trunk.to_f32_vec(&dev)?, &refd)?;
    for (name, neck) in &necks {
        let t = Instant::now();
        let outs = neck.forward(&dev, &trunk)?;
        dev.sync()?;
        println!("neck {name} {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        for (i, o) in outs.iter().enumerate() {
            report(&format!("neck_{name}_{i}"), &o.to_f32_vec(&dev)?, &refd)?;
        }
    }
    
    // ---- text + detector
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(refdir.join("meta.json"))?)?;
    let text = loki_segment::text::TextEncoder::load(&w, "detector.backbone.language_backbone")?;
    let det = loki_segment::detector::Detector::load(&w, "detector")?;
    let fpn = necks[0].1.forward(&dev, &trunk)?;
    let t = Instant::now();
    let dimg = det.prepare(&fpn)?;
    dev.sync()?;
    println!("detector prepare {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    for (ti, prompt) in meta["text"].as_array().unwrap().iter().enumerate() {
        let prompt = prompt.as_str().unwrap();
        let reftok: Vec<u32> = meta[format!("tokens_{ti}")].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let t = Instant::now();
        let tf = text.encode(prompt)?;
        dev.sync()?;
        println!("-- prompt {prompt:?}: text {:.1} ms, tokens match: {}", t.elapsed().as_secs_f64() * 1e3, tf.tokens == reftok);
        report(&format!("text{ti}_features"), &tf.feats.to_f32_vec(&dev)?, &refd)?;
        report(&format!("text{ti}_geo"), &dimg.geo.to_f32_vec(&dev)?, &refd)?;
        let t = Instant::now();
        let d = det.detect(&dimg, &tf)?;
        dev.sync()?;
        let t_det = t.elapsed().as_secs_f64() * 1e3;
        let all: Vec<usize> = (0..200).collect();
        let t = Instant::now();
        let m = det.masks(&d, &all)?;
        dev.sync()?;
        println!("detect {t_det:.1} ms, masks(200) {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        report(&format!("text{ti}_memory"), &d.memory.to_f32_vec(&dev)?, &refd)?;
        report(&format!("text{ti}_hs"), &d.hs.to_f32_vec(&dev)?, &refd)?;
        let pres = refd.read(&format!("text{ti}_presence"))?;
        println!("{:28} ours {:.4} ref {:.4}", format!("text{ti}_presence(last)"), d.presence_logit, pres[pres.len() - 1]);
        report(&format!("text{ti}_logits"), &d.logits, &refd)?;
        let bflat: Vec<f32> = d.boxes.iter().flat_map(|b| b.to_vec()).collect();
        report(&format!("text{ti}_boxes"), &bflat, &refd)?;
        report(&format!("text{ti}_masks"), &m.to_f32_vec(&dev)?, &refd)?;
        report(&format!("text{ti}_semantic"), &det.semantic(&d)?, &refd)?;
        let rl = refd.read(&format!("text{ti}_logits"))?;
        let kept_ref: Vec<(usize, f32)> = rl.iter().enumerate().filter(|(_, l)| **l > 0.0).map(|(i, l)| (i, 1.0 / (1.0 + (-l).exp()))).collect();
        let kept: Vec<(usize, f32)> = d.probs.iter().enumerate().filter(|(_, p)| **p > 0.5).map(|(i, p)| (i, *p)).collect();
        println!("kept (>0.5) ours {kept:?}
            ref  {kept_ref:?}");
    }
    
    // ---- interactive head
    let pts_meta = meta["points"].as_array().cloned().unwrap_or_default();
    if !pts_meta.is_empty() || !meta["box"].is_null() {
        use loki_segment::interactive::{Interactive, Point};
        let (iw, ih) = (meta["width"].as_f64().unwrap() as f32, meta["height"].as_f64().unwrap() as f32);
        let inter = Interactive::load(&w, "tracker.model")?;
        let ifpn = necks[1].1.forward(&dev, &trunk)?;
        let simg = inter.prepare(&ifpn)?;
        report("sam_s0", &simg.s0.to_f32_vec(&dev)?, &refd)?;
        report("sam_s1", &simg.s1.to_f32_vec(&dev)?, &refd)?;
        let mut pts = Vec::new();
        for p in &pts_meta {
            let v: Vec<f32> = p.as_str().unwrap().split(',').map(|t| t.parse().unwrap()).collect();
            pts.push(Point { x: v[0] / iw * 1008.0, y: v[1] / ih * 1008.0, label: if v.len() > 2 { v[2] as i32 } else { 1 } });
        }
        if let Some(b) = meta["box"].as_str() {
            let v: Vec<f32> = b.split(',').map(|t| t.parse().unwrap()).collect();
            pts.push(Point { x: v[0] / iw * 1008.0, y: v[1] / ih * 1008.0, label: 2 });
            pts.push(Point { x: (v[0] + v[2]) / iw * 1008.0, y: (v[1] + v[3]) / ih * 1008.0, label: 3 });
        }
        let multimask = meta["sam_multimask"].as_bool().unwrap_or(false);
        report("sam_sparse", &inter.embed_points(&pts), &refd)?;
        let t = Instant::now();
        let o = inter.decode(&simg, &pts, multimask)?;
        dev.sync()?;
        println!("sam decode {:.1} ms (multimask {multimask})", t.elapsed().as_secs_f64() * 1e3);
        let all = o.all.to_f32_vec(&dev)?;
        let p = 288 * 288;
        let (lo, ious) = if multimask { (all[p..].to_vec(), o.ious[1..].to_vec()) } else { (o.mask.to_f32_vec(&dev)?, vec![o.iou]) };
        report("sam_lowres", &lo, &refd)?;
        report("sam_ious", &ious, &refd)?;
        println!("{:28} ours {:.4} ref {:?}", "sam_obj", o.obj_score, refd.read("sam_obj")?);
    }
    Ok(())
}
