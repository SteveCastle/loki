//! Validate the native tracker against ref/ref_video.py: same preprocessed frames (the reference's own), same
//! clicks; compares the per-frame low-res mask logits and reports per-frame mask IoU.
//!     video_check REF_DIR FRAMES_DIR [MODEL]
use anyhow::Result;
use loki_segment::dump::{compare, Dump};
use loki_segment::image::Rgb8;
use loki_segment::model::{Model, ObjPrompt};
use loki_segment::video::{TrackOpts, VideoSession};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let refd = Dump::open(Path::new(&args[1]))?;
    let fdir = PathBuf::from(&args[2]);
    let model_path = PathBuf::from(args.get(3).cloned().unwrap_or_else(|| "models/sam3.1_multiplex_fp16.safetensors".into()));
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(Path::new(&args[1]).join("meta.json"))?)?;
    let n = meta["frames"].as_u64().unwrap() as usize;
    let mut frames = Vec::new();
    let mut names: Vec<_> = std::fs::read_dir(&fdir)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    names.sort();
    for p in names.iter().take(n) {
        frames.push(Rgb8::load(p)?);
    }
    let dev = loki_segment::cuda::Device::new(0)?;
    let model = Model::load(&model_path, dev.clone())?;
    let mut sess = VideoSession::from_frames(fdir.to_str().unwrap(), frames, 30.0);
    let imgs = refd.read("images")?;
    let per = 3 * 1008 * 1008;
    sess.parity = std::env::var("PARITY").is_ok();
    sess.debug = Some(std::cell::RefCell::new(Dump::create(Path::new("native_out/vdbg"))?));
    sess.ref_inputs = Some((0..n).map(|i| imgs[i * per..(i + 1) * per].to_vec()).collect());
    let mut prompts = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    for (oid, o) in meta["objects"].as_object().unwrap() {
        let pts: Vec<Vec<f32>> = o["points"].as_array().unwrap().iter().map(|p| p.as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect()).collect();
        ids.push(oid.parse()?);
        prompts.push(ObjPrompt { obj: Some(oid.parse()?), frame: Some(0), points: pts, ..Default::default() });
    }
    let t = Instant::now();
    let tr = sess.compute(&model, &prompts, 0.5, &TrackOpts { looped: false, bidir: false, smooth: 0, sticky: std::env::var("STICKY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0) }, false)?;
    dev.sync()?;
    println!("tracked {} frames x {} objects in {:.2} s", n, tr.objs.len(), t.elapsed().as_secs_f64());
    let p = 288 * 288;
    let mut nd = Dump::create(Path::new("native_out/vcheck"))?;
    for (f, l) in &tr.logits {
        let flat: Vec<f32> = l.iter().flat_map(|x| x.iter().copied()).collect();
        nd.write(&format!("f{f}_out_lowres"), &[l.len(), 288, 288], &flat)?;
    }
    let (mut worst_cos, mut miou_sum, mut cnt) = (1f64, 0f64, 0);
    for f in 0..n {
        let name = format!("f{f}_out_lowres");
        let r = refd.read(&name)?;
        let ours: Vec<f32> = tr.logits[&f].iter().flat_map(|l| l.iter().copied()).collect();
        let (cos, rel, _) = compare(&r, &ours);
        let mut ious = Vec::new();
        for o in 0..tr.objs.len() {
            let (a, b) = (&r[o * p..(o + 1) * p], &ours[o * p..(o + 1) * p]);
            let (mut i, mut u) = (0, 0);
            for (x, y) in a.iter().zip(b) {
                let (x, y) = (*x > 0.0, *y > 0.0);
                i += (x && y) as usize;
                u += (x || y) as usize;
            }
            let iou = if u == 0 { 1.0 } else { i as f64 / u as f64 };
            miou_sum += iou;
            cnt += 1;
            ious.push(format!("{iou:.4}"));
        }
        worst_cos = worst_cos.min(cos);
        println!("frame {f:3}: logits cos {cos:.6} rel {rel:.2e}  mask IoU vs ref {}", ious.join(" "));
    }
    println!("worst cos {worst_cos:.6}, mean IoU {:.4}", miou_sum / cnt as f64);
    let _ = ids;
    Ok(())
}
