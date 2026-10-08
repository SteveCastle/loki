//! Qwen Image 2.1 conditioning: Qwen3-VL sees the prompt plus the reference image(s), the last
//! hidden layer is sliced to the user turn with the vision spans removed, and the positions where
//! the image latents get spliced into the DiT sequence are recorded (one slot per image).
use crate::cuda::Device;
use crate::image::Rgb8;
use crate::llm::{self, Llm};
use crate::ops;
use crate::safetensors::SafeTensors;
use crate::tensor::{DType, Tensor};
use crate::tokenizer::{self, Tokenizer};
use crate::vision::VisionTower;
use crate::weights::Loader;
use anyhow::{ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub const SYSTEM_PROMPT: &str = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";
pub const VISION_BLOCK: &str = "<|vision_start|><|image_pad|><|vision_end|>";

/// The chat template for a prompt with `n_images` references (ComfyUI's QwenImage21Tokenizer): every
/// reference is announced as `<imageN>` followed by a vision block, the blocks separated by spaces, then the prompt.
pub fn template(prompt: &str, n_images: usize) -> String {
    let refs: Vec<String> = (1..=n_images).map(|i| format!("<image{i}>{VISION_BLOCK}")).collect();
    format!("{SYSTEM_PROMPT}<|im_start|>user\n{}{prompt}<|im_end|>\n<|im_start|>assistant\n", refs.join(" "))
}

pub struct TextEncoder {
    dev: Arc<Device>,
    pub vision: VisionTower,
    pub llm: Llm,
    pub tok: Tokenizer,
}

pub struct Conditioning {
    /// [L, 4096] bf16
    pub context: Tensor,
    /// per reference image, in order: index in `context` where its latents are spliced in (ascending)
    pub slots: Vec<usize>,
}

impl TextEncoder {
    pub fn load(dev: Arc<Device>, path: &Path) -> Result<TextEncoder> {
        let st = SafeTensors::open(path)?;
        let mut l = Loader::new(&st, dev.clone());
        let t0 = std::time::Instant::now();
        let vision = VisionTower::load(&mut l).context("loading vision tower")?;
        let llm = Llm::load(&mut l).context("loading Qwen3 decoder")?;
        crate::info!("  text encoder: {:.1} GB uploaded in {:.1}s", l.uploaded as f64 / 1e9, t0.elapsed().as_secs_f64());
        Ok(TextEncoder { dev, vision, llm, tok: Tokenizer::new()? })
    }

    /// `refs` must already be resized to multiples of 32 (the same pixels the VAE encodes); one slot per image.
    pub fn encode(&self, prompt: &str, refs: &[Rgb8]) -> Result<Conditioning> {
        ensure!(!refs.is_empty(), "at least one reference image is needed");
        let dev = &self.dev;
        // ---- vision tower per image (downscaled if above the processor's max_pixels, like process_qwen2vl_images)
        let mut vis = Vec::with_capacity(refs.len());
        for r in refs {
            let (vh, vw, vpix) = vision_input(r);
            vis.push(self.vision.forward(&vpix, vh, vw).context("vision tower")?);
        }
        // ---- tokens: one <|image_pad|> per image, each expanded to that image's merged vision tokens
        let toks = self.tok.encode(&template(prompt, refs.len()));
        let pads: Vec<usize> = toks.iter().enumerate().filter(|(_, &t)| t == tokenizer::IMAGE_PAD).map(|(i, _)| i).collect();
        ensure!(pads.len() == refs.len(), "template has {} image pads for {} images", pads.len(), refs.len());
        let l = toks.len() - pads.len() + vis.iter().map(|v| v.merged.shape[0]).sum::<usize>();
        // ---- embeddings (text rows from the int8 table, vision rows copied in) and MRoPE position ids
        // (qwen2vl_mrope_position_ids: text advances all three axes by one per token; an image puts its tokens
        // at the next position on the time axis with row/column offsets on the others, then skips max(gh, gw))
        let x = Tensor::new(dev, DType::F32, &[l, llm::HIDDEN])?;
        let mut pos: [Vec<i64>; 3] = [vec![0; l], vec![0; l], vec![0; l]];
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(refs.len()); // (first row, rows) of each image
        let mut deep: Vec<(usize, &[Tensor])> = Vec::with_capacity(refs.len());
        let (mut row, mut next, mut cursor) = (0usize, 0i64, 0usize);
        for (v, &pad) in vis.iter().zip(&pads) {
            row = self.place_text(&x, &toks[cursor..pad], row, &mut next, &mut pos)?;
            let n_img = v.merged.shape[0];
            let (gh, gw) = (v.grid_h / 2, v.grid_w / 2);
            dev.dtod(x.ptr + (row * llm::HIDDEN * 4) as u64, v.merged.ptr, v.merged.bytes())?;
            for i in 0..n_img {
                pos[0][row + i] = next;
                pos[1][row + i] = next + (i / gw) as i64;
                pos[2][row + i] = next + (i % gw) as i64;
            }
            spans.push((row, n_img));
            deep.push((row, v.deepstack.as_slice()));
            next += gh.max(gw) as i64;
            row += n_img;
            cursor = pad + 1;
        }
        row = self.place_text(&x, &toks[cursor..], row, &mut next, &mut pos)?;
        ensure!(row == l);
        let rope = self.llm.rope_table(&pos)?;
        // ---- decoder (DeepStack features added to each image's rows after the first layers)
        self.llm.forward(&x, &rope, &deep).context("Qwen3 decoder")?;
        // ---- slice: drop everything before the second <|im_start|> and every vision span; a slot is the
        // number of kept rows before that image
        let im_starts: Vec<usize> = toks.iter().enumerate().filter(|(_, &t)| t == tokenizer::IM_START).map(|(i, _)| i).collect();
        ensure!(im_starts.len() >= 2, "template must contain two <|im_start|>");
        let drop_before = im_starts[1]; // the same in expanded coordinates: the system turn precedes every image
        ensure!(drop_before <= spans[0].0);
        let mut keep: Vec<i32> = Vec::with_capacity(l);
        let mut slots = Vec::with_capacity(spans.len());
        let mut i = drop_before;
        for &(start, n) in &spans {
            keep.extend((i..start).map(|j| j as i32));
            slots.push(keep.len());
            i = start + n;
        }
        keep.extend((i..l).map(|j| j as i32));
        let idx = Tensor::from_buf(dev.upload(&keep)?, DType::F32, &[keep.len()]);
        let ctx32 = Tensor::new(dev, DType::F32, &[keep.len(), llm::HIDDEN])?;
        ops::gather_rows_f32(dev, &x, &idx, &ctx32)?;
        let context = Tensor::new(dev, DType::BF16, &[keep.len(), llm::HIDDEN])?;
        ops::to_bf16(dev, &ctx32, &context)?;
        Ok(Conditioning { context, slots })
    }

    /// Embed a run of text tokens into rows `row..` of `x`, with consecutive positions on all three axes.
    fn place_text(&self, x: &Tensor, toks: &[u32], row: usize, next: &mut i64, pos: &mut [Vec<i64>; 3]) -> Result<usize> {
        if toks.is_empty() {
            return Ok(row);
        }
        let ids: Vec<i32> = toks.iter().map(|&t| t as i32).collect();
        let e = self.llm.embed(&ids)?;
        self.dev.dtod(x.ptr + (row * llm::HIDDEN * 4) as u64, e.ptr, e.bytes())?;
        for i in 0..ids.len() {
            for axis in pos.iter_mut() {
                axis[row + i] = *next + i as i64;
            }
        }
        *next += ids.len() as i64;
        Ok(row + ids.len())
    }
}

/// Vision-tower input: the reference image at its (multiple-of-32) size, bilinearly downscaled when it
/// exceeds Qwen's max_pixels (12845056), mirroring process_qwen2vl_images.
fn vision_input(img: &Rgb8) -> (usize, usize, Vec<f32>) {
    const MAX_PIXELS: usize = 12_845_056;
    const FACTOR: usize = 32;
    let (h, w) = (img.h, img.w);
    let (mut hb, mut wb) = (h, w); // already multiples of 32
    if hb * wb > MAX_PIXELS {
        let beta = ((h * w) as f64 / MAX_PIXELS as f64).sqrt();
        hb = FACTOR.max(((h as f64 / beta / FACTOR as f64).floor() as usize) * FACTOR);
        wb = FACTOR.max(((w as f64 / beta / FACTOR as f64).floor() as usize) * FACTOR);
    }
    let src = img.to_f32();
    if hb == h && wb == w {
        return (h, w, src);
    }
    // bilinear, align_corners=False
    let mut out = vec![0f32; hb * wb * 3];
    let sy = h as f32 / hb as f32;
    let sx = w as f32 / wb as f32;
    for oy in 0..hb {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).max(0.0);
        let y0 = (fy.floor() as usize).min(h - 1);
        let y1 = (y0 + 1).min(h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..wb {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).max(0.0);
            let x0 = (fx.floor() as usize).min(w - 1);
            let x1 = (x0 + 1).min(w - 1);
            let wx = fx - x0 as f32;
            for c in 0..3 {
                let p = |yy: usize, xx: usize| src[(yy * w + xx) * 3 + c];
                let v = p(y0, x0) * (1.0 - wy) * (1.0 - wx) + p(y0, x1) * (1.0 - wy) * wx + p(y1, x0) * wy * (1.0 - wx) + p(y1, x1) * wy * wx;
                out[(oy * wb + ox) * 3 + c] = v;
            }
        }
    }
    (hb, wb, out)
}
