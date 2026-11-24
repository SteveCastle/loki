//! Preprocessing into the run folder's `cache/`, one model at a time: the Qwen3.5 text encoder (every caption, the
//! empty prompt for caption dropout, the sample prompts) and the FLUX.2 VAE (every image at its training size, and
//! mirrored with `--flip`). Training then only reads these files, so the 19 GB text encoder never shares the GPU
//! with the DiT. Entries are keyed by content (text / file identity + size), so a resumed or re-run training reuses
//! them and a changed caption or image is encoded again.
use crate::data::{self, Item};
use crate::safetensors_out::{self, Out};
use anyhow::{Context, Result};
use loki_dream::cuda::Device;
use loki_dream::safetensors::SafeTensors;
use loki_dream::tensor::{DType, Tensor};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// FNV-1a 64, hex: cache keys.
pub fn key(parts: &[&str]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in parts {
        for b in p.bytes().chain(std::iter::once(0u8)) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    format!("{h:016x}")
}

pub struct Cache {
    pub dir: PathBuf,
}

impl Cache {
    pub fn new(run: &Path) -> Result<Cache> {
        let dir = run.join("cache");
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        Ok(Cache { dir })
    }

    pub fn text_path(&self, text: &str) -> PathBuf {
        self.dir.join(format!("txt-{}.safetensors", key(&["qwen3.5-9b", text])))
    }

    pub fn latent_path(&self, it: &Item) -> PathBuf {
        let meta = std::fs::metadata(&it.path).ok();
        let stamp = meta.as_ref().map(|m| format!("{}:{:?}", m.len(), m.modified().ok())).unwrap_or_default();
        self.dir.join(format!("lat-{}.safetensors", key(&["flux2-vae-small", &it.path.display().to_string(), &stamp, &it.bw.to_string(), &it.bh.to_string()])))
    }

    /// Encode every text that has no cache entry yet. `texts` may repeat.
    pub fn encode_texts(&self, dev: &Arc<Device>, te_path: &Path, texts: &[String], quiet: bool) -> Result<()> {
        let mut todo: Vec<String> = texts.iter().filter(|t| !self.text_path(t).is_file()).cloned().collect();
        todo.sort();
        todo.dedup();
        if todo.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        crate::info!("  text encoder: {} prompt(s) ({})", todo.len(), te_path.display());
        let te = loki_dream::text_encoder::TextEncoder::open(dev.clone(), te_path)?;
        let bar = crate::progress::bar(quiet, todo.len(), "  text encoder");
        for chunk in todo.chunks(32) {
            let enc = te.encode(chunk, &|_| {}).context("text encoder")?;
            for (t, e) in chunk.iter().zip(enc) {
                let bits: Vec<u16> = dev.dtoh_at(e.hidden.ptr, e.hidden.numel())?;
                let mut out = Out::default();
                out.add("hidden", "BF16", &e.hidden.shape, bits.iter().flat_map(|b| b.to_le_bytes()).collect());
                let mut meta = BTreeMap::new();
                meta.insert("text".to_string(), t.clone());
                safetensors_out::write(&self.text_path(t), &out, &meta)?;
            }
            bar.inc(chunk.len() as u64);
        }
        bar.finish_and_clear();
        te.close();
        crate::info!("  text encoder: done in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    /// Encode every image (and its mirror with `flip`) that has no cache entry yet.
    pub fn encode_images(&self, dev: &Arc<Device>, vae_path: &Path, items: &[Item], flip: bool, quiet: bool) -> Result<()> {
        let todo: Vec<&Item> = items
            .iter()
            .filter(|it| {
                let p = self.latent_path(it);
                match SafeTensors::open(&p) {
                    Ok(st) => flip && !st.has("latent_flip"),
                    Err(_) => true,
                }
            })
            .collect();
        if todo.is_empty() {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        crate::info!("  vae: {} image(s)", todo.len());
        let vae = loki_dream::vae::Vae::load(dev.clone(), vae_path)?;
        let bar = crate::progress::bar(quiet, todo.len(), "  vae encode");
        // decode + resize on a helper thread while the GPU encodes the previous image
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<(usize, loki_dream::image::Rgb8)>>(2);
        std::thread::scope(|s| -> Result<()> {
            let todo_ref = &todo;
            s.spawn(move || {
                for (i, it) in todo_ref.iter().enumerate() {
                    let r = data::load_rgb(&it.path).map(|img| (i, data::fit(&img, it.bw, it.bh)));
                    if tx.send(r).is_err() {
                        break;
                    }
                }
            });
            for r in rx {
                let (i, img) = r?;
                let it = todo[i];
                let mut out = Out::default();
                let mut variants = vec![("latent", img.clone())];
                if flip {
                    variants.push(("latent_flip", data::mirror(&img)));
                }
                for (name, im) in variants {
                    let lat = vae.encode(&im.to_f32(), im.h, im.w).with_context(|| format!("vae encode of {}", it.path.display()))?;
                    let v = lat.to_f32_vec(dev)?;
                    out.add(name, "F32", &lat.shape, v.iter().flat_map(|x| x.to_le_bytes()).collect());
                }
                let mut meta = BTreeMap::new();
                meta.insert("path".to_string(), it.path.display().to_string());
                meta.insert("size".to_string(), format!("{}x{}", it.bw, it.bh));
                safetensors_out::write(&self.latent_path(it), &out, &meta)?;
                bar.inc(1);
            }
            Ok(())
        })?;
        bar.finish_and_clear();
        crate::info!("  vae: done in {:.1}s", t0.elapsed().as_secs_f64());
        Ok(())
    }

    pub fn load_text(&self, dev: &Device, text: &str) -> Result<Tensor> {
        let st = SafeTensors::open(&self.text_path(text)).with_context(|| format!("text features for {text:?} are not cached"))?;
        let info = st.info("hidden")?.clone();
        let t = Tensor::new(dev, DType::BF16, &info.shape)?;
        dev.htod_at(t.ptr, st.bytes("hidden")?)?;
        Ok(t)
    }

    /// The latent f32 [tokens, 128] of an item (mirrored when `flipped`).
    pub fn load_latent(&self, dev: &Device, it: &Item, flipped: bool) -> Result<Tensor> {
        let st = SafeTensors::open(&self.latent_path(it)).with_context(|| format!("latent of {} is not cached", it.path.display()))?;
        let name = if flipped { "latent_flip" } else { "latent" };
        let info = st.info(name)?.clone();
        let t = Tensor::new(dev, DType::F32, &info.shape)?;
        dev.htod_at(t.ptr, st.bytes(name)?)?;
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn keys_separate_parts() {
        assert_ne!(super::key(&["ab", "c"]), super::key(&["a", "bc"]));
        assert_eq!(super::key(&["x"]).len(), 16);
    }
}
