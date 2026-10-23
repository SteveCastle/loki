//! LoRAs on the MOSS-Image block linears, applied unmerged: y = W x + (x down^T) up^T * strength * alpha / rank.
//!
//! The Kromoss README is explicit that the 8-step DMD LoRAs must not be merged into the bf16 weights (the delta is close
//! to bf16's rounding step), and merging into int8 weights would be worse, so every LoRA is evaluated as two extra small
//! GEMMs per targeted linear (rank 128: about 10% more DiT work). Several LoRAs stack by concatenating their ranks.
//!
//! Accepted key styles for `layers.N.proj.{q,k,v,qkv,o,ffn_up_gate,ffn_down}`:
//!   kohya / Kromoss      `diffusion_model.layers.N.proj.q.lora_down.weight` + `lora_up.weight` + `alpha`
//!   ComfyUI / bare       `layers.N.proj.q.lora_down.weight` (any of the prefixes diffusion_model., transformer.,
//!                        model.diffusion_model., base_model.model.)
//!   kohya flattened      `lora_unet_layers_N_proj_q.lora_down.weight`
//!   diffusers / PEFT     `transformer.layers.N.proj.q.lora_A.weight` + `lora_B.weight` (alpha = rank when absent)
//! q / k / v deltas land on their slices of the fused qkv projection (as ComfyUI maps them). Keys for other modules
//! (text refiner, time embedding, ...) are reported and skipped.
use crate::cuda::Device;
use crate::ops::{self, Act, Epi};
use crate::safetensors::SafeTensors;
use crate::tensor::{bf16_bits, DType, Tensor};
use anyhow::{bail, ensure, Context, Result};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// One linear's combined delta: down [R, K], up [N, R] (scale folded in), both bf16, R a multiple of 8.
pub struct LoraDelta {
    pub down: Tensor,
    pub up: Tensor,
}
impl LoraDelta {
    /// out[M, N] += (x[M, K] down^T) up^T
    pub fn apply(&self, dev: &Device, x: &Tensor, out: &Tensor) -> Result<()> {
        let m = x.shape[0];
        let r = self.down.shape[0];
        let mid = Tensor::new(dev, DType::BF16, &[m, r])?;
        ops::gemm_bf16(dev, x, &self.down, None, Act::None, Epi::Store, &mid)?;
        ops::gemm_bf16(dev, &mid, &self.up, None, Act::None, Epi::AddRes(out), out)
    }
}

#[derive(Default)]
pub struct BlockLora {
    pub qkv: Option<LoraDelta>,
    pub o: Option<LoraDelta>,
    pub up_gate: Option<LoraDelta>,
    pub down: Option<LoraDelta>,
}

/// A LoRA file and its strength, as given on the command line (`FILE[:STRENGTH]`).
#[derive(Clone, Debug)]
pub struct LoraSpec {
    pub path: PathBuf,
    pub strength: f32,
}

impl LoraSpec {
    pub fn parse(s: &str) -> Result<LoraSpec> {
        if let Some((p, v)) = s.rsplit_once(':') {
            if let Ok(x) = v.trim().parse::<f32>() {
                if !p.is_empty() {
                    return Ok(LoraSpec { path: PathBuf::from(p), strength: x });
                }
            }
        }
        Ok(LoraSpec { path: PathBuf::from(s), strength: 1.0 })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Target {
    Qkv,
    O,
    UpGate,
    Down,
}

/// (layer, target, first output column of the fused projection, its width) for a module name.
fn classify(module: &str) -> Option<(usize, Target, Option<usize>)> {
    let m = module.replace("lora_unet_", "");
    let m = if m.contains('.') { m } else { m.replace("layers_", "layers.").replacen("_proj_", ".proj.", 1) };
    let rest = m.split("layers.").nth(1)?;
    let (idx, tail) = rest.split_once('.')?;
    let layer: usize = idx.parse().ok()?;
    let name = tail.strip_prefix("proj.")?;
    Some(match name {
        "q" => (layer, Target::Qkv, Some(0)),
        "k" => (layer, Target::Qkv, Some(1)),
        "v" => (layer, Target::Qkv, Some(2)),
        "qkv" => (layer, Target::Qkv, None),
        "o" => (layer, Target::O, None),
        "ffn_up_gate" => (layer, Target::UpGate, None),
        "ffn_down" => (layer, Target::Down, None),
        _ => return None,
    })
}

/// Raw little-endian tensor bytes with their dtype, read element by element (no copies of the whole file).
#[derive(Clone, Copy)]
struct Raw<'a> {
    b: &'a [u8],
    dt: u8, // 0 f32, 1 bf16, 2 f16
}
impl Raw<'_> {
    #[inline]
    fn at(&self, i: usize) -> f32 {
        match self.dt {
            0 => f32::from_le_bytes(self.b[i * 4..i * 4 + 4].try_into().unwrap()),
            1 => half::bf16::from_bits(u16::from_le_bytes([self.b[i * 2], self.b[i * 2 + 1]])).to_f32(),
            _ => half::f16::from_bits(u16::from_le_bytes([self.b[i * 2], self.b[i * 2 + 1]])).to_f32(),
        }
    }
}
fn raw<'a>(st: &'a SafeTensors, name: &str) -> Result<Raw<'a>> {
    let dt = match st.info(name)?.dtype.as_str() {
        "F32" => 0,
        "BF16" => 1,
        "F16" => 2,
        o => bail!("{name}: unsupported LoRA dtype {o}"),
    };
    Ok(Raw { b: st.bytes(name)?, dt })
}

struct Piece<'a> {
    down: Raw<'a>, // [r, k]
    up: Raw<'a>,   // [n_t, r]
    r: usize,
    k: usize,
    n_t: usize,
    col0: usize,
    scale: f32,
}

/// Width of each fused projection.
fn out_width(t: Target) -> usize {
    use crate::dit::{DIM, FFN};
    match t {
        Target::Qkv => 3 * DIM,
        Target::O => DIM,
        Target::UpGate => 2 * FFN,
        Target::Down => DIM,
    }
}

/// Load LoRAs and build the per-block deltas for `layers` blocks.
pub fn load(dev: &Device, specs: &[LoraSpec], layers: usize) -> Result<Vec<BlockLora>> {
    let files: Vec<SafeTensors> = specs.iter().map(|spec| SafeTensors::open(&spec.path).with_context(|| format!("opening LoRA {}", spec.path.display()))).collect::<Result<_>>()?;
    let mut pieces: BTreeMap<(usize, Target), Vec<Piece>> = BTreeMap::new();
    for (spec, st) in specs.iter().zip(&files) {
        let mut used = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        let mut names: Vec<&String> = st.tensors.keys().collect();
        names.sort();
        for key in names {
            let (module, up_key) = if let Some(m) = key.strip_suffix(".lora_down.weight") {
                (m.to_string(), format!("{m}.lora_up.weight"))
            } else if let Some(m) = key.strip_suffix(".lora_A.weight") {
                (m.to_string(), format!("{m}.lora_B.weight"))
            } else {
                continue;
            };
            let Some((layer, target, slot)) = classify(&module) else {
                skipped.push(module);
                continue;
            };
            ensure!(layer < layers, "{}: LoRA for block {layer}, the model has {layers}", spec.path.display());
            let dinfo = st.info(key)?.clone();
            let uinfo = st.info(&up_key).with_context(|| format!("{}: {key} has no matching up weight", spec.path.display()))?.clone();
            ensure!(dinfo.shape.len() == 2 && uinfo.shape.len() == 2 && dinfo.shape[0] == uinfo.shape[1], "{}: {module}: bad LoRA shapes {:?} / {:?}", spec.path.display(), dinfo.shape, uinfo.shape);
            let (r, k, n_t) = (dinfo.shape[0], dinfo.shape[1], uinfo.shape[0]);
            let alpha = if st.has(&format!("{module}.alpha")) { st.f32s(&format!("{module}.alpha"))?[0] } else { r as f32 };
            let (col0, expect) = match slot {
                Some(i) => (i * crate::dit::DIM, crate::dit::DIM),
                None => (0, out_width(target)),
            };
            ensure!(n_t == expect, "{}: {module}: output width {n_t}, expected {expect}", spec.path.display());
            let want_k = match target {
                Target::Down => crate::dit::FFN,
                _ => crate::dit::DIM,
            };
            ensure!(k == want_k, "{}: {module}: input width {k}, expected {want_k}", spec.path.display());
            pieces.entry((layer, target)).or_default().push(Piece { down: raw(st, key)?, up: raw(st, &up_key)?, r, k, n_t, col0, scale: spec.strength * alpha / r as f32 });
            used += 1;
        }
        if used == 0 {
            bail!("{}: no MOSS-Image block LoRA weights found (expected keys like diffusion_model.layers.0.proj.q.lora_down.weight)", spec.path.display());
        }
        if !skipped.is_empty() {
            crate::info!("  lora {}: {} module(s) outside the DiT blocks skipped (e.g. {})", spec.path.display(), skipped.len(), skipped[0]);
        }
        crate::info!("  lora {} x{}: {used} linears", spec.path.file_name().unwrap_or_default().to_string_lossy(), spec.strength);
    }
    // bf16 conversion and packing on all cores, uploads after
    let groups: Vec<((usize, Target), Vec<Piece>)> = pieces.into_iter().collect();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8).min(16);
    let built: Vec<(usize, Target, usize, usize, Vec<u16>, Vec<u16>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let groups = &groups;
                sc.spawn(move || {
                    let mut v = Vec::new();
                    for ((layer, target), ps) in groups.iter().skip(t).step_by(threads) {
                        let k = ps[0].k;
                        let r_real: usize = ps.iter().map(|p| p.r).sum();
                        let r = (r_real + 7) / 8 * 8;
                        let n = out_width(*target);
                        let mut down = vec![0u16; r * k];
                        let mut up = vec![0u16; n * r];
                        let mut r0 = 0;
                        for p in ps {
                            for i in 0..p.r {
                                for j in 0..k {
                                    down[(r0 + i) * k + j] = bf16_bits(p.down.at(i * p.k + j));
                                }
                            }
                            for row in 0..p.n_t {
                                for i in 0..p.r {
                                    up[(p.col0 + row) * r + r0 + i] = bf16_bits(p.up.at(row * p.r + i) * p.scale);
                                }
                            }
                            r0 += p.r;
                        }
                        v.push((*layer, *target, r, k, down, up));
                    }
                    v
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut out: Vec<BlockLora> = (0..layers).map(|_| BlockLora::default()).collect();
    for (layer, target, r, k, down, up) in built {
        let n = out_width(target);
        let delta = LoraDelta { down: Tensor::from_bf16(dev, &down, &[r, k])?, up: Tensor::from_bf16(dev, &up, &[n, r])? };
        let b = &mut out[layer];
        match target {
            Target::Qkv => b.qkv = Some(delta),
            Target::O => b.o = Some(delta),
            Target::UpGate => b.up_gate = Some(delta),
            Target::Down => b.down = Some(delta),
        }
    }
    for f in files {
        f.drop_in_background();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_key_styles() {
        assert_eq!(classify("diffusion_model.layers.3.proj.q"), Some((3, Target::Qkv, Some(0))));
        assert_eq!(classify("transformer.layers.41.proj.v"), Some((41, Target::Qkv, Some(2))));
        assert_eq!(classify("lora_unet_layers_7_proj_ffn_up_gate"), Some((7, Target::UpGate, None)));
        assert_eq!(classify("layers.0.proj.qkv"), Some((0, Target::Qkv, None)));
        assert_eq!(classify("diffusion_model.text_embedder.blocks.0.qkv"), None);
    }

    #[test]
    fn spec_strength() {
        let s = LoraSpec::parse("dmd.safetensors:0.75").unwrap();
        assert_eq!((s.path.to_str().unwrap(), s.strength), ("dmd.safetensors", 0.75));
        let s = LoraSpec::parse(r"C:\loras\x.safetensors").unwrap();
        assert_eq!((s.path.to_str().unwrap(), s.strength), (r"C:\loras\x.safetensors", 1.0));
        let s = LoraSpec::parse(r"C:\loras\x.safetensors:1.5").unwrap();
        assert_eq!((s.path.to_str().unwrap(), s.strength), (r"C:\loras\x.safetensors", 1.5));
    }
}
