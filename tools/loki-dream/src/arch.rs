//! Diffusion-model architecture detection from a checkpoint's tensor names.
//!
//! loki-dream runs five architectures; the diffusion model file decides which one, so `--dit` (or `--model`) can point at
//! any checkpoint and the matching text encoder, VAE and sampler follow. Detection only reads the safetensors header.
//! Checkpoints of other architectures are named when recognised (FLUX.1, Qwen-Image, SDXL, ...) and refused.
use crate::safetensors::SafeTensors;
use anyhow::{bail, Result};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    /// MOSS-Image (Kromoss and same-architecture fine-tunes): Qwen3.5-9B + FLUX.2 small VAE
    Moss,
    /// Krea 2 (12B single-stream MMDiT): Qwen3-VL-4B + Qwen-Image VAE
    Krea2,
    /// FLUX.2 klein (4B / 9B, base and distilled): Qwen3-4B / 8B + FLUX.2 VAE
    Flux2,
    /// Anima (Cosmos-Predict2 2B with an LLM adapter): Qwen3-0.6B + Qwen-Image VAE
    Anima,
    /// SDXL and SDXL-architecture checkpoints (UNet, CLIP-L + OpenCLIP bigG and the VAE in one file)
    Sdxl,
}

impl Arch {
    pub fn name(self) -> &'static str {
        match self {
            Arch::Moss => "MOSS-Image",
            Arch::Krea2 => "Krea 2",
            Arch::Flux2 => "FLUX.2 klein",
            Arch::Anima => "Anima",
            Arch::Sdxl => "SDXL",
        }
    }
    /// The short id used on the command line and in generation records.
    pub fn id(self) -> &'static str {
        match self {
            Arch::Moss => "moss",
            Arch::Krea2 => "krea2",
            Arch::Flux2 => "flux2",
            Arch::Anima => "anima",
            Arch::Sdxl => "sdxl",
        }
    }
    pub fn parse(s: &str) -> Result<Arch> {
        Ok(match s.to_ascii_lowercase().replace(['-', '_', '.', ' '], "").as_str() {
            "moss" | "mossimage" | "kromoss" => Arch::Moss,
            "krea2" | "krea" => Arch::Krea2,
            "flux2" | "flux2klein" | "klein" | "flux" => Arch::Flux2,
            "anima" => Arch::Anima,
            "sdxl" | "xl" => Arch::Sdxl,
            o => bail!("unknown architecture {o:?} (supported: {})", SUPPORTED),
        })
    }
}

pub const SUPPORTED: &str = "MOSS-Image (Kromoss), Krea 2, FLUX.2 klein, Anima, SDXL";

/// Prefixes some exporters put in front of the diffusion model's own tensor names.
pub const PREFIXES: [&str; 4] = ["model.diffusion_model.", "diffusion_model.", "model.", ""];

/// The prefix under which `probe` exists (if any).
pub fn prefix_of(st: &SafeTensors, probe: &str) -> Option<&'static str> {
    PREFIXES.iter().copied().find(|p| st.has(&format!("{p}{probe}")))
}

/// Detect from tensor names alone (`has` answers whether a name exists, `any` whether some name contains a fragment).
pub fn detect_names(has: &dyn Fn(&str) -> bool, any: &dyn Fn(&str) -> bool) -> Result<Arch> {
    let with = |n: &str| PREFIXES.iter().any(|p| has(&format!("{p}{n}")));
    if with("shared_block_modulation.weight") && (with("layers.0.proj.ffn_up_gate.weight") || with("layers.0.proj.ffn_up_gate.comfy_quant")) {
        return Ok(Arch::Moss);
    }
    if with("txtfusion.projector.weight") && with("blocks.0.mod.lin") {
        return Ok(Arch::Krea2);
    }
    if with("double_stream_modulation_img.lin.weight") && with("single_stream_modulation.lin.weight") {
        return Ok(Arch::Flux2);
    }
    if with("net.llm_adapter.embed.weight") || with("llm_adapter.embed.weight") {
        return Ok(Arch::Anima);
    }
    if has("model.diffusion_model.input_blocks.0.0.weight") && has("model.diffusion_model.label_emb.0.0.weight") {
        if (has("conditioner.embedders.1.model.token_embedding.weight") || has("conditioner.embedders.1.transformer.text_model.embeddings.token_embedding.weight"))
            && has("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight")
            && any("first_stage_model.decoder.")
        {
            return Ok(Arch::Sdxl);
        }
        bail!("this is an SDXL UNet without both text encoders and the VAE (a refiner, or a UNet-only file); loki-dream runs full SDXL checkpoints");
    }
    // recognised but unsupported
    let what = if with("double_blocks.0.img_attn.qkv.weight") || with("double_blocks.0.img_attn.qkv.comfy_quant") {
        if with("guidance_in.in_layer.weight") || with("vector_in.in_layer.weight") { "a FLUX.1 model" } else { "a FLUX.1-family model (Chroma / HunyuanVideo)" }
    } else if with("net.blocks.0.mlp.layer1.weight") || with("blocks.0.mlp.layer1.weight") {
        "a Cosmos-Predict2 model (without Anima's LLM adapter)"
    } else if with("transformer_blocks.0.img_mod.1.weight") || with("img_in.weight") && with("txt_norm.weight") {
        "a Qwen-Image model"
    } else if any("input_blocks.") {
        "a Stable Diffusion 1.x / 2.x model"
    } else if any("down_blocks.0.resnets") {
        "a UNet in diffusers layout (use a single-file SDXL checkpoint)"
    } else if any("joint_blocks.") {
        "a Stable Diffusion 3 model"
    } else if any("adaLN_modulation") && any("noise_refiner") {
        "a Lumina / Z-Image model"
    } else if any("patch_embedding") && any("blocks.0.self_attn") {
        "a Wan video model"
    } else if any("adaln_single") || any("scale_shift_table") && any("audio") {
        "an LTX video model"
    } else if any("caption_projection") {
        "a HiDream / PixArt model"
    } else if any("decoder.conv_in") || any("decoder.conv1") {
        "a VAE, not a diffusion model"
    } else if any("model.layers.0.self_attn") || any("model.embed_tokens") {
        "a language model (text encoder), not a diffusion model"
    } else {
        "a model of an unrecognised architecture"
    };
    bail!("this is {what}; loki-dream supports {SUPPORTED}")
}

pub fn detect(st: &SafeTensors) -> Result<Arch> {
    detect_names(&|n| st.has(n), &|frag| st.tensors.keys().any(|k| k.contains(frag)))
}

/// Detect the architecture of a diffusion model file.
pub fn detect_file(path: &Path) -> Result<Arch> {
    let st = SafeTensors::open(path)?;
    detect(&st).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(names: &[&str]) -> Result<Arch> {
        detect_names(&|n| names.contains(&n), &|f| names.iter().any(|k| k.contains(f)))
    }

    #[test]
    fn detects_each_architecture() {
        assert_eq!(det(&["shared_block_modulation.weight", "layers.0.proj.ffn_up_gate.weight"]).unwrap(), Arch::Moss);
        assert_eq!(det(&["txtfusion.projector.weight", "blocks.0.mod.lin"]).unwrap(), Arch::Krea2);
        assert_eq!(det(&["model.diffusion_model.txtfusion.projector.weight", "model.diffusion_model.blocks.0.mod.lin"]).unwrap(), Arch::Krea2);
        assert_eq!(det(&["double_stream_modulation_img.lin.weight", "single_stream_modulation.lin.weight"]).unwrap(), Arch::Flux2);
        assert_eq!(det(&["net.llm_adapter.embed.weight", "net.blocks.0.mlp.layer1.weight"]).unwrap(), Arch::Anima);
        let xl = [
            "model.diffusion_model.input_blocks.0.0.weight",
            "model.diffusion_model.label_emb.0.0.weight",
            "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight",
            "conditioner.embedders.1.model.token_embedding.weight",
            "first_stage_model.decoder.conv_in.weight",
        ];
        assert_eq!(det(&xl).unwrap(), Arch::Sdxl);
        assert!(det(&xl[..2]).unwrap_err().to_string().contains("refiner"));
    }

    #[test]
    fn refuses_others_by_name() {
        let e = det(&["model.diffusion_model.double_blocks.0.img_attn.qkv.weight", "model.diffusion_model.guidance_in.in_layer.weight"]).unwrap_err().to_string();
        assert!(e.contains("a FLUX.1 model") && e.contains("supports"), "{e}");
        let e = det(&["model.diffusion_model.input_blocks.0.0.weight"]).unwrap_err().to_string();
        assert!(e.contains("Stable Diffusion 1"), "{e}");
        assert!(det(&["foo"]).unwrap_err().to_string().contains("unrecognised"));
    }

    #[test]
    fn parses_names() {
        assert_eq!(Arch::parse("FLUX.2").unwrap(), Arch::Flux2);
        assert_eq!(Arch::parse("krea-2").unwrap(), Arch::Krea2);
        assert_eq!(Arch::parse("SDXL").unwrap(), Arch::Sdxl);
        assert!(Arch::parse("sd3").is_err());
    }
}
