"""Shared helpers for the loki-dream reference harness.

Builds the official MOSS-Image components (pipeline_moss_image.py from lodestones/Kromoss, the ground truth for numerics)
from the same single files the native engine loads, so both sides see identical weights:

  diffusion model : kromoss_bf16.safetensors (ComfyUI layout, fused proj.qkv -> split back into q/k/v)
  text encoder    : qwen3.5_9b_bf16.safetensors (ComfyUI single file) or the diffusers text_encoder/ folder
  VAE             : flux2-vae_small.safetensors (diffusers AutoencoderKLFlux2 keys)
  tokenizer       : ref_src/kromoss/tokenizer

Run with the venv's interpreter in isolated mode:  tools/ref-env/Scripts/python -I ref/<script>.py ...
"""
from __future__ import annotations

import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
KROMOSS = ROOT / "ref_src" / "kromoss"


def models_dir() -> Path:
    for d in [os.environ.get("LOKI_MODELS"), str(Path.home() / "bin" / "models"), str(ROOT / "models")]:
        if d and Path(d).is_dir():
            return Path(d)
    raise SystemExit("set LOKI_MODELS to the folder holding the Kromoss files")


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("pipeline_moss_image", KROMOSS / "pipeline_moss_image.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_moss_image"] = mod
    spec.loader.exec_module(mod)
    return mod


def read_safetensors(path: Path, prefix_filter=None, device="cpu", dtype=None):
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), "pt", device=device) as f:
        for k in f.keys():
            if prefix_filter and not prefix_filter(k):
                continue
            t = f.get_tensor(k)
            out[k] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    return out


def load_transformer(mod, path: Path, device="cuda"):
    """MOSSImageTransformer from the ComfyUI single file (fused qkv split back into q, k, v)."""
    sd = read_safetensors(path)
    if any(k.endswith(".comfy_quant") for k in sd):
        raise SystemExit("the reference needs the bf16 diffusion model, not the int8 file")
    for k in [k for k in sd if k.endswith(".proj.qkv.weight")]:
        q, kk, v = sd.pop(k).chunk(3, 0)
        p = k[: -len("qkv.weight")]
        sd[p + "q.weight"], sd[p + "k.weight"], sd[p + "v.weight"] = q, kk, v
    with torch.device("meta"):
        model = mod.MOSSImageTransformer()
    model = model.to_empty(device="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if m != "rope.bands"]
    if missing or unexpected:
        raise SystemExit(f"transformer keys: missing {missing[:5]} unexpected {unexpected[:5]}")
    model.rope.bands = model.rope._default_bands(torch.device("cpu"))
    return model.to(device=device, dtype=torch.bfloat16).eval()


def load_text_encoder(device="cuda"):
    """Qwen3_5Model (transformers) with the weights of qwen3.5_9b_bf16.safetensors (language model only)."""
    from transformers import AutoConfig, AutoModel
    cfg = AutoConfig.from_pretrained(str(KROMOSS / "text_encoder"))
    with torch.device("meta"):
        model = AutoModel.from_config(cfg, dtype=torch.bfloat16)
    model = model.to_empty(device=device)
    path = models_dir() / "qwen3.5_9b_bf16.safetensors"
    sd = read_safetensors(path, device=device)
    want = dict(model.state_dict())
    mapped = {}
    for k, v in sd.items():
        for cand in (k, k.replace("model.language_model.", "language_model."), k.replace("model.", "language_model.", 1),
                     k.replace("model.visual.", "visual."), "language_model." + k):
            if cand in want:
                mapped[cand] = v.to(torch.bfloat16) if v.is_floating_point() else v  # the official text_encoder/ folder is all bf16
                break
    missing = [k for k in want if k not in mapped]
    if missing:
        raise SystemExit(f"text encoder: {len(missing)} missing keys, e.g. {missing[:5]}; file keys e.g. {list(sd)[:5]}")
    model.load_state_dict(mapped, strict=True, assign=True)
    # rotary inv_freq buffers are non-persistent (not in the file) and were left uninitialized by the meta-device
    # construction: rebuild them, or RoPE silently becomes the identity
    n = 0
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "compute_default_rope_parameters"):
            inv, _ = m.compute_default_rope_parameters(m.config, device)
            m.inv_freq = inv.to(device)
            m.original_inv_freq = inv.clone().to(device)
            n += 1
    if n == 0:
        raise SystemExit("text encoder: no rotary embedding found to initialize")
    return model.eval()


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(KROMOSS / "tokenizer"))


def load_vae(device="cuda", dtype=torch.float32):
    from diffusers import AutoencoderKLFlux2
    cfg = json.loads((KROMOSS / "vae" / "config.json").read_text())
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    vae = AutoencoderKLFlux2.from_config(cfg)
    sd = read_safetensors(models_dir() / "flux2-vae_small.safetensors")
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"vae keys: missing {missing[:5]} unexpected {unexpected[:5]}")
    return vae.to(device=device, dtype=dtype).eval()


def save(out: Path, name: str, t):
    """Raw little-endian dump + a JSON sidecar entry: <name>.bin holds f32 (or i32 for integer tensors)."""
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
        a = t.to(torch.int32).numpy() if not t.is_floating_point() else t.float().numpy()
    else:
        a = np.asarray(t)
        a = a.astype(np.int32) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
    a = np.ascontiguousarray(a)
    (out / f"{name}.bin").write_bytes(a.tobytes())
    idx_path = out / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    idx[name] = {"shape": list(a.shape), "dtype": "i32" if a.dtype == np.int32 else "f32"}
    idx_path.write_text(json.dumps(idx, indent=1))


def free_cuda():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
