"""Reference for the MiniMax H3 latent upscaler (LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler, node
`MinimaxH3LatentUpscaler3D`): runs the node's own `LatentResizer3D` (fp32, CPU) on a seeded random normalized latent.

Usage (any python with torch, safetensors, einops):
    python ref_upscaler.py <minimax_h3_latent_upscaler_3d_nodes.py> <weights.safetensors> <out_dir> [T H W HO WO]
Writes into <out_dir>:
    latent.bin  f32 [24, T, H, W]    input LATENT (a normalized H3 latent, the DiT's latent space)
    out.bin     f32 [24, T, HO, WO]  the node's output: model((latent - mean) / std, ...) * std + mean
    meta.json
`tests/upscaler.rs` (LOKI_UPSCALER_REF=<out_dir>) compares the engine against it.
"""
import sys, os, json, types
import numpy as np
import torch

node_py, weights, out_dir = sys.argv[1:4]
T, H, W, HO, WO = (int(v) for v in (sys.argv[4:9] if len(sys.argv) >= 9 else (40, 6, 8, 9, 12)))

# the node imports ComfyUI modules at load time; stub just enough of them
fp = types.ModuleType("folder_paths")
fp.folder_names_and_paths = {"latent_upscale_models": ([], set())}
fp.models_dir = "."
fp.add_model_folder_path = lambda *a, **k: None
sys.modules["folder_paths"] = fp
src = open(node_py, encoding="utf-8").read()
mod = types.ModuleType("up3d")
exec(compile(src, node_py, "exec"), mod.__dict__)

from safetensors.torch import load_file
sd = mod._extract_upscaler_sd(load_file(weights))
cfg = mod._detect_arch(sd)
model = mod.LatentResizer3D(**cfg)
model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
model.eval()

rng = np.random.default_rng(1234)
z = rng.standard_normal((24, T, H, W)).astype(np.float32)
scale = (WO / W + HO / H) / 2.0
with torch.inference_mode():
    # MinimaxH3LatentUpscaler3D.execute: normalize the LATENT (already the normalized H3 latent) with the node's
    # stats, run, de-normalize
    mean = torch.tensor(mod.LATENTS_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(mod.LATENTS_STD).view(1, -1, 1, 1, 1)
    x = (torch.from_numpy(z)[None] - mean) / std
    out = (model(x, scale=scale, target_size=(T, HO, WO), enable_chunking=True) * std + mean)[0].numpy()
os.makedirs(out_dir, exist_ok=True)
z.tofile(os.path.join(out_dir, "latent.bin"))
out.astype(np.float32).tofile(os.path.join(out_dir, "out.bin"))
json.dump({"T": T, "H": H, "W": W, "HO": HO, "WO": WO, "scale": scale, "cfg": cfg}, open(os.path.join(out_dir, "meta.json"), "w"))
print("wrote", out_dir, out.shape, float(np.abs(out).mean()))
