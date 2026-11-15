"""MiniMax-H3-Image end to end through ComfyUI's own nodes (no server): MiniMaxH3ImageConditioning -> KSampler
(euler / simple / 6 steps / cfg 1) -> VAEDecode, for an apples-to-apples comparison with the Rust engine.

Usage (cwd = a ComfyUI >= 0.45 root, embedded python; LOKI_PYLIB = a folder with comfy-kitchen >= 0.2.41):
    python ref_e2e_image.py --prompt "..." --width 1024 --height 1024 --seed 1 --out out.png [--image a.png ...]
"""
import argparse, os, sys, time
if os.environ.get("LOKI_PYLIB"):
    sys.path.insert(0, os.environ["LOKI_PYLIB"])
sys.path.insert(0, os.getcwd())
import numpy as np
import torch
from PIL import Image

import comfy.sd
import comfy.utils
import nodes
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

M = r"C:\Users\steph\bin\models"
ap = argparse.ArgumentParser()
ap.add_argument("--prompt", required=True)
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--height", type=int, default=0)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--out", required=True)
a = ap.parse_args()
torch.set_grad_enabled(False)

t0 = time.time()
clip = comfy.sd.load_clip(ckpt_paths=[os.path.join(M, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")], clip_type=comfy.sd.CLIPType.MINIMAX)
vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(os.path.join(M, "minimax_h3_image_vae_fp16.safetensors")))
images = {}
for i, p in enumerate(a.image):
    images[f"image_{i + 1}"] = torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
res = MiniMaxH3ImageConditioning.execute(clip, a.prompt, a.width, a.height, vae=vae, images=images or None)
positive, latent = res.result if hasattr(res, "result") else res
print("conditioning", time.time() - t0, "latent", tuple(latent["samples"].shape))
del clip
model = comfy.sd.load_diffusion_model(os.path.join(M, "minimax_h3_image_pruned_int8_convrot.safetensors"))
t1 = time.time()
(out,) = nodes.common_ksampler(model, a.seed, a.steps, 1.0, "euler", "simple", positive, positive, latent, denoise=1.0)
print("sampled", time.time() - t1)
del model
img = vae.decode(out["samples"])
img = img.reshape(-1, img.shape[-3], img.shape[-2], 3)[0]
Image.fromarray((img.float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)).save(a.out)
print("wrote", a.out, tuple(img.shape), "total", time.time() - t0)
