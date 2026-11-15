"""Reference dump of ComfyUI's MiniMax-H3-Image VAE (comfy.sd.VAE path, fp16): H3 encoder + 2D ViT decoder.

Usage (cwd = a ComfyUI >= 0.45 root, embedded python; LOKI_PYLIB = a folder with comfy-kitchen >= 0.2.41 when the
embedded one is older):
    python ref_ivae.py <out_dir> <image_path>
For each case writes into <out_dir>/<case>/:
    pixels.u8      u8  [H, W, 3]   input image
    moments.bin    f32 [48, H/16, W/16]  tiled encoder moments (mean | logvar) before sampling
    latent.bin     f32 [24, H/16, W/16]  vae.encode(pixels/255): posterior sample (seed 42), normalized
    decoded.bin    f32 [H, W, 3] vae.decode(latent) in [0, 1]
    meta.json
"""
import sys, os, json
if os.environ.get("LOKI_PYLIB"):
    sys.path.insert(0, os.environ["LOKI_PYLIB"])
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.utils

MODEL = os.environ.get("LOKI_IVAE", r"C:\Users\steph\bin\models\minimax_h3_image_vae_fp16.safetensors")


@torch.no_grad()
def main():
    out_dir, img_path = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(img_path).convert("RGB")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(MODEL))
    fsm = vae.first_stage_model
    print("vae", type(fsm).__name__, "dtype", vae.vae_dtype)
    moments = {}
    orig = fsm._adaptive_encode

    def enc_hook(x):
        m = orig(x)
        moments["m"] = m.float().cpu()
        return m

    fsm._adaptive_encode = enc_hook
    for name, (w, h) in {"small": (320, 224), "tiled": (704, 480)}.items():
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        px = np.asarray(img.resize((w, h), Image.LANCZOS), dtype=np.uint8)
        px.tofile(os.path.join(d, "pixels.u8"))
        x = torch.from_numpy(px).float().div(255.0)[None]  # [1, H, W, 3]
        z = vae.encode(x)  # [1, 24, 1, h, w]
        moments["m"][0, :, 0].numpy().astype(np.float32).tofile(os.path.join(d, "moments.bin"))
        z.float()[0, :, 0].cpu().numpy().astype(np.float32).tofile(os.path.join(d, "latent.bin"))
        y = vae.decode(z)  # [1, H, W, 3] (image VAE: not_video)
        y = y.reshape(-1, h, w, 3)[0]
        y.float().cpu().numpy().astype(np.float32).tofile(os.path.join(d, "decoded.bin"))
        json.dump({"w": w, "h": h, "latent": list(z.shape)}, open(os.path.join(d, "meta.json"), "w"))
        print(name, w, h, "latent", tuple(z.shape), "decoded", tuple(y.shape), "zstd", float(z.float().std()))


if __name__ == "__main__":
    main()
