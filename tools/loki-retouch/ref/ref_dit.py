"""Reference dump of ComfyUI's Qwen Image 2.1 VAE + DiT for one denoising step.

Usage (ComfyUI dir, embedded python): python ref_dit.py <te_ref_dir> <out_dir> <target_w> <target_h> <sigma> [seed]
Reads te_ref_dir/{resized.png,context.bin,meta.json}.
Writes: ref_latent.bin (f32 [64,h,w], raw VAE output), x.bin (f32 [64,H,W] model input), denoised.bin (f32 [64,H,W]),
        decoded.bin (f32 [H*16, W*16, 3] VAE decode of denoised), meta.json
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from PIL import Image

import comfy.cli_args
if os.environ.get("REF_SAGE") == "1":
    comfy.cli_args.args.use_sage_attention = True
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.latent_formats

@torch.no_grad()
def main():
    te_dir, out_dir = sys.argv[1], sys.argv[2]
    tw, th = int(sys.argv[3]), int(sys.argv[4])
    sigma = float(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    os.makedirs(out_dir, exist_ok=True)
    models = os.path.expanduser("~/bin/models")
    meta = json.load(open(os.path.join(te_dir, "meta.json")))
    L, D = meta["shape"]
    ctx = torch.from_numpy(np.fromfile(os.path.join(te_dir, "context.bin"), dtype=np.float32).reshape(1, L, D))
    slot = meta["image_slots"][0]

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(os.path.join(models, "qwen_image_2.1_vae_bf16.safetensors")))
    im = Image.open(os.path.join(te_dir, "resized.png")).convert("RGB")
    pix = torch.from_numpy(np.array(im).astype(np.float32) / 255.0)[None]
    ref_latent = vae.encode(pix)  # [1,64,h,w]
    ref_latent.float().cpu().numpy().tofile(os.path.join(out_dir, "ref_latent.bin"))
    print("ref latent", ref_latent.shape, float(ref_latent.mean()), float(ref_latent.std()))

    model = comfy.sd.load_diffusion_model(os.path.join(models, "qwen_image_2.1_int8_convrot.safetensors"))
    comfy.model_management.load_models_gpu([model])
    dev = model.load_device
    lf = comfy.latent_formats.QwenImage21()

    torch.manual_seed(seed)
    noise = torch.randn((1, 64, th // 16, tw // 16), dtype=torch.float32)
    x = noise * sigma  # CONST noise scaling with an empty latent
    x.numpy().tofile(os.path.join(out_dir, "x.bin"))
    ref_in = lf.process_in(ref_latent).to(dev)
    with torch.no_grad():
        denoised = model.model.apply_model(x.to(dev), torch.tensor([sigma], device=dev), c_crossattn=ctx.to(dev),
                                           ref_latents=[ref_in], image_slots=[slot])
    denoised = denoised.float().cpu()
    denoised.numpy().tofile(os.path.join(out_dir, "denoised.bin"))
    print("denoised", denoised.shape, float(denoised.mean()), float(denoised.std()))
    comfy.model_management.unload_all_models()
    dec = vae.decode(denoised)  # [1,H,W,3]
    dec.float().cpu().numpy().tofile(os.path.join(out_dir, "decoded.bin"))
    Image.fromarray((dec[0].numpy() * 255).round().clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, "decoded.png"))
    json.dump({"ref_latent_shape": list(ref_latent.shape), "x_shape": list(x.shape), "sigma": sigma, "slot": slot,
               "decoded_shape": list(dec.shape)}, open(os.path.join(out_dir, "meta.json"), "w"))
    print("done")

if __name__ == "__main__":
    main()
