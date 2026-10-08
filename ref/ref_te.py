"""Reference dump of ComfyUI's Qwen Image 2.1 text-encoder path for one image + prompt.

Usage (from the ComfyUI directory, with the embedded python):
    python ref_te.py <image> <out_dir> [resolution]
Writes: tokens.json (token ids of the full template, before image expansion),
        context.bin (f32 [L,4096]), meta.json (shape, image_slots, vision size).
"""
import sys, os, json, math
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.utils
import comfy.model_management
from comfy.sd import CLIPType

PROMPT = open(os.path.join(os.path.dirname(__file__), "prompt.txt"), encoding="utf-8").read()

def main():
    img_path, out_dir = sys.argv[1], sys.argv[2]
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(out_dir, exist_ok=True)
    te_path = os.path.expanduser("~/dev/loki-retouch/models/qwen3vl_8b_int8_convrot.safetensors")
    clip = comfy.sd.load_clip(ckpt_paths=[te_path], clip_type=CLIPType.QWEN_IMAGE)

    im = Image.open(img_path).convert("RGB")
    arr = torch.from_numpy(np.array(im).astype(np.float32) / 255.0)[None]  # [1,H,W,3]
    samples = arr.movedim(-1, 1)
    if resolution > 0:
        ratio = samples.shape[3] / samples.shape[2]
        width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
    else:
        width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
    width, height = max(32, width), max(32, height)
    if (width, height) == (samples.shape[3], samples.shape[2]):
        s = arr
    else:
        s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
    # save the exact resized pixels the model sees
    Image.fromarray((s[0].numpy() * 255.0).round().clip(0, 255).astype(np.uint8)).save(os.path.join(out_dir, "resized.png"))

    tokens = clip.tokenize(PROMPT, images=[s[:, :, :, :3]], keep_vision=False, prevent_empty_text=True)
    key = next(iter(k for k in tokens if k != "keep_vision"))
    ids = [t[0] if isinstance(t[0], int) else "<IMG>" for t in tokens[key][0]]
    json.dump(ids, open(os.path.join(out_dir, "tokens.json"), "w"))

    cond = clip.encode_from_tokens_scheduled(tokens)
    ctx = cond[0][0].float().cpu().numpy()[0]
    extra = cond[0][1]
    ctx.astype(np.float32).tofile(os.path.join(out_dir, "context.bin"))
    json.dump({"shape": list(ctx.shape), "image_slots": extra.get("image_slots"), "width": int(width), "height": int(height),
               "has_mask": "attention_mask" in extra}, open(os.path.join(out_dir, "meta.json"), "w"))
    print("context", ctx.shape, "slots", extra.get("image_slots"), "size", width, height)

if __name__ == "__main__":
    main()
