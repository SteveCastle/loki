"""Dump ComfyUI's vision-tower outputs (merged + 3 deepstack, f32) for one image: <img> <out_dir>"""
import sys, os
sys.path.insert(0, os.getcwd())
import torch, numpy as np
from PIL import Image
import comfy.sd
from comfy.sd import CLIPType
TE = r"C:\Users\steph\bin\models\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
img, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=CLIPType.MINIMAX)
clip.load_model() if hasattr(clip, "load_model") else None
import comfy.model_management as mm
mm.load_models_gpu([clip.patcher])
tr = clip.cond_stage_model.qwen3vl_32b.transformer
x = torch.from_numpy(np.array(Image.open(img).convert("RGB")).astype(np.float32) / 255.0)[None]
dev = mm.get_torch_device()
with torch.no_grad():
    merged, extra = tr.preprocess_embed({"type": "image", "data": x}, dev)
print("merged", merged.dtype, merged.shape, "visual weight dtype", tr.visual.blocks[0].attn.qkv.weight.dtype)
merged.float().cpu().numpy().astype(np.float32).tofile(os.path.join(out, "merged.bin"))
for i, d in enumerate(extra["deepstack"]):
    d.float().cpu().numpy().astype(np.float32).tofile(os.path.join(out, "deep%d.bin" % i))
