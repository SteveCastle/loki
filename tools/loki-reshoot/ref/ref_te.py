"""Reference dump of ComfyUI's MiniMax H3 text/vision conditioning (Qwen3-VL-32B, 50 layers).

Usage (cwd = the ComfyUI root, embedded python):
    python <repo>/ref/ref_te.py <out_root> [case ...]
Creates synthetic/test inputs under <out_root>/inputs and, per case, <out_root>/<case>/:
    case.json   (prompt + ref items with input file paths, consumed by `te_check`)
    tokens.json (token ids; vision blocks as {"vision": n_tokens})
    context.bin (f32 [L,5120] hidden state after layer 50), tags.json, meta.json
"""
import sys, os, json, time
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.model_management
from comfy.sd import CLIPType

TE = r"C:\Users\steph\dev\loki-reshoot\models\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
TESTIMGS = r"C:\Users\steph\dev\4kify\testimgs"
PROMPT = ("A woman in a red coat walks through a snowy street at dusk, talking to the camera; "
          "she says: \"It's 5 degrees, can you believe it?\" Warm lights, 35mm film look.")


def make_inputs(root):
    d = os.path.join(root, "inputs")
    os.makedirs(d, exist_ok=True)
    src = Image.open(os.path.join(TESTIMGS, "portrait.jpg")).convert("RGB")
    src2 = Image.open(os.path.join(TESTIMGS, "workflow_input.jpg")).convert("RGB")
    out = {}
    out["img_a"] = (src.resize((320, 192), Image.LANCZOS), "img_a.png")
    out["img_b"] = (src2.resize((256, 256), Image.LANCZOS), "img_b.png")
    out["img_odd"] = (src2.resize((250, 170), Image.LANCZOS), "img_odd.png")
    for k, (im, name) in out.items():
        im.save(os.path.join(d, name))
    # 5 frames (odd -> repeat-padded) of a slow pan, 288x160
    W, H = src2.size
    frames = []
    for i in range(5):
        x0 = int(i * W * 0.04)
        crop = src2.crop((x0, 0, x0 + int(W * 0.8), int(H * 0.8)))
        fr = crop.resize((288, 160), Image.LANCZOS)
        name = "vid_%d.png" % i
        fr.save(os.path.join(d, name))
        frames.append(name)
    return {k: os.path.join(d, v[1]) for k, v in out.items()}, [os.path.join(d, f) for f in frames]


def load_img(p):
    return torch.from_numpy(np.array(Image.open(p).convert("RGB")).astype(np.float32) / 255.0)[None]


def cases(imgs, frames):
    vid = {"type": "video", "frames": frames, "timestamps": [i / 2.0 for i in range(len(frames))]}
    return {
        "text": {"prompt": PROMPT, "items": []},
        "img1": {"prompt": PROMPT, "items": [{"type": "image", "path": imgs["img_a"]}, {"type": "audio"}]},
        "img2": {"prompt": PROMPT, "items": [{"type": "image", "path": imgs["img_a"]}, {"type": "audio"},
                                             {"type": "image", "path": imgs["img_b"]}]},
        "video": {"prompt": PROMPT, "items": [{"type": "audio"}, vid]},
        "odd": {"prompt": "close-up, (soft light:1.2) <d> test", "items": [{"type": "image", "path": imgs["img_odd"]}]},
    }


def to_comfy_items(items):
    out = []
    for it in items:
        if it["type"] == "image":
            out.append({"type": "image", "data": load_img(it["path"])})
        elif it["type"] == "audio":
            out.append({"type": "audio"})
        else:
            fr = torch.cat([load_img(p) for p in it["frames"]], 0)
            out.append({"type": "video", "data": fr, "timestamps": list(it["timestamps"])})
    return out


def main():
    root = sys.argv[1]
    want = sys.argv[2:]
    imgs, frames = make_inputs(root)
    allc = cases(imgs, frames)
    t0 = time.time()
    clip = comfy.sd.load_clip(ckpt_paths=[TE], clip_type=CLIPType.MINIMAX)
    print("load", time.time() - t0, flush=True)
    for name, c in allc.items():
        if want and name not in want:
            continue
        od = os.path.join(root, name)
        os.makedirs(od, exist_ok=True)
        json.dump(c, open(os.path.join(od, "case.json"), "w"), indent=1)
        tokens = clip.tokenize(c["prompt"], minimax_ref_items=to_comfy_items(c["items"]))
        key = next(iter(tokens))
        ids = []
        for t in tokens[key][0]:
            ids.append(int(t[0]) if isinstance(t[0], (int, np.integer)) else "<VIS>")
        json.dump(ids, open(os.path.join(od, "tokens.json"), "w"))
        torch.cuda.synchronize()
        t1 = time.time()
        cond = clip.encode_from_tokens_scheduled(tokens)
        torch.cuda.synchronize()
        dt = time.time() - t1
        ctx = cond[0][0].float().cpu().numpy()[0]
        extra = cond[0][1]
        tags = extra.get("minimax_token_tags")
        tags = [int(v) for v in tags.reshape(-1).tolist()] if tags is not None else None
        ctx.astype(np.float32).tofile(os.path.join(od, "context.bin"))
        json.dump(tags, open(os.path.join(od, "tags.json"), "w"))
        json.dump({"shape": list(ctx.shape), "dtype": str(cond[0][0].dtype), "encode_s": dt}, open(os.path.join(od, "meta.json"), "w"))
        print(name, ctx.shape, cond[0][0].dtype, "encode %.2fs" % dt, "tags0", None if tags is None else tags.count(0), flush=True)


if __name__ == "__main__":
    main()
