"""Reference dumps for loki-dream's SDXL path, from diffusers' StableDiffusionXLPipeline loaded from the same single-file
checkpoint the native engine reads (`from_single_file`, configs from stabilityai/stable-diffusion-xl-base-1.0), in fp32.

  python -I ref/ref_sdxl.py te   OUT --ckpt FILE -p "a cat" -p ""         text conditioning (context + pooled) per prompt
  python -I ref/ref_sdxl.py unet OUT --ckpt FILE --size 1024x768 --t 500  one UNet forward on prompt 0's conditioning
  python -I ref/ref_sdxl.py vae  OUT --ckpt FILE --size 512x512           decode a latent (and encode the result)
  python -I ref/ref_sdxl.py full OUT --ckpt FILE --size 768x768 --steps 20 --cfg 6 --seed 3   euler + normal schedule,
                                                                  prompt 0 against prompt 1 as the negative -> OUT/full.png

`unet` and `vae` read the conditioning `te` wrote into the same OUT folder. Dumps are OUT/<name>.bin + index.json,
compared by `arch_check sdxl te|unet|vae OUT --ckpt FILE`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

_spec = importlib.util.spec_from_file_location("ref_common", Path(__file__).resolve().parent / "ref_common.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)


def pipeline(ckpt: str):
    from diffusers import StableDiffusionXLPipeline
    return StableDiffusionXLPipeline.from_single_file(ckpt, torch_dtype=torch.float32)


def load(out: Path, name: str) -> torch.Tensor:
    idx = json.loads((out / "index.json").read_text())
    a = np.frombuffer((out / f"{name}.bin").read_bytes(), dtype=np.float32).reshape(idx[name]["shape"])
    return torch.from_numpy(a.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["te", "unet", "vae", "full"])
    ap.add_argument("out")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("-p", "--prompt", action="append", default=[])
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--t", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=6.0)
    a = ap.parse_args()
    out = Path(a.out)
    w, h = (int(x) for x in a.size.lower().split("x"))
    pipe = pipeline(a.ckpt)
    with torch.no_grad():
        if a.stage == "te":
            pipe.text_encoder.to("cuda")
            pipe.text_encoder_2.to("cuda")
            prompts = a.prompt or ["a photo of a cat"]
            (out).mkdir(parents=True, exist_ok=True)
            (out / "prompts.json").write_text(json.dumps(prompts))
            for i, p in enumerate(prompts):
                ids = pipe.tokenizer(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
                ids2 = pipe.tokenizer_2(p, padding="max_length", max_length=77, truncation=True, return_tensors="pt").input_ids
                emb, _, pooled, _ = pipe.encode_prompt(p, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
                rc.save(out, f"te{i}_ids_l", ids[0])
                rc.save(out, f"te{i}_ids_g", ids2[0])
                rc.save(out, f"te{i}_ctx", emb[0])
                rc.save(out, f"te{i}_pooled", pooled[0])
        elif a.stage == "unet":
            pipe.unet.to("cuda")
            g = torch.Generator().manual_seed(a.seed)
            x = torch.randn(1, 4, h // 8, w // 8, generator=g)
            ctx = load(out, "te0_ctx")[None].cuda()
            pooled = load(out, "te0_pooled")[None].cuda()
            time_ids = torch.tensor([[h, w, 0, 0, h, w]], dtype=torch.float32, device="cuda")
            eps = pipe.unet(x.cuda(), torch.tensor([a.t], device="cuda"), encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
            (out / "unet.json").write_text(json.dumps({"size": [w, h], "t": a.t}))
            rc.save(out, "unet_x", x[0])
            rc.save(out, "unet_eps", eps[0])
        elif a.stage == "full":
            # ComfyUI's KSampler(euler, normal) over diffusers' UNet: prompt 0 against prompt 1 as the negative
            pipe.unet.to("cuda")
            pipe.vae.to("cuda")
            betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
            acp = torch.cumprod(1 - betas, 0)
            sig = ((1 - acp) / acp) ** 0.5
            log_sig = sig.float().log()
            ts = torch.linspace(999, 0, a.steps)
            def sigma_at(t):
                lo, hi, w_ = int(t.floor()), int(t.ceil()), float(t - t.floor())
                return float(((1 - w_) * log_sig[lo] + w_ * log_sig[hi]).exp())
            sigmas = [sigma_at(t) for t in ts] + [0.0]
            g = torch.Generator().manual_seed(a.seed)
            x = torch.randn(1, 4, h // 8, w // 8, generator=g).cuda() * (1 + sigmas[0] ** 2) ** 0.5
            ctx = torch.stack([load(out, "te0_ctx"), load(out, "te1_ctx")]).cuda()
            pooled = torch.stack([load(out, "te0_pooled"), load(out, "te1_pooled")]).cuda()
            time_ids = torch.tensor([[h, w, 0, 0, h, w]] * 2, dtype=torch.float32, device="cuda")
            for i in range(a.steps):
                s, sn = sigmas[i], sigmas[i + 1]
                t = (log_sig.cuda() - torch.tensor(s).log().cuda()).abs().argmin().float()
                eps = pipe.unet(torch.cat([x, x]) / (s * s + 1) ** 0.5, t, encoder_hidden_states=ctx, added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids}).sample
                e = eps[1] + a.cfg * (eps[0] - eps[1])
                den = x - s * e
                x = x + (x - den) / s * (sn - s)
            img = pipe.vae.decode(x / pipe.vae.config.scaling_factor).sample.clamp(-1, 1)
            arr = ((img[0] + 1) * 127.5).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            from PIL import Image
            Image.fromarray(arr).save(out / "full.png")
        else:
            pipe.vae.to("cuda")
            g = torch.Generator().manual_seed(a.seed)
            # a smooth latent with the statistics of a real one: low-frequency noise at the scaled latent's spread
            lat = torch.nn.functional.interpolate(torch.randn(1, 4, h // 64, w // 64, generator=g), size=(h // 8, w // 8), mode="bilinear") * 0.9
            img = pipe.vae.decode(lat.cuda() / pipe.vae.config.scaling_factor).sample.clamp(-1, 1)
            (out / "vae.json").write_text(json.dumps({"size": [w, h]}))
            rc.save(out, "vae_lat", lat[0])
            rc.save(out, "vae_rgb", ((img[0] + 1) * 127.5).permute(1, 2, 0))
            enc = pipe.vae.encode(img).latent_dist.mean * pipe.vae.config.scaling_factor
            rc.save(out, "vae_enc", enc[0])


if __name__ == "__main__":
    main()
