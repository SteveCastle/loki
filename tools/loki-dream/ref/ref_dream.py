"""Dump reference tensors from the official MOSS-Image pipeline code (pipeline_moss_image.py) for loki-dream's stage checks.

  python -I ref/ref_dream.py tokens OUT -p "a cat" -p "..."      tokenizer ids + chat-templated text
  python -I ref/ref_dream.py te     OUT -p "a cat" [-n ""]        Qwen3.5 last_hidden_state (+ a few layers) per prompt
  python -I ref/ref_dream.py sched  OUT                           MOSSImageScheduler sigmas/timesteps for several step counts
  python -I ref/ref_dream.py dit    OUT --size 512x512 --t 800    one transformer forward (cond + uncond with layer 22 skipped)
  python -I ref/ref_dream.py vae    OUT [--image IMG]             VAE decode of a latent, encode of an image (img2img path)
  python -I ref/ref_dream.py full   OUT -p "a cat" --steps 8 --cfg 3 [--lora FILE]   whole pipeline at a fixed seed

`dit` and `full` read the text features written by `te` in the same OUT folder (so the 18 GB text encoder and the
10 GB transformer never share the GPU). Each dump is OUT/<name>.bin (f32 / i32 little endian) + OUT/index.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import importlib.util, sys
_spec = importlib.util.spec_from_file_location("ref_common", Path(__file__).resolve().parent / "ref_common.py")
rc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rc)


def template(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)


def cmd_tokens(a):
    tok = rc.load_tokenizer()
    cases = []
    for p in a.prompt:
        text = template(tok, p)
        ids = tok(text)["input_ids"]
        plain = tok(p)["input_ids"]
        cases.append({"prompt": p, "template": text, "ids": ids, "plain_ids": plain})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "tokens.json").write_text(json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cases:
        print(len(c["ids"]), repr(c["template"][:120]))


def load_prompts(a):
    prompts = list(a.prompt) + [n for n in (a.negative or [])]
    return prompts


def cmd_te(a):
    tok = rc.load_tokenizer()
    te = rc.load_text_encoder()
    out = Path(a.out)
    meta = []
    for i, p in enumerate(load_prompts(a)):
        text = template(tok, p)
        inputs = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = te(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
        rc.save(out, f"te{i}_ids", inputs["input_ids"][0])
        rc.save(out, f"te{i}_last", o.last_hidden_state[0])
        hs = o.hidden_states  # embeddings + after each layer
        for l in range(31):  # hidden_states[32] is already final-normed
            rc.save(out, f"te{i}_h{l}", hs[l + 1][0])
        rc.save(out, f"te{i}_emb", hs[0][0])
        meta.append({"prompt": p, "len": int(inputs["input_ids"].shape[1])})
        print(f"te{i}: {inputs['input_ids'].shape[1]} tokens, last |x| {o.last_hidden_state.float().abs().mean():.4f}")
    (out / "te.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def cmd_te32(a):
    """fp32 CPU run of the whole text encoder (bf16 weights, fp32 math): the numerical truth both bf16 engines approximate."""
    tok = rc.load_tokenizer()
    te = rc.load_text_encoder(device="cpu").float()
    out = Path(a.out)
    for i, p in enumerate(load_prompts(a)):
        inputs = tok(template(tok, p), return_tensors="pt")
        with torch.no_grad():
            o = te(**inputs, use_cache=False, return_dict=True, output_hidden_states=True)
        rc.save(out, f"te{i}_last32", o.last_hidden_state[0])
        for l in range(31):
            rc.save(out, f"te{i}_h{l}_32", o.hidden_states[l + 1][0])
        print(f"te{i} fp32 done", flush=True)


def cmd_sched(a):
    mod = rc.load_pipeline_module()
    out = Path(a.out)
    for n in (1, 4, 8, 20, 28, 50):
        s = mod.MOSSImageScheduler()
        s.set_timesteps(n)
        rc.save(out, f"sigmas_{n}", s.sigmas)
        rc.save(out, f"timesteps_{n}", s.timesteps)
        print(n, [round(x, 5) for x in s.sigmas.tolist()][:10], s.timesteps.tolist()[:4])


def te_rows(out, i):
    idx = json.loads((out / "index.json").read_text())
    name = f"te{i}_last"
    shape = idx[name]["shape"]
    return torch.from_numpy(np.fromfile(out / f"{name}.bin", dtype=np.float32).reshape(shape))


def parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def cmd_dit(a):
    mod = rc.load_pipeline_module()
    out = Path(a.out)
    model = rc.load_transformer(mod, rc.models_dir() / "kromoss_bf16.safetensors")
    w, h = parse_size(a.size)
    g = torch.Generator("cpu").manual_seed(a.seed)
    lat = torch.randn((1, 128, h // 16, w // 16), generator=g, dtype=torch.float32)
    rc.save(out, "dit_latent", lat[0])
    cond = te_rows(out, 0).to("cuda", torch.bfloat16)[None]
    unc = te_rows(out, a.neg_index).to("cuda", torch.bfloat16)[None]
    t = torch.tensor([a.t], dtype=torch.float32, device="cuda")
    caps = {}

    def hook(name):
        def f(m, args, o):
            caps[name] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()
        return f
    hs = [model.text_embedder.register_forward_hook(hook("txt")), model.layers[0].register_forward_hook(hook("blk0")),
          model.layers[21].register_forward_hook(hook("blk21")), model.layers[41].register_forward_hook(hook("blk41"))]
    with torch.no_grad():
        o = model(lat.to("cuda"), t, cond).sample.float()
    for hh in hs:
        hh.remove()
    for k, v in caps.items():
        rc.save(out, f"dit_{k}", v[0])
    rc.save(out, "dit_out_cond", o[0])
    with torch.no_grad():
        u = model(lat.to("cuda"), t, unc, skip_layers=[22]).sample.float()
    rc.save(out, "dit_out_uncond_skip22", u[0])
    (out / "dit.json").write_text(json.dumps({"t": a.t, "w": w, "h": h, "seed": a.seed, "neg_index": a.neg_index}))
    print("dit out", o.abs().mean().item(), u.abs().mean().item())


def vae_decode(vae, z):
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    z = z.to(vae.dtype)
    z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
    b, c, h, w = z.shape
    z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    return vae.decode(z, return_dict=False)[0].clamp(-1, 1)


def vae_encode(vae, img):
    """img [1,3,H,W] in [-1,1] -> normalized packed latent [1,128,H/16,W/16] (distribution mode, as diffusers' Flux2 img2img)."""
    lat = vae.encode(img.to(vae.dtype)).latent_dist.mode()
    b, c, h, w = lat.shape
    lat = lat.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
    bn = vae.bn
    eps = vae.config.batch_norm_eps
    return (lat - bn.running_mean.view(1, -1, 1, 1)) / (bn.running_var + eps).sqrt().view(1, -1, 1, 1)


def cmd_vae(a):
    from PIL import Image
    out = Path(a.out)
    vae = rc.load_vae()
    g = torch.Generator("cpu").manual_seed(7)
    if a.latent:
        idx = json.loads((out / "index.json").read_text())
        z = torch.from_numpy(np.fromfile(out / f"{a.latent}.bin", dtype=np.float32).reshape(idx[a.latent]["shape"]))[None]
    else:
        z = torch.randn((1, 128, 16, 24), generator=g) * 0.8
    rc.save(out, "vae_z", z[0])
    with torch.no_grad():
        img = vae_decode(vae, z.cuda()).float()
    rc.save(out, "vae_dec", img[0])
    Image.fromarray(((img[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)).save(out / "vae_dec.png")
    if a.image:
        im = Image.open(a.image).convert("RGB")
        W, H = im.size
        W, H = W // 16 * 16, H // 16 * 16
        im = im.crop((0, 0, W, H))
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float() / 127.5 - 1
        rc.save(out, "vae_enc_in", x[0])
        with torch.no_grad():
            lat = vae_encode(vae, x.cuda()).float()
        rc.save(out, "vae_enc", lat[0])
        print("encoded", tuple(lat.shape), lat.abs().mean().item())
    print("decoded", tuple(img.shape))


def apply_lora(transformer, path, strength=1.0):
    """The Kromoss README's unmerged LoRA hook (fp32 delta added to the bf16 output)."""
    from safetensors.torch import load_file
    state = load_file(path, device="cuda")
    modules = dict(transformer.named_modules())
    n = 0
    for key in [k for k in state if k.endswith(".lora_down.weight")]:
        name = key[len("diffusion_model."):-len(".lora_down.weight")]
        down, up = state[key].float(), state[key.replace("lora_down", "lora_up")].float()
        scale = strength * state[f"diffusion_model.{name}.alpha"].item() / down.shape[0]

        def hook(module, args, out, down=down, up=up, scale=scale):
            return out + (scale * (args[0].float() @ down.T) @ up.T).to(out.dtype)
        modules[name].register_forward_hook(hook)
        n += 1
    print(f"lora: {n} modules")


def cmd_full(a):
    from PIL import Image
    if a.sdpa == "math":  # perturbation baseline: same math, another attention kernel
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    mod = rc.load_pipeline_module()
    out = Path(a.out)
    w, h = parse_size(a.size)
    model = rc.load_transformer(mod, rc.models_dir() / "kromoss_bf16.safetensors")
    if a.lora:
        apply_lora(model, a.lora, a.lora_scale)
    sched = mod.MOSSImageScheduler()
    pipe = mod.MOSSImagePipeline(model, sched)
    cond = te_rows(out, 0).to("cuda", torch.bfloat16)[None]
    unc = te_rows(out, a.neg_index).to("cuda", torch.bfloat16)[None]
    g = torch.Generator("cpu").manual_seed(a.seed)
    lat = torch.randn((1, 128, h // 16, w // 16), generator=g, dtype=torch.float32)
    tag = a.tag
    rc.save(out, f"{tag}_noise", lat[0])
    skip = [] if a.no_skip else [22]
    with torch.no_grad():
        res = pipe(prompt_embeds=cond, negative_prompt_embeds=unc, height=h, width=w, num_inference_steps=a.steps,
                   guidance_scale=a.cfg, latents=lat.clone(), output_type="latent", uncond_skip_layers=skip).images
    rc.save(out, f"{tag}_latent", res[0].float())
    (out / f"{tag}.json").write_text(json.dumps({"w": w, "h": h, "steps": a.steps, "cfg": a.cfg, "seed": a.seed,
                                                 "neg_index": a.neg_index, "no_skip": a.no_skip, "lora": a.lora}))
    del pipe, model
    rc.free_cuda()
    vae = rc.load_vae()
    with torch.no_grad():
        img = vae_decode(vae, res.float())
    arr = ((img[0].float().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(out / f"{tag}.png")
    print("wrote", out / f"{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("out")
    ap.add_argument("-p", "--prompt", action="append", default=[])
    ap.add_argument("-n", "--negative", action="append", default=None)
    ap.add_argument("--size", default="512x512")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t", type=float, default=800.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--neg-index", type=int, default=1)
    ap.add_argument("--no-skip", action="store_true")
    ap.add_argument("--lora")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--image")
    ap.add_argument("--latent")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--sdpa", default="default")
    a = ap.parse_args()
    {"tokens": cmd_tokens, "te": cmd_te, "te32": cmd_te32, "sched": cmd_sched, "dit": cmd_dit, "vae": cmd_vae, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
