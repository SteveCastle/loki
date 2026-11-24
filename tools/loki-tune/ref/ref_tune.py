"""Reference gradients for loki-tune: one training step through the official MOSSImageTransformer (pipeline_moss_image.py,
bf16 weights, torch autograd), with the LoRA attached to every block linear as a forward hook, on the inputs and LoRA
weights `tune_check step --dump DIR` wrote. Compares the loss and the LoRA gradients with the engine's.

  tools/ref-env/Scripts/python -I ../loki-tune/ref/ref_tune.py DIR          (from tools/loki-dream, bf16 Kromoss in LOKI_MODELS)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

DREAM_REF = Path(__file__).resolve().parents[2] / "loki-dream" / "ref"
_spec = importlib.util.spec_from_file_location("ref_common", DREAM_REF / "ref_common.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

NAMES = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


def main():
    d = Path(sys.argv[1])
    meta = json.loads((d / "step.json").read_text())
    h, w, sigma, rank, alpha, tl = meta["h"], meta["w"], meta["sigma"], meta["rank"], meta["alpha"], meta["text_len"]
    load = lambda n, shape: torch.from_numpy(np.fromfile(d / f"{n}.bin", dtype=np.float32).reshape(shape))
    ctx = load("ctx", (tl, 4096)).cuda()
    x0 = load("x0", (h * w, 128)).cuda()
    noise = load("noise", (h * w, 128)).cuda()
    lora = rc.read_safetensors(d / "lora.safetensors", device="cuda")
    eng = rc.read_safetensors(d / "grads.safetensors")

    mod = rc.load_pipeline_module()
    model = rc.load_transformer(mod, rc.models_dir() / "kromoss_bf16.safetensors")
    model.requires_grad_(False)
    f32 = len(sys.argv) > 2 and sys.argv[2] == "f32"
    if f32:
        model = model.float()
    sc = alpha / rank
    params = {}
    hooks = []
    for i, blk in enumerate(model.layers):
        for name in NAMES:
            pre = f"diffusion_model.layers.{i}.proj.{name}"
            A = lora[pre + ".lora_down.weight"].float().clone().requires_grad_(True)
            B = lora[pre + ".lora_up.weight"].float().clone().requires_grad_(True)
            params[pre] = (A, B)

            def hook(m, args, out, A=A, B=B):
                x = args[0]
                return out + ((x.float() @ A.t()) @ B.t() * sc).to(out.dtype)
            hooks.append(blk.proj[name].register_forward_hook(hook))

    dt = torch.float32 if f32 else torch.bfloat16
    xt = ((1 - sigma) * x0 + sigma * noise).to(torch.bfloat16).to(dt)
    hs = xt.reshape(h, w, 128).permute(2, 0, 1)[None]
    t = torch.tensor([sigma * 1000.0], dtype=torch.float32, device="cuda")
    v = model(hs, t, ctx.to(torch.bfloat16).to(dt)[None]).sample.float()[0]  # [128, h, w]
    v = v.permute(1, 2, 0).reshape(h * w, 128)
    loss = ((v - (noise - x0)) ** 2).mean()
    loss.backward()
    print(f"loss: reference {loss.item():.5f}  engine {meta['loss']:.5f}")

    tag = "f32" if f32 else "bf16"
    torch.save({k: (A.grad.cpu(), B.grad.cpu()) for k, (A, B) in params.items()}, d / f"ref_grads_{tag}.pt")
    other = d / f"ref_grads_{'bf16' if f32 else 'f32'}.pt"
    if other.exists():
        o = torch.load(other)
        a = torch.cat([torch.cat([o[k][0].flatten(), o[k][1].flatten()]) for k in params]).double()
        b = torch.cat([torch.cat([params[k][0].grad.flatten().cpu(), params[k][1].grad.flatten().cpu()]) for k in params]).double()
        print(f"  reference bf16 vs f32: cos {(a @ b / (a.norm() * b.norm())).item():.5f}")
        for i in (0, 20, 41):
            ks = [k for k in params if f".layers.{i}." in k]
            a = torch.cat([torch.cat([o[k][0].flatten(), o[k][1].flatten()]) for k in ks]).double()
            b = torch.cat([torch.cat([params[k][0].grad.flatten().cpu(), params[k][1].grad.flatten().cpu()]) for k in ks]).double()
            print(f"    block {i}: cos {(a @ b / (a.norm() * b.norm())).item():.5f}")

    def cmp(label, keys):
        a = torch.cat([eng[k].flatten() for k in keys]).double()
        b = torch.cat([params[k.rsplit('.', 2)[0]][0 if '.lora_down.' in k else 1].grad.flatten().cpu() for k in keys]).double()
        cos = (a @ b / (a.norm() * b.norm())).item()
        rel = ((a - b).norm() / b.norm()).item()
        print(f"  {label:<28} cos {cos:.5f}  rel {rel:.3e}  |ref| {b.norm().item():.3e}  |eng| {a.norm().item():.3e}")

    for name in NAMES:
        for part in ("lora_down", "lora_up"):
            cmp(f"{name}.{part}", [f"diffusion_model.layers.{i}.proj.{name}.{part}.weight" for i in range(42)])
    for i in (0, 20, 41):
        cmp(f"block {i}", [f"diffusion_model.layers.{i}.proj.{n}.{p}.weight" for n in NAMES for p in ("lora_down", "lora_up")])


if __name__ == "__main__":
    main()
