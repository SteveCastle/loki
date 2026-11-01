"""Reference dumps for loki-dream's Krea 2, FLUX.2 klein and Anima paths, from the official diffusers implementations
(diffusers 0.42: Krea2Pipeline / Krea2Transformer2DModel, Flux2KleinPipeline / Flux2Transformer2DModel, the Anima modular
pipeline with CosmosTransformer3DModel + AnimaTextConditioner) and transformers' Qwen3 / Qwen3-VL text models, loaded from
the same ComfyUI single files the native engine reads (fp8 weights dequantized to bf16 here).

  python -I ref/ref_arch.py te   ARCH OUT --dit FILE --te FILE -p "a cat" [-n ""]   text conditioning per prompt
  python -I ref/ref_arch.py dit  ARCH OUT --dit FILE --size 768x512 --sigma 0.8 --seed 1   one forward (cond + uncond)
  python -I ref/ref_arch.py vae  ARCH OUT --vae FILE [--image IMG]                  decode a latent / encode an image
  python -I ref/ref_arch.py full ARCH OUT --dit FILE --te FILE --vae FILE -p ... --steps 8 --cfg 1 --seed 42 --tag t

ARCH is krea2, flux2 or anima. `dit` reads the conditioning `te` wrote into the same OUT folder. Configs and tokenizers come
from ref_src/ (ref_src/anima = circlestone-labs/Anima-Base-v1.0-Diffusers, ref_src/klein4b = black-forest-labs/FLUX.2-klein-4B
json + tokenizer, ref_src/qwen3vl4b = Qwen/Qwen3-VL-4B-Instruct json + tokenizer). Dumps are OUT/<name>.bin + index.json.
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

SRC = rc.ROOT / "ref_src"
PREFIXES = ("model.diffusion_model.", "diffusion_model.")


def dequant(sd: dict) -> dict:
    """ComfyUI scaled fp8 / plain fp8 -> bf16; drop quantization side tensors and model prefixes."""
    out = {}
    for k, v in sd.items():
        if k.endswith((".weight_scale", ".input_scale", ".comfy_quant", ".scale_weight", ".scaled_fp8")):
            continue
        if v.dtype == torch.float8_e4m3fn:
            base = k[: -len(".weight")] if k.endswith(".weight") else k
            s = sd.get(base + ".weight_scale", sd.get(base + ".scale_weight"))
            v = v.to(torch.float32) * (s.to(torch.float32) if s is not None else 1.0)
        elif v.dtype == torch.int8 or v.dtype == torch.uint8:
            raise SystemExit(f"{k}: int8 / nvfp4 checkpoints cannot be used as the reference; use a bf16 or fp8 file")
        for p in PREFIXES:
            if k.startswith(p):
                k = k[len(p):]
        out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
    return out


def cfg_json(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text()).items() if not k.startswith("_")}


# ---------------------------------------------------------------------------------------------------- text encoders
def load_qwen3(te_path: str, arch: str, device="cuda"):
    from transformers import AutoConfig, AutoModel
    sd = dequant(rc.read_safetensors(Path(te_path)))
    if arch == "krea2":
        from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
        full = json.loads((SRC / "qwen3vl4b" / "config.json").read_text())
        cfg = Qwen3VLTextConfig(**full["text_config"])
        model = Qwen3VLTextModel(cfg)
    else:
        d = "anima/text_encoder" if arch == "anima" else "klein4b/text_encoder"
        cfg = AutoConfig.from_pretrained(str(SRC / d))
        hidden = next(v.shape[1] for k, v in sd.items() if k.endswith("embed_tokens.weight"))
        if hidden != cfg.hidden_size:  # klein 9B: Qwen3-8B
            cfg = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
        model = AutoModel.from_config(cfg)
    want = model.state_dict()
    mapped = {}
    for k, v in sd.items():
        for cand in (k, k.removeprefix("model."), k.removeprefix("model.language_model."), k.replace("model.language_model.", "")):
            if cand in want:
                mapped[cand] = v
                break
    missing = [k for k in want if k not in mapped]
    if missing:
        raise SystemExit(f"text encoder: missing {missing[:5]}")
    model.load_state_dict(mapped, strict=True)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    # .to(bf16) also casts the rotary inv_freq buffers; from_pretrained(torch_dtype=bf16) keeps them in fp32
    for m in model.modules():
        if hasattr(m, "inv_freq") and hasattr(m, "compute_default_rope_parameters"):
            inv, _ = m.compute_default_rope_parameters(m.config)
            m.inv_freq = inv.to(device=device, dtype=torch.float32)
            m.original_inv_freq = m.inv_freq.clone()
    return model


def tokenizer(arch: str, which="qwen"):
    from transformers import AutoTokenizer
    if arch == "anima" and which == "t5":
        return AutoTokenizer.from_pretrained(str(SRC / "anima" / "t5_tokenizer"))
    d = {"krea2": "qwen3vl4b", "flux2": "klein4b/tokenizer", "anima": "anima/tokenizer"}[arch]
    return AutoTokenizer.from_pretrained(str(SRC / d))


def krea2_te(te, tok, prompt):
    """Krea2Pipeline.get_text_hidden_states for one prompt (max_sequence_length 512)."""
    prefix = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
              "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n"
    start, n_suffix, max_len = 34, 5, 512
    text_tokens = tok([prefix + prompt], truncation=True, padding="max_length", max_length=max_len + start - n_suffix, return_tensors="pt").to("cuda")
    suffix_tokens = tok([suffix], return_tensors="pt").to("cuda")
    ids = torch.cat([text_tokens.input_ids, suffix_tokens.input_ids], dim=1)
    mask = torch.cat([text_tokens.attention_mask, suffix_tokens.attention_mask], dim=1).bool()
    pos = (mask.long().cumsum(dim=-1) - 1).clamp(min=0).unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        o = te(input_ids=ids, attention_mask=mask, position_ids=pos, output_hidden_states=True)
    taps = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    hs = torch.stack([o.hidden_states[i] for i in taps], dim=2)[:, start:]
    return hs[0], mask[0, start:], ids[0]


def flux2_te(te, tok, prompt):
    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    e = Flux2KleinPipeline._get_qwen3_prompt_embeds(te, tok, prompt, dtype=torch.bfloat16, device="cuda")
    return e[0]


def anima_te(te, tok, t5tok, conditioner, prompt):
    from diffusers.modular_pipelines.anima.encoders import AnimaTextEncoderStep

    class C:
        pass
    c = C()
    c.tokenizer, c.text_encoder, c.t5_tokenizer = tok, te, t5tok
    q, qm = AnimaTextEncoderStep._get_qwen_prompt_embeds(c, [prompt], 512, "cuda", torch.bfloat16)
    t5, t5m = AnimaTextEncoderStep._get_t5_prompt_ids(c, [prompt], 512, "cuda")
    with torch.no_grad():
        ctx = conditioner(source_hidden_states=q, target_input_ids=t5, target_attention_mask=t5m, source_attention_mask=qm)
    return q[0], qm[0], t5[0], ctx[0]


def load_conditioner(dit_path: str):
    from diffusers.models.condition_embedders.condition_embedder_anima import AnimaTextConditioner
    cond = AnimaTextConditioner(**cfg_json(SRC / "anima" / "text_conditioner" / "config.json"))
    sd = dequant(rc.read_safetensors(Path(dit_path)))
    sub = {k.removeprefix("net.llm_adapter."): v for k, v in sd.items() if k.startswith("net.llm_adapter.")}
    missing, unexpected = cond.load_state_dict(sub, strict=False)
    if missing or unexpected:
        raise SystemExit(f"conditioner keys: missing {missing[:5]} unexpected {unexpected[:5]}")
    return cond.to(device="cuda", dtype=torch.bfloat16).eval()


def cmd_te(a):
    out = Path(a.out)
    te = load_qwen3(a.te, a.arch)
    tok = tokenizer(a.arch)
    texts = list(a.prompt) + list(a.negative or [])
    meta = []
    if a.arch == "anima":
        cond = load_conditioner(a.dit)
        t5tok = tokenizer(a.arch, "t5")
    for i, p in enumerate(texts):
        if a.arch == "krea2":
            hs, mask, ids = krea2_te(te, tok, p)
            rc.save(out, f"te{i}_taps", hs)
            rc.save(out, f"te{i}_mask", mask.int())
            rc.save(out, f"te{i}_ids", ids)
        elif a.arch == "flux2":
            rc.save(out, f"te{i}_ctx", flux2_te(te, tok, p))
        else:
            q, qm, t5, ctx = anima_te(te, tok, t5tok, cond, p)
            rc.save(out, f"te{i}_qwen", q)
            rc.save(out, f"te{i}_qmask", qm)
            rc.save(out, f"te{i}_t5", t5)
            rc.save(out, f"te{i}_ctx", ctx)
        meta.append(p)
    (out / "prompts.json").write_text(json.dumps(meta, ensure_ascii=False))
    print("te done:", len(texts), "text(s)")


# ---------------------------------------------------------------------------------------------------- transformers
def load_transformer(arch: str, dit_path: str, device="cuda", offload=False, dtype=torch.bfloat16):
    from diffusers.loaders import single_file_utils as sfu
    sd = dequant(rc.read_safetensors(Path(dit_path)))
    if arch == "krea2":
        from diffusers import Krea2Transformer2DModel
        model = Krea2Transformer2DModel()
        sd = sfu.convert_krea2_transformer_checkpoint_to_diffusers(sd)
    elif arch == "flux2":
        from diffusers import Flux2Transformer2DModel
        cfg = cfg_json(SRC / "klein4b" / "transformer" / "config.json")
        d = sd["img_in.weight"].shape[0]
        cfg.update(num_attention_heads=d // 128, joint_attention_dim=sd["txt_in.weight"].shape[1],
                   num_layers=len({k.split(".")[1] for k in sd if k.startswith("double_blocks.")}),
                   num_single_layers=len({k.split(".")[1] for k in sd if k.startswith("single_blocks.")}))
        model = Flux2Transformer2DModel(**cfg)
        sd = sfu.convert_flux2_transformer_checkpoint_to_diffusers(sd)
    else:
        from diffusers import CosmosTransformer3DModel
        model = CosmosTransformer3DModel(**cfg_json(SRC / "anima" / "transformer" / "config.json"))
        sd = {k: v for k, v in sd.items() if not k.startswith("net.llm_adapter.")}
        sd = sfu.convert_cosmos_transformer_checkpoint_to_diffusers(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"transformer keys: missing {missing[:8]} unexpected {unexpected[:8]}")
    model = model.to(dtype=dtype).eval()
    if offload:
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(model, onload_device=torch.device(device), offload_type="block_level", num_blocks_per_group=1)
    else:
        model = model.to(device)
    return model


def parse_size(s):
    w, h = s.lower().split("x")
    return int(w), int(h)


def te_inputs(out: Path, arch: str, i: int, dump):
    if arch == "krea2":
        return torch.from_numpy(dump(f"te{i}_taps")).to("cuda", torch.bfloat16)[None], torch.from_numpy(dump(f"te{i}_mask")).to("cuda").bool()[None]
    return torch.from_numpy(dump(f"te{i}_ctx")).to("cuda", torch.bfloat16)[None], None


def loader(out: Path):
    idx = json.loads((out / "index.json").read_text())

    def f(name):
        e = idx[name]
        dt = np.int32 if e["dtype"] == "i32" else np.float32
        return np.fromfile(out / f"{name}.bin", dtype=dt).reshape(e["shape"])
    return f


def cmd_dit(a):
    out = Path(a.out)
    dump = loader(out)
    w, h = parse_size(a.size)
    dt = torch.float32 if a.fp32 else torch.bfloat16
    model = load_transformer(a.arch, a.dit, offload=a.offload, dtype=dt)
    g = torch.Generator("cpu").manual_seed(a.seed)
    sfx = "_32" if a.fp32 else ""
    sigma = a.sigma
    n_texts = len(json.loads((out / "prompts.json").read_text()))
    with torch.no_grad():
        if a.arch == "flux2":
            from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline as P
            lat = torch.randn((1, 128, h // 16, w // 16), generator=g)
            ids = P._prepare_latent_ids(lat).to("cuda")
            x = P._pack_latents(lat).to("cuda", torch.bfloat16)
            rc.save(out, "dit_x", x[0])
            t = torch.tensor([sigma * 1000.0]).to(torch.bfloat16).to("cuda")
            for i in range(n_texts):
                ctx, _ = te_inputs(out, a.arch, i, dump)
                tids = P._prepare_text_ids(ctx).to("cuda")
                v = model(hidden_states=x, timestep=t / 1000, guidance=None, encoder_hidden_states=ctx, txt_ids=tids, img_ids=ids, return_dict=False)[0]
                rc.save(out, f"dit_v{i}", v[0])
        elif a.arch == "krea2":
            from diffusers.pipelines.krea2.pipeline_krea2 import Krea2Pipeline as P
            lat = torch.randn((1, 16, h // 8, w // 8), generator=g)
            x = P._pack_latents(type("X", (), {"patch_size": 2})(), lat, 1, 16, h // 8, w // 8).to("cuda", torch.bfloat16)
            rc.save(out, "dit_x", x[0])
            x = x.to(dt)
            t = torch.tensor([sigma]).to(torch.bfloat16).to("cuda").to(dt)
            seen = {}
            model.txt_in.register_forward_hook(lambda m, args, o: seen.__setitem__("txt", o))
            model.text_fusion.register_forward_hook(lambda m, args, o: seen.__setitem__("fusion", o))
            for i in range(n_texts):
                taps, mask = te_inputs(out, a.arch, i, dump)
                pos = P.prepare_position_ids(taps.shape[1], h // 16, w // 16, "cuda")
                v = model(hidden_states=x, encoder_hidden_states=taps.to(dt), timestep=t, position_ids=pos, encoder_attention_mask=mask, return_dict=False)[0]
                rc.save(out, f"dit_v{i}{sfx}", v[0])
                rc.save(out, f"dit_txt{i}{sfx}", seen["txt"][0][mask[0]])
        else:
            lat = torch.randn((1, 16, 1, h // 8, w // 8), generator=g)
            rc.save(out, "dit_x5", lat[0, :, 0])
            x = lat.to("cuda", torch.bfloat16)
            t = (torch.tensor([sigma * 1000.0]).to(torch.bfloat16) / 1000).to("cuda")
            pad = torch.zeros(1, 1, h, w, device="cuda", dtype=torch.bfloat16)
            for i in range(n_texts):
                ctx, _ = te_inputs(out, a.arch, i, dump)
                v = model(hidden_states=x, timestep=t, encoder_hidden_states=ctx, padding_mask=pad, return_dict=False)[0]
                rc.save(out, f"dit_v{i}", v[0, :, 0])
    (out / "dit.json").write_text(json.dumps({"size": [w, h], "sigma": sigma, "seed": a.seed}))
    print("dit done")


# ---------------------------------------------------------------------------------------------------- VAE
def load_vae(arch: str, vae_path: str, device="cuda", dtype=torch.float32):
    from diffusers.loaders import single_file_utils as sfu
    sd = rc.read_safetensors(Path(vae_path))
    if arch == "flux2":
        from diffusers import AutoencoderKLFlux2
        import inspect
        ok = inspect.signature(AutoencoderKLFlux2.__init__).parameters
        vae = AutoencoderKLFlux2(**{k: v for k, v in cfg_json(SRC / "klein4b" / "vae" / "config.json").items() if k in ok})
    else:
        from diffusers import AutoencoderKLQwenImage
        vae = AutoencoderKLQwenImage(**cfg_json(SRC / "anima" / "vae" / "config.json"))
        sd = sfu.convert_wan_vae_to_diffusers(sd)
    missing, unexpected = vae.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"vae keys: missing {missing[:5]} unexpected {unexpected[:5]}")
    return vae.to(device=device, dtype=dtype).eval()


def cmd_vae(a):
    out = Path(a.out)
    vae = load_vae(a.arch, a.vae)
    g = torch.Generator("cpu").manual_seed(a.seed)
    w, h = parse_size(a.size)
    with torch.no_grad():
        if a.arch == "flux2":
            z = torch.randn((1, 32, h // 8, w // 8), generator=g)
            img = vae.decode(z.to("cuda"), return_dict=False)[0]
        else:
            z = torch.randn((1, 16, 1, h // 8, w // 8), generator=g)
            img = vae.decode(z.to("cuda"), return_dict=False)[0][:, :, 0]
        rc.save(out, "vae_z", z.reshape(z.shape[1], h // 8, w // 8))
        rc.save(out, "vae_img", img[0].clamp(-1, 1))
        if a.image:
            from PIL import Image
            im = Image.open(a.image).convert("RGB")
            x = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1)[None] / 127.5 - 1
            rc.save(out, "vae_in", x[0])
            if a.arch == "flux2":
                mean = vae.encode(x.to("cuda")).latent_dist.mode()
            else:
                mean = vae.encode(x[:, :, None].to("cuda")).latent_dist.mode()[:, :, 0]
            rc.save(out, "vae_enc", mean[0])
    print("vae done")


# ---------------------------------------------------------------------------------------------------- full pipelines
def cmd_full(a):
    out = Path(a.out)
    w, h = parse_size(a.size)
    prompt = a.prompt[0]
    neg = (a.negative or [""])[0]
    g = torch.Generator("cpu").manual_seed(a.seed)
    from diffusers import FlowMatchEulerDiscreteScheduler
    if a.arch == "anima":
        from diffusers import ModularPipeline
        from diffusers.guiders import ClassifierFreeGuidance
        pipe = ModularPipeline.from_pretrained(str(SRC / "anima"))
        pipe.load_components(torch_dtype=torch.bfloat16)
        if a.dit:
            pipe.update_components(transformer=load_transformer("anima", a.dit, device="cpu"), text_conditioner=load_conditioner(a.dit).cpu())
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=a.cfg))
        pipe.to("cuda")
        res = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=a.steps, height=h, width=w, generator=g, output_type="np", output="images")
        img = torch.from_numpy(np.asarray(res[0] if isinstance(res, list) else res)).float().reshape(h, w, 3)
    else:
        te = load_qwen3(a.te, a.arch, device="cuda")
        tok = tokenizer(a.arch)
        tr = load_transformer(a.arch, a.dit, offload=True)
        vae = load_vae(a.arch, a.vae, device="cuda", dtype=torch.bfloat16)
        if a.arch == "krea2":
            from diffusers import Krea2Pipeline
            sch = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15, base_image_seq_len=256, max_image_seq_len=6400)
            pipe = Krea2Pipeline(scheduler=sch, vae=vae, text_encoder=te, tokenizer=tok, transformer=tr, is_distilled=a.distilled)
            res = pipe(prompt=prompt, negative_prompt=neg if a.cfg != 1 else None, height=h, width=w, num_inference_steps=a.steps, guidance_scale=a.cfg - 1, generator=g, output_type="np")
        else:
            from diffusers import Flux2KleinPipeline
            sch = FlowMatchEulerDiscreteScheduler(**cfg_json(SRC / "klein4b" / "scheduler" / "scheduler_config.json"))
            pipe = Flux2KleinPipeline(scheduler=sch, vae=vae, text_encoder=te, tokenizer=tok, transformer=tr, is_distilled=a.distilled)
            res = pipe(prompt=prompt, height=h, width=w, num_inference_steps=a.steps, guidance_scale=a.cfg, generator=g, output_type="np")
        img = torch.from_numpy(res.images[0]).float()
    from PIL import Image
    arr = (img.numpy().clip(0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(arr).save(out / f"full_{a.tag}.png")
    (out / f"full_{a.tag}.json").write_text(json.dumps({"prompt": prompt, "negative": neg, "steps": a.steps, "cfg": a.cfg, "seed": a.seed, "size": [w, h], "distilled": a.distilled}))
    print("full done", out / f"full_{a.tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arch", choices=["krea2", "flux2", "anima"])
    ap.add_argument("out")
    ap.add_argument("-p", "--prompt", action="append", default=[])
    ap.add_argument("-n", "--negative", action="append", default=None)
    ap.add_argument("--dit")
    ap.add_argument("--te")
    ap.add_argument("--vae")
    ap.add_argument("--size", default="768x512")
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--distilled", action="store_true")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--image")
    ap.add_argument("--tag", default="full")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"te": cmd_te, "dit": cmd_dit, "vae": cmd_vae, "full": cmd_full}[a.cmd](a)


if __name__ == "__main__":
    main()
