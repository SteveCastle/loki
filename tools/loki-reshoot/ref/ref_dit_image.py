"""Reference dump of ComfyUI's MiniMax-H3-Image DiT (MiniMaxH3ImageModel.forward) on a small synthetic case.

Usage (cwd = a ComfyUI >= 0.45 root, embedded python; LOKI_PYLIB = a folder with comfy-kitchen >= 0.2.41):
    python ref_dit_image.py <out_dir> [W=512] [H=384]

Writes into out_dir:
    text.bin        f32 [L, 5120]   (bf16-representable values; fed to the model as bf16)
    tags.bin        u8  [L]
    x_in.bin        f32 [24, 1, H/16, W/16]  target image latent
    ref<i>.bin      f32 [24, 1, h, w]        reference image latents
    schedule.json   the sampler's sigmas (simple scheduler, 6 steps, the checkpoint's PDD grid)
    out_<s>.bin     model output (negated velocity) at sigma s
    h_<s>_<tag>.bin hidden state snapshots
    meta.json
"""
import sys, os, json, time
if os.environ.get("LOKI_PYLIB"):
    sys.path.insert(0, os.environ["LOKI_PYLIB"])
sys.path.insert(0, os.getcwd())
import torch

import comfy.sd
import comfy.samplers
import comfy.model_management

MODEL = os.environ.get("LOKI_H3I_DIT", r"C:\Users\steph\bin\models\minimax_h3_image_pruned_int8_convrot.safetensors")
# REF_HP=1: high-precision reference (no activation quantization: rotated activation x dequantized int8 weight in fp32;
# exact attention instead of the checkpoint's comfy_kitchen_int8)
HP = os.environ.get("REF_HP") == "1"

if HP:
    import comfy.quant_ops
    from comfy_kitchen.backends._activations import apply_input_act, apply_residual
    from comfy_kitchen.tensor.int8_utils import _build_hadamard, _rotate_activation

    def int8_linear_w8a16(x, weight, weight_scale, bias=None, out_dtype=torch.bfloat16, convrot=False, convrot_groupsize=256,
                          input_act=None, input_act_weight=None, input_act_eps=0.0, residual=None, residual_scale=None):
        x = apply_input_act(x, input_act, input_act_weight, input_act_eps)
        xf = x.float()
        if convrot:
            h = _build_hadamard(convrot_groupsize, device=x.device, dtype=torch.float32)
            xf = _rotate_activation(xf, h, convrot_groupsize)
        w = weight.to(x.device).float() * weight_scale.to(x.device, torch.float32).reshape(-1, 1)
        out = torch.nn.functional.linear(xf, w).to(out_dtype)
        if bias is not None:
            out = out + bias.to(out.device, out.dtype)
        return apply_residual(out, residual, residual_scale)

    comfy.quant_ops.ck.int8_linear = int8_linear_w8a16
    torch.backends.cuda.matmul.allow_tf32 = False


def dump(path, t):
    t.detach().float().cpu().contiguous().numpy().tofile(path)


@torch.no_grad()
def main():
    out_dir = sys.argv[1]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    H = int(sys.argv[3]) if len(sys.argv) > 3 else 384
    os.makedirs(out_dir, exist_ok=True)
    lh, lw = H // 16, W // 16

    g = torch.Generator("cpu").manual_seed(1234)
    L = 60
    text = (torch.randn(L, 5120, generator=g) * 2.0).to(torch.bfloat16)
    tags = torch.ones(L, dtype=torch.long)
    tags[10:30] = 0  # one vision span
    x = torch.randn(1, 24, 1, lh, lw, generator=g)
    refs = []
    for (rh, rw) in ((8, 14), (12, 10)):
        z = torch.randn(1, 24, 1, rh, rw, generator=g)
        refs.append({"kind": "image", "latent_h": rh, "latent_w": rw, "latent": z})

    model = comfy.sd.load_diffusion_model(MODEL)
    comfy.model_management.load_models_gpu([model])
    dev = model.load_device
    dm = model.model.diffusion_model
    if HP:
        for blk in dm.blocks:
            blk.attn.comfy_attention.function = None
            blk.attn.comfy_attention.method = None
    dtype = model.model.get_dtype_inference() if hasattr(model.model, "get_dtype_inference") else torch.bfloat16
    ms = model.get_model_object("model_sampling")
    sig = comfy.samplers.calculate_sigmas(ms, "simple", 6).tolist()
    print("compute dtype", dtype, "pdd", dm.final_layer.pdd_steps, dm.pdd_sigmas, "schedule", sig)
    json.dump({"simple_6": sig, "simple_8": comfy.samplers.calculate_sigmas(ms, "simple", 8).tolist(),
               "simple_4": comfy.samplers.calculate_sigmas(ms, "simple", 4).tolist(), "pdd_sigmas": dm.pdd_sigmas},
              open(os.path.join(out_dir, "schedule.json"), "w"))

    payload = {
        "text_token_tags": tags,
        "refs": refs,
        "cond_video_latents": [r["latent"] for r in refs],
        "cond_audio_latents": [],
        "visual_cond_noise_aug": 1.0,
        "seed": 0,
    }
    ctx = text[None].to(dev, dtype)

    snaps = {}
    def pre0(mod, args, kwargs=None):
        snaps["in"] = args[0].clone()
    hooks = [dm.blocks[0].register_forward_pre_hook(pre0)]
    for bi in (0, 1, len(dm.blocks) // 2, len(dm.blocks) - 1):
        def mk(bi):
            def f(mod, args, out):
                snaps[f"b{bi}"] = out.clone()
            return f
        hooks.append(dm.blocks[bi].register_forward_hook(mk(bi)))

    dump(os.path.join(out_dir, "text.bin"), text)
    tags.to(torch.uint8).numpy().tofile(os.path.join(out_dir, "tags.bin"))
    dump(os.path.join(out_dir, "x_in.bin"), x[0])
    for i, r in enumerate(refs):
        dump(os.path.join(out_dir, f"ref{i}.bin"), r["latent"][0])

    seq_len = None
    sample_sigmas = torch.tensor(sig)
    evals = [sig[0], sig[3]]
    for s in evals:
        snaps.clear()
        torch.cuda.synchronize()
        t0 = time.time()
        out = dm.forward(x.to(dev), torch.tensor([s * 1000.0], device=dev), ctx,
                         transformer_options={"sample_sigmas": sample_sigmas}, minimax_payload=dict(payload))
        torch.cuda.synchronize()
        print(f"sigma {s}: forward {time.time() - t0:.2f}s", tuple(out.shape), float(out.float().std()))
        dump(os.path.join(out_dir, f"out_{s:.6f}.bin"), out[0])
        for k, v in snaps.items():
            dump(os.path.join(out_dir, f"h_{s:.6f}_{k}.bin"), v)
            seq_len = v.shape[0]
    for h in hooks:
        h.remove()
    json.dump({"text_len": L, "latent_h": lh, "latent_w": lw, "evals": evals, "schedule": sig,
               "refs": [{"latent_h": r["latent_h"], "latent_w": r["latent_w"]} for r in refs], "seq_len": seq_len,
               "snap_blocks": [0, 1, len(dm.blocks) // 2, len(dm.blocks) - 1]},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("done", out_dir, "seq_len", seq_len)


if __name__ == "__main__":
    main()
