"""Reference dump of ComfyUI's MiniMax H3 DiT (MiniMaxH3Model.forward) on a small synthetic ref2va case.

Usage (cwd = ComfyUI root, embedded python):
    python <loki-reshoot>/ref/ref_dit.py <out_dir> [sigmas=0.7,0.2] [W=448] [H=256] [frames=22]

Writes into out_dir:
    text.bin        f32 [L, 5120]   (bf16-representable values; fed to the model as bf16)
    tags.bin        u8  [L]
    video_in.bin    f32 [24, T, H, W]  (target video latent x)
    audio_in.bin    f32 [32, 2, Ta]    (carried audio variable y, as the sampler holds it)
    ref<i>_video.bin / ref<i>_audio.bin  reference latents
    out_video_<s>.bin, out_audio_<s>.bin   model outputs (already negated, carry-converted) per sigma
    text_states.bin f32 [L, 5376]  token refiner output (once)
    h_<s>_<tag>.bin  hidden state snapshots (block inputs/outputs) per sigma
    meta.json
"""
import sys, os, json
sys.path.insert(0, os.getcwd())
import torch
import numpy as np

import comfy.sd
import comfy.utils
import comfy.model_management

MODEL = r"C:\Users\steph\dev\loki-reshoot\models\minimax_h3_ref2va_pruned_int8_convrot.safetensors"

if os.environ.get("REF_W8A16") == "1":
    # High-precision reference: no activation quantization (rotated activation x dequantized int8 weight in fp32).
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
    sigmas = [float(s) for s in (sys.argv[2] if len(sys.argv) > 2 else "0.7,0.2").split(",")]
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 448
    H = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    frames = int(sys.argv[5]) if len(sys.argv) > 5 else 22
    os.makedirs(out_dir, exist_ok=True)

    from comfy_extras.nodes_minimax_h3 import temporal_shape
    frame_count, latent_t, audio_t = temporal_shape(frames)
    lh, lw = H // 16, W // 16

    g = torch.Generator("cpu").manual_seed(1234)
    L = 60
    text = (torch.randn(L, 5120, generator=g) * 2.0).to(torch.bfloat16)
    tags = torch.ones(L, dtype=torch.long)
    tags[10:30] = 0  # one vision span
    video = torch.randn(1, 24, latent_t, lh, lw, generator=g)
    audio = torch.randn(1, 32, 2, audio_t, generator=g)

    # references: image 8x14 latents, video+audio (latent_t 2, 8x14, 10 audio frames), standalone audio 40 frames
    refs = []
    ref_img = torch.randn(1, 24, 1, 8, 14, generator=g)
    refs.append({"kind": "image", "latent_h": 8, "latent_w": 14, "latent": ref_img})
    ref_vid = torch.randn(1, 24, 2, 8, 14, generator=g)
    ref_vid_a = torch.randn(1, 32, 2, 10, generator=g)
    refs.append({"kind": "video_audio", "latent_t": 2, "latent_h": 8, "latent_w": 14, "ref_audio_t": 10,
                 "latent": ref_vid, "audio_latent": ref_vid_a})
    ref_aud = torch.randn(1, 32, 2, 40, generator=g)
    refs.append({"kind": "audio", "ref_audio_t": 40, "audio_latent": ref_aud})

    model = comfy.sd.load_diffusion_model(MODEL)
    comfy.model_management.load_models_gpu([model])
    dev = model.load_device
    dm = model.model.diffusion_model
    dtype = model.model.get_dtype_inference() if hasattr(model.model, "get_dtype_inference") else torch.bfloat16
    print("compute dtype", dtype)

    payload = {
        "text_token_tags": tags,
        "refs": refs,
        "cond_video_latents": [r["latent"] for r in refs if "latent" in r],
        "cond_audio_latents": [r["audio_latent"] for r in refs if r.get("audio_latent") is not None],
        "visual_cond_noise_aug": 1.0,
        "seed": 0,
        "audio_scale": 12.0 / 3.0,
    }
    to = {"minimax_h3_sigma_shift_video": 12.0, "minimax_h3_sigma_shift_audio": 3.0}
    ctx = text[None].to(dev, dtype)

    # hooks: snapshot the hidden state entering block 0 and leaving selected blocks
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
    def ref_hook(mod, args, out):
        snaps["text_states"] = out.clone()
    hooks.append(dm.token_refiner.register_forward_hook(ref_hook))

    dump(os.path.join(out_dir, "text.bin"), text)
    tags.to(torch.uint8).numpy().tofile(os.path.join(out_dir, "tags.bin"))
    dump(os.path.join(out_dir, "video_in.bin"), video[0])
    dump(os.path.join(out_dir, "audio_in.bin"), audio[0])
    meta_refs = []
    for i, r in enumerate(refs):
        m = {k: v for k, v in r.items() if k not in ("latent", "audio_latent")}
        if "latent" in r:
            dump(os.path.join(out_dir, f"ref{i}_video.bin"), r["latent"][0])
            m["video_shape"] = list(r["latent"][0].shape)
        if r.get("audio_latent") is not None:
            dump(os.path.join(out_dir, f"ref{i}_audio.bin"), r["audio_latent"][0])
            m["audio_shape"] = list(r["audio_latent"][0].shape)
        if r["kind"] == "image":
            m["latent_t"] = 1
        meta_refs.append(m)

    seq_len = None
    for s in sigmas:
        snaps.clear()
        torch.cuda.synchronize()
        import time
        t0 = time.time()
        out = dm.forward([video.to(dev), audio.to(dev)], torch.tensor([s * 1000.0], device=dev), ctx,
                         transformer_options=dict(to), minimax_payload=dict(payload))
        torch.cuda.synchronize()
        print(f"sigma {s}: forward {time.time() - t0:.2f}s", out[0].shape, out[1].shape,
              float(out[0].float().std()), float(out[1].float().std()))
        dump(os.path.join(out_dir, f"out_video_{s}.bin"), out[0][0])
        dump(os.path.join(out_dir, f"out_audio_{s}.bin"), out[1][0])
        cmp_dir = os.environ.get("REF_COMPARE_DIR")
        if cmp_dir:
            for nm, o in (("video", out[0][0]), ("audio", out[1][0])):
                p = os.path.join(cmp_dir, f"out_{nm}_{s}.bin")
                if os.path.exists(p):
                    other = torch.from_numpy(np.fromfile(p, dtype=np.float32)).reshape(o.shape)
                    a = o.float().cpu()
                    rel = float((other - a).norm() / a.norm())
                    cos = float((other * a).sum() / (other.norm() * a.norm()))
                    print(f"   {cmp_dir} vs this, {nm}: rel_l2 {rel:.4e} cos {cos:.6f} max_abs {float((other - a).abs().max()):.4f}")
        for k, v in snaps.items():
            if k == "text_states":
                dump(os.path.join(out_dir, "text_states.bin"), v[0] if v.ndim == 3 else v)
            else:
                dump(os.path.join(out_dir, f"h_{s}_{k}.bin"), v)
                seq_len = v.shape[0]
    for h in hooks:
        h.remove()
    json.dump({"text_len": L, "latent_t": latent_t, "latent_h": lh, "latent_w": lw, "audio_t": audio_t,
               "sigmas": sigmas, "refs": meta_refs, "seq_len": seq_len, "hidden": 5376,
               "snap_blocks": [0, 1, len(dm.blocks) // 2, len(dm.blocks) - 1]},
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("done", out_dir, "seq_len", seq_len)


if __name__ == "__main__":
    main()
