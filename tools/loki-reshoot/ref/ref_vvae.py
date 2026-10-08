"""Reference dump of ComfyUI's MiniMax H3 video VAE (comfy.sd.VAE path, fp16).

Usage (cwd = ComfyUI root, embedded python):
    python ref_vvae.py <out_dir> [image_path]
For each case writes into <out_dir>/<case>/:
    pixels.u8      u8  [T, H, W, 3]   input frames
    latent.bin     f32 [24, Tl, H/16, W/16]  vae.encode(pixels/255)
    decoded.bin    f32 [Tout, H, W, 3] vae.decode(latent) in [0, 1]
    meta.json
Plus debug intermediates for the first case (dbg_*.bin): the first tile's raw encoder moments and the
first decoder call's raw output.
"""
import sys, os, json, time, math
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.ldm.minimax.vae as mvae
from comfy.ldm.modules import attention as cattn

MODEL = r"C:\Users\steph\bin\models\minimax_h3_video_vae_fp16.safetensors"


def make_clip(img, w, h, t, seed):
    """Smooth pan + slow zoom across a real photo -> u8 [t, h, w, 3]."""
    rng = np.random.RandomState(seed)
    W0, H0 = img.size
    frames = []
    cx0, cy0 = W0 * (0.35 + 0.1 * rng.rand()), H0 * (0.4 + 0.1 * rng.rand())
    for i in range(t):
        a = i / max(1, t - 1)
        zoom = 1.0 + 0.25 * a
        cw = W0 * 0.45 / zoom
        ch = cw * h / w
        cx = cx0 + W0 * 0.12 * a
        cy = cy0 + H0 * 0.04 * math.sin(3.0 * a)
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        fr = img.transform((w, h), Image.EXTENT, box, resample=Image.BICUBIC)
        frames.append(np.asarray(fr, dtype=np.uint8))
    return np.stack(frames)


@torch.no_grad()
def main():
    out_dir = sys.argv[1]
    img_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join("input", "016.JPG")
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(img_path).convert("RGB")

    sd = comfy.utils.load_torch_file(MODEL)
    vae = comfy.sd.VAE(sd=sd)
    fsm = vae.first_stage_model
    print("vae dtype", vae.vae_dtype, "attention", getattr(cattn.optimized_attention, "__name__", "?"))

    # debug hooks: first tile encode moments, first decoder call output
    dbg = {}
    orig_enc = fsm._encode_moments
    orig_dec = fsm._decode_pixels

    def enc_hook(x):
        out = orig_enc(x)
        if "enc_in" not in dbg:
            dbg["enc_in"] = x.float().cpu()
            dbg["enc_out"] = out.float().cpu()
        return out

    def dec_hook(z):
        out = orig_dec(z)
        if "dec_in" not in dbg:
            dbg["dec_in"] = z.float().cpu()
            dbg["dec_out"] = out.float().cpu()
        return out

    fsm._encode_moments = enc_hook
    fsm._decode_pixels = dec_hook

    cases = [
        ("v448", 448, 256, 22),
        ("v448_39", 448, 256, 39),
        ("v640", 640, 384, 22),
        ("img448", 448, 256, 1),
    ]
    if len(sys.argv) > 3:
        cases = [c for c in cases if c[0] in sys.argv[3].split(",")]
    for ci, (name, w, h, t) in enumerate(cases):
        d = os.path.join(out_dir, name)
        os.makedirs(d, exist_ok=True)
        pix = make_clip(img, w, h, t, seed=ci if name != "img448" else 0)
        pix.tofile(os.path.join(d, "pixels.u8"))
        x = torch.from_numpy(pix.astype(np.float32) / 255.0)
        torch.cuda.synchronize()
        t0 = time.time()
        lat = vae.encode(x)
        torch.cuda.synchronize()
        t_enc = time.time() - t0
        lat = lat.float().cpu()
        assert lat.shape[0] == 1
        lat[0].numpy().astype(np.float32).tofile(os.path.join(d, "latent.bin"))
        t0 = time.time()
        dec = vae.decode(lat)
        torch.cuda.synchronize()
        t_dec = time.time() - t0
        dec = dec.float().cpu()
        print(name, "latent", tuple(lat.shape), "decoded", tuple(dec.shape), f"enc {t_enc:.2f}s dec {t_dec:.2f}s")
        # -> [T, H, W, 3]
        if dec.ndim == 5:
            if dec.shape[1] == 3:
                dec = dec[0].permute(1, 2, 3, 0)
            else:
                dec = dec[0]
        elif dec.ndim == 4 and dec.shape[-1] != 3:
            dec = dec.permute(1, 2, 3, 0)
        dec = dec.contiguous()
        dec.numpy().astype(np.float32).tofile(os.path.join(d, "decoded.bin"))
        expect = fsm.decode_output_shape(tuple(lat.shape))
        meta = dict(t=t, h=h, w=w, latent_shape=list(lat.shape[1:]), decoded_shape=list(dec.shape),
                    decode_output_shape=list(expect), t_enc=t_enc, t_dec=t_dec, vae_dtype=str(vae.vae_dtype))
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        if ci == 0:
            for k, v in dbg.items():
                v.numpy().astype(np.float32).tofile(os.path.join(d, f"dbg_{k}.bin"))
                meta[f"dbg_{k}"] = list(v.shape)
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        dbg.clear()


if __name__ == "__main__":
    main()
