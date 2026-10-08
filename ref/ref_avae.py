"""Reference dump of ComfyUI's MiniMax H3 audio VAE (encode + decode).

Usage (cwd = ComfyUI root, embedded python):
    python ref_avae.py <out_dir> [music_interleaved_f32]
The optional music file is raw f32le interleaved stereo at 32 kHz (e.g. from
`ffmpeg -ss 40 -t 2.5 -i song.wav -ac 2 -ar 32000 -f f32le music_interleaved.f32`).

Loads the VAE exactly as comfy.sd does (comfy.sd.VAE(sd=...) -> "pre_block.attn.zero_k_bias" branch) and calls
vae.encode / vae.decode with the same layouts the MiniMax H3 nodes use ([B, L, 2] waveform -> [B, 32, 2, T] latent).
Two precision modes are dumped:
  tf32 : ComfyUI defaults (cuDNN convs may use TF32 tensor cores: torch.backends.cudnn.allow_tf32 = True)
  fp32 : strict fp32 (allow_tf32 = False for cuDNN and cuBLAS) - the accuracy target for the Rust engine
Files (all raw little-endian f32):
  <case>_wav.bin              [2, n]       stereo planar input
  <case>_<mode>_latent.bin    [32, 2, T]   vae.encode
  <case>_<mode>_decoded.bin   [2, n]       vae.decode(<case>_<mode>_latent)
  synth_fp32_<name>.bin       intermediates (encoder output, head output, decoder stages) for debugging
  resample_in_44100.bin / resample_out_32000.bin   comfy.audio.resample reference (mono, 1 s)
  meta.json
"""
import sys, os, json, math
sys.path.insert(0, os.getcwd())
import numpy as np
import torch

import comfy.sd
import comfy.utils
import comfy.model_management

MODEL = r"C:\Users\steph\dev\loki-reshoot\models\minimax_h3_audio_vae_fp32.safetensors"
SR = 32000


def synth_signal(n):
    t = np.arange(n, dtype=np.float64) / SR
    rng = np.random.default_rng(1234)
    # L: log chirp 60 Hz -> 12 kHz + 440 Hz sine, noise burst in the middle
    f0, f1 = 60.0, 12000.0
    k = math.log(f1 / f0) / t[-1]
    chirp = np.sin(2 * math.pi * f0 * (np.exp(k * t) - 1) / k)
    left = 0.35 * chirp + 0.2 * np.sin(2 * math.pi * 440.0 * t)
    burst = (t > 0.4) & (t < 0.55)
    left[burst] += 0.3 * rng.standard_normal(burst.sum())
    # R: chord of sines with slow AM + a click train + different noise burst
    right = 0.18 * (np.sin(2 * math.pi * 220.0 * t) + np.sin(2 * math.pi * 277.18 * t + 0.3) + np.sin(2 * math.pi * 329.63 * t + 1.1))
    right *= 0.6 + 0.4 * np.sin(2 * math.pi * 3.0 * t)
    clicks = (np.arange(n) % 4000) < 8
    right[clicks] += 0.5
    burst2 = (t > 0.7) & (t < 0.8)
    right[burst2] += 0.25 * rng.standard_normal(burst2.sum())
    wav = np.stack([left, right]).astype(np.float32)
    return np.clip(wav, -1.0, 1.0)


def set_mode(mode):
    strict = mode == "fp32"
    torch.backends.cudnn.allow_tf32 = not strict
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def main():
    out_dir = sys.argv[1]
    os.makedirs(out_dir, exist_ok=True)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(MODEL))
    print("vae", type(vae.first_stage_model).__name__, "dtype", vae.vae_dtype, "device", vae.device)

    cases = {"synth": synth_signal(SR)}
    if len(sys.argv) > 2:
        m = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, 2).T.copy()
        n = (m.shape[1] // 800) * 800
        cases["music"] = m[:, :n]

    meta = {"sample_rate": SR, "cases": {}}
    for name, wav in cases.items():
        wav.tofile(os.path.join(out_dir, f"{name}_wav.bin"))
        n = wav.shape[1]
        meta["cases"][name] = {"n": int(n), "modes": {}}
        for mode in ["fp32", "tf32"]:
            set_mode(mode)
            hooks = []
            inter = {}
            if name == "synth" and mode == "fp32":
                fsm = vae.first_stage_model

                def grab(key, which="out"):
                    def h(mod, inp, out):
                        t = (inp[0] if which == "in" else out)
                        inter[key] = t.detach().float().cpu().contiguous()
                    return h
                hooks.append(fsm.encoder.block[0].register_forward_hook(grab("enc_conv0")))
                for i in range(1, 6):
                    hooks.append(fsm.encoder.block[i].register_forward_hook(grab(f"enc_block{i}")))
                hooks.append(fsm.encoder.register_forward_hook(grab("enc_out")))
                hooks.append(fsm.pre_block.register_forward_hook(grab("head_out")))
                hooks.append(fsm.pre_block.attn.register_forward_hook(grab("attn_out")))
                hooks.append(fsm.dec_in_proj.register_forward_hook(grab("dec_in")))
                hooks.append(fsm.decoder.conv_pre.register_forward_hook(grab("dec_pre")))
                for i in range(7):
                    hooks.append(fsm.decoder.ups[i][0].register_forward_hook(grab(f"dec_up{i}")))
                hooks.append(fsm.decoder.resblocks[0].activations[0].register_forward_hook(grab("dec_rb0_act0")))
                hooks.append(fsm.decoder.resblocks[0].register_forward_hook(grab("dec_rb0")))
                hooks.append(fsm.decoder.activation_post.register_forward_hook(grab("dec_act_post")))
                hooks.append(fsm.decoder.activation_post.register_forward_hook(grab("dec_stage6", "in")))

            x = torch.from_numpy(wav)[None].movedim(1, -1)  # [1, n, 2] as the H3 nodes pass it
            lat = vae.encode(x)  # [1, 32, 2, T]
            lat = lat.float().cpu()
            dec = vae.decode(lat)  # [1, n, 2]
            dec = dec.float().cpu()[0].movedim(-1, 0).contiguous()  # [2, n]
            for h in hooks:
                h.remove()
            lat[0].numpy().tofile(os.path.join(out_dir, f"{name}_{mode}_latent.bin"))
            dec.numpy().tofile(os.path.join(out_dir, f"{name}_{mode}_decoded.bin"))
            shapes = {}
            for k, v in inter.items():
                v.numpy().tofile(os.path.join(out_dir, f"{name}_{mode}_{k}.bin"))
                shapes[k] = list(v.shape)
            T = lat.shape[-1]
            err = (dec - torch.from_numpy(wav)).pow(2).mean() / torch.from_numpy(wav).pow(2).mean()
            print(f"{name}/{mode}: latent {list(lat.shape)} mean {lat.mean():.4f} std {lat.std():.4f}  decoded {list(dec.shape)}  roundtrip SNR {-10*math.log10(float(err)):.2f} dB")
            meta["cases"][name]["modes"][mode] = {"T": int(T), "latent_shape": list(lat.shape[1:]), "decoded_len": int(dec.shape[1]), "intermediates": shapes,
                                                  "roundtrip_snr_db": -10 * math.log10(float(err))}
        # fp32 vs tf32 gap (how far ComfyUI's default sits from strict fp32)
        a = np.fromfile(os.path.join(out_dir, f"{name}_fp32_decoded.bin"), dtype=np.float32)
        b = np.fromfile(os.path.join(out_dir, f"{name}_tf32_decoded.bin"), dtype=np.float32)
        la = np.fromfile(os.path.join(out_dir, f"{name}_fp32_latent.bin"), dtype=np.float32)
        lb = np.fromfile(os.path.join(out_dir, f"{name}_tf32_latent.bin"), dtype=np.float32)
        snr = 10 * math.log10(np.sum(a.astype(np.float64) ** 2) / max(np.sum((a.astype(np.float64) - b) ** 2), 1e-30))
        lrel = math.sqrt(np.sum((la.astype(np.float64) - lb) ** 2) / np.sum(la.astype(np.float64) ** 2))
        print(f"{name}: tf32 vs fp32  decoded SNR {snr:.2f} dB  latent rel-L2 {lrel:.3e}")
        meta["cases"][name]["tf32_vs_fp32"] = {"decoded_snr_db": snr, "latent_rel_l2": lrel}
    # comfy.audio.resample reference (44.1 kHz -> 32 kHz), used by the node for non-32k reference audio
    import comfy.audio
    sr = 44100
    t = np.arange(sr) / sr
    x = (0.4 * np.sin(2 * math.pi * 1000 * t) + 0.2 * np.sin(2 * math.pi * 15000 * t) + 0.1 * np.random.default_rng(0).standard_normal(sr)).astype(np.float32)
    y = comfy.audio.resample(torch.from_numpy(x)[None], sr, SR)[0].numpy().astype(np.float32)
    x.tofile(os.path.join(out_dir, "resample_in_44100.bin"))
    y.tofile(os.path.join(out_dir, "resample_out_32000.bin"))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
