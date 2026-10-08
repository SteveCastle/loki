"""Run the video_minimax_h3_r2v graph through a private headless ComfyUI server (API format) for an apples-to-apples
comparison with h3ref2va. Usage (any python):
  python ref/ref_e2e.py --image testdata/ref.jpg --audio testdata/music3s.wav --prompt "..." --width 448 --height 256 --length 22 --seed 1 --out ref_out/e2e
Starts ComfyUI on port 8199 with --input-directory/--output-directory pointed at the out dir, kills it afterwards.
"""
import argparse, json, os, shutil, subprocess, sys, time, urllib.request

COMFY = r"C:\Users\steph\Downloads\ComfyUI_windows_portable_nvidia_cu126\ComfyUI_windows_portable"
ap = argparse.ArgumentParser()
ap.add_argument("--image"); ap.add_argument("--audio"); ap.add_argument("--video"); ap.add_argument("--prompt", required=True)
ap.add_argument("--width", type=int, required=True); ap.add_argument("--height", type=int, required=True)
ap.add_argument("--length", type=int, required=True); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--out", required=True)
a = ap.parse_args()
out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
inp = os.path.join(out, "input"); os.makedirs(inp, exist_ok=True)
refs = {}
if a.image: shutil.copy(a.image, os.path.join(inp, "ref_image" + os.path.splitext(a.image)[1])); refs["image"] = "ref_image" + os.path.splitext(a.image)[1]
if a.audio: shutil.copy(a.audio, os.path.join(inp, "ref_audio" + os.path.splitext(a.audio)[1])); refs["audio"] = "ref_audio" + os.path.splitext(a.audio)[1]

g = {
 "1": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
 "2": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
 "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
 "4": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
 "7": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {"clip": ["4", 0], "vae": ["1", 0], "audio_vae": ["2", 0], "prompt": a.prompt,
        "width": a.width, "height": a.height, "length": a.length, "ref_image_size": "match"}},
 "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
 "9": {"class_type": "BasicScheduler", "inputs": {"model": ["3", 0], "scheduler": "simple", "steps": a.steps, "denoise": 1.0}},
 "10": {"class_type": "BasicGuider", "inputs": {"model": ["3", 0], "conditioning": ["7", 0]}},
 "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
 "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": ["7", 1]}},
 "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 0]}},
 "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["2", 0]}},
 "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": 24.0}},
 "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": "e2e", "format": "auto", "codec": "auto"}},
}
if "image" in refs:
    g["5"] = {"class_type": "LoadImage", "inputs": {"image": refs["image"]}}
    g["7"]["inputs"]["ref_images.ref_image_0"] = ["5", 0]
if "audio" in refs:
    g["6"] = {"class_type": "LoadAudio", "inputs": {"audio": refs["audio"]}}
    g["7"]["inputs"]["ref_audios.ref_audio_0"] = ["6", 0]

port = 8199
srv = subprocess.Popen([os.path.join(COMFY, "python_embeded", "python.exe"), "-s", os.path.join("ComfyUI", "main.py"), "--windows-standalone-build",
        "--use-sage-attention", "--port", str(port), "--input-directory", inp, "--output-directory", out, "--disable-auto-launch"], cwd=COMFY)
try:
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try: urllib.request.urlopen(base + "/system_stats", timeout=2); break
        except Exception: time.sleep(2)
    t0 = time.time()
    req = urllib.request.Request(base + "/prompt", data=json.dumps({"prompt": g}).encode(), headers={"Content-Type": "application/json"})
    try:
        pid = json.loads(urllib.request.urlopen(req).read())["prompt_id"]
    except urllib.error.HTTPError as e:
        print(e.read().decode()); raise
    while True:
        h = json.loads(urllib.request.urlopen(base + "/history/" + pid).read())
        if pid in h:
            st = h[pid]["status"]
            print("status", st["status_str"], f"{time.time()-t0:.1f}s")
            if st["status_str"] != "success": print(json.dumps(st["messages"])[:3000])
            print(json.dumps(h[pid]["outputs"])[:500]); break
        time.sleep(2)
finally:
    srv.terminate()
    try: srv.wait(30)
    except Exception: srv.kill()
