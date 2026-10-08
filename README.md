# h3ref2va

MiniMax **H3 reference-to-video** (`ref2va`) as a single self-contained Windows binary: any mix of reference
**images, videos and audio** in, an **mp4 with native generated sound** out. No ComfyUI, no Python, no PyTorch: the
whole engine (int8 tensor-core GEMMs, Sage-style int8/fp8 attention, NVFP4 text encoder, 3-D video VAE, BigVGAN audio VAE)
is CUDA written from scratch and driven from Rust. Runtime dependencies: the NVIDIA driver and `ffmpeg` (auto-downloaded on
Windows when missing).

It reproduces the ComfyUI `video_minimax_h3_r2v` workflow (`MiniMaxH3ReferenceToVideo` → `BasicGuider` (cfg 1, one DiT pass
per step) → `SamplerCustomAdvanced` `res_multistep` / `simple` schedule / 20 steps → video + audio VAE decode → mp4), and
every component was validated against ComfyUI's implementation (see *Validation*). Sibling of [4kify](../4kify).

## Usage

```
h3ref2va --prompt TEXT [OPTIONS]

  -p, --prompt TEXT / --prompt-file F   prompt; address references with <Picture 1>, <Video 1>, <Audio 1> tags
  -i, --ref-image PATH                  reference image            (repeatable, up to 9)
  -v, --ref-video FILE[@START[,DUR]]    reference video, seconds   (repeatable, up to 3)
  -a, --ref-audio FILE[@START[,DUR]]    reference audio            (repeatable, up to 3)
      --no-video-audio                  ignore the soundtracks of reference videos
  -o, --out PATH                        output mp4 (default h3_<seed>.mp4)
  -d, --duration SECONDS                default 5 (snapped up to the model's 17k+5 frame grid @ 24 fps)
      --frames N                        exact frame count instead of --duration
      --size WxH | --aspect W:H --megapixels F    output size (default: first reference's aspect at 0.4 MP)
      --steps N (20)  --seed N  --ref-image-size match|max  --no-audio  --crf N (18)
      --dit / --text-encoder / --video-vae / --audio-vae PATH   model files
      --ffmpeg PATH
```

Reference tags are numbered per type in command-line order: `<Picture i>` for `--ref-image`, `<Video k>` for `--ref-video`, and
`<Audio j>` for the soundtracks of the reference videos first (in video order) and then the `--ref-audio` clips. They are printed
at start-up. The model is very sensitive to prompt wording: say explicitly which reference drives which part of the shot, e.g.

```
h3ref2va -i hero.png -a theme.mp3 -d 6 -p "Animate <Picture 1> as one continuous shot: she dances to the music of <Audio 1>. Audio: loud rhythmic club music from <Audio 1>."
h3ref2va -i face.png -v dance.mp4@2,4 -p "The woman of <Picture 1> performs the motion of <Video 1>, keeping the voice and sound of <Audio 1>."
```

## Quick mode: living photos

```
h3ref2va --animate photo.jpg --describe "the woman in a black swimsuit taking a mirror selfie in a sunlit room" -d 5
```

`--animate IMAGE` gives natural ambient life, subtle resting movement and a subtle camera shake (`--shake none|subtle|handheld`) with
ambient sound, keeping the photo's identity and framing. It implies `--native`: the model works best at five canvases (1:1 768×768,
4:3 1024×768, 3:4 768×1024, 16:9 1344×768, 9:16 768×1344); the first image is snapped to the nearest ratio and fitted with
`--fit pad|crop|stretch` (pad = black bars the model fills in, as in the ComfyUI `MiniMaxH3AutoRatio` node).

## What the model can do (limits enforced by the CLI)

| | |
|---|---|
| **length** | trained range ≈ 5.2–15 s = 124–362 frames @ 24 fps; shorter clips run (a note is printed), longer than 362 frames (15.08 s) is refused. Frame counts snap **up** to `17k+5` (5, 22, 39, …, 124, …, 362); audio is 40 latent frames/s. |
| **size** | multiples of 32, up to ≈2K: ComfyUI's template table tops out at 1920×1088 (2.0 MP); the hard cap is 2048×1152. Default is 0.4 MP (e.g. 864×480); the node's own default is 1344×768 (0.98 MP). |
| **references** | up to 9 images (scaled down only, aspect kept, to the generation's pixel area with `match`, or up to a 2048 px short edge with `max`), 3 videos (≥5 frames, resampled to 24 fps, canvas ≤ 768×1344 pixels, at most the output length), 3 audio clips (≤30 s here). Reference tokens ride along in every sampling step, so each one costs time. |
| **output** | H.264 + AAC mp4, 24 fps, 32 kHz stereo audio generated jointly with the video. |

Cost grows with tokens `latent_t × (H/32) × (W/32)`: doubling both sides is 4× tokens (16× attention); doubling the length is 2×.
Length is much cheaper than resolution for the same added value.

## Models

Looked up next to the binary, in `./models`, and in `$H3_MODELS`; any that are missing are downloaded (resumable) from
[Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) into `models/` next to the binary (~42 GB total).

| flag | file | size |
|---|---|---|
| `--dit` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 21.0 GB |
| `--text-encoder` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 15.7 GB |
| `--video-vae` | `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB |
| `--audio-vae` | `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB |

Models are loaded in phases so everything fits a 24 GB card: text encoder (evicted) → VAE encode of the references (evicted)
→ DiT sampling (evicted) → VAE decode streamed straight into ffmpeg.

## Building

Rust (stable, MSVC), CUDA Toolkit 12.x (`nvcc`), Visual Studio 2022 C++ tools. `cargo build --release` → `target/release/h3ref2va.exe`
(`build.rs` compiles `kernels/*.cu` to sm_89 fatbins embedded in the binary; `cudarc` loads the driver dynamically).

See `docs/NOTES.md` for the reference-code map, and `ref/` for the ComfyUI reference-dump scripts used to validate each component.
