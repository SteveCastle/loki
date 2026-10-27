# loki-dream

**Text to image from the command line**, with **Kromoss** (lodestones' full fine-tune of OpenMOSS's MOSS-Image), the
**Qwen3.5-9B** text encoder and the **FLUX.2** small VAE, as one self-contained binary. It does text-to-image,
img2img, inpainting, LoRAs and an 8-step turbo mode. The whole stack is reimplemented in CUDA and Rust: the 5B single-stream DiT
(int8 ConvRot tensor-core GEMMs, int8/fp8 attention), Qwen3.5's hybrid Gated-DeltaNet / gated-attention encoder,
and the VAE. No Python, ONNX, PyTorch or ComfyUI at runtime; the only dependency is the NVIDIA driver.

```
loki-dream -p "a tabby cat holding a sign that says KROMOSS"            # -> dream_<seed>.png (50 steps, CFG 3)
loki-dream --turbo --count 4 -o drafts/ -p "a lighthouse in a storm"     # 8-step DMD2 LoRA, ~2.6 s per 1024² image
loki-dream --aspect 16:9 --seed 7 -p "a misty mountain lake at dawn" -o lake.png
loki-dream --init photo.jpg --mask sky.png --strength 1 -p "a sky full of hot air balloons"
```

## Part of the loki- toolkit

| tool | does | engine |
|---|---|---|
| [loki-retouch](../loki-retouch) | image editing, upscaling, restoration, compositing | Qwen Image 2.1 |
| [loki-reshoot](../loki-reshoot) | reference images / videos / audio -> video with sound | MiniMax H3 |
| [loki-segment](../loki-segment) | concept / click / box segmentation, tracked roto of video | SAM 3.1 |
| **loki-dream** (this tool) | text -> image, img2img, inpainting, LoRAs | Kromoss (MOSS-Image) |

Shared conventions (`../CONVENTIONS.md`): stdout carries only results (the path of every image, one per line;
`--json` one object; `-o -` the PNG). Progress, timings and warnings go to stderr (`-q` silences them). Exit status is 0 ok,
1 error, 2 usage. Models are found next to the binary, in `./models` or `$LOKI_MODELS`, and downloaded when missing. Every
PNG carries the `loki` generation record (iTXt chunk).

```
loki-segment -p sky -o sky.png photo.jpg && loki-dream --init photo.jpg --mask sky.png --strength 1 -p "an aurora" -o - | loki-reshoot --animate - -d 5 -o aurora.mp4
```

## Usage

```
loki-dream [OPTIONS] -p PROMPT
  -p, --prompt TEXT        what to draw (repeatable: several prompts in one run, models load once)
  -P, --prompt-file FILE   the whole file is one prompt (- = stdin);  --batch FILE: one prompt per line
  --negative TEXT          negative prompt (default empty, as the official pipeline; ignored with --turbo / --cfg 1)
  --size WxH | --aspect W:H [--megapixels MP] | --megapixels MP     default 1024x1024; --snap keeps the x16 size
  --steps N (50)  --cfg F (3)  --seed N (random)  --count N (images per prompt)
  --turbo                  8-step DMD2 LoRA: 8 steps, CFG 1, no negative branch
  --lora FILE[:SCALE]      repeatable; kohya (Kromoss/MOSS), ComfyUI, diffusers/PEFT keys
  --init IMAGE --strength F (0.6)   img2img;   --mask MASK (with --init): inpaint, white = regenerate, soft edges
  --shift F (5)  --skip-layers LIST|none (22)   official sampler knobs
  -o PATH|DIR|-  --json  -q  --show-prompt  --meta JSON | --meta-file FILE
  --precision int8|bf16    the diffusion model (default int8 ConvRot when present)
  --dit / --text-encoder / --vae PATH
```

**Output naming.** `-o FILE.png` for one image. With several images (`--count`, several prompts), `-o FILE.png` gives
`FILE_<seed>.png` next to it. `-o DIR` (an existing folder, or a path ending in `/` or `\`) gives `DIR/dream_<seed>.png`.
Without `-o`: `dream_<seed>.png` in the current folder (img2img: `<init>_dream_<seed>.png` next to the init image).
Existing files are never overwritten: the name gets `_1`, `_2`, ... Seeds are `seed, seed+1, ...` in prompt-major order.

**Progress.** On a terminal you get bars. Otherwise (e.g. under the media server) stderr carries `image k/N: seed S WxH`
and `step i/n` lines (`step (\d+)/(\d+)`, restarting for every image).

**`--json`** prints one object for the run:

```json
{"images":[{"path":"out/dream_7.png","seed":7,"width":1024,"height":1024,"prompt":"...","generation":{...}}],
 "prompt":"...","negative":"","steps":50,"cfg":3.0,"turbo":false,"width":1024,"height":1024}
```

`generation` is the record stamped into the PNG: `prompt`, `negative`, `seed`, `steps`, `cfg`, `shift`, `turbo`,
`sampler`, `skip_layers`, `precision`, `width`/`height` (and `generated_width`/`generated_height` when resampled),
`strength`, `loras` (path and strength), `inputs` (`init` and `mask`), `meta`, `args`.

**Sizes.** The model works in multiples of 16 pixels, and about one megapixel is its home ground (Kromoss was fine-tuned at
1024 px). Other sizes are generated at the nearest multiple and Lanczos-resampled to exactly what was asked (`--snap`
keeps the native size). With `--init` and no size flag the output has the init image's size. `--aspect` and `--megapixels`
choose a multiple-of-16 size near the requested pixel count.

**Turbo.** `--turbo` applies `8-steps-dmd2-lora.safetensors` (strength 1), 8 steps, CFG 1 and no negative branch, as the
Kromoss card prescribes. Of the two turbo LoRAs, DMD2 is the default: the card ranks it best on PartiPrompts (CLIP 28.13 vs
27.99 for DMD), and side by side it gives more texture and contrast at the same composition. The hidden
`--turbo-lora dmd` selects the other one. `--turbo` combines with `--lora` and with img2img and inpainting.

**img2img and inpainting.** `--strength F` keeps the first `1-F` of the schedule from the encoded image (diffusers'
img2img semantics: noise to that step's sigma, then denoise the rest). The shift-5 schedule spends most steps at high
noise, so colours and layout survive up to F ~ 0.9; use 0.9-0.97 to change the subject while keeping the composition,
and 1.0 to ignore the image. `--mask` (any size, resized to the init image; luminance, or alpha for a transparent cut-out)
regenerates the white area. Every step, the init latent noised to the current sigma is re-imposed outside the mask
(latent mask = the max over each 16x16 cell, so soft edges never pin old content inside the region). The decoded result
is then composited over the init at full resolution with the soft mask. Use `--strength 1` with a mask to replace content.

**Layout prompts.** MOSS-Image understands a structured layout prompt, given as the prompt text (one line of JSON):

```json
{"summary":{"caption":"a minimalist poster for a jazz night, deep blue background"},
 "items":[{"label":"text","caption":"big bold title","text":"BLUE NOTE","bbox":[-420,-400,-220,400]},
          {"label":"object","caption":"a golden saxophone","bbox":[-150,-250,420,250]}],
 "bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}
```

`bbox` is `[y0, x0, y1, x1]` in output pixels with the origin at the image centre. `label` is `text` (rendered with its
`text`), `object` or `character`; an optional `palette` lists colours. This is the format ComfyUI's MOSSImageLayoutPrompt
node builds. It goes through the text encoder like any prompt (`--show-prompt` prints the exact templated text).

**Seeds.** The initial noise is drawn exactly as `torch.randn([1, 128, H/16, W/16], generator=torch.Generator("cpu").manual_seed(seed))`
(MT19937 plus ATen's Box-Muller; bit-identical, verified). A seed therefore means the same starting noise as in the official
diffusers pipeline and in ComfyUI. Runs are deterministic: the same command gives the same PNG, bit for bit.

## Models

Found next to `loki-dream.exe`, in a `models/` folder next to it, in `$LOKI_MODELS` (a path list) or `./models`; missing
files are downloaded (resumable) from [lodestones/Kromoss](https://huggingface.co/lodestones/Kromoss) (Apache-2.0, ungated)
into `models/` next to the binary.

| role | file | size | when |
|---|---|---|---|
| diffusion model | `kromoss_int8_convrot.safetensors` | 5.4 GB | default |
| diffusion model | `kromoss_bf16.safetensors` | 10.2 GB | `--precision bf16` |
| text encoder | `qwen3.5_9b_bf16.safetensors` (Qwen3.5-9B, language model used) | 19.3 GB | always |
| VAE | `flux2-vae_small.safetensors` (FLUX.2 encoder + small decoder) | 0.25 GB | always |
| turbo LoRA | `8-steps-dmd2-lora.safetensors` (`8-steps-dmd-lora.safetensors` with `--turbo-lora dmd`) | 1.2 GB | `--turbo` |

First run total: 25 GB (26.2 GB with `--turbo`). The tokenizer (Qwen3.5's 248k BPE, from the repo's `tokenizer/`) is
embedded in the binary (`assets/`).

## Building

Rust (stable) plus CUDA Toolkit 12.6+ (`nvcc`) and a host C++ compiler, then `cargo build --release`. Same layout as the
siblings: `build.rs` compiles `kernels/*.cu` to fatbins (sm_89 + compute_80 PTX) that are embedded; `cudarc` loads the
driver dynamically.

```
cargo test --release --lib --test tokenizer      # GPU-free (CI): unit tests + tokenizer vs Hugging Face fixtures
cargo run --release --bin dream_check -- ...     # stage checks against reference dumps (GPU, models; below)
```

## Engine

- **Text encoder (Qwen3.5-9B, 32 layers).** 24 Gated DeltaNet layers: fused in-projection, depthwise causal conv + SiLU,
  L2-normalized q/k, the delta rule in its recurrent form (fp32 state in registers, one block per value head), and a gated
  RMSNorm. 8 gated full-attention layers: GQA 16/4 heads of 256, RoPE on the first 64 dims (theta 1e7), causal softmax and a
  sigmoid output gate. Zero-centred `(1+w)` RMSNorms, SwiGLU MLP. bf16 weights are streamed layer by layer, so every
  prompt in a batch (and the negative) passes a layer before the next one is uploaded and the encoder never holds more
  than ~0.5 GB of weights. The input is the official chat template (`apply_chat_template(..., enable_thinking=True)`:
  the user turn trimmed, the assistant turn left on an open `<think>` block); the conditioning is `last_hidden_state`.
- **DiT (42 blocks, dim 3072, 24 heads of 128, FFN 8192).** The text refiner (2 blocks over the text tokens only) does not
  depend on the timestep and runs once per prompt. One shared modulation per step (shift/scale/gate for attention and FFN)
  plus a head modulation. The SiTU FFN activation `4 tanh(g/4) sigmoid(g) * 25 tanh(h/25)` and the modulated RMSNorm are
  fused into the int8 activation quantizer (Hadamard-256 rotation + per-token scales), and the gated residual into the int8
  GEMM epilogue. RoPE uses 4 axes of 16 frequencies `(r, f, h, w)`, image tokens at `(0, T, row-(H-1)/2, col-(W-1)/2)` and text
  token i at `(0, i, 0, 0)`. Every op rounds to bf16 where the reference does. Attention is int8 QK / fp8 PV (Sage-style) with
  the int8 model and bf16 flash attention with the bf16 model (`LOKI_ATTN=bf16|sage` overrides).
- **Sampler.** The official `MOSSImageScheduler` (shift 5 over a linspace, snapped to the 1000-step training grid,
  duplicates dropped), flow-matching Euler in fp32. CFG runs the negative branch with block 22 skipped
  (`uncond_skip_layers=(22,)`, the official default), and cond and uncond each run at their true text length.
- **LoRA.** Unmerged (the card warns that the DMD deltas are near bf16's rounding step): `y += (x down^T) up^T * s` as two
  small bf16 GEMMs per targeted linear. q/k/v deltas land on their slices of the fused qkv. Rank 128 adds ~10% to a step.
- **VAE.** Implicit-GEMM convolutions on channels-last bf16 planes (shared with loki-retouch), two-pass deterministic
  GroupNorm, single-head mid attention, 2x2 (un)patchify and BatchNorm (de)standardization fused into the latent kernels.
- **Loading.** Weight files are memory-mapped and staged through pinned buffers by 8 threads (first-touch page faults
  parallelize), and the mappings are unmapped on a background thread (unmapping a 19 GB view costs ~3 s on Windows).
- **Determinism.** No float atomics anywhere (the attention's K/V means and the GroupNorm statistics are two-pass
  reductions), so runs are bit-reproducible.

## Validation against the official pipeline

`ref/ref_dream.py` dumps reference tensors from `pipeline_moss_image.py` (lodestones/Kromoss; diffusers 0.42,
transformers 5.20, torch 2.14 cu126), loaded from the same single files the engine uses. `dream_check` runs the native stage
on the same inputs. The venv is `tools/ref-env` (git-ignored), run with `python -I`:

```
python -m venv tools/ref-env
tools/ref-env/Scripts/python -I -m pip install torch --index-url https://download.pytorch.org/whl/cu126
tools/ref-env/Scripts/python -I -m pip install "diffusers>=0.38" "transformers>=5.3" accelerate safetensors pillow numpy
# ref_src/kromoss/: pipeline_moss_image.py, configs, tokenizer from the HF repo (git-ignored)
tools/ref-env/Scripts/python -I ref/ref_dream.py te    ref_out/a -p "a tabby cat holding a sign that says KROMOSS" -p "" -p "..." -p "<layout json>"
tools/ref-env/Scripts/python -I ref/ref_dream.py te32  ref_out/a -p ...        # the same in fp32 on the CPU (numerical truth)
tools/ref-env/Scripts/python -I ref/ref_dream.py sched ref_out/a
tools/ref-env/Scripts/python -I ref/ref_dream.py dit   ref_out/a --size 768x512 --t 871
tools/ref-env/Scripts/python -I ref/ref_dream.py vae   ref_out/a --image testdata/apple.png
tools/ref-env/Scripts/python -I ref/ref_dream.py full  ref_out/a --size 768x768 --steps 20 --cfg 3 --seed 42 --tag cfg20
tools/ref-env/Scripts/python -I ref/ref_dream.py full  ref_out/a --size 1024x1024 --steps 8 --cfg 1 --seed 42 --lora models/8-steps-dmd2-lora.safetensors --tag turbo
target/release/dream_check te|sched|dit|vae ref_out/a ;  dream_check full ref_out/a cfg20 --int8|--bf16 [--lora FILE]
```

Text encoder (`last_hidden_state`, 4 prompts of 10-72 tokens incl. the empty negative and a layout JSON). The native
engine is as close to the fp32 truth as the official bf16 pipeline is:

| prompt | native vs reference (cos) | native vs fp32 | reference vs fp32 |
|---|---|---|---|
| cat sign (22 tok) | 0.999883 | 0.999923 | 0.999919 |
| empty negative (10) | 0.999816 | 0.999842 | 0.999829 |
| long photo prompt (48) | 0.999864 | 0.999840 | 0.999722 |
| layout JSON (72) | 0.999861 | 0.999929 | 0.999908 |

Token ids and the chat template are identical (also a GPU-free test). Scheduler: sigmas bit-exact for 1/4/8/20/28/50 steps.
Noise: bit-exact with `torch.randn`.

One DiT forward (768x512, t = 871, the cat prompt; cos / relative L2):

| stage | bf16 model | int8 model + int8/fp8 attention |
|---|---|---|
| text refiner output | 0.999992 / 0.0039 | 0.999992 / 0.0039 |
| after block 0 / 21 / 41 | 0.999993 / 0.999985 / 0.999826 | 0.999981 / 0.999986 / 0.998804 |
| velocity, cond | 0.999923 / 0.0124 | 0.999720 / 0.0237 |
| velocity, uncond (block 22 skipped) | 0.999938 / 0.0112 | 0.999771 / 0.0216 |

VAE: decode PSNR 57.7 dB vs the fp32 reference (8-bit output); encode cos 0.99989.

Whole images at a fixed seed (same noise), PSNR vs the reference image. Diffusion amplifies any rounding difference, so
the scale is the reference itself with only its attention kernel swapped (SDPA flash vs math):

| run | native bf16 | native int8 | reference vs itself |
|---|---|---|---|
| 768², 20 steps, CFG 3 | 26.0 dB | 24.1 dB | 24.6 dB |
| 1024², 50 steps, CFG 3 | 27.7 dB | 26.1 dB | 17.4 dB |
| 1024², 8 steps, DMD2 LoRA, CFG 1 | 27.9 dB | 24.4 dB | |

Same composition, subject and text. Fine details such as the exact glyphs on a sign vary as much as they do between two
attention kernels in PyTorch.

## Performance (RTX 4090, Windows, model files in the OS cache)

| 1024x1024 | sampling | end to end | peak VRAM (above the desktop) |
|---|---|---|---|
| int8, 50 steps, CFG 3 | 16.8 s (0.17 s per DiT pass) | 21.6 s | 7.6 GB |
| int8, `--turbo` (8 steps) | 2.6 s | 7.5 s | 7.6 GB |
| bf16, 50 steps, CFG 3 | 38.8 s | 45.3 s | 12.0 GB |
| bf16, `--turbo` | 4.3 s | 10.3 s | 12.0 GB |

Fixed costs per run: text encoder 2.7-2.9 s (14 GB of bf16 weights streamed), DiT upload 1.1 s (int8) / 2.0 s (bf16),
LoRA prep 0.4 s, VAE decode 0.2 s. `--count N` and several prompts pay them once. At 1024² the int8 GEMMs run at 320-360
TOPS and attention takes about a quarter of a step (`LOKI_PROFILE=1` prints the per-kernel breakdown).

## Deviations

- **Official pipeline vs ComfyUI.** ComfyUI's MOSS-Image samples with `simple` sigmas over ModelSamplingDiscreteFlow(shift 5),
  i.e. picked from the *shifted* 1000-point table, not the official linspace-then-snap schedule. It also has no
  negative-branch block skip by default. loki-dream follows the official pipeline on both. Model, text template (ComfyUI
  matches the official one), LoRA mapping and int8 ConvRot format agree with ComfyUI.
- **CLI.** `--json` prints one object for the run (as specified for this tool) instead of one per result. The default seed is
  random (always recorded). Additions: several `-p`, `--batch`, `--aspect`/`--megapixels`/`--snap`, `--shift`,
  `--skip-layers`, `--turbo-lora` (hidden). LoRA keys for modules outside the 42 DiT blocks (text refiner, time
  embedding) are reported and skipped.
- **Tokenizer.** The NFC normalizer and the Qwen2 split pattern are reimplemented. Letters/digits use Rust's Unicode
  `alphabetic`/`numeric`, which can differ from the regex's `\p{L}`/`\p{N}` on rare scripts.

## Gotchas found on the way

- The reference harness must rebuild Qwen3.5's non-persistent `inv_freq` buffers after a meta-device load, or RoPE
  silently becomes the identity (`ref_common.load_text_encoder` does). The official `text_encoder/` folder rounds `A_log`
  and the DeltaNet norm weight to bf16, while the single file keeps them fp32. Both sides round to bf16, as the official
  pipeline does.
- Float `atomicAdd` in the attention's K/V mean (inherited from loki-retouch's `sage_attn.cu`) made runs differ by
  ~26 dB through the diffusion. It is now a fixed-order two-pass reduction. loki-retouch still has the atomic version.
- Unmapping a 19 GB mapping on Windows takes ~3 s; it now happens on a background thread.
