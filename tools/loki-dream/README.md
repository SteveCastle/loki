# loki-dream

**Text to image from the command line**, with **Kromoss** (lodestones' full fine-tune of OpenMOSS's MOSS-Image), the
**Qwen3.5-9B** text encoder and the **FLUX.2** small VAE, as one self-contained binary. It does text-to-image,
img2img, inpainting, LoRAs and an 8-step turbo mode. The whole stack is reimplemented in CUDA and Rust: the 5B single-stream DiT
(int8 ConvRot tensor-core GEMMs, int8/fp8 attention), Qwen3.5's hybrid Gated-DeltaNet / gated-attention encoder,
and the VAE. No Python, ONNX, PyTorch or ComfyUI at runtime; the only dependency is the NVIDIA driver.

It also runs three other architectures, each with its own text encoder, VAE and sampler, picked by the model file
(`--model FILE`; the architecture is detected from the checkpoint): **Krea 2** (12B, Qwen3-VL-4B + Qwen-Image VAE),
**FLUX.2 klein** (4B / 9B, Qwen3-4B / 8B + FLUX.2 VAE) and **Anima** (2B Cosmos-Predict2 with an LLM adapter, Qwen3-0.6B
+ Qwen-Image VAE). See [Other architectures](#other-architectures-krea-2-flux2-klein-anima).

```
loki-dream -p "a tabby cat holding a sign that says KROMOSS"            # -> dream_<seed>.png (50 steps, CFG 3)
loki-dream --turbo --count 4 -o drafts/ -p "a lighthouse in a storm"     # 8-step DMD2 LoRA, ~2.6 s per 1024² image
loki-dream --aspect 16:9 --seed 7 -p "a misty mountain lake at dawn" -o lake.png
loki-dream --init photo.jpg --mask sky.png --strength 1 -p "a sky full of hot air balloons"
loki-dream --model krea2_turbo_fp8_scaled -p "a portrait of an old fisherman at golden hour"      # Krea 2 Turbo, 8 steps
loki-dream --model flux-2-klein-4b -p "a red apple on a wooden table"                            # FLUX.2 klein, 4 steps
loki-dream --arch anima -p "masterpiece, best quality, safe, 1girl, silver hair, cherry blossoms"  # Anima (downloads Turbo)
```

## Part of the loki- toolkit

| tool | does | engine |
|---|---|---|
| [loki-retouch](../loki-retouch) | image editing, upscaling, restoration, compositing | Qwen Image 2.1 |
| [loki-reshoot](../loki-reshoot) | reference images / videos / audio -> video with sound | MiniMax H3 |
| [loki-segment](../loki-segment) | concept / click / box segmentation, tracked roto of video | SAM 3.1 |
| **loki-dream** (this tool) | text -> image, img2img, inpainting, LoRAs | Kromoss (MOSS-Image), Krea 2, FLUX.2 klein, Anima |

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
  --steps N (50)  --cfg F (3)  --seed N (random)  --count N (images per prompt)   (defaults: each model's own)
  --turbo                  8-step DMD2 LoRA: 8 steps, CFG 1, no negative branch (other models: their distilled settings)
  --lora FILE[:SCALE]      repeatable; kohya (Kromoss/MOSS), ComfyUI, diffusers/PEFT keys (MOSS-Image only)
  --init IMAGE --strength F (0.6)   img2img;   --mask MASK (with --init): inpaint, white = regenerate, soft edges
  --shift F (5)  --skip-layers LIST|none (22)   official sampler knobs (--skip-layers: MOSS-Image only)
  -o PATH|DIR|-  --json  -q  --show-prompt  --meta JSON | --meta-file FILE
  --precision int8|bf16    the diffusion model (default int8 ConvRot when present)
  -m, --model / --dit PATH|NAME   the diffusion model, any supported architecture (a bare NAME is looked up in the model folders)
  --arch moss|krea2|flux2|anima   an architecture's default model when no --model is given
  --text-encoder / --vae PATH|NAME   override the architecture's own
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
`sampler`, `skip_layers`, `precision`, `dit` (the diffusion model file), `width`/`height` (and `generated_width`/`generated_height` when resampled),
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

## Other architectures: Krea 2, FLUX.2 klein, Anima

`--model FILE` (alias `--dit`; a path, or a bare file name looked up in the model folders) takes a checkpoint of any of
four architectures. The architecture is detected from the file's tensor names (the safetensors header only), and the
matching text encoder, VAE, prompt template, schedule and defaults follow. `--arch NAME` picks an architecture's default
model when no `--model` is given (and must match the file when one is). Checkpoints of other architectures are refused by
name (`this is a FLUX.1 model; loki-dream supports ...`): FLUX.1 / Chroma, Qwen-Image, SD / SDXL UNets, SD3, Lumina,
Wan, LTX, HiDream, Cosmos without Anima's adapter, and text encoders or VAEs given by mistake.

| arch | checkpoints | text encoder | VAE | defaults |
|---|---|---|---|---|
| `moss` | Kromoss, MOSS-Image fine-tunes | Qwen3.5-9B | FLUX.2 small | 50 steps, CFG 3 (`--turbo`: DMD2 LoRA, 8 steps, CFG 1) |
| `krea2` | Krea 2 Raw / Turbo and fine-tunes | Qwen3-VL-4B (12 hidden states) | Qwen-Image | Turbo: 8 steps, CFG 1. Raw: 28 steps, CFG 5.5 |
| `flux2` | FLUX.2 klein 4B / 9B, base and distilled | Qwen3-4B / Qwen3-8B (hidden states 9, 18, 27) | FLUX.2 (full) | distilled: 4 steps, CFG 1. base: 50 steps, CFG 4 |
| `anima` | Anima base / aesthetic / turbo | Qwen3-0.6B + T5 query tokens | Qwen-Image | 30 steps, CFG 4. Turbo: 10 steps, CFG 1 |

**Variants.** Distillation does not show in the weights, so the file name decides: Krea 2 runs with Turbo settings unless the
name says `raw` or `base` (Turbo is the released default and community fine-tunes build on it; a Turbo-based fine-tune at
Raw settings burns), a klein file is distilled unless it says `base`, an Anima file is base unless it says `turbo`.
`--turbo` forces the distilled settings and `--steps` / `--cfg` override any default.

**Guidance.** `--cfg` is classic classifier-free guidance everywhere: `uncond + cfg * (cond - uncond)`, 1 = no negative
branch. Krea 2's own convention (`cond + g * (cond - uncond)`, "guidance 4.5") is `--cfg 5.5`.

**Weights.** bf16, fp16, fp32, ComfyUI scaled fp8 (`weight_scale`, `.comfy_quant` or `_quantization_metadata`), plain fp8
casts, ComfyUI NVFP4 (as in `qwen_3_8b_fp8mixed`) and ComfyUI int8 ConvRot, for the diffusion model and the text encoder
alike. Everything that is not int8 ConvRot is dequantized on the GPU as it loads, then either kept in bf16 or rotated and
quantized to int8 ConvRot (W8A8, the same kernels as Kromoss). `--precision` picks; the default is int8 for quantized files
and for float files too large to sit in bf16 next to the activations (Krea 2 bf16, klein 9B bf16), bf16 otherwise. Layers a
quantizer marked `full_precision_matrix_mult` (Krea 2's attention gate and output projection) stay bf16. Text encoders run in
bf16 and are streamed layer by layer, as the Qwen3.5 encoder is. mxfp8 / int4 files are refused.

**Not (yet) for the new architectures:** LoRAs (`--lora` is MOSS-Image only), `--skip-layers`, layout prompts, Anima's
ComfyUI prompt weighting (`(word:1.5)` is read literally, as diffusers does), and FLUX.2's reference-image editing.

```
loki-dream --model krea2_turbo_fp8_scaled --aspect 3:2 -p "a portrait photo of an old fisherman, golden hour"
loki-dream --model krea2_raw_fp8_scaled --steps 28 --cfg 5.5 -p "..."           # Krea 2 Raw (the defaults, spelled out)
loki-dream --model flux-2-klein-base-9b-fp8 --text-encoder qwen_3_8b_fp8mixed -p "..."
loki-dream --model anima-base-v1.0 --negative "worst quality, low quality, score_1, score_2, score_3, blurry" \
           -p "masterpiece, best quality, score_7, safe, 1girl, @artist name, ..."
loki-dream --model anima-turbo-v1.1 --init sketch.png --strength 0.7 -p "..."    # img2img / inpainting work for all four
```

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

**Swapping the diffusion model.** Any fine-tune with the same MOSS-Image architecture (the same tensor names and shapes
as Kromoss) can stand in for it: `--dit FILE` (or `--model`), or `LOKI_DREAM_DIT=FILE` for every run on the machine
(`--dit`, `--arch` and `--precision` override it). Both also accept a Krea 2, FLUX.2 klein or Anima checkpoint, which
switches the whole pipeline (see Other architectures). bf16, fp16 (converted to bf16 as it loads) and int8 ConvRot files work; a bare file name is
looked up in the model folders. The file name is recorded as `dit` in each PNG's generation record. The media server
passes its `dreamModel` config value (config.json) as `--dit`.

First run total: 25 GB (26.2 GB with `--turbo`). The tokenizer (Qwen3.5's 248k BPE, from the repo's `tokenizer/`) is
embedded in the binary (`assets/`).

**The other architectures** look for their files by name the same way (official names first) and download what is missing
from the ungated Comfy-Org / CircleStone repos. The checkpoint itself is usually given with `--model`; `--arch` alone
downloads the default listed here. A ComfyUI model tree works as is: list its `diffusion_models`, `text_encoders` and `vae`
folders in `$LOKI_MODELS`.

| arch | role | file (searched as) | from | size |
|---|---|---|---|---|
| krea2 | diffusion model (`--arch` default) | `krea2_turbo_int8_convrot` (`krea2*turbo*int8`) | [Comfy-Org/Krea-2](https://huggingface.co/Comfy-Org/Krea-2) | 13.5 GB |
| krea2 | text encoder | `qwen3vl_4b_bf16` (`qwen3vl_4b`, `qwen3-vl-4b`, `qwen3vl4b`) | Comfy-Org/Krea-2 | 8.9 GB |
| krea2, anima | VAE | `qwen_image_vae` | Comfy-Org/Krea-2 | 0.25 GB |
| flux2 | diffusion model (`--arch` default) | `flux-2-klein-4b` | [Comfy-Org/Flux-2-Klein-4B](https://huggingface.co/Comfy-Org/Flux-2-Klein-4B) | 7.8 GB |
| flux2 4B | text encoder | `qwen_3_4b` | Comfy-Org/Flux-2-Klein-4B | 8.0 GB |
| flux2 9B | text encoder | `qwen_3_8b` (also `qwen_3_8b_fp8mixed`, NVFP4/fp8) | [Comfy-Org/Flux-2-Klein-9B](https://huggingface.co/Comfy-Org/Flux-2-Klein-9B) | 16.4 GB |
| flux2 | VAE | `flux2-vae` (the full decoder) | Comfy-Org/Flux-2-Klein-4B | 0.34 GB |
| anima | diffusion model (`--arch` default) | `anima-base` (Anima 1.0; `anima-turbo-v1.1`, the distilled 10-step one, downloads when named) | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | 4.2 GB |
| anima | text encoder | `qwen_3_06b_base` | circlestone-labs/Anima | 1.2 GB |

Which klein text encoder a run needs is read from the model (`txt_in` width: 7680 = 4B, 12288 = 9B). The Qwen3 tokenizer
(151k BPE, shared by all three) and Anima's T5 tokenizer (SentencePiece Unigram with its precompiled normalizer) are
embedded too (`assets/qwen3`, `assets/t5`); both match the Hugging Face tokenizers on the fixtures in `tests/`.

## Building

Rust (stable) plus CUDA Toolkit 12.6+ (`nvcc`) and a host C++ compiler, then `cargo build --release`. Same layout as the
siblings: `build.rs` compiles `kernels/*.cu` to fatbins (sm_89 + compute_80 PTX) that are embedded; `cudarc` loads the
driver dynamically.

```
cargo test --release --lib --test tokenizer      # GPU-free (CI): unit tests + tokenizers vs Hugging Face fixtures
cargo run --release --bin dream_check -- ...     # MOSS-Image stage checks against reference dumps (GPU, models; below)
cargo run --release --bin arch_check -- ...      # the same for Krea 2 / FLUX.2 klein / Anima (ref/ref_arch.py dumps)
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
- **Qwen3 text encoder** (Krea 2's Qwen3-VL-4B language model, klein's Qwen3-4B / 8B, Anima's Qwen3-0.6B). Dense GQA
  decoder, head dim 128, q/k RMSNorm and split-half rope (theta 5e6 for Qwen3-VL, whose text-only MRoPE positions are equal
  on all axes, 1e6 otherwise), SwiGLU; sizes read from the checkpoint. Streamed layer by layer like the Qwen3.5 encoder,
  with HF's bf16 rounding (`bf16(bf16(x r) w)` norms, bf16 rope products). Callers tap HF `hidden_states[i]` and/or the
  final-normed state. Klein pads every prompt to 512 tokens like its pipeline (the padding attends to the real tokens
  only, as HF's mask does); Krea 2's pipeline pads in the middle of its template and masks the padding everywhere, so the
  padding is dropped instead (exactly equivalent: no position depends on it).
- **Krea 2 DiT (28 blocks, dim 6144, GQA 48/12 heads of 128, SwiGLU 16384).** Text fusion first, once per prompt: 2
  "layer-wise" blocks attend across each token's 12 tapped hidden states (a small grouped-attention kernel), a learned
  Linear(12 -> 1) collapses them, 2 refiner blocks attend across the tokens, then RMSNorm + Linear-GELU-Linear. Blocks share
  one timestep modulation plus a per-block table, use zero-centred `(1 + w)` RMSNorms (eps 1e-5), q/k norms, 3-axis rope
  (t / h / w = 32 / 48 / 48 dims, theta 1000) and a sigmoid gate on the attention output. q/k/v and the MLP run int8; the
  gate and output projections bf16 (the fp8 checkpoints mark them precision-sensitive).
- **FLUX.2 klein DiT.** 5 + 20 (4B) or 8 + 24 (9B) double / single-stream blocks, LayerNorm modulation, 4-axis rope
  (theta 2000, interleaved pairs), parallel attention + SwiGLU in the single blocks, no biases. Same 128-channel latent
  space as MOSS-Image, so it shares the FLUX.2 VAE code (with the full decoder).
- **Anima.** The LLM adapter (6 blocks: self-attention over the prompt's T5 tokens, cross-attention to the Qwen3-0.6B states,
  GELU MLP; head dim 64 through the flash kernel) turns the prompt into 512 context tokens; the Cosmos-Predict2 DiT (28
  blocks, dim 2048) cross-attends to them, with the keys and values of that context computed once per prompt for every
  block. AdaLN-LoRA modulation, 3D rope with the extrapolation ratio 4 on height and width, GELU MLP.
- **Qwen-Image VAE** (Krea 2, Anima). The Wan 2.1 causal 3D VAE reduced to images: for one frame every 3x3x3 convolution
  only meets the zero padding in front, so the last temporal slice of each kernel is all that runs; RMS norms, nearest-2x
  up / zero-padded stride-2 down resamplers and single-head mid attention on the same implicit-GEMM convolution kernels
  as the FLUX.2 VAE. Latents are standardized with the Wan 2.1 statistics and packed 2x2 for the DiTs.
- **Flow sampler** (Krea 2, klein, Anima). diffusers' `FlowMatchEulerDiscreteScheduler`: `linspace(1, 1/n, n)` shifted
  by a resolution-dependent `mu` (klein's empirical mu; Krea 2's 0.5 .. 1.15 over 256 .. 6400 tokens, 1.15 for Turbo) or
  statically (Anima: 3), Euler on fp32 latents, the same img2img and inpainting as MOSS-Image. The initial noise is
  torch.randn of each pipeline's latent shape, so seeds match diffusers.
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

### Krea 2, FLUX.2 klein, Anima

`ref/ref_arch.py` runs diffusers 0.42's `Krea2Pipeline` / `Krea2Transformer2DModel`, `Flux2KleinPipeline` /
`Flux2Transformer2DModel` and the Anima modular pipeline (`CosmosTransformer3DModel` + `AnimaTextConditioner`), with
transformers' Qwen3 / Qwen3-VL text models, on the same single files (fp8 dequantized to bf16); `arch_check` runs the native
stage on the same inputs. Configs and tokenizers come from the ungated diffusers repos (`ref_src/`, see the script's header).

```
tools/ref-env/Scripts/python -I ref/ref_arch.py te   flux2 ref_out/flux2 --te qwen_3_4b.safetensors -p "..." -n ""
tools/ref-env/Scripts/python -I ref/ref_arch.py dit  flux2 ref_out/flux2 --dit flux-2-klein-base-4b-fp8.safetensors --size 768x512 --sigma 0.8
tools/ref-env/Scripts/python -I ref/ref_arch.py vae  anima ref_out/qvae --vae qwen_image_vae.safetensors --image testdata/apple.png
tools/ref-env/Scripts/python -I ref/ref_arch.py full krea2 ref_out/krea2 --dit ... --te ... --vae ... -p "..." --steps 8 --cfg 1 --distilled
target/release/arch_check te|dit|vae ARCH DIR --te FILE | --dit FILE [--int8|--bf16] | --vae FILE
```

Text conditioning (cos vs the bf16 reference):

| | native vs reference |
|---|---|
| klein, Qwen3-4B taps 9/18/27, prompt tokens | 0.999998 (the 512-token padding: 0.991-0.996, about the reference's own bf16-vs-fp32 spread there) |
| Krea 2, Qwen3-VL-4B, 12 taps | 0.99977-0.99992 |
| Anima, Qwen3-0.6B last state | 0.99939-0.99988 (vs an fp32 run: native relL2 0.022, the bf16 reference 0.052) |
| Anima, LLM adapter output | 0.99979-0.99996 (empty negative included) |
| token ids (Qwen3, T5) | identical |

One DiT forward (768x512, sigma 0.8, three prompts; cos / relative L2):

| model | bf16 | int8 |
|---|---|---|
| FLUX.2 klein base 4B | 0.99997 / 0.0074 | 0.99990 / 0.014 |
| Anima base | 0.99998 / 0.0069 | 0.99993 / 0.012 |
| Krea 2 Turbo | | 0.9973-0.99975 / 0.022-0.073 |

Krea 2 is unusually sensitive: the bf16 reference against an fp32 run of itself scores 0.92-0.997 on the same inputs, and the
native output is exactly as close to that fp32 run as the reference is. VAEs: Qwen-Image decode 56.1 dB, encode cos
0.99999; FLUX.2 (full decoder) decode 57.5 dB.

Whole images at a fixed seed, PSNR vs the official pipeline's image:

| run | native |
|---|---|
| FLUX.2 klein base 4B, 768², 20 steps, CFG 4 | 38.5 dB (bf16), 37.9 dB (int8) |
| Krea 2 Turbo, 768², 8 steps | 43.4 dB (int8) |
| Anima base, 768², 30 steps, CFG 4 | 25.5 dB (bf16; same composition, the MOSS-Image range above) |

## Performance (RTX 4090, Windows, model files in the OS cache)

| 1024x1024 | sampling | end to end | peak VRAM (above the desktop) |
|---|---|---|---|
| int8, 50 steps, CFG 3 | 16.8 s (0.17 s per DiT pass) | 21.6 s | 7.6 GB |
| int8, `--turbo` (8 steps) | 2.6 s | 7.5 s | 7.6 GB |
| bf16, 50 steps, CFG 3 | 38.8 s | 45.3 s | 12.0 GB |
| bf16, `--turbo` | 4.3 s | 10.3 s | 12.0 GB |
| Krea 2 Turbo (fp8 file, int8), 8 steps | 4.6 s | 9.3 s | |
| FLUX.2 klein base 4B (fp8 file, int8), 50 steps, CFG 4 | 13.0 s | 16.5 s | |
| FLUX.2 klein 9B (fp8 file, int8), 4 steps | 1.0 s | 5.2 s | |
| Anima base (bf16), 30 steps, CFG 4 | 8.6 s | 11.1 s | |

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
- **Krea 2, FLUX.2 klein, Anima.** They follow the diffusers pipelines (the official ones for Krea 2 and klein, the
  CircleStone-published modular pipeline for Anima), not ComfyUI: flow Euler on diffusers' schedules (ComfyUI users of Anima
  often pick er_sde or euler_a), Anima prompts read literally (ComfyUI parses `(word:1.5)` weights), no klein reference-image
  editing. Latents stay fp32 between steps where the Krea 2 and klein pipelines keep them in bf16. Krea 2's prompt padding is
  dropped rather than masked (exactly equivalent). Distilled variants are recognised by file name (see above).
- **Tokenizer.** The NFC normalizer and the Qwen2 split pattern are reimplemented. Letters/digits use Rust's Unicode
  `alphabetic`/`numeric`, which can differ from the regex's `\p{L}`/`\p{N}` on rare scripts.

## Gotchas found on the way

- Casting a transformers model with `.to(bf16)` also casts its rope `inv_freq` buffer, which quietly costs the reference
  precision at large positions (klein's 512-token padding); `ref_arch.py` restores it in fp32, as `from_pretrained` keeps it.
- The flash attention kernel writes nothing for a query whose keys are all masked; Anima's empty negative prompt masks its
  whole source, so that cross-attention is skipped (its output is exactly zero in the reference).

- The reference harness must rebuild Qwen3.5's non-persistent `inv_freq` buffers after a meta-device load, or RoPE
  silently becomes the identity (`ref_common.load_text_encoder` does). The official `text_encoder/` folder rounds `A_log`
  and the DeltaNet norm weight to bf16, while the single file keeps them fp32. Both sides round to bf16, as the official
  pipeline does.
- Float `atomicAdd` in the attention's K/V mean (inherited from loki-retouch's `sage_attn.cu`) made runs differ by
  ~26 dB through the diffusion. It is now a fixed-order two-pass reduction. loki-retouch still has the atomic version.
- Unmapping a 19 GB mapping on Windows takes ~3 s; it now happens on a background thread.
