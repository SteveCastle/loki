# loki-retouch

**AI image editing from the command line.** Change the content of a photo with a text instruction, upscale it
faithfully, restore it, composite several images, or outpaint it into a 4K wallpaper, with **Qwen Image 2.1** (or
**MiniMax H3 Image** with `--model h3`: up to 16 pictures, native 4-16 MP), as a single self-contained binary. No ComfyUI, no Python, no PyTorch: the whole inference engine (int8/bf16/fp8
tensor-core kernels, flash attention, VAE convolutions) is written from scratch in CUDA and driven from Rust.
The only runtime dependency is the NVIDIA driver.

```
loki-retouch -p "make it night, with rain" photo.jpg          # edit content   -> photo_edit.png
loki-retouch --upscale 2 small.jpg                            # faithful 2x    -> small_up.png
loki-retouch --preset 4kify photo.jpg                         # restore + outpaint to a 4K wallpaper
loki-retouch -p "put the jacket of <image2> on <image1>" -r jacket.png me.png
loki-retouch --model h3 -p "<image1> at night in the rain, same framing" --long-edge 4096 photo.jpg   # MiniMax H3 Image
```

## Part of the loki- toolkit

`loki-retouch` is one of a family of small, composable media tools that share one set of conventions, so they
chain with each other and with ordinary Unix tools, and are easy for agents to drive:

| tool | does | engine |
|---|---|---|
| **loki-retouch** (this repo) | image editing, upscaling, restoration, compositing | Qwen Image 2.1, MiniMax H3 Image |
| [loki-reshoot](../loki-reshoot) | reference images / videos / audio -> video with sound | MiniMax H3 |

The shared conventions (also see `../CONVENTIONS.md`):

- **Inputs** are files, globs, directories, or `-` for stdin; **results** are written to a file whose path is printed
  on stdout (`--json` prints a JSON object instead), or streamed as the media itself with `-o -`.
- **Progress and diagnostics go to stderr only**; `-q` silences them. Exit status: 0 ok, 1 error, 2 usage error.
- Every PNG written carries a `loki` text chunk: the prompt, preset, seed, steps, size, inputs and command line as JSON
  (the Lowkey Media Viewer shows it in its Generation panel; see `../CONVENTIONS.md`, "Provenance").
- `-p/--prompt`, `-P/--prompt-file`, `--seed`, `--steps`, `--show-prompt` mean the same thing in every tool.
- Models are looked up next to the binary, in `./models`, and in `$LOKI_MODELS`; missing ones are downloaded
  (and the download notice is printed even with `-q`).
- `lokictl` (the Lowkey Media Server client) is separate and unrelated: these tools never talk to a server.

```
loki-retouch --preset restore -o - old.jpg | loki-retouch -p "colorize" - -o - | loki-reshoot --animate - -d 5 -o alive.mp4
```

Agent skills for both tools live in `.claude/skills/` (Claude Code and OpenCode read that folder).

## Usage

`loki-retouch` edits content (any prompt), upscales faithfully, restores, composites several images, and
outpaints. Presets (`--preset NAME`) bundle a prompt, an output size and reference sizing: `4kify` is the
restore + outpaint into a 4K wallpaper treatment this project started as, `4kify-phone` the vertical
variant, `upscale` faithful super-resolution, `restore` restoration at the input's own size.

```
loki-retouch [OPTIONS] [INPUT]...

  INPUT                files, directories, globs ("photos/*.jpg"), or - for stdin; each input is one job (models load once)
  -p, --prompt TEXT    what to do (-P FILE reads it from a file, - = stdin); <image1>, <image2>... name the images
  -r, --ref FILE       extra reference image for every job (<image2>, ...), repeatable
  --preset NAME        4kify | 4kify-phone | upscale | restore | none      (--list-presets)
  --append TEXT        extend a preset's prompt;  --prompt replaces it
  SIZE (default: same as the input; a preset's own size wins over that; an explicit flag wins over the preset)
  --size same|WxH      --scale 2 (each side, ratio kept)   --width/--height/--long-edge PX   --megapixels MP
  --upscale [N]        = --preset upscale --scale N (faithful super-resolution, default 2x)
  --snap               keep the model's multiple-of-16 size (otherwise results are resampled to exactly the requested size)
  -o, --out PATH|DIR|- output (default <name>_edit.png next to the input; the preset's suffix e.g. _4k; stdin -> stdout)
  --steps N (25)  --seed N  --shift F (0.69)  --ref-resolution N  --json  -q  --show-prompt  --meta JSON|--meta-file FILE
  --dit / --text-encoder / --vae PATH   model files; by default discovered next to the binary or in ./models
```

Examples:

```
loki-retouch -p "make it night, with rain" photo.jpg                   # content edit at the input's size -> photo_edit.png
loki-retouch --upscale 2 small.jpg                                     # faithful 2x super-resolution, ratio kept
loki-retouch --scale 1.5 -p "replace the sky with a sunset" photo.jpg  # content edit + 1.5x
loki-retouch -p "put the jacket of <image2> on <image1>" -r jacket.png me.png
loki-retouch --preset restore -o - old.jpg | loki-retouch -p "colorize" - -o - | loki-retouch --preset 4kify - -o wall.png
loki-retouch --preset 4kify photo.jpg                           # -> photo_4k.png (3840x2160)
loki-retouch --preset 4kify-phone photo.jpg                     # -> photo_phone.png (1296x2800)
loki-retouch --preset 4kify -q -o wallpapers "camera/*.jpg"      # batch; the result paths are printed on stdout
loki-retouch --preset 4kify --seq -o out frames/                 # the frames of a clip: one seed, framing pinned
```

Batch mode is phase-ordered: every input is encoded (VAE + text encoder), then the text encoder is
released and the DiT samples the images one by one, each one decoded and written as soon as it is
sampled. Each of the two large models is loaded exactly once per run. The model needs sizes in multiples
of 16; any other requested size (e.g. `--size same` on a 1001x747 image) is generated at the nearest
multiple and Lanczos-resampled to exactly what you asked for.

Sequence mode (`--seq`, for the frames of a clip extracted in order, e.g. `loki-retouch --preset 4kify --seq -o out frames/`
then `ffmpeg -framerate 10 -i out/frame_%03d_4k.png -c:v libx264 -crf 16 -pix_fmt yuv420p clip.mp4`) uses
one seed for every frame and a stricter prompt: a "faithful enlargement" restoration paragraph instead of
the free restoration, and a framing paragraph that spells out where the input sits on the canvas (full
height, centred, nothing cropped), because independently restored frames otherwise drift in zoom by about
+-15%. See `loki-retouch --preset 4kify --seq --show-prompt`.

Tried for video and dropped, for the record: feeding a restored frame back as a second reference pins the
framing but the model copies its rendering and over-restores it into grain unless it is shrunk to the
source's size; seeding each frame from the previous output's latent locks the geometry to the previous
frame, and chaining frames (even warped by optical flow) compounds errors until the clip collapses within
ten frames; seeding every frame from one flow-warped keyframe is stable but needs a Python driver with
OpenCV and brings its own seams and occlusion handling. The measured results of every attempt, the
evaluation scripts (`tools/metrics`) and a ranked plan for a re-attempt are in the
"video upscaling plan" doc: https://claude.ai/code/artifact/6626f241-bb72-4324-80c9-fd7916a8a886

The wallpaper prompt is parameterized by orientation; see `loki-retouch --preset 4kify --show-prompt` (and `--list-presets`).

## Models

Put these next to `loki-retouch.exe` (or in a `models/` folder next to it, or in `$LOKI_MODELS`), or pass them explicitly. Any that
can't be found are downloaded automatically (~17 GB total) from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) into `models/` next to the
binary; interrupted downloads resume on the next run.

| flag             | file                                        |
|------------------|---------------------------------------------|
| `--dit`          | `qwen_image_2.1_int8_convrot.safetensors`   |
| `--text-encoder` | `qwen3vl_8b_int8_convrot.safetensors`       |
| `--vae`          | `qwen_image_2.1_vae_bf16.safetensors`       |

The int8 files are ComfyUI's "int8 ConvRot" format (per-row int8 with a 256-wide Hadamard rotation).

## Second model: MiniMax H3 Image (`--model h3`)

`--model h3` edits with **MiniMax H3 Image** instead: MiniMax's open 33.1B DiT (text to image and editing in one
checkpoint, distilled to 6 steps at CFG 1) with the Qwen3-VL-32B conditioning encoder of the H3 family and its own image
VAE, run by loki-reshoot's H3 engine (linked as a library; the same DiT, encoder and kernels as its video model). What it
adds over Qwen Image 2.1:

- **up to 16 pictures** per edit: the input is `<Picture 1>` and every `-r` adds `<Picture 2>`, `<Picture 3>`, ...;
  `<imageN>` tags in prompts and presets are translated, so commands written for Qwen Image work unchanged;
- **native 4-16 MP output**: `--long-edge 4096`, `--megapixels 16`, `--upscale 2` on a 2K photo run in one pass;
- **6 steps**: a 4 MP edit samples in ~4 s once the models are loaded (~35-50 s cold on a 24 GB card, where the 15 GB
  text encoder and the 21 GB DiT load in turn; several inputs share one load of each).

```
loki-retouch --model h3 -p "Make <image1> a rainy night street, neon reflections on the wet asphalt, same framing" photo.jpg
loki-retouch --model h3 -p "The man from <image1> sits in the armchair of the bookshop in <image2>, reading" -r shop.png me.png
loki-retouch --model h3 --long-edge 4096 -p "Restore <image1>: remove the scratches and grain, same framing" old.jpg
```

Sizing follows the official pipeline: the output keeps the input's size (or what a size flag / preset says), drawn on a
32 px grid of at least 4 MP and Lanczos-resampled to exactly that size (`--snap` keeps the drawn canvas). The pictures are
given to the text encoder at a 1024 px short edge and to the DiT as VAE latents at a 1024-2048 px short edge (ComfyUI's
`MiniMaxH3ImageConditioning`). Long, specific English descriptions of the result work best. Not with `--model h3`:
`--shift` and `--ref-resolution` (Qwen Image settings); the presets were tuned on Qwen Image but run on either model.
An H3 Image checkpoint given with `--dit` selects the model by itself.

| flag | file | from | size |
|---|---|---|---|
| `--dit` | `minimax_h3_image_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | 21.0 GB |
| `--text-encoder` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (shared with loki-reshoot) | Comfy-Org/MiniMax-H3-Image | 15.7 GB |
| `--vae` | `minimax_h3_image_vae_fp16.safetensors` | Comfy-Org/MiniMax-H3-Image | 1.2 GB |

Validation (ComfyUI 0.45's own implementation, scripts and checks in loki-reshoot's `ref/` and `src/bin/`): the same
edit prompt, picture and seed give the same result as ComfyUI (22.3 dB PSNR; differences are int8 rounding carried
through the steps), the image VAE decodes at 58.6 dB.

## Building

Requirements: Rust (stable, MSVC toolchain), CUDA Toolkit 12.x (`nvcc`), Visual Studio 2022 C++ tools.

```
cargo build --release          # -> target/release/loki-retouch.exe
cargo test --release           # kernel tests against CPU references
cargo run --release --bin bench
```

`build.rs` compiles `kernels/*.cu` with nvcc into fatbins (sm_89 SASS + compute_80 PTX) that are
embedded in the binary. `cudarc` loads the driver (`nvcuda.dll`) dynamically.

## Tests

`cargo test --release` runs the kernel tests against CPU references and the prompt/preset unit tests. `ref/` holds the
scripts that dump reference tensors from ComfyUI's own code (generated dumps are git-ignored; this repo never tracks
media or test fixtures). `te_check` and `dit_check` compare this engine against such dumps.

## Engine notes

- **GEMMs**: hand-written mma.sync kernels. int8 x int8 (activations quantized per token with the
  same Hadamard rotation as the weights) with the dequantization, bias, residual and gating fused
  into the epilogue; bf16 for the unquantized layers and the VAE (implicit-GEMM convolutions on
  channels-last planes). ~460 TOPS int8 / ~150 TFLOPS bf16 on an RTX 4090.
- **Attention**: a flash-attention kernel (bf16) for the text encoder and vision tower, and a
  SageAttention-style kernel for the DiT: int8 QKᵀ with per-token scales and per-channel K smoothing,
  fp8 (e4m3) P·V with per-channel V smoothing and per-tile scales, fused online softmax.
- **Prefix KV cache**: the text and reference-image tokens do not depend on the timestep (they are
  modulated at t = 0), so their K/V are computed once per image and cached (quantized) for all steps;
  every step only runs the target-image tokens.
- **Memory**: weights are memory-mapped and uploaded directly; VAE residual blocks run in row strips
  with halos so 4K decoding never materializes full-resolution intermediates; the reference image is
  capped at the output's pixel budget.
- **Reference size**: a small input (a 400 px video frame) is upscaled to 1024 before the model sees it.
  At native size it gives the DiT a 26x16 token grid centred on a 240x135 target, too little aligned
  structure to copy, and the output re-draws eyes, fingers and contours (fidelity to the aligned source
  SSIM 0.91 vs 0.94 at 1024; 2048 gains little more and shifts the tone). The prompt alone does not fix it.
- Environment switches: `LOKI_ATTN=bf16` (bf16 attention + bf16 cache), `LOKI_PROFILE=1`
  (per-kernel timing of the DiT step), `LOKI_GEMM_NARROW=1` (128x128 GEMM tiles only).

## Performance (RTX 4090, 25 steps)

| job                                      | time   |
|------------------------------------------|--------|
| 3840x2160 from a 900x1600 photo          | ~100 s end to end (about 3.3 s per step, ~10 s of model loading) |
| 1296x2800 phone wallpaper                | ~40 s  |
| per-step breakdown at 4K                 | attention 2.0 s, int8 GEMMs 1.0 s, everything else 0.3 s |

The first ComfyUI-equivalent run of the same workflow in this engine took 175 s; the Sage-style attention,
the 128-key tiles, fused epilogues and grouped GEMM rasterization brought it to ~100 s.

## Validation

`ref/ref_te.py` and `ref/ref_dit.py` dump references from ComfyUI's own code (run from the ComfyUI
directory with its embedded Python); `te_check` and `dit_check` compare this engine against them.
On the test image: text encoder cosine 0.999, one DiT step cosine 0.996-0.997 (ComfyUI's own
SDPA-vs-Sage backends differ by 0.9956), VAE encode 0.9999, VAE decode 0.99998.
