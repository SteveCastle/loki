# 4kify

Restore a photo and outpaint it into a 4K desktop wallpaper (or a vertical phone wallpaper) with
**Qwen Image 2.1**, as a single self-contained Windows binary. No ComfyUI, no Python, no PyTorch:
the whole inference engine (int8/bf16/fp8 tensor-core kernels, flash attention, VAE convolutions)
is written from scratch in CUDA and driven from Rust. The only runtime dependency is the NVIDIA driver.

It reproduces the "Qwen Image 2.1 restore + outpaint" ComfyUI workflow (Qwen3-VL 8B conditioning with
the reference image, euler / simple schedule, 25 steps, cfg 1, shift 0.69) and was validated layer by
layer against ComfyUI's implementation (text encoder, one DiT step, VAE encode/decode).

## Usage

`qedit` is a generic image-editing CLI for Qwen Image 2.1: it edits content (any prompt), upscales
faithfully, restores, composites several images, and outpaints. `4kify` is the same program with
`--preset 4kify` as its default (restore + outpaint into a 4K wallpaper). Both follow Unix conventions so
they chain: images in from files, globs, directories or stdin (`-`); results out as a file (its path is
printed on stdout, or one JSON object per result with `--json`) or as PNG bytes on stdout (`-o -`);
progress goes to stderr only (`-q` silences it); non-zero exit status on failure.

```
qedit [OPTIONS] [INPUT]...

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
  --steps N (25)  --seed N  --shift F (0.69)  --ref-resolution N  --json  -q  --show-prompt
  --dit / --text-encoder / --vae PATH   model files; by default discovered next to the binary or in ./models
```

Examples:

```
qedit -p "make it night, with rain" photo.jpg                   # content edit at the input's size -> photo_edit.png
qedit --upscale 2 small.jpg                                     # faithful 2x super-resolution, ratio kept
qedit --scale 1.5 -p "replace the sky with a sunset" photo.jpg  # content edit + 1.5x
qedit -p "put the jacket of <image2> on <image1>" -r jacket.png me.png
qedit --preset restore -o - old.jpg | qedit -p "colorize" - -o - | qedit --preset 4kify - -o wall.png
4kify photo.jpg                                                 # -> photo_4k.png (3840x2160)
4kify --phone photo.jpg                                         # -> photo_phone.png (1296x2800)
4kify -q -o wallpapers "camera/*.jpg"                           # batch; the result paths are printed on stdout
4kify --seq -o out frames/                                      # the frames of a clip: one seed, framing pinned
```

Batch mode is phase-ordered: every input is encoded (VAE + text encoder), then the text encoder is
released and the DiT samples the images one by one, each one decoded and written as soon as it is
sampled. Each of the two large models is loaded exactly once per run. The model needs sizes in multiples
of 16; any other requested size (e.g. `--size same` on a 1001x747 image) is generated at the nearest
multiple and Lanczos-resampled to exactly what you asked for.

Sequence mode (`--seq`, for the frames of a clip extracted in order, e.g. `4kify --seq -o out frames/`
then `ffmpeg -framerate 10 -i out/frame_%03d_4k.png -c:v libx264 -crf 16 -pix_fmt yuv420p clip.mp4`) uses
one seed for every frame and a stricter prompt: a "faithful enlargement" restoration paragraph instead of
the free restoration, and a framing paragraph that spells out where the input sits on the canvas (full
height, centred, nothing cropped), because independently restored frames otherwise drift in zoom by about
+-15%. See `4kify --seq --show-prompt`.

Tried for video and dropped, for the record: feeding a restored frame back as a second reference pins the
framing but the model copies its rendering and over-restores it into grain unless it is shrunk to the
source's size; seeding each frame from the previous output's latent locks the geometry to the previous
frame, and chaining frames (even warped by optical flow) compounds errors until the clip collapses within
ten frames; seeding every frame from one flow-warped keyframe is stable but needs a Python driver with
OpenCV and brings its own seams and occlusion handling. The measured results of every attempt, the
evaluation scripts (`tools/metrics`) and a ranked plan for a re-attempt are in the
"4kify video upscaling plan" doc: https://claude.ai/code/artifact/6626f241-bb72-4324-80c9-fd7916a8a886

The wallpaper prompt is parameterized by orientation; see `4kify --show-prompt` (and `qedit --list-presets`).

## Models

Put these next to `4kify.exe` (or in a `models/` folder next to it), or pass them explicitly. Any that
can't be found are downloaded automatically (~17 GB total) from
[Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1) into `models/` next to the
binary; interrupted downloads resume on the next run.

| flag             | file                                        |
|------------------|---------------------------------------------|
| `--dit`          | `qwen_image_2.1_int8_convrot.safetensors`   |
| `--text-encoder` | `qwen3vl_8b_int8_convrot.safetensors`       |
| `--vae`          | `qwen_image_2.1_vae_bf16.safetensors`       |

The int8 files are ComfyUI's "int8 ConvRot" format (per-row int8 with a 256-wide Hadamard rotation).

## Building

Requirements: Rust (stable, MSVC toolchain), CUDA Toolkit 12.x (`nvcc`), Visual Studio 2022 C++ tools.

```
cargo build --release          # -> target/release/4kify.exe
cargo test --release           # kernel tests against CPU references
cargo run --release --bin bench
```

`build.rs` compiles `kernels/*.cu` with nvcc into fatbins (sm_89 SASS + compute_80 PTX) that are
embedded in the binary. `cudarc` loads the driver (`nvcuda.dll`) dynamically.

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
- Environment switches: `FOURKIFY_ATTN=bf16` (bf16 attention + bf16 cache), `FOURKIFY_PROFILE=1`
  (per-kernel timing of the DiT step), `FOURKIFY_GEMM_NARROW=1` (128x128 GEMM tiles only).

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
