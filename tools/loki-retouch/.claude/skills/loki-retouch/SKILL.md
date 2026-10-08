---
name: loki-retouch
description: Edit, upscale, restore or composite images with the local loki-retouch CLI (Qwen Image 2.1, Rust/CUDA, no ComfyUI). Use when the user wants to change the content of a photo with a text instruction, upscale or restore an image, combine several images, outpaint, or make a 4K/phone wallpaper from a photo (--preset 4kify), or wants a quick image-editing building block in a pipeline. Composes with loki-reshoot (images -> video) through stdin/stdout.
---

# loki-retouch: image in, edited image out

`loki-retouch` changes the content of images from a text instruction ("make it night", "put the jacket from <image2> on <image1>"),
upscales them faithfully (real super-resolution, ratio kept), restores old photos, composites several images and outpaints. It runs the
Qwen Image 2.1 edit model locally on the NVIDIA GPU (RTX 4090 class, 24 GB) in a standalone engine. It is one of the `loki-` toolkit's
small, composable tools: see the sibling `loki-reshoot` (references -> video with sound).

## 0. Locate and sanity-check
- Binary: `loki-retouch` is on PATH (installed in `C:\Users\steph\bin`, with the three models in `bin\models` next to it), so run it
  from any directory. Source: the loki monorepo, `tools/loki-retouch`; to update the installed copy, `cargo build --release` there and
  copy `target/release/loki-retouch.exe` over the one in `C:\Users\steph\bin`. If it can't find the models it downloads ~17 GB from Hugging
  Face (it prints a notice even with `-q`); `LOKI_MODELS` points it at another model folder. CI builds release binaries for Windows and Linux.
- The GPU must be free (~12-20 GB). If ComfyUI or another big GPU job is running, the run fails with an allocation error. Check
  `nvidia-smi --query-gpu=memory.used --format=csv`; ask before stopping the user's processes.
- ALWAYS read the live docs first: `loki-retouch --help` (all options, SIZE rules, REFERENCES, EXAMPLES) and `loki-retouch --list-presets`.
  The binary is the source of truth if this skill and it disagree.

## Conventions shared by all loki- tools (this is what makes them composable)
- Inputs: files, directories, globs, or `-` = an image on stdin. Every input is one job; the models load once for all of them.
- Outputs: the path of each written file is printed on **stdout** (`--json`: one JSON object per result); `-o -` writes the PNG itself to
  stdout. Everything else (progress, timings) goes to **stderr**; `-q` silences it. Never parse stderr.
- Exit status: 0 ok, 1 error, 2 usage error. One input failing in a batch reports `skipping <file>: ...` and exits 1 at the end.
- Shared env: `LOKI_MODELS` (model folder), `LOKI_PROFILE=1` (per-kernel timing).
- Chaining: `loki-retouch --preset restore -o - old.jpg | loki-retouch -p "colorize" - -o - | loki-reshoot --animate - -d 5 -o alive.mp4`

## What to run (pick by intent)
| Intent | Command |
|---|---|
| change content, keep size | `loki-retouch -p "turn the scene into a snowy winter day, keep the people and composition" photo.jpg` → `photo_edit.png` |
| change content + resize | `loki-retouch --scale 1.5 -p "replace the sky with a sunset" photo.jpg` |
| faithful upscale | `loki-retouch --upscale 2 small.jpg` (any factor; `--preset upscale --scale 4`) |
| restore an old/compressed photo | `loki-retouch --preset restore old.jpg` |
| composite / transfer | `loki-retouch -p "put the jacket of <image2> on the person of <image1>" -r jacket.png me.png` |
| 4K desktop wallpaper | `loki-retouch --preset 4kify photo.jpg` → `photo_4k.png` (3840x2160); `--preset 4kify-phone` → 1296x2800 |
| frames of a clip | `loki-retouch --preset 4kify --seq -o out frames/` (one seed, framing pinned) |
| batch | `loki-retouch -q --preset restore -o outdir "shots/*.jpg"` (result paths on stdout) |

## Sizes (default: same as the input)
`--size same|WxH`, `--scale F` (each side, ratio kept), `--width/--height/--long-edge PX`, `--megapixels MP`, `--upscale [N]`. A preset's own size
(4kify) wins over the default; any explicit flag wins over the preset. The model works in multiples of 16: other sizes are generated at the
nearest multiple and Lanczos-resampled to exactly the size asked for (`--snap` keeps the native multiple). Memory/time grow with output
pixels (4K ≈ 100 s end to end; 512x640 ≈ 15 s including ~10 s of model loading).

## Prompts
- Write an instruction, not a description: "make it night, with rain", "remove the person on the left", "change the jacket to red leather".
  Add "keep the people and composition" (or similar) when only part of the image should change.
- Refer to images as `<image1>` (the input), `<image2>`... (the `-r` references, in order). The tags are optional for one image.
- A preset's prompt is replaced by `--prompt`/`-P FILE`, or extended with `--append TEXT`. `--show-prompt` prints the final prompt and exits.
- Upscaling and content edits combine: `--scale 2 -p "add soft film grain"`.

## Check the result and iterate
- Open the PNG (or build a before/after montage) and look; compare sizes with `python -c "from PIL import Image; print(Image.open(p).size)"`.
- Different outcome: change `--seed N` (default 0, incremented per input), then `--steps` (default 25), then the wording. The engine is slightly
  nondeterministic run to run (~44 dB PSNR between identical runs), so tiny pixel differences are normal.
- Errors: "allocating ... OUT_OF_MEMORY" → free the GPU or lower `--size`/`--scale`; "unknown preset" → `--list-presets`.
