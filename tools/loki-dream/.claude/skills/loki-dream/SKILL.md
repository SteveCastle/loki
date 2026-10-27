---
name: loki-dream
description: Generate images from text with the local loki-dream CLI (Kromoss, a MOSS-Image fine-tune; Rust/CUDA, no Python/ComfyUI). Use when the user wants a new image from a prompt, several variations, a fast 8-step draft (--turbo), an img2img variation of an existing picture, inpainting a masked region, a LoRA applied, or a structured layout prompt (text and objects placed in boxes). Composes with loki-retouch / loki-segment / loki-reshoot through stdin/stdout.
---

# loki-dream: prompt in, image out

`loki-dream` runs Kromoss (lodestones' full fine-tune of OpenMOSS MOSS-Image, 5B DiT) with the Qwen3.5-9B text encoder and
the FLUX.2 small VAE, locally on the NVIDIA GPU (RTX 4090 class, 24 GB) in a standalone engine. It is one of the `loki-`
toolkit's small composable tools: `loki-retouch` edits images, `loki-segment` makes masks, `loki-reshoot` animates them.

## 0. Locate and sanity-check
- Source: the loki monorepo, `tools/loki-dream` (`cargo build --release` -> `target/release/loki-dream.exe`); installed copies live in
  `C:\Users\steph\bin` with the models in `bin\models`. Missing models are downloaded once from Hugging Face (lodestones/Kromoss:
  5.4 GB int8 DiT + 19.3 GB text encoder + 0.25 GB VAE, + 1.2 GB for --turbo; a notice prints even with `-q`). `LOKI_MODELS`
  points at another model folder.
- The GPU must be free: check `tasklist` for other loki-*.exe jobs and `nvidia-smi` before running; never start while another big
  GPU job runs.
- ALWAYS read `loki-dream --help` first (OUTPUT / SIZE / SAMPLING / IMG2IMG / LAYOUT PROMPTS / EXAMPLES). The binary is the truth.

## Conventions shared by all loki- tools
- stdout = results only: the path of every image written, one per line (`--json`: one JSON object for the run); `-o -` = the PNG.
- stderr = progress (`step i/n`), timings, warnings; `-q` silences them. Exit 0 ok, 1 error, 2 usage.
- Every PNG carries its generation record (prompt, seed, steps, cfg, size, LoRAs, command line) in an iTXt chunk `loki`.
  To vary or reproduce a result, read it instead of guessing.

## What to run
| Intent | Command |
|---|---|
| one image | `loki-dream -p "a tabby cat holding a sign that says HELLO" -o cat.png` |
| fast drafts (8 steps, ~10x faster) | `loki-dream --turbo --count 4 -o drafts/ -p "..."` |
| best quality | default: 50 steps, CFG 3 (official settings) |
| widescreen / portrait | `--aspect 16:9` or `--size 1344x768` (about 1 MP is the sweet spot; multiples of 16) |
| reproduce | `--seed N` (same seed + settings = same image, also matching diffusers/ComfyUI noise) |
| variation of a picture | `loki-dream --init photo.jpg --strength 0.5 -p "same scene in winter"` (0.3 subtle .. 0.8 loose) |
| inpaint a region | `loki-dream --init photo.jpg --mask region.png --strength 1 -p "..."` (white = redraw; mask from loki-segment) |
| a style/character LoRA | `--lora file.safetensors:0.8` (repeatable; stacks with --turbo) |
| text + layout | a layout JSON as the prompt (see --help LAYOUT PROMPTS) |
| check the encoder input | `--show-prompt` |

## Prompts
- Describe the image (subject, setting, style, lighting, camera); quoted text in the prompt is rendered as text.
- `--negative` only matters with CFG > 1 (not --turbo).
- Masks from loki-segment chain directly: `loki-segment -p sky -o sky.png photo.jpg && loki-dream --init photo.jpg --mask sky.png --strength 1 -p "a sky full of balloons"`.
