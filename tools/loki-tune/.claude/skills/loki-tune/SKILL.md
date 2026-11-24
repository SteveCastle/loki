---
name: loki-tune
description: Train LoRAs for MOSS-Image models (Kromoss bf16/int8, or any MOSS-Image checkpoint) with the local loki-tune CLI (Rust/CUDA, no Python/kohya/ComfyUI). Use when the user wants a LoRA of a person/character, a style, an object/product or a concept from a set of images, wants to caption a training set, check or compare training checkpoints and previews, continue or extend a training run, or branch from an earlier checkpoint. The resulting LoRA loads in loki-dream (--lora) and ComfyUI.
---

# loki-tune: images in, LoRA out

`loki-tune` trains a LoRA on the frozen MOSS-Image DiT (default Kromoss bf16, else int8; `--model` takes any MOSS-Image
checkpoint) on the local NVIDIA GPU (24 GB). It captions the images with loki-scribe, encodes them (Qwen3.5-9B, FLUX.2 VAE),
trains with checkpoints + preview images, and resumes from any checkpoint. One run = one folder.

## 0. Before running
- Installed copy: `C:\Users\steph\bin\loki-tune.exe` (source `tools/loki-tune`, `cargo build --release`). Models are shared
  with loki-dream (`bin\models`, `LOKI_MODELS`).
- The GPU must be free for the whole run (minutes to an hour): check `nvidia-smi` and ask the user before starting; a
  resident loki-scribe or another loki- job holding memory makes training spill to system RAM and crawl.
- ALWAYS read `loki-tune --help` first (INPUTS / RUN FOLDER / CHECKPOINTS AND RESUMING / OUTPUT / EXAMPLES).

## 1. Pick the kind, trigger and class
The caption rules follow community practice: captions describe what should stay promptable, the trigger carries what the
LoRA teaches. loki-scribe first lists the traits all images share and keeps them out of every caption.
- person / character: `--kind character --trigger zkx --class woman` (identity left to "zkx"; clothing, pose, setting described)
- art / photo style: `--kind style --trigger inkwash` (content described, style never named; trains ~70 epochs, at least 2000 steps; prompts that ask for "photography", "depth of field", "natural skin" fight the style)
- product / object: `--kind object --trigger zkxmug --class mug`
- pose / action / effect: `--kind concept --trigger zkxpose`
- everything plainly: `--kind general`
Use a rare made-up trigger word. 15-50 varied images (poses, places, light) work best; mirror with `--flip` only for
symmetric subjects without text.

## 2. Plan, review captions, train
```
loki-tune --dry-run --kind character --trigger zkx --class man -o runs/zkx photos/   # captions + plan -> runs/zkx/run.json
# review runs/zkx/run.json: items[].caption / short, traits, samples; edit if needed
loki-tune --resume runs/zkx                                                           # train (Ctrl-C or a STOP file: checkpoint + exit)
```
Inputs: images, folders, `.txt` lists, `.jsonl` lists with `{"path", "caption"}` (captions given this way, or as sidecar
`photo.txt` files, are used as is). Defaults: rank 16, lr 1e-4, ~50 epochs (1000-3000 steps), 1024 px, checkpoints +
previews every 250 steps. ~1.7 s/step at 1 MP on a 4090 (bf16), `--resolution 512` ~3x faster.

## 3. Judge checkpoints
- `runs/zkx/samples/step-NNNNNN_0.png` reproduces a training caption, `_1.png` is a new scene (generalization);
  `step-000000` is the base model. Look at them (Read the PNGs) rather than at the loss: flow-matching loss is noisy and
  barely moves. MOSS LoRAs keep improving after an apparent plateau; overtraining shows as burnt colours, copied
  backgrounds/poses from the training set, or the subject appearing without the trigger.
- Try a checkpoint: `loki-dream --turbo --lora runs/zkx/checkpoints/step-000750/zkx-step000750.safetensors -p "zkx, a man, ..."`
  (works on the int8 model too; lower the strength with `FILE:0.8` if it overpowers prompts).

## 4. Continue, extend, branch
- `--resume runs/zkx` continues from the latest checkpoint; add `--steps 3000` to extend a finished run.
- `--resume runs/zkx/checkpoints/step-000500` branches into `runs/zkx-from500` (original untouched).
- On resume only training settings may change (`--steps --lr --warmup --cosine --save-every --sample* --caption-dropout`);
  images, captions, model, rank and resolution belong to the run. Resumes are exact (same losses as an uninterrupted run).

## Output
stdout: the final LoRA path (`runs/zkx/zkx.safetensors`), or the last checkpoint when stopped (exit 130). `--json`: one object
with `lora`, `checkpoints`, `samples`, `loss`. stderr: `step i/n loss ... eta` lines. The LoRA uses Kromoss's key layout
(`diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`), bf16, with the `loki` record in its metadata.

## From the Lowkey app / server
Context palette (shift+right-click a selection) -> LoRA -> "Train LoRA on N images…" opens a dialog (kind, trigger, class,
captions, length, detail, rank) that queues the server's `train-lora` task (runs in `<server data>/loras/<name>`; pausing or
cancelling the job saves a checkpoint; the `resume` option continues a run).
