# loki-tune

**LoRA training for MOSS-Image models from the command line**: Kromoss (bf16 or int8) or any other MOSS-Image
checkpoint, from a folder of images to a checkpointed LoRA that loki-dream and ComfyUI load. It captions the images
(loki-scribe), encodes them (Qwen3.5-9B, FLUX.2 VAE), trains, writes checkpoints with preview images and resumes from
any of them. Standalone Rust/CUDA on loki-dream's engine: no Python, PyTorch, kohya or ComfyUI.

```
loki-tune --kind character --trigger zkx --class man photos/            # -> ./zkx/zkx.safetensors
loki-tune --kind style --trigger inkwash -o runs/inkwash art/
loki-tune --model my_moss_finetune.safetensors set.jsonl                 # train against another MOSS-Image checkpoint
loki-tune --dry-run --kind character --trigger zkx photos/               # caption + plan only; edit run.json, then:
loki-tune --resume zkx                                                   # start / continue (Ctrl-C saves and stops)
loki-tune --resume zkx/checkpoints/step-000500 --steps 1500              # branch from an earlier checkpoint
loki-dream --turbo --lora zkx/zkx.safetensors -p "zkx, a man, sitting on a park bench"
```

## Part of the loki- toolkit

| tool | does | engine |
|---|---|---|
| [loki-dream](../loki-dream) | text -> image, img2img, inpainting, LoRAs | Kromoss (MOSS-Image), Krea 2, FLUX.2 klein, Anima, SDXL, MiniMax H3 Image |
| **loki-tune** (this tool) | images -> LoRA | MOSS-Image (Kromoss) |
| [loki-scribe](../loki-scribe) | images / text -> text (captions here) | Qwen3.5 / 3.8 |

Conventions (`../CONVENTIONS.md`): stdout carries only the result (the final LoRA's path; `--json` one object), progress
goes to stderr, errors are one `loki-tune: error: ...` line, exit 0 ok / 1 error / 2 usage / 130 stopped with Ctrl-C.
Models are found next to the binary, in `./models` or `$LOKI_MODELS`, and downloaded when missing.

## Inputs

Image files, folders (recursive), list files and `-` (a list on stdin). A `.txt` / `.lst` list holds one path per
line; a `.jsonl` list one `{"path": "...", "caption": "..."}` per line, which is how a caller with its own captions
(the media server's descriptions) hands them over. The binary knows nothing about libraries or queries: the media
server's `train-lora` task resolves a selection or a search to files and calls it with a list.

## Captions

First found wins: the list's caption, a sidecar text file (`photo.txt` or `photo.jpg.txt`), then (`--captions auto`,
the default) loki-scribe. `--captions given` uses only given captions, `--captions none` trains every image on the
trigger alone. Captions are kept in `run.json`; review them with `--dry-run` and edit them there before training.

loki-scribe captions follow the community practice for each kind of LoRA (`--kind`): a caption describes what should
stay promptable and leaves what the LoRA teaches to the trigger.

| `--kind` | the trigger carries | the captions describe |
|---|---|---|
| `character` | the person: face, hair, body, age, signature items | clothing, pose, expression, action, setting, lighting, camera, medium |
| `style` | the look: medium, technique, palette, rendering | the content in full; the style is never named |
| `object` | the object's own design: shape, materials, markings | placement, surroundings, hands, lighting, camera |
| `concept` | the pose / action / effect | everything else |
| `general` (default) | nothing in particular | everything |

For character, style and object LoRAs loki-scribe first looks at up to 8 images together and lists what they all share
(`traits` in `run.json`, or give `--traits`); every caption is told to leave those out. `--class man` makes captions
name the subject "zkx, a man," then "zkx". Each image gets a long paragraph and a short sentence (MOSS-Image was trained
on several caption lengths per image); every step trains on one of the two at random. `--trigger` is put first in
captions that do not already contain it.

## Training

What happens, and the defaults (all overridable):

- **Base model**: `--model` takes any MOSS-Image checkpoint (bf16, fp16 or int8 ConvRot; ComfyUI single files with or
  without the `model.diffusion_model.` prefix). Default: Kromoss bf16 when present, else int8. The LoRA is trained
  against the frozen base and loads on either Kromoss file: loki-dream applies LoRAs unmerged, so an int8 base
  does not change the LoRA's effect. Training on the bf16 file gives cleaner gradients (see Validation).
- **Targets**: q, k, v, o, ffn_up_gate and ffn_down of all 42 blocks (the set the Kromoss turbo LoRAs cover; the text
  refiner stays frozen, as OpenMOSS's guide says). `--rank 16`, `--alpha` = rank.
- **Optimizer**: AdamW (lr `1e-4`; weight decay 0.01), 50 warmup steps, constant learning rate
  (`--cosine` decays to 10%), gradient clipping at 1. `--batch N` accumulates N images per step.
- **Steps**: about 50 epochs, 1000 to 3000 (`--steps`); styles about 70 epochs, at least 2000 (a style has to carry over
  to subjects it never saw: watch the new-scene preview). MOSS-Image LoRAs converge slowly: likeness keeps improving
  after it seems to plateau, so compare checkpoints rather than stopping early.
- **Noise levels** (`--timesteps moss`): MOSS-Image's final pretraining schedule, sigma = 1 - sigmoid(N(-1.09, 0.95))
  with 1% uniform, plus `--clean-share 0.3` of the steps drawn from sigma < 0.5. The base schedule spends only ~12% of
  steps on the clean half, where faces and fine detail are learned; raising it is the fix that made MOSS LoRAs hold
  likeness in community trainers. `uniform` and `logit-normal` are the alternatives. Loss: flow matching on
  v = noise - x0, mean over image tokens, no weighting (as the MOSS-Image report).
- **Sizes**: about `--resolution 1024` squared pixels per image, aspect ratio kept, sides multiples of 16, centre crop;
  smaller images are not upscaled past their own pixel count. `--flip` also trains mirrored images.
- **Captions dropout** `--caption-dropout` (default 0).

Each step is deterministic in the seed and the step number (image order, noise level, noise), so a resumed run
continues exactly where its checkpoint left off.

## The run folder

```
NAME/run.json                         settings, images with captions and sizes, shared traits, preview prompts
NAME/checkpoints/step-000250/         NAME-step000250.safetensors (the LoRA) + state.safetensors (optimizer)
NAME/samples/step-000250_0.png        previews (step-000000: the base model without the LoRA)
NAME/log.jsonl                        per step: loss, smoothed loss, learning rate, gradient norm, seconds
NAME/NAME.safetensors                 the final LoRA
NAME/cache/                           text features + latents, deleted when the run completes (--keep-cache)
```

Checkpoints are written every `--save-every 250` steps and when the run is stopped: Ctrl-C once finishes the step,
saves and exits (130); twice exits at once. `--resume NAME` continues from the latest checkpoint, `--resume
NAME/checkpoints/step-000500` from that one; an earlier checkpoint branches into `NAME-from500` (or `-o`), leaving the
original run as it was. `--steps`, `--lr`, `--warmup`, `--cosine`, `--save-every`, the preview settings, caption dropout,
weight decay and clipping may change on resume; the images, model, rank, alpha, resolution, seed and captions belong
to the run.

Previews: at every checkpoint (`--sample-every`), two images by default: the first image's short caption (does it
reproduce?) and a new scene with the trigger (does it generalize?). `--sample PROMPT` (repeatable) sets your own.
Rendered with the 8-step DMD2 turbo LoRA at CFG 1 (`--sample-steps 8`; more steps render at CFG 3 without it), seed
42, at `--sample-size` (default the training resolution). `--no-samples` skips them.

The LoRA file: Kromoss's key layout (`diffusion_model.layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}.lora_down.weight` /
`.lora_up.weight` / `.alpha`, bf16), the loki generation record under the `loki` metadata key (base model, trigger,
rank, steps, images, loss, `--meta`) and kohya's `ss_*` keys.

## From the Lowkey app

Context palette (shift+right-click) → LoRA: **Train LoRA on N images…** (the selection, or every image of the library
query when the scope is widened) opens the dialog's *New LoRA* tab; **Continue a LoRA run…** opens *Continue a run*:
pick a run and a checkpoint (with its two previews), the new total steps and learning rate. The server's `train-lora`
task runs loki-tune (`resume` option); an earlier checkpoint than the run's last branches into `NAME-fromSTEP`.
Pausing or cancelling the job saves a checkpoint; resuming a paused job continues from where it stopped. Finished
LoRAs (and checkpoints) appear in Transform's Generate / Reimagine LoRA picker (`GET /api/loras`).

## Speed and memory (RTX 4090)

| base | 1 MP step | VRAM |
|---|---|---|
| Kromoss bf16 (default) | ~1.7 s | ~16 GB |
| Kromoss int8 | ~1.4 s | less (5 GB smaller weights) |

At 512 px a step is roughly 3x faster. Previews add ~5 s per image per checkpoint (8-step turbo).

Preprocessing loads one model at a time (loki-scribe, the 19 GB Qwen3.5 text encoder streamed layer by layer, the VAE),
then the DiT trains alone.

## How it works

`src/train.rs` runs loki-dream's MOSS-Image forward (the same kernels and bf16 rounding as inference, the LoRA as two
extra GEMMs per linear) keeping only the residual stream at each block input, then walks the blocks backwards,
re-running each forward to keep its intermediates and differentiating it (block-level gradient checkpointing: ~1 GB of
saved activations at 1 MP instead of ~20 GB). Backward kernels (`kernels/train.cu`, `kernels/attn_train.cu`): modulated
RMSNorm, q/k RMSNorm + rope, SiTU, a FlashAttention-2 backward split into a dK/dV and a dQ kernel (deterministic, no
atomics), the Hadamard-256 that undoes ConvRot's rotation for int8 linears (the activation quantizer is a
straight-through estimator), and a tensor-core GEMM for the LoRA weight gradients. Gradients accumulate in f32 (the
residual stream too), GEMM inputs are bf16. LoRA weights, gradients and Adam moments live in one f32 buffer; q, k and
v are three LoRAs trained on the fused qkv projection through a block-diagonal up matrix.

## Validation

- `tune_check ops`: every backward kernel against an f64 CPU reference or central differences (attention dq/dk/dv
  ~2e-3 relative error, the norms ~1e-7, SiTU / rope ~2e-3, the LoRA gradient GEMM ~3e-7).
- `tune_check int8`: int8 ConvRot linears against the same layers in bf16 (forward and input gradient ~1%).
- `tune_check step --model kromoss_bf16 --dump DIR` + `ref/ref_tune.py DIR f32`: a full training step against torch
  autograd through the official `MOSSImageTransformer` (pipeline_moss_image.py) in f32. The engine's LoRA gradients
  match with cosine 0.93-0.99 per module type (0.99 at the last block, 0.92 at the first), closer to the f32
  reference than torch's own bf16 autograd (0.68 at the first block): the deep backward through 42 bf16 blocks is
  noisy, and keeping the residual gradient in f32 helps. The int8 base adds W8A8 noise at every linear and its
  first-block gradients agree with the f32 reference much less, hence the bf16 default.
