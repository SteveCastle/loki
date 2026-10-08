---
name: loki-reshoot
description: Generate video with native sound using the local loki-reshoot CLI (MiniMax H3 reference-to-video, Rust/CUDA, no ComfyUI). Use when the user wants to animate a photo ("living photo"), make a clip from reference images, videos or audio (music, voice), asks for an H3/MiniMax generation, or wants to re-run or iterate a video generation with different refs, prompt, size, length or seed. Composes with loki-retouch (image editing) through stdin/stdout.
---

# loki-reshoot: references in, video with sound out

`loki-reshoot` turns any mix of reference images (≤9), videos (≤3) and audio clips (≤3) plus a prompt into an mp4 with generated
native audio. It runs the whole model locally on the NVIDIA GPU (RTX 4090 class, 24 GB). It is one of the `loki-` toolkit's small,
composable tools: see the sibling `loki-retouch` (image editing/upscaling) and the shared conventions below.

## 0. Locate and sanity-check
- Binary: `C:\Users\steph\dev\loki-reshoot\target\release\loki-reshoot.exe` (`cargo build --release` in that repo if missing).
  Run it **from the repo root** so `./models` (the four model files) is found, or set `LOKI_MODELS` to the folder holding them.
  Otherwise it downloads ~42 GB from Hugging Face (it prints a notice even with `-q`). Never run it from a random cwd without `LOKI_MODELS`.
- The GPU must be free (~22 GB). If ComfyUI or another big GPU job is running the run fails with an allocation error. Check
  `nvidia-smi --query-gpu=memory.used --format=csv`; ask before stopping the user's processes.
- ffmpeg/ffprobe are required (PATH, or next to the exe; auto-downloaded on Windows if absent).
- ALWAYS read the live docs first: `loki-reshoot --help` (options, limits, a condensed PROMPT WRITING section, composing) and, before
  writing a prompt for the first time in a task, `loki-reshoot --prompt-guide` (MiniMax's full reference-mode guide). The binary is the
  source of truth if this skill and it disagree.

## Conventions shared by all loki- tools (this is what makes them composable)
- Inputs: files, or `-` = that one input read from stdin (`-i -`, `--animate -`, `-v -`, `-a -`, `-P -`; at most one per run).
- Outputs: the path of the written file is printed on **stdout** (`--json`: one JSON object instead); `-o -` writes the media itself to
  stdout (mp4 is fragmented). Everything else (progress, timings, warnings) goes to **stderr**; `-q` silences it. Never parse stderr.
- Exit status: 0 ok, 1 error, 2 usage error. Errors are one `loki-reshoot: error: ...` line on stderr.
- Shared env: `LOKI_MODELS` (model folder), `LOKI_PROFILE=1` (per-kernel timing).
- Chaining example: `loki-retouch --preset restore -o - old.jpg | loki-reshoot --animate - -d 5 -o shot.mp4`

## Quick mode: bring a still photo to life (the most common job)
```
loki-reshoot --animate photo.jpg --describe "the young woman in a black swimsuit taking a mirror selfie in a sunlit room" -d 5 -o alive.mp4
```
`--animate IMAGE` = natural ambient life + subtle resting movement + subtle camera shake, identity/framing kept, ambient sound, no speech
or music. It adds the image as `<Picture 1>`, uses `--native` (below) and 5 s by default. Always pass `--describe` (one sentence on subject and
setting: it anchors identity and what may move). `--shake none|subtle|handheld`; `--prompt "..."` appends extra direction;
`--ref-image-size max` for best identity. `--show-prompt` prints the composed prompt and exits.
Audio caveat: with no audio reference the model renders ambient-only prompts almost silent (about -60 dB, ComfyUI does the same). For audible
sound pass `-a` (a room-tone/nature recording or music slice).

## Native canvas / supported ratios (`--native`, implied by `--animate`)
The model is built around five canvases: 1:1 768x768, 4:3 1024x768, 3:4 768x1024, 16:9 1344x768, 9:16 768x1344. `--native` snaps the first
reference's aspect to the nearest one and fits the first image to it with `--fit auto` (default: centre-crop when within ~12% of the supported
ratio, else pad), `crop`, `pad` (black bars; the model keeps them in the video rather than filling them) or `stretch`. `--size WxH` overrides.

## 1. Gather inputs
- Reference images: `-i FILE` (repeat). Tags `<Picture 1>`, `<Picture 2>`... in the order given.
- Reference videos: `-v FILE[@START[,DURATION]]` (seconds). Tags `<Video k>`. Resampled to 24 fps; ≥5 frames; at most the output length. Its
  soundtrack is used too (becomes an `<Audio j>`) unless `--no-video-audio`.
- Reference audio (music to dance to, a voice to clone, sound effects): `-a FILE[@START[,DURATION]]` (≤30 s). Tags `<Audio j>`: video
  soundtracks first (in video order), then `-a` clips in order.
- The CLI prints the tag map at start-up (stderr); the prompt must use exactly those tags.

## 2. Choose geometry and length (limits are enforced)
- `-d SECONDS` (default 5) or `--frames N`; frames snap UP to 17k+5 (22, 39, ..., 124 ≈ 5.2 s, ..., 362 ≈ 15 s). Trained range is 124-362 frames
  (5.2-15 s); shorter runs but quality may suffer; >362 is refused. Use ≥5 s for real results.
- `--size WxH` (multiples of 32; max 2048x1152) or `--aspect W:H --megapixels F` (default aspect = first reference, 0.4 MP).
  Template sizes: 864x480, 1216x672, 1344x768 (0.98 MP, node default), 1920x1088 max in ComfyUI's table.
- Cost ∝ tokens = latent_t × (H/32) × (W/32): doubling both sides is 4× tokens and ~16× attention; length is much cheaper than resolution.
  Measured on a 4090: ~11 s per sampling step at 768x1344 / 5 s (20 steps ≈ 4 min, ~4.5 min total); a 15 s 1344x768 clip is ~1 min per step.
  Draft small and short first (`--size 384x672 --frames 22 --steps 12`, under a minute), then render the final at full settings.
- Faces are small and soft when the subject is far away or the output is small: use a larger size, a closer-framed reference,
  `--ref-image-size max` and ≥5 s clips.
- Fixed `--seed N` for reproducibility (otherwise random, printed); changing the seed is the cheapest way to get another take.

## 3. Write the prompt (this decides quality)
The model is very sensitive to prompt wording. Follow `--prompt-guide`:
- English, detailed, shot-by-shot, 350-500 words for generation; never a bare plot summary.
- Define each reference's job with the tags: `<Subject N>` for reusable visible content (person, place, outfit, motion, style) with its source
  (`<Subject 1> is the woman in <Picture 1>, long dark hair, blue cardigan`), `<Picture N>` for concrete frame anchors, `<Video N>` for
  whole-video relations (edit/continue/camera-and-rhythm source), `<Audio N>` for copied/referenced audio.
- Structure: `subject_definitions`, `summary` (starts with `[reference generation]`, `[reference generation + audio reference]`, ...),
  `retention_analysis`, `detailed_description` (style opening, then `[Shot 1] ...`, `[Shot 2] At 00:03.000, ...` with framing, appearance,
  environment/lighting, actions, camera movement, current sound, where each reference applies), `overall_soundscape`, `non_diegetic_music`
  (`N/A` when none). Speakers `(S1)`, `(S2)`; dialogue `<d>[English] ...</d>`.
- Say explicitly which reference drives what: "<Audio 1> is the music; she dances on its beat".
- Put long prompts in a file: `-P prompt.txt` (quoting `<`/`>` in shells is painful). Short prompts are fine for drafts.

## 4. Run
```
cd C:\Users\steph\dev\loki-reshoot
.\target\release\loki-reshoot.exe -i hero.png -a theme.mp3@30,6 -d 6 --size 1216x672 --seed 7 -P prompt.txt -o out\hero.mp4
.\target\release\loki-reshoot.exe -i face.png -v dance.mp4@2,4 -d 5 -P p.txt -o out\dance.mp4
```
Run long jobs in the background and poll; progress prints per step on stderr. Default output is `reshoot_<seed>.mp4` in the cwd. `--no-audio`
skips audio. Phases load/evict models, so expect ~10 s of model loading per phase.

## 5. Check the result and iterate
- `ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of compact out.mp4` (frame count is `N/A` for `-o -` streams).
- Look at it: contact sheet `ffmpeg -y -i out.mp4 -vf "select='not(mod(n,12))',scale=480:-1,tile=4x2" -frames:v 1 sheet.png`, then read the PNG;
  check audio with `ffmpeg -i out.mp4 -af volumedetect -vn -f null -`.
- Fix problems by prompt first (be more explicit about what each tag does), then seed, then size/ref-image-size. If identity drifts, describe the
  subject's features in `subject_definitions` and use `--ref-image-size max`. If the length is wrong, remember the 17k+5 snapping.
- Errors: "allocating ... OUT_OF_MEMORY" → free the GPU or reduce size/duration/ref count; "frames exceeds ~15 s limit" → shorten.
