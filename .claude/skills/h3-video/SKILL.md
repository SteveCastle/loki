---
name: h3-video
description: Generate videos with sound using the local MiniMax H3 reference-to-video CLI (h3ref2va, Rust/CUDA, no ComfyUI). Use when the user wants to make/animate a video from reference images, reference videos or audio (music, voice), asks for an H3/MiniMax generation, or wants to re-run/iterate a generation with different refs, prompt, size, length or seed.
---

# h3-video: MiniMax H3 reference-to-video with `h3ref2va`

`h3ref2va` turns any mix of reference images (≤9), videos (≤3) and audio clips (≤3) plus a prompt into an mp4 with
generated native audio. It runs the whole model locally on the NVIDIA GPU (RTX 4090 class, 24 GB).

## 0. Locate and sanity-check
- Binary: `C:\Users\steph\dev\h3ref2va\target\release\h3ref2va.exe` (build with `cargo build --release` in that repo if missing).
  Run it **from the repo root** so `./models` (the four model files) is found; otherwise models are looked up next to the exe,
  in `$H3_MODELS`, or downloaded (~42 GB).
- The GPU must be free: it needs ~22 GB. If ComfyUI (or another big GPU job) is running, VRAM is exhausted and the run fails with an
  allocation error. Check `nvidia-smi --query-gpu=memory.used --format=csv`; ask before stopping the user's processes.
- ffmpeg/ffprobe are needed (found on PATH / next to the exe; downloaded automatically on Windows if absent).
- ALWAYS read the live docs first: `h3ref2va --help` (options, limits, a condensed PROMPT WRITING section) and, before writing a
  prompt for the first time in a task, `h3ref2va --prompt-guide` (MiniMax's full reference-mode prompt guide). They are the source
  of truth if this skill and the binary disagree.

## Quick mode: bring a still photo to life (most common job)
```
h3ref2va --animate photo.jpg --describe "the young woman in a black swimsuit taking a mirror selfie in a sunlit room" -d 5 -o outlive.mp4
```
`--animate IMAGE` = natural ambient life + subtle resting movement + subtle camera shake, identity/framing kept, ambient sound, no
speech or music. It adds the image as `<Picture 1>`, uses `--native` (below) and 5 s by default. Always pass `--describe` (one sentence on
subject and setting: it anchors identity and what may move). `--shake none|subtle|handheld`; `--prompt "..."` appends extra direction;
`--ref-image-size max` for best identity. Also works with other flags (`-a music.mp3` to add audio refs, `--seed`, `--steps`).

Audio caveat: with no audio reference the model renders ambient-only prompts almost silent (about -60 dB; ComfyUI does the same).
If you want audible ambience, pass a reference audio with `-a` (a room-tone/nature recording or music slice). `--show-prompt` prints the
composed prompt and exits.

## Native canvas / supported ratios (`--native`, implied by `--animate`)
The model is built around five canvases: 1:1 768x768, 4:3 1024x768, 3:4 768x1024, 16:9 1344x768, 9:16 768x1344. `--native` snaps the first
reference's aspect to the nearest one and fits the first image to it with `--fit auto` (default: centre-crop when within ~12% of the
supported ratio, else pad), `crop`, `pad` (black bars; the model keeps them in the video rather than filling them) or `stretch`. Use `--size WxH` to override.

## 1. Gather inputs
- Reference images: `-i FILE` (repeat). Tags `<Picture 1>`, `<Picture 2>`... in the order given.
- Reference videos: `-v FILE[@START[,DURATION]]` (seconds). Tags `<Video k>`. Resampled to 24 fps; need ≥5 frames; at most the output
  length. Its soundtrack is used too (becomes an `<Audio j>`) unless `--no-video-audio`.
- Reference audio (music to dance to, a voice to clone, sound effects): `-a FILE[@START[,DURATION]]`. Use `@START,DURATION` to pick a
  slice (≤30 s). Tags `<Audio j>`: video soundtracks first (in video order), then `-a` clips in order.
- The CLI prints the tag map at start-up; the prompt must use exactly those tags.

## 2. Choose geometry and length (limits are enforced)
- `-d SECONDS` (default 5) or `--frames N`; frames snap UP to 17k+5 (22, 39, ..., 124 ≈ 5.2 s, ..., 362 ≈ 15 s). Trained range is
  124-362 frames (5.2-15 s); shorter runs but quality may suffer; >362 is refused. Use ≥5 s for real results.
- `--size WxH` (multiples of 32; max 2048x1152) or `--aspect W:H --megapixels F` (default aspect = first reference, 0.4 MP).
  Template sizes: 864x480 (0.4 MP), 1216x672, 1344x768 (0.98 MP, node default), 1920x1088 max in ComfyUI's table.
- Cost ∝ tokens = latent_t × (H/32) × (W/32): doubling both sides is 4× tokens and ~16× attention; length is much cheaper than
  resolution. Reference tokens ride along every step. Indicative speed on a 4090: ~8 s per sampling step at 1344x768 / 4.5 s of video
  (20 steps ≈ 3 min) plus ~40 s load/encode/decode overhead; a 15 s 1344x768 clip is ~1 min per step. Do quick drafts small and short
  (e.g. 640x352, 1-2 s via `--frames 22 --steps 12`), then the final at full settings.
- Faces are small and soft when the subject is far away or the output is small: for face fidelity use a larger size, a closer-framed
  reference, `--ref-image-size max` (better identity, several times slower) and ≥5 s clips.
- Fixed `--seed N` for reproducibility; otherwise random (printed). Changing the seed is the cheapest way to get a different take.

## 3. Write the prompt (this is what decides quality)
The model is very sensitive to prompt wording. Follow `--prompt-guide`:
- English, detailed, shot-by-shot, 350-500 words for generation; not a plot summary.
- Define each reference's job with the tags: `<Subject N>` for reusable visible content (person, place, outfit, motion, style) with its
  source (`<Subject 1> is the woman in <Picture 1>, long dark hair, blue cardigan`), `<Picture N>` for concrete frame anchors,
  `<Video N>` for whole-video relations (edit/continue/camera-and-rhythm source), `<Audio N>` for copied/referenced audio.
- Structure: `subject_definitions`, `summary` (starts with `[reference generation]`, `[reference generation + audio reference]`, ...),
  `retention_analysis`, `detailed_description` (style opening, then `[Shot 1] ...`, `[Shot 2] At 00:03.000, ...` with framing,
  appearance, environment/lighting, actions, camera movement, current sound, where each reference applies), `overall_soundscape`,
  `non_diegetic_music` (`N/A` when none). Speakers `(S1)`, `(S2)`; dialogue `<d>[English] ...</d>`.
- Say explicitly which reference drives what: "<Audio 1> is the music; she dances on its beat", "keep the voice timbre of <Audio 2>".
- Put the prompt in a file and use `--prompt-file prompt.txt` (long text and `<`/`>` characters are painful to quote in shells).
  A short prompt is fine for quick drafts: e.g. "Animate <Picture 1> as one continuous shot ... Audio: rhythmic music from <Audio 1>."

## 4. Run
```
cd C:\Users\steph\dev\h3ref2va
.\target\release\h3ref2va.exe -i hero.png -a theme.mp3@30,6 -d 6 --size 1216x672 --seed 7 --prompt-file prompt.txt -o out\hero.mp4
.\target\release\h3ref2va.exe -i face.png -v dance.mp4@2,4 -d 5 --prompt-file p.txt -o out\dance.mp4
```
Run long jobs in the background and poll; progress prints per step. Output defaults to `h3_<seed>.mp4` in the cwd. Add `--no-audio`
to skip audio generation. Phases load/evict models, so expect ~10 s per phase of model loading.

## 5. Check the result and iterate
- `ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames,duration -of compact out.mp4`
- Look at it: extract a contact sheet, e.g. `ffmpeg -y -i out.mp4 -vf "select='not(mod(n,12))',scale=480:-1,tile=4x2" -frames:v 1 sheet.png`, and read the PNG;
  check audio with `ffmpeg -i out.mp4 -af volumedetect -vn -f null -`.
- Fix problems by prompt first (be more explicit about what each tag does), then seed, then size/ref-image-size. If identity drifts,
  describe the subject's features in `subject_definitions` and use `--ref-image-size max`. If the length is wrong, remember the 17k+5 snapping.
- Errors: "allocating ... OUT_OF_MEMORY" → free the GPU or reduce size/duration/ref count; "frames exceeds ~15 s limit" → shorten.
