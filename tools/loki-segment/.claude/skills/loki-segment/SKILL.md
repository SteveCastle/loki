---
name: loki-segment
description: Segment images and roto video with the local loki-segment CLI (Meta SAM 3.1, Rust/CUDA, no Python). Use when the user wants a mask or cut-out of "all the people", "the sky", "the dog" (concept/text prompts), of the object under a click or inside a box, every object in an image, or per-frame masks (roto/mattes) of objects tracked through a video clip, including from an editor via the --serve JSON-lines session. Composes with loki-retouch / loki-reshoot through stdin/stdout.
---

# loki-segment: image or clip in, soft masks out

`loki-segment` runs Meta's Segment Anything 3.1 locally on the NVIDIA GPU (RTX 4090 class; ~2 GB VRAM for images, ~5 GB for
video). It finds every instance of a named concept, the object under a click or in a box, or everything. On video it tracks the
objects through the clip with SAM 3.1's memory and adds stability knobs on top. It's one of the `loki-` toolkit's small, composable
tools; see `loki-retouch` (image editing) and `loki-reshoot` (video generation).

## 0. Locate and sanity-check
- Source: the loki monorepo, `tools/loki-segment`. Binary: `target/release/loki-segment.exe` there (`cargo build --release`), or a
  copy on PATH (`C:\Users\steph\bin`, with `models\sam3.1_multiplex_fp16.safetensors` next to it). If the model is missing it is
  downloaded once (1.75 GB, from Comfy-Org/sam3.1; a notice prints even with `-q`). `LOKI_MODELS` points at another model folder.
- Video needs ffmpeg/ffprobe on PATH or next to the exe.
- ALWAYS read `loki-segment --help` first (PROMPTS / OUTPUT / SERVE / EXAMPLES). The binary is the source of truth.

## Conventions shared by all loki- tools
- Input: one image or video file, or `-` for an image on stdin.
- stdout: the written paths, one per line; with `--json`, one JSON object instead; `-o -` streams the PNG (one object only).
  Progress goes to stderr (`-q` silences it). Never parse stderr.
- Exit 0 ok (also when nothing matched: `"objects": []`), 1 error, 2 usage.
- Every mask PNG carries the `loki` generation record (iTXt): prompts, threshold, object id/label/score, command line.

## What to run (pick by intent)
| Intent | Command |
|---|---|
| every instance of a thing | `loki-segment -p person --json -o masks/ photo.jpg` -> masks/1.png, 2.png ... best first |
| several concepts at once | `loki-segment -p "the sky" -p tree -o masks/ photo.jpg` (ids run across both) |
| the object under a click | `loki-segment --point 410,220 photo.jpg` -> photo_mask.png (add `--neg X,Y` to exclude a part) |
| the object in a box | `loki-segment --box 100,80,300,420 photo.jpg` (several `--box` = several objects) |
| a cut-out (RGBA) | `loki-segment -p dog --alpha -o dog.png photo.jpg` |
| hard mask | add `--binary` (default masks are soft: 0..255 with feathered edges) |
| everything | `loki-segment --everything -o all/ photo.jpg` (images only, ~0.7 s) |
| roto a clip | `loki-segment -p "the girl" -o roto/ clip.mp4` -> roto/<id>/00000.png ... (every frame) |
| roto from a click on frame N | `loki-segment --point 300,250 --frame 40 -o roto/ clip.mp4` |
| several clicked objects | `--prompts objs.json` with `[{"obj":1,"points":[[x,y,1]]},{"obj":2,"box":[x,y,w,h]}]` |
| looping clip | add `--loop` (first and last frames agree); `--smooth 3` for extra temporal calm |

Coordinates are input pixels (x right, y down). Concepts take a short noun phrase ("red car", "person in a hat"), not an
instruction. `--threshold 0.3` finds more (and weaker) instances, and 0.7 fewer.

## Video stability knobs
- Bidirectional tracking is on by default: frames before the prompted one are tracked backwards. `--no-bidir` turns it off.
- `--loop`: for seamless loops; both directions go all the way around and the two estimates are merged.
- `--smooth N`: temporal smoothing of mask logits over N frames (3 is mild). The prompted frame is kept as is.
- `--sticky F` (default 0.15): stops near-tie candidate flips, such as a limb blinking in and out. `0` gives stock SAM behaviour.
- All objects of one run must be prompted on the same frame. Speed is ~100 ms/frame (one object or one concept) on a 4090.

## Editors: `--serve`
`loki-segment --serve` loads the models once, then takes one JSON request per line and gives one JSON reply per line:
`{"cmd":"open","path":...}` (an image is encoded once, ~100 ms), then `{"cmd":"segment","point":[x,y],"out":"m.png"}` (~10 ms) or
`{"cmd":"segment","text":"dog","out":"dir/"}` (~20 ms). On an opened video, `segment` previews one frame (`"frame":N`) and
`{"cmd":"track",...}` propagates and writes `dir/<id>/<frame>.png`. Errors are `{"error":"..."}` and the session continues.

## Check the result
- Overlay a mask on the image (PIL: blend a colour where mask > 127), or open the RGBA cut-out.
- No objects -> try a simpler noun, a lower `--threshold`, or a click instead.
- Wrong part with one click: add a `--neg` click on the unwanted part, or use a `--box`. One click is ambiguous (the decoder picks
  among whole / part / sub-part).
- "out of memory": another GPU job is running; check `nvidia-smi --query-gpu=memory.used --format=csv` and ask before stopping it.
