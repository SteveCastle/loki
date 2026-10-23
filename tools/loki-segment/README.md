# loki-segment

**Segment Anything 3.1 from the command line, for stills and video.** Name a concept ("person", "the sky", "red car")
and get every instance, or click / box one object, and get soft-edged masks at the input resolution. On a clip the
objects are tracked with SAM 3.1's memory so the roto holds from frame to frame. It's one self-contained binary: the whole
SAM 3.1 stack (ViT-L vision trunk, text encoder, DETR-style concept detector, SAM prompt decoder, multiplexed video
tracker) is reimplemented in CUDA and Rust. It uses no Python, ONNX or PyTorch at runtime; the only dependencies are the NVIDIA
driver and ffmpeg (video only).

```
loki-segment -p person photo.jpg                        # every person -> photo_mask_1.png, photo_mask_2.png ...
loki-segment --point 410,220 --neg 380,400 photo.jpg    # the object under the click -> photo_mask.png
loki-segment --box 100,80,300,420 --alpha -o cut.png photo.jpg
loki-segment -p dog -o roto/ clip.mp4                   # roto/<id>/00000.png ... per object, tracked
loki-segment --serve                                    # JSON-lines session for editors (models load once)
```

## Part of the loki- toolkit

| tool | does | engine |
|---|---|---|
| [loki-retouch](../loki-retouch) | image editing, upscaling, restoration, compositing | Qwen Image 2.1 |
| [loki-reshoot](../loki-reshoot) | reference images / videos / audio -> video with sound | MiniMax H3 |
| **loki-segment** (this tool) | concept / click / box segmentation of images, tracked roto of video | SAM 3.1 |

The shared conventions are in `../CONVENTIONS.md`. In short: results go to files whose paths are printed on stdout (`--json`
prints one JSON object instead, `-o -` streams the PNG), progress goes to stderr (`-q` silences it), and the exit status is 0 ok,
1 error, 2 usage. Models are found next to the binary, in `./models` or `$LOKI_MODELS`, and downloaded when missing. Every mask PNG
carries the `loki` generation record (iTXt chunk).

```
loki-retouch -p "make it night" -o - photo.jpg | loki-segment -p person -o - - > people.png
```

## Usage

```
loki-segment [OPTIONS] INPUT            # INPUT = image or video file (or - for an image on stdin)
  -p, --prompt TEXT        concept prompt, repeatable ("person", "sky") -> every matching instance
  -P, --prompt-file FILE   concept prompts, one per line (- = stdin)
  --point X,Y              positive click (input pixels), repeatable;  --neg X,Y negative click
  --box X,Y,W,H            box prompt (input pixels), repeatable
  --frame N                video: frame the prompts are on (default 0)
  --prompts FILE.json      full spec: [{ "obj": 1, "frame": 0, "text": "...", "points": [[x,y,1|0],...], "box": [x,y,w,h] }, ...]
  --everything             every object (automatic mask generation, images)
  --threshold F            concept detector score threshold (default 0.5)
  --loop, --no-bidir, --smooth N, --sticky F   video stability knobs (below)
  -o, --out PATH|DIR|-     image: mask PNG(s); video: DIR/<obj>/<frame:05>.png per object
  --alpha / --binary       RGBA cut-out instead of a gray mask / hard 0-255 masks
  --json                   one JSON object on stdout (below)
  --serve                  JSON-lines session on stdin/stdout
  --model PATH  --meta JSON  --meta-file FILE  -q
```

**Prompts.** A concept (`-p`, an entry with only `"text"`) goes to the SAM 3 detector and yields **every** instance whose joint
score `sigmoid(class) * sigmoid(presence)` exceeds `--threshold`, best first. Clicks and a box (`--point`, `--neg`, `--box`, an entry
with `"points"`/`"box"`) go to the SAM prompt decoder and pick **one** object. With one positive click the decoder's ambiguity mode is
used (3 candidates, best predicted IoU). Several `--box` without clicks give one object per box. In `--prompts`, an entry with both
clicks and `"text"` is the clicked object, and the text is only its label.

**Outputs.** Masks are 8-bit grayscale PNGs at the input resolution with **soft edges**: the sigmoid of the 288x288 mask logits,
bilinearly upsampled the way the official pipeline does it. Threshold at 128 for a hard mask, or pass `--binary`. Naming:

- image, one object: `-o FILE.png`, or `-o DIR/` -> `DIR/<id>.png`. The default is `<input>_mask.png` next to the input.
- image, several objects: `-o DIR/` -> `DIR/<id>.png`, `-o FILE.png` -> `FILE_<id>.png`. The default is `<input>_mask_<id>.png`.
- `-o -` writes the PNG to stdout (exactly one object; a terminal is refused).
- video: `DIR/<id>/<frame:05>.png` for every frame and object. The default DIR is `<input>_masks/`.

`--json` prints:

```json
{"width":1800,"height":1200,"frames":1,"fps":null,"model":"SAM 3.1",
 "objects":[{"id":1,"label":"truck","score":0.8789,"bbox":[84,282,1625,566],"mask":"out/1.png"}]}
```

`bbox` is the tight box of the mask (>= 128) in input pixels. Video objects carry `"frames": "DIR/<id>"` instead of `"mask"`, their
`bbox` is on the prompted frame, and the object adds `"prompt_frame"`. An empty `objects` list (exit 0) means nothing matched. For
clicks, `score` is the decoder's object-presence probability; for `--everything` it is `sigmoid` of the predicted IoU.

**Contract notes** (relative to the brief this tool was built against): `-P/--prompt-file`, `--alpha`, `--binary`, `--sticky` and
`--model` are additions. On a video, all prompts of one run must be on the same frame for now (`--frame`, or the same `"frame"` in
every entry). `--everything` is image-only.

## `--serve` (interactive editors)

`loki-segment --serve [INPUT]` loads the models once (about 1.4 s) and answers one JSON line per request line:

```
{"cmd":"open","path":"img.jpg"}                         -> {"ok":true,"width":..,"height":..,"frames":1,"fps":null,"ms":..}
{"cmd":"segment","text":"dog","out":"dir/"}             -> the --json object, plus "ms"
{"cmd":"segment","point":[x,y],"neg":[[x,y]],"box":[x,y,w,h],"out":"click.png"}
{"cmd":"segment","prompts":[{...}, ...],"threshold":0.4,"alpha":true,"binary":false,"everything":false}
{"cmd":"open","path":"clip.mp4"}                        -> decodes the frames (no model work yet)
{"cmd":"segment","frame":40,"point":[x,y],"out":"dir/"} -> video: that frame only (a fast preview, dir/<id>/00040.png)
{"cmd":"track","frame":40,"prompts":[...],"out":"dir/","loop":true,"smooth":3,"sticky":0.15,"no_bidir":false}
{"cmd":"close"}   {"cmd":"info"}
```

Errors come back as `{"error":"..."}` and the session continues; EOF ends it. Fields mean what the flags mean (`text`/`prompt` may be
a string or a list, and `point`/`points`/`neg`/`box` take one item or a list). `prompts` entries without a `"frame"` inherit the
request's. An opened image is encoded once (about 100 ms); after that every `segment` runs only the prompt decoders. On an RTX 4090
with a 1800x1200 image, PNG writing included:

| request | time |
|---|---|
| `open` (decode + resize + ViT-L + three necks) | 95-110 ms |
| `segment` click / clicks + box | 7-10 ms |
| `segment` concept (cached text) / new concept | 10-30 ms / +6 ms text encoding |
| video `segment` preview on a new frame / same frame again | ~100 ms / ~10 ms |

## Video and temporal stability

The tracker is SAM 3.1's own: memory encoder, decoupled RoPE memory attention over the prompted frame plus the 6 most recent frames,
and object pointers from the last 15 frames, all through the multiplex decoder. Objects are bucketed the way the official demo does
it. Every clicked object gets its own bucket, with independent memory. The instances of one concept are multiplexed up to 16 to a
bucket, which costs about one object's worth of compute. Each frame's backbone runs once per pass, whatever the number of buckets.

On top of the stock model, these knobs target flicker:

- **bidirectional** (default; `--no-bidir` turns it off): frames before the prompted frame are tracked by a backward pass. Without it
  they get empty masks.
- **`--loop`**: the clip is a seamless loop. Both passes run all the way around, so every frame gets a forward and a backward estimate.
  They are merged in logit space with weights `1 / (1 + distance to the prompt)`, so the last frame agrees with the first.
- **`--smooth N`**: a triangular temporal filter of width N over the mask logits before thresholding and output (it wraps around with
  `--loop`). The prompted frame is left as prompted. 3 is mild.
- **`--sticky F`** (default 0.15): when the decoder's candidates score within F of each other, keep the one that overlaps the object's
  previous mask the most. In the stock model a near-tie flips a limb in and out between frames. `--sticky 0` restores stock behaviour.

Measured on the bedroom clip (200 frames, 960x540, two kids jumping, crossing and occluding each other): identities hold through full
occlusion. Motion-compensated frame-to-frame mask IoU (`tools/flicker.py`) is 0.860 stock and 0.879 with `--smooth 3`. On a ping-pong
loop (frames m and N-1-m are the same image), `--loop` raises the worst mirror-pair IoU from 0.920 to 0.946.

Speed on an RTX 4090: about 100 ms per frame for one object or one concept bucket (55 ms of that is the ViT-L), and about 120 ms for
two separately clicked objects. Frames are decoded with ffmpeg at native resolution and every frame is kept, with no frame-rate
conversion. Peak VRAM is about 5 GB, most of it the feature cache that `--loop` uses for its second visit (150 frames by default,
`LOKI_SEGMENT_CACHE_FRAMES=N`).

## Models

The single model file is found next to `loki-segment.exe`, in `./models`, or in `$LOKI_MODELS` (a path list). When it's missing it is
downloaded (1.75 GB, resumable) from **[Comfy-Org/sam3.1](https://huggingface.co/Comfy-Org/sam3.1)**:
`checkpoints/sam3.1_multiplex_fp16.safetensors`. That file is Comfy-Org's ungated fp16 repackaging of Meta's official
[facebook/sam3.1](https://huggingface.co/facebook/sam3.1) `sam3.1_multiplex.pt`. The official repo is gated (manual approval). The
key names are the official ones (`detector.*`, `tracker.model.*`, 1590 fp16 tensors). Loaded into the official `sam3` code, every key
is used except the computed RoPE buffers and the unused `text_projection`. Matrices go to the GPU as bf16 and small vectors as f32.
The CLIP BPE vocabulary (`assets/bpe_simple_vocab_16e6.txt.gz`, from the official repo, originally OpenAI CLIP) is embedded in the
binary.

### License

SAM 3 / 3.1 weights and code are released by Meta under the **SAM License** (Last Updated: November 19, 2025; the copy shipped with
`facebook/sam3.1`, `Comfy-Org/sam3.1` and `github.com/facebookresearch/sam3` is identical). It grants a non-exclusive, worldwide,
royalty-free license to use, reproduce, distribute, create derivative works of and modify the SAM Materials. The conditions:

- anything redistributed must ship with a copy of the license;
- publications must acknowledge SAM;
- use must comply with law and trade controls, with no ITAR, military, nuclear or espionage end uses;
- use must not involve reverse engineering the SAM Materials;
- the materials come with no warranty, and the license terminates if you sue Meta over IP.

This engine is an independent reimplementation from the published code and paper, and it loads the released weights unmodified.
Anyone redistributing the model file must include Meta's license text (it ships next to it on Hugging Face).

## Building

You need Rust (stable, MSVC toolchain), CUDA Toolkit 12.x (`nvcc`) and the Visual Studio 2022 C++ tools.

```
cargo build --release                       # -> target/release/loki-segment.exe (+ stage_check, video_check)
cargo test --release --lib --test tokenizer # GPU-free tests (prompt parsing, output naming, tokenizer parity)
```

`build.rs` compiles `kernels/*.cu` into sm_89 fatbins (plus compute_80 PTX) that are embedded in the binary. `common.cuh`,
`gemm_bf16.cu` and the implicit-GEMM `conv.cu` are shared with loki-retouch. `attention.cu` is retouch's flash attention extended to
head dims 32/64/128, a batch grid (the ViT's 9 windows), key-padding masks and an additive bias (the detector's box bias).

## Validation

The reference is Meta's own `sam3` package (github.com/facebookresearch/sam3) running on the same fp16 file. It lives in a gitignored
venv:

```
python -m venv tools/ref-env
tools/ref-env/Scripts/python -I -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
tools/ref-env/Scripts/python -I -m pip install timm numpy ftfy regex iopath tqdm pillow safetensors psutil huggingface_hub opencv-python pycocotools "triton-windows<3.6"
git clone https://github.com/facebookresearch/sam3 ref_src/sam3 && tools/ref-env/Scripts/python -I -m pip install --no-deps ./ref_src/sam3
tools/ref-env/Scripts/python -I ref/ref_image.py testdata/truck.jpg ref_out/truck_fp32 --fp32 --text truck --text wheel --point 900,500
tools/ref-env/Scripts/python -I ref/ref_video.py testdata/frames ref_out/clip_fp32 --obj "1:394,211" --obj "2:230,269" --fp32
target/release/stage_check ref_out/truck_fp32 models/sam3.1_multiplex_fp16.safetensors testdata/truck.jpg
target/release/video_check ref_out/clip_fp32 testdata/frames          # PARITY=1: the official multi-object add path
```

The reference scripts patch a few things in upstream. The official predictors enable bf16 autocast for the whole process, so `--fp32`
switches it off. The ViT MLP is forced to bf16, so `--fp32` patches that too. On Windows the FLASH SDPA backend is unavailable, and
some upstream calls have drifted (`ref/ref_common.py`, `ref/ref_video.py`).

Results are measured against the fp32 reference on 3 images and a 24-frame clip. The noise floor in brackets is the official
bf16 path against its own fp32 result:

| stage | cosine vs fp32 reference |
|---|---|
| ViT-L trunk output (72x72x1024) | 0.9973-0.9990 (official bf16: 0.9970) |
| FPN necks, 3 heads x 3 levels | 0.9990-0.9999 (official bf16: 0.9990-0.9999) |
| text features | 0.99987 |
| fusion encoder memory | 0.9995-0.9998 |
| decoder queries (hs) on prompts that match | 0.997-0.9995 (official bf16: 0.983-0.995) |
| mask logits of all 200 queries | 0.9975-0.9997 (official bf16: 0.988-0.995) |
| kept detections and scores | identical sets, probabilities within 0.003 |
| SAM click decoder mask logits / IoU / object score | 0.99995 / 0.99999 / 20.17 vs 20.15 |
| tracker: memory, pointers, memory-conditioned features | 0.9995 / 0.99998 / 0.9994 |
| tracked masks, 24 frames, 2 objects | logits cos >= 0.9999 on 20 of 24 frames; mask IoU vs reference mean 0.9966 |

Frames that deviate do so because of near-tie candidate flips in the multimask head (e.g. predicted IoUs of 3.338 vs 3.352). The
reference's mask matches one of our other candidates at cosine 0.9999. These flips are numerical, the official bf16 path makes them too,
and they motivated `--sticky`. Concept prompts that match nothing (presence logit < 0) drift more (hs cos 0.96-0.99) without changing
what is kept (nothing). The tokenizer matches the official one on 16 test strings
(`tests/tokenizer_cases.json`). Preprocessing (antialiased bilinear to 1008x1008) matches torchvision to cos 0.999995.

## Engine notes

- **Precision**: residual streams are f32, and GEMM inputs bf16 with f32 accumulation (mma.sync). LayerNorm statistics are f32, and
  attention softmax is f32 on bf16 Q/K/V.
- **ViT**: tokens stay in window-major order for the whole trunk. A windowed block is then one batched attention over 9 contiguous
  runs, and a global block uses per-token RoPE angles (global RoPE is interpolated to the 24x24 pretraining grid, scale 1/3). The
  absolute position table is tiled 24x24 as in the official `tile_abs_pos`.
- **Caching**: an image's trunk, the three necks (detector, interactive, propagation), `conv_s0`/`conv_s1` and the geometry-encoder
  CLS token are computed once per image. Text features are cached per prompt string.
- **Tracker memory encoder input**: the per-object masks (`sigmoid * 2 - 1`) and conditioning flags are multiplexed into 32
  channels at 1008x1008, resampled (antialiased bilinear) to 1152, then put through 4 stride-2 convs with LayerNorm2d and GELU. Then
  come `pix_feat_proj`, 2 ConvNeXt blocks (the layer-scale gamma is folded into `pwconv2`), and the per-slot no-object embeddings
  (padding slots included, as upstream).
- Environment: `LOKI_MODELS` (model folder or path list) and `LOKI_SEGMENT_CACHE_FRAMES` (video feature cache, frames).

## Gotchas

- The official `Sam3Processor` (image API) multiplies the presence probability in twice on the 3.1 detector. That detector already
  folds presence into its logits (`supervise_joint_box_scores`), so the processor reports truck 0.763 where the joint score is 0.869.
  This tool reports the joint score, which is what the official video predictor uses (0.879 there with its own resize).
- In the official demo, the second and later clicked objects enter through the mask-input path: the object's own logits are scaled
  x20-10, pooled to 72x72, and the score is pinned to 10. The tool decodes every clicked object the same way, which is cleaner.
  `video_check` with `PARITY=1` reproduces the official path for validation.
- A prompted frame's masks are consolidated through video resolution (bilinear up, antialiased down) before tracking, as upstream.
  Skipping that step costs about 0.002 cosine on the prompted frame.
- ComfyUI's SAM3 port differs from the official model in several places: the sine position encoding, the order of the box sine
  embedding, and the geometry-encoder CLS projection. It can't serve as the reference.
- The interactive decoder's IoU head returns logit-like values (about 1-4), not IoUs.
