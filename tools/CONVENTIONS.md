# loki- toolkit conventions

Small tools, one job each, composable with each other and with ordinary Unix tools. Every `loki-*` tool follows these rules
(`loki-retouch`: image editing, `loki-reshoot`: references -> video with sound, `loki-segment`: masks of images and tracked video
objects, `loki-dream`: text -> image; `lokictl`, the Lowkey Media Server client, is a separate tool and not part of this contract).

## Streams
- **stdin**: an input may be `-`. The tool reads that one input from stdin (at most one per run). Images are sniffed from content.
- **stdout**: the result and nothing else. Either the path of the written file (one line per result), one JSON object per result with
  `--json`, or the media bytes themselves with `-o -` (PNG for images, fragmented mp4 for video). A tool refuses to write binary media to a
  terminal.
- **stderr**: all progress, timings and warnings. `-q/--quiet` silences them. Errors always print, as a single line
  `<tool>: error: <message>`. Large downloads (models) are announced on stderr even with `-q`.
- **exit status**: 0 ok, 1 runtime error, 2 usage error (clap default).

## Flags with the same meaning everywhere
`-p/--prompt TEXT`, `-P/--prompt-file FILE` (`-` = stdin), `-o/--out PATH|DIR|-`, `-q/--quiet`, `--json`, `--seed N`, `--steps N`,
`--show-prompt` (print the final prompt and exit), `--dit / --text-encoder / --vae ... PATH` (model overrides), `-h/--help`, `-V/--version`.
`--help` ends with OUTPUT / EXAMPLES sections written for agents: it must be enough to use the tool without any other documentation.

## Provenance: the `loki` generation record
Every result carries one JSON object describing the run, so a file found months later still says what made it. The Lowkey Media
Viewer and Server show it in the metadata panel's Generation section (next to ComfyUI / WebUI records), with the prompt, the
settings, the inputs and a copyable command line. Where it lives:
- **PNG** (loki-retouch, loki-segment masks): an `iTXt` chunk with the keyword `loki`, before the image data.
- **mp4** (loki-reshoot): the container's `comment` tag (`ffprobe -show_entries format_tags=comment`).

The record is recognised by two marker fields and is otherwise open; fields are added over time, never renamed or re-typed,
so a reader of version 1 keeps working:

```json
{"loki": 1, "tool": "loki-retouch", "version": "0.1.0", "model": "Qwen Image 2.1", "created": "2026-10-16T21:04:05Z",
 "args": ["--preset", "4kify", "photo.jpg"],           // the command line as given (prompt-file contents are NOT inlined)
 "prompt": "...",                                        // the final prompt text sent to the model
 "preset": "4kify", "seed": 0, "steps": 25, "shift": 0.69,
 "width": 3840, "height": 2160,                          // delivered size; generated_width/height when the model drew at another
 "ref_resolution": 0,
 "inputs": [{"tag": "<image1>", "path": "photo.jpg", "width": 1024, "height": 768}]}
```
loki-reshoot adds `model_id` (`h3` | `fast`, with `model` "MiniMax H3" / "FastH3 Omni"), `frames`, `fps`, `seconds`, `audio`, `crf`, `ref_image_size`, `native`/`fit`, `animate`/`describe`/`shake`,
and its inputs are tagged `<Picture n>` / `<Video n>` (`start`, `duration`, `audio_tag`) / `<Audio n>`. An input read from stdin is
reported as `-`. `--json` prints the same record under `generation`. Parsers: `src/renderer/generation-metadata.ts` (the app),
writers: each tool's `src/genmeta.rs`.

`--meta JSON` (or `--meta-file FILE`) stores the caller's own JSON object verbatim as the record's `meta` field, and is left out
of `args`. A value that is not a JSON object fails the run before any model loads. The Lowkey Transform flow puts its recipe
there (`{"kind": "loki-transform", "v": 1, "intent", "settings", "paths"}`: the intent and every setting as the user set them,
and the job's inputs in order), which is what lets the Generation panel save a result as a Transform task again
(`src/renderer/components/transform/recipe.ts`). The media server only passes `--meta` to an engine whose `--help` lists it.

## Worker mode (`--worker`)
Every tool can run as a resident process for a supervisor (`loki-worker`, see [`worker/`](worker)): `loki-<tool> --worker`
keeps its models on the GPU between requests instead of loading them for every run. It speaks JSON lines:
- **stdin**: one request per line. A run is `{"id": "j1", "args": ["-p", "...", "-o", "out.png", "in.png"]}` — exactly the
  command line the tool takes (without the executable). Other commands: `{"id", "cmd": "warm"}` (resolve, download and load
  the models; loki-reshoot takes `"model": "fast"`), `{"id", "cmd": "evict"}` (drop every idle model), `{"id", "cmd":
  "status"}`, and `{"cancel": "j1"}`, which stops the named run at its next sampling step. loki-segment adds named
  sessions on the same resident model: `{"id", "cmd": "session", "session": NAME, "req": {...a --serve request...}}` and
  `{"cmd": "close-session", "session": NAME}`.
- **stdout**: protocol events only. `{"event": "ready", "tool", "version", "model", "protocol": 1, "gpu"}` once at start;
  per request the run's result lines as `{"id", "event": "stdout", "line"}` (sent when the run ends), then `{"id", "event":
  "done", "ok", "error", "oom", "cancelled", "reply", "secs", "resident": [{"key", "mb"}], "gpu": {"total_mb", "free_mb"}}`.
- **stderr**: the run's progress log exactly as on the command line, closed by the sentinel line `\x1eloki-worker done <id>`
  so a supervisor can attribute every line to its request.
- **memory**: before a load, the least recently used models the current request is not using are evicted until the load
  fits with `LOKI_WORKER_RESERVE_MB` (default 1.5-5 GB per tool) left for activations; a run that still runs out of memory is
  retried once with every idle model evicted. On a 24 GB card the big phases still take turns; on 48-80 GB cards a tool's
  whole pipeline stays loaded and a warm run skips loading entirely.
- **failure**: a CUDA error that poisons the context makes the tool report the request and exit with status 3; the
  supervisor restarts it. Input from stdin (`-`) and output to stdout (`-o -`) are refused in worker mode.

## Environment
- `LOKI_MODELS`: extra folder(s) searched for model files (after the binary's folder and `./models`; an OS path list, `;` on
  Windows, `:` on Linux).
- `LOKI_MODELS_DOWNLOAD`: where missing model files are downloaded (default `<binary dir>/models`). Downloads are resumable
  and take a lock (`<file>.lock`), so several workers sharing one models volume fetch each file once.
- `LOKI_CUDA_ARCHS` / `LOKI_CUDA_PTX` (build time): SASS architectures (default `89`) and the PTX fallback (default `80`).
  The FP8 kernels of loki-retouch, loki-reshoot and loki-dream need 89 or later; the worker image builds `89,90` + PTX `90`.
- `LOKI_PROFILE=1`: per-kernel GPU timing report on stderr. Other `LOKI_*` variables are engine tuning knobs (see each README).

## Models and hardware
Standalone CUDA engines (no Python/PyTorch); runtime needs only an NVIDIA driver (CUDA 12.x build, sm_89 kernels, RTX 4090 class,
24 GB). Models are discovered locally and downloaded (resumable) from Hugging Face when missing. Phases load and evict models one after
another so a run fits in 24 GB.

## Repository rules
Media, test fixtures, model weights, reference dumps and generated output are never committed (see `.gitignore`).
Agent skills live in `.claude/skills/<tool>/SKILL.md` (read by Claude Code and OpenCode).
