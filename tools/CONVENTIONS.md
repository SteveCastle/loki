# loki- toolkit conventions

Small tools, one job each, composable with each other and with ordinary Unix tools. Every `loki-*` tool follows these rules
(`loki-retouch`: image editing, `loki-reshoot`: references -> video with sound; `lokictl`, the Lowkey Media Server client, is a
separate tool and not part of this contract).

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
- **PNG** (loki-retouch): an `iTXt` chunk with the keyword `loki`, before the image data.
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
loki-reshoot adds `frames`, `fps`, `seconds`, `audio`, `crf`, `ref_image_size`, `native`/`fit`, `animate`/`describe`/`shake`,
and its inputs are tagged `<Picture n>` / `<Video n>` (`start`, `duration`, `audio_tag`) / `<Audio n>`. An input read from stdin is
reported as `-`. `--json` prints the same record under `generation`. Parsers: `src/renderer/generation-metadata.ts` (the app),
writers: each tool's `src/genmeta.rs`.

## Environment
- `LOKI_MODELS`: extra folder searched for model files (after the binary's folder and `./models`).
- `LOKI_PROFILE=1`: per-kernel GPU timing report on stderr. Other `LOKI_*` variables are engine tuning knobs (see each README).

## Models and hardware
Standalone CUDA engines (no Python/PyTorch); runtime needs only an NVIDIA driver (CUDA 12.x build, sm_89 kernels, RTX 4090 class,
24 GB). Models are discovered locally and downloaded (resumable) from Hugging Face when missing. Phases load and evict models one after
another so a run fits in 24 GB.

## Repository rules
Media, test fixtures, model weights, reference dumps and generated output are never committed (see `.gitignore`).
Agent skills live in `.claude/skills/<tool>/SKILL.md` (read by Claude Code and OpenCode).
