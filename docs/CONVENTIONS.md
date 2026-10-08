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
