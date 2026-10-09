# tools/ — standalone AI CLIs

Small, composable command-line tools that run AI models locally on an NVIDIA GPU, written from scratch in CUDA and Rust
(no Python, no PyTorch). The Lowkey Media Server's `retouch` and `reshoot` tasks, and the Transform flow in the
app, drive them; they are just as usable from a shell or an agent.

## Quickstart

**You need** Windows or Linux and an NVIDIA RTX 40-series (or newer) GPU with 24 GB of video memory. Nothing else: no Python, no CUDA toolkit
(the NVIDIA driver is enough). `loki-reshoot` also uses ffmpeg, which it fetches by itself on Windows.

### 1. Download

Latest release: **[github.com/SteveCastle/loki/releases/latest](https://github.com/SteveCastle/loki/releases/latest)**

| tool | Windows | Linux | checksums |
|---|---|---|---|
| **loki-retouch** (images) | [loki-retouch-windows-amd64.zip](https://github.com/SteveCastle/loki/releases/latest/download/loki-retouch-windows-amd64.zip) | [loki-retouch-linux-amd64.zip](https://github.com/SteveCastle/loki/releases/latest/download/loki-retouch-linux-amd64.zip) | [windows](https://github.com/SteveCastle/loki/releases/latest/download/loki-retouch-windows-amd64.zip.sha256) · [linux](https://github.com/SteveCastle/loki/releases/latest/download/loki-retouch-linux-amd64.zip.sha256) |
| **loki-reshoot** (video) | [loki-reshoot-windows-amd64.zip](https://github.com/SteveCastle/loki/releases/latest/download/loki-reshoot-windows-amd64.zip) | [loki-reshoot-linux-amd64.zip](https://github.com/SteveCastle/loki/releases/latest/download/loki-reshoot-linux-amd64.zip) | [windows](https://github.com/SteveCastle/loki/releases/latest/download/loki-reshoot-windows-amd64.zip.sha256) · [linux](https://github.com/SteveCastle/loki/releases/latest/download/loki-reshoot-linux-amd64.zip.sha256) |

Each zip holds the executable and its README. (If you run the [Lowkey Media Server](../media-server), it installs these for you the first time
you use a Transform: no download needed.)

### 2. Put them on your PATH

```powershell
# Windows (PowerShell): unzip, then put the .exe files in a folder on your PATH, e.g. ~\bin
Expand-Archive loki-retouch-windows-amd64.zip -DestinationPath .
Copy-Item .\loki-retouch-windows-amd64\loki-retouch.exe $HOME\bin\
```

```bash
# Linux
unzip loki-retouch-linux-amd64.zip && install -m 755 loki-retouch-linux-amd64/loki-retouch ~/.local/bin/
```

Check it runs (no GPU is touched yet): `loki-retouch --version`.

### 3. Run

```
loki-retouch --preset restore old_photo.jpg                       # clean up a photo      -> old_photo_restored.png
loki-retouch -p "make it night, with rain" photo.jpg             # edit by prompt
loki-retouch --upscale 2 small.jpg                               # faithful 2x upscale
loki-reshoot --animate photo.jpg -d 5                            # a 5 s living photo with sound
loki-reshoot -i face.png -a theme.mp3 -P prompt.txt -o clip.mp4  # image + music -> video
```

The first run downloads the model files (**17 GB** for loki-retouch, **42 GB** for loki-reshoot) from Hugging Face into a `models/` folder beside
the executable (or `$LOKI_MODELS`). It says so before it starts, resumes if interrupted, and never repeats. Every result is a file next to the
input and its path is printed, so the tools chain: `loki-retouch --preset restore -o - old.jpg | loki-reshoot --animate - -d 5 -o alive.mp4`.
`--help` on either tool lists every option; `loki-reshoot --prompt-guide` prints how to write a good video prompt.

## What's here

| tool | does | model | dir |
|---|---|---|---|
| **loki-retouch** | edit images by prompt, upscale, restore, composite, wallpapers | Qwen Image 2.1 | [`loki-retouch/`](loki-retouch) |
| **loki-reshoot** | reference images / videos / audio → video with generated sound | MiniMax H3 | [`loki-reshoot/`](loki-reshoot) |

They share one set of conventions (streams, flags, exit codes, model discovery): see [`CONVENTIONS.md`](CONVENTIONS.md).
`lokictl` (the media-server client) is separate and unrelated.

## Getting the binaries

- **Through the Lowkey Media Server (no setup)**: the engines are dependencies like the tagger and Whisper. The first
  Retouch/Reshoot job installs what it needs, logging progress in the job: the executable (about 50 MB, from the latest
  GitHub release, verified against its `.sha256`) and the model files (17 GB / 42 GB, from Hugging Face, SHA-256 pinned
  in `media-server/deps/models/manifest.json`, resumable). The setup wizard and the Dependencies page offer the same
  installs ahead of time ("AI image editing", "AI video from references"). The weights are shared: the server runs the
  engine with `LOKI_MODELS` pointing at its copy, and ffmpeg comes from the server's bundled one. A `loki-retouch` /
  `loki-reshoot` already on `PATH` always wins, and then nothing is installed for you.
- **Releases**: see the [Quickstart](#quickstart) above (zips and `.sha256` checksums attached to every GitHub release, built by CI).
- **From source**: Rust (stable) plus CUDA Toolkit 12.6+ (`nvcc`) and a host C++ compiler (MSVC on Windows, gcc ≤ 13 on Linux), then
  `cargo build --release` inside the tool's directory. Kernels are compiled for sm_89 (RTX 40 series) with a compute_80 PTX fallback.
- **Models** are not in the archives (17 GB for loki-retouch, 42 GB for loki-reshoot). Each tool finds them next to the executable,
  in `./models`, or in `$LOKI_MODELS`, and downloads whatever is missing from Hugging Face on first use (resumable). Keeping the
  executables and a shared `models/` folder together (this repo's author uses `~/bin`) means every tool and the media server find them.
  If you change a model file name or add one, update `media-server/deps/models/manifest.json` too (`qwen-image-2.1`,
  `minimax-h3-ref2va`; the `loki-*` tool entries point at release assets and need no edit).
- Runtime needs only the NVIDIA driver. `loki-reshoot` also needs `ffmpeg` (downloaded automatically on Windows when missing).
- macOS is not supported (no CUDA).

## CI

| workflow | when | what |
|---|---|---|
| [`tools.yml`](../.github/workflows/tools.yml) | PRs and branch pushes touching `tools/**` | build, test, smoke-test and package both tools on Windows and Linux; archives are kept for 7 days |
| [`release.yml`](../.github/workflows/release.yml) job *Build AI tools* | pushes to `master` | the same build; archives are attached to the GitHub release next to the Electron app and the media server |

Both use the composite action [`.github/actions/build-ai-tool`](../.github/actions/build-ai-tool/action.yml). Hosted runners have no GPU,
so the CI runs the GPU-free tests and checks that the executable starts and prints its help with no NVIDIA driver present; it installs only
what compiling needs (`nvcc`, `cudart` headers, CCCL on Linux; the full toolkit on Windows). In `release.yml` the job is `continue-on-error`,
so a toolchain hiccup never blocks the app/server release (the tools are then simply missing from that release): remove that line once the
jobs have proven stable.

## Tests

```bash
# what CI runs (no GPU needed)
cd tools/loki-retouch && cargo test --release --lib --test tokenizer
cd tools/loki-reshoot && cargo test --release --lib --test tokenizer --test te_cpu --test avae_cpu --test media   # media needs ffmpeg

# by hand on a machine with an NVIDIA GPU and the models (not in CI)
cargo test --release --test kernels                       # kernel tests against CPU references (both tools)
cargo test --release --test sampler --test vvae_frames    # loki-reshoot
cargo run  --release --bin dit_check                      # per-component checks against ComfyUI dumps (ref/*.py write the dumps)
```

`ref/` holds the scripts that dump reference tensors from ComfyUI's own code; the dumps (`ref_out/`), models, test media and results are
git-ignored and never committed.

## Layout

```
tools/
  CONVENTIONS.md        shared conventions of the loki- tools
  loki-retouch/         Cargo crate: src/, kernels/*.cu (nvcc -> embedded fatbins), tests/, ref/, .claude/skills/
  loki-reshoot/         same; plus reference_4kify/ (Qwen-Image sources kept as reading material, not compiled)
```
