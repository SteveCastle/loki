# Lowkey Media Viewer

**A fast, minimalist viewer for the images, videos, audio, and comics on your hard drive. Free and open source. Built for people who care about their personal media library and want to actually look at it instead of fighting their software.**

<video src="https://stevecastle.github.io/loki/static/viewer-overview.mp4" autoplay loop muted playsinline controls></video>

Most media apps want to take over your files — importing them into a private library, re-encoding them, asking you to sign in. Lowkey Media Viewer doesn't. Point it at a directory and it just opens. Files stay where you put them, in the formats you already have. Tags live in a local SQLite database you own and can back up. No account, no cloud sync, no telemetry.

It's built for the kind of library that's hard to handle with existing tools: tens of thousands of images and videos accumulated over years, mixed formats, comic archives, audio files, screenshots, weird old `.webm`s. If you want to view those files, tag them, find them again later, and never lose them to a service shutting down — this is for you.

**Download free** at [**lowkeyviewer.com**](https://lowkeyviewer.com) for **Windows, macOS, and Linux**.

This repository is also home to **Lowkey Media Server** — an optional companion that adds batch processing, AI auto-tagging, transcription, and a web UI. See further down.

---

## Why use it

- **Fast.** Opens enormous folders without hanging. Renders 4K videos and high-res images without stutter. Battle-tested against libraries with 100,000+ files.
- **Local-first.** Your media stays where you put it. Tags live in a SQLite database in your home directory that you own and can back up. No sync, no cloud, no account.
- **Free and open source.** MIT licensed. No paywalled features. Support the project on [Patreon](https://www.patreon.com/c/lowkeyviewer) if you want, but you never have to.
- **Quiet by design.** Dark UI, no chrome you don't need, full-screen modes, hotkeys for everything.
- **Handles awkward stuff.** Comic archives (`.cbz`, `.cbr`) open like directories. Video subtitles get picked up automatically. Audio gets a WebGPU visualizer. Transcripts are editable in place.
- **Cross-platform.** Native installer for Windows, signed/notarized DMG for macOS (Apple Silicon + Intel), AppImage for Linux.

---

## Features

### Viewing

- **Images** — JPG, JPEG, PNG, GIF, WebP, JFIF, BMP, SVG, and animated GIF
- **Video** — MP4, MOV, MKV, WebM, FLV, M4V, with subtitle pickup (SRT/VTT) and audio-track selection on multi-track files
- **Audio** — MP3, M4A, AAC, OGG, WAV with a WebGPU spectrum visualizer
- **Comic archives** — `.cbz` and `.cbr` (CBR uses a bundled UnRAR binary so it Just Works on every platform)
- **Detail, grid, and masonry layouts** with multiple scale modes, shuffle, and infinite-scroll
- **Background autoplay** with looping, fit modes, and configurable transitions

### Organizing

- **Tags and categories** — drag any tag onto any image to apply it. Bulk-apply to a whole filter. Hold Ctrl to apply to every visible item.
- **Custom tag previews** — use any image you're viewing as the cover for a tag
- **Stored category slots** (1–9) — jump between your most-used categories with a single keystroke
- **Command palette** for fast tag/category/file actions
- **Context palette** (Shift + right-click) for in-place edits
- **Battle mode** — pairwise comparison with ELO ranking, then apply the resulting order as a custom sort
- **Duplicate detection** with perceptual hashing
- **Reorder tags and media** by drag-and-drop within a category

### Finding things

- **Tag filtering** with OR / AND / Exclusive modes
- **Fuzzy tag search** across every tag in every category
- **Filename search** as you type
- **Description search** if you've generated descriptions via the server
- **Transcript-driven video navigation** — click a line of a transcript to jump to that moment

### Metadata & transcripts

- **Tag and category descriptions** — markdown notes that travel with your tags
- **In-place transcript editing** — Enter to save, Shift+Enter for a newline, per-cue delete, insert button
- **In-panel substring search** through transcripts with prev/next navigation
- **Captions and subtitles** picked up from SRT/VTT sidecars next to video files (auto-converted to WebVTT with BOM/CRLF normalization)

### Built for big libraries

- Virtualized scrolling — 50,000+ tags or media items render without stutter
- Thumbnails generated on-demand and cached
- Selection state, query state, and library state persisted across launches
- File-association registration on Windows and macOS so you can set it as your default viewer

---

## Get it

Prebuilt binaries are at [**lowkeyviewer.com**](https://lowkeyviewer.com). Builds are signed/notarized on macOS, NSIS-installed on Windows, AppImage on Linux. New releases ship straight from `master`.

If you'd rather build from source, see [Building from source](#building-from-source) below.

---

## Lowkey Media Server (optional companion)

> **⚠️ The server is in beta.** The viewer is stable. Only the server has rough edges — expect occasional breaking changes, missing features, and bugs that need a workaround. The two products are versioned independently in spirit but ship from the same release.

The server is what you reach for when the desktop app shouldn't block on a long-running task — auto-tagging 10,000 photos with an AI model, transcribing a 4-hour video, transcoding a library to adaptive HLS, ingesting a YouTube channel. Run it on the same machine as the viewer, or on a NAS/home server.

What you get:

- **A job queue** with real-time progress via SSE — start things, walk away, come back later
- **Visual workflow editor** to chain tasks into reusable pipelines (transcode → screenshot → auto-tag → ingest)
- **Auto-tagging** using ONNX models (WD-EVA02-Large-Tagger v3) or Ollama vision models
- **Transcription** of any video via Faster-Whisper
- **Ingestion** from local folders, YouTube (via yt-dlp), galleries (via gallery-dl), and Discord exports
- **HLS adaptive streaming** so the viewer can scrub through massive videos remotely
- **FFmpeg toolkit** with 16 presets (scale, convert, extract audio, screenshots, thumbnail sheets, etc.) plus raw passthrough
- **S3-compatible storage** alongside local disks
- **A web UI** for everything above — works in any browser, embeds the same React app the desktop viewer uses
- **Browser extensions** for Chrome and Firefox — right-click a page and send it to your queue

Authentication is on by default (JWT + bcrypt). First launch creates a temporary `admin` / `admin` user and forces you through a setup wizard to replace it.

The viewer detects the server automatically once both are running and unlocks the bulk-job actions.

**Get it the same way:** prebuilt binaries on [lowkeyviewer.com/server](https://lowkeyviewer.com/server), or in Docker. Full setup, configuration, environment variables, task catalog, and API reference live in [`media-server/README.md`](media-server/README.md).

---

## Building from source

The viewer is an Electron + React + TypeScript app. The server is a single Go binary. They live in the same repo because they share a React renderer.

### Prerequisites

- **Node 18** (CI builds against `18.x`)
- **npm** — `package.json` declares `"packageManager": "npm@10.9.0"`. Yarn is no longer supported; there's no `yarn.lock`.
- **Go 1.24+** (only for the server)
- A C compiler is **not** required for either product. SQLite uses pure-Go drivers everywhere.

### Build & run the viewer

```bash
npm install
npm start        # webpack-dev-server + electronmon
npm test         # builds first, then runs jest
npm run package  # build a distributable for the current OS
```

`npm run package` produces an NSIS installer on Windows, a universal DMG on macOS, and an AppImage on Linux. Artifacts land in `release/build/`. FFmpeg/FFprobe binaries are auto-downloaded as part of packaging.

### Build & run the server

```bash
npm run build:server   # builds the React UI, embeds it, then go build
cd media-server
./media-server          # or media-server.exe on Windows
```

Then open <http://localhost:8090> and log in as `admin` / `admin`.

For faster iteration:

```bash
npm run build:web       # build the renderer and copy into media-server/loki-static/
cd media-server
go run .
```

After renderer changes, re-run `build:web` before the Go binary will serve the updated assets (the static files are embedded at compile time).

### Tests

```bash
npm test               # viewer (Jest)
npm run test:server    # server (go test ./...)
```

---

## Repo layout

```
loki/
├── src/                # Electron app
│   ├── main/           # Main process (Node): IPC, archive extraction, thumbnails, transcripts
│   └── renderer/       # React + XState SPA (also embedded by the media server)
├── media-server/       # Go module: HTTP API, auth, job queue, HLS, S3 storage, web UI
│   ├── auth/           # JWT + bcrypt user auth
│   ├── tasks/          # Self-registering task implementations
│   ├── jobqueue/       # SQLite-backed job + workflow DAG engine
│   ├── storage/        # Local + S3-compatible storage registry
│   ├── runners/        # Worker pool that dispatches jobs to task fns
│   ├── deps/           # On-demand download manager (ffmpeg, yt-dlp, gallery-dl, whisper, onnx)
│   ├── renderer/       # Go HTML templates for the admin web UI
│   └── loki-static/    # Built React renderer embedded at compile time
└── docs/               # Source for lowkeyviewer.com
```

`src/renderer/platform.ts` is the abstraction layer that lets the same React code run as the Electron renderer process or as the web UI served by the Go server.

## Releases

`.github/workflows/release.yml` runs on every push to `master`:

1. Runs the Electron Jest tests and Go tests.
2. Tags the commit `vX.Y.Z` from `package.json` if the tag doesn't already exist.
3. Builds the Electron app for Windows, macOS (arm64 + x64), and Linux in parallel.
4. Cross-compiles the Go server for `windows/amd64`, `darwin/{arm64,amd64}`, and `linux/amd64`.
5. Generates a changelog from commits and contributors since the previous tag.
6. Publishes everything to GitHub Releases.

To cut a release: bump the version in `package.json`, `package-lock.json`, `release/app/package.json`, `release/app/package-lock.json`, and `docs/index.html`, then push to `master`.

## Contributing

Fork, branch, and open a pull request against `master`. If you've forked the project and built something interesting on top, I'd love to hear about it.

## License

MIT — see [LICENSE](LICENSE).
