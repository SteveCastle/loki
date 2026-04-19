# Loki

Loki is the source repository for two companion products that share a single React frontend:

- **Lowkey Media Viewer** — a free, minimalist Electron desktop app for viewing and curating images, video, audio, and comic book archives on Windows and macOS.
- **Lowkey Media Server** — an HTTP server (Go) with a web UI and job queue for batch processing, auto-tagging, transcription, and media ingestion. Serves the same renderer as a web app.

https://stevecastle.github.io/loki/static/viewer-overview.mp4

Prebuilt binaries and end-user documentation are available at [lowkeyviewer.com](https://lowkeyviewer.com). The media server docs are at [lowkeyviewer.com/server](https://lowkeyviewer.com/server).

---

## Repo Layout

```
loki/
├── src/                # Electron app
│   ├── main/           # Main process (Node): IPC, menus, file I/O, archive extraction
│   └── renderer/       # React + XState SPA (also embedded by the media server)
├── media-server/       # Go module: HTTP API, job queue, web UI, task runners
│   ├── tasks/          # Self-registering task implementations
│   ├── jobqueue/       # SQLite-backed job + workflow DAG engine
│   ├── storage/        # Local + S3-compatible storage registry
│   ├── runners/        # Faster-Whisper, ONNX tagger, Ollama, FFmpeg wrappers
│   └── loki-static/    # Built renderer embedded at compile time
└── docs/               # Source for lowkeyviewer.com
```

The renderer under `src/renderer/` targets both environments. `src/renderer/platform.ts` abstracts over Electron IPC vs. HTTP so a single codebase powers the desktop app and the server's web UI.

---

## Lowkey Media Viewer (Electron)

A fast, distraction-free viewer for large personal media libraries. Highlights:

- Native playback for images, video, audio
- Comic book archives (`.cbz` / `.zip`) open like directories — opened from the file dialog or drag-and-drop
- Tag and category system with stored slots and a command palette
- Battle mode (ELO rating), duplicate detection, transcript-based video navigation
- Grid and masonry layouts, scale modes, shuffle, bulk tagging

### Build Requirements

- Node 18
- Yarn
- `ffmpeg`, `ffprobe`, `ffplay`, and `exiftool` placed in `src/main/resources/bin/<platform>/`

### Development

```bash
yarn
yarn dev
```

### Tests

```bash
npm test              # runs build first, then jest
npx jest <pattern>    # single test (build must exist)
```

### Build a Distributable

```bash
yarn package
```

Binaries land in `release/build/`.

---

## Lowkey Media Server (Go)

> **⚠️ Alpha.** The server binds to `:8090` and exposes your library over HTTP with no authentication by default. Run only on trusted networks or behind a firewall.

A companion service for long-running work the desktop app shouldn't block on: auto-tagging via ONNX models, Whisper transcription, Ollama-powered image descriptions, FFmpeg conversion, ingestion, and arbitrary workflows chained as a DAG.

See [`media-server/README.md`](media-server/README.md) for full server documentation, Docker setup, and API reference.

### Quick Start

From the repo root:

```bash
npm run build:server   # builds the renderer, copies it into media-server/loki-static/, then go build
cd media-server
./media-server         # or media-server.exe on Windows
```

Open <http://localhost:8090>.

### Running the Server in Development

```bash
npm run build:web      # build the renderer and copy to media-server/loki-static/
cd media-server
go run .
```

After renderer changes, re-run `build:web` before the Go binary will serve the updated assets (the static files are embedded at compile time).

### Server Tests

```bash
npm run test:server
# or
cd media-server && go test ./...
```

---

## Contributing

Fork, branch, and open a pull request against `master`. If you've forked the project and done something interesting with it, I'd love to hear about it.

## License

MIT — see [LICENSE](LICENSE).
