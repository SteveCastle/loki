# Loki

Loki is the source repository for two companion products that share a single React frontend:

- **Lowkey Media Viewer** — a free, minimalist Electron desktop app for viewing and curating images, video, audio, and comic book archives on Windows, macOS, and Linux.
- **Lowkey Media Server** — a Go HTTP server with a web UI, authenticated multi-user access, a job queue, an HLS streamer, and a workflow DAG engine for batch processing media. It also serves the same React UI as a web app so the desktop and browser experiences stay in sync.

<video src="https://stevecastle.github.io/loki/static/viewer-overview.mp4" autoplay loop muted playsinline controls></video>

Prebuilt binaries and end-user documentation live at [lowkeyviewer.com](https://lowkeyviewer.com). Media server docs are at [lowkeyviewer.com/server](https://lowkeyviewer.com/server).

> **Status: beta.** Both products are in active development and ship from `master` on every push. Versions are tagged automatically by CI. Expect occasional breaking schema changes between releases — the viewer migrates on startup, but back up your `dream.sqlite` if you're paranoid.

---

## Repo Layout

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

The renderer under `src/renderer/` targets both environments. `src/renderer/platform.ts` abstracts over Electron IPC vs. HTTP so a single codebase powers the desktop app and the server's web UI.

---

## Lowkey Media Viewer (Electron)

A fast, distraction-free viewer for large personal media libraries. Highlights:

- Native playback for images, video, audio (with multi-track / subtitle pickup, transcript-driven navigation, and editable cues)
- Comic book archives (`.cbz`, `.cbr`) open like directories from the file dialog or drag-and-drop. CBR is supported out of the box via a bundled UnRAR binary.
- Tag and category system: drag-to-tag, bulk apply, category-scoped stored slots, and a context palette
- Battle mode (ELO rating), duplicate detection, ORB-based perceptual hashing
- Grid, masonry, and detail layouts; scale modes; shuffle; bulk tagging
- WebGPU audio visualizer for audio files
- Optional pairing with Lowkey Media Server for HLS playback of large videos

### Build Requirements

- **Node 18** (CI builds against `18.x`)
- **npm** (yarn is no longer used; `package.json` declares `"packageManager": "npm@10.9.0"`)
- FFmpeg and FFprobe binaries under `src/main/resources/bin/<platform>/` — these are auto-downloaded by `npm run package`, or you can run `./download-binaries.sh` in each platform subdirectory manually. ExifTool and FFplay are no longer required; dimensions come from FFprobe.
- On Windows, the bundled `unrar.exe` ships in-tree so CBR Just Works.

### Development

```bash
npm install
npm start       # webpack-dev-server + electronmon
```

### Tests

```bash
npm test              # builds first, then jest
npx jest <pattern>    # single test (a build must already exist)
```

### Build a Distributable

```bash
npm run package
```

`electron-builder` produces:

- **Windows:** NSIS installer (`.exe`)
- **macOS:** universal DMG (`arm64` + `x64`)
- **Linux:** AppImage

Binaries land in `release/build/`. CI builds and publishes all three targets to GitHub Releases on every push to `master`.

---

## Lowkey Media Server (Go)

> **Beta with auth enabled by default.** On first launch the server creates a temporary `admin` / `admin` account, then forces you through a setup wizard at `/login?setup=true` to replace it with a real user. JWT sessions are signed with `LOWKEY_JWT_SECRET` (auto-generated and persisted if not set) and stored as `HttpOnly` cookies. All `/api/*`, `/media/*`, and admin pages require login. There is still no per-route authorization beyond "logged-in" — every authenticated user gets admin role today, so don't give credentials to anyone you wouldn't trust with the box itself.

A companion service for the long-running work the desktop app shouldn't block on:

- **HLS adaptive streaming** with on-disk segment cache (480p / 720p / 1080p plus passthrough)
- **Auto-tagging** via ONNX models (WD-EVA02-Large-Tagger v3) or Ollama vision models
- **Transcription** via Faster-Whisper (bundled in the "Generate Metadata" task)
- **Ingestion** from local directories, YouTube (yt-dlp), galleries (gallery-dl), and Discord exports
- **Workflow DAGs** — chain any registered task; persisted in SQLite, editable from the web UI
- **FFmpeg toolkit** — 16 preset operations (scale, convert, extract audio, screenshot, thumbnail sheet, etc.) plus raw command pass-through
- **S3-compatible storage** alongside local disks; per-root thumbnail prefixes; signed URLs
- **File browser** over local roots and S3 buckets
- **LoRA dataset builder** for AI training workflows
- **SSE** streaming for real-time job updates
- **System tray** integration on Windows and macOS
- **Browser extensions** (Chrome + Firefox) for sending URLs into the job queue

See [`media-server/README.md`](media-server/README.md) for full server documentation, Docker setup, environment variables, the task catalog, and the API reference.

### Quick Start

From the repo root:

```bash
npm run build:server   # builds the renderer, copies it into media-server/loki-static/, then go build
cd media-server
./media-server         # or media-server.exe on Windows
```

Open <http://localhost:8090>, log in as `admin` / `admin`, and follow the setup wizard to create your real account.

### Running the Server in Development

```bash
npm run build:web      # build the renderer and copy to media-server/loki-static/
cd media-server
go run .
```

After renderer changes, re-run `build:web` before the Go binary will serve the updated assets — the static files are embedded at compile time.

### Server Tests

```bash
npm run test:server
# or
cd media-server && go test ./...
```

---

## Releases & CI

`.github/workflows/release.yml` runs on every push to `master`:

1. Runs Electron Jest tests and Go tests.
2. Tags the commit `vX.Y.Z` from `package.json` if the tag doesn't already exist.
3. Builds Electron app for Windows, macOS (arm64 + x64), and Linux in parallel.
4. Cross-compiles the Go server for `windows/amd64`, `darwin/{arm64,amd64}`, and `linux/amd64`.
5. Generates a changelog from commits and contributors since the previous tag.
6. Publishes everything to GitHub Releases.

To cut a release, bump the version in `package.json`, `package-lock.json`, `release/app/package.json`, `release/app/package-lock.json`, and the `RELEASE_BASE` / version label in `docs/index.html`, then push to `master`.

---

## Contributing

Fork, branch, and open a pull request against `master`. If you've forked the project and built something interesting on top, I'd love to hear about it.

## License

MIT — see [LICENSE](LICENSE).
