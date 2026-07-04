# Lowkey Media Server

Lowkey Media Server is the back-end companion to the Lowkey Media Viewer. It manages a SQLite media library, runs long-lived jobs (auto-tagging, transcription, ingestion, ffmpeg pipelines, HLS transcodes), serves files over HTTP with optional S3 storage, streams video as HLS, and embeds the same React UI the Electron app uses so the whole interface works in a browser.

> **⚠️ Beta software.** Authentication is now enabled by default — the server creates a temporary `admin` / `admin` user on first launch and forces you through a setup wizard to replace it. All `/api/*`, `/media/*`, and admin pages require a valid session. There is currently no per-route authorization beyond "is logged in," so every authenticated user has admin-level access. JWTs are signed with `LOWKEY_JWT_SECRET` (auto-generated and persisted on first run if not provided) and stored as HttpOnly cookies. Don't expose the server to the open internet without a reverse proxy and TLS.

<img width="1628" height="494" alt="Screenshot 2025-09-20 083904" src="https://github.com/user-attachments/assets/e814d2a5-7088-46b2-9a8c-d537b989b018" />

<img width="1661" height="1606" alt="Screenshot 2025-09-20 082929" src="https://github.com/user-attachments/assets/850672bc-da70-4269-8049-cc651b0abed6" />

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Docker Quick Start](#docker-quick-start)
- [Installation (End Users)](#installation-end-users)
- [Authentication](#authentication)
- [Development Setup](#development-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Available Tasks](#available-tasks)
- [API Documentation](#api-documentation)
- [Browser Extensions](#browser-extensions)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Authenticated multi-user access** — JWT sessions, bcrypt password hashes, HttpOnly cookies, setup wizard, user management UI.
- **Job queue** — create, monitor, copy, cancel, and clear long-running media-processing jobs. SQLite-backed so jobs survive restarts.
- **Workflow DAG engine** — chain tasks into reusable workflows, persisted under the `workflows` table; visual editor at `/editor`.
- **Media browser** — search, filter, paginate, preview, and tag your library from the web UI.
- **HLS adaptive streaming** — on-demand transcode to 480p / 720p / 1080p with on-disk segment cache (`/media/hls/...`).
- **Swipe mode** — paginated random-sample view designed for quick triage on touch devices.
- **File system browser** — list local roots and S3 buckets, drill into folders, ingest in-place.
- **Auto-tagging** — ONNX (WD-EVA02-Large-Tagger v3) or Ollama vision models against the tag set already in your DB.
- **Transcription** — Faster-Whisper integration (bundled under the "Generate Metadata" task).
- **Ingestion** — bulk import from local paths, YouTube (yt-dlp), arbitrary galleries (gallery-dl), or Discord exports.
- **FFmpeg toolkit** — 16 preset operations (scale, convert, extract audio, screenshot, thumbnail sheet, blur, crop, reverse, speed, caption, etc.) plus raw passthrough.
- **LoRA dataset builder** — assemble a captioned image dataset from tagged media.
- **Storage abstraction** — multiple local roots and S3-compatible buckets side-by-side. Per-root thumbnail prefixes. Default-root configurable.
- **Bundled binaries** — `ffmpeg`, `ffprobe`, `ffplay`, `exiftool`, `onnxtag`, and `onnxruntime` ship inside the release archive (no first-run downloads, no Gatekeeper headaches on macOS). Optional tools (`yt-dlp`, `gallery-dl`, `ollama`) are detected on PATH with copy-paste install instructions per OS. AI models are downloaded on demand from a checksummed manifest.
- **SSE updates** — `/stream` pushes live job state and download progress to all connected clients.
- **System tray** — Windows and macOS get a tray icon with Open Web UI / Quit shortcuts.
- **Browser extensions** — Chrome and Firefox extensions for sending the current URL to the job queue.

---

## System Requirements

- **Operating System:** Windows 10/11 (x64), macOS 11+ (arm64 or amd64), or Linux (amd64).
- **Disk Space:** ~250 MB for the release archive (binary + bundled ffmpeg / exiftool / onnxruntime); downloaded AI models add ~1–3 GB depending on which you enable.
- **RAM:** 4 GB minimum, 8 GB+ recommended once ONNX tagging or HLS transcodes are running.

The server is a single statically-linked Go binary. SQLite uses `modernc.org/sqlite` (pure Go) so **no CGO is required** at build time.

---

## Docker Quick Start

The fastest way to run the server is with Docker. The compose stack handles persistence, sane defaults, and a MinIO instance wired in as the default S3 storage root so everything works out of the box.

> **Build context note:** the image build needs the **repo root** as its context (the React SPA source lives there), not `media-server/`. The compose file handles this for you; for manual builds pass `-f media-server/Dockerfile` from the repo root.

### 1. Build and run

**With docker compose (recommended)** — from the `media-server/` directory:

```bash
cd media-server
docker compose up -d --build
```

Open **http://localhost:18090** (set `LOWKEY_PORT` to change the host port), log in as `admin` / `admin`, and follow the setup wizard to create your real account.

**Or build and run manually** — from the **repo root**:

```bash
docker build -f media-server/Dockerfile -t lowkey-media-server .
docker run -d --name lowkey-media-server \
  -p 8090:8090 \
  -v lowkey-data:/data \
  lowkey-media-server
```

Then open **http://localhost:8090**. The manual container has no storage roots until you configure some (next section); the compose stack comes with MinIO preconfigured.

### 2. Mount your media

Bind-mount local directories so the server can browse and process your files:

```bash
docker run -d --name lowkey-media-server \
  -p 8090:8090 \
  -v lowkey-data:/data \
  -v /path/to/photos:/mnt/photos:ro \
  -v /path/to/videos:/mnt/videos:ro \
  lowkey-media-server
```

Then register them as storage roots using environment variables (see below).

### 3. Environment variables

All configuration can be set via environment variables — no config file needed.

| Variable | Default | Description |
|----------|---------|-------------|
| `LOWKEY_DB_PATH` | `/data/db/media.db` | SQLite database path |
| `LOWKEY_DOWNLOAD_PATH` | `/data/downloads` | Download directory (deprecated — prefer a default root) |
| `LOWKEY_OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Ollama API endpoint |
| `LOWKEY_OLLAMA_MODEL` | `llama3.2-vision` | Vision model for descriptions and tagging |
| `LOWKEY_JWT_SECRET` | auto-generated and persisted | JWT signing secret. Override to share sessions across replicas. |
| `LOWKEY_DISCORD_TOKEN` | | Discord token for Discord export ingestion |
| `LOWKEY_FASTER_WHISPER_PATH` | | Path to faster-whisper binary (overrides the on-demand download) |
| `LOWKEY_ROOT_1`, `_2`, ... | | Local storage roots (see below) |
| `LOWKEY_DEFAULT_ROOT` | `1` | Which root receives uploads/downloads (1-based index or label) |
| `LOWKEY_ROOTS` | | JSON storage roots array with S3 support (see below) |

#### Storage roots via environment

**Local paths** — use numbered `LOWKEY_ROOT_<N>` variables. Format: `path` or `path:label`.

```bash
docker run -d --name lowkey-media-server \
  -p 8090:8090 \
  -v lowkey-data:/data \
  -v ~/photos:/mnt/photos:ro \
  -v ~/videos:/mnt/videos:ro \
  -e LOWKEY_ROOT_1=/mnt/photos:Photos \
  -e LOWKEY_ROOT_2=/mnt/videos:Videos \
  lowkey-media-server
```

**S3-compatible storage** — use `LOWKEY_ROOTS` with a JSON array. Supports all StorageRoot fields including S3 credentials, custom endpoints, and thumbnail prefixes. Mix local and S3 roots freely.

```bash
docker run -d --name lowkey-media-server \
  -p 8090:8090 \
  -v lowkey-data:/data \
  -e 'LOWKEY_ROOTS=[
    {"type":"local","path":"/mnt/photos","label":"Photos"},
    {
      "type":"s3",
      "label":"My S3 Bucket",
      "endpoint":"https://s3.us-east-1.amazonaws.com",
      "region":"us-east-1",
      "bucket":"my-media-bucket",
      "prefix":"images/",
      "accessKey":"AKIAIOSFODNN7EXAMPLE",
      "secretKey":"wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
      "thumbnailPrefix":"thumbs/"
    }
  ]' \
  lowkey-media-server
```

> When `LOWKEY_ROOTS` is set, it takes priority over any `LOWKEY_ROOT_<N>` variables. Either way, environment roots replace roots from the config file.

### 4. MinIO (S3-compatible storage)

The compose stack includes a MinIO instance wired in as the server's default storage root, so `docker compose up` gives you working S3 storage with nothing to configure:

| Service | URL | Credentials |
|---------|-----|-------------|
| Media Server | http://localhost:18090 | `admin` / `admin` on first run; replaced via setup wizard |
| MinIO Console | http://localhost:19001 | `admin` / `adminadmin` |
| MinIO S3 API | http://localhost:19000 | same |

The one-shot `minio-setup` container creates the `media` and `media-thumbnails` buckets automatically. Upload files through the MinIO console and they'll appear in the media server's file browser.

To use real S3 / Cloudflare R2 / local mounts instead, override `LOWKEY_ROOTS` (see above).

Stop everything with `make down` (or `docker compose down`; add `-v` to also delete the data volumes).

### 5. Makefile shortcuts

Run these from the `media-server/` directory (requires `make`; on Windows use the underlying `docker compose` / `docker` commands shown above):

```bash
make up               # Build and start all services (media server + MinIO)
make down             # Stop all services
make docker-build     # Build the image only
make docker-run       # Run standalone container (detached)
make docker-run-it    # Run in foreground (auto-removes on exit)
make docker-logs      # Tail logs
make docker-stop      # Stop and remove standalone container
make docker-rebuild   # Rebuild from scratch (no cache)
```

### 6. Data persistence

All server state lives in the `/data` volume:

```
/data/
  db/media.db                    # SQLite database (jobs, workflows, users, media, tags)
  downloads/                     # Downloaded media (when no S3 / explicit roots configured)
  lowkey-media-viewer/
    config.json                  # Auto-generated config incl. the JWT secret
    models/                      # On-demand AI model files (downloaded via the wizard)
```

The named volumes (`media-server_lowkey-data`, `media-server_minio-data`) survive container restarts and rebuilds.

---

## Installation (End Users)

Prebuilt binaries for Windows (x64), macOS (arm64 + amd64), and Linux (amd64) are published to GitHub Releases on every push to `master`. Download `media-server-<os>-<arch>` from the latest release.

1. **Run the binary.** On Windows it starts the background server and creates a system tray icon. On macOS and Linux it runs in the foreground; use your init system of choice to keep it alive.

   <img width="408" height="247" alt="Screenshot 2025-09-20 080659" src="https://github.com/user-attachments/assets/4b8a0141-08d4-4fb9-9c42-78db5dbd25ad" />

2. **Open <http://localhost:8090>.** Log in as `admin` / `admin`. You'll be redirected to a setup wizard — pick a real username and password. The default admin is deleted automatically once a real user exists.

3. **Open the Config tab** to verify the Lowkey Database path and configure model paths for ONNX tagging, Ollama, or Faster Whisper.

4. **Walk the welcome wizard** at <http://localhost:8090/> on first run. It shows what's bundled, gives install instructions for optional tools (`yt-dlp`, `gallery-dl`, `ollama`), and lets you pick which AI models to download. You can skip it and revisit any time.

   - [Download model files for the ONNX tagger](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main)
   - [Install Ollama for LLM-based descriptions and tagging](https://ollama.com/)
   - Faster-Whisper for transcription downloads with one click from the wizard's AI features step (or bring your own [standalone build](https://github.com/Purfview/whisper-standalone-win) via `fasterWhisperPath`)

   <img width="1233" height="1693" alt="Screenshot 2025-09-20 080416" src="https://github.com/user-attachments/assets/5eb008ae-88fb-4519-af03-4e55afbb6601" />

---

## Authentication

### How it works

- On first launch, if the `users` table is empty, the server creates a temporary `admin` user with password `admin`.
- All protected routes redirect to `/login?setup=true` while the only user is the default admin, forcing you to register a real account.
- `POST /auth/login` exchanges credentials for a JWT (returned in the response body **and** set as an `auth_token` HttpOnly cookie, `SameSite=Lax`, 24 hour expiry).
- Subsequent requests are accepted with either the cookie or an `Authorization: Bearer <token>` header. Browser extensions and the Electron app use the Bearer header; the web UI uses the cookie.
- `POST /auth/logout` clears the cookie.
- `GET /auth/status` reports whether a user is logged in and whether setup is still pending.
- `/auth/users` (admin) lists and deletes users; the last remaining user can't be deleted, so you can't lock yourself out.

### What's still TODO

- All authenticated users get full admin access. Per-route role checks (`RolePublic` / `RoleAdmin`) exist in the middleware but every real user is treated as admin.
- No password reset flow. Recover by stopping the server, deleting the user row from SQLite, and letting the default admin re-spawn.
- No rate limiting on `/auth/login`.

Override the JWT secret by setting `LOWKEY_JWT_SECRET` (otherwise one is generated and persisted in the config dir on first run).

---

## Development Setup

### Prerequisites

#### 1. Install Go

Requires **Go 1.24.0** or later (CI builds against `go1.24`).

**Windows:**

```powershell
winget install GoLang.Go
# or download the MSI from https://go.dev/dl/
go version
```

**macOS:**

```bash
brew install go
```

**Linux:**

Install the official tarball from <https://go.dev/dl/> or your distro's package.

#### 2. Install Git

Required for cloning the repository and Go module management.

#### 3. C Compiler

**Not required.** SQLite uses `modernc.org/sqlite` (pure Go) and the project sets `CGO_ENABLED=0` in CI. Cross-compilation just works.

#### 4. Optional: External Tools

For full functionality you'll want the following tools. Bundled ones ship with the server; optional ones you install yourself (the welcome wizard shows the right command for your OS); models download on demand:

| Tool | Purpose | How you get it |
|------|---------|----------------|
| FFmpeg / FFprobe / FFplay | Media probing, conversion, HLS, thumbnails | Bundled in the release |
| ExifTool | Image and video metadata extraction | Bundled in the release |
| ONNX Runtime + ONNX Tagger binary | ML inference plumbing | Bundled in the release |
| WD-EVA02-Large-Tagger v3 (model files) | Image auto-tagging | Welcome wizard → AI features |
| Faster-Whisper | Video transcription | Welcome wizard → AI features (one-click download), or set `fasterWhisperPath` to your own binary |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | YouTube and other video downloads | Optional — install via `brew`/`winget`/`pipx`; wizard shows the command |
| [gallery-dl](https://github.com/mikf/gallery-dl) | Image gallery downloads | Optional — install via `brew`/`pip`/`pipx`; wizard shows the command |
| [Ollama](https://ollama.com/) | LLM-based image descriptions / vision tagging | Optional — install via the official installer |

### Building from Source

#### 1. Clone the Repository

The Go module path is `github.com/stevecastle/shrike` even though the public repo is `loki`.

```bash
git clone https://github.com/stevecastle/loki.git
cd loki/media-server
```

#### 2. Build the React UI (required)

The Go binary embeds `loki-static/**` at compile time, so the renderer must be built first:

```bash
# From the repo root
npm install
npm run build:web   # builds the renderer and copies it into media-server/loki-static/
```

For a full one-shot build (renderer + server, with any running `media-server` killed first):

```bash
npm run build:server
```

#### 3. Build the Go Binary

```bash
# From media-server/

# Standard build
go build -o media-server .

# Optimized build
go build -ldflags="-s -w" -o media-server .

# Cross-compile (works with no extra setup since CGO is off)
GOOS=windows GOARCH=amd64 go build -ldflags="-s -w" -o media-server-windows-amd64.exe .
GOOS=darwin  GOARCH=arm64 go build -ldflags="-s -w" -o media-server-darwin-arm64 .
GOOS=linux   GOARCH=amd64 go build -ldflags="-s -w" -o media-server-linux-amd64 .
```

#### 4. Run the Server

```bash
./media-server        # or .\media-server.exe on Windows
```

The server listens on `http://localhost:8090`. On Windows and macOS a system tray icon appears.

### Development Mode

For faster iteration:

```bash
go run .
```

Make renderer changes? Re-run `npm run build:web` from the repo root before reloading.

### Running Tests

```bash
# All tests
go test ./...

# Verbose
go test -v ./...

# Single package
go test -v ./media/...

# Or from the repo root
npm run test:server
```

### Project Layout

The actual layout is more complex than a typical Go HTTP project because the entry point is split across three build-tagged files (Windows tray, macOS tray, Linux headless):

```
media-server/
├── main.go                 # Windows entry point: HTTP server + system tray
├── main_darwin.go          # macOS entry point: HTTP server + system tray
├── main_linux.go           # Linux entry point: HTTP server (headless)
├── loki_api.go             # JSON REST API used by the React SPA
├── hls.go                  # HLS transcode/segment cache, /media/hls/* handlers
├── fsbrowser.go            # /api/fs/list filesystem browser (local + S3)
├── thumbnail.go            # On-demand image and video thumbnail generation
├── db_dsn.go               # SQLite connection string helpers
│
├── auth/                   # JWT + bcrypt user management
├── appconfig/              # Config file load/save, env-var overrides
├── deps/                   # On-demand dependency downloader (ffmpeg, yt-dlp, whisper, onnx)
├── downloads/              # Bulk install / progress tracking for deps
├── jobqueue/               # SQLite-backed job + workflow DAG engine
├── media/                  # Media table queries, search, random sampler
├── onnxtag/                # ONNX-based image tagging
├── platform/               # Per-OS path/process helpers
├── querylog/               # Slow-query logger
├── renderer/               # Go HTML templates + middleware (RolePublic / RoleAdmin)
│   └── templates/          # Templates for /jobs, /config, /editor, /login, etc.
├── runners/                # Worker pool that dispatches jobs to task fns
├── storage/                # Local + S3 storage registry, signed URLs, thumbnails
├── stream/                 # Server-Sent Events broker (/stream, /downloads/stream)
├── tasks/                  # Self-registering task implementations
│   ├── registry.go         # init() registers every built-in task
│   ├── autotag.go          # ONNX auto-tagging
│   ├── autotag_vision.go   # Ollama vision-based tagging
│   ├── metadata_ops.go     # Descriptions + transcripts + hashes + dimensions
│   ├── ffmpeg*.go          # 16 ffmpeg preset variants + custom passthrough
│   ├── hls.go              # HLS transcode task
│   ├── ingest_*.go         # Local / YouTube / gallery / Discord ingestion
│   ├── lora_dataset.go     # LoRA training dataset builder
│   ├── media_*.go          # ingest / move / metadata / cleanup / remove
│   ├── save.go             # Save File task
│   └── ...
│
├── loki-static/            # React renderer bundle (embedded at compile time)
├── chrome-extension/       # Chrome extension for sending URLs into the queue
├── firefox-extension/      # Firefox extension (same functionality)
├── cmd/                    # Side CLIs
│   ├── onnxtag/            # Standalone ONNX tagger
│   ├── dbcopy/             # Database copy utility
│   └── sbs/                # Side-by-side comparison tool
├── API_DOCUMENTATION.md    # Detailed API reference
└── README.md               # This file
```

Changes to HTTP handlers, startup, or tray integration usually need to be mirrored across `main.go`, `main_darwin.go`, and `main_linux.go`. They share helpers from `tasks`, `jobqueue`, `storage`, `runners`, and `stream`.

### Dependencies

The server ships with everything it needs to run out of the box. Three categories:

- **Bundled** — `ffmpeg`, `ffprobe`, `ffplay` (not on macOS), `exiftool`, `onnxtag`, `onnxruntime` live in `<install-root>/bin/` alongside the executable. Populated per-OS-arch by CI (`scripts/fetch-bundled-deps.sh` / `.ps1` driven by `scripts/bundled-versions.json` with pinned SHA-256s). On macOS the server strips `com.apple.quarantine` from each binary at boot so Gatekeeper doesn't kill it.
- **Optional** — `yt-dlp`, `gallery-dl`, and `ollama` are detected on `PATH`. The welcome wizard (`/onboarding`) shows the right install command per OS; the server never auto-installs them.
- **Models** — AI model files (currently the WD-EVA02 tagger) download on demand via the wizard or `POST /api/deps/models/{id}/download`. Each model has a manifest entry with URLs and SHA-256 checksums. Downloads are atomic (`.partial` → rename), resumable, and verified before the model is marked installed.

Inspect runtime state via `GET /api/deps/status` or the wizard at `/settings/dependencies`. The welcome wizard appears once on first run and can be reopened any time.

---

## Configuration

Configuration file location:

- **Windows:** `%APPDATA%\Lowkey Media Viewer\config.json`
- **macOS:** `~/Library/Application Support/Lowkey Media Viewer/config.json`
- **Linux:** `~/.config/lowkeymediaviewer/config.json`

Most settings can also be set via environment variables (see the Docker section above). Env vars take precedence over the config file.

### Example config.json

```json
{
  "dbPath": "C:\\path\\to\\database.db",
  "jwtSecret": "auto-generated-on-first-run",

  "ollamaBaseUrl": "http://localhost:11434",
  "ollamaModel": "llama3.2-vision",
  "describePrompt": "Please describe this image...",
  "autotagPrompt": "Please analyze this image and select tags...",

  "onnxTagger": {
    "modelPath": "C:\\path\\to\\model.onnx",
    "labelsPath": "C:\\path\\to\\selected_tags.csv",
    "configPath": "C:\\path\\to\\config.json",
    "ortSharedLibraryPath": "C:\\path\\to\\onnxruntime.dll",
    "generalThreshold": 0.35,
    "characterThreshold": 0.85
  },

  "fasterWhisperPath": "C:\\path\\to\\faster-whisper-xxl.exe",

  "storageRoots": [
    { "type": "local", "path": "C:\\Media\\Photos", "label": "Photos" },
    { "type": "local", "path": "C:\\Media\\Videos", "label": "Videos" }
  ]
}
```

### ONNX Tagger Setup

1. Download model files from [HuggingFace](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/tree/main):
   - `model.onnx` — the neural network model
   - `selected_tags.csv` — tag labels
   - `config.json` — model configuration (optional)

2. Download [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases):
   - Windows: `onnxruntime-win-x64-*.zip` → extract `onnxruntime.dll` from `lib/`
   - macOS: `onnxruntime-osx-arm64-*.tgz` → extract `libonnxruntime.dylib`
   - Linux: `onnxruntime-linux-x64-*.tgz` → extract `libonnxruntime.so`

3. Configure the paths in the Config tab of the web UI.

---

## Usage

Once the server is running, the Lowkey Media Viewer detects it automatically and can dispatch jobs to it. Bulk jobs against any search result are also creatable from the Media Browser tab.

### Web Interface

Access the web UI at: <http://localhost:8090>

- **Home / Tasks** — quick task creation
- **Jobs** — view, cancel, copy, and remove jobs; real-time updates via SSE
- **Workflows** — create reusable DAGs; visual editor at `/editor`
- **Media** — browse, search, preview, and tag media; bulk-job action on results
- **Swipe** — paginated random-sample view (`/swipe`)
- **Config** — server settings, model paths, storage roots, user management
- **Dependencies** — view bundled-binary status, install hints for optional tools, and download/delete AI models
- **Stats** — database statistics

### System Tray

Right-click the tray icon (Windows/macOS) for:

- **Open Web UI** — opens <http://localhost:8090> in your default browser
- **Quit** — shutdown the server

Linux builds run headless.

---

## Available Tasks

Tasks register themselves in `tasks/registry.go`'s `init()`. The current catalog:

| Task ID                      | Name                       | Description                                              |
| ---------------------------- | -------------------------- | -------------------------------------------------------- |
| `wait`                       | Wait                       | Test task that sleeps 5 seconds                          |
| `ingest`                     | Ingest Media Files         | Dispatches by URL: local path, YouTube, gallery, Discord |
| `metadata`                   | Generate Metadata          | Descriptions, transcripts (Whisper), hashes, dimensions  |
| `autotag`                    | Auto Tag (ONNX)            | ML image tagging against the existing tag set            |
| `hls`                        | HLS Transcode              | Build the 480p / 720p / 1080p HLS ladder for a video     |
| `move`                       | Move Media Files           | Move files on disk and update DB paths                   |
| `remove`                     | Remove Media               | Delete entries from the database                         |
| `cleanup`                    | CleanUp                    | Remove orphaned database entries                         |
| `save`                       | Save File                  | Copy/persist a file with metadata                        |
| `lora-dataset`               | Create LoRA Dataset        | Assemble a captioned image dataset                       |
| `ffmpeg`                     | ffmpeg                     | Raw ffmpeg passthrough with custom args                  |
| `ffmpeg-scale`               | FFmpeg Scale               |                                                          |
| `ffmpeg-convert`             | FFmpeg Convert             |                                                          |
| `ffmpeg-extract-audio`       | FFmpeg Extract Audio       |                                                          |
| `ffmpeg-extract-audio-clip`  | FFmpeg Extract Audio Clip  |                                                          |
| `ffmpeg-screenshot`          | FFmpeg Screenshot          |                                                          |
| `ffmpeg-thumbnail`           | FFmpeg Thumbnail           |                                                          |
| `ffmpeg-reverse`             | FFmpeg Reverse             |                                                          |
| `ffmpeg-speed`               | FFmpeg Speed               |                                                          |
| `ffmpeg-grayscale`           | FFmpeg Grayscale           |                                                          |
| `ffmpeg-blur`                | FFmpeg Blur                |                                                          |
| `ffmpeg-resize`              | FFmpeg Resize              |                                                          |
| `ffmpeg-crop`                | FFmpeg Crop                |                                                          |
| `ffmpeg-rotate`              | FFmpeg Rotate              |                                                          |
| `ffmpeg-caption`             | FFmpeg Caption             |                                                          |
| `ffmpeg-thumbsheet`          | FFmpeg Thumbnail Sheet     |                                                          |

To add a new task: implement a `TaskFn` (`func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error`) and add a `RegisterTask(...)` line in `tasks/registry.go`. Tasks can register output files via `RegisterOutputFile` so downstream workflow steps can chain off them.

---

## API Documentation

See [API_DOCUMENTATION.md](./API_DOCUMENTATION.md) for the detailed HTTP API reference including:

- Endpoint specifications (REST + SSE)
- Request/response formats
- Auth flow and Bearer token usage
- Example curl commands

---

## Browser Extensions

Both a [Chrome](./chrome-extension/README.md) and [Firefox](./firefox-extension/README.md) extension are bundled. They let you create jobs (yt-dlp, gallery-dl, ffmpeg, ingest, etc.) from the page you're currently viewing, with real-time job status via SSE. Auth is supported via the same `/auth/login` endpoint.

---

## Troubleshooting

### Locked out / forgot password

Stop the server, delete the offending user from SQLite:

```bash
sqlite3 /path/to/media.db "DELETE FROM users;"
```

Restart the server. The default `admin` / `admin` will be re-created and you'll be sent through the setup wizard again.

### Port 8090 already in use

Bind address is currently hardcoded to `:8090` in the entry-point files. To run on a different port, edit `main.go` / `main_darwin.go` / `main_linux.go` and rebuild.

### Database connection errors

1. Ensure the database path in `config.json` (or `LOWKEY_DB_PATH`) exists and is writable.
2. Check file permissions on the database file.
3. Ensure no other process has the database open in WAL-incompatible exclusive mode.

### ONNX Tagger not working

1. Verify all model files (`model.onnx`, `selected_tags.csv`, optional `config.json`) are present.
2. Verify the ONNX Runtime shared library path is correct for your OS.
3. Make sure the model opset is ≥ 11.
4. On Windows, check Event Viewer for native crash logs.

### System tray icon not appearing (Windows)

1. Check the Windows notification area settings for hidden icons.
2. Restart Windows Explorer if necessary.

### Linux-specific issues

**Permission denied when running:**

```bash
chmod +x media-server-linux-amd64
./media-server-linux-amd64
```

**Missing shared libraries (only relevant if you point ONNX at a system install):**

```bash
export LD_LIBRARY_PATH=/path/to/libs:$LD_LIBRARY_PATH
./media-server-linux-amd64
```

---

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes; keep handler logic mirrored across `main.go`, `main_darwin.go`, and `main_linux.go`.
4. Run tests: `go test ./...` (and `npm run test:server` from the repo root).
5. Submit a pull request.

---

## License

See [LICENSE](../LICENSE) for details.
