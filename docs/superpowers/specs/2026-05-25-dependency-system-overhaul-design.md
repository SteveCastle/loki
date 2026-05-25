# Dependency & Setup System Overhaul — Design

**Date:** 2026-05-25
**Scope:** Lowkey Media Server (`media-server/`) only. The Electron desktop app at the repo root bundles its own binaries under `src/main/resources/bin/<platform>/` and is unaffected by this work.

## Goal

Replace the current "download everything on demand" setup system with a clear three-category model:

1. **Bundled binaries** — `ffmpeg`, `ffprobe`, `ffplay`, `exiftool`, `onnxtag`, `onnxruntime` ship with the server, downloaded into the release artifact by CI per platform. The runtime never downloads them.
2. **Optional tools** — `yt-dlp`, `gallery-dl`, `ollama` are detected on `PATH`. If missing, the UI shows copy-paste install instructions per OS. The runtime never installs them.
3. **AI models** — the only category that downloads at runtime. A robust downloader with manifest, SHA-256 checksums, atomic install, and resume support.

A one-time, skippable welcome wizard surfaces this information on first run. The wizard is also reachable from settings at any time. There is no forced `/setup` redirect.

## Non-goals

- Changes to the Electron desktop app's bundled binaries.
- Changes to `jobqueue`, `tasks`, `workflows`, or the runners that *use* these dependencies — only the resolution/install layer changes. Callers continue to ask for a path; the path-resolver implementation changes underneath.
- Auto-update of bundled binaries between server releases. A new ffmpeg requires a new server release.
- Auto-install of yt-dlp / gallery-dl / ollama. The user installs these themselves.
- A user-editable model registry (e.g., "paste a HuggingFace URL"). The model manifest is compile-time only in this phase.
- Replacement of the `main.go` / `main_darwin.go` / `main_linux.go` build-tag split. We delete code from each but do not unify them.
- ARM Linux support for bundled binaries — Linux release is x86_64 only in this phase (matches current state).

## Architecture

```
media-server/deps/
├── bundled/      stable native binaries shipped with the server
│   ├── bundled.go         exported: Resolve(id) (string, error); IDs()
│   ├── manifest.go        var Manifest = []Bundled{...}  (compile-time)
│   ├── verify.go          VerifyAll() — boot-time presence check, logs missing
│   └── quarantine_darwin.go / quarantine_other.go
├── optional/     PATH-detected user-installed tools
│   ├── optional.go        Detect(id) (Status, error); IDs()
│   ├── manifest.go        var Manifest = []Optional{...}
│   └── hints.go           per-OS install instructions
├── models/       on-demand AI model downloader
│   ├── manifest.go        embedded JSON via //go:embed, parsed at init
│   ├── store.go           paths, atomic install (.partial → rename)
│   ├── downloader.go      resumable HTTP, retries, sha256 verify
│   ├── progress.go        in-memory progress + event channel
│   └── state.go           state.json — installed/partial/missing per model
└── status/       cross-category aggregation for the UI
    └── status.go          Snapshot() []DepStatus
```

**Rules:**

- `bundled`, `optional`, `models` do **not** import each other. `status` imports all three but only to read snapshots; it never triggers an install.
- No package uses `init()` to self-register. Each category exposes a `Manifest` slice you can grep for.
- No package writes shared on-disk state. Bundled writes nothing. Optional writes nothing. Only `models/state.go` writes (its own `state.json`).
- macOS-specific code lives in `_darwin.go` files within the relevant package, not in a new global platform split.

The old `deps/` files — `ffmpeg.go`, `whisper.go`, `onnx.go`, `dce.go`, `ytdlp.go`, `gallerydl.go`, `ollama.go`, `onnxtag.go`, `deps.go`, `metadata.go`, plus `downloads/manager.go`, `downloads/extract.go`, `downloads/http.go` — are deleted. The new code totals ~1500–1800 LOC versus ~4500 today.

## Bundled binaries

### What's bundled

| ID | What | Source URL pattern |
|----|------|---------------------|
| `ffmpeg` | ffmpeg executable | BtbN/FFmpeg-Builds (Win/Linux), evermeet.cx (macOS) |
| `ffprobe` | ffprobe executable | same archives as ffmpeg, except macOS uses separate evermeet.cx zip |
| `ffplay` | ffplay executable | from the BtbN/FFmpeg-Builds archive; not present on macOS evermeet build (acceptable — features that need ffplay log "unsupported on macOS") |
| `exiftool` | exiftool executable + lib dir | exiftool.org (Win: `exiftool-*.zip`; Linux/macOS: `Image-ExifTool-*.tar.gz`) |
| `onnxtag` | the project's own ONNX tagger CLI | `github.com/SteveCastle/loki/releases/download/onnxtag-v<ver>/...` |
| `onnxruntime` | shared library (`.dll` / `.so` / `.dylib`) | microsoft/onnxruntime releases, per-OS-arch |

### Where they live in the release

For all platforms, the server binary and its bundled deps live in the same install root:

```
<install-root>/
├── lowkeymediaserver(.exe)
└── bin/
    ├── ffmpeg(.exe)
    ├── ffprobe(.exe)
    ├── ffplay(.exe)     # not present on macOS
    ├── exiftool(.exe)
    ├── exiftool_files/  # exiftool's lib directory (Windows-style packaging)
    ├── onnxtag(.exe)
    └── onnxruntime.{dll,so,dylib}
```

`bundled.Resolve(id)` returns `filepath.Join(execDir, "bin", relPath)` where `execDir = filepath.Dir(os.Executable())`. We resolve `os.Executable()` once at init and cache it.

We do **not** use `//go:embed` for the binaries. They sit alongside the executable as plain files. Reasons:
- Single-file go binary is not a goal — the server is already a directory deployment.
- Faster builds (Go doesn't have to compress/embed ~120MB at every build).
- Easier to swap a binary for debugging without rebuilding.
- Docker users mount `bin/` from a separate stage; no extraction-on-first-launch races.

### How CI populates `bin/`

A new `media-server/scripts/fetch-bundled-deps.sh` (and `.ps1` for Windows runner) takes `GOOS` and `GOARCH` arguments, downloads each upstream archive, extracts the binaries we need, verifies a pinned SHA-256, and drops them into `media-server/bin/<goos>-<goarch>/`. The script is idempotent: skips downloads if files already exist and checksums match.

A new GitHub Actions workflow `release-server.yml` runs the fetch script for each target triple, then runs `go build`, then bundles the resulting `lowkeymediaserver(.exe)` + `bin/` directory into a release archive (zip on Windows, tar.gz elsewhere).

Pinned versions and SHA-256 hashes live in `media-server/scripts/bundled-versions.json`:

```json
{
  "ffmpeg": {
    "windows-amd64": { "version": "n7.1", "url": "...", "sha256": "...",
                       "extract": ["bin/ffmpeg.exe","bin/ffprobe.exe","bin/ffplay.exe"] },
    "linux-amd64":   { ... },
    "darwin-arm64":  { ... },
    "darwin-amd64":  { ... }
  },
  "exiftool":   { ... },
  "onnxtag":    { ... },
  "onnxruntime":{ ... }
}
```

The script reads this file and is the **only** place upstream URLs and versions live.

### Boot-time verification

`bundled.VerifyAll()` is called once during server startup, after the HTTP listener is bound but before the welcome wizard is served. For each entry in `Manifest`, it:

1. Checks the file exists at the resolved path.
2. On macOS: calls `removeQuarantine(path)` (no-op on other OSes) — runs `xattr -d com.apple.quarantine <path>` if the xattr is present. Failures are logged but not fatal.
3. Runs the binary with its `--version` flag (or equivalent) with a 5-second timeout. If it exits non-zero, the dep is marked "broken."

Result is cached in-memory and exposed via `status.Snapshot()`. A missing or broken bundled dep is **not fatal** — the server starts, the welcome wizard surfaces the problem, and the user sees actionable text ("ffmpeg failed to run — please reinstall the server"). The current behavior (silently using the wrong binary, then crashing in a task) is the bug we're fixing.

### macOS quarantine handling

`bundled/quarantine_darwin.go`:

```go
func removeQuarantine(path string) {
    // Best-effort: xattr returns an error if attribute is absent. We only log.
    _ = exec.Command("xattr", "-d", "com.apple.quarantine", path).Run()
}
```

Called for each bundled binary at boot. This is the single fix that addresses the user's reported macOS crash. The CI release tarball preserves xattrs when extracted via `tar -xzf`, but the quarantine attribute is added by macOS the moment the user expands a `.tar.gz` they downloaded from a browser. We strip it idempotently every boot.

## Optional tools

| ID | What |
|----|------|
| `yt-dlp` | Video downloader; updates weekly to track site changes |
| `gallery-dl` | Image gallery downloader; similar update cadence |
| `ollama` | Local LLM server; heavyweight, runs as its own service |

`optional.Detect(id)` returns:

```go
type Status struct {
    Installed bool
    Path      string  // empty if not installed
    Version   string  // best-effort; empty if probe failed
    Hint      InstallHint
}

type InstallHint struct {
    Description string   // "yt-dlp lets the server import from YouTube, Twitch, ..."
    Commands    []OSCmd  // per-OS install commands
    DocsURL     string
}
```

Detection runs `exec.LookPath(name)` then probes `<path> --version` with a 3-second timeout. No caching beyond the request — these change when the user runs `brew upgrade`.

Install hints are static data in `optional/hints.go`. For example, `yt-dlp`:

```
macOS:   brew install yt-dlp
Windows: winget install yt-dlp
Linux:   pipx install yt-dlp
Docs:    https://github.com/yt-dlp/yt-dlp#installation
```

There is no "install" endpoint. The UI shows the commands; the user runs them.

## Models

### Manifest format

`models/manifest.json` is embedded via `//go:embed`. The Go side parses it once at init.

```json
{
  "schema_version": 1,
  "models": [
    {
      "id": "wd-eva02-large-tagger-v3",
      "name": "WD EVA02 Large Tagger v3",
      "description": "Image autotagging classifier (general + character tags).",
      "version": "1.0.0",
      "consumers": ["autotag"],
      "size_bytes": 1257385984,
      "files": [
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx",
          "rel_path": "model.onnx",
          "sha256": "PLACEHOLDER_TO_FILL_DURING_IMPLEMENTATION_FROM_UPSTREAM"
        },
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv",
          "rel_path": "selected_tags.csv",
          "sha256": "..."
        },
        {
          "url": "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/config.json",
          "rel_path": "config.json",
          "sha256": "..."
        }
      ]
    }
  ]
}
```

Models known to be in use today (from existing code): the WD-EVA02 autotagger and at least one Whisper model used by the transcribe task. The implementation phase enumerates the full set by grepping the current `deps/onnx.go` and `deps/whisper.go` and porting URLs.

For models without a publisher-supplied SHA-256, the implementation computes the SHA-256 once by downloading the file and pins it. We do not skip checksums.

### On-disk layout

```
<data-dir>/models/
└── <model-id>/
    ├── model.onnx         # final file (whatever rel_path says)
    ├── ...other files
    └── .meta.json         # { version, sha256_verified, installed_at }
```

`<data-dir>` is `%LOCALAPPDATA%\Lowkey Media Server` on Windows, `~/Library/Application Support/Lowkey Media Server` on macOS, `~/.local/share/lowkey-media-server` on Linux. (We adopt platform-correct user data dirs; the current Windows path is `%ProgramData%` which is wrong for a per-user app.)

A model is "installed" iff `.meta.json` exists, `version` matches the manifest, and every file in the manifest exists at the expected path. We do **not** re-verify SHA-256 on every status check; that happens at install time only. Re-verification on demand is exposed as `POST /api/deps/models/:id/verify`.

### Downloader

`models/downloader.go` does one install per model at a time (semaphore = 1 per model id; concurrent installs of *different* models are allowed). For each file in the model:

1. Stream-download to `<file>.partial` using `http.Get` with `Range: bytes=<existing-size>-` if `.partial` exists.
2. Stream the bytes through `sha256.New()` while writing.
3. On `io.Copy` success, verify hash. Mismatch → delete `.partial`, return error.
4. `os.Rename(.partial, final)`. (Same filesystem; atomic on POSIX and Windows.)

Retries: up to 3 attempts per file with exponential backoff (1s, 4s, 16s). Network errors trigger retry. HTTP 4xx (except 416) does not retry — that's a permanent manifest problem.

Cancellation: each install runs with a `context.Context` stored in the `progress.Tracker`. `POST /api/deps/models/:id/cancel` cancels it; partial files are deleted on cancel to keep state clean.

Failure modes are all surfaced. After any failure, `state.json` is updated to `{status: "failed", error: "<message>"}`. The UI displays the error inline.

### Progress

`models/progress.go` tracks per-install state:

```go
type ModelInstall struct {
    ID            string
    State         string  // "queued" | "downloading" | "verifying" | "installed" | "failed" | "cancelled"
    CurrentFile   string
    BytesDone     int64
    BytesTotal    int64
    Error         string
    StartedAt     time.Time
    UpdatedAt     time.Time
}
```

Updates are emitted on a buffered channel. The HTTP layer exposes Server-Sent Events at `/api/deps/models/progress` — subscribers get every update. No polling endpoint; the existing `/stream` SSE pattern in the codebase is the model.

### State

`models/state.go` persists per-model install state to `<data-dir>/models/state.json`. State is rebuilt from disk on startup by walking `<data-dir>/models/` and checking each model's `.meta.json` against the manifest. The persisted `state.json` is treated as a cache of derived state — if it's missing or corrupt, we rebuild from scratch without erroring.

All writes go through a single goroutine (channel-based serialization) so we never have concurrent-write JSON corruption — the bug class the current `metadata.go` is vulnerable to.

## Status aggregation

`status.Snapshot()` returns one slice that the UI iterates:

```go
type DepStatus struct {
    ID       string
    Category string   // "bundled" | "optional" | "model"
    Name     string
    State    string   // "ready" | "missing" | "broken" | "downloading" | "failed"
    Version  string
    SizeBytes int64    // for models, total; for others, 0
    Detail   any      // category-specific: install hints, progress, error message
}
```

Used by `GET /api/deps/status` (returns the snapshot) and by the welcome wizard / settings page.

## HTTP API

New endpoints, all under `/api/deps/`:

| Method | Path | Purpose |
|--------|------|---------|
| GET    | `/api/deps/status` | Snapshot of all deps across categories |
| GET    | `/api/deps/models/progress` | SSE stream of model install events |
| POST   | `/api/deps/models/:id/download` | Start install (idempotent — already-installed returns 200; already-downloading returns 202) |
| POST   | `/api/deps/models/:id/cancel` | Cancel in-flight install |
| POST   | `/api/deps/models/:id/verify` | Re-verify SHA-256 of installed files |
| DELETE | `/api/deps/models/:id` | Uninstall (delete model directory) |

Removed endpoints: `/dependencies`, `/dependencies/download`, `/setup`, and the SSE `/stream` handlers specific to legacy dep state. The generic `/stream` (used for jobqueue updates) is unaffected.

The handlers live in `handlers/deps_handlers.go` and are registered from a new shared `routes_deps.go` that all three `main_*.go` files call. We do not duplicate handler registration across the platform files.

## Onboarding wizard

A new React route in the SPA at `/onboarding`. It renders once when:

- `GET /api/deps/onboarding-state` returns `{ shown: false }` AND
- The user has not dismissed it.

State is persisted in `<data-dir>/onboarding.json`: `{ shown: true, dismissed_at: "..." }`. From settings, a "Show welcome again" button clears `shown` so the user can revisit.

The wizard has three steps, all skippable:

1. **Bundled status** — "Here's what's already installed and ready." Shows a green-check list. If any are broken/missing, shows red with the path and a "Reinstall the server" hint.
2. **Optional tools** — "These features need extra software you install yourself." Shows yt-dlp / gallery-dl / ollama with status badges. Each row has a "Show install command" disclosure with per-OS commands and a docs link.
3. **Models** — "Download the AI models you want to use." Shows model cards with size, description, "Download" button. Buttons trigger `POST /api/deps/models/:id/download`. Progress shown inline from the SSE stream. The user can close the wizard while downloads continue in the background.

A persistent "Skip" / "I'll do this later" button is on every step. Closing the wizard sets `shown: true` regardless of step.

The same component is rendered (without the "first run" auto-redirect) at `/settings/dependencies` for revisits.

## Removed code & behaviors

- `main_*.go`: delete the `setupMode` flag, the `CheckAnyMissing`-driven middleware, and the `/setup` redirect. The server boots straight to `/` (which redirects to `/onboarding` on first run via the SPA route, not via server-side redirect).
- `main_*.go`: delete the `/dependencies` and `/dependencies/download` handlers.
- `deps/`: delete `ffmpeg.go`, `whisper.go`, `onnx.go`, `dce.go`, `ytdlp.go`, `gallerydl.go`, `ollama.go`, `onnxtag.go`, `deps.go`, `metadata.go`, `paths.go`, `exec.go`, `exec_*.go`. (Replaced by the new packages.)
- `downloads/`: delete the entire package — `manager.go`, `extract.go`, `http.go`. Replaced by `deps/models/downloader.go`. Anything else in the codebase that imports `downloads/` (only the legacy dep installers, verified during implementation) is rewired.
- `dependencies.json` metadata file and the legacy on-disk install locations are **not** migrated. The new code uses different paths and a different on-disk model layout, so we treat any previous installs as gone. On first boot, if the legacy file exists, we rename it to `dependencies.json.bak` (so the user can recover paths from it if needed) and log a one-line notice. Users who had downloaded models before will need to re-download via the new wizard. This is consistent with the "complete overhaul" goal and avoids carrying brittle migration code forever.

## Caller migration

Tasks/runners currently call helpers like `deps.GetFFmpegPath()`, `deps.GetOnnxModelPath()`, etc. Two replacement helpers are added at the package boundary:

- `deps.MustBundled(id string) string` — wraps `bundled.Resolve(id)`; panics if missing (callers can assume bundled deps are present after `VerifyAll` at boot, and panicking is correct for a misconfigured release).
- `deps.ModelPath(id, relPath string) (string, error)` — wraps `models.Path(id, relPath)`; returns a typed `ErrModelNotInstalled` so task code can surface a friendly "download this model first" error to the UI instead of crashing.

Optional tools are looked up at task execution time via `optional.Detect(id)`; tasks that need them must check status and return `ErrOptionalToolMissing` with the hint payload if absent.

All `GetXxxPath` and `GetXxxDownloadURL` call sites get rewritten as part of this work. Approximate scope: ~40 call sites across `tasks/`, `loki_api.go`, and the three `main_*.go` files, identified during implementation via grep.

## Testing strategy

- `bundled/`: table-driven tests for `Resolve` with a fake `execDir` and table of present/missing files. `removeQuarantine` is tested via build tag on a darwin runner only (CI must include a macOS job — already exists per `release-onnxtag.yml`).
- `optional/`: tests use a temp dir prepended to `PATH` containing fake binaries that print known version strings. Verifies detection + version parsing.
- `models/`:
  - `downloader_test.go` uses `httptest.Server` to serve a fixture file with a known SHA-256. Covers: happy path, resume after partial download, checksum mismatch deletes `.partial`, retry on transient 500, no retry on 404, cancellation deletes `.partial`.
  - `state_test.go` covers: rebuild from disk when `state.json` missing, rebuild detects partial install (some files missing), concurrent updates serialized correctly.
  - `manifest_test.go` validates the embedded JSON parses and every URL is well-formed.
- `status/`: tests assemble fake snapshots from each category and verify the aggregation.
- `handlers/deps_handlers_test.go` uses `httptest.NewRecorder` and a fake `status.Snapshotter` injected via interface.

Integration smoke: a new `media-server/scripts/smoke-bundled.sh` runs after CI builds the release archive, extracts it to a temp dir, starts the server, hits `/api/deps/status`, and asserts every bundled dep reports `state: ready`. This catches "we forgot to add ffmpeg to bundled-versions.json" before the release ships.

## Error handling

| Failure | Behavior |
|---------|----------|
| Bundled dep file missing at boot | Logged with path; status reports `missing`; server keeps running; tasks needing it fail with a typed error the UI converts to "Reinstall the server" |
| Bundled dep present but `--version` non-zero | Status reports `broken` with stderr captured (first 200 chars); same surfacing as missing |
| Optional tool not on PATH | Normal state; UI shows install hints |
| Optional tool on PATH but `--version` fails | Status reports `installed: true, version: ""`; tasks may still work; we don't preempt |
| Model download network error | Retry up to 3x; on final failure, `state.json` records error; `.partial` kept so user can retry resume-style |
| Model download checksum mismatch | `.partial` deleted; state records "checksum mismatch — file may be corrupt or manifest may be stale"; no auto-retry |
| Model download cancelled | `.partial` deleted; state records `cancelled` |
| State file corrupt | Logged; rebuilt from filesystem walk; no user-visible error |
| First-run migration of legacy `dependencies.json` fails | Logged; we proceed with empty state; legacy file is renamed to `dependencies.json.bak` rather than deleted |

## Out-of-scope follow-ups

These are noted for later but explicitly not built in this phase:

- A user-editable model registry (paste a HuggingFace URL).
- Mirror/CDN selection for users in regions where HuggingFace is slow.
- Per-model GPU/CPU variant selection.
- Notifying the user when a new server release ships with a newer bundled ffmpeg.
- ARM Linux release builds.
- A "Doctor" command that runs `--version` on every bundled binary and prints a report (the wizard already does this via the UI; CLI version is nice-to-have).
