# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo Shape

This repo contains **two distinct products** that share a frontend:

1. **Lowkey Media Viewer** — an Electron desktop app (TypeScript + React + XState) at repo root (`src/`).
2. **Lowkey Media Server** (`media-server/`, Go module `github.com/stevecastle/shrike`) — a standalone HTTP server with a job queue and web UI. It embeds the same React SPA as the desktop renderer and serves it at `/`.

The React renderer under `src/renderer/` targets **both** environments. `src/renderer/platform.ts` is the abstraction layer: `isElectron` detection, `capabilities` flags, and routing API calls to either Electron IPC or HTTP fetch. When editing renderer code, assume it may run in either environment unless the file is clearly Electron-only.

## Commands

### Electron app (root)
- `yarn dev` / `npm start` — run in dev (webpack-dev-server + electronmon)
- `yarn package` — build distributable binary for the current OS
- `yarn lint` — ESLint (`.js,.jsx,.ts,.tsx`)
- `npm test` — **builds first, then runs Jest**. To run a single test quickly: `npm run build` once, then `npx jest <pattern>`. Don't skip the build — `setupFiles` includes `check-build-exists.ts`.
- Tests live in `src/__tests__/` and use jsdom + ts-jest.

### Go media server
- `npm run build:web` — builds the renderer and copies output into `media-server/loki-static/` (where Go embeds it).
- `npm run build:server` — runs `build:web` + `go build` (via `scripts/build-server.js`). This also kills any running `media-server` process first.
- `npm run test:server` — `cd media-server && go test ./...`
- From `media-server/`: `go test ./...`, `go build -o media-server.exe .`, `go run .`

**Important:** the Go server embeds `loki-static/**` at compile time. After renderer changes, you must run `build:web` (or `build:server`) before the Go binary will serve the new assets.

## Architecture Notes

### Go server platform split
`media-server/main.go`, `main_darwin.go`, `main_linux.go` are **separate files gated by build tags** (`//go:build windows` etc.). Changes that touch HTTP handlers, startup, or tray integration often need to be mirrored across all three. They share helpers from packages like `tasks`, `jobqueue`, `storage`, `runners`, `stream`.

### Task registry
All server tasks register themselves in `media-server/tasks/registry.go`'s `init()` via `RegisterTask(id, name, options, fn)`. To add a new task: implement a `TaskFn` (signature `func(j *jobqueue.Job, q *jobqueue.Queue, r *sync.Mutex) error`), add a line in `init()`. Tasks can register output files via `RegisterOutputFile` for downstream workflow steps.

### Workflow DAG
`jobqueue/workflows.go` persists saved workflows (`workflows` table in SQLite). Jobs can chain via a DAG; `ErrorJob` cancels pending dependents (see recent commits for the contract). When touching job state transitions, check `jobqueue_test.go` and `workflows_test.go`.

### Storage abstraction
`storage/registry.go` manages multiple storage roots (local paths and S3-compatible buckets) configured via `LOWKEY_ROOT_<N>` env vars or the `LOWKEY_ROOTS` JSON array. Tasks needing uploads use `tasks.SetStorageRegistry` wired at startup. Don't hardcode paths — go through the registry.

### Two "renderers"
- `src/renderer/` — React SPA (Electron renderer process + web UI).
- `media-server/renderer/` — Go HTML templates (server-rendered admin pages for Jobs, Config, etc.). These are **different**; the Go templates are a separate UI used when the SPA isn't appropriate.

### Frontend state
`src/renderer/state.tsx` hosts the primary XState machine (`@xstate/react`). Renderer components consume it via context. Session persistence flows through `hooks/useSessionStore.ts` → Electron's `electron-store` or HTTP in web mode.

### Pre-existing type errors (don't chase)
`webpack/types.d.ts` symbol errors, `src/main/media.ts` tuple naming, `layout.tsx` RefObject null mismatches, and various metadata type mismatches are known issues. TypeScript 5.0.2 is pinned and has no bundled WebGPU types — use numeric GPU buffer constants or `any` casts.

### External binaries
The Electron app expects `ffmpeg`, `ffprobe`, `ffplay`, and `exiftool` under `src/main/resources/bin/<platform>/`. The Go server manages its own copies under `%ProgramData%\Lowkey Media Server\deps\` (Windows) or `/var/lib/lowkeymediaserver/deps/` (Linux), downloaded on demand via the Dependencies web UI, with a fallback to system PATH.
