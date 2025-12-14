# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Shrike Media Server is a Windows-based media processing server written in Go that manages long-running tasks through a job queue system. It provides a web UI, real-time updates via Server-Sent Events (SSE), and integrates with external tools like FFmpeg, yt-dlp, gallery-dl, and Ollama for AI-powered media tagging and descriptions.

**Key characteristics:**
- Windows-only (system tray integration, Windows-specific paths)
- No authentication - intended for local/trusted networks
- Binds to `localhost:8090` by default
- Embeds binaries (FFmpeg, yt-dlp, etc.) in the executable

## Build and Development Commands

### Building
```powershell
# Standard build
go build -o shrike.exe .

# Optimized build (smaller binary, no debug symbols)
go build -ldflags="-s -w" -o shrike.exe .
```

### Running
```powershell
# Run directly
.\shrike.exe

# Development mode (faster iteration)
go run .
```

### Testing
```powershell
# Run all tests
go test ./...

# Verbose output
go test -v ./...

# Test specific package
go test -v ./media/...
```

### Dependencies
```powershell
# Download dependencies
go mod download

# Verify dependencies
go mod verify
```

## Architecture

### Core Components

**Job Queue System** (`jobqueue/`):
- Thread-safe queue with SQLite persistence
- Jobs have dependencies (directed acyclic graph support)
- Job states: Pending → InProgress → Completed/Cancelled/Error
- Jobs resume as Pending if server restarts during execution
- Uses Go contexts for cancellation
- Broadcasts updates via SSE for real-time UI updates

**Task Registry** (`tasks/`):
- Tasks are registered functions that process jobs
- Each task has an ID (e.g., "ffmpeg", "ingest") and a name
- Task functions receive: `*jobqueue.Job`, `*jobqueue.Queue`, `*sync.Mutex`
- Tasks registered in `tasks/registry.go` init()
- Available tasks: wait, gallery-dl, yt-dlp, ffmpeg, ingest, metadata, move, autotag, remove, cleanup, lora-dataset

**Runners** (`runners/`):
- Worker pool that claims and executes jobs
- Configurable concurrency (default: 1 worker)
- Listens to queue signal channel for new jobs
- Automatically picks up next job when one completes
- Uses goroutines for parallel execution

**Media Database** (`media/`):
- Queries against Lowkey Media Viewer SQLite database
- Main tables: `media`, `media_tag_by_category`
- Provides search with filters: tag:label, category:label, path:pattern, etc.
- Supports pagination and infinite scroll
- Functions for metadata operations (hash, dimensions, descriptions)

**Server-Sent Events** (`stream/`):
- Broadcasts job updates to connected web clients
- Event types: create, update, delete, stdout-{job-id}
- Connection limits: 1000 max, 50 message buffer per client
- Keep-alive every 30s, cleanup every 60s

**Template Renderer** (`renderer/`):
- Go html/template system
- Templates in `renderer/templates/*.go.html`
- HTMX integration for dynamic UI updates
- Middleware for error handling

**Configuration** (`appconfig/`):
- Stored at `%APPDATA%\Lowkey Media Viewer\config.json`
- Contains database path, Ollama settings, ONNX tagger config, etc.
- Can be updated via web UI `/config` endpoint

**Embedded Executables** (`embedexec/`):
- Binaries embedded at build time from `embedexec/bin/`
- Extracted to `%ProgramData%\Shrike\tmp\` at runtime
- Includes: ffmpeg, ffprobe, ffplay, yt-dlp, gallery-dl, exiftool, faster-whisper-xxl, dedupe

### Key Architectural Patterns

**Database Switching**:
- Users can switch databases at runtime via `/config` endpoint
- When database changes: new DB opened, new queue initialized, old DB closed
- Handled by `switchDatabase()` in main.go

**Job Lifecycle**:
1. Job created via API → assigned UUID → added to queue → persisted to DB
2. Runner claims job when dependencies satisfied and worker available
3. Task function executes, streams stdout via `PushJobStdout()`
4. Job marked Completed/Error/Cancelled → final state saved to DB
5. SSE broadcasts all state changes to connected clients

**Template + HTMX Pattern**:
- Server renders HTML fragments (e.g., job rows)
- HTMX swaps fragments into DOM on SSE events
- Combines SSR with real-time interactivity without full SPA

**Context Cancellation**:
- Each job has a context and cancel function
- Cancelling a job calls `job.Cancel()` which propagates to running task
- Tasks must respect context cancellation (check `j.Ctx.Done()`)

## Database Schema Expectations

Shrike expects to connect to a Lowkey Media Viewer database with these tables:

**media**:
- `path` (primary key): absolute file path
- `description`: generated text description
- `hash`: content hash (e.g., SHA-256)
- `size`: file size in bytes
- `width`, `height`: image/video dimensions

**media_tag_by_category**:
- `media_path`: references media.path
- `tag_label`: tag name
- `category_label`: category name

Indexes are created automatically by `ensureIndexes()` in main.go for performance.

## Configuration

Configuration file location: `%APPDATA%\Lowkey Media Viewer\config.json`

Key settings:
- `dbPath`: Path to SQLite database (required)
- `downloadPath`: Default download location for media ingestion
- `ollamaBaseUrl`: Ollama API endpoint (default: http://localhost:11434)
- `ollamaModel`: Model for vision tasks (default: llama3.2-vision)
- `onnxTagger`: ONNX model configuration for image auto-tagging
- `fasterWhisperPath`: Path to Faster Whisper executable for transcription

## Task Development

To add a new task:

1. Create task function in `tasks/` with signature:
   ```go
   func myTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error
   ```

2. Register in `tasks/registry.go` init():
   ```go
   RegisterTask("my-task-id", "My Task Name", myTask)
   ```

3. Task responsibilities:
   - Stream output via `q.PushJobStdout(j.ID, "message")`
   - Mark completion via `q.CompleteJob(j.ID)` or `q.ErrorJob(j.ID)`
   - Respect context cancellation: check `j.Ctx.Done()`
   - Parse arguments from `j.Arguments` and `j.Input`

4. Job input parsing:
   - `j.Command`: task ID (e.g., "ffmpeg")
   - `j.Arguments`: flags/options (e.g., ["-i", "input.mp4"])
   - `j.Input`: main input (file path, URL, or newline-delimited list)

## HTTP API

See `API_DOCUMENTATION.md` for full details. Key endpoints:

- `POST /create` - Create job (JSON: `{"input": "command args..."}`)
- `GET /stream` - SSE connection for real-time updates
- `GET /media/api` - Paginated media items (supports search queries)
- `GET /media/file?path=...` - Serve media files
- `POST /job/{id}/cancel` - Cancel running job
- `POST /jobs/clear` - Remove all non-running jobs
- `GET /health` - Server health and statistics

## Common Gotchas

1. **Windows-only**: Code uses Windows system tray, path separators, and assumes Windows paths. Don't introduce Unix-specific assumptions.

2. **Database locking**: SQLite can lock on concurrent writes. The queue's mutex protects job updates, but be mindful of long-running DB operations.

3. **Context cancellation**: Tasks must check `j.Ctx.Done()` regularly, especially in loops. Cancelled jobs should exit gracefully.

4. **Stdout streaming**: Use `q.PushJobStdout()` for progress updates. This both stores output and broadcasts via SSE.

5. **Job state management**: Always call `q.CompleteJob()` or `q.ErrorJob()` when task finishes. The runner has fallback logic but explicit is better.

6. **Embedded binary extraction**: Binaries are extracted once at startup. If you modify `embedexec/bin/`, rebuild the executable.

7. **Template rendering**: Templates execute on every request. Heavy computation should be in handlers, not templates.

8. **CORS**: `/health` endpoint has permissive CORS for external monitoring. Other endpoints do not.

9. **File path encoding**: Media paths may contain Unicode. Use `url.PathUnescape()` not `url.QueryUnescape()` for path parameters to avoid issues with '+' in filenames.

10. **Port 8090**: Hardcoded in main.go. If changing, update system tray URL and documentation.

## Testing Considerations

- Most packages lack comprehensive tests (only `media/media_test.go` exists)
- When adding tests, use temporary databases or mock the `*sql.DB`
- Test job queue with in-memory queue (no database)
- SSE testing: check message formats and connection limits
- Test task cancellation by triggering `job.Cancel()`

## External Dependencies

**Required at runtime** (optional, configured in settings):
- Ollama for vision-based image descriptions
- ONNX Runtime DLL for ML-based image tagging
- Faster Whisper for video transcription

**Embedded** (included in binary):
- FFmpeg, FFprobe, FFplay
- yt-dlp
- gallery-dl
- ExifTool
- dedupe

## Debug and Logs

- Server logs to stdout (viewable when running from terminal)
- Job stdout stored in `job.Stdout` array and database
- Windows Event Viewer may contain crash logs for ONNX Runtime issues
- SSE connection stats available via `/health` endpoint
