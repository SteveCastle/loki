# Web Mode Filesystem Browser

**Date:** 2026-03-20
**Status:** Approved

## Problem

The Electron app supports loading media from the filesystem via native OS dialogs (`select-directory` → `load-files`). In web mode (served by media-server), these channels are stubbed and `capabilities.fileSystemAccess` is `false`, so the state machine skips filesystem loading entirely and only supports DB-based browsing.

## Solution

Add a server-side file browser modal to web mode, backed by two new Go API endpoints, with configurable root path constraints.

## Approach: Single endpoint per concern, client-side navigation

Two new endpoints handle browsing and scanning separately. The platform layer maps existing IPC channels to these endpoints so the XState state machine requires no code changes (one behavioral change documented below).

## Config

Add `RootPaths []string` to the existing `Config` struct in `appconfig/config.go`.

- Default: empty (unrestricted filesystem access)
- When set: only directories under the configured roots can be browsed or scanned
- Editable via the existing config page at `POST /config` (form-based HTML endpoint)

## Go API Endpoints

### `POST /api/fs/list` — Directory browser

- **Request:** `{ "path": string }` (empty string = show roots)
- **Response:** `{ "entries": [{ "name": string, "path": string, "isDir": bool, "mtimeMs": number }], "parent": string|null, "roots": string[] }`
- When `path` is empty and roots are configured: returns root dirs as entries
- When `path` is empty and no roots: returns OS filesystem roots (`/` on Unix, drive letters on Windows — enumerate drives by checking A-Z existence)
- Directories always shown; files only shown if they match the media extension regex:
  `jpg|jpeg|jfif|webp|png|webm|mp4|mov|mpeg|gif|mkv|m4v|mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape`
- `parent` is the parent directory path (for "up" navigation), or `null` when at a root boundary
- The media extension regex must be defined once as a package-level variable to avoid drift with `/api/fs/scan`

### `POST /api/fs/scan` — Load directory as library

- **Request:** `{ "path": string, "recursive": bool }`
- **Response:** `{ "library": [{ "path": string, "mtimeMs": number }], "cursor": 0 }`
- Walks the directory, filters to media files using the shared extension regex
- Inserts discovered files into the SQLite media table (same as Electron's `insertBulkMedia`)
- Returns the full file list in one response (no streaming for v1)
- Returns files in filesystem walk order (unsorted); the client-side `onDone` handler re-sorts by the user's chosen sort order and recalculates cursor position, so `cursor: 0` is always correct here
- Same root path jail enforcement as `/api/fs/list`
- For recursive walks, use `filepath.WalkDir` and skip symlinked directories to prevent symlink loops
- **v1 limitation:** No size cap on response. Very large directories (100k+ files) may be slow.

## Security

- **Path traversal prevention:** `filepath.Clean` the requested path, then `filepath.EvalSymlinks` to resolve symlinks, then verify the resolved path is under a configured root using `filepath.Rel`. If `Rel` returns a path starting with `..`, reject with 403.
- **Symlink loop protection:** During recursive walks in `/api/fs/scan`, skip symlinked directories entirely to prevent infinite loops.
- **Empty roots = unrestricted:** When `rootPaths` is empty, all paths are allowed. Intentional for personal/local use.
- **Auth:** Both endpoints use existing auth middleware (cookie-based JWT).

## Platform Layer Changes (`src/renderer/platform.ts`)

- Remove `select-directory` and `load-files` from `stubbedChannels`
- `select-directory`: opens the file browser modal, returns a Promise that resolves with the chosen directory path **as a bare string** (not a wrapped object). This matches the `setPath` action's expectation that `event.data` is a string. On cancel, the promise rejects.
- `load-files`: maps to `POST /api/fs/scan` with `argsToBody: (args) => ({ path: args[0], recursive: args[2] })`. Note: `args[1]` (sortOrder) and `args[3]` (options) are intentionally dropped — the server returns unsorted results and the client re-sorts in the `onDone` handler.
- `select-file`: stays stubbed (not needed for v1)
- Set `capabilities.fileSystemAccess = true` in web mode

### State machine behavioral change (no code changes needed)

Setting `fileSystemAccess = true` changes the `init` state guard flow. Previously, web users always hit the `!capabilities.fileSystemAccess` guard and went to `restoringWebSession`. Now:

- Users with `hasPersistedLibrary` → `loadingFromPersisted` (unchanged, second guard)
- Users without persisted library → `selecting` (previously went to `restoringWebSession`)

This means `restoringWebSession` is no longer reachable in web mode. This is **intentional** — web users now get the same init flow as Electron users. The `restoringWebSession` state primarily restored DB query state from session store; with filesystem access enabled, the `selecting` state is the correct entry point for users without a persisted library.

### Cancel behavior note

When the user cancels the file browser modal, the promise rejects. The `selectingDirectory.onError` handler transitions to `loadedFromFS`. If the user already had a loaded library, they return to it. If this was the first selection (empty library), they land in `loadedFromFS` with an empty library and can re-trigger `SELECT_DIRECTORY` from the command palette.

## File Browser Modal Component

New file: `src/renderer/components/controls/file-browser-modal.tsx`

- Rendered at app root level, hidden by default
- Controlled via a promise-based `openFileBrowser()` function called by the web-mode `select-directory` implementation
- **UI:**
  - Breadcrumb path bar at top — each segment clickable to navigate up
  - Entry list — directories (folder icon) and media files (file icon), sorted dirs-first then alphabetical
  - "Open" button — resolves promise with current directory path as a bare string
  - "Cancel" button — rejects promise
- Calls `POST /api/fs/list` on each navigation click
- When roots are configured and user is at top level: shows root entries only, no "up" navigation above roots
- Dark theme styling consistent with existing app CSS

## Integration Flow

1. Web app boots → `init` → `capabilities.fileSystemAccess` is `true` → normal init checks apply
2. User clicks folder icon → `SELECT_DIRECTORY` → state enters `selectingDirectory`
3. `selectingDirectory` invokes `select-directory` → platform opens file browser modal
4. Modal fetches `/api/fs/list` with empty path → shows roots
5. User navigates by clicking directories → each click fetches `/api/fs/list`
6. User clicks "Open" → promise resolves with path string → `onDone` → `setPath` → `loadingFromFS`
7. `loadingFromFS` invokes `load-files` → platform calls `/api/fs/scan` → server walks, filters, inserts, returns library
8. State receives library → client re-sorts → `loadedFromFS` with `currentStateType: 'fs'`

**Session restore:** `hasPersistedLibrary` still takes priority in `init`, so returning users with persisted data restore their session.

**No streaming in v1:** Electron emits `load-files-batch` for incremental UI updates. Web mode skips this — `/api/fs/scan` returns everything at once. The state machine handles this fine since `loadingFromFS.onDone` accepts the complete library.

## Files to Create/Modify

### New files
- `src/renderer/components/controls/file-browser-modal.tsx` — Modal component
- `src/renderer/components/controls/file-browser-modal.css` — Styles

### Modified files
- `media-server/appconfig/config.go` — Add `RootPaths` field
- `media-server/loki_api.go` — Add `lokiFsListHandler`, `lokiFsScanHandler`, and shared media extension regex
- `media-server/main.go` — Register new routes (routes are only in `main.go`, not platform variants)
- `src/renderer/platform.ts` — Un-stub channels, add modal integration, set `fileSystemAccess = true`
