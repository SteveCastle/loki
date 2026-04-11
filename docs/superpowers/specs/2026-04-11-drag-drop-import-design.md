# Drag & Drop File Import — Design Spec

**Date:** 2026-04-11
**Status:** Approved

## Problem

There's no way to add new files to the library by dragging them into the app. Users must manually copy files to the right directory, then refresh. The app should accept native file drops and route them to the right location based on the current browsing context.

## Design

A global drop zone on the main content area accepts native file drops. The destination is determined by the current browse mode:

- **FS mode (directory browsing):** Copy the file into the currently browsed directory.
- **DB mode (tag browsing):** Place the file in the default storage root, then apply all active tags.
- **Search mode:** Place the file in the default storage root, no tags applied.

After placement: insert into the database, refresh the library view, show a toast notification.

### Behavior details

- **Copy by default, Shift+drop to move.** Standard OS convention. Move removes the source file after successful copy.
- **Name collisions:** Append a counter suffix (e.g. `image_1.jpg`, `image_2.jpg`).
- **Multiple files:** All files in a single drop are processed together. Toast shows count: "Added 3 files to /photos/vacation".
- **Non-media files:** Silently skipped (use the existing media-type validation from `file-handling.ts` / server-side validation).

### Cross-platform support

Works in both Electron and web modes:

- **Electron:** New `import-files` IPC handler copies/moves files locally, inserts into SQLite, applies tags.
- **Web:** POST to existing `/api/upload` endpoint (multipart form), then POST to `/api/assignments` for tag application.

## Drop routing

### FS mode

The renderer reads `initialFile` from the state machine context to determine the current directory path.

**Electron:** Calls `invoke('import-files', { files, destination: currentDir, move, tags: [] })`. The main process copies/moves each file to `destination/filename`, resolving collisions, then calls `insertBulkMedia` to add to the database.

**Web:** FS mode is local-directory browsing — in web mode, directories are server-side storage roots browsed via the file browser. Dropped files POST to `/api/upload` which places them in the default root's `uploads/` directory. The server's auto-ingest handles database insertion.

### DB mode

The renderer reads `dbQuery.tags` to get the active tag list.

**Electron:** Calls `invoke('import-files', { files, destination: defaultImportPath + '/imports/', move, tags })`. The `defaultImportPath` is a user-configured setting (folder picker) stored via `electron-store`. After file placement and DB insertion, `createAssignment` is called for each tag on each file.

**Web:** POST files to `/api/upload`. After upload response, POST to `/api/assignments` for each file+tag combination. The upload endpoint handles DB insertion via auto-ingest.

### Search mode

Same as DB mode but with an empty tags array — files go to the default storage root, no tags applied.

## Renderer: useFileDrop hook

New file: `src/renderer/hooks/useFileDrop.ts`

Follows the pattern of the existing `useTagDrop.tsx`. Uses `useDrop` from `react-dnd` accepting `NativeTypes.FILE`.

**Inputs (from state machine via useSelector):**
- `currentStateType` — 'fs', 'db', or 'search'
- `initialFile` — current directory path (FS mode)
- `dbQuery.tags` — active tags (DB mode)

**Shift detection:** The `monitor.getItem()` from `NativeTypes.FILE` provides the drop event. Check `event.dataTransfer` or use a keyboard listener to detect Shift for move-vs-copy.

**Return value:** `{ dropRef, isOver, canDrop }` — same pattern as useTagDrop.

**After successful import:**
- FS mode: send `LOAD_FROM_PATH` event to re-scan the directory
- DB mode: send `LOAD_FROM_DB` to re-query by active tags
- Search mode: re-send the current search query
- Show toast with result message

## Renderer: drop zone in panels.tsx

Attach `dropRef` to the main content panel. When `isOver && canDrop`, render a subtle border highlight so the user knows the drop target is active. No overlay text.

## Electron main process: import-files IPC

New IPC handler in `src/main/media.ts`.

**Channel:** `import-files`

**Args:**
```typescript
{
  files: string[],        // absolute source paths
  destination: string,    // target directory
  move: boolean,          // shift held = move instead of copy
  tags?: { label: string, category: string }[]
}
```

**Flow:**
1. For each source file, determine target path: `destination/filename`. If collision, try `filename_1.ext`, `filename_2.ext`, etc.
2. Copy file (`fs.copyFile`). If `move` is true, delete source after successful copy (`fs.unlink`).
3. Call `insertBulkMedia` with all successfully placed file paths.
4. If `tags` is non-empty, call `createAssignment` for each tag on each new file.
5. Return `{ imported: string[], failed: string[] }`.

**Preload registration:** Add `'import-files'` to the channels enum in `src/main/preload.ts`.

## Web mode: platform.ts mapping

Add `import-files` to the web mode channel-endpoint map in `src/renderer/platform.ts`. This mapping builds a `FormData` from the file list and POSTs to `/api/upload`, then handles tag assignment as a second step.

Since the web upload flow differs from the simple JSON POST pattern used by other channels (it needs multipart form data), the hook handles web uploads directly via `fetch` rather than going through the `invoke` abstraction. The platform mapping is not needed — the hook checks `isElectron` and branches.

## Electron: defaultImportPath setting

A new setting stored via `electron-store`:
- Key: `defaultImportPath`
- Value: absolute filesystem path
- Set via folder picker dialog (same pattern as DB path selection)

Used as the destination when importing files in DB/search mode. If not set, the hook prompts the user to configure it before the first DB-mode import (via a toast with an action button or a dialog).

## Files to create/modify

| File | Change |
|------|--------|
| `src/renderer/hooks/useFileDrop.ts` | **New** — native file drop hook |
| `src/renderer/components/layout/panels.tsx` | Attach drop zone, add drop indicator styling |
| `src/main/media.ts` | Add `import-files` IPC handler |
| `src/main/preload.ts` | Register `import-files` channel |
| `src/renderer/platform.ts` | No change needed — web upload handled directly in hook |

## What doesn't change

- State machine (`state.tsx`) — no new states, uses existing reload events
- Existing drag-and-drop (`useTagDrop.tsx`, tag/media drag sources) — unaffected
- Media server upload endpoint — already supports the needed flow
- Database schema — no changes
