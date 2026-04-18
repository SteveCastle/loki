# Archive-as-Directory Design

**Date:** 2026-04-18
**Scope:** Electron app only (Lowkey Media Viewer). Web mode / Go server out of scope.

## Summary

Let users open `.cbz` / `.zip` archives as if they were directories. On open, the archive is extracted to a cached temp directory and the existing directory-loading pipeline runs unchanged. Supports menu "Open Archive…", drag-and-drop, and OS file association. MVP is read-only, ZIP-family only (no CBR/RAR).

## Non-goals

- CBR / RAR / 7z support.
- Web-mode (Go server `/api/fs/scan`) support.
- Nested archive browsing (archive inside archive).
- Writing back into archives.
- In-place, on-demand streaming without extraction to disk.

## Architecture

New module **`src/main/archives.ts`** with a small, focused API:

```ts
isArchivePath(p: string): boolean
extractArchive(p: string): Promise<string>   // returns temp dir path; dedupes in-flight + uses LRU cache
cleanupArchives(): Promise<void>             // called on app quit
```

Integration points (all minimal):

1. **`src/main/load-files.ts`** — at the top of `loadFiles`, if the incoming path is an archive, replace it with `await extractArchive(path)` and proceed as normal with recursive mode forced on. Everything downstream (walk, filter, stream into library) works unchanged.
2. **`src/main/main.ts`** — add `select-archive` IPC handler that calls `dialog.showOpenDialog` with `{ properties: ['openFile'], filters: [{ name: 'Comic Archive', extensions: ['cbz', 'zip'] }] }`. Register an app `will-quit` handler that calls `cleanupArchives()`.
3. **`src/file-types.ts`** — add `Archive` to the `Extensions` enum and extend `getFileType()` to recognize `.cbz` / `.zip` as `Archive` (distinct from `Image`/`Video`/`Audio`/`Document`/`Other`).
4. **`src/renderer/hooks/useFileDrop.ts`** — when a dropped file is an archive, route to open-as-directory (emit the existing directory-selection event with the archive's path) instead of `import-files`. Extend `resolveDirectory()` so an archive path is passed through unchanged (not collapsed to its parent).
5. **`src/renderer/state.tsx`** — no state-machine changes needed. `loadingFromFS` accepts arbitrary paths; the translation happens in the main process.
6. **Menu** (wherever "Select Directory" is defined) — add an "Open Archive…" item that invokes `select-archive` and, on success, dispatches the same `SELECT_DIRECTORY` / `setPath` flow used for a chosen directory.
7. **`package.json`** — add `yauzl` dep. Extend `build.fileAssociations` with `.cbz` and `.zip` entries.
8. **`src/renderer/platform.ts`** — add `select-archive` to the Electron-only channel list. In web mode it throws a "not supported" error (consistent with other Electron-only channels).

## Extraction & caching

**Library:** `yauzl` — maintained, streaming, no native code.

**Temp dir layout:**

```
%TEMP%/lowkey-archives/
  <hash>/
    <extracted files, preserving internal paths>
    .meta.json         # { sourcePath, extractedAt, sizeBytes }
```

- **Hash:** first 16 chars of `sha1(absolutePath + mtimeMs)`. Modifying or replacing the archive produces a new hash → re-extract on next open.
- **On extract:** iterate entries; skip directory entries; skip entries whose resolved path escapes the extraction root (zip-slip guard); skip entries that don't match the existing media-file extension filter (images/video/audio).
- **Failure:** if any error occurs mid-extraction, `rm -rf <hash>/` and reject. No partial cache entries.
- **Concurrency:** a `Map<hash, Promise<string>>` dedupes simultaneous extractions of the same archive.

**LRU cache:**

- Limits: **N = 8 recent archives** or **M = 2 GB total**, whichever hits first. Constants at the top of `archives.ts`.
- Index rebuilt on startup by scanning `%TEMP%/lowkey-archives/*/.meta.json` — no separate DB.
- On each `extractArchive`, update an in-memory access-time map. If over limits, `rm -rf` the oldest entries, skipping any entry currently in use (refcount).

**Cleanup on app quit:**

- `will-quit` handler deletes `%TEMP%/lowkey-archives/` entirely. (If this proves too aggressive in practice, switch to keep-on-quit + LRU-only eviction.)

## User flow

**Entry points (all land in the same state):**

1. **Menu → "Open Archive…"** → `select-archive` IPC → dispatch same XState event as directory selection with the chosen archive path.
2. **Drag-and-drop** of a single archive file → route to open-as-directory. Multiple archives dropped: open the first, ignore the rest.
3. **OS file association / command-line arg** → existing `process.argv` / `initialFile` path handles it. `resolveDirectory()` is extended so archive paths pass through rather than being collapsed to their parent.

**While viewing an archive:**

- Library shows extracted media files, flat (subfolders within the archive are walked recursively), sorted by path with existing natural-sort.
- Title bar / breadcrumb shows the archive path (e.g., `D:\Comics\Vol01.cbz`). No special badge.
- Recursive toggle has no effect while in an archive (archive is always treated as recursive). Not hidden; we may grey it out in a follow-up if it's confusing.
- The existing "loading files" spinner covers extraction. No dedicated progress UI for MVP.

**File filter:** reuse the existing media-type filter — `ComicInfo.xml`, `.DS_Store`, `Thumbs.db`, etc. are dropped during extraction; any embedded videos or audio are preserved and shown alongside images.

## Error cases

| Case | Behavior |
|---|---|
| Archive missing / unreadable | Error toast via existing `load-files` failure path. No temp dir created. |
| Corrupted / not a valid zip | Error toast with `yauzl`'s message. Partial `<hash>/` deleted. |
| Password-protected zip | Error toast: "Password-protected archives are not supported." |
| Zip-slip entry (absolute path or `..` escape) | Skip entry, log warning; do not abort whole extraction. |
| Case-folded name clash on Windows | Later entry wins. Acceptable; these archives are malformed. |
| Archive has zero media entries | Extraction succeeds; library loads empty; existing empty-dir UX renders. |
| Disk full / write error mid-extraction | `rm -rf <hash>/`, error toast, no half-populated cache. |
| Simultaneous opens of same archive | Dedupe via in-flight promise map; both callers await the same extraction. |
| Archive modified while open | Not detected mid-session. Next open sees new mtime → new hash → re-extract. |
| LRU eviction race with active use | Refcount guard; do not evict an entry currently referenced. |

## Testing

**`src/__tests__/archives.test.ts`** (unit) — uses fixture archives checked into `src/__tests__/fixtures/archives/`:

- `flat.cbz` (5 jpegs at root) → extracts to temp, all 5 present.
- `nested.cbz` (images in subfolders + `ComicInfo.xml`) → xml skipped, media preserved in subfolders.
- `zipslip.zip` (entry `../evil.txt`) → entry skipped, no file written outside extraction dir.
- `corrupt.zip` (truncated) → rejects with error, no partial temp dir remains.
- Cache hit: calling `extractArchive` twice on same path returns same dir and does not re-extract (spy on `fs.writeFile` or compare mtimes).
- LRU eviction: force cache past limit → oldest entry's dir is gone.

**Extend `load-files.test.ts`** (or create) — `loadFiles` on an archive path returns the same shape as on a directory, with the expected media files.

**Manual test plan:**

- Open `.cbz` via menu.
- Open `.cbz` via drag-drop onto window.
- Open `.cbz` via double-click in OS file browser.
- Confirm temp dir cleanup on app quit (inspect `%TEMP%/lowkey-archives/`).
- Reopen same archive → instant (cache hit).

## Dependencies

- `yauzl` (new) — zip reader.

## Open questions / future work

- CBR / RAR support via a bundled `unrar` binary, following the existing `src/main/resources/bin/<platform>/` pattern used for ffmpeg/exiftool.
- Go server / web-mode parity.
- Progress UI for large archives.
- "Navigate into archive" UX (preserve internal folder hierarchy) if user feedback asks for it.
