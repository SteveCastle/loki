# Performance & correctness audit — remaining work

Audit of 2026-10-22 (server + React client), judged at real scale: the live
library is **2.85M media, 19.4M tag assignments, 22.8 GB** (`dream-x.sqlite`).
Fixed items are committed on `feature/scene-editor` (see "Done" below). The
items here were found and verified, but are not yet fixed. They are ordered
by expected user impact.

## 1. Covering indexes for tag/category counts — HIGH (needs a maintenance window)

- **Problem:** `COUNT(DISTINCT media_path) … WHERE tag_label = ?` (`/api/tags/count`,
  the sidebar's per-tag counts, `library_api.go` category counts) has no covering
  index. SQLite walks `idx_mtc_tag_weight` and then does a table lookup per row.
- **Measured on the live DB:** 30.8 s for `breasts` (309k rows).
- **Measured on a compacted copy of the table:**

  | | Count query |
  |---|---|
  | Without the new index | 1.8 s |
  | With `(tag_label, media_path)` | 98 ms |

- **Cost:** building the index took 58 s on the compacted copy, and will take
  longer on the live file. It holds the write lock for that whole time, so the
  viewer and jobs get "database is locked". That is why it was **not**
  auto-applied at startup.
- **Plan:**
  - Add `CREATE INDEX IF NOT EXISTS idx_mtbc_tag_path ON media_tag_by_category(tag_label, media_path)`
    and `(category_label, media_path)`.
  - Run them as an explicit maintenance task (lokictl / a Jobs-page button)
    that the user starts, or on startup only when the table is small.
- **Cleanup while there:** drop the duplicate indexes. `idx_mtc_tag_label`
  duplicates `idx_mtbc_tag_label`; `idx_mtbc_media_path` and
  `idx_mtc_media_path` duplicate the PK prefix; `idx_media_path` duplicates
  the `media.path` PK. That is pure write and disk cost on 19M rows. Check the
  viewer's `src/main` schema too, since it re-creates some of them.
- **Also:** batch the sidebar counts into one `GROUP BY` request instead of
  one POST per tag (`components/taxonomy/tag-count.tsx`).

## 2. Ingest follow-ups create one job per file — HIGH at scale

- **Where:** `tasks/media_ingest.go:134`.
- **Problem:** this calls `AddJob` per file per follow-up, so a 1M-file ingest
  with `--autotag --filemeta` queues 2M jobs. Each one costs a row insert, a
  template render, a broadcast and a claim.
- **Fix:** queue one batch job per follow-up with a path-list input; the
  `process` task already takes lists.

## 3. Web: thumbnails cost two round trips — MED-HIGH

- **Where:** `image.tsx`, `video.tsx`.
- **Problem:** each tile does a POST `/api/media/preview` to get a path, then
  GETs `/media/file`.
- **Fix:**
  - Point `<img src>` straight at `/media/thumbnail?path=&cache=&ts=`
    (`loki_api.go`). It already exists, is cacheable, and is unused by the SPA.
  - Fix its `immutable` cache header, which is keyed by media path rather than
    by thumbnail content.

## 4. Web requests never time out or supersede — MED-HIGH (correctness)

- **Where:** `platform.ts`. `authFetch`/`jsonPost` have no default timeout,
  and web `loadMediaByQuery` has no latest-wins abort.
- **Problem:** a hung server leaves the UI loading forever, and a superseded
  query keeps downloading.
- **Fix:**
  - Give `authFetch` a default `AbortSignal.timeout`, with a longer one for
    queries and scans.
  - Add an `AbortController` per query kind. The server side already cancels
    on disconnect (`QueryContext`).

## 5. Masonry layout is O(library) per image load and per scroll — HIGH when masonry is used

- **Where:** `components/list/list.tsx:567-646`.
- **Problems:**
  - Each thumbnail load copies the dimensions Map and re-lays-out every item.
  - Each scroll runs `layout.items.filter` over everything.
  - `CACHE_MASONRY_DIMENSIONS` spreads an ever-growing object in context.
- **Fix:**
  - Lay out incrementally from the first changed index, batching arrivals per rAF.
  - Binary-search the visible window.
  - Keep the dimensions in a ref'd Map.

## 6. Moved files keep stale paths in the grid — MEDIUM (correctness)

- **Where:** `filter.ts`'s cache key (`libraryLoadId::filters::sortBy`).
- **Problem:** the key ignores library identity. `updateFilePath`
  (`state.tsx`) changes `library` without a new load id, and in the
  single-item case mutates `item.path` in place, so memoised cells never
  re-render.
- **Fix:** mint a new item and a new `libraryLoadId` on path updates.

## 7. Whole library cloned over IPC on every load — MEDIUM at scale

- **Where:** `state.tsx` `setSessionValues({ library })` after each
  query/load/delete.
- **Problem:** this structured-clones up to millions of items over IPC.
  `session.json` no longer rewrites on cursor moves (done), but the per-load
  clone and write remain.
- **Fix:** persist the query/folder and re-derive on boot, or cap the
  persisted library.

## 8. Electron session store first-load parse — LOW-MED

- **Where:** `session-library.json`.
- **Problem:** it is still parsed in full on the main process at boot.
- **Fix:** consider streaming, or deferring it until the renderer asks for it.

## 9. HLS "generating" spinner can stick forever — MEDIUM (correctness)

- **Where:** `video.tsx` (~426).
- **Problem:** `if (!res.ok || cancelled) return;` stops polling on 4xx/5xx
  but leaves `hlsGenerating = true`. The HLS check, poll and generate fetches
  also have no timeout or abort.

## 10. Job detail page races the live job — MEDIUM

- **Where:** `main.go:383` (and the darwin/linux copies).
- **Problem:** `GetJob(id)` hands the template the live `*Job`. It serialises
  `Stdout` while `PushJobStdout` appends to it, which is torn data and a
  data race.
- **Fix:** use `JobSnapshot(id, tail)` like `job_api.go`, and make `GetJob`
  return a copy.

## 11. SSE payloads and drop policy — MEDIUM

- **Payload size:** `serializeListUpdate` marshals the whole `Job`, including
  `OutputFiles`/`Input` (a child's input concatenates a parent's 1M outputs),
  into every event. Send counts instead.
- **Dropped events:** state events share the hub/client buffers with stdout
  lines and are dropped when full, so a log flood can lose the "completed"
  event. Use a separate never-dropped (coalesced) channel for state changes.
- **Memory:** cap in-memory `Job.Stdout` to a ring of a few thousand lines;
  the full log stays in the DB. `loadJobsFromDB` reloads every historical
  job's full stdout and outputs at startup.

## 12. Smaller items

- **Tag sidebar previews (web):** Go `/api/taxonomy/tags` omits
  `thumbnail_path_600`, so every tag card POSTs `/api/tags/preview`.
- **Folder scan blocks the response (web):** `fsbrowser.go:178` runs
  `insertBulkMediaPaths` (1M upserts) before responding. Respond first and
  insert in the background, in batches.
- **`/stream` is open to any site:** it is unauthenticated with
  `Access-Control-Allow-Origin: *`, so any website can read job output and
  media paths through it. Gate it with `RolePublicRead` and echo trusted
  origins only.
- **Admin `/media/api` paging:** it runs `COUNT(*)` per page and pages with
  OFFSET (`media/media.go:864`).
- **`useMediaDimensions` listeners:** they never attach in list cells, because
  the effect depends on `[ref]` and the element mounts later. Use a callback
  ref.
- **Image decoding:** `image.tsx` is missing `decoding="async"`.
- **Tag-file consolidation blocks the main process:** `fs.copyFileSync` per
  file (`src/main/taxonomy.ts:713`). Use async copies with progress, and wrap
  the UPDATEs in one transaction.
- **HLS data races (`hls.go`):** `entry.queued` and `entry.progress` are
  written without `hlsInflightMu`, and `os.RemoveAll` runs while holding the
  global HLS mutex.
- **Context palette polling:** it polls dependency status every 3 s,
  bypassing `fetchStatusShared`.
- **Leaked watcher goroutines:** each command job leaks one `ctx.Done`
  watcher goroutine. Don't just cancel the ctx after the task returns: the
  watcher would then `taskkill` a PID that may have been reused. Stop the
  watcher once `Wait` returns instead.

## Done (this audit)

| Commit | Fix |
|---|---|
| 06d38e8 | SSE: send-on-closed-channel crash; idle tabs spinning a core after being reaped |
| 88b70f4 | Task panics no longer kill the server; jobs a task leaves running are finalised |
| 14bc0f3 | Command/ingest tasks drain pipes before `Wait` (successful runs were marked Error; yt-dlp lost its path) |
| 2f1f6c6 | Runner signal no longer blocks under `q.mu` (queue-wide deadlock) |
| 8584737 | Grid-wide re-renders during playback; thumbnails clobbering the detail video's subtitles and audio track |
| af1d2b4 | Network-drive detection without `wmic` (all thumbnails were throttled to 3) |
| 7896e42 | WAL for the server's DB; no 2 GB file cap; HTTP header/idle timeouts |
| 903c87e | Job stdout/outputs persisted in the background: 20k log lines went from 89 s to 0.06 s |
| 1f164da | Media list endpoints stream typed rows, cancel on disconnect, and check `rows.Err` |
| 80a6c18 | gzip for remote clients; content-hash ETag for the SPA bundle |
| c3efe41 | Session store: cursor moves no longer rewrite the whole library; batch merge fix |
| 29ae2cb + 556a288 | Thumbnail generation honours cancellation; web tiles abort their preview requests |
| bbde491 | Folder scans linear (200k files: 21.7 s to 0.46 s); stale batches no longer leak into DB results |
| 78e1aad | Vector indexes build in the background, so HTTP is up immediately |
| 832e7e8 | Tag/category writes are all-or-nothing (409 on conflict) |
| 5dc2ab9 | `/api/media/metadata` is scoped for public visitors |
| b4be898 | `ClaimJob` skips finished history; clearing finished jobs is a single pass (2.25 s to 0.12 s for 2.5k jobs) |
| 8e6e000 | A cancelled job keeps its GPU/host buckets until its task actually stops |
