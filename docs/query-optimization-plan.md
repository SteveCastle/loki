# Media query optimization plan

Living document. Step 1 (instrumentation) is complete. Steps 2+ depend on the
profile data the user collects from real usage.

## Step 1 — Instrumentation (done)

Both runtimes now append a JSONL record per executed media query.

- **Electron renderer / main process** — every call through `Database.run`,
  `Database.get`, `Database.all` is timed and written. The two query-builder
  hot paths (`loadMediaByTags`, `loadMediaByDescriptionSearch`) are tagged
  with `name` so they're trivial to filter from the noise.
- **Go media server** — six hot query paths instrumented via `querylog.Start`:
  - `GetItems`, `GetItems.count`
  - `GetRandomItems`
  - `getItemsWithExistenceFilter`
  - `getRandomItemsWithExistenceFilter`
  - `GetTags`
  - `GetPathsByQuery`

### Log file locations

| Source | Path |
|---|---|
| Electron | `%APPDATA%\<app>\query-log.jsonl` (whatever `app.getPath('userData')` resolves to) |
| Go server (Windows) | `%ProgramData%\Lowkey Media Server\query-log.jsonl` |
| Go server (Linux) | `/var/lib/lowkeymediaserver/query-log.jsonl` |
| Go server (macOS) | platform-specific (whatever `platform.GetDataDir()` returns on Darwin) |

### Record format (JSONL, one record per line)

```json
{
  "ts": "2026-04-30T15:30:21.123Z",
  "source": "electron" | "go-server",
  "name": "loadMediaByTags",
  "sql": "SELECT mtc.media_path, ... GROUP BY media_path HAVING COUNT(DISTINCT tag_label) = 3 ORDER BY weight",
  "params": ["dog", "outdoor", "summer"],
  "duration_ms": 412.831,
  "rows": 287,
  "error": null
}
```

### How to use

1. Move the existing `query-log.jsonl` aside (or just rotate it) to start with
   a clean run.
2. Use the app for a while, doing the things that feel slow — especially:
   typed query searches with multiple tags, EXCLUSIVE-mode filtering, large
   result sets, sort changes, and any view that mixes search with tag
   filtering.
3. Send back the `query-log.jsonl` file. We'll sort by `duration_ms`, group by
   `name` / SQL shape, and tackle the worst offenders first.

A useful ad-hoc analysis:

```bash
jq -c 'select(.duration_ms > 100)' query-log.jsonl \
  | jq -s 'sort_by(-.duration_ms) | .[0:50]'
```

…or by SQL shape:

```bash
jq -c '{name, duration_ms, rows, params_n: (.params|length)}' query-log.jsonl \
  | sort -t: -k3 -n -r | head -50
```

## Step 2 — Likely hot spots (hypotheses, validate from log)

These are the query patterns I expect to dominate the slow tail. Each comes
with the index/rewrite I'd try.

### 2.1 Tag filtering — Electron `loadMediaByTags` (`src/main/media.ts`)

The current SQL is:

```sql
SELECT ... FROM media_tag_by_category mtc
LEFT JOIN media m ON m.path = mtc.media_path
WHERE tag_label = $1 OR tag_label = $2 OR ...
GROUP BY media_path
HAVING COUNT(DISTINCT tag_label) = N   -- AND mode
ORDER BY weight
```

Failure modes likely to show in the log:

- **Cost grows with tag count.** With many tags, `COUNT(DISTINCT tag_label)`
  forces a sort/aggregate over potentially every matching row.
- **`ORDER BY weight` with no index** — silently sorts the entire result set.
- **`LEFT JOIN media`** — pulls every column from `media` even though only
  a few are projected; if `media` is wide (preview blob, transcript, etc.)
  this churns pages from the cache.

Fixes to try after confirming with the log:

```sql
-- Required indexes
CREATE INDEX IF NOT EXISTS idx_mtc_tag_label ON media_tag_by_category(tag_label);
CREATE INDEX IF NOT EXISTS idx_mtc_media_path_tag ON media_tag_by_category(media_path, tag_label);
CREATE INDEX IF NOT EXISTS idx_mtc_weight ON media_tag_by_category(weight);
CREATE INDEX IF NOT EXISTS idx_media_path ON media(path); -- usually already exists as PK
```

Better SQL for AND mode (replaces `COUNT(DISTINCT)` aggregation):

```sql
SELECT m.path, m.elo, m.height, m.width
FROM media m
WHERE NOT EXISTS (
  SELECT 1 FROM (VALUES (?), (?), (?)) AS req(tag)
  EXCEPT
  SELECT mtc.tag_label FROM media_tag_by_category mtc
  WHERE mtc.media_path = m.path
)
ORDER BY m.elo;
```

…or, simpler and almost always faster on SQLite for small N:

```sql
SELECT m.path, m.elo, m.height, m.width FROM media m
WHERE EXISTS (SELECT 1 FROM media_tag_by_category WHERE media_path = m.path AND tag_label = ?)
  AND EXISTS (SELECT 1 FROM media_tag_by_category WHERE media_path = m.path AND tag_label = ?)
  AND EXISTS (SELECT 1 FROM media_tag_by_category WHERE media_path = m.path AND tag_label = ?)
ORDER BY m.elo;
```

The Electron `loadMediaByDescriptionSearch` already uses this `EXISTS-per-tag`
shape. Bringing `loadMediaByTags` in line removes the GROUP BY/HAVING cost
entirely.

For OR mode the current single `EXISTS (... tag IN (...))` is fine; just need
the indexes.

### 2.2 Description search LIKE patterns

`loadMediaByDescriptionSearch` runs `LIKE '%token%'` against `media.description`,
`media.path`, `media.hash`, plus an `EXISTS` against `media_tag_by_category` for
each token. Leading-wildcard `LIKE` defeats every B-tree index, so the
description column always full-scans.

If the log shows description search dominating: add an FTS5 virtual table
mirror of `media(description, path)` and route the description path through
FTS. Tag-prefix `LIKE` (e.g. `tag:dog*`) can use a regular index if rewritten
to `tag_label >= 'dog' AND tag_label < 'doh'` — worth doing only if the data
shows tag prefix searches matter.

### 2.3 Go server `GetItems` and existence filtering

The log will show whether the count query (`SELECT COUNT(*) FROM media m
WHERE …`) or the page query is the slow one. They share a WHERE clause; if
the count is the bottleneck, options are:

- Drop `totalCount` for typed queries (return `-1`, page until empty); the UI
  already handles `-1`.
- Maintain a denormalized counter table (only worth it if the same query
  shape repeats often — tag/path counts).

The existence-filter loop (`getItemsWithExistenceFilter`,
`getRandomItemsWithExistenceFilter`) may run `db.Query` many times per user
request when results are sparse. Watch for the same SQL repeated within
milliseconds in the log — that's the symptom.

### 2.4 `GetTags` post-fetch

After every `GetItems`/`GetRandomItems`, a second `IN (?, ?, ?, …)` query
fetches tags for the page. With page size 100 and many tags per item this can
get large.

If the log shows `GetTags` dominating: bundle it into the main query via a
`json_group_array` or correlated subquery. Or — simpler — keep the separate
fetch but ensure `media_tag_by_category(media_path)` is indexed.

### 2.5 Typed-query AST → SQL pathology

The query builder in `media-server/media/search.go` composes nested AND/OR/NOT
nodes. Each node returns its own SQL fragment that gets parenthesized and
concatenated. Two things to verify from the log:

- **Repeated identical sub-queries.** AND of three tags emits three near-
  identical `EXISTS (SELECT 1 FROM media_tag_by_category …)` blocks. SQLite
  should fold these but doesn't always; if `EXPLAIN QUERY PLAN` shows three
  separate index scans for them, batch them via a single subquery with
  `tag_label IN (…) GROUP BY media_path HAVING COUNT(DISTINCT) = N`.
- **NOT EXISTS with no anchoring tag.** A query like `NOT tag:foo` with no
  positive predicate scans the whole `media` table and runs the NOT EXISTS
  per row. Always anchor at least one positive predicate, or special-case
  pure-NOT into `WHERE m.path NOT IN (SELECT media_path FROM mtc WHERE tag = ?)`.

## Step 3 — Priorities once log data arrives

1. **Find the p95 query.** Sort by `duration_ms` desc, look at the SQL +
   params shape. That's the first target.
2. **Confirm with `EXPLAIN QUERY PLAN`.** Run the captured SQL with the
   captured params against a copy of the user's DB; verify the plan is using
   indexes (or not).
3. **Add indexes first** — they're zero-risk and often catch 80% of the
   regression. Migration goes in `database.ts` (Electron) and
   `media/media.go::InitializeSchema` (Go).
4. **Rewrite only if indexes don't fix it.** GROUP BY/HAVING → EXISTS is the
   biggest single win available without schema changes.
5. **Re-measure** by running the same workflow again with the log open and
   confirming `duration_ms` dropped.

## Disabling logging later

- Electron: call `setQueryLogEnabled(false)` from `queryLog.ts`, or remove the
  `logQuery(...)` calls from `database.ts`.
- Go: comment out `querylog.Start(...)` calls or add an env-var gate
  (`if os.Getenv("LOWKEY_QUERY_LOG") == ""`) inside `querylog.Log` once we're
  done profiling.
