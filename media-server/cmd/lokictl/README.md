# lokictl

Command-line client for the Lowkey Media Server HTTP API, designed for AI
agents first: deterministic JSON on stdout, JSON errors on stderr, stable exit
codes. Ships next to `media-server(.exe)`.

## Build

```sh
# from the repo root
npm run build:cli            # -> media-server/lokictl(.exe)
# or directly
cd media-server && go build -o lokictl.exe ./cmd/lokictl
```

`npm run build:server` also builds it (skip with `SKIP_CLI=1`).

## Output contract

- **stdout**: the result, pretty-printed JSON (`-o table` renders arrays of
  objects as TSV for humans).
- **stderr**: errors as a single JSON object `{"error":..., "status":..., "hint":..., "detail":...}`,
  plus progress lines for `--follow`/`--wait`.
- **Exit codes**: `0` success · `1` server/network error · `2` usage error ·
  `3` an awaited job/workflow failed, was cancelled, or timed out.

## Connection & auth

Server URL and token resolve as: flag (`--server`, `--token`) > env
(`LOKICTL_SERVER`, `LOKICTL_TOKEN`) > config file > `http://localhost:8090`.

```sh
lokictl health                            # no auth needed
lokictl login --password <pw>             # default --username admin
# token is stored at <UserConfigDir>/lokictl/config.json (0600)
```

A `401` on any command means: run `lokictl login` again.

## Command reference

Run `lokictl help` for the always-current list. Highlights:

| Area | Commands |
|---|---|
| Discovery | `health`, `stats`, `task list`, `task show <id>`, `lokictl help` |
| Jobs | `job run <task> [args...] [--field k=v] [--wait] [--follow] [--timeout D]`, `job list [--state S]`, `job get/wait/logs/cancel/copy/remove <id>`, `job clear --yes` |
| Workflows | `workflow list/get/create/update/delete/run`, `workflow run-adhoc --dag FILE\|-` |
| Library queries | `media query [--tag ... --visual ... --similar ...]`, `media search/similar/visual/metadata/tags/describe/delete` |
| Raw SQL (read-only) | `db query "SELECT ..." [--arg V]`, `db tables`, `db schema [table]` |
| Taxonomy | `taxonomy [--category C]`, `tag create/delete/rename/move/assign/unassign`, `category create/delete/rename` |
| Dependencies | `deps status`, `deps download <model-id> --wait`, `deps verify/delete` |
| Server admin | `config get`, `config set --json '{...}'`, `fs list/scan`, `upload <file>...` |
| Escape hatch | `api <METHOD> <path> [--body JSON\|@file\|-]` — any endpoint, auth attached |

Destructive commands (`job clear`, `media delete`, `tag delete`,
`category delete`, `workflow delete`, `deps delete`) refuse to run without
`--yes` (exit 2).

## Agent cookbook

**Run a task and wait for the result** (task options are free-form tokens;
`--field k=v` sends values verbatim, safe for spaces/quotes/newlines):

```sh
lokictl task show metadata                     # discover its options first
lokictl job run metadata --type description --apply all "C:/pics/x.jpg" --wait --timeout 10m
# → final job JSON; exit 0 completed, 3 error/cancelled/timeout
```

**Watch a long job's output live:**

```sh
lokictl job run ffmpeg-scale --width 1280 "C:/vids/in.mp4" --follow
# stdout lines stream to stderr; final job JSON lands on stdout
```

**Ask the database anything (read-only, bind args via `?`):**

```sh
lokictl db tables
lokictl db schema media_tag_by_category
lokictl db query "SELECT hash, COUNT(*) n, GROUP_CONCAT(path, CHAR(10)) paths FROM media WHERE hash IS NOT NULL GROUP BY hash HAVING n > 1 ORDER BY n DESC" --limit 50
lokictl db query "SELECT tag_label, COUNT(*) n FROM media_tag_by_category GROUP BY tag_label ORDER BY n DESC" --limit 25
```

**Search the library three ways:**

```sh
lokictl media query --tag sunset --exclude-tag blurry --mode AND
lokictl media visual "a red car in the snow" --limit 20      # text -> image embeddings
lokictl media similar "C:/pics/x.jpg"                        # image -> image
```

**Run a saved workflow and wait for every job it spawns:**

```sh
lokictl workflow list
lokictl workflow run <id> --input "C:/pics/new" --wait
```

**Install a model dependency:**

```sh
lokictl deps status
lokictl deps download siglip2-base-patch16-224 --wait
```

## Notes & limits

- `job logs` only works while a job is running — the server streams stdout
  over SSE and does not persist it.
- `db query` accepts a single `SELECT`/`WITH` statement; the server enforces
  read-only at the SQLite level (`query_only`). Rows are capped (default
  1000, max 10000) with `"truncated": true` when clipped.
- `--arg` values are sent as strings; SQLite type affinity coerces them for
  comparisons against numeric columns.
- Job-input tokens containing double quotes are rejected (the server's
  splitter has no escape syntax) — pass such values with `--field name=value`.
- `stats` is Windows-server-only; `workflow` saved-routes need a server built
  from this branch or later on macOS/Linux.
- `config set` merges over `config get` and never echoes redacted secrets or
  storage roots back; edit credentials in the web Config UI.

Real sample:

```text
$ lokictl health
{
  "jobs": { "cancelled": 5, "completed": 147, "error": 11, "in_progress": 1, "pending": 0, "total": 164 },
  "status": "healthy",
  ...
}
```
