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

For automation, prefer a long-lived **API key** over the login JWT. Keys are
`lk_`-prefixed, tied to a user, revocable, and accepted anywhere a token is
(`--token`, `LOKICTL_TOKEN`, config file, `Authorization: Bearer`, or an
`X-API-Key` header). Create one in the web UI (Config → API Keys) or:

```sh
lokictl login --password <pw>             # bootstrap once
lokictl key create --name ci --save       # mint a key and store it as the CLI token
lokictl key list                          # id, owner, prefix, created, last used
lokictl key revoke --id 3
```

A `401` on any command means: run `lokictl login` again (or the API key was
revoked — create a new one).

## Command reference

Run `lokictl help` for the always-current list. Highlights:

| Area | Commands |
|---|---|
| Discovery | `health`, `stats`, `task list`, `task show <id>`, `lokictl help` |
| Jobs | `job run <task> [args...] [--field k=v] [--wait] [--follow] [--timeout D]`, `job list [--state S]`, `job get/wait/logs/cancel/copy/remove <id>`, `job clear --yes` |
| Workflows | `workflow list/get/create/update/delete/run`, `workflow run-adhoc --dag FILE\|-` |
| Library queries | `media query [--tag ... --visual ... --similar ...]`, `media search/similar/visual/image-search/metadata/tags/delete` |
| Media data | `media describe <path> (--text D\|--clear)`, `media transcript <path> [--text T\|--clear]`, `media rate <path> [--elo E --views N --wins N --losses N]`, `media thumbs <path> [--regenerate]`, `media generate <path> --type T [--wait]` |
| Embeddings index | `index status/models/rebuild`, `index missing [--model M]`, `index get <path> [--vector]`, `index delete <path> --yes`, `index prune --yes`, `index embed [args...] [--wait]` |
| Raw SQL (read-only) | `db query "SELECT ..." [--arg V]`, `db tables`, `db schema [table]` |
| Taxonomy | `taxonomy [--category C]`, `tag create/delete/rename/move/assign/unassign/assign-bulk/unassign-bulk`, `tag list/count/weight/has/timestamp/assignment-weight`, `category create/delete/rename/count` |
| Dependencies | `deps status`, `deps download <model-id> --wait`, `deps verify/delete` |
| Server admin | `config get`, `config set --json '{...}'`, `fs list/scan`, `upload <file>...`, `whoami` |
| API keys | `key create --name N [--username U] [--save]`, `key list`, `key revoke --id N` |
| Escape hatch | `api <METHOD> <path> [--body JSON\|@file\|-]` — any endpoint, auth attached |

Destructive commands (`job clear`, `media delete`, `tag delete`,
`tag unassign-bulk`, `category delete`, `workflow delete`, `deps delete`,
`index delete`, `index prune`) refuse to run without `--yes` (exit 2).

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

**Search the library four ways:**

```sh
lokictl media query --tag sunset --exclude-tag blurry --mode AND
lokictl media visual "a red car in the snow" --limit 20      # text -> image embeddings
lokictl media similar "C:/pics/x.jpg"                        # image -> image (library item)
lokictl media image-search "C:/downloads/some.jpg"           # image -> image (any local file)
```

**Keep the embedding index healthy:**

```sh
lokictl index status                       # coverage, orphans, per-model vector counts
lokictl index missing --limit 0            # how many items still need embeddings
lokictl index embed --query "SELECT path FROM media" --wait  # or: index embed <paths...>
lokictl index prune --yes                  # drop vectors for deleted media
lokictl index rebuild --timeout 10m        # rebuild ANN index (also runs at startup)
```

**Curate tags in bulk** (pipe any path list — one per line — into `--stdin`):

```sh
lokictl tag list -o table                                    # every tag with usage counts
lokictl db query "SELECT path FROM media WHERE path LIKE '%/vacation/%'" -q \
  | jq -r '.rows[][]' | lokictl tag assign-bulk vacation --category trips --stdin
lokictl tag unassign-bulk blurry --stdin --yes < paths.txt
```

**Write metadata directly, or have AI generate it:**

```sh
lokictl media describe "C:/pics/x.jpg" --text "Two dogs on a beach"
lokictl media transcript "C:/vids/talk.mp4"                  # read
lokictl media generate "C:/vids/talk.mp4" --type transcript --wait
lokictl media rate "C:/pics/x.jpg" --elo 1600
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
