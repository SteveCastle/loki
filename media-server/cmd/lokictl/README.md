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
(`LOKICTL_SERVER`, `LOKICTL_TOKEN`) > config file > auto-detected default.
The default is `http://localhost:<port>` where the port comes from the
`LOWKEY_PORT` env var, then the local media-server's own `config.json`
(`port` field), then `10111` ("L0K1" — the server's default port).

```sh
lokictl health                            # no auth needed
lokictl login                             # opens the browser to authorize (recommended)
lokictl login --no-browser                # prints the authorization URL instead
lokictl login --password <pw>             # headless / scripted (default --username admin)
# credential is stored at <UserConfigDir>/lokictl/config.json (0600)
```

The browser flow is the RFC 8252 loopback pattern used by modern CLIs: the
CLI listens on an ephemeral `127.0.0.1` port, the browser authorizes it on an
approve page (logging you in first if needed), and the CLI receives a
one-time code it exchanges (PKCE) for a freshly minted `lk_` API key — a
long-lived, revocable credential, unlike the JWT the password flow returns
(valid for a year and not individually revocable). It requires the browser to run on the same machine as the CLI; over
SSH, use `--password`.

API keys remain fully supported everywhere a token is accepted (`--token`,
`LOKICTL_TOKEN`, config file, `Authorization: Bearer`, or an `X-API-Key`
header). Manage them in the web UI (Config → API Keys) or:

```sh
lokictl key create --name ci --save       # mint a key and store it as the CLI token
lokictl key list                          # id, owner, prefix, created, last used
lokictl key revoke --id 3
```

A `401` on any command means: run `lokictl login` again (or the API key was
revoked — create a new one).

## Command reference

Run `lokictl help` for the always-current list. The full reference, with every
flag, is published at <https://lowkeyviewer.com/server/cli.html>
(`docs/server/cli.html`; `TestDocsCoverEveryCommand` fails when a command or
flag is missing from it). Highlights:

| Area | Commands |
|---|---|
| Discovery | `health`, `stats`, `task list`, `task show <id>`, `lokictl help` |
| Jobs | `job run <task> [args...] [--field k=v] [--wait] [--follow] [--timeout D]`, `job copy/resume <id> [--wait] [--follow] [--timeout D]`, `job list [--state S]`, `job get/wait/logs/cancel/remove <id>`, `job clear --yes` |
| Workflows | `workflow list/get/create/update/delete/run`, `workflow run-adhoc --dag FILE\|-` |
| Library queries | `media query [--tag ... --visual ... --similar ...]`, `media search/similar/visual/image-search/metadata/tags/delete` |
| Media data | `media describe <path> (--text D\|--clear)`, `media transcript <path> [--text T\|--clear]`, `media rate <path> [--elo E --views N --wins N --losses N]`, `media thumbs <path> [--regenerate]`, `media generate <path> --type T [--wait]` |
| Library bookkeeping | `media move <from> <to> [--prefix] [--dry-run]` (you moved the file; re-point the DB), `media forget <path> --yes` (drop every DB reference, keep the file) |
| Missing media | `media cleanup [--dir D] [--dry-run] [--max-missing-percent N] [--skip-orphans] [--detach]` — forget media whose files no longer exist, removing every tag/embedding/face/battle row that names them and sweeping dangling rows; scans then re-verifies before deleting, skips offline volumes, and holds back any volume with more than N% missing (default 50) |
| Thumbnail cache | `media thumbnail-cleanup [--dir D] [--dry-run] [--db PATH]... [--no-discover] [--manifest FILE] [--detach]` — delete cached thumbnails no library still names. Every SQLite database beside the configured one counts as a library sharing the cache; `--db` adds others, `--manifest` writes per-file ownership, `--dir` scopes to a folder (recursive) and names the media path of every file it removes |
| Embeddings index | `index status/models/rebuild`, `index missing [--model M]`, `index get <path> [--vector]`, `index delete <path> --yes`, `index prune --yes`, `index embed [args...] [--wait]` |
| Raw SQL (read-only) | `db query "SELECT ..." [--arg V]`, `db tables`, `db schema [table]` |
| Taxonomy | `taxonomy [--category C]`, `tag create/delete/rename/move/assign/unassign/assign-bulk/unassign-bulk`, `tag list/count/weight/has/timestamp/assignment-weight`, `category create/delete/rename/count` |
| Dependencies | `deps status`, `deps download <model-id> --wait`, `deps verify/delete` |
| GPU worker pools | `pool status` (`-o table`), `pool check`, `pool strategy local\|pool\|hybrid [--fallback=true]`, `pool route <engine> local\|pool\|pool:NAME\|hybrid\|default`, `pool add-node <pool> <url> --token T`, `pool add-runpod <pool> --endpoint ID --api-key K [--transfer-root LABEL]`, `pool remove <pool> [url]`, `pool warm <pool\|all> <engine>...`, `pool create runpod-pods\|runpod-serverless <pool> [--count N --gpu T --network-volume V --preload E]` (needs `$RUNPOD_API_KEY`), `pool scale <pool> --count N`, `pool destroy <pool>` — see `tools/worker/README.md` |
| Server admin | `config get`, `config set --json '{...}'`, `fs list/scan`, `upload <file>...`, `whoami` |
| API keys | `key create --name N [--username U] [--save]`, `key list`, `key revoke --id N` |
| Escape hatch | `api <METHOD> <path> [--body JSON\|@file\|-]` — any endpoint, auth attached |

Destructive commands (`job clear`, `media delete`, `media forget`, `tag delete`,
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

lokictl job copy <id> --follow     # re-run a job and tail it until it finishes
lokictl job resume <id> --follow   # same for a paused job
```

`--follow` attaches to the stream before the job is created, so a fast job's
first lines aren't lost, and exits when the job finishes (exit 3 if it errored
or was cancelled). `--wait` waits without streaming.

**Forget media that no longer exists** (two passes: scan, then re-verify; a
volume with most of its items missing is held back, not purged):

```sh
lokictl media cleanup --dry-run                        # report only, whole library
lokictl media cleanup --dir "D:/photos/2019" --dry-run   # bounded test
lokictl media cleanup                                  # forget missing items + sweep dangling sidecar rows
lokictl media cleanup --max-missing-percent 100        # a volume is gone for good: purge it deliberately
```

**Reclaim thumbnail cache space** (the task log streams to stderr; the
final job JSON on stdout carries the whole log in `stdout`):

```sh
lokictl media thumbnail-cleanup --dry-run --manifest C:/tmp/thumbs.tsv   # who needs what: ownership report + per-file TSV
lokictl media thumbnail-cleanup --dir "D:/photos/2019" --dry-run   # bounded test: lists each file and the media path it belonged to
lokictl media thumbnail-cleanup --dir "D:/photos/2019"             # delete, that folder only
lokictl media thumbnail-cleanup --dry-run                          # whole cache, report only
lokictl media thumbnail-cleanup                                    # whole cache, delete
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

**Group tags inside a category** (a visual level between category and tag —
e.g. a `Series` category with one group per show holding per-season tags;
grouping never changes what a tag query matches):

```sh
lokictl group list --category Series -o table               # groups in order
lokictl group add "Dirty Pair" --category Series "Dirty Pair - Season 1" "Dirty Pair - Season 2"
lokictl group weight "Dirty Pair" --category Series --weight 1   # reorder
lokictl group rename "Dirty Pair" "The Dirty Pair" --category Series
lokictl group remove "Dirty Pair - Specials"                 # ungroup a tag
lokictl group hide "Dirty Pair" --category Series           # out of the panel (Settings → Show Hidden Tag Groups)
lokictl group show "Dirty Pair" --category Series
lokictl group view "Dirty Pair" list --category Series      # card | list | auto (follow the category)
lokictl group delete "Dirty Pair" --category Series --yes    # tags stay, ungrouped
lokictl category grouping Series on                          # show/hide groups
```

Assigning a tag that doesn't exist yet creates it in `--category`; an existing
tag keeps its own category (naming another is a 409).

**Deleting a category is guarded.** It always previews what it would remove,
and refuses to untag anything unless you also pass `--allow-nonempty`:

```sh
lokictl category delete Anime --dry-run                      # tags, groups, assignments, media
lokictl category delete Anime --yes                          # only if it owns no assignments
lokictl category delete Anime --yes --allow-nonempty         # deliberately untag its media
```

**Write metadata directly, or have AI generate it:**

```sh
lokictl media describe "C:/pics/x.jpg" --text "Two dogs on a beach"
lokictl media transcript "C:/vids/talk.mp4"                  # read
lokictl media generate "C:/vids/talk.mp4" --type transcript --wait
lokictl media rate "C:/pics/x.jpg" --elo 1600
```

**You moved files on disk — make the library agree.** The move re-points every
reference (media row, tags, embedding, faces, scan markers, battle log) in one
transaction; nothing on disk is touched. `--dry-run` performs the move and
rolls it back, so its counts are exactly what a real run would change:

```sh
lokictl media move "C:/pics/old.jpg" "D:/archive/old.jpg"      # one file
lokictl media move "C:/pics/2023" "D:/archive/2023" --prefix --dry-run
lokictl media move "C:/pics/2023" "D:/archive/2023" --prefix   # the folder
```

`--prefix` is segment-aligned: moving `2023` never drags `2023extra` along. A
destination that already belongs to another item is a 409 with the offending
paths — a move will not merge two items. To drop an item from the library
while leaving the file where it is (or to clean up after a file that's gone):

```sh
lokictl media forget "C:/pics/gone.jpg" --yes
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
  over SSE and does not persist it. It exits when the job finishes.
- `db query` accepts a single `SELECT`/`WITH` statement; the server enforces
  read-only at the SQLite level (`query_only`). Rows are capped (default
  1000, max 10000) with `"truncated": true` when clipped.
- `--arg` values are sent as strings; SQLite type affinity coerces them for
  comparisons against numeric columns.
- Job-input tokens containing double quotes are rejected (the server's
  splitter has no escape syntax) — pass such values with `--field name=value`.
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
