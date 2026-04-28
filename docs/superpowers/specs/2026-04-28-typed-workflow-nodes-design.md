# Typed Workflow Nodes — Design

**Status:** Draft
**Date:** 2026-04-28
**Scope:** `media-server` (Go server + server-rendered Drawflow editor at `media-server/renderer/templates/editor.go.html`). The React SPA is unaffected; it has no workflow editor.

## Problem

The current workflow editor models every node as having exactly one input port and one output port. Inputs flow through `WorkflowTask.Input` (a single string), which the runtime fills by concatenating upstream `OutputFiles`. Per-node configuration lives in a parallel `Arguments []string` channel as CLI-style `--flag value` tokens.

This produces three concrete problems:

1. **No typed contracts.** A node receiving "input" doesn't know whether it's a list of file paths, a list of URLs, or a search query string. Today this is implicit per-task and inconsistent (compare `ingest` vs. `queries`).
2. **Options are dead ends.** A user cannot drive any option (e.g. `--scale 0.5`, `--folder /tmp`) from the output of another node. Options are literal-only.
3. **No editor-level validation.** Any output port can connect to any input port; type mismatches surface only at runtime, often as silent misbehavior.

We want a true typed node-editing system: every option is a wireable input; every task declares the contract for what it consumes and produces; the editor validates wires by type before runtime.

## Goals

- Tasks declare a structured input contract (one entry per accepted parameter, including the "primary" data input) and a single typed output.
- Every declared input is renderable as both a literal field (default behavior) and a wireable port (when an upstream node's output type is compatible).
- Editor refuses incompatible wires; runtime coerces compatible-but-not-equal types.
- Existing saved workflows continue to load and run.
- Existing task functions (`TaskFn` implementations) need no changes; the runtime synthesizes their legacy `j.Input` / `j.Arguments` from the new binding model.

## Non-Goals

- Multiple output ports per node. Out of scope; one declared output type per task.
- A new graph engine. We keep Drawflow.
- React SPA workflow editor. Out of scope.
- Cycles, conditionals, or expression languages. Bindings are literal-or-wire only.
- Streaming / partial outputs. Outputs remain materialized as the existing `OutputFiles` list.

## Type Lattice

A small fixed set of types. The first iteration:

| Type          | Wire payload                          | Notes                                                            |
|---------------|---------------------------------------|------------------------------------------------------------------|
| `string`      | UTF-8 string                          | Free text, queries, single labels                                |
| `number`      | float64                               | All numerics; UI may further constrain (int / range)             |
| `bool`        | bool                                  |                                                                  |
| `enum`        | string                                | Constrained by `Choices`                                         |
| `multi-enum`  | []string                              | Constrained by `Choices`                                         |
| `path`        | string                                | Filesystem path                                                  |
| `path-list`   | []string                              | The most common output; produced by ingest, ffmpeg-*, save, etc. |
| `url`         | string                                |                                                                  |
| `url-list`    | []string                              |                                                                  |
| `ref-list`    | []string                              | Heterogeneous paths and URLs (e.g. `ingest` accepts either)      |
| `json`        | arbitrary JSON                        | Escape hatch; UI shows a textarea                                |

### Coercion rules

The runtime applies these when a wire feeds a port of a different declared type:

- Scalar → matching list: wrap in a length-1 list (`path` → `path-list`, `url` → `url-list`).
- Scalar → `ref-list`: wrap (any of `path` / `url`).
- `path-list` ↔ `url-list` ↔ `ref-list`: pass-through (treated as opaque string lists at the wire level).
- `path` / `url` / `number` / `bool` → `string`: lossless format.
- `string` → `number`: parse; runtime error if invalid (job goes to `Error` state with a clear message).
- `string` → `bool`: only `"true"`/`"false"`/`"1"`/`"0"`; otherwise runtime error.
- All other combinations: editor refuses the connection at draw time.

### Fan-in

- For list-typed inputs (`path-list`, `url-list`, `ref-list`, `multi-enum`): multiple incoming wires concatenate in connection order.
- For scalar-typed inputs: at most one incoming wire. The editor refuses additional connections.

## Task Contract Schema

In `media-server/tasks/options.go`:

```go
type TaskInput struct {
    Name        string   `json:"name"`
    Label       string   `json:"label"`
    Type        string   `json:"type"`              // one of the type-lattice keys
    Choices     []string `json:"choices,omitempty"` // for enum / multi-enum
    Default     any      `json:"default,omitempty"`
    Required    bool     `json:"required,omitempty"`
    Primary     bool     `json:"primary,omitempty"` // marks the conventional first input
    Description string   `json:"description,omitempty"`
}

type TaskOutput struct {
    Type        string `json:"type"`
    Description string `json:"description,omitempty"`
}

type Task struct {
    ID     string      `json:"id"`
    Name   string      `json:"name"`
    Inputs []TaskInput `json:"inputs"`
    Output TaskOutput  `json:"output"`
    Fn     TaskFn      `json:"-"`
}
```

`Options []TaskOption` is removed in favor of `Inputs []TaskInput`. Migration:

- The current "primary" input (today: `j.Input`) becomes a leading `TaskInput` entry with `Primary: true`. Type chosen per task (most are `path-list` or `ref-list`; `queries` declares `string`).
- Each former `TaskOption` becomes a `TaskInput` with `Primary: false`. Field names map directly; `Type` carries forward (`"string"` → `string`, etc.).
- Every task gains an `Output` declaration; default `{Type: "path-list"}` for tasks that currently call `RegisterOutputFile`, `string` or `json` for the few that don't.

`RegisterTask`'s signature changes from `(id, name, options, fn)` to `(id, name, inputs, output, fn)`. All call sites in `media-server/tasks/registry.go` are updated in the same change.

## Workflow Storage Schema

In `media-server/jobqueue/jobqueue.go`:

```go
type Binding struct {
    Kind  string `json:"kind"`            // "literal" | "wire"
    Value any    `json:"value,omitempty"` // for "literal"
    From  string `json:"from,omitempty"`  // for "wire": source task ID
    Port  string `json:"port,omitempty"`  // for "wire": always "output" in v1
}

type WorkflowTask struct {
    ID           string             `json:"id"`
    Command      string             `json:"command"`
    Bindings     map[string]Binding `json:"bindings"`     // keyed by input name
    Dependencies []string           `json:"dependencies"`
    PosX         float64            `json:"pos_x,omitempty"`
    PosY         float64            `json:"pos_y,omitempty"`

    // Deprecated; populated only when reading legacy rows. Never written.
    Arguments []string `json:"arguments,omitempty"`
    Input     string   `json:"input,omitempty"`
}
```

`Dependencies` is now derived from `Bindings` (the set of `from` IDs across all `wire` bindings). It remains in the persisted shape so the existing scheduling code (`canClaim`, `ClaimJob`) is undisturbed.

### Backward-compat read path

When `GetWorkflow` (or any other reader) deserializes a stored `WorkflowTask` that has empty `Bindings` but non-empty `Input` / `Arguments`:

1. Look up the task's current `Inputs` declaration by `Command`.
2. Set the primary input's binding to `{Kind: "literal", Value: <Input>}`.
3. Run the existing CLI-flag parser (`ParseOptions` logic, repurposed) to extract `--name value` pairs from `Arguments`, then set each one as `{Kind: "literal", Value: <parsed>}`.
4. Walk `Dependencies`: for the primary input only, append a `wire` binding from each dep (preserves the implicit upstream-feeds-primary behavior).
5. The migrated bindings are returned in-memory but **not** written back; old workflows continue to live in the legacy shape until the user re-saves them. Re-saves persist the new shape and stop populating `Input`/`Arguments`.

## Runtime: Resolving Bindings to a Job

`AddWorkflow` (and `RunWorkflow`'s submission step) translates each `WorkflowTask` to a `Job`. The new step happens at job-claim time, in `ClaimJob`, replacing the current `inputBuilder` block:

For each input on the task:
- If the binding is `literal`: use `Value` directly.
- If the binding is `wire`: fetch the source `Job`'s output payload (currently always `OutputFiles`; `string` outputs are recovered from a single-element output payload; `json` outputs from a new `OutputJSON` field — see below) and apply coercion to the target type.
- Multiple wires into a list-typed input concatenate.

The resolved input map is then materialized to the legacy `Job.Input` and `Job.Arguments` so existing task functions don't need to change:

- The input flagged `Primary: true` is joined into a newline-delimited string and stored on `Job.Input`. (List types newline-join their elements; scalar primary types are stored verbatim.)
- All other inputs are emitted as `--name value` tokens in `Job.Arguments` using the same encoding the editor uses today, so `ParseOptions` (rebound to operate over `[]TaskInput`) round-trips them.

### Output payloads

The current output channel is `Job.OutputFiles []string` plus a parallel `SourceFiles []string`. To support non-list output types:

- `path-list` / `url-list` / `ref-list` outputs continue to use `OutputFiles` exactly as today. No task changes.
- `path` / `url` outputs use `OutputFiles` constrained to length 1.
- `string` / `number` / `bool` / `json` outputs land in a new `Job.OutputValue any` field (persisted as JSON in the SQLite jobs table). Tasks that produce these call a new `q.RegisterOutputValue(jobID, value)` helper. None of the existing tasks need this in v1; it's a forward hook.

## Editor (`editor.go.html`)

### Node rendering

`/tasks` already returns each task's option list. Extend it to return `inputs` and `output` from the new `Task` schema. The editor's `buildNodeHTML` becomes:

- `editor.addNode(name, inputs.length, 1, ...)` — one input port per declared input; one output port.
- For each input: a labeled row with the port handle on the left, the existing inline editor (text / number / checkbox / select / multi-checkbox) on the right.
- The output port carries a CSS class derived from `output.type` (e.g. `port-path-list`) for color-coding.
- Each input port carries a class derived from its declared type and the task ID + input name, so the connection-validation hook can read both endpoints' types.

### Connection validation

Drawflow exposes `connectionStart` and `connectionCreated` events. We hook `connectionCreated`:

1. Read the source node's output type and the target node's input type from the DOM port classes.
2. Run the same coercion rules used by the runtime (kept in a tiny shared JS helper that mirrors the Go list).
3. If incompatible: call `editor.removeSingleConnection(...)` immediately and surface a status-bar message.
4. If the target is scalar and already has one connection: refuse the new one.
5. If accepted: add a CSS class to the target's input row to dim the inline literal editor (it's overridden by the wire).

When a connection is removed: re-enable the inline editor.

### Export shape

`exportDAG` produces the new `WorkflowTask` shape directly:

- For each Drawflow node, emit `bindings` keyed by input name. Each input is a `literal` binding from the inline editor's value unless that input has at least one incoming connection, in which case it becomes a `wire` binding (or array of wires for fan-in).
- `dependencies` is computed as the unique set of source node IDs across all wires.
- Old `arguments` and `input` fields are no longer emitted.

### Backward-compat load

`loadWorkflow` may receive legacy-shape DAGs (with `arguments` and `input` but no `bindings`). The same migration logic from the Go read path is mirrored in JS (reuses the `taskOptionsMap` / `taskInputsMap` already cached) so the editor can render and re-save them in the new shape.

## API Surface

Existing endpoints keep their paths. Behavior changes:

- `GET /tasks`: response now includes `inputs` (replacing `options`) and `output`. We keep emitting `options` as a deprecated alias for one release so any external consumers don't break instantly; the editor switches to `inputs`.
- `POST /workflow`, `POST /workflows/create`, `PUT /workflows/{id}`: accept the new `WorkflowTask` shape. For one release, also accept the legacy `{arguments, input}` shape and apply the read-path migration before storing.
- `POST /workflow` (run-now): unchanged URL; payload is the new shape.

## Validation

`validateDAG` (in `media-server/jobqueue/workflows.go`) gains:

- For every wire binding, the source task ID exists in the DAG.
- For every task, every `Required: true` input has either a non-empty literal binding or at least one wire binding.
- For every wire binding, the source's declared output type is coercible to the target's input type per the lattice. Tasks are looked up via `tasks.GetTasks()`; an unknown command is a validation error.
- For scalar inputs, at most one wire binding.

These checks run on `CreateWorkflow` and `UpdateWorkflow`, the same place existing structural checks run.

## Module Layout

New / changed files:

- `media-server/tasks/options.go` — replace `TaskOption` with `TaskInput`/`TaskOutput`/`Task` schema; rewrite `ParseOptions` to operate over `[]TaskInput` (logic largely unchanged).
- `media-server/tasks/registry.go` — update `RegisterTask` signature and all call sites; add `Output` to every existing task; add `Primary: true` to the conventional leading input.
- `media-server/tasks/types.go` *(new)* — type lattice constants + Go-side coercion helpers shared by the runtime and validation.
- `media-server/jobqueue/jobqueue.go` — add `Binding`, change `WorkflowTask`, change `ClaimJob`'s input-construction block, add `OutputValue` field + persistence.
- `media-server/jobqueue/workflows.go` — extend `validateDAG`; add legacy-shape migration in `GetWorkflow`.
- `media-server/main*.go` (and platform variants) — `/tasks` handler returns new shape; workflow endpoints accept legacy and new shapes.
- `media-server/renderer/templates/editor.go.html` — multi-port nodes, connection validation, export shape, dim-on-wire styling.
- Tests: `tasks/options_test.go`, `jobqueue/workflows_test.go`, `jobqueue/jobqueue_core_test.go` get cases for binding resolution, coercion, migration, and validation.

## Risks and Trade-offs

- **Drawflow port limitations.** Drawflow's per-node port count is fixed at creation. Since task contracts are static per `Command`, this is fine. If a user changes a task's input list in the future, existing saved workflows referencing that task may need a re-save; the editor will surface this as a "stale node — refresh" warning.
- **Multi-checkbox `multi-enum` widget** is awkward to wire because the literal editor is a set of checkboxes that can't fully be replaced by a single string output. Mitigation: when a `multi-enum` input is wired, hide all checkboxes and show a "wired from <node>" pill; on disconnect, restore the checkboxes.
- **Legacy migration is one-way at first read.** Old DAGs render in the new shape immediately. Users who never re-save them retain the legacy stored bytes. We accept this; no automatic mass-rewrite.
- **`json` output coercion** is intentionally narrow — it only feeds a `json`-typed input. We're not building a JSONPath / expression language in v1.

## Test Plan

- Unit tests for the coercion table (every cell, including expected refusals).
- Unit tests for `ParseOptions` on `[]TaskInput`, including each type.
- Round-trip test: a DAG built from the new shape, persisted, reloaded, executed via `RunWorkflow`, asserts the resolved `Job.Input` and `Job.Arguments` match expected.
- Migration test: a stored DAG in the legacy shape (`Input` + `Arguments`) loads as the new shape with equivalent bindings.
- Validation test: scalar fan-in is refused; wire to incompatible type is refused; missing required input is refused.
- Editor smoke test (Playwright): drag two nodes, wire output to a non-primary input, confirm the inline editor dims; export and verify the JSON shape.
