# Custom Description Prompt — Design

**Date:** 2026-05-30
**Status:** Approved (verbal — user asked to proceed without further review)

## Problem

The metadata component's "Generate Description" / "Regenerate Description" button
posts a job to the media server that runs a vision-LLM call using the prompt
text stored in `appconfig.Config.DescribePrompt`. Users currently have no way to
try a different prompt without editing the server config (which then affects
every subsequent description job globally).

We want a per-invocation override: the user can optionally type a custom prompt
in the React UI before generating, and that prompt is used for *this* job only.
The stored config default is unchanged. If the user provides no override, the
default is used exactly as today.

## Scope

In scope:
- React UI: collapsible "Custom prompt" panel in `GenerateDescription`
- Frontend session memory of the last-used custom prompt
- Frontend fetch of the current default prompt (for placeholder)
- HTTP `/create` extension: optional `fields` map on the request body
- Server `metadata` task: new `prompt` `TaskOption` consumed by the description path
- Wire-through: `processDescriptionForFile` → `describeFileWithOllama` → `callVisionLLM`
- New JSON endpoint: `GET /api/prompts/describe` returning the current default

Out of scope:
- Per-file persistence of custom prompts (no DB schema change)
- Custom prompts for autotag / other metadata types (description only)
- Changes to the Go HTML config page or admin Jobs UI
- Changes to the `jobqueue.Job` / `Workflow` data model (no new `Fields` field)
- Mirror of any of the above in the Electron-only IPC path (the feature is web-mode driven via HTTP)

## Architecture

### Frontend (React)

`src/renderer/components/metadata/generate-description.tsx`:

- Add a collapsible panel rendered alongside the existing button. Toggle is a
  small text link ("Customize prompt" / "Hide prompt"). When expanded, a
  `<textarea>` is shown with:
  - `placeholder` set to the default prompt fetched from
    `GET /api/prompts/describe` (cached at module scope after first fetch)
  - `value` controlled by component state, seeded from
    `lastCustomPrompt` (module-level `let` in this file)
- A toggle is also displayed in the inline ("Regenerate") variant via the same
  component — both variants get the panel.
- On submit:
  - Read the textarea value, `.trim()` it.
  - If empty, send the existing request body unchanged.
  - If non-empty, write the trimmed value to `lastCustomPrompt`, and include it
    in the POST body as `fields: { prompt: <value> }`.
- Cleanup: when the user clears the textarea, leave `lastCustomPrompt` as the
  last non-empty value (so toggling the panel back open shows it again). Only
  overwrite on submit.

Module-level singleton (suitable for "session-only, in-memory, cross-file"):

```ts
let lastCustomPrompt = '';
let cachedDefaultPrompt: string | null = null;
```

No XState change. No new context. No localStorage.

### HTTP API

`POST /create` request body becomes:

```json
{
  "input": "metadata --type description --apply all --overwrite \"<path>\"",
  "fields": { "prompt": "describe only the foreground subject" }
}
```

`fields` is optional. If present, the server appends `--<key> <value>` for each
entry to the parsed `Arguments` slice *before* enqueueing the workflow task.
This keeps the existing `ParseOptions` flow as the single source of option
parsing — no special-casing in the metadata task.

Escape behavior: since `fields` values are appended directly (not re-parsed by
`ParseCommand`), they may contain any characters including double quotes,
newlines, etc. No quoting required.

Empty-string values in `fields` are ignored (defensive: prevents accidental
override with empty text). The frontend already trims and drops empties before
sending; this is a server-side belt-and-suspenders.

`GET /api/prompts/describe` returns:

```json
{ "prompt": "Please describe this image, paying special attention to ..." }
```

Reads from `appconfig.Get().DescribePrompt`. Admin-role gated (same middleware
as `/config`).

### Server task

`media-server/tasks/media_metadata.go`:

- Add a new entry to `metadataOptions`:
  ```go
  {Name: "prompt", Label: "Custom Description Prompt", Type: "string",
   Description: "Override the configured describe prompt for this run"},
  ```
- In `metadataTask`, read `customPrompt, _ := opts["prompt"].(string)`.
- Pass `customPrompt` through to `processDescriptionForFile`.

`media-server/tasks/metadata_ops.go`:

- `processDescriptionForFile` gains a `customPrompt string` parameter.
- Pass through to `describeFileWithOllama(ctx, filePath, model, customPrompt)`.
- `describeFileWithOllama` gains a `customPrompt string` parameter.
- Pass through to `callOllamaVision(ctx, resizedPath, model, customPrompt)`.
- `callOllamaVision` gains a `customPrompt string` parameter. Its body becomes:
  ```go
  prompt := customPrompt
  if strings.TrimSpace(prompt) == "" {
      prompt = appconfig.Get().DescribePrompt
  }
  return callVisionLLM(timeoutCtx, imagePath, prompt)
  ```

Because `describeFileWithOllama`'s signature changes, every existing caller
must be updated to pass an extra arg:
- `media-server/tasks/metadata_ops.go` line 69 (inside `generateDescriptions`,
  which is currently unused — pass `""`, do not extend its own signature).
- `media-server/tasks/lora_dataset.go` line 210 (LoRA dataset generation —
  pass `""` so it continues to use the config default).

`media-server/tasks/media_ingest.go` continues to enqueue `metadata` jobs
without a `--prompt` flag, so those jobs continue to use the configured default.

## Data flow

```
React: GenerateDescription
   │  fields.prompt = trimmed textarea value (if non-empty)
   ▼
POST /create  { input, fields }
   │  createJobHandler appends --prompt <value> to Arguments
   ▼
jobqueue.AddWorkflow → metadataTask
   │  ParseOptions reads opts["prompt"]
   ▼
processDescriptionForFile(... customPrompt)
   │
   ▼
describeFileWithOllama(... customPrompt)
   │
   ▼
callOllamaVision(... customPrompt)
   │  prompt = customPrompt or appconfig.Get().DescribePrompt
   ▼
callVisionLLM(ctx, image, prompt)
```

## Error handling

- Frontend: if `GET /api/prompts/describe` fails (e.g. job server down), fall
  back to a generic placeholder string. Don't block the panel from opening.
- Server: invalid `fields` shape in `/create` body (e.g. non-object, non-string
  values) → 400 Bad Request with a short error message. Server defensively
  ignores empty-string values inside `fields`.
- Server task: if `prompt` option is the empty string (which it will be when
  not specified), behavior is identical to today.

## Testing

- Go: extend `media-server/tasks/registry_test.go` if option schemas are checked
  there. Add a `metadata_ops_test.go` case (or extend existing) verifying that
  `processDescriptionForFile` calls into the vision path with the override when
  provided and the config default when empty. Mocking `callVisionLLM` may
  require a small refactor to a package-level function variable; if too
  invasive, settle for an integration-style test that exercises `ParseOptions`
  with `--prompt foo` and asserts the option value reaches the right callee
  (verified via a thin seam).
- Go: a handler test for `POST /create` confirming a `fields` map is appended
  as `--key value` arguments to the queued job.
- Go: a handler test for `GET /api/prompts/describe` confirming it returns the
  current `DescribePrompt`.
- React: extend `src/__tests__/` with a render test for `GenerateDescription`
  verifying the panel toggle, default-prompt placeholder fetch, and that a
  trimmed non-empty value is sent under `fields.prompt`.

## Migration / compatibility

- `/create` callers that don't send `fields` see no change.
- `media_ingest.go` and any other internal callers that build `metadata` jobs
  programmatically continue to omit `--prompt`, so they continue to use the
  config default.
- No DB migration. No persisted state change.
- No change to the Electron-only IPC path (`invoke('update-description', ...)`
  is a separate code path for manual text edits; it doesn't run the LLM).

## File-level change list

**New:**
- `media-server` route + handler for `GET /api/prompts/describe` (in `main.go`,
  `main_darwin.go`, `main_linux.go` — same handler body, registered in each
  platform file per the repo's split-build convention).

**Modified:**
- `media-server/main.go` + `main_darwin.go` + `main_linux.go`:
  - `CreateJobHandlerRequest` gets `Fields map[string]string \`json:"fields,omitempty\"\``
  - `createJobHandler` appends `--<k> <v>` to args for each non-empty field
- `media-server/tasks/media_metadata.go`: new `prompt` option, threaded into
  `processDescriptionForFile`.
- `media-server/tasks/metadata_ops.go`: add `customPrompt` parameter to
  `processDescriptionForFile`, `describeFileWithOllama`, and
  `callOllamaVision`; resolve to default inside `callOllamaVision`. Update the
  existing call inside the dead `generateDescriptions` to pass `""` so the
  file continues to compile.
- `media-server/tasks/lora_dataset.go`: update the `describeFileWithOllama`
  call site (LoRA path) to pass `""` so it continues to use the config default.
- `src/renderer/components/metadata/generate-description.tsx`: panel UI, fetch
  default, module-level session memory, `fields.prompt` in POST body.
- `src/renderer/components/metadata/generate-description.css`: styles for the
  toggle link, panel, and textarea.

## Open questions

None — user authorized blanket approval on remaining shape choices.
