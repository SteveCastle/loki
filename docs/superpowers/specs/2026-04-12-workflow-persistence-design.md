# Workflow Persistence Design

## Overview

Add the ability to save, recall, and run reusable multi-step task workflows (DAGs) in the media-server. Saved workflows appear as actions in the context palette and can be managed through the API and the existing Drawflow visual editor.

## Database Schema

New `workflows` table in the job queue database (same SQLite DB as the `jobs` table):

```sql
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    dag TEXT NOT NULL  -- JSON blob
)
```

The `dag` column stores a JSON array of node definitions using the existing `WorkflowTask` shape:

```json
[
  { "id": "step1", "command": "autotag", "arguments": ["--apply", "all"], "input": "", "dependencies": [] },
  { "id": "step2", "command": "metadata", "arguments": ["--type", "transcript", "--apply", "all"], "input": "", "dependencies": ["step1"] },
  { "id": "step3", "command": "metadata", "arguments": ["--type", "description", "--apply", "all"], "input": "", "dependencies": ["step1"] }
]
```

- `id` fields are stable identifiers within the DAG template for linking dependencies
- Root nodes (empty `dependencies`) receive run-time input when the workflow is executed
- The format is identical to what the `/workflow` endpoint and Drawflow editor already produce

## Output Chaining

The existing job queue mechanism handles data flow between nodes:

1. Root nodes receive the run-time input (query string or file list)
2. Each task executes and writes output to stdout via `PushJobStdout` (paths, URLs, values)
3. When a downstream node is claimed, its effective `Input` = its own `input` field + stdout from all completed dependency jobs
4. This is already implemented in `ClaimJob` in `jobqueue.go` — no changes needed

Tasks that want to feed downstream nodes should write output paths/URLs/values to stdout. This is an existing contract.

## API Endpoints

Six new endpoints, all requiring admin auth middleware:

### `GET /workflows`

List all saved workflows. Returns id and name only (no DAG blob).

**Response:**
```json
[
  { "id": "abc-123", "name": "Full Processing Pipeline" },
  { "id": "def-456", "name": "Ingest and Tag" }
]
```

### `GET /workflows/{id}`

Get a single workflow with full DAG.

**Response:**
```json
{
  "id": "abc-123",
  "name": "Full Processing Pipeline",
  "dag": [
    { "id": "step1", "command": "autotag", "arguments": [], "input": "", "dependencies": [] },
    { "id": "step2", "command": "metadata", "arguments": ["--type", "transcript", "--apply", "all"], "input": "", "dependencies": ["step1"] }
  ]
}
```

### `POST /workflows`

Create a new workflow.

**Request:**
```json
{
  "name": "Full Processing Pipeline",
  "dag": [ ... ]
}
```

**Response:** `201 Created`
```json
{
  "id": "abc-123",
  "name": "Full Processing Pipeline"
}
```

Validates:
- `name` is non-empty and unique
- `dag` is a valid JSON array of workflow tasks
- All dependency references within the DAG resolve to node IDs that exist in the same DAG
- All commands reference registered tasks

### `PUT /workflows/{id}`

Update an existing workflow's name and/or DAG.

**Request:**
```json
{
  "name": "Updated Name",
  "dag": [ ... ]
}
```

Same validation as POST. Returns `200 OK` with updated workflow.

### `DELETE /workflows/{id}`

Delete a workflow. Returns `204 No Content`.

### `POST /workflows/{id}/run`

Instantiate and execute a saved workflow.

**Request:**
```json
{
  "input": "--query64=dGFnOmxhbmRzY2FwZQ=="
}
```

Or with a file list:
```json
{
  "input": "/path/to/file1.jpg\n/path/to/file2.jpg"
}
```

**Execution flow:**
1. Load the saved DAG from the database
2. Generate fresh UUIDs for each node (replacing template IDs)
3. Remap all dependency references to the new UUIDs
4. Inject the `input` value into all root nodes (nodes with empty `dependencies`)
5. Call the existing `AddWorkflow` with the instantiated tasks
6. Return the live job IDs

**Response:** `201 Created`
```json
{
  "ids": ["live-uuid-1", "live-uuid-2", "live-uuid-3"]
}
```

## Context Palette Integration

The context palette gains a dynamic "Workflows" section below the existing hardcoded actions.

**On palette open** (when `serverAvailable && authToken`):
- Fetch `GET /workflows` to get the list of saved workflows
- Render each as a row with the workflow name and a "Run" button
- Cache the list while the palette is open (no polling)

**On "Run" click:**
- Build the input from the current target context (same query64 logic used by existing actions)
- POST to `/workflows/{id}/run` with that input
- Close palette, job toasts appear via existing SSE/ToastSystem

The existing hardcoded actions (Transcripts, Tags, Descriptions) remain as quick shortcuts for single-task operations. Saved workflows are for multi-step pipelines.

If no saved workflows exist, the section is hidden.

## Drawflow Editor Updates

The existing editor at `/editor` already builds DAGs and posts to `/workflow` for immediate execution. Add save/load functionality:

### Save
- "Save" button prompts for a workflow name
- POSTs the current Drawflow canvas (exported and converted to DAG format) to `POST /workflows`
- On success, the editor tracks the saved workflow ID for subsequent updates

### Load
- "Load" dropdown fetches `GET /workflows` to populate the list
- Selecting a workflow fetches `GET /workflows/{id}` for the full DAG
- Converts the DAG back to Drawflow node format and imports it via `editor.import()`
- Sets the editor into "editing existing workflow" mode (shows Update/Delete buttons)

### Update
- Visible when editing a previously loaded workflow
- PUTs the current canvas to `PUT /workflows/{id}`

### Delete
- Visible when editing a previously loaded workflow
- DELETEs via `DELETE /workflows/{id}` after confirmation
- Clears the canvas

### Run (existing)
- The existing "Run" button continues to work — it runs the current canvas DAG immediately without saving

### DAG Format Conversion

The editor already converts Drawflow export → DAG format for the "Run" function. The same logic is reused for Save. The reverse (DAG → Drawflow import) is needed for Load:
- Each DAG node becomes a Drawflow node positioned in a layout
- Dependency edges become Drawflow connections (output → input)
- Auto-layout positions nodes left-to-right based on dependency depth

## Files Modified

### Server (Go)
- `media-server/jobqueue/jobqueue.go` — Add `workflows` table creation in schema init, add CRUD methods for workflows
- `media-server/main.go` — Add 6 new HTTP handler functions and route registrations
- No changes to existing job execution, dependency resolution, or output chaining

### Client (TypeScript)
- `src/renderer/components/controls/context-palette.tsx` — Fetch and render saved workflows as actions
- `src/renderer/components/controls/context-palette.css` — Style the workflows section

### Editor (HTML/JS)
- `media-server/renderer/templates/editor.go.html` — Add save/load/update/delete UI and DAG↔Drawflow conversion
