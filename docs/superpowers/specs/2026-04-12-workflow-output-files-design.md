# Workflow Output Files & Temp Directory

## Problem

When jobs are chained in a workflow, the downstream job receives its input by concatenating all `Stdout` lines from parent jobs. This mixes ffmpeg diagnostics, progress messages, and log lines with actual output file paths, causing downstream jobs to fail when they try to parse non-path lines as file inputs.

Additionally, intermediate workflow steps write output files next to the originals. A chain like grayscale -> blur -> resize produces `_grayscale`, `_grayscale_blurred`, and `_grayscale_blurred_resized` files, leaving unwanted intermediate artifacts behind.

## Design

Three changes to the job system:

### 1. Dedicated OutputFiles Channel

Add an `OutputFiles []string` field to the `Job` struct — a dedicated list of file paths that a task explicitly registers as its output, separate from diagnostic stdout.

**Job struct changes:**

```go
type Job struct {
    // ... existing fields ...
    OutputFiles []string `json:"output_files"`
}
```

**Database schema:** Add `output_files TEXT` column to the jobs table (JSON array, same pattern as `stdout` and `dependencies`).

**New Queue method:**

```go
func (q *Queue) RegisterOutputFile(id string, path string) error
```

Appends a path to the job's `OutputFiles` slice. Acquires the lock, updates the job, persists to DB. Does NOT broadcast to SSE (these are internal plumbing, not user-facing log lines). Tasks continue to call `PushJobStdout` for all logging — `RegisterOutputFile` is only for file paths that downstream jobs should receive.

**ClaimJob changes:** Replace the current Stdout-based input construction (lines 520-533 of jobqueue.go) with:

```go
for _, depID := range job.Dependencies {
    if depJob, ok := q.Jobs[depID]; ok {
        source := depJob.OutputFiles
        if len(source) == 0 {
            // Backwards compatibility: legacy tasks that don't register output files
            source = depJob.Stdout
        }
        for _, line := range source {
            if inputBuilder.Len() > 0 {
                inputBuilder.WriteString("\n")
            }
            inputBuilder.WriteString(line)
        }
    }
}
```

This means: prefer `OutputFiles` when available, fall back to `Stdout` for tasks that haven't been updated (backwards compatible).

### 2. All Processing Nodes Write to Temp Directories

Every file-producing task always writes its output to a `.loki-temp` directory. There is no concept of "terminal" vs "intermediate" nodes for output location — all processing nodes are pure transforms that write to temp. The user adds an explicit **Save File** task to control where and how final files are persisted.

**Temp directory location:** `.loki-temp/<jobID>/` sibling to the source file's directory. Using the job ID as a subfolder prevents collisions between concurrent workflows operating on files in the same directory.

Example: source file `Z:\gallery-dl\video.mp4` with job ID `abc-123` produces output at `Z:\gallery-dl\.loki-temp\abc-123\video_grayscale.mp4`.

**When temp directories are used:** Any job that is part of a workflow (has dependencies OR has dependents). Standalone jobs (submitted individually, not part of a workflow) continue to write output next to the source file as they do today — no behavior change for single-job execution.

**Job struct changes:**

```go
type Job struct {
    // ... existing fields ...
    WorkflowID string `json:"workflow_id"` // Non-empty when job is part of a workflow
}
```

`WorkflowID` is set by `AddWorkflow`/`RunWorkflow` to a shared identifier for all jobs in the workflow. Tasks check `j.WorkflowID != ""` to decide whether to use temp directories.

**runFFmpegOnFiles changes:** Before calling `buildArgs`, if the job has a `WorkflowID`, compute `tempDir = filepath.Join(dir, ".loki-temp", j.ID)` and create it. Pass `tempDir` as `dir` to `buildArgs` so the output path lands in the temp directory. For standalone jobs, pass the original `dir` as today.

This way all existing `buildArgs` callbacks (which use `dir` to construct output paths) automatically write to the correct location without individual changes.

**Cleanup:** Temp directories are cleaned up when the workflow is done:

1. **On Save File completion:** After the save task finishes (successfully or not), it cleans up the `.loki-temp` directories for all ancestor jobs in the workflow by walking the dependency graph.
2. **On workflow error:** If any job in a workflow errors, clean up all `.loki-temp` directories for that workflow's jobs. Walk the job graph using `WorkflowID` to find all related jobs.
3. **Orphan cleanup:** On server startup, scan for any `.loki-temp` directories whose job IDs no longer exist in the queue (crashed/killed workflows) and remove them.

### 3. Save File Task

A new `save` task that is the explicit final step in any file-producing workflow. It takes files from its parent's `OutputFiles` and writes them to a user-controlled destination.

**Registration:**

```go
RegisterTask("save", "Save File", saveOptions, saveTask)
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `destination` | enum | `original` | Where to save: `original` (same directory as the original input file), `directory` (specific directory) |
| `directory` | string | `""` | Target directory (only used when destination=directory) |
| `conflict` | enum | `suffix` | How to handle existing files: `overwrite`, `suffix` (append _1, _2, etc.), `skip` |
| `suffix` | string | `""` | Custom suffix to append to filename before extension (e.g. `_edited`). Empty = use the suffix from the processing task |
| `flatten` | bool | `false` | When using a directory destination, flatten all files into it (ignore relative paths) |

**Behavior:**

For each file in the parent's `OutputFiles`:

1. **Determine the destination path:**
   - `original`: Strip the `.loki-temp/<jobID>/` segment from the path to recover the original directory. The filename keeps whatever suffix the processing task gave it (e.g. `video_grayscale.mp4`), unless a custom `suffix` is specified, in which case replace the processing suffix with the custom one applied to the original filename.
   - `directory`: Move/copy to the specified directory. If `flatten` is false, preserve relative path structure from the temp dir.

2. **Handle conflicts** based on the `conflict` option:
   - `overwrite`: Replace existing file.
   - `suffix`: If `output.mp4` exists, try `output_1.mp4`, `output_2.mp4`, etc.
   - `skip`: Log and skip if file exists.

3. **Move the file** (not copy) from the temp dir to the destination — same-drive rename is atomic and instant. If cross-drive, fall back to copy + delete.

4. **Register the final path** via `RegisterOutputFile` so further downstream jobs (if any) get the real paths.

5. **Clean up** the `.loki-temp` directories for all ancestor jobs in the workflow.

**Save to original directory example:**

Workflow: `ffmpeg-grayscale` -> `ffmpeg-blur` -> `save`

- Input: `Z:\gallery-dl\video.mp4`
- Grayscale outputs to: `Z:\gallery-dl\.loki-temp\<job1>\video_grayscale.mp4`
- Blur reads that, outputs to: `Z:\gallery-dl\.loki-temp\<job2>\video_grayscale_blurred.mp4`
- Save (destination=original, suffix=_edited) writes: `Z:\gallery-dl\video_edited.mp4`
- Save (destination=original, suffix="", conflict=suffix) writes: `Z:\gallery-dl\video_grayscale_blurred.mp4` (keeps processing name, appends _1 if exists)

### Task OutputFiles Registration

Every task that produces files calls `q.RegisterOutputFile(j.ID, path)`. Tasks that pass through their input files (metadata, autotag) register the original paths so downstream jobs receive them.

| Task | OutputFiles behavior |
|------|---------------------|
| ffmpeg presets | Register each output file path |
| ffmpeg custom | Register each output file path |
| hls | Register master playlist path |
| move | Register each destination path |
| metadata | Register each processed source path (passthrough) |
| autotag | Register each processed source path (passthrough) |
| ingest (local) | Register each ingested file path |
| ingest (youtube) | Register each downloaded file path |
| ingest (gallery) | Register each downloaded file path |
| ingest (discord) | Register each downloaded file path |
| lora-dataset | Register each output dataset file path |
| save | Register each final saved path |
| remove | No output files |
| cleanup | No output files |
| wait | No output files |

## Implementation Order

1. **Add OutputFiles and WorkflowID to Job struct and DB schema** — fields, columns, migration, serialization/deserialization
2. **Add RegisterOutputFile method** — Queue method with lock, DB persist
3. **Update AddWorkflow and RunWorkflow to set WorkflowID** — generate a workflow run ID, assign to all jobs
4. **Update ClaimJob** — prefer OutputFiles over Stdout for input construction
5. **Add temp directory logic to runFFmpegOnFiles** — check WorkflowID, create .loki-temp dir, pass to buildArgs
6. **Implement Save File task** — save task with all options, conflict resolution, temp dir cleanup
7. **Add cleanup logic** — in ErrorJob, walk workflow graph via WorkflowID, remove temp dirs. Startup orphan scan.
8. **Update all tasks to call RegisterOutputFile** — ffmpeg presets, ffmpeg custom, hls, move, metadata, autotag, ingest variants, lora-dataset
9. **Update tests** — workflow tests for OutputFiles propagation, temp dir creation/cleanup, save task behavior
