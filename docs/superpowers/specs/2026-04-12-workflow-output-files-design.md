# Workflow Output Files & Temp Directory

## Problem

When jobs are chained in a workflow, the downstream job receives its input by concatenating all `Stdout` lines from parent jobs. This mixes ffmpeg diagnostics, progress messages, and log lines with actual output file paths, causing downstream jobs to fail when they try to parse non-path lines as file inputs.

Additionally, intermediate workflow steps write output files next to the originals. A chain like grayscale -> blur -> resize produces `_grayscale`, `_grayscale_blurred`, and `_grayscale_blurred_resized` files, leaving unwanted intermediate artifacts behind.

## Design

Two changes to the job system:

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

### 2. Temp Directory for Intermediate Steps

When a job produces files and is not the terminal step in a workflow, its output should go to a `.loki-temp` directory sibling to the source file. Only the terminal step writes to the final location.

**Job struct changes:**

```go
type Job struct {
    // ... existing fields ...
    HasDependents bool `json:"has_dependents"`
}
```

**Setting HasDependents:** In `AddWorkflow` and `RunWorkflow`, after building all jobs, scan the full task list to determine which job IDs appear in any dependency list. Mark those jobs with `HasDependents = true`. Standalone jobs (not part of a workflow) always have `HasDependents = false`.

**Temp directory location:** `.loki-temp/<jobID>/` sibling to the source file's directory. Using the job ID as a subfolder prevents collisions between concurrent workflows operating on files in the same directory.

Example: source file `Z:\gallery-dl\video.mp4` with job ID `abc-123` produces intermediate output at `Z:\gallery-dl\.loki-temp\abc-123\video_grayscale.mp4`.

**runFFmpegOnFiles changes:** The `buildArgs` callback currently receives `(abs, dir, name, ext)`. Add awareness of the temp directory:

- Before calling `buildArgs`, if `j.HasDependents` is true, compute `tempDir = filepath.Join(dir, ".loki-temp", j.ID)` and create it. Pass `tempDir` as `dir` to `buildArgs` so the output path lands in the temp directory.
- If `j.HasDependents` is false (terminal step), pass the original `dir` as today.

This way all existing `buildArgs` callbacks (which use `dir` to construct output paths) automatically write to the correct location without individual changes.

**Cleanup:** When a job completes or errors, if it has a `.loki-temp/<jobID>` directory, that directory should be cleaned up. Two cleanup points:

1. **On job completion:** In `CompleteJob`, after marking complete, if the job's `OutputFiles` contain paths under `.loki-temp`, schedule cleanup. However, we should NOT clean up until all dependents have consumed the files.
2. **On workflow completion:** When a terminal job (no dependents) completes, walk its dependency chain and remove all `.loki-temp/<depJobID>/` directories. This is the safe point — all downstream jobs have finished.
3. **On error:** If any job in a workflow errors, clean up all `.loki-temp` directories for that workflow's jobs. The workflow is dead, temp files are waste.

The cleanup logic lives in the Queue, triggered from `CompleteJob` and `ErrorJob`. To find related jobs in a workflow, walk the dependency graph from the completing job.

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
| remove | No output files |
| cleanup | No output files |
| wait | No output files |

## Implementation Order

1. **Add OutputFiles to Job struct and DB schema** — field, column, migration, serialization/deserialization
2. **Add RegisterOutputFile method** — Queue method with lock, DB persist
3. **Add HasDependents to Job struct and DB schema** — field, column, migration
4. **Update AddWorkflow to set HasDependents** — scan dependency lists
5. **Update RunWorkflow to set HasDependents** — same scan after ID remapping
6. **Update ClaimJob** — prefer OutputFiles over Stdout
7. **Add temp directory logic to runFFmpegOnFiles** — create .loki-temp dir, pass to buildArgs
8. **Add cleanup logic** — in CompleteJob/ErrorJob, walk dependency graph, remove temp dirs
9. **Update all tasks to call RegisterOutputFile** — ffmpeg presets, ffmpeg custom, hls, move, metadata, autotag, ingest variants, lora-dataset
10. **Update tests** — workflow tests for OutputFiles propagation, temp dir creation/cleanup
