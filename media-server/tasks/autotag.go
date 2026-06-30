package tasks

import (
	"fmt"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

func autotagTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	fromQuery := false
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "Failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
		fromQuery = true
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(paths)))
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "No image path provided")
			q.CompleteJob(j.ID)
			return nil
		}
		paths = parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input", len(paths)))
	}

	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	if err := EnsureCategoryExists(q.Db, "Suggested", 0); err != nil {
		q.PushJobStdout(j.ID, "Failed to ensure category: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}

	// Tag via a pool of persistent workers (the tagger model loads once per
	// worker, not per image). Pool size + ONNX threads come from the autotag
	// performance config.
	processed, skipped, err := runAutotagPool(ctx, j, q, paths, fromQuery)
	if err != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, "Tagging failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d tagged, %d skipped", processed, skipped))
	q.CompleteJob(j.ID)
	return nil
}
