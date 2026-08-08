package tasks

import (
	"fmt"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// removeLogInterval throttles the per-batch stdout lines. Every PushJobStdout
// rewrites the job row with its whole (growing) stdout array, so a line per
// 500-path batch would itself become the slow part of a large removal. Progress
// bars come from SetJobProgress (throttled and persisted with a targeted
// UPDATE); the log lines are for the human reading the job later.
const removeLogInterval = 2 * time.Second

func removeFromDB(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	q.PushJobStdout(j.ID, "Resolving items to remove")

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		q.PushJobStdout(j.ID, "Failed to resolve input: "+rerr.Error())
		q.ErrorJob(j.ID)
		return rerr
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("Query: %s", res.Query))
	}
	paths := res.Paths
	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No paths provided for removal")
		q.CompleteJob(j.ID)
		return nil
	}

	total := len(paths)
	q.PushJobStdout(j.ID, fmt.Sprintf("Starting removal of %d media items from database", total))
	// Publish the total before any work happens so the UI shows 0/N immediately
	// rather than an indeterminate spinner for the whole run.
	q.SetJobProgress(j.ID, 0, total)

	// Output files are registered once at the end, in bulk: each path removed is
	// still reported for downstream workflow steps, but registering them one at
	// a time re-serializes the whole list per call (quadratic — the dominant
	// cost of a large removal before this).
	removed := make([]string, 0, total)
	lastLog := time.Now()

	result, err := media.RemoveItemsFromDBStream(ctx, q.Db, paths, func(b media.RemovalBatch) {
		removed = append(removed, b.Paths...)
		q.SetJobProgress(j.ID, b.Done, b.Total)
		if b.Done == b.Total || time.Since(lastLog) >= removeLogInterval {
			lastLog = time.Now()
			q.PushJobStdout(j.ID, fmt.Sprintf(
				"Removed %d/%d items (%d media rows, %d tag associations)",
				b.Done, b.Total, b.MediaItemsRemoved, b.TagsRemoved))
		}
	})

	// Whatever got committed is durable, so register it either way — a
	// cancelled run still reports the items it actually removed.
	q.RegisterOutputFiles(j.ID, removed)

	if err != nil {
		// Cancellation (the Pause button) is not a failure: the committed
		// batches stay removed and re-running the job picks up the rest.
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf(
				"Task was canceled after removing %d of %d items — progress is saved",
				len(removed), total))
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error removing media items: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Index eviction happens inside RemoveItemsFromDBStream via the media
	// removal hook (registry.go init), per batch.
	q.PushJobStdout(j.ID, fmt.Sprintf("Processed %d paths", len(result.ProcessedPaths)))
	if result.MediaItemsRemoved == 0 {
		q.PushJobStdout(j.ID, "No matching media items found in database")
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Successfully removed %d media items and %d tag associations", result.MediaItemsRemoved, result.TagsRemoved))
	}

	// A cancel that lands after the last batch committed still has to mark the
	// job cancelled — completing it would make the Pause button look broken.
	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}

	q.CompleteJob(j.ID)
	return nil
}
