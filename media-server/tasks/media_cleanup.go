package tasks

import (
	"fmt"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

func cleanUpFn(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	q.PushJobStdout(j.ID, "Starting database cleanup - finding and removing media items that don't exist in file system")

	progressCallback := func(found, removed int) {
		q.PushJobStdout(j.ID, fmt.Sprintf("Progress: Found %d orphaned items, removed %d so far", found, removed))
	}

	result, err := media.StreamingCleanupNonExistentItems(ctx, q.Db, progressCallback)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error during cleanup: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	if result.MediaItemsRemoved == 0 {
		q.PushJobStdout(j.ID, "No orphaned media items found - database is clean!")
	} else {
		q.PushJobStdout(j.ID, "Cleanup completed successfully:")
		q.PushJobStdout(j.ID, fmt.Sprintf("- Processed %d orphaned media items", len(result.ProcessedPaths)))
		q.PushJobStdout(j.ID, fmt.Sprintf("- Removed %d media items from database", result.MediaItemsRemoved))
		q.PushJobStdout(j.ID, fmt.Sprintf("- Removed %d tag associations", result.TagsRemoved))
	}

	if len(result.Errors) > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Note: %d errors occurred during cleanup (but cleanup continued)", len(result.Errors)))
	}

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
