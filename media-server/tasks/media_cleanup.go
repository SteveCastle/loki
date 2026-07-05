package tasks

import (
	"fmt"
	"strings"
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
		// Cancellation (the Pause button) is not a failure — mark the job
		// cancelled so it can be restarted, not errored.
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Task was canceled after removing %d items — progress is saved", result.MediaItemsRemoved))
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, fmt.Sprintf("Error during cleanup: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	if result.MediaItemsRemoved == 0 {
		q.PushJobStdout(j.ID, "No orphaned media items found - database is clean!")
	} else {
		q.PushJobStdout(j.ID, "Cleanup completed successfully:")
		q.PushJobStdout(j.ID, fmt.Sprintf("- Removed %d media items from database", result.MediaItemsRemoved))
		q.PushJobStdout(j.ID, fmt.Sprintf("- Removed %d tag associations", result.TagsRemoved))
	}

	// Items on offline volumes are deliberately left alone: an unmounted drive
	// stats exactly like a deleted file, and purging a whole volume's library
	// because it was unplugged is unrecoverable.
	if result.SkippedUnavailable > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf(
			"WARNING: skipped %d missing items because their volume(s) are offline: %s — reconnect the drive(s) and run cleanup again to process them",
			result.SkippedUnavailable, strings.Join(result.UnavailableRoots, ", ")))
	}

	if len(result.Errors) > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Note: %d errors occurred during cleanup", len(result.Errors)))
	}

	q.CompleteJob(j.ID)
	return nil
}
