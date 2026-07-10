package tasks

import (
	"fmt"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

func removeFromDB(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

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

	q.PushJobStdout(j.ID, "Starting removal of media items from database")

	result, err := media.RemoveItemsFromDB(ctx, q.Db, paths)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error removing media items: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Processed %d paths", len(result.ProcessedPaths)))
	q.PushJobStdout(j.ID, fmt.Sprintf("Removed %d tag associations", result.TagsRemoved))
	q.PushJobStdout(j.ID, fmt.Sprintf("Removed %d media items from database", result.MediaItemsRemoved))
	// Index eviction happens inside RemoveItemsFromDB via the media removal
	// hook (registry.go init); only output-file registration is needed here.
	for _, p := range result.ProcessedPaths {
		q.RegisterOutputFile(j.ID, p)
	}
	if result.MediaItemsRemoved == 0 {
		q.PushJobStdout(j.ID, "No matching media items found in database")
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Successfully removed %d media items and %d tag associations", result.MediaItemsRemoved, result.TagsRemoved))
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
