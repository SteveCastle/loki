package tasks

import (
	"fmt"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

func removeFromDB(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Error loading media paths for query: %v", err))
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
	} else {
		pathsStr := strings.TrimSpace(j.Input)
		if pathsStr == "" {
			q.PushJobStdout(j.ID, "No paths provided for removal")
			q.CompleteJob(j.ID)
			return nil
		}
		rawPaths := strings.Split(pathsStr, "\n")
		for _, path := range rawPaths {
			cleanPath := strings.TrimSpace(path)
			if cleanPath != "" {
				paths = append(paths, cleanPath)
			}
		}
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
