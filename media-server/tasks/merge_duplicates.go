package tasks

import (
	"fmt"
	"path/filepath"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
)

// Accepting every pending duplicate group at once.
//
// The Duplicates panel is built for one-group-at-a-time review; this task is
// the bulk "I trust these" path. It walks every PENDING group and merges each
// one into its preferred keeper (highest resolution, then largest file —
// media.PreferredDuplicateKeeper) with media.MergeInto — the same merge the panel's Merge
// button and the dedupe task use — so the anchor gains the other members'
// tags, embeddings and transcript and the other members are deleted from
// disk and the database. Excluded members are never merged; dismissed groups
// are never touched. Fully merged groups disappear on their own (their member
// rows go with the deleted media).
//
// Long-running by design: a library with tens of thousands of groups deletes
// tens of thousands of files. Progress is per group, pause/resume and cancel
// work between groups (every merge already committed stays committed), and
// the Duplicates panel refreshes as groups vanish.

var mergeDuplicatesOptions = []TaskOption{
	{Name: "min-similarity", Label: "Minimum Similarity (%)", Type: "number", Default: 0.0,
		Description: "Only merge groups whose loosest member is at least this similar to the anchor; 0 = every pending group. Use it to auto-accept the certain (100%) groups and leave the rest for review"},
	{Name: "dry-run", Label: "Dry Run", Type: "bool",
		Description: "Report how many groups and files would be merged without touching anything"},
}

// mergeDuplicatesLogGroups caps the per-group log lines before the task
// switches to periodic totals.
const mergeDuplicatesLogGroups = 100

const mergeDuplicatesBroadcastInterval = 1500 * time.Millisecond

func mergeDuplicatesTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	db := q.Db
	if db == nil {
		q.PushJobStdout(j.ID, "Error: database connection not available")
		q.ErrorJob(j.ID)
		return fmt.Errorf("database connection not available")
	}
	opts := ParseOptions(j, mergeDuplicatesOptions)
	minPct, _ := opts["min-similarity"].(float64)
	dryRun, _ := opts["dry-run"].(bool)
	minScore := 0.0
	if minPct > 0 {
		minScore = float64(duplicateThresholdScore(minPct))
	}

	ids, err := media.PendingDuplicateGroupIDs(db)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error listing pending groups: %v", err))
		q.ErrorJob(j.ID)
		return err
	}
	if len(ids) == 0 {
		q.PushJobStdout(j.ID, "No pending duplicate groups to merge")
		q.CompleteJob(j.ID)
		return nil
	}
	if minPct > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("Merging pending duplicate groups at ≥ %.1f%% similarity (%d pending in total)", minPct, len(ids)))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Merging all %d pending duplicate group(s) into their best copies (highest resolution, then largest file)", len(ids)))
	}
	if dryRun {
		q.PushJobStdout(j.ID, "Dry run: nothing will be merged or deleted")
	} else {
		q.PushJobStdout(j.ID, "Each group's other active members are deleted after their tags, embeddings and transcript move to the kept copy")
	}

	var merged, skippedLoose, skippedEmpty, filesDeleted, failures int
	var tagsGained, embGained, facesRemoved int64
	var wouldDelete int
	_ = q.SetJobProgress(j.ID, 0, len(ids))
	lastBroadcast := time.Time{}
	lastLog := time.Now()
	for i, id := range ids {
		select {
		case <-ctx.Done():
			broadcastDuplicatesUpdated(0, 0)
			q.PushJobStdout(j.ID, fmt.Sprintf("Canceled after %d group(s) — every merge already done is kept; run again to continue", merged))
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}
		if q.PauseRequested(j.ID) {
			broadcastDuplicatesUpdated(0, 0)
			q.PushJobStdout(j.ID, fmt.Sprintf("Paused at group %d/%d — resume to continue", i, len(ids)))
			return jobqueue.ErrPaused
		}
		_ = q.SetJobProgress(j.ID, i, len(ids))

		g, found, err := media.GetDuplicateGroup(db, id)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not load group #%d: %v", id, err))
			failures++
			continue
		}
		if !found || g.Status != media.DuplicateStatusPending {
			continue // merged or dismissed by hand while this ran
		}
		if minScore > 0 && g.MinScore < minScore {
			skippedLoose++
			continue
		}
		// Keeper: the shared rule (highest resolution, then largest file) —
		// the same default the panel highlights and the endpoint applies.
		keeper, _, err := media.DuplicateGroupKeeper(db, g)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: group #%d: could not pick a keeper: %v", id, err))
			failures++
			continue
		}
		var sources []string
		for _, m := range g.Members {
			if !m.Excluded && m.Path != keeper {
				sources = append(sources, m.Path)
			}
		}
		if len(sources) == 0 {
			skippedEmpty++
			continue
		}
		if dryRun {
			wouldDelete += len(sources)
			merged++
			if merged <= mergeDuplicatesLogGroups {
				q.PushJobStdout(j.ID, fmt.Sprintf("  Would merge group #%d: keep %s, delete %d", id, filepath.Base(keeper), len(sources)))
			}
			continue
		}

		res, err := media.MergeInto(ctx, db, keeper, sources)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: group #%d merge failed: %v", id, err))
			failures++
			continue
		}
		merged++
		filesDeleted += len(res.Deleted)
		failures += len(res.Failed)
		tagsGained += res.Tags
		embGained += res.Embeddings
		facesRemoved += res.FacesRemoved
		if merged <= mergeDuplicatesLogGroups {
			q.PushJobStdout(j.ID, fmt.Sprintf("  Group #%d: kept %s, deleted %d, gained %d tag(s) %d embedding(s)%s",
				id, filepath.Base(keeper), len(res.Deleted), res.Tags, res.Embeddings, failNote(len(res.Failed))))
		} else if merged == mergeDuplicatesLogGroups+1 {
			q.PushJobStdout(j.ID, "  (further groups are counted in the periodic totals below)")
		}
		now := time.Now()
		if now.Sub(lastLog) >= findDuplicatesLogInterval {
			lastLog = now
			q.PushJobStdout(j.ID, fmt.Sprintf("  %d/%d groups — %d merged, %d file(s) deleted, %d failure(s)", i+1, len(ids), merged, filesDeleted, failures))
		}
		if now.Sub(lastBroadcast) >= mergeDuplicatesBroadcastInterval {
			lastBroadcast = now
			broadcastDuplicatesUpdated(0, 0)
		}
	}
	_ = q.SetJobProgress(j.ID, len(ids), len(ids))

	if dryRun {
		q.PushJobStdout(j.ID, fmt.Sprintf("Dry run complete: %d group(s) would merge, deleting %d file(s); %d group(s) below the similarity floor, %d with nothing to merge",
			merged, wouldDelete, skippedLoose, skippedEmpty))
		q.CompleteJob(j.ID)
		return nil
	}
	broadcastDuplicatesUpdated(0, 0)
	if facesRemoved > 0 {
		broadcastPeopleUpdated([]string{})
	}
	q.PushJobStdout(j.ID, fmt.Sprintf(
		"Merge complete: %d group(s) merged, %d file(s) deleted, %d tag(s) and %d embedding(s) consolidated, %d skipped below the similarity floor, %d failure(s)",
		merged, filesDeleted, tagsGained, embGained, skippedLoose, failures))
	q.CompleteJob(j.ID)
	return nil
}

func failNote(n int) string {
	if n == 0 {
		return ""
	}
	return fmt.Sprintf(", %d could not be deleted", n)
}
