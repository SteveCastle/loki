package jobqueue

// items.go — the path→job index.
//
// Jobs are created with either an explicit path list or a search query, so
// "which jobs are touching this file?" has no single answer at creation time.
// This index tracks it in two phases:
//
//   - Definition time (AddJob / CopyJob / DB load): an injected
//     ItemsResolverFunc parses the job's definition into concrete paths when
//     that's possible without the database (explicit path lists). Query jobs
//     return nil here — their membership is unknowable until they resolve.
//   - Resolution time: when a per-item task resolves its input to a concrete
//     item list (tasks.runItemOps), it calls SetJobItems with the real list,
//     replacing the definition-time entry. This is when query jobs become
//     path-queryable.
//
// Only ACTIVE jobs (pending / in-progress / paused) are indexed; terminal
// transitions drop the entry. That keeps the index self-cleaning, bounds its
// memory to in-flight work, and means it needs no persistence: it rebuilds
// from the jobs table at load and re-fills at claim time for query jobs.

import (
	"path/filepath"
	"runtime"
	"strings"
)

// ItemsResolverFunc maps a job definition to the concrete media paths it will
// operate on, when that is derivable without database access. Return nil when
// membership can't be known yet (query jobs) — the task itself will report
// the resolved list via SetJobItems once it runs.
type ItemsResolverFunc func(command string, arguments []string, input string) []string

var itemsResolver ItemsResolverFunc

// SetItemsResolver installs the definition-time items resolver. Call once at
// startup (alongside SetHostResolver, before any AddJob / loadJobsFromDB).
// nil disables definition-time indexing; SetJobItems still works.
func SetItemsResolver(fn ItemsResolverFunc) {
	itemsResolver = fn
}

// maxIndexedItemsPerJob bounds the index's memory. A query job spanning the
// whole library (millions of paths) is not worth per-path lookups — the jobs
// list already shows it — so beyond this cap the job simply isn't indexed.
const maxIndexedItemsPerJob = 25000

// normalizeItemKey canonicalizes a path for index lookups so the path a
// client asks about matches the path a task resolved: cleaned, and
// case-folded on Windows (NTFS is case-insensitive). s3:// identities are
// compared verbatim.
func normalizeItemKey(p string) string {
	if strings.HasPrefix(p, "s3://") {
		return p
	}
	p = filepath.Clean(p)
	if runtime.GOOS == "windows" {
		p = strings.ToLower(p)
	}
	return p
}

// isActiveState reports whether a job in this state belongs in the index.
func isActiveState(s JobState) bool {
	return s == StatePending || s == StateInProgress || s == StatePaused
}

// setJobItemsLocked replaces the indexed item set for a job. Must be called
// with mu held (or before the queue is shared, e.g. during load).
func (q *Queue) setJobItemsLocked(jobID string, paths []string) {
	q.dropJobItemsLocked(jobID)
	if len(paths) == 0 || len(paths) > maxIndexedItemsPerJob {
		return
	}
	if q.jobItems == nil {
		q.jobItems = make(map[string][]string)
		q.pathJobs = make(map[string]map[string]struct{})
	}
	keys := make([]string, 0, len(paths))
	for _, p := range paths {
		k := normalizeItemKey(p)
		if k == "" {
			continue
		}
		set, ok := q.pathJobs[k]
		if !ok {
			set = make(map[string]struct{})
			q.pathJobs[k] = set
		}
		if _, dup := set[jobID]; dup {
			continue
		}
		set[jobID] = struct{}{}
		keys = append(keys, k)
	}
	q.jobItems[jobID] = keys
}

// dropJobItemsLocked removes a job from the index. Must be called with mu held.
func (q *Queue) dropJobItemsLocked(jobID string) {
	keys, ok := q.jobItems[jobID]
	if !ok {
		return
	}
	for _, k := range keys {
		if set, ok := q.pathJobs[k]; ok {
			delete(set, jobID)
			if len(set) == 0 {
				delete(q.pathJobs, k)
			}
		}
	}
	delete(q.jobItems, jobID)
}

// indexJobFromDefinitionLocked indexes a job from its definition via the
// installed resolver. OriginalInput is used (not Input) because Input is
// rewritten at claim time with dependency outputs; when that happens the
// task's SetJobItems call supersedes this entry anyway.
func (q *Queue) indexJobFromDefinitionLocked(job *Job) {
	if itemsResolver == nil || !isActiveState(job.State) {
		return
	}
	q.setJobItemsLocked(job.ID, itemsResolver(job.Command, job.Arguments, job.OriginalInput))
}

// SetJobItems records the concrete item list a job is operating on. Tasks
// call it once their input resolves (this is how query jobs enter the index).
// Ignored for unknown or already-terminal jobs.
func (q *Queue) SetJobItems(jobID string, paths []string) {
	q.mu.Lock()
	defer q.mu.Unlock()
	job, ok := q.Jobs[jobID]
	if !ok || !isActiveState(job.State) {
		return
	}
	q.setJobItemsLocked(jobID, paths)
}

// GetJobsForPath returns copies of all ACTIVE jobs whose indexed item set
// contains the given path, in queue order. Query jobs that haven't started
// yet (membership unresolved) and jobs over the index cap are not included.
func (q *Queue) GetJobsForPath(path string) []Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	set, ok := q.pathJobs[normalizeItemKey(path)]
	if !ok || len(set) == 0 {
		return nil
	}
	out := make([]Job, 0, len(set))
	for _, id := range q.JobOrder {
		if _, hit := set[id]; !hit {
			continue
		}
		if job, ok := q.Jobs[id]; ok {
			out = append(out, *job)
		}
	}
	return out
}
