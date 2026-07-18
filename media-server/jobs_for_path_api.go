package main

// jobs_for_path_api.go — GET /api/jobs/for-path?path=<media path>
//
// Answers "which jobs are currently operating on this file?" from the
// jobqueue's path→job index (jobqueue/items.go). Only ACTIVE jobs (pending,
// in-progress, paused) are returned: path-list jobs appear the moment they
// are created; query jobs appear once they resolve their input at claim
// time. The SPA uses this to swap per-item action buttons (e.g. Generate
// Transcript) for a live status indicator, then follows the job over the
// /stream SSE events by id. Shared by all three platform mains.

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

type jobForPathSummary struct {
	ID            string            `json:"id"`
	Command       string            `json:"command"`
	Arguments     []string          `json:"arguments"`
	State         jobqueue.JobState `json:"state"`
	ProgressDone  int               `json:"progress_done"`
	ProgressTotal int               `json:"progress_total"`
	CreatedAt     time.Time         `json:"created_at"`
}

func jobsForPathHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		path := r.URL.Query().Get("path")
		if path == "" {
			http.Error(w, "missing path parameter", http.StatusBadRequest)
			return
		}

		jobs := deps.Queue.GetJobsForPath(path)
		out := make([]jobForPathSummary, 0, len(jobs))
		for _, j := range jobs {
			out = append(out, jobForPathSummary{
				ID:            j.ID,
				Command:       j.Command,
				Arguments:     j.Arguments,
				State:         j.State,
				ProgressDone:  j.ProgressDone,
				ProgressTotal: j.ProgressTotal,
				CreatedAt:     j.CreatedAt,
			})
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"path": path,
			"jobs": out,
		})
	}
}
