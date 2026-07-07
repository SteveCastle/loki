package main

// Job pause/resume handlers. Intentionally free of build tags so every
// platform main registers the same routes.
//
// Pause is graceful: RequestPause sets a flag the task's item loop polls
// between items, so the in-flight item finishes and commits before the job
// parks in the Paused state. All per-item work is already durable, so a
// paused (or cancelled) job never loses completed items; Resume re-queues
// the job and the task's skip-existing checks (or the persisted progress
// offset for overwrite runs) continue where it left off.

import (
	"net/http"
)

func pauseJobHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		if err := deps.Queue.RequestPause(r.PathValue("id")); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("Job pause requested"))
	}
}

func resumeJobHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		if err := deps.Queue.ResumeJob(r.PathValue("id")); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("Job resumed"))
	}
}
