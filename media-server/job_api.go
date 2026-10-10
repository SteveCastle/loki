package main

// job_api.go — GET /api/job/{id}
//
// One job as JSON for pollers that are not the server's own pages: state,
// item progress, registered output files and the tail of its log. Lowkey
// Studio follows a splat-training job this way (cross-origin, API key in a
// header) and downloads the result through /media/file once it completes.
// Shared by all three platform mains.

import (
	"encoding/json"
	"net/http"
	"strconv"
)

func jobAPIHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		tail := 20
		if v, err := strconv.Atoi(r.URL.Query().Get("tail")); err == nil && v >= 0 && v <= 500 {
			tail = v
		}
		job, lines, ok := deps.Queue.JobSnapshot(r.PathValue("id"), tail)
		if !ok {
			http.Error(w, "job not found", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":             job.ID,
			"command":        job.Command,
			"state":          job.State,
			"progress_done":  job.ProgressDone,
			"progress_total": job.ProgressTotal,
			"output_files":   job.OutputFiles,
			"created_at":     job.CreatedAt,
			"completed_at":   job.CompletedAt,
			"log":            lines,
		})
	}
}
