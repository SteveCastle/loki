package main

// Auto-scheduler HTTP surface. Untagged so every platform main registers the
// same routes:
//
//	GET  /api/scheduler        → current status (mode/state/reason/signals)
//	POST /api/scheduler/mode   → {"mode":"off"|"auto"} (persisted)
//	POST /api/scheduler/run    → one-shot forced run (old "Run everything")

import (
	"encoding/json"
	"net/http"
)

func schedulerStatusHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		if autoSched == nil {
			http.Error(w, "scheduler not running", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(autoSched.Status())
	}
}

func schedulerModeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		if autoSched == nil {
			http.Error(w, "scheduler not running", http.StatusServiceUnavailable)
			return
		}
		var req struct {
			Mode string `json:"mode"`
		}
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid JSON body", http.StatusBadRequest)
			return
		}
		if req.Mode != "off" && req.Mode != "auto" {
			http.Error(w, `mode must be "off" or "auto"`, http.StatusBadRequest)
			return
		}
		autoSched.SetMode(req.Mode)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(autoSched.Status())
	}
}

func schedulerRunHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		if autoSched == nil {
			http.Error(w, "scheduler not running", http.StatusServiceUnavailable)
			return
		}
		autoSched.ForceRun()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(autoSched.Status())
	}
}
