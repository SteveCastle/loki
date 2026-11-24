package main

// loras_api.go — GET /api/loras
//
// The LoRA files the Generate / Reimagine picker offers (tasks.ListLoras: the training runs folder and a `loras`
// folder next to the engines' models), each with the architecture it applies to, its trigger word and training
// step. A dream job takes the chosen ones as its `loras` field (one `path:scale` per line).

import (
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

func lorasHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, tasks.ListLoras())
}

// GET /api/lora-runs — the training runs in the training folder (settings, progress, resumable checkpoints and
// their preview images), for the Train LoRA dialog's "Continue a run".
func loraRunsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, map[string]any{"runs": tasks.ListLoraRuns(), "dir": tasks.LoraTrainingDir()})
}
