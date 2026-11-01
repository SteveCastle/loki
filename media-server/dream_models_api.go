package main

// dream_models_api.go — GET /api/dream/models[?refresh=1]
//
// The diffusion models loki-dream can run (tasks.DreamModels: the engine's
// own --list-models), for the Generate / Reimagine model picker. A job picks
// one by file name in its `model` field.

import (
	"net/http"
	"time"

	"github.com/stevecastle/shrike/features"
	"github.com/stevecastle/shrike/tasks"
)

func dreamModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	if !features.Dream {
		writeJSON(w, tasks.DreamModelList{Models: []tasks.DreamModel{}, Note: "this server is built without image generation", Fetched: time.Now()})
		return
	}
	refresh := r.URL.Query().Get("refresh") == "1"
	writeJSON(w, tasks.DreamModels(r.Context(), refresh))
}
