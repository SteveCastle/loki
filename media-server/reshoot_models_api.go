package main

// reshoot_models_api.go — GET /api/reshoot/models
//
// The extra loki-reshoot diffusion models a job can pick (config.json
// reshootModels), for the Transform flow's Model control. H3 and FastH3
// Omni are always there and not listed.

import (
	"net/http"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/tasks"
)

func reshootModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, map[string]any{"models": tasks.ReshootModelChoices(appconfig.Get())})
}
