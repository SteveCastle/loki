package main

import (
	"encoding/json"
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

// textModelsHandler lists the GGUF models on this machine for the Config
// page's text-model picker (GET /api/text-models).
func textModelsHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Use GET", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"models": tasks.LocalTextModels()})
}
