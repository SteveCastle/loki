package main

// transform_enhance_api.go — POST /api/transform/enhance-prompt
//
// Rewrites a Transform prompt for its target model with loki-scribe
// (tasks.EnhancePrompt). Only ever on request: nothing is queued, and the
// answer goes back into the prompt box for the user to keep, edit or undo.

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

func transformEnhanceHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	if r.Method == http.MethodGet {
		_ = json.NewEncoder(w).Encode(map[string][]string{"models": tasks.PromptGuideModels()})
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "Use POST", http.StatusMethodNotAllowed)
		return
	}
	var req tasks.EnhanceRequest
	if err := readJSONBody(r, &req); err != nil {
		http.Error(w, "bad json", http.StatusBadRequest)
		return
	}
	res, err := tasks.EnhancePrompt(r.Context(), req)
	if err != nil {
		status := http.StatusInternalServerError
		if errors.Is(err, tasks.ErrEnhanceInput) {
			status = http.StatusUnprocessableEntity
		}
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
		return
	}
	_ = json.NewEncoder(w).Encode(res)
}
