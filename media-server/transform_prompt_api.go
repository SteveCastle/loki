package main

// transform_prompt_api.go — POST /api/transform/prompt
//
// Previews the prompt a retouch / reshoot job will run, for the Transform
// flow's review step. The body is exactly what POST /create takes
// ({input, fields}); nothing is queued.

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

func transformPromptHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		var req CreateJobHandlerRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		args := ParseCommand(req.Input)
		if len(args) < 2 {
			http.Error(w, "Invalid input", http.StatusBadRequest)
			return
		}
		cmd, input := args[0], args[len(args)-1]
		args = appendFieldArgs(args[1:len(args)-1], req.Fields)

		preview, err := tasks.PreviewPrompt(r.Context(), cmd, args, input)
		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, tasks.ErrPreviewUnsupported) {
				status = http.StatusUnprocessableEntity
			}
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		_ = json.NewEncoder(w).Encode(preview)
	}
}
