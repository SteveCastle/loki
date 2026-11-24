package main

// transform_enhance_api.go — /api/transform/enhance-prompt
//
// Rewrites a Transform prompt for its target model with loki-scribe. Only ever
// on request (the Enhance button, or a run queued with "Enhance first"), and
// as a queued job (tasks.QueueEnhanceJob) so the text model takes its turn
// with the other model work:
//
//	POST {EnhanceRequest}  -> 202 {"jobId"}; the job's state arrives on /stream
//	POST {EnhanceRequest, "then": {input, fields}}
//	                       -> 202 {"jobId"}; the job queues `then` (a /create
//	                          body) with the rewrite as its prompt when done
//	GET ?job=ID            -> {state, result?, error?} (tasks.EnhanceJobState)
//	GET                    -> {"models": [...]}, the targets with a prompt guide
//
// Cancelling is the job's own POST /job/{id}/cancel.

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

func transformEnhanceHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodGet {
			if id := r.URL.Query().Get("job"); id != "" {
				st, ok := tasks.EnhanceJobStatus(deps.Queue, id)
				if !ok {
					w.WriteHeader(http.StatusNotFound)
					_ = json.NewEncoder(w).Encode(map[string]string{"error": "no such enhance job"})
					return
				}
				_ = json.NewEncoder(w).Encode(st)
				return
			}
			_ = json.NewEncoder(w).Encode(map[string][]string{"models": tasks.PromptGuideModels()})
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			tasks.EnhanceRequest
			Then *CreateJobHandlerRequest `json:"then"`
		}
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		var id string
		var err error
		if req.Then != nil {
			// parsed as POST /create parses its body
			args := ParseCommand(req.Then.Input)
			if len(args) == 0 {
				http.Error(w, "Invalid input", http.StatusBadRequest)
				return
			}
			then := tasks.EnhanceThen{Command: args[0]}
			if len(args) > 1 {
				then.Input = args[len(args)-1]
				then.Arguments = args[1 : len(args)-1]
			}
			then.Arguments = appendFieldArgs(then.Arguments, req.Then.Fields)
			id, err = tasks.QueueEnhanceThen(deps.Queue, req.EnhanceRequest, then)
		} else {
			id, err = tasks.QueueEnhanceJob(deps.Queue, req.EnhanceRequest)
		}
		if err != nil {
			status := http.StatusInternalServerError
			if errors.Is(err, tasks.ErrEnhanceInput) {
				status = http.StatusUnprocessableEntity
			}
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(map[string]string{"jobId": id})
	}
}
