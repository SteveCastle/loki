package main

// Saved-workflow CRUD + run handlers. This file is intentionally free of
// build tags so every platform main can register the /workflows routes;
// the ad-hoc POST /workflow handler stays in the per-platform mains.

import (
	"encoding/json"
	"net/http"

	"github.com/stevecastle/shrike/jobqueue"
)

func workflowsListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		list, err := deps.Queue.ListWorkflows()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if list == nil {
			list = []jobqueue.SavedWorkflowSummary{}
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(list)
	}
}

func workflowDetailHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		if id == "" {
			http.Error(w, "missing id", http.StatusBadRequest)
			return
		}

		switch r.Method {
		case http.MethodGet:
			wf, err := deps.Queue.GetWorkflow(id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusNotFound)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(wf)

		case http.MethodPut:
			var req struct {
				Name string                  `json:"name"`
				DAG  []jobqueue.WorkflowTask `json:"dag"`
			}
			if err := readJSONBody(r, &req); err != nil {
				http.Error(w, "bad json", http.StatusBadRequest)
				return
			}
			if err := deps.Queue.UpdateWorkflow(id, req.Name, req.DAG); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			wf, err := deps.Queue.GetWorkflow(id)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(wf)

		case http.MethodDelete:
			if err := deps.Queue.DeleteWorkflow(id); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusNoContent)

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func workflowCreateHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Name string                  `json:"name"`
			DAG  []jobqueue.WorkflowTask `json:"dag"`
		}
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		wf, err := deps.Queue.CreateWorkflow(req.Name, req.DAG)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(wf)
	}
}

func workflowRunHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		id := r.PathValue("id")
		if id == "" {
			http.Error(w, "missing id", http.StatusBadRequest)
			return
		}

		var req struct {
			Input string `json:"input"`
		}
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		ids, err := deps.Queue.RunWorkflow(id, req.Input)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"ids": ids})
	}
}
