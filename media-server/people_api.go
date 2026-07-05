package main

import (
	"net/http"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/tasks"
)

// RegisterPeopleRoutes wires the person-management API onto mux (called from
// RegisterFacesRoutes so the three platform mains stay one-line).
//
//	GET    /api/people               — list persons (face/media counts, cover)
//	POST   /api/people               — create a named person {name}
//	POST   /api/people/{id}/rename   — {name}; cascades to taxonomy rows
//	POST   /api/people/{id}/merge    — {intoId}; moves faces + taxonomy rows
//	DELETE /api/people/{id}          — unassigns faces, removes taxonomy rows
//	GET    /api/people/{id}/media    — the person's media, renderer item shape
//	POST   /api/faces/{id}/assign    — {personId} or {name} (creates person)
//	POST   /api/faces/{id}/unassign
//	DELETE /api/faces/all?confirm=true — privacy wipe of ALL face data
func RegisterPeopleRoutes(mux *http.ServeMux, deps *Dependencies) {
	mux.HandleFunc("/api/people", renderer.ApplyMiddlewares(peopleHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/rename", renderer.ApplyMiddlewares(personRenameHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/merge", renderer.ApplyMiddlewares(personMergeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}", renderer.ApplyMiddlewares(personDeleteHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/media", renderer.ApplyMiddlewares(personMediaHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/assign", renderer.ApplyMiddlewares(faceAssignHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/unassign", renderer.ApplyMiddlewares(faceUnassignHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/all", renderer.ApplyMiddlewares(facesWipeHandler(deps), renderer.RoleAdmin))
}

// pathID parses the {id} path value.
func pathID(r *http.Request) (int64, bool) {
	id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
	return id, err == nil && id > 0
}

// userError maps a media-package validation error (unknown id, name conflict,
// …) to 400/404; anything else stays 500.
func userErrorStatus(err error) int {
	msg := err.Error()
	switch {
	case strings.Contains(msg, "no person with id"), strings.Contains(msg, "no face with id"):
		return http.StatusNotFound
	case strings.Contains(msg, "already"), strings.Contains(msg, "required"),
		strings.Contains(msg, "cannot merge"), strings.Contains(msg, "must be"):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func peopleHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			people, err := media.GetPeople(deps.DB)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if people == nil {
				people = []media.Person{}
			}
			writeJSON(w, people)
		case http.MethodPost:
			var req struct {
				Name string `json:"name"`
			}
			if err := readJSON(r, &req); err != nil {
				httpError(w, "bad request", http.StatusBadRequest)
				return
			}
			id, err := media.CreatePerson(deps.DB, req.Name)
			if err != nil {
				httpError(w, err.Error(), userErrorStatus(err))
				return
			}
			writeJSON(w, map[string]any{"id": id, "name": strings.TrimSpace(req.Name)})
		default:
			httpError(w, "use GET or POST", http.StatusMethodNotAllowed)
		}
	}
}

func personRenameHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid person id", http.StatusBadRequest)
			return
		}
		var req struct {
			Name string `json:"name"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if err := media.RenamePerson(deps.DB, id, req.Name); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		writeJSON(w, map[string]any{"id": id, "name": strings.TrimSpace(req.Name)})
	}
}

func personMergeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid person id", http.StatusBadRequest)
			return
		}
		var req struct {
			IntoID int64 `json:"intoId"`
		}
		if err := readJSON(r, &req); err != nil || req.IntoID <= 0 {
			httpError(w, "intoId required", http.StatusBadRequest)
			return
		}
		if err := media.MergePersons(deps.DB, id, req.IntoID); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		writeJSON(w, map[string]any{"merged": id, "into": req.IntoID})
	}
}

func personDeleteHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			httpError(w, "use DELETE", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid person id", http.StatusBadRequest)
			return
		}
		if err := media.DeletePerson(deps.DB, id); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		writeJSON(w, map[string]any{"deleted": id})
	}
}

func personMediaHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid person id", http.StatusBadRequest)
			return
		}
		if _, found, err := media.GetPersonByID(deps.DB, id); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		} else if !found {
			httpError(w, "no such person", http.StatusNotFound)
			return
		}
		paths, err := media.PersonMediaPaths(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Rank preserved via descending pseudo-scores (best detection first).
		hits := make([]tasks.SimilarHit, len(paths))
		for i, p := range paths {
			hits[i] = tasks.SimilarHit{Path: p, Score: float32(len(paths) - i)}
		}
		items, err := enrichScoredItems(deps.DB, hits)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, items)
	}
}

func faceAssignHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		faceID, ok := pathID(r)
		if !ok {
			httpError(w, "invalid face id", http.StatusBadRequest)
			return
		}
		var req struct {
			PersonID int64  `json:"personId"`
			Name     string `json:"name"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		personID := req.PersonID
		if personID == 0 {
			name := strings.TrimSpace(req.Name)
			if name == "" {
				httpError(w, "personId or name required", http.StatusBadRequest)
				return
			}
			// Assign by name: reuse the person when it exists, create otherwise.
			if p, found, err := media.GetPersonByName(deps.DB, name); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			} else if found {
				personID = p.ID
			} else {
				id, err := media.CreatePerson(deps.DB, name)
				if err != nil {
					httpError(w, err.Error(), userErrorStatus(err))
					return
				}
				personID = id
			}
		}
		if err := media.AssignFace(deps.DB, faceID, personID, "user"); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		writeJSON(w, map[string]any{"faceId": faceID, "personId": personID})
	}
}

func faceUnassignHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		faceID, ok := pathID(r)
		if !ok {
			httpError(w, "invalid face id", http.StatusBadRequest)
			return
		}
		if err := media.UnassignFace(deps.DB, faceID); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		writeJSON(w, map[string]any{"faceId": faceID, "unassigned": true})
	}
}

// facesWipeHandler is the privacy escape hatch: DELETE /api/faces/all wipes
// every face row, scan marker, person, and People-category taxonomy row.
// Requires ?confirm=true — this is not undoable (though rescanning rebuilds
// everything except manual labels).
func facesWipeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			httpError(w, "use DELETE", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Query().Get("confirm") != "true" {
			httpError(w, "add ?confirm=true to delete all face data", http.StatusBadRequest)
			return
		}
		if err := media.DeleteAllFaceData(deps.DB); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		tasks.SetFaceIndexForModel(nil, "", nil)
		writeJSON(w, map[string]any{"deleted": true})
	}
}
