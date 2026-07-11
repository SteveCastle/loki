package main

import (
	"encoding/json"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/stream"
	"github.com/stevecastle/shrike/tasks"
)

// broadcastPeopleChanged pushes the same "people-updated" SSE event the
// clustering job emits, so live UIs (People grid, person-filtered library
// views — in every connected window) refetch immediately after a MANUAL
// mutation: assign, reject, curate, merge, rename, delete, lock, cover.
func broadcastPeopleChanged() {
	payload, err := json.Marshal(map[string]any{"models": []string{}})
	if err != nil {
		return
	}
	stream.Broadcast(stream.Message{Type: "people-updated", Msg: string(payload)})
}

// RegisterPeopleRoutes wires the person-management API onto mux (called from
// RegisterFacesRoutes so the three platform mains stay one-line).
//
//	GET    /api/people               — list persons (face/media counts, cover)
//	POST   /api/people               — create a named person {name}
//	POST   /api/people/{id}/rename   — {name}; cascades to taxonomy rows
//	POST   /api/people/{id}/merge    — {intoId}; moves faces + taxonomy rows
//	DELETE /api/people/{id}          — unassigns faces, removes taxonomy rows
//	GET    /api/people/{id}/media    — the person's media, renderer item shape
//	GET    /api/people/{id}/faces    — the person's faces with typicality, for review
//	POST   /api/people/{id}/lock     — promote every auto face to a user assignment
//	POST   /api/people/{id}/curate   — {keepFaceIds}: lock the keeps, reject the rest
//	GET    /api/faces/ungrouped      — count of faces not yet assigned to any person
//	POST   /api/faces/{id}/assign    — {personId} or {name} (creates person)
//	POST   /api/faces/{id}/unassign
//	POST   /api/faces/{id}/reject    — {personId?}: veto + cannot-links + unassign
//	DELETE /api/faces/all?confirm=true — privacy wipe of ALL face data
func RegisterPeopleRoutes(mux *http.ServeMux, deps *Dependencies) {
	// GET (list) is public-readable so people browsing works in view-only
	// mode; POST (create) stays admin-only while public access is on.
	mux.HandleFunc("/api/people", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			peopleHandler(deps)(w, r)
			return
		}
		requireAuthWhenPublic(deps, peopleHandler(deps))(w, r)
	}, renderer.RolePublicRead))
	mux.HandleFunc("/api/people/{id}/rename", renderer.ApplyMiddlewares(personRenameHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/merge", renderer.ApplyMiddlewares(personMergeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}", renderer.ApplyMiddlewares(personDeleteHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/media", renderer.ApplyMiddlewares(personMediaHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/people/{id}/faces", renderer.ApplyMiddlewares(personFacesHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/people/{id}/lock", renderer.ApplyMiddlewares(personLockHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/curate", renderer.ApplyMiddlewares(personCurateHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/people/{id}/cover", renderer.ApplyMiddlewares(personCoverHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/assign-person", renderer.ApplyMiddlewares(mediaAssignPersonHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/reject-person", renderer.ApplyMiddlewares(mediaRejectPersonHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/assign", renderer.ApplyMiddlewares(faceAssignHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/unassign", renderer.ApplyMiddlewares(faceUnassignHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/reject", renderer.ApplyMiddlewares(faceRejectHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/all", renderer.ApplyMiddlewares(facesWipeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/stats", renderer.ApplyMiddlewares(facesStatsHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/faces/ungrouped", renderer.ApplyMiddlewares(facesUngroupedHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/tuning", renderer.ApplyMiddlewares(facesTuningHandler(deps), renderer.RoleAdmin))
}

// facesTuningHandler reads (GET) and persists (POST) the grouping tuner — the
// People panel's Tune sliders. Values live in the server config, so they
// apply to EVERY clustering pass on every client (buttons, tuned regroups,
// in-scan incremental passes) until changed. GET always reports the EFFECTIVE
// values (built-in defaults where nothing is saved).
func facesTuningHandler(deps *Dependencies) http.HandlerFunc {
	type tuning struct {
		ThresholdOffset float64 `json:"thresholdOffset"`
		MinCluster      int     `json:"minCluster"`
		MinQuality      float64 `json:"minQuality"`
	}
	current := func() tuning {
		cfg := appconfig.Get()
		t := tuning{ThresholdOffset: cfg.FaceClusterThresholdOffset, MinCluster: cfg.FaceClusterMinCluster, MinQuality: cfg.FaceClusterMinQuality}
		if t.ThresholdOffset < -0.2 || t.ThresholdOffset > 0.3 {
			t.ThresholdOffset = 0
		}
		if t.MinCluster < 1 {
			t.MinCluster = 3
		}
		if t.MinQuality <= 0 || t.MinQuality >= 1 {
			t.MinQuality = 0.75
		}
		return t
	}
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			writeJSON(w, current())
		case http.MethodPost:
			var req tuning
			if err := readJSON(r, &req); err != nil {
				httpError(w, "bad request", http.StatusBadRequest)
				return
			}
			if req.ThresholdOffset < -0.2 || req.ThresholdOffset > 0.3 {
				httpError(w, "thresholdOffset must be between -0.2 and 0.3", http.StatusBadRequest)
				return
			}
			if req.MinCluster < 1 || req.MinCluster > 50 {
				httpError(w, "minCluster must be between 1 and 50", http.StatusBadRequest)
				return
			}
			if req.MinQuality <= 0 || req.MinQuality >= 1 {
				httpError(w, "minQuality must be between 0 and 1", http.StatusBadRequest)
				return
			}
			cfg := appconfig.Get()
			cfg.FaceClusterThresholdOffset = req.ThresholdOffset
			cfg.FaceClusterMinCluster = req.MinCluster
			cfg.FaceClusterMinQuality = req.MinQuality
			if _, err := appconfig.Save(cfg); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, current())
		default:
			httpError(w, "use GET or POST", http.StatusMethodNotAllowed)
		}
	}
}

// facesUngroupedHandler reports how many stored faces have no person yet —
// the workload a "Group new faces" run would pick up. GET /api/faces/ungrouped;
// with ?faces=1 (plus optional limit/offset, best detections first) the
// response also carries a page of the faces themselves for the manual-review
// pseudo-group.
func facesUngroupedHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		count, err := tasks.CountUngroupedFaces(deps.DB)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		resp := map[string]any{"count": count}
		if r.URL.Query().Get("faces") == "1" {
			limit := 120
			if v, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && v >= 1 && v <= 500 {
				limit = v
			}
			offset := 0
			if v, err := strconv.Atoi(r.URL.Query().Get("offset")); err == nil && v > 0 {
				offset = v
			}
			faces, err := tasks.ListUngroupedFaces(deps.DB, limit, offset)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			resp["faces"] = faces
		}
		writeJSON(w, resp)
	}
}

// facesStatsHandler reports how much face data is stored — per-model face
// vectors (count + blob bytes), scan markers, people counts, and the live
// in-memory search index — for the Config page's storage panel.
func facesStatsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}

		type modelStats struct {
			Model string `json:"model"`
			Count int    `json:"count"`
			Bytes int64  `json:"bytes"`
		}
		faces := []modelStats{}
		var totalFaces int
		var totalBytes int64
		rows, err := deps.DB.Query(
			`SELECT model, COUNT(*), COALESCE(SUM(LENGTH(vector)), 0)
			 FROM face GROUP BY model ORDER BY model`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for rows.Next() {
			var s modelStats
			if err := rows.Scan(&s.Model, &s.Count, &s.Bytes); err == nil {
				faces = append(faces, s)
				totalFaces += s.Count
				totalBytes += s.Bytes
			}
		}
		rows.Close()

		var scans int
		_ = deps.DB.QueryRow(`SELECT COUNT(*) FROM face_scan`).Scan(&scans)
		var people, named int
		_ = deps.DB.QueryRow(`SELECT COUNT(*),
			COALESCE(SUM(CASE WHEN name IS NOT NULL AND TRIM(name) <> '' THEN 1 ELSE 0 END), 0)
			FROM person`).Scan(&people, &named)

		writeJSON(w, map[string]any{
			"faces":       faces,
			"total_faces": totalFaces,
			"total_bytes": totalBytes,
			"scans":       scans,
			"people": map[string]any{
				"total":   people,
				"named":   named,
				"unnamed": people - named,
			},
			"index": map[string]any{
				"model":   tasks.FaceIndexedModel(),
				"vectors": tasks.FaceIndexSize(),
			},
		})
	}
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
			broadcastPeopleChanged()
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
		// Report the RESOLVED stored name (RenamePerson may suffix it, e.g.
		// _cluster on a collision with a non-People tag) — clients use it to
		// keep an active tag filter pointing at the renamed person.
		name := strings.TrimSpace(req.Name)
		if p, found, err := media.GetPersonByID(deps.DB, id); err == nil && found {
			name = p.Name
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"id": id, "name": name})
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
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"merged": id, "into": req.IntoID})
	}
}

// personDeleteHandler removes a person. By default their faces are kept
// (unassigned, free to re-cluster) and the dissolved membership is recorded
// as a group ban — a negative attractor stopping the SAME group from simply
// re-forming on the next clustering pass (?ban=false skips recording it).
// With ?deleteFaces=true the face rows are deleted too — the purge for a
// hopelessly mixed cluster; deleted faces stay gone until their media is
// explicitly rescanned.
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
		if r.URL.Query().Get("deleteFaces") == "true" {
			faceIDs, err := media.DeletePersonAndFaces(deps.DB, id)
			if err != nil {
				httpError(w, err.Error(), userErrorStatus(err))
				return
			}
			tasks.FaceIndexDeleteFaceIDs(faceIDs)
			broadcastPeopleChanged()
			writeJSON(w, map[string]any{"deleted": id, "facesDeleted": len(faceIDs)})
			return
		}
		banned := 0
		if r.URL.Query().Get("ban") != "false" {
			// Snapshot the membership BEFORE the delete unassigns it. The name
			// is provenance only (helps debugging the ban table).
			p, found, err := media.GetPersonByID(deps.DB, id)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if found {
				if banned, err = media.BanFaceGroup(deps.DB, id, p.Name); err != nil {
					httpError(w, err.Error(), http.StatusInternalServerError)
					return
				}
			}
		}
		if err := media.DeletePerson(deps.DB, id); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"deleted": id, "banned": banned})
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

// personCoverHandler regenerates a person's cover crop: it walks the person's
// faces best-first (detection confidence × bbox area) and picks the first one
// whose source actually renders — decoding images directly and re-extracting
// the scan's midpoint frame for videos — so a person with usable faces always
// ends up with a visible preview. POST /api/people/{id}/cover.
func personCoverHandler(deps *Dependencies) http.HandlerFunc {
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
		if _, found, err := media.GetPersonByID(deps.DB, id); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		} else if !found {
			httpError(w, "no such person", http.StatusNotFound)
			return
		}
		faces, err := media.PersonFacesByQuality(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(faces) == 0 {
			httpError(w, "person has no faces to pick a cover from", http.StatusUnprocessableEntity)
			return
		}
		// Try a bounded number of candidates — decoding (and possibly frame
		// extraction) per face is not free, and if the ten best faces all fail
		// the rest almost certainly will too.
		const maxCandidates = 10
		for i, f := range faces {
			if i >= maxCandidates {
				break
			}
			if _, err := decodeFaceSource(r.Context(), deps.DB, f.MediaPath); err != nil {
				continue
			}
			if err := media.SetPersonCover(deps.DB, id, f.ID); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			broadcastPeopleChanged()
			writeJSON(w, map[string]any{"personId": id, "coverFaceId": f.ID})
			return
		}
		httpError(w, "none of the person's best faces could be rendered", http.StatusUnprocessableEntity)
	}
}

// mediaAssignPersonHandler assigns a media item's face to a person (dragging
// a person card onto media). POST /api/media/assign-person with
// {path, personId, setCover?} — or {path, newPerson: true, name?} to MINT a
// brand-new group from this image (dragging the "New group" chip onto media):
// the item's face is pulled out of whatever cluster it sits in (a user move,
// so a veto keeps it from drifting back) and becomes the founding, locked seed
// of a new person, which immediately joins the People grid ready to collect
// more faces by drag or clustering.
//
// The media is scanned on the fly when it has no stored face vectors yet.
// When the item contains several faces, the one most similar to the person's
// existing faces wins (people often appear alongside others); for a person
// with no faces yet — always the case for newPerson — the largest face wins.
// setCover additionally makes that face the person's preview crop (implied
// for a new person).
func mediaAssignPersonHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Path      string `json:"path"`
			PersonID  int64  `json:"personId"`
			SetCover  bool   `json:"setCover"`
			NewPerson bool   `json:"newPerson"`
			Name      string `json:"name"`
		}
		if err := readJSON(r, &req); err != nil || strings.TrimSpace(req.Path) == "" ||
			(req.PersonID <= 0 && !req.NewPerson) {
			httpError(w, "path and personId (or newPerson) required", http.StatusBadRequest)
			return
		}
		if !req.NewPerson {
			if _, found, err := media.GetPersonByID(deps.DB, req.PersonID); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			} else if !found {
				httpError(w, "no such person", http.StatusNotFound)
				return
			}
		}

		// Stored faces when scanned; scan on the fly (persist + index)
		// otherwise — "create the vector first". Runs BEFORE a new person is
		// created so a face-less item never leaves an empty group behind.
		faces, _, err := tasks.FacesForPathOrScan(r.Context(), deps.DB, req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(faces) == 0 {
			httpError(w, "no face detected in this media item", http.StatusUnprocessableEntity)
			return
		}

		personName := ""
		if req.NewPerson {
			personName = strings.TrimSpace(req.Name)
			if personName == "" {
				personName, err = media.NextUnknownName(deps.DB)
				if err != nil {
					httpError(w, err.Error(), http.StatusInternalServerError)
					return
				}
			}
			id, err := media.CreatePerson(deps.DB, personName)
			if err != nil {
				httpError(w, err.Error(), userErrorStatus(err))
				return
			}
			req.PersonID = id
			req.SetCover = true // a brand-new group always needs a preview
		}

		personFaces, err := media.PersonFacesByQuality(deps.DB, req.PersonID)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		best := pickFaceForPerson(faces, personFaces)

		if err := media.AssignFace(deps.DB, best.ID, req.PersonID, "user"); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		if req.SetCover {
			if err := media.SetPersonCover(deps.DB, req.PersonID, best.ID); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{
			"faceId":   best.ID,
			"personId": req.PersonID,
			"faces":    len(faces),
			"setCover": req.SetCover,
			"created":  req.NewPerson,
			"name":     personName,
		})
	}
}

// mediaRejectPersonHandler is the inverse of assign-person: POST
// /api/media/reject-person with {path, personId} (or {path, name}) rejects
// every face of that person on the item — veto + cannot-links + unassign, the
// same permanent assertion as the per-face reject. Used when a person's tag
// is removed from a media item so the group actually loses those faces (the
// web-mode assignment DELETE calls the same logic server-side; this endpoint
// exists for the Electron viewer, whose tag deletes go straight to SQLite).
func mediaRejectPersonHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Path     string `json:"path"`
			PersonID int64  `json:"personId"`
			Name     string `json:"name"`
		}
		if err := readJSON(r, &req); err != nil || strings.TrimSpace(req.Path) == "" ||
			(req.PersonID <= 0 && strings.TrimSpace(req.Name) == "") {
			httpError(w, "path and personId (or name) required", http.StatusBadRequest)
			return
		}
		personID := req.PersonID
		if personID <= 0 {
			p, found, err := media.GetPersonByDisplayName(deps.DB, strings.TrimSpace(req.Name))
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if !found {
				httpError(w, "no such person", http.StatusNotFound)
				return
			}
			personID = p.ID
		}
		ids, err := media.RejectPersonFacesOnMedia(deps.DB, req.Path, personID)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if len(ids) > 0 {
			broadcastPeopleChanged()
		}
		writeJSON(w, map[string]any{"personId": personID, "rejectedFaceIds": ids})
	}
}

// pickFaceForPerson chooses which of a media item's faces to assign: the one
// most similar to the person's existing faces when the person has any
// (comparing only same-dimension vectors, i.e. the same recognizer), else the
// largest face in the item.
func pickFaceForPerson(faces, personFaces []media.Face) media.Face {
	best := faces[0]
	if len(personFaces) > 0 {
		var bestScore float32 = -2
		matched := false
		for _, f := range faces {
			for _, pf := range personFaces {
				if len(f.Vec) != len(pf.Vec) {
					continue
				}
				if sc := embedvec.CosineSim(f.Vec, pf.Vec); sc > bestScore {
					bestScore, best, matched = sc, f, true
				}
			}
		}
		if matched {
			return best
		}
	}
	for _, f := range faces[1:] {
		if f.W*f.H > best.W*best.H {
			best = f
		}
	}
	return best
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
			PersonID  int64  `json:"personId"`
			Name      string `json:"name"`
			NewPerson bool   `json:"newPerson"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		personID := req.PersonID
		created := false
		personName := ""
		if personID == 0 {
			name := strings.TrimSpace(req.Name)
			if name == "" {
				// {newPerson:true} mints a fresh auto-named group seeded by
				// this face — "make a person out of this stray".
				if !req.NewPerson {
					httpError(w, "personId, name, or newPerson required", http.StatusBadRequest)
					return
				}
				var err error
				if name, err = media.NextUnknownName(deps.DB); err != nil {
					httpError(w, err.Error(), http.StatusInternalServerError)
					return
				}
			}
			// Assign by name: reuse the person when it exists (including the
			// _cluster-suffixed form), create otherwise.
			if p, found, err := media.GetPersonByDisplayName(deps.DB, name); err != nil {
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
				created = true
			}
			personName = name
		}
		if err := media.AssignFace(deps.DB, faceID, personID, "user"); err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		if created {
			// A brand-new group needs a preview; its only face is the cover.
			if err := media.SetPersonCover(deps.DB, personID, faceID); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"faceId": faceID, "personId": personID, "created": created, "name": personName})
	}
}

// faceRejectHandler is the "this is not them" action: POST
// /api/faces/{id}/reject with an optional {personId} (defaults to the face's
// current person). It records a permanent negative assertion — a veto against
// the person plus cannot-links against the group's exemplar faces — and
// unassigns the face. No clustering pass, under any settings, will put the
// face back in that group, even after the group is dissolved and re-forms.
func faceRejectHandler(deps *Dependencies) http.HandlerFunc {
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
			PersonID int64 `json:"personId"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		personID := req.PersonID
		if personID == 0 {
			f, found, err := media.GetFaceByID(deps.DB, faceID)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if !found {
				httpError(w, "no face with id "+strconv.FormatInt(faceID, 10), http.StatusNotFound)
				return
			}
			if f.PersonID == 0 {
				httpError(w, "face is unassigned; personId required", http.StatusBadRequest)
				return
			}
			personID = f.PersonID
		}
		links, err := media.RejectFaceFromPerson(deps.DB, faceID, personID)
		if err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{
			"faceId":      faceID,
			"personId":    personID,
			"cannotLinks": links,
		})
	}
}

// personLockHandler promotes every auto-assigned face of a person to a user
// assignment — "this whole group is right". POST /api/people/{id}/lock.
// Locked faces survive every regroup (including --reset-all) and seed future
// clustering with extra weight.
func personLockHandler(deps *Dependencies) http.HandlerFunc {
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
		n, err := media.LockPersonFaces(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"personId": id, "locked": n})
	}
}

// personCurateHandler applies a whole-group review in one shot: POST
// /api/people/{id}/curate with {keepFaceIds: [...]}. Faces in the list are
// locked (user assignments); every other face of the person is rejected —
// removed with a permanent guarantee it never returns to this group. The
// one-click "keep the checked, discard the rest" cleanup for a group with
// false positives.
func personCurateHandler(deps *Dependencies) http.HandlerFunc {
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
			KeepFaceIDs []int64 `json:"keepFaceIds"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		kept, rejected, err := media.CuratePersonFaces(deps.DB, id, req.KeepFaceIDs)
		if err != nil {
			httpError(w, err.Error(), userErrorStatus(err))
			return
		}
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"personId": id, "kept": kept, "rejected": rejected})
	}
}

// reviewFace is one entry of the person face-review payload.
type reviewFace struct {
	ID         int64   `json:"id"`
	MediaPath  string  `json:"path"`
	FrameTS    float64 `json:"frameTs"`
	DetScore   float64 `json:"detScore"`
	AssignedBy string  `json:"assignedBy"`
	// Typicality is the face's cosine against the group's center — the mean
	// of the user-CONFIRMED faces when any exist (self excluded), else the
	// plain group mean. Low values are the outliers a human should look at
	// first. 0 when it can't be computed (single face or mixed-dimension
	// group).
	Typicality float32 `json:"typicality"`
}

// personFacesHandler returns a person's faces enriched with typicality for
// the review UI, least typical first (the faces most worth a human look).
// GET /api/people/{id}/faces.
func personFacesHandler(deps *Dependencies) http.HandlerFunc {
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
		faces, err := media.PersonFacesByQuality(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out := make([]reviewFace, len(faces))
		for i, f := range faces {
			out[i] = reviewFace{
				ID:         f.ID,
				MediaPath:  f.MediaPath,
				FrameTS:    f.FrameTS,
				DetScore:   f.Score,
				AssignedBy: f.AssignedBy,
				Typicality: 0,
			}
		}
		// Centroid per vector dimension (a merged person can hold faces from
		// different recognizers; only same-dimension vectors are comparable).
		// When the person has user-confirmed faces, the center is the mean of
		// THOSE alone — typicality then reads "similarity to the confirmed
		// core", matching the clustering anchor, and stays honest even when a
		// bad regroup floods the group with someone else's faces (a weighted
		// all-face mean drifts toward the flood, sorting the strangers as
		// typical and the confirmed faces as suspects). Groups with no
		// confirmations use the plain all-face mean.
		sums := map[int][]float32{}
		userSums := map[int][]float32{}
		userCounts := map[int]int{}
		for i := range faces {
			v := embedvec.Normalize(faces[i].Vec)
			faces[i].Vec = v
			s := sums[len(v)]
			if s == nil {
				s = make([]float32, len(v))
				sums[len(v)] = s
			}
			for k, x := range v {
				s[k] += x
			}
			if faces[i].AssignedBy == "user" {
				us := userSums[len(v)]
				if us == nil {
					us = make([]float32, len(v))
					userSums[len(v)] = us
				}
				for k, x := range v {
					us[k] += x
				}
				userCounts[len(v)]++
			}
		}
		for i, f := range faces {
			rest := make([]float32, len(f.Vec))
			// A user face measures against the OTHER confirmed faces (self
			// excluded); with only itself confirmed there is no confirmed
			// center left, so it falls back to the all-face mean like an
			// unconfirmed group.
			uc := userCounts[len(f.Vec)]
			if uc > 0 && !(f.AssignedBy == "user" && uc == 1) {
				copy(rest, userSums[len(f.Vec)])
				if f.AssignedBy == "user" {
					for k := range rest {
						rest[k] -= f.Vec[k]
					}
				}
			} else {
				s := sums[len(f.Vec)]
				for k := range rest {
					rest[k] = s[k] - f.Vec[k] // exclude self from its own center
				}
			}
			var norm float64
			for _, x := range rest {
				norm += float64(x) * float64(x)
			}
			if norm > 1e-9 {
				out[i].Typicality = float32(float64(dotFloat32(f.Vec, rest)) / math.Sqrt(norm))
			}
		}
		sort.SliceStable(out, func(a, b int) bool { return out[a].Typicality < out[b].Typicality })
		writeJSON(w, map[string]any{"personId": id, "faces": out})
	}
}

// dotFloat32 is a plain dot product (embedvec.CosineSim renormalizes, which
// the exclude-self centroid math above must control itself).
func dotFloat32(a, b []float32) float32 {
	var d float64
	for i := range a {
		d += float64(a[i]) * float64(b[i])
	}
	return float32(d)
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
		broadcastPeopleChanged()
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
		broadcastPeopleChanged()
		writeJSON(w, map[string]any{"deleted": true})
	}
}
