package main

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/stream"
	"github.com/stevecastle/shrike/tasks"
)

// broadcastDuplicatesChanged pushes the same "duplicates-updated" SSE event
// the find-duplicates job emits while it runs, so open Duplicates panels (in
// every connected window) refetch after a MANUAL mutation: merge, dismiss,
// restore, exclude, delete, wipe.
func broadcastDuplicatesChanged() {
	payload, err := json.Marshal(map[string]any{"groups": 0, "items": 0})
	if err != nil {
		return
	}
	stream.Broadcast(stream.Message{Type: "duplicates-updated", Msg: string(payload)})
}

// RegisterDuplicatesRoutes wires the duplicate-candidate review API onto mux.
// Groups are produced by the "find-duplicates" task (tasks/find_duplicates.go)
// and stored in duplicate_group / duplicate_member (media/duplicates.go);
// this API is how the user acts on them. Reads are public-readable so the
// panel works in view-only mode; every mutation is admin-only.
//
//	GET    /api/duplicates?status=pending|dismissed|all&sort=newest|members&limit&offset — groups with members (renderer item shape)
//	GET    /api/duplicates/stats             — pending/dismissed group counts, item counts
//	GET    /api/duplicates/for-path?path=    — {groupId, status} for one media item (0 = none)
//	GET    /api/duplicates/{id}              — one group
//	POST   /api/duplicates/{id}/merge        — {keep, sources?}: media.MergeInto keep ← sources (default: every other active member)
//	POST   /api/duplicates/{id}/dismiss      — not duplicates: hide the group, remember the decision
//	POST   /api/duplicates/{id}/restore      — back to pending
//	POST   /api/duplicates/{id}/exclude      — {path, excluded}: pull one member out (or back in)
//	DELETE /api/duplicates/{id}              — forget the group; its items may be regrouped next run
//	DELETE /api/duplicates/all?confirm=true[&status=dismissed] — forget every group (of one status)
func RegisterDuplicatesRoutes(mux *http.ServeMux, deps *Dependencies) {
	mux.HandleFunc("/api/duplicates", renderer.ApplyMiddlewares(duplicatesListHandler(deps), renderer.RolePublicRead))
	// Literal paths — Go's mux prefers them over the "/api/duplicates/{id}"
	// wildcard, so "stats" / "all" / "for-path" never parse as an id.
	mux.HandleFunc("/api/duplicates/stats", renderer.ApplyMiddlewares(duplicatesStatsHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/duplicates/for-path", renderer.ApplyMiddlewares(duplicatesForPathHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/duplicates/all", renderer.ApplyMiddlewares(duplicatesWipeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/duplicates/{id}", renderer.ApplyMiddlewares(duplicateGroupHandler(deps), renderer.RolePublicRead))
	mux.HandleFunc("/api/duplicates/{id}/merge", renderer.ApplyMiddlewares(duplicateMergeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/duplicates/{id}/dismiss", renderer.ApplyMiddlewares(duplicateStatusHandler(deps, media.DuplicateStatusDismissed), renderer.RoleAdmin))
	mux.HandleFunc("/api/duplicates/{id}/restore", renderer.ApplyMiddlewares(duplicateStatusHandler(deps, media.DuplicateStatusPending), renderer.RoleAdmin))
	mux.HandleFunc("/api/duplicates/{id}/exclude", renderer.ApplyMiddlewares(duplicateExcludeHandler(deps), renderer.RoleAdmin))
}

// duplicateGroupView is the wire shape of a group: the group row plus its
// members in the flat media-item shape the renderer's grids expect (path,
// elo, dimensions, battles) with the member's similarity as "score" and its
// "excluded" flag. Member order is the group's (anchor, then by similarity,
// excluded last), not the score sort enrichScoredItems applies.
func duplicateGroupView(deps *Dependencies, g media.DuplicateGroup) (map[string]any, error) {
	hits := make([]tasks.SimilarHit, len(g.Members))
	for i, m := range g.Members {
		hits[i] = tasks.SimilarHit{Path: m.Path, Score: float32(m.Score)}
	}
	enriched, err := enrichScoredItems(deps.DB, hits)
	if err != nil {
		return nil, err
	}
	// The default keeper (highest resolution, then largest file) and the
	// facts behind it, so the panel can show them and highlight the keeper.
	keeper, facts, err := media.DuplicateGroupKeeper(deps.DB, g)
	if err != nil {
		return nil, err
	}
	byPath := make(map[string]map[string]any, len(enriched))
	for _, it := range enriched {
		if p, ok := it["path"].(string); ok {
			byPath[p] = it
		}
	}
	items := make([]map[string]any, 0, len(g.Members))
	for _, m := range g.Members {
		it := byPath[m.Path]
		if it == nil {
			it = map[string]any{"path": m.Path, "mtimeMs": int64(0)}
		}
		it["score"] = m.Score
		it["excluded"] = m.Excluded
		it["anchor"] = m.Path == g.AnchorPath
		it["keeper"] = m.Path == keeper
		if f, ok := facts[m.Path]; ok && f.Size > 0 {
			it["size"] = f.Size
		}
		items = append(items, it)
	}
	return map[string]any{
		"id":          g.ID,
		"model":       g.Model,
		"threshold":   g.Threshold,
		"anchorPath":  g.AnchorPath,
		"keeperPath":  keeper,
		"status":      g.Status,
		"createdAt":   g.CreatedAt,
		"updatedAt":   g.UpdatedAt,
		"memberCount": g.MemberCount,
		"activeCount": g.ActiveCount,
		"minScore":    g.MinScore,
		"members":     items,
	}, nil
}

func duplicatesListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		status := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("status")))
		switch status {
		case "", media.DuplicateStatusPending:
			status = media.DuplicateStatusPending
		case media.DuplicateStatusDismissed:
		case "all":
			status = ""
		default:
			httpError(w, "status must be pending, dismissed, or all", http.StatusBadRequest)
			return
		}
		limit := 50
		if v, err := strconv.Atoi(r.URL.Query().Get("limit")); err == nil && v >= 1 && v <= 500 {
			limit = v
		}
		sort := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("sort")))
		switch sort {
		case "", media.DuplicateSortNewest:
			sort = media.DuplicateSortNewest
		case media.DuplicateSortMembers:
		default:
			httpError(w, "sort must be newest or members", http.StatusBadRequest)
			return
		}
		offset := 0
		if v, err := strconv.Atoi(r.URL.Query().Get("offset")); err == nil && v > 0 {
			offset = v
		}
		groups, total, err := media.ListDuplicateGroups(deps.DB, status, sort, limit, offset)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		views := make([]map[string]any, 0, len(groups))
		for _, g := range groups {
			v, err := duplicateGroupView(deps, g)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			views = append(views, v)
		}
		writeJSON(w, map[string]any{
			"groups": views,
			"total":  total,
			"limit":  limit,
			"offset": offset,
		})
	}
}

func duplicatesStatsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		s, err := media.GetDuplicateStats(deps.DB)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, s)
	}
}

func duplicatesForPathHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		p := strings.TrimSpace(r.URL.Query().Get("path"))
		if p == "" {
			httpError(w, "path required", http.StatusBadRequest)
			return
		}
		id, err := media.DuplicateGroupForPath(deps.DB, p)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		resp := map[string]any{"groupId": id}
		if id != 0 {
			if g, found, err := media.GetDuplicateGroup(deps.DB, id); err == nil && found {
				resp["status"] = g.Status
				resp["activeCount"] = g.ActiveCount
			}
		}
		writeJSON(w, resp)
	}
}

func duplicateGroupHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid group id", http.StatusBadRequest)
			return
		}
		switch r.Method {
		case http.MethodGet:
			g, found, err := media.GetDuplicateGroup(deps.DB, id)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if !found {
				httpError(w, "no such duplicate group", http.StatusNotFound)
				return
			}
			v, err := duplicateGroupView(deps, g)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			writeJSON(w, v)
		case http.MethodDelete:
			// Deleting is a write: in public mode the read-role middleware let
			// the request through, so re-check here.
			requireAuthWhenPublic(deps, func(w http.ResponseWriter, r *http.Request) {
				if err := media.DeleteDuplicateGroup(deps.DB, id); err != nil {
					httpError(w, err.Error(), duplicateErrorStatus(err))
					return
				}
				broadcastDuplicatesChanged()
				writeJSON(w, map[string]any{"deleted": id})
			})(w, r)
		default:
			httpError(w, "use GET or DELETE", http.StatusMethodNotAllowed)
		}
	}
}

// duplicateErrorStatus maps media-package validation errors to 400/404.
func duplicateErrorStatus(err error) int {
	msg := err.Error()
	switch {
	case strings.Contains(msg, "no duplicate group"), strings.Contains(msg, "no member"):
		return http.StatusNotFound
	case strings.Contains(msg, "must be"), strings.Contains(msg, "required"):
		return http.StatusBadRequest
	default:
		return http.StatusInternalServerError
	}
}

func duplicateStatusHandler(deps *Dependencies, status string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid group id", http.StatusBadRequest)
			return
		}
		if err := media.SetDuplicateGroupStatus(deps.DB, id, status); err != nil {
			httpError(w, err.Error(), duplicateErrorStatus(err))
			return
		}
		broadcastDuplicatesChanged()
		writeJSON(w, map[string]any{"id": id, "status": status})
	}
}

func duplicateExcludeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid group id", http.StatusBadRequest)
			return
		}
		var req struct {
			Path     string `json:"path"`
			Excluded *bool  `json:"excluded"`
		}
		if err := readJSON(r, &req); err != nil || strings.TrimSpace(req.Path) == "" {
			httpError(w, "bad request: path required", http.StatusBadRequest)
			return
		}
		excluded := true
		if req.Excluded != nil {
			excluded = *req.Excluded
		}
		if err := media.SetDuplicateMemberExcluded(deps.DB, id, strings.TrimSpace(req.Path), excluded); err != nil {
			httpError(w, err.Error(), duplicateErrorStatus(err))
			return
		}
		broadcastDuplicatesChanged()
		g, found, err := media.GetDuplicateGroup(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !found {
			writeJSON(w, map[string]any{"id": id, "group": nil})
			return
		}
		v, err := duplicateGroupView(deps, g)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"id": id, "group": v})
	}
}

// duplicateMergeHandler is the review action that actually deduplicates:
// the chosen keeper absorbs the sources' tags, embeddings, and transcript
// and the sources are deleted (media.MergeInto — the same merge behind the
// viewer's Merge action and the dedupe task). Sources default to every
// other ACTIVE member; excluded members are never merged unless named
// explicitly. The group's member rows follow the deletions, so a fully
// merged group disappears on its own and a partially merged one stays
// pending with what is left.
func duplicateMergeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		id, ok := pathID(r)
		if !ok {
			httpError(w, "invalid group id", http.StatusBadRequest)
			return
		}
		var req struct {
			Keep    string   `json:"keep"`
			Sources []string `json:"sources"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		g, found, err := media.GetDuplicateGroup(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !found {
			httpError(w, "no such duplicate group", http.StatusNotFound)
			return
		}
		members := map[string]media.DuplicateMember{}
		for _, m := range g.Members {
			members[m.Path] = m
		}
		keep := strings.TrimSpace(req.Keep)
		if keep == "" {
			// No explicit choice: the shared keeper rule (highest resolution,
			// then largest file), never blindly the anchor.
			k, _, err := media.DuplicateGroupKeeper(deps.DB, g)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			keep = k
		}
		if _, ok := members[keep]; !ok {
			httpError(w, "keep must be a member of the group", http.StatusBadRequest)
			return
		}
		var sources []string
		if len(req.Sources) > 0 {
			for _, s := range req.Sources {
				s = strings.TrimSpace(s)
				if s == "" || s == keep {
					continue
				}
				if _, ok := members[s]; !ok {
					httpError(w, "sources must be members of the group: "+s, http.StatusBadRequest)
					return
				}
				sources = append(sources, s)
			}
		} else {
			for _, m := range g.Members {
				if m.Path != keep && !m.Excluded {
					sources = append(sources, m.Path)
				}
			}
		}
		if len(sources) == 0 {
			httpError(w, "nothing to merge: the group has no other active members", http.StatusBadRequest)
			return
		}

		res, err := media.MergeInto(r.Context(), deps.DB, keep, sources)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if res.FacesRemoved > 0 {
			broadcastPeopleChanged()
		}
		// Member rows of the deleted sources are gone (RemoveItemsFromDB), and
		// the group was pruned if fewer than two remain; a failed deletion
		// keeps its member so the user sees what is still there.
		broadcastDuplicatesChanged()
		resp := map[string]any{"merge": res, "group": nil}
		if g, found, err := media.GetDuplicateGroup(deps.DB, id); err == nil && found {
			if v, err := duplicateGroupView(deps, g); err == nil {
				resp["group"] = v
			}
		}
		writeJSON(w, resp)
	}
}

func duplicatesWipeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			httpError(w, "use DELETE", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Query().Get("confirm") != "true" {
			httpError(w, "confirm=true required", http.StatusBadRequest)
			return
		}
		status := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("status")))
		switch status {
		case "", "all":
			status = ""
		case media.DuplicateStatusPending, media.DuplicateStatusDismissed:
		default:
			httpError(w, "status must be pending, dismissed, or all", http.StatusBadRequest)
			return
		}
		n, err := media.DeleteDuplicateGroups(deps.DB, status)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		broadcastDuplicatesChanged()
		writeJSON(w, map[string]any{"deleted": n})
	}
}
