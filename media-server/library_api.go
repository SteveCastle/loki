package main

import (
	"database/sql"
	"net/http"
	"strings"
)

// -----------------------------------------------------------------------------
// Library data API (shared across all platform mains).
//
//   POST /api/media/transcript — set or clear a media item's transcript
//   POST /api/media/rating     — read/set elo, views, wins, losses
//   GET  /api/tags/list        — all tags with usage counts (?category= filter)
// -----------------------------------------------------------------------------

func mediaTranscriptHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Path       string `json:"path"`
			Transcript string `json:"transcript"`
		}
		if err := readJSON(r, &req); err != nil || req.Path == "" {
			httpError(w, "bad request: path required", http.StatusBadRequest)
			return
		}
		res, err := deps.DB.Exec("UPDATE media SET transcript = ? WHERE path = ?", req.Transcript, req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if n, _ := res.RowsAffected(); n == 0 {
			httpError(w, "media not found", http.StatusNotFound)
			return
		}
		writeJSON(w, map[string]string{"status": "ok"})
	}
}

// mediaRatingHandler reads and writes the rating columns. Only fields present
// in the request are updated; a body with just {"path": ...} reads.
func mediaRatingHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Path   string   `json:"path"`
			Elo    *float64 `json:"elo"`
			Views  *int64   `json:"views"`
			Wins   *int64   `json:"wins"`
			Losses *int64   `json:"losses"`
		}
		if err := readJSON(r, &req); err != nil || req.Path == "" {
			httpError(w, "bad request: path required", http.StatusBadRequest)
			return
		}

		var sets []string
		var args []any
		if req.Elo != nil {
			sets, args = append(sets, "elo = ?"), append(args, *req.Elo)
		}
		if req.Views != nil {
			sets, args = append(sets, "views = ?"), append(args, *req.Views)
		}
		if req.Wins != nil {
			sets, args = append(sets, "wins = ?"), append(args, *req.Wins)
		}
		if req.Losses != nil {
			sets, args = append(sets, "losses = ?"), append(args, *req.Losses)
		}
		if len(sets) > 0 {
			args = append(args, req.Path)
			res, err := deps.DB.Exec(
				"UPDATE media SET "+strings.Join(sets, ", ")+" WHERE path = ?", args...)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			if n, _ := res.RowsAffected(); n == 0 {
				httpError(w, "media not found", http.StatusNotFound)
				return
			}
		}

		var elo sql.NullFloat64
		var views, wins, losses sql.NullInt64
		err := deps.DB.QueryRow(
			"SELECT elo, views, wins, losses FROM media WHERE path = ?", req.Path).
			Scan(&elo, &views, &wins, &losses)
		if err == sql.ErrNoRows {
			httpError(w, "media not found", http.StatusNotFound)
			return
		} else if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		nullable := func(v any, valid bool) any {
			if !valid {
				return nil
			}
			return v
		}
		writeJSON(w, map[string]any{
			"path":   req.Path,
			"elo":    nullable(elo.Float64, elo.Valid),
			"views":  nullable(views.Int64, views.Valid),
			"wins":   nullable(wins.Int64, wins.Valid),
			"losses": nullable(losses.Int64, losses.Valid),
		})
	}
}

// tagsListHandler returns every tag with its category, weight, and usage count
// (distinct media). ?category= filters to one category.
func tagsListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		query := `
			SELECT t.label, COALESCE(t.category_label, ''), COALESCE(t.weight, 0),
			       COUNT(DISTINCT m.media_path)
			FROM tag t
			LEFT JOIN media_tag_by_category m ON m.tag_label = t.label`
		var args []any
		if category := r.URL.Query().Get("category"); category != "" {
			query += ` WHERE t.category_label = ?`
			args = append(args, category)
		}
		query += ` GROUP BY t.label ORDER BY COALESCE(t.category_label, ''), t.label`

		rows, err := deps.DB.Query(query, args...)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		tags := []map[string]any{}
		for rows.Next() {
			var label, category string
			var weight float64
			var count int
			if err := rows.Scan(&label, &category, &weight, &count); err != nil {
				continue
			}
			tags = append(tags, map[string]any{
				"label":       label,
				"category":    category,
				"weight":      weight,
				"media_count": count,
			})
		}
		writeJSON(w, map[string]any{"tags": tags})
	}
}
