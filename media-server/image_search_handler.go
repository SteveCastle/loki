package main

import (
	"database/sql"
	"errors"
	"io"
	"net/http"

	"github.com/stevecastle/shrike/tasks"
)

// enrichScoredItems turns ranked (path,score) hits into the flat item shape the
// renderer expects (matching lokiMediaQueryHandler), reading media metadata for
// each path and attaching score, sorted by score descending.
func enrichScoredItems(db *sql.DB, hits []tasks.SimilarHit) ([]map[string]any, error) {
	scoreByPath := make(map[string]float32, len(hits))
	items := make([]map[string]any, 0, len(hits))
	for _, h := range hits {
		scoreByPath[h.Path] = h.Score
		item := map[string]any{"path": h.Path, "mtimeMs": int64(0)}
		var elo sql.NullFloat64
		var height, width sql.NullInt64
		err := db.QueryRow(
			`SELECT elo, height, width FROM media WHERE path = ?`, h.Path,
		).Scan(&elo, &height, &width)
		if err == nil {
			if elo.Valid {
				item["elo"] = elo.Float64
			}
			if height.Valid {
				item["height"] = height.Int64
			}
			if width.Valid {
				item["width"] = width.Int64
			}
		} // a missing media row (orphan embedding) still yields a path+score item
		items = append(items, item)
	}
	sortItemsByScore(items, scoreByPath) // existing helper in loki_api.go
	return items, nil
}

func lokiImageSearchHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		image, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20)) // 32 MB cap
		if err != nil {
			if errors.As(err, new(*http.MaxBytesError)) {
				httpError(w, "image too large (max 32 MB)", http.StatusRequestEntityTooLarge)
			} else {
				httpError(w, "image body required", http.StatusBadRequest)
			}
			return
		}
		if len(image) == 0 {
			httpError(w, "image body required", http.StatusBadRequest)
			return
		}
		hits, err := tasks.SearchByImage(r.Context(), deps.DB, image, 50)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		items, err := enrichScoredItems(deps.DB, hits)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, items)
	}
}
