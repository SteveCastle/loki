package main

import (
	"net/http"
	"strconv"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/tasks"
)

// -----------------------------------------------------------------------------
// Embeddings-index management API (shared across all platform mains).
//
//   GET    /api/index/status      — installed index + per-model DB stats
//   GET    /api/index/models      — embed model registry
//   POST   /api/index/rebuild     — rebuild vector index for the active model
//   GET    /api/index/missing     — media paths lacking an embedding
//   GET    /api/embeddings        — stored embedding rows for one path
//   DELETE /api/embeddings        — delete stored embedding rows for one path
//   POST   /api/embeddings/prune  — drop embeddings whose media row is gone
// -----------------------------------------------------------------------------

func indexStatusHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}

		type modelStats struct {
			Model string `json:"model"`
			Count int    `json:"count"`
			Dim   int    `json:"dim"`
			Bytes int64  `json:"bytes"` // stored vector blob bytes for this model
		}
		stats := []modelStats{}
		var totalCount int
		var totalBytes int64
		rows, err := deps.DB.Query(
			`SELECT model, COUNT(*), MAX(dim), COALESCE(SUM(LENGTH(vector)), 0)
			 FROM media_embedding GROUP BY model ORDER BY model`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for rows.Next() {
			var s modelStats
			if err := rows.Scan(&s.Model, &s.Count, &s.Dim, &s.Bytes); err == nil {
				stats = append(stats, s)
				totalCount += s.Count
				totalBytes += s.Bytes
			}
		}
		rows.Close()

		var mediaTotal, orphaned, missing int
		deps.DB.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&mediaTotal)
		deps.DB.QueryRow(`
			SELECT COUNT(*) FROM media_embedding e
			LEFT JOIN media m ON m.path = e.media_path
			WHERE m.path IS NULL`).Scan(&orphaned)

		active := tasks.ActiveEmbedModel()
		deps.DB.QueryRow(`
			SELECT COUNT(*) FROM media m
			LEFT JOIN media_embedding e ON e.media_path = m.path AND e.model = ?
			WHERE e.media_path IS NULL`, active.ID).Scan(&missing)

		indexedModel := tasks.IndexedModel()
		writeJSON(w, map[string]any{
			"index": map[string]any{
				"installed": indexedModel != "" || tasks.IndexSize() > 0,
				"model":     indexedModel,
				"vectors":   tasks.IndexSize(),
			},
			"active_model":         active.ID,
			"media_total":          mediaTotal,
			"missing_active_model": missing,
			"orphaned":             orphaned,
			"embeddings":           stats,
			"total_count":          totalCount,
			"total_bytes":          totalBytes,
		})
	}
}

// embeddingsWipeHandler is the embeddings twin of facesWipeHandler:
// DELETE /api/embeddings/all wipes every stored embedding row (ALL models)
// and clears the in-memory search index. Requires ?confirm=true — not
// undoable, though re-running the embed task rebuilds everything.
func embeddingsWipeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			httpError(w, "use DELETE", http.StatusMethodNotAllowed)
			return
		}
		if r.URL.Query().Get("confirm") != "true" {
			httpError(w, "add ?confirm=true to delete all stored embeddings", http.StatusBadRequest)
			return
		}
		res, err := deps.DB.Exec(`DELETE FROM media_embedding`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		deleted, _ := res.RowsAffected()
		// Drop the live search index too — similarity search falls back to a
		// brute-force scan over the (now empty) table until the next rebuild.
		tasks.SetVectorIndexForModel(nil, "")
		writeJSON(w, map[string]any{"deleted": deleted})
	}
}

func indexModelsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		activeID := tasks.ActiveEmbedModel().ID
		indexedID := tasks.IndexedModel()
		models := []map[string]any{}
		for _, m := range tasks.EmbedModelList() {
			models = append(models, map[string]any{
				"id":           m.ID,
				"display_name": m.DisplayName,
				"dim":          m.Dim,
				"multimodal":   m.Multimodal,
				"active":       m.ID == activeID,
				"indexed":      m.ID == indexedID,
			})
		}
		writeJSON(w, map[string]any{"models": models})
	}
}

func indexRebuildHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}
		model, count, err := tasks.RebuildActiveIndex(deps.DB, nil)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"status": "ok", "model": model, "vectors": count})
	}
}

func indexMissingHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		model := r.URL.Query().Get("model")
		if model == "" {
			model = tasks.ActiveEmbedModel().ID
		} else if _, ok := tasks.EmbedModelByID(model); !ok {
			httpError(w, "unknown model "+model, http.StatusBadRequest)
			return
		}
		limit := 100
		if v := r.URL.Query().Get("limit"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n < 0 {
				httpError(w, "invalid limit", http.StatusBadRequest)
				return
			}
			limit = min(n, 10000)
		}

		var total int
		deps.DB.QueryRow(`
			SELECT COUNT(*) FROM media m
			LEFT JOIN media_embedding e ON e.media_path = m.path AND e.model = ?
			WHERE e.media_path IS NULL`, model).Scan(&total)

		paths := []string{}
		if limit > 0 {
			rows, err := deps.DB.Query(`
				SELECT m.path FROM media m
				LEFT JOIN media_embedding e ON e.media_path = m.path AND e.model = ?
				WHERE e.media_path IS NULL ORDER BY m.path LIMIT ?`, model, limit)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer rows.Close()
			for rows.Next() {
				var p string
				if err := rows.Scan(&p); err == nil {
					paths = append(paths, p)
				}
			}
		}
		writeJSON(w, map[string]any{"model": model, "total_missing": total, "paths": paths})
	}
}

// embeddingsHandler serves GET (inspect) and DELETE (remove) for one media
// path's stored embedding rows, addressed by ?path= and optionally ?model=.
func embeddingsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Query().Get("path")
		if path == "" {
			httpError(w, "path required", http.StatusBadRequest)
			return
		}
		model := r.URL.Query().Get("model")

		switch r.Method {
		case http.MethodGet:
			includeVector := r.URL.Query().Get("vector") == "true"
			query := `SELECT model, dim, created_at, vector FROM media_embedding WHERE media_path = ?`
			args := []any{path}
			if model != "" {
				query += ` AND model = ?`
				args = append(args, model)
			}
			rows, err := deps.DB.Query(query, args...)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer rows.Close()

			embeddings := []map[string]any{}
			for rows.Next() {
				var m string
				var dim int
				var createdAt int64
				var blob []byte
				if err := rows.Scan(&m, &dim, &createdAt, &blob); err != nil {
					continue
				}
				row := map[string]any{"model": m, "dim": dim, "created_at": createdAt}
				if includeVector {
					if vec, err := embedvec.Decode(blob); err == nil {
						row["vector"] = vec
					}
				}
				embeddings = append(embeddings, row)
			}
			writeJSON(w, map[string]any{"path": path, "embeddings": embeddings})

		case http.MethodDelete:
			query := `DELETE FROM media_embedding WHERE media_path = ?`
			args := []any{path}
			if model != "" {
				query += ` AND model = ?`
				args = append(args, model)
			}
			res, err := deps.DB.Exec(query, args...)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			n, _ := res.RowsAffected()
			// Drop the path from the live index when its vectors are gone for
			// the indexed model (Delete tombstones; absent paths are a no-op).
			if model == "" || model == tasks.IndexedModel() {
				tasks.IndexDelete(path)
			}
			writeJSON(w, map[string]any{"deleted": n})

		default:
			httpError(w, "use GET or DELETE", http.StatusMethodNotAllowed)
		}
	}
}

// embeddingsPruneHandler deletes embedding rows whose media row no longer
// exists (orphans left behind by out-of-band deletions).
func embeddingsPruneHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST", http.StatusMethodNotAllowed)
			return
		}

		rows, err := deps.DB.Query(`
			SELECT DISTINCT e.media_path FROM media_embedding e
			LEFT JOIN media m ON m.path = e.media_path
			WHERE m.path IS NULL`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var paths []string
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err == nil {
				paths = append(paths, p)
			}
		}
		rows.Close()

		res, err := deps.DB.Exec(`
			DELETE FROM media_embedding WHERE media_path IN (
				SELECT e.media_path FROM media_embedding e
				LEFT JOIN media m ON m.path = e.media_path
				WHERE m.path IS NULL)`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		deleted, _ := res.RowsAffected()
		for _, p := range paths {
			tasks.IndexDelete(p)
		}
		writeJSON(w, map[string]any{"pruned_rows": deleted, "pruned_paths": len(paths)})
	}
}
