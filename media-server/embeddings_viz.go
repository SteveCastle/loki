package main

import (
	_ "embed"
	"net/http"
	"strconv"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/tasks"
)

// -----------------------------------------------------------------------------
// Embedding-space visualization (shared across all platform mains).
//
//   GET /viz/embeddings              — WebGL 3D scatter of the embedding space
//   GET /api/embeddings/projection   — PCA-projected [x,y,z] points for a model
//
// The page lets you orbit/zoom a 3D projection of the vector index, hover
// points for media previews, and switch between embedding models to compare
// how each one clusters the same library.
// -----------------------------------------------------------------------------

//go:embed vizstatic/embeddings.html
var embeddingsVizHTML []byte

// RegisterVizRoutes wires the visualization page + its data API onto mux.
// Called from each platform's main file alongside the other API routes.
func RegisterVizRoutes(mux *http.ServeMux, deps *Dependencies) {
	mux.HandleFunc("/viz/embeddings", renderer.ApplyMiddlewares(embeddingsVizPageHandler(), renderer.RoleAdmin))
	mux.HandleFunc("/api/embeddings/projection", renderer.ApplyMiddlewares(embeddingsProjectionHandler(deps), renderer.RoleAdmin))
}

func embeddingsVizPageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(embeddingsVizHTML)
	}
}

// embeddingsProjectionHandler reduces a model's stored embeddings to 3D via
// PCA and returns them as parallel paths/points arrays. When the library
// exceeds ?limit (default 4000, max 20000) it stride-samples deterministically
// so reloads show the same cloud. ?model accepts any model with stored rows —
// including ones no longer in the registry — so old and new models can be
// compared side by side.
func embeddingsProjectionHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		model := r.URL.Query().Get("model")
		if model == "" {
			model = tasks.ActiveEmbedModel().ID
		}
		limit := 4000
		if v := r.URL.Query().Get("limit"); v != "" {
			n, err := strconv.Atoi(v)
			if err != nil || n <= 0 {
				httpError(w, "invalid limit", http.StatusBadRequest)
				return
			}
			limit = n
		}
		if limit > 20000 {
			limit = 20000
		}

		all, err := media.LoadAllEmbeddings(deps.DB, model)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Rows must share one dimension for PCA; anchor on the first row's dim
		// and drop strays (e.g. rows written mid-migration by another model
		// build). LoadAllEmbeddings returns a stable DB order, keeping the
		// stride sample below deterministic across reloads.
		var dim int
		rows := all[:0]
		for _, e := range all {
			if dim == 0 {
				dim = len(e.Vec)
			}
			if len(e.Vec) == dim && dim > 0 {
				rows = append(rows, e)
			}
		}
		total := len(rows)

		// Deterministic stride sample down to limit.
		sampled := rows
		if total > limit {
			sampled = make([]media.StoredEmbedding, 0, limit)
			stride := float64(total) / float64(limit)
			for i := 0; i < limit; i++ {
				sampled = append(sampled, rows[int(float64(i)*stride)])
			}
		}

		// Normalize (embedding similarity is cosine, so unit-sphere geometry
		// is what the visualization should reflect), then project.
		paths := make([]string, len(sampled))
		vecs := make([][]float32, len(sampled))
		for i, e := range sampled {
			paths[i] = e.Path
			vecs[i] = embedvec.Normalize(e.Vec)
		}
		points, variance := embedvec.ProjectPCA3(vecs)

		writeJSON(w, map[string]any{
			"model":    model,
			"dim":      dim,
			"total":    total,
			"count":    len(points),
			"variance": []float64{variance[0], variance[1], variance[2]},
			"paths":    paths,
			"points":   points,
		})
	}
}
