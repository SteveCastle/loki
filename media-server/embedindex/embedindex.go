// Package embedindex wraps a pure-Go HNSW graph behind a small interface so
// the rest of the server depends only on VectorIndex, not the ANN library.
package embedindex

import (
	"github.com/coder/hnsw"
)

// SearchHit is a single result from a Search call.
type SearchHit struct {
	Path  string
	Score float32 // cosine similarity: higher is more similar (1.0 = identical)
}

// VectorIndex is the narrow interface the server uses for ANN vector search.
type VectorIndex interface {
	Add(path string, vec []float32)
	Search(query []float32, k int) []SearchHit
	Len() int
}

type hnswIndex struct {
	g *hnsw.Graph[hnsw.Vector]
}

// New returns a fresh, empty VectorIndex backed by a cosine-distance HNSW graph.
func New() VectorIndex {
	g := hnsw.NewGraph[hnsw.Vector]()
	g.Distance = hnsw.CosineDistance
	return &hnswIndex{g: g}
}

// Add inserts a vector associated with path into the index.
// vec must already be L2-normalised.
func (h *hnswIndex) Add(path string, vec []float32) {
	h.g.Add(hnsw.MakeVector(path, vec))
}

// Search returns the k nearest neighbours to query, ordered by descending
// cosine similarity (most similar first).
// query must already be L2-normalised.
func (h *hnswIndex) Search(query []float32, k int) []SearchHit {
	nodes := h.g.Search(query, k)
	hits := make([]SearchHit, 0, len(nodes))
	for _, n := range nodes {
		// Distance returns lower-is-closer; convert to higher-is-better.
		hits = append(hits, SearchHit{
			Path:  n.ID(),
			Score: 1 - h.g.Distance(query, n.Embedding()),
		})
	}
	return hits
}

// Len returns the number of vectors currently in the index.
func (h *hnswIndex) Len() int { return h.g.Len() }
