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
	Delete(path string)
	Search(query []float32, k int) []SearchHit
	Len() int
}

// hnswIndex wraps the HNSW graph with two auxiliary sets:
//   - keys  — paths that are actively present (used by callers).
//   - ghosts — paths removed via Delete but still present in the underlying
//     graph because hnsw v0.1.0 does not update its entry-point pointer on
//     deletion, which would cause nil-panics on subsequent Add/Search.
//     Ghosts are filtered out in Search; the graph is rebuilt from the
//     database on server restart, clearing them automatically.
type hnswIndex struct {
	g      *hnsw.Graph[hnsw.Vector]
	keys   map[string]struct{} // active paths
	ghosts map[string]struct{} // tombstoned paths still in g
}

// New returns a fresh, empty VectorIndex backed by a cosine-distance HNSW graph.
func New() VectorIndex {
	g := hnsw.NewGraph[hnsw.Vector]()
	g.Distance = hnsw.CosineDistance
	return &hnswIndex{
		g:      g,
		keys:   map[string]struct{}{},
		ghosts: map[string]struct{}{},
	}
}

// Add inserts (or no-ops if already present) a vector for path.
//
// If path was previously Delete-d (ghost), we reactivate it using the
// vector that is still resident in the underlying graph rather than
// calling g.Add again — doing so would trigger the HNSW library's
// internal delete-then-reinsert path which panics on its Len() invariant
// check (github.com/coder/hnsw v0.1.0 bug).
//
// vec must already be L2-normalised.
func (h *hnswIndex) Add(path string, vec []float32) {
	if _, exists := h.keys[path]; exists {
		// Already active — no-op to prevent the "node not added" panic that
		// occurs when hnsw re-adds an existing key (internal delete+reinsert
		// leaves Len() unchanged, failing the library's own invariant check).
		return
	}
	if _, isGhost := h.ghosts[path]; isGhost {
		// The node is still in the underlying graph as a tombstone.
		// Reactivate in place; the old vector remains, which is acceptable
		// because the caller (embedTask) uses shouldSkipEmbed to avoid
		// re-embedding files that already have a stored vector.
		delete(h.ghosts, path)
		h.keys[path] = struct{}{}
		return
	}
	h.g.Add(hnsw.MakeVector(path, vec))
	h.keys[path] = struct{}{}
}

// Delete removes path from the active set. It is a safe no-op if path is
// not present.
//
// We do NOT call g.Delete because hnsw v0.1.0 does not update its
// layer entry-point pointer on deletion. If the deleted node was the
// entry point, subsequent g.Add or g.Search calls would start from an
// isolated (nil-neighbor) node and panic. Instead the node is tombstoned
// and filtered from Search results via the keys map. Ghost nodes are
// cleared automatically when the index is rebuilt from the database on
// server restart.
func (h *hnswIndex) Delete(path string) {
	if _, exists := h.keys[path]; !exists {
		return
	}
	delete(h.keys, path)
	h.ghosts[path] = struct{}{}
}

// Search returns up to k nearest neighbours to query, ordered by descending
// cosine similarity (most similar first).
// query must already be L2-normalised.
func (h *hnswIndex) Search(query []float32, k int) []SearchHit {
	// Guard: the underlying HNSW library panics when k exceeds the total graph
	// size (which may include ghost nodes from soft-deleted entries).
	gLen := h.g.Len()
	if gLen == 0 {
		return nil
	}
	if k > gLen {
		k = gLen
	}
	nodes := h.g.Search(query, k)
	hits := make([]SearchHit, 0, len(nodes))
	for _, node := range nodes {
		// Filter: only return nodes that are in the active keys set.
		// Ghost nodes (soft-deleted) remain in the HNSW graph but are excluded
		// here via the keys map.
		if _, present := h.keys[node.ID()]; !present {
			continue
		}
		// Distance returns lower-is-closer; convert to higher-is-better.
		hits = append(hits, SearchHit{
			Path:  node.ID(),
			Score: 1 - h.g.Distance(query, node.Embedding()),
		})
	}
	return hits
}

// Len returns the number of active (non-deleted) vectors in the index.
func (h *hnswIndex) Len() int { return len(h.keys) }
