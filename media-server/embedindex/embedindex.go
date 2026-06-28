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
// Implementations are NOT internally synchronized; callers must serialize all
// Add/Delete/Search calls (the server does this via tasks.vectorIndexMu).
type VectorIndex interface {
	Add(path string, vec []float32)
	Delete(path string)
	Search(query []float32, k int) []SearchHit
	Len() int
}

// hnswIndex wraps the HNSW graph with auxiliary maps:
//   - keys    — paths that are actively present (used by callers).
//   - ghosts  — paths removed via Delete but still present in the underlying
//     graph because hnsw v0.1.0 does not update its entry-point pointer on
//     deletion, which would cause nil-panics on subsequent Add/Search.
//     Ghosts are filtered out in Search; the graph is rebuilt from the
//     database on server restart, clearing them automatically.
//   - vectors — authoritative current vector per active path. Updated on
//     every Add (all three branches: active-refresh, ghost-reactivate, and
//     fresh-insert). Because hnsw v0.1.0 cannot move a node once inserted,
//     Search scores are computed against this overlay rather than
//     node.Embedding(), so results always reflect the latest embedding even
//     within a single session where a path was deleted and re-embedded.
type hnswIndex struct {
	g       *hnsw.Graph[hnsw.Vector]
	keys    map[string]struct{}  // active paths
	ghosts  map[string]struct{}  // tombstoned paths still in g
	vectors map[string][]float32 // authoritative current vector per active path
}

// New returns a fresh, empty VectorIndex backed by a cosine-distance HNSW graph.
func New() VectorIndex {
	g := hnsw.NewGraph[hnsw.Vector]()
	g.Distance = hnsw.CosineDistance
	return &hnswIndex{
		g:       g,
		keys:    map[string]struct{}{},
		ghosts:  map[string]struct{}{},
		vectors: map[string][]float32{},
	}
}

// Add inserts a vector for path, or refreshes the authoritative vector overlay
// if path is already active or is being reactivated from a soft-delete.
//
// In all three branches the vectors overlay is updated so Search always scores
// against the latest embedding regardless of the underlying graph topology.
//
//   - Active path: the graph node's position cannot be updated in hnsw v0.1.0;
//     the overlay ensures Search scoring is still correct.
//   - Ghost path: reactivated in place (calling g.Add again panics in hnsw
//     v0.1.0); the graph node retains its old position but the overlay makes
//     scoring correct.
//   - New path: inserted into both the graph and the overlay.
//
// vec must already be L2-normalised.
func (h *hnswIndex) Add(path string, vec []float32) {
	if _, exists := h.keys[path]; exists {
		// Already active. The graph node's position can't be updated in
		// hnsw v0.1.0, but refresh the authoritative vector so Search scores
		// reflect the latest embedding. (Graph topology may lag; it is rebuilt
		// from the DB on restart.)
		h.vectors[path] = vec
		return
	}
	if _, isGhost := h.ghosts[path]; isGhost {
		// Reactivate a soft-deleted node in place (g.Add of an existing key
		// panics in hnsw v0.1.0). The graph node retains its old position, but
		// the authoritative vector overlay makes Search scoring correct.
		delete(h.ghosts, path)
		h.keys[path] = struct{}{}
		h.vectors[path] = vec
		return
	}
	h.g.Add(hnsw.MakeVector(path, vec))
	h.keys[path] = struct{}{}
	h.vectors[path] = vec
}

// Delete removes path from the active set. It is a safe no-op if path is
// not present.
//
// We do NOT call g.Delete because hnsw v0.1.0 does not update its
// layer entry-point pointer on deletion. If the deleted node was the
// entry point, subsequent g.Add or g.Search calls would start from an
// isolated (nil-neighbor) node and panic. Instead the node is tombstoned
// and filtered from Search results. Ghost nodes are cleared automatically
// when the index is rebuilt from the database on server restart.
func (h *hnswIndex) Delete(path string) {
	if _, exists := h.keys[path]; !exists {
		return
	}
	delete(h.keys, path)
	delete(h.vectors, path)
	h.ghosts[path] = struct{}{}
}

// Search returns up to k nearest neighbours to query, ordered by descending
// cosine similarity (most similar first).
// query must already be L2-normalised.
func (h *hnswIndex) Search(query []float32, k int) []SearchHit {
	gLen := h.g.Len()
	if gLen == 0 {
		return nil
	}
	// Over-fetch by the ghost count so soft-deleted nodes ranking ahead of live
	// ones don't cause us to return fewer than k live results.
	fetch := k + len(h.ghosts)
	if fetch > gLen {
		fetch = gLen
	}
	nodes := h.g.Search(query, fetch)
	hits := make([]SearchHit, 0, k)
	for _, node := range nodes {
		id := node.ID()
		vec, present := h.vectors[id]
		if !present {
			continue // ghost / not active
		}
		// Score against the authoritative overlay vector, not node.Embedding(),
		// so the result reflects the latest embedding even when the graph
		// topology hasn't been rebuilt yet (graph positions are fixed in hnsw
		// v0.1.0 after first insertion).
		hits = append(hits, SearchHit{
			Path:  id,
			Score: 1 - h.g.Distance(query, vec),
		})
		if len(hits) == k {
			break
		}
	}
	return hits
}

// Len returns the number of active (non-deleted) vectors in the index.
func (h *hnswIndex) Len() int { return len(h.keys) }
