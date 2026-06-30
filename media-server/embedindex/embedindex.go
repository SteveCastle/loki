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

// hnswIndex wraps the HNSW graph with two auxiliary sets:
//   - keys   — paths that are actively present. Search results are filtered to
//     this set, and Len() reports its size.
//   - ghosts — paths removed via Delete but still resident in the underlying
//     graph, because hnsw v0.1.0 does not update its entry-point pointer on
//     deletion (calling g.Delete can leave a subsequent Search/Add starting
//     from an isolated node and panic). Ghosts are filtered out of Search
//     results and cleared when the index is rebuilt from the DB on restart.
//
// Memory: exactly ONE copy of each vector is kept — inside the HNSW graph.
// Search scores against node.Embedding() (the graph's own copy), so the index
// does not keep a second full copy of every vector (which at millions of items
// would cost gigabytes). The cost of this choice is a known, bounded staleness:
// hnsw v0.1.0 cannot move or replace a node's vector once inserted, so re-adding
// an already-present path (active, or reactivated from a soft-delete) does NOT
// update its vector until the next startup rebuild from the DB (which always
// reflects the current vector). In normal operation this never bites — the embed
// task skips already-embedded paths — and it only matters for delete-then-
// re-embed of the SAME path within a single session, where that path keeps
// scoring against its pre-delete vector until restart.
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

// Add inserts a vector for path. Re-adding an already-active path is a no-op,
// and re-adding a soft-deleted (ghost) path reactivates it in place; in both
// cases hnsw v0.1.0 cannot update the stored vector, so the node keeps its
// original vector until the next rebuild-from-DB at startup (see the type doc).
//
// vec must already be L2-normalised.
func (h *hnswIndex) Add(path string, vec []float32) {
	if _, exists := h.keys[path]; exists {
		return // already active; vector cannot be updated in place (see type doc)
	}
	if _, isGhost := h.ghosts[path]; isGhost {
		// Reactivate a soft-deleted node (g.Add of an existing key panics in
		// hnsw v0.1.0). The node retains its original graph vector.
		delete(h.ghosts, path)
		h.keys[path] = struct{}{}
		return
	}
	h.g.Add(hnsw.MakeVector(path, vec))
	h.keys[path] = struct{}{}
}

// Delete removes path from the active set (a safe no-op if absent). The node is
// tombstoned rather than removed from the graph — see the type doc for why.
func (h *hnswIndex) Delete(path string) {
	if _, exists := h.keys[path]; !exists {
		return
	}
	delete(h.keys, path)
	h.ghosts[path] = struct{}{}
}

// Search returns up to k nearest active neighbours to query, ordered by
// descending cosine similarity. query must already be L2-normalised.
func (h *hnswIndex) Search(query []float32, k int) []SearchHit {
	gLen := h.g.Len()
	if gLen == 0 {
		return nil
	}
	// Over-fetch by the ghost count so soft-deleted nodes ranking ahead of live
	// ones don't cause us to return fewer than k active results.
	fetch := k + len(h.ghosts)
	if fetch > gLen {
		fetch = gLen
	}
	nodes := h.g.Search(query, fetch)
	hits := make([]SearchHit, 0, k)
	for _, node := range nodes {
		id := node.ID()
		if _, active := h.keys[id]; !active {
			continue // ghost / not active
		}
		hits = append(hits, SearchHit{
			Path:  id,
			Score: 1 - h.g.Distance(query, node.Embedding()),
		})
		if len(hits) == k {
			break
		}
	}
	return hits
}

// Len returns the number of active (non-deleted) vectors in the index.
func (h *hnswIndex) Len() int { return len(h.keys) }
