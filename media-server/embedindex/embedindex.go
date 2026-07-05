// Package embedindex provides the in-memory vector index used for visual
// similarity search. It performs an EXACT, parallelized cosine scan over all
// vectors rather than approximate nearest-neighbour search.
//
// Why exact scan and not ANN: the index previously wrapped coder/hnsw v0.1.0,
// which delivered very poor recall in practice — its default EfSearch (20) was
// far below the k values the server requests (50 for "find similar", 1000 for
// visual predicates in composed queries), and its search loop terminates as
// soon as one expansion fails to improve the single best hit, so everything
// beyond the top few results was whatever nodes the greedy descent happened to
// visit. Symptoms: the query image itself often missing from its own results,
// and clearly-similar items outranked by dissimilar ones. The graph also could
// not delete nodes or update vectors in place, forcing ghost/tombstone
// workarounds here.
//
// The exact scan fixes all of that with the same memory footprint (the HNSW
// graph already kept every vector in RAM): scores are true cosine
// similarities, ranking is exact (the query item scores ~1.0 and ranks first),
// deletes are real, and re-adding a path updates its vector immediately. Cost
// is O(N·dim) per search, parallelized across CPUs — roughly 1ms per 10k
// vectors at dim 768, comfortably interactive at library scale.
package embedindex

import (
	"runtime"
	"sort"
	"sync"

	"github.com/stevecastle/shrike/embedvec"
)

// SearchHit is a single result from a Search call.
type SearchHit struct {
	Path  string
	Score float32 // cosine similarity: higher is more similar (1.0 = identical)
}

// VectorIndex is the narrow interface the server uses for vector search.
// Implementations are NOT internally synchronized; callers must serialize all
// Add/Delete/Search calls (the server does this via tasks.vectorIndexMu).
type VectorIndex interface {
	// Add inserts or replaces the vector for path. The vector is L2-normalized
	// internally, so callers may pass raw model output.
	Add(path string, vec []float32)
	// Delete removes path from the index (safe no-op if absent).
	Delete(path string)
	// Search returns up to k nearest neighbours to query, ordered by
	// descending cosine similarity. The query is L2-normalized internally.
	Search(query []float32, k int) []SearchHit
	// Len returns the number of vectors in the index.
	Len() int
}

// exactIndex stores one normalized vector per path in parallel slices (for
// cache-friendly scanning) plus a path→slot map for O(1) update and delete
// (delete swap-removes the last slot into the hole).
type exactIndex struct {
	paths []string
	vecs  [][]float32
	slot  map[string]int
}

// New returns a fresh, empty exact-scan VectorIndex.
func New() VectorIndex {
	return &exactIndex{slot: map[string]int{}}
}

func (x *exactIndex) Add(path string, vec []float32) {
	v := embedvec.Normalize(vec)
	if i, ok := x.slot[path]; ok {
		x.vecs[i] = v // in-place vector update (e.g. re-embed after edit)
		return
	}
	x.slot[path] = len(x.paths)
	x.paths = append(x.paths, path)
	x.vecs = append(x.vecs, v)
}

func (x *exactIndex) Delete(path string) {
	i, ok := x.slot[path]
	if !ok {
		return
	}
	last := len(x.paths) - 1
	if i != last {
		x.paths[i] = x.paths[last]
		x.vecs[i] = x.vecs[last]
		x.slot[x.paths[i]] = i
	}
	x.paths = x.paths[:last]
	x.vecs = x.vecs[:last]
	delete(x.slot, path)
}

func (x *exactIndex) Len() int { return len(x.paths) }

func (x *exactIndex) Search(query []float32, k int) []SearchHit {
	n := len(x.paths)
	if n == 0 || k <= 0 {
		return nil
	}
	q := embedvec.Normalize(query)

	// Score every vector, parallelized in contiguous chunks.
	scores := make([]float32, n)
	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				scores[i] = embedvec.Cosine(q, x.vecs[i])
			}
		}(lo, hi)
	}
	wg.Wait()

	if k > n {
		k = n
	}
	top := topKIndices(scores, k)

	hits := make([]SearchHit, 0, k)
	for _, i := range top {
		hits = append(hits, SearchHit{Path: x.paths[i], Score: scores[i]})
	}
	// Deterministic order: score descending, then path ascending on ties.
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Score != hits[b].Score {
			return hits[a].Score > hits[b].Score
		}
		return hits[a].Path < hits[b].Path
	})
	return hits
}

// topKIndices selects the indices of the k largest scores using a min-heap of
// size k (O(N log k)); most items short-circuit on a single compare with the
// heap root.
func topKIndices(scores []float32, k int) []int {
	h := make([]int, 0, k) // heap of indices; h[0] holds the smallest score
	siftDown := func(pos int) {
		for {
			l, r := 2*pos+1, 2*pos+2
			small := pos
			if l < len(h) && scores[h[l]] < scores[h[small]] {
				small = l
			}
			if r < len(h) && scores[h[r]] < scores[h[small]] {
				small = r
			}
			if small == pos {
				return
			}
			h[pos], h[small] = h[small], h[pos]
			pos = small
		}
	}
	siftUp := func(pos int) {
		for pos > 0 {
			parent := (pos - 1) / 2
			if scores[h[parent]] <= scores[h[pos]] {
				return
			}
			h[pos], h[parent] = h[parent], h[pos]
			pos = parent
		}
	}
	for i := range scores {
		if len(h) < k {
			h = append(h, i)
			siftUp(len(h) - 1)
		} else if scores[i] > scores[h[0]] {
			h[0] = i
			siftDown(0)
		}
	}
	return h
}
