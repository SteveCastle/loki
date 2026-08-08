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
//
// Mean-centering (NewCentered): image-embedding spaces carry a large common
// component every vector shares ("is a photo"), which compresses the useful
// dynamic range of cosine similarity and creates hub items that score well
// against every query. A centered index subtracts the library's mean vector
// from both query and candidates before comparing (the "all-but-the-top"
// trick), sharpening rankings. Scores become centered cosines: still 1.0 for
// the query item itself, but no longer comparable to plain-cosine thresholds —
// consumers that gate on absolute cosine values (the face-identity indexes and
// their calibrated thresholds) must keep using New().
package embedindex

import (
	"math"
	"runtime"
	"sort"
	"sync"

	"github.com/stevecastle/shrike/embedvec"
)

// MinCenterCount is the minimum number of vectors before a centered index
// actually centers: below it the mean is a statistic of almost nothing and
// centering would distort more than it corrects, so the index scores plain
// cosine exactly like New() until the library grows past this size.
const MinCenterCount = 64

// SearchHit is a single result from a Search call.
type SearchHit struct {
	Path  string
	Score float32 // cosine similarity: higher is more similar (1.0 = identical)
}

// SharedSpec is the input to SearchShared: a shared-concept ("must match all")
// query over several positive example vectors, optionally steered away from
// negative ones. Weights are magnitudes (PosW nil = equal); all vectors must
// be in the index's embedding space.
type SharedSpec struct {
	Pos  [][]float32
	PosW []float32
	Neg  [][]float32
	NegW []float32
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
	// SearchFiltered is Search restricted to the paths in allow (nil = no
	// restriction). Only allowed vectors are scored, so a narrow allow set
	// costs less than a full scan, not more.
	SearchFiltered(query []float32, k int, allow map[string]struct{}) []SearchHit
	// SearchShared ranks candidates by similarity to ALL of the spec's
	// positive examples at once (see embedvec.SharedQuery) instead of to
	// their average, restricted to allow when non-nil.
	SearchShared(spec SharedSpec, k int, allow map[string]struct{}) ([]SearchHit, error)
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

	// Mean-centering state (NewCentered only). sum accumulates the stored
	// (normalized) vectors so the mean is O(dim) to produce; mu/muN2/cnorm are
	// the center actually used for scoring, refreshed lazily by maybeRecenter
	// once the index has drifted enough — nil mu = centering dormant. cnorm[i]
	// caches ‖vecs[i]−mu‖ so centered scoring costs one extra multiply per
	// candidate instead of a vector pass. The index holds one model's vectors
	// (tasks guards this via vectorIndexModel), so all dims match.
	centering bool
	sum       []float64
	mu        []float32
	muN2      float64
	cnorm     []float32
	nCentered int
}

// New returns a fresh, empty exact-scan VectorIndex that scores plain cosine.
// Use it whenever absolute cosine values matter (face thresholds).
func New() VectorIndex {
	return &exactIndex{slot: map[string]int{}}
}

// NewCentered returns an exact-scan VectorIndex that mean-centers both query
// and candidates before comparing (see the package comment), once it holds at
// least MinCenterCount vectors. Rankings sharpen; absolute scores shift.
func NewCentered() VectorIndex {
	return &exactIndex{slot: map[string]int{}, centering: true}
}

func (x *exactIndex) sumAdd(v []float32) {
	if !x.centering {
		return
	}
	if x.sum == nil {
		x.sum = make([]float64, len(v))
	}
	for d, s := range v {
		x.sum[d] += float64(s)
	}
}

func (x *exactIndex) sumSub(v []float32) {
	if !x.centering || x.sum == nil {
		return
	}
	for d, s := range v {
		x.sum[d] -= float64(s)
	}
}

func (x *exactIndex) Add(path string, vec []float32) {
	v := embedvec.Normalize(vec)
	if i, ok := x.slot[path]; ok {
		x.sumSub(x.vecs[i])
		x.vecs[i] = v // in-place vector update (e.g. re-embed after edit)
		x.sumAdd(v)
		if x.mu != nil {
			x.cnorm[i] = x.centeredNorm(v)
		}
		return
	}
	x.slot[path] = len(x.paths)
	x.paths = append(x.paths, path)
	x.vecs = append(x.vecs, v)
	x.sumAdd(v)
	if x.mu != nil {
		x.cnorm = append(x.cnorm, x.centeredNorm(v))
	}
}

func (x *exactIndex) Delete(path string) {
	i, ok := x.slot[path]
	if !ok {
		return
	}
	x.sumSub(x.vecs[i])
	last := len(x.paths) - 1
	if i != last {
		x.paths[i] = x.paths[last]
		x.vecs[i] = x.vecs[last]
		x.slot[x.paths[i]] = i
		if x.cnorm != nil {
			x.cnorm[i] = x.cnorm[last]
		}
	}
	x.paths = x.paths[:last]
	x.vecs = x.vecs[:last]
	if x.cnorm != nil {
		x.cnorm = x.cnorm[:last]
	}
	delete(x.slot, path)
}

func (x *exactIndex) Len() int { return len(x.paths) }

// centeredNorm returns ‖v−mu‖ for unit v: √(1 − 2⟨v,mu⟩ + ‖mu‖²).
func (x *exactIndex) centeredNorm(v []float32) float32 {
	n2 := 1 - 2*float64(embedvec.Cosine(v, x.mu)) + x.muN2
	if n2 < 0 {
		n2 = 0
	}
	return float32(math.Sqrt(n2))
}

// maybeRecenter refreshes the centering state when the index has grown or
// shrunk ~10% past the last refresh (or has never centered). Callers already
// serialize Search with Add/Delete, so mutating here is safe. Cost is one
// O(N·dim) parallel pass — the same as a search — paid rarely.
func (x *exactIndex) maybeRecenter() {
	if !x.centering {
		return
	}
	n := len(x.paths)
	if n < MinCenterCount {
		x.mu, x.cnorm, x.nCentered = nil, nil, n
		return
	}
	if x.mu != nil {
		drift := n - x.nCentered
		if drift < 0 {
			drift = -drift
		}
		if drift <= x.nCentered/10+32 {
			return
		}
	}
	mu := make([]float32, len(x.sum))
	var n2 float64
	for d, s := range x.sum {
		m := s / float64(n)
		mu[d] = float32(m)
		n2 += m * m
	}
	x.mu, x.muN2 = mu, n2
	cn := make([]float32, n)
	parallelChunks(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			cn[i] = x.centeredNorm(x.vecs[i])
		}
	})
	x.cnorm = cn
	x.nCentered = n
}

// singleScorer returns a per-worker scorer factory for one query vector:
// plain cosine when centering is dormant, centered cosine otherwise —
// (⟨qc,v⟩ − ⟨qc,mu⟩)/‖v−mu‖ with qc = normalize(q−mu), algebraically equal to
// cos(qc, v−mu) so no per-candidate vector pass is needed.
func (x *exactIndex) singleScorer(query []float32) func() func(slot int) float32 {
	q := embedvec.Normalize(query)
	if x.mu == nil || len(q) != len(x.mu) {
		return func() func(int) float32 {
			return func(i int) float32 { return embedvec.Cosine(q, x.vecs[i]) }
		}
	}
	qc := make([]float32, len(q))
	for d := range q {
		qc[d] = q[d] - x.mu[d]
	}
	qc = embedvec.Normalize(qc)
	cq := embedvec.Cosine(qc, x.mu)
	return func() func(int) float32 {
		return func(i int) float32 {
			cn := x.cnorm[i]
			if cn < 1e-6 {
				return 0 // candidate sits at the library mean: no direction
			}
			return (embedvec.Cosine(qc, x.vecs[i]) - cq) / cn
		}
	}
}

func (x *exactIndex) Search(query []float32, k int) []SearchHit {
	x.maybeRecenter()
	return x.scanTopK(nil, k, x.singleScorer(query))
}

// SearchFiltered scans only the vectors whose path is in allow, so the cost is
// O(|allow|·dim) rather than O(N·dim).
func (x *exactIndex) SearchFiltered(query []float32, k int, allow map[string]struct{}) []SearchHit {
	if allow == nil {
		return x.Search(query, k)
	}
	x.maybeRecenter()
	slots := x.collectSlots(allow)
	if len(slots) == 0 {
		return nil
	}
	return x.scanTopK(slots, k, x.singleScorer(query))
}

// SearchShared ranks candidates against every positive example at once with
// intersection semantics (embedvec.SharedQuery): per-example centered cosines
// with the examples' variation subspace attenuated, aggregated as weighted
// mean − λ·std, minus weighted negative-term cosines.
func (x *exactIndex) SearchShared(spec SharedSpec, k int, allow map[string]struct{}) ([]SearchHit, error) {
	x.maybeRecenter()
	// Move the query vectors into scoring space: normalized, and centered when
	// centering is active (candidates are centered per-slot in the scorer).
	prep := func(vs [][]float32) [][]float32 {
		out := make([][]float32, len(vs))
		for i, v := range vs {
			n := embedvec.Normalize(v)
			if x.mu != nil && len(n) == len(x.mu) {
				c := make([]float32, len(n))
				for d := range n {
					c[d] = n[d] - x.mu[d]
				}
				n = embedvec.Normalize(c)
			}
			out[i] = n
		}
		return out
	}
	sq, err := embedvec.NewSharedQuery(prep(spec.Pos), spec.PosW, prep(spec.Neg), spec.NegW,
		embedvec.DefaultSharedBeta, embedvec.DefaultSharedLambda)
	if err != nil {
		return nil, err
	}
	var slots []int
	if allow != nil {
		slots = x.collectSlots(allow)
		if len(slots) == 0 {
			return nil, nil
		}
	}
	dim := 0
	if len(x.vecs) > 0 {
		dim = len(x.vecs[0])
	}
	newScorer := func() func(int) float32 {
		score := sq.Scorer()
		if x.mu == nil {
			return func(i int) float32 { return score(x.vecs[i]) }
		}
		scratch := make([]float32, dim)
		return func(i int) float32 {
			cn := x.cnorm[i]
			if cn < 1e-6 {
				return 0
			}
			inv := 1 / cn
			v := x.vecs[i]
			for d := range v {
				scratch[d] = (v[d] - x.mu[d]) * inv
			}
			return score(scratch)
		}
	}
	return x.scanTopK(slots, k, newScorer), nil
}

// collectSlots resolves an allow set to index slots, iterating whichever of
// allow / the index is smaller; paths absent from the other side are skipped.
func (x *exactIndex) collectSlots(allow map[string]struct{}) []int {
	if len(x.paths) == 0 || len(allow) == 0 {
		return nil
	}
	var slots []int
	if len(allow) < len(x.paths) {
		slots = make([]int, 0, len(allow))
		for p := range allow {
			if i, ok := x.slot[p]; ok {
				slots = append(slots, i)
			}
		}
	} else {
		slots = make([]int, 0, len(x.paths))
		for i, p := range x.paths {
			if _, ok := allow[p]; ok {
				slots = append(slots, i)
			}
		}
	}
	return slots
}

// scanTopK scores every candidate (slots nil = all of them) in parallel with
// per-worker scorers from newScorer, then returns the top-k hits ordered by
// descending score (path ascending on ties).
func (x *exactIndex) scanTopK(slots []int, k int, newScorer func() func(slot int) float32) []SearchHit {
	n := len(x.paths)
	if slots != nil {
		n = len(slots)
	}
	if n == 0 || k <= 0 {
		return nil
	}
	scores := make([]float32, n)
	parallelChunks(n, func(lo, hi int) {
		score := newScorer()
		for j := lo; j < hi; j++ {
			s := j
			if slots != nil {
				s = slots[j]
			}
			scores[j] = score(s)
		}
	})
	if k > n {
		k = n
	}
	top := topKIndices(scores, k)
	hits := make([]SearchHit, 0, k)
	for _, j := range top {
		s := j
		if slots != nil {
			s = slots[j]
		}
		hits = append(hits, SearchHit{Path: x.paths[s], Score: scores[j]})
	}
	sortHits(hits)
	return hits
}

// parallelChunks runs fn over [0,n) split into contiguous per-CPU chunks.
func parallelChunks(n int, fn func(lo, hi int)) {
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
			fn(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// sortHits orders hits deterministically: score descending, then path
// ascending on ties.
func sortHits(hits []SearchHit) {
	sort.Slice(hits, func(a, b int) bool {
		if hits[a].Score != hits[b].Score {
			return hits[a].Score > hits[b].Score
		}
		return hits[a].Path < hits[b].Path
	})
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
