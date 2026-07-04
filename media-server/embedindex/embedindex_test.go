package embedindex

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
)

func TestIndexSearchReturnsNearest(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	idx.Add("b", embedvec.Normalize([]float32{0, 1}))
	idx.Add("c", embedvec.Normalize([]float32{-1, 0}))
	hits := idx.Search(embedvec.Normalize([]float32{0.9, 0.1}), 1)
	if len(hits) != 1 || hits[0].Path != "a" {
		t.Fatalf("expected nearest 'a', got %+v", hits)
	}
}

func TestIndexSearchOrdersByDescendingSimilarity(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	idx.Add("b", embedvec.Normalize([]float32{0, 1}))
	idx.Add("c", embedvec.Normalize([]float32{-1, 0}))
	hits := idx.Search(embedvec.Normalize([]float32{0.9, 0.1}), 2)
	if len(hits) != 2 {
		t.Fatalf("expected 2 hits, got %d", len(hits))
	}
	if hits[0].Score < hits[1].Score {
		t.Errorf("results not nearest-first: scores %v, %v", hits[0].Score, hits[1].Score)
	}
	if hits[0].Path != "a" {
		t.Errorf("expected nearest 'a', got %q", hits[0].Path)
	}
}

func TestIndexLen(t *testing.T) {
	idx := New()
	if idx.Len() != 0 {
		t.Fatalf("expected Len 0 before any Add, got %d", idx.Len())
	}
	idx.Add("x", embedvec.Normalize([]float32{1, 0}))
	idx.Add("y", embedvec.Normalize([]float32{0, 1}))
	if idx.Len() != 2 {
		t.Fatalf("expected Len 2 after two Adds, got %d", idx.Len())
	}
}

// TestAddUpdatesVectorInPlace verifies that re-adding an existing key with a
// different vector replaces the stored vector (the exact index has no
// stale-vector caveat) and leaves exactly one entry.
func TestAddUpdatesVectorInPlace(t *testing.T) {
	idx := New()
	vec1 := embedvec.Normalize([]float32{1, 0})
	vec2 := embedvec.Normalize([]float32{0, 1})
	idx.Add("a", vec1)
	idx.Add("a", vec2)
	if idx.Len() != 1 {
		t.Fatalf("expected Len 1 after re-Add, got %d", idx.Len())
	}
	hits := idx.Search(vec2, 1)
	if len(hits) != 1 || hits[0].Path != "a" {
		t.Fatalf("expected Search to return 'a' after re-add, got %+v", hits)
	}
	if hits[0].Score < 0.999 {
		t.Errorf("expected 'a' to score ~1.0 against its NEW vector, got %v", hits[0].Score)
	}
	// Against the OLD vector it must no longer be a perfect match.
	old := idx.Search(vec1, 1)
	if len(old) == 1 && old[0].Score > 0.5 {
		t.Errorf("stale vector still in index: query at old position scored %v", old[0].Score)
	}
}

// TestDeleteRemovesNode verifies Delete removes a node from the index and that
// deleting a missing key is a safe no-op.
func TestDeleteRemovesNode(t *testing.T) {
	idx := New()
	vecA := embedvec.Normalize([]float32{1, 0})
	vecB := embedvec.Normalize([]float32{0, 1})
	idx.Add("a", vecA)
	idx.Add("b", vecB)

	// Delete "a" — safe no-op on a missing key must also not panic.
	idx.Delete("zzz")
	idx.Delete("a")

	if idx.Len() != 1 {
		t.Fatalf("expected Len 1 after Delete, got %d", idx.Len())
	}
	hits := idx.Search(vecA, 2)
	for _, h := range hits {
		if h.Path == "a" {
			t.Errorf("deleted node 'a' still appears in Search results: %+v", hits)
		}
	}
}

// TestDeleteThenReAddUsesNewVector verifies Delete followed by re-Add stores
// the NEW vector (the old HNSW-backed index kept the original vector until a
// full rebuild; the exact index must not).
func TestDeleteThenReAddUsesNewVector(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	idx.Add("b", embedvec.Normalize([]float32{0, 1}))
	idx.Delete("a")
	idx.Add("a", embedvec.Normalize([]float32{-1, 0}))
	if idx.Len() != 2 {
		t.Fatalf("expected Len 2 after re-add, got %d", idx.Len())
	}
	hits := idx.Search(embedvec.Normalize([]float32{-1, 0}), 1)
	if len(hits) != 1 || hits[0].Path != "a" || hits[0].Score < 0.999 {
		t.Fatalf("expected 'a' at ~1.0 against its new vector, got %+v", hits)
	}
}

// TestSearchNormalizesInputs verifies Add and Search accept raw (unnormalized)
// vectors and still produce true cosine similarities.
func TestSearchNormalizesInputs(t *testing.T) {
	idx := New()
	idx.Add("a", []float32{10, 0}) // raw, not unit length
	hits := idx.Search([]float32{3, 0}, 1)
	if len(hits) != 1 || hits[0].Path != "a" {
		t.Fatalf("expected 'a', got %+v", hits)
	}
	if hits[0].Score < 0.999 || hits[0].Score > 1.001 {
		t.Errorf("expected cosine ~1.0 for parallel raw vectors, got %v", hits[0].Score)
	}
}

// TestSearchIsExactTopK cross-checks Search against an independent brute-force
// ranking on a few hundred random vectors, including a self-query: the query
// item must come back first at ~1.0 and the full top-k must match exactly.
// This pins the property whose absence motivated dropping ANN: recall = 100%.
func TestSearchIsExactTopK(t *testing.T) {
	const n, dim, k = 300, 32, 50
	rng := rand.New(rand.NewSource(42))
	idx := New()
	vecs := make(map[string][]float32, n)
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		path := fmt.Sprintf("img-%03d", i)
		v = embedvec.Normalize(v)
		vecs[path] = v
		idx.Add(path, v)
	}

	query := vecs["img-123"]
	hits := idx.Search(query, k)
	if len(hits) != k {
		t.Fatalf("expected %d hits, got %d", k, len(hits))
	}
	if hits[0].Path != "img-123" || hits[0].Score < 0.999 {
		t.Fatalf("self-query must rank first at ~1.0, got %+v", hits[0])
	}

	// Independent brute-force reference.
	type ps struct {
		path  string
		score float32
	}
	ref := make([]ps, 0, n)
	for p, v := range vecs {
		ref = append(ref, ps{p, embedvec.Cosine(query, v)})
	}
	sort.Slice(ref, func(a, b int) bool {
		if ref[a].score != ref[b].score {
			return ref[a].score > ref[b].score
		}
		return ref[a].path < ref[b].path
	})
	for i := 0; i < k; i++ {
		if hits[i].Path != ref[i].path {
			t.Fatalf("rank %d mismatch: got %q (%v), want %q (%v)",
				i, hits[i].Path, hits[i].Score, ref[i].path, ref[i].score)
		}
	}
}

// BenchmarkSearch50k measures one exact search over 50k 768-dim vectors —
// the scale/shape of a real SigLIP 2 library. Guards against regressing the
// "exact scan is fast enough" assumption this package is built on.
func BenchmarkSearch50k(b *testing.B) {
	const n, dim = 50_000, 768
	rng := rand.New(rand.NewSource(1))
	idx := New()
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		idx.Add(fmt.Sprintf("img-%06d", i), v)
	}
	query := make([]float32, dim)
	for d := range query {
		query[d] = float32(rng.NormFloat64())
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.Search(query, 50)
	}
}

// TestSearchKLargerThanIndex verifies k > Len returns everything, ordered.
func TestSearchKLargerThanIndex(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	idx.Add("b", embedvec.Normalize([]float32{0, 1}))
	hits := idx.Search(embedvec.Normalize([]float32{1, 0}), 10)
	if len(hits) != 2 || hits[0].Path != "a" {
		t.Fatalf("expected [a b], got %+v", hits)
	}
}
