package embedindex

import (
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

// TestAddIsIdempotent verifies that re-adding an existing key with a different
// vector does not panic and leaves exactly one active node.
func TestAddIsIdempotent(t *testing.T) {
	idx := New()
	vec1 := embedvec.Normalize([]float32{1, 0})
	vec2 := embedvec.Normalize([]float32{0, 1})
	idx.Add("a", vec1)
	// Re-add same key with a different vector — must not panic.
	idx.Add("a", vec2)
	if idx.Len() != 1 {
		t.Fatalf("expected Len 1 after idempotent Add, got %d", idx.Len())
	}
	hits := idx.Search(vec2, 1)
	if len(hits) != 1 || hits[0].Path != "a" {
		t.Fatalf("expected Search to return 'a' after re-add, got %+v", hits)
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

// TestAddRefreshesVectorForActiveKey verifies that re-adding an active key with
// a different vector updates the authoritative overlay, so Search scores against
// the new vector rather than the stale graph position.
func TestAddRefreshesVectorForActiveKey(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	// Re-add active key with a vector pointing elsewhere.
	idx.Add("a", embedvec.Normalize([]float32{0, 1}))
	if idx.Len() != 1 {
		t.Fatalf("expected Len 1, got %d", idx.Len())
	}
	// Query aligned with the NEW vector should score ~1.0.
	hits := idx.Search(embedvec.Normalize([]float32{0, 1}), 1)
	if len(hits) != 1 || hits[0].Score < 0.9 {
		t.Fatalf("expected score to reflect refreshed vector (~1.0), got %+v", hits)
	}
}

// TestReactivateUsesNewVector verifies that after Delete+Add (ghost reactivation)
// the Search score uses the new overlay vector, not the stale graph position.
//
// With k=2 both nodes are fetched regardless of graph topology, so the test is
// deterministic even on tiny 2-node graphs. "a" is inserted at {1,0} in the
// graph but the overlay is {-1,0} after reactivation; scoring against the
// overlay yields ~1.0, whereas scoring against node.Embedding() would yield ~-1.0.
func TestReactivateUsesNewVector(t *testing.T) {
	idx := New()
	idx.Add("a", embedvec.Normalize([]float32{1, 0}))
	idx.Add("b", embedvec.Normalize([]float32{0, 1})) // keep graph non-trivial
	idx.Delete("a")
	idx.Add("a", embedvec.Normalize([]float32{-1, 0})) // reactivate with new vector

	// Request k=2 so both nodes are fetched regardless of topology. Find "a" in
	// results and assert its score reflects the overlay {-1,0}, not the stale
	// graph position {1,0}.
	hits := idx.Search(embedvec.Normalize([]float32{-1, 0}), 2)
	var aHit *SearchHit
	for i := range hits {
		if hits[i].Path == "a" {
			aHit = &hits[i]
			break
		}
	}
	if aHit == nil {
		t.Fatalf("expected 'a' in Search results, got %+v", hits)
	}
	// cos({-1,0},{-1,0})=1 → score≈1.0 (overlay correct)
	// cos({-1,0},{1,0})=-1 → score≈-1.0 (stale graph vector — would catch pre-fix bug)
	if aHit.Score < 0.9 {
		t.Fatalf("score reflects stale graph vector; got Score=%v, expected ~1.0 (overlay vector)", aHit.Score)
	}
}
