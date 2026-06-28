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
// vector does not panic and leaves exactly one node with the updated vector.
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
