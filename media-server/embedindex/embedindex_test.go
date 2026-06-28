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
