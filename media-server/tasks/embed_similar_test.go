package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// embedvecNormalize wraps embedvec.Normalize so later task tests (11, 13)
// in this package can reuse the helper without importing embedvec directly.
func embedvecNormalize(v []float32) []float32 { return embedvec.Normalize(v) }

func norm(t *testing.T, v []float32) []float32 {
	t.Helper()
	return embedvecNormalize(v)
}

func TestSimilarByPathRanksByCosine(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	// query vector points along x; a is closest, b orthogonal, c opposite.
	_ = media.UpsertEmbedding(db, "q.jpg", EmbedModelID, norm(t, []float32{1, 0}), 0)
	_ = media.UpsertEmbedding(db, "a.jpg", EmbedModelID, norm(t, []float32{0.9, 0.1}), 0)
	_ = media.UpsertEmbedding(db, "b.jpg", EmbedModelID, norm(t, []float32{0, 1}), 0)
	_ = media.UpsertEmbedding(db, "c.jpg", EmbedModelID, norm(t, []float32{-1, 0}), 0)

	hits, err := SimilarByPath(db, EmbedModelID, "q.jpg", 10)
	if err != nil {
		t.Fatalf("similar: %v", err)
	}
	// The query item itself is INCLUDED (cosine 1.0 → ranks first).
	if len(hits) != 4 {
		t.Fatalf("expected 4 hits (self included), got %d", len(hits))
	}
	if hits[0].Path != "q.jpg" {
		t.Errorf("expected q.jpg first (self, score ~1.0), got %s", hits[0].Path)
	}
	if hits[1].Path != "a.jpg" {
		t.Errorf("expected a.jpg second, got %s", hits[1].Path)
	}
	if hits[len(hits)-1].Path != "c.jpg" {
		t.Errorf("expected c.jpg last, got %s", hits[len(hits)-1].Path)
	}
}
