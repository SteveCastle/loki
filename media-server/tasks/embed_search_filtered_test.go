package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func newFilteredSearchDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	_ = media.UpsertEmbedding(db, "nearest.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0}), 0)
	_ = media.UpsertEmbedding(db, "mid.jpg", EmbedModelID, embedvecNormalize([]float32{0.5, 0.5}), 0)
	_ = media.UpsertEmbedding(db, "far.jpg", EmbedModelID, embedvecNormalize([]float32{0, 1}), 0)
	return db
}

// TestSearchByVectorWithinBruteForce verifies the DB brute-force path ranks
// only within the allow set — the globally-nearest item stays out when it
// isn't allowed.
func TestSearchByVectorWithinBruteForce(t *testing.T) {
	db := newFilteredSearchDB(t)
	q := embedvecNormalize([]float32{1, 0})
	allow := PathSet{"mid.jpg": {}, "far.jpg": {}}
	hits, err := searchByVectorWithin(db, EmbedModelID, q, 10, allow)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 || hits[0].Path != "mid.jpg" || hits[1].Path != "far.jpg" {
		t.Fatalf("expected [mid.jpg far.jpg], got %+v", hits)
	}
	// nil allow = whole library, nearest first.
	all, err := searchByVectorWithin(db, EmbedModelID, q, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(all) != 3 || all[0].Path != "nearest.jpg" {
		t.Fatalf("nil allow: expected nearest.jpg first of 3, got %+v", all)
	}
}

// TestSearchByVectorWithinUsesIndex verifies the installed index honors the
// allow set too (SearchFiltered), not just the brute-force fallback.
func TestSearchByVectorWithinUsesIndex(t *testing.T) {
	db := newFilteredSearchDB(t)
	idx := embedindex.New()
	idx.Add("nearest.jpg", embedvecNormalize([]float32{1, 0}))
	idx.Add("mid.jpg", embedvecNormalize([]float32{0.5, 0.5}))
	// far.jpg deliberately NOT indexed: index results must come from the
	// index alone, proving the filtered search didn't fall back to the DB.
	SetVectorIndexForModel(idx, EmbedModelID)
	t.Cleanup(func() { SetVectorIndex(nil) })

	hits, err := searchByVectorWithin(db, EmbedModelID, embedvecNormalize([]float32{1, 0}), 10,
		PathSet{"mid.jpg": {}, "far.jpg": {}})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Path != "mid.jpg" {
		t.Fatalf("expected [mid.jpg] from the index, got %+v", hits)
	}
}

// TestSearchFacesWithinFiltersByMediaPath verifies a face search restricted to
// an allow set only returns faces whose media is in the set, even when a face
// index is installed (the filtered path brute-forces — the index is keyed by
// face id and cannot see media paths).
func TestSearchFacesWithinFiltersByMediaPath(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	seedFaces(t, db, "a.jpg", "m1", []float32{1, 0, 0})
	idsB := seedFaces(t, db, "b.jpg", "m1", []float32{0.9, 0.1, 0})

	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	hits, err := searchFacesByVectorWithin(db, "m1", []float32{1, 0, 0}, 5, PathSet{"b.jpg": {}})
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].FaceID != idsB[0] || hits[0].MediaPath != "b.jpg" {
		t.Fatalf("expected only b.jpg's face, got %+v", hits)
	}
}
