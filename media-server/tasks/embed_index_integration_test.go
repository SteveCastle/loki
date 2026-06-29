package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func TestBuildIndexFromDBAndSimilarUsesIt(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	_ = media.UpsertEmbedding(db, "q.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0}), 0)
	_ = media.UpsertEmbedding(db, "a.jpg", EmbedModelID, embedvecNormalize([]float32{0.9, 0.1}), 0)
	_ = media.UpsertEmbedding(db, "b.jpg", EmbedModelID, embedvecNormalize([]float32{-1, 0}), 0)

	idx, err := BuildIndexFromDB(db, EmbedModelID)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	SetVectorIndex(idx)
	defer SetVectorIndex(nil)

	hits, err := SimilarByPath(db, EmbedModelID, "q.jpg", 2)
	if err != nil {
		t.Fatalf("similar: %v", err)
	}
	// Self (q.jpg) is included and ranks first; a.jpg is the nearest neighbour.
	if len(hits) != 2 || hits[0].Path != "q.jpg" || hits[1].Path != "a.jpg" {
		t.Errorf("expected [q.jpg, a.jpg] via index, got %+v", hits)
	}
}

func TestIncrementalInsertIsSearchable(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	// Seed the query row in the DB (SimilarByPath reads the query vector from the DB).
	_ = media.UpsertEmbedding(db, "q.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0}), 0)

	// Empty index installed; then incrementally insert two candidates.
	SetVectorIndex(embedindex.New())
	defer SetVectorIndex(nil)
	indexAdd("a.jpg", embedvecNormalize([]float32{0.9, 0.1}))
	indexAdd("b.jpg", embedvecNormalize([]float32{-1, 0}))

	hits, err := SimilarByPath(db, EmbedModelID, "q.jpg", 1)
	if err != nil {
		t.Fatalf("similar: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != "a.jpg" {
		t.Errorf("expected a.jpg via incremental index, got %+v", hits)
	}
}
