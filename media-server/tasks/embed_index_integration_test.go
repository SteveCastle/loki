package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func TestBuildIndexFromDBAndSimilarUsesIt(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
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

	hits, err := SimilarByPath(db, EmbedModelID, "q.jpg", 1)
	if err != nil {
		t.Fatalf("similar: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != "a.jpg" {
		t.Errorf("expected a.jpg via index, got %+v", hits)
	}
}
