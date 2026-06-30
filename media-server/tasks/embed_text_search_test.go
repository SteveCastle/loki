package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func TestSearchByVectorRanks(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	_ = media.UpsertEmbedding(db, "cat.jpg", EmbedModelID, embedvecNormalize([]float32{1, 0}), 0)
	_ = media.UpsertEmbedding(db, "dog.jpg", EmbedModelID, embedvecNormalize([]float32{0, 1}), 0)
	hits, err := SearchByVector(db, EmbedModelID, embedvecNormalize([]float32{0.8, 0.2}), 1)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(hits) != 1 || hits[0].Path != "cat.jpg" {
		t.Errorf("expected cat.jpg, got %+v", hits)
	}
}
