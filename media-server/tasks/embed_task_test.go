package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func TestShouldSkipEmbedRespectsModelKey(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	if err := media.UpsertEmbedding(db, "a.jpg", EmbedModelID, []float32{1, 0}, 0); err != nil {
		t.Fatalf("seed: %v", err)
	}
	if !shouldSkipEmbed(db, "a.jpg", EmbedModelID) {
		t.Error("expected skip for already-embedded path")
	}
	if shouldSkipEmbed(db, "b.jpg", EmbedModelID) {
		t.Error("expected no skip for new path")
	}
}
