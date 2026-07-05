package main

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

func TestEnrichScoredItemsAttachesScoreAndSorts(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	// Seed media rows so enrichment can read metadata.
	for _, p := range []string{"a.jpg", "b.jpg"} {
		if _, err := db.Exec(`INSERT INTO media (path, width, height, elo) VALUES (?,?,?,?)`, p, 100, 200, 1500.0); err != nil {
			t.Fatal(err)
		}
	}
	hits := []tasks.SimilarHit{{Path: "a.jpg", Score: 0.2}, {Path: "b.jpg", Score: 0.9}}
	items, err := enrichScoredItems(db, hits)
	if err != nil {
		t.Fatalf("enrich: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}
	// Sorted by score desc: b first.
	if items[0]["path"] != "b.jpg" {
		t.Errorf("expected b.jpg first (higher score), got %v", items[0]["path"])
	}
	if s, _ := items[0]["score"].(float32); s != 0.9 {
		t.Errorf("expected score 0.9 on first item, got %v", items[0]["score"])
	}
	if items[0]["width"] == nil {
		t.Errorf("expected media metadata (width) attached")
	}
}
