package tasks

import (
	"context"
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// Deleting media rows (cleanup task, remove task — anything that goes through
// media.RemoveItemsFromDB) must evict the paths from the live vector index via
// the removal hook wired in registry.go init. Before the hook, only the remove
// task evicted; the cleanup task left ghosts that similarity search kept
// returning until the next index rebuild.
func TestRemoveItemsFromDB_EvictsFromVectorIndex(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec("INSERT INTO media (path) VALUES ('a.jpg')"); err != nil {
		t.Fatal(err)
	}

	idx := embedindex.New()
	idx.Add("a.jpg", embedvecNormalize([]float32{1, 0}))
	SetVectorIndex(idx)
	defer SetVectorIndex(nil)
	if IndexSize() != 1 {
		t.Fatalf("index size = %d before removal, want 1", IndexSize())
	}

	if _, err := media.RemoveItemsFromDB(context.Background(), db, []string{"a.jpg"}); err != nil {
		t.Fatalf("RemoveItemsFromDB() error = %v", err)
	}

	if IndexSize() != 0 {
		t.Fatalf("index size = %d after removal, want 0 (path evicted by hook)", IndexSize())
	}
}
