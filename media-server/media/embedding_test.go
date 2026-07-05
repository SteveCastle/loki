package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func newEmbedDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	return db
}

func TestUpsertAndGetEmbedding(t *testing.T) {
	db := newEmbedDB(t)
	defer db.Close()
	vec := []float32{0.1, 0.2, 0.3}
	if err := UpsertEmbedding(db, "a.jpg", "m1", vec, 0); err != nil {
		t.Fatalf("upsert: %v", err)
	}
	got, ok, err := GetEmbedding(db, "a.jpg", "m1")
	if err != nil || !ok {
		t.Fatalf("get: ok=%v err=%v", ok, err)
	}
	if len(got) != 3 || got[1] != 0.2 {
		t.Errorf("roundtrip mismatch: %v", got)
	}
}

func TestUpsertOverwrites(t *testing.T) {
	db := newEmbedDB(t)
	defer db.Close()
	_ = UpsertEmbedding(db, "a.jpg", "m1", []float32{1, 1}, 0)
	if err := UpsertEmbedding(db, "a.jpg", "m1", []float32{2, 2}, 0); err != nil {
		t.Fatalf("second upsert: %v", err)
	}
	got, _, _ := GetEmbedding(db, "a.jpg", "m1")
	if got[0] != 2 {
		t.Errorf("expected overwrite, got %v", got)
	}
}

func TestHasEmbeddingModelScoped(t *testing.T) {
	db := newEmbedDB(t)
	defer db.Close()
	_ = UpsertEmbedding(db, "a.jpg", "m1", []float32{1}, 0)
	if ok, _ := HasEmbedding(db, "a.jpg", "m1"); !ok {
		t.Error("expected has=true for m1")
	}
	if ok, _ := HasEmbedding(db, "a.jpg", "m2"); ok {
		t.Error("expected has=false for m2 (model-scoped)")
	}
}

func TestLoadAllEmbeddingsFiltersByModel(t *testing.T) {
	db := newEmbedDB(t)
	defer db.Close()
	_ = UpsertEmbedding(db, "a.jpg", "m1", []float32{1, 0}, 0)
	_ = UpsertEmbedding(db, "b.jpg", "m1", []float32{0, 1}, 0)
	_ = UpsertEmbedding(db, "c.jpg", "m2", []float32{1, 1}, 0)
	all, err := LoadAllEmbeddings(db, "m1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(all) != 2 {
		t.Errorf("expected 2 m1 rows, got %d", len(all))
	}
}
