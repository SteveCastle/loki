package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func newFaceDB(t *testing.T) *sql.DB {
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

func TestReplaceFacesAndGet(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()

	faces := []NewFace{
		{X: 0.1, Y: 0.2, W: 0.3, H: 0.4, Score: 0.95, Vec: []float32{1, 0, 0}},
		{X: 0.5, Y: 0.5, W: 0.2, H: 0.2, Score: 0.80, Vec: []float32{0, 1, 0}, FrameTS: 12.5},
	}
	ids, err := ReplaceFaces(db, "a.jpg", "sface", faces, 111)
	if err != nil {
		t.Fatalf("replace: %v", err)
	}
	if len(ids) != 2 || ids[0] == ids[1] || ids[0] == 0 {
		t.Fatalf("inserted ids = %v, want 2 distinct non-zero ids", ids)
	}

	got, err := GetFaces(db, "a.jpg", "sface")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d faces, want 2", len(got))
	}
	// Ordered det_score descending.
	if got[0].Score != 0.95 || got[1].Score != 0.80 {
		t.Fatalf("order: %v, %v", got[0].Score, got[1].Score)
	}
	if got[0].X != 0.1 || got[0].W != 0.3 || got[0].Vec[0] != 1 {
		t.Fatalf("row roundtrip mismatch: %+v", got[0])
	}
	if got[1].FrameTS != 12.5 {
		t.Fatalf("frame_ts = %v, want 12.5", got[1].FrameTS)
	}
	if got[0].PersonID != 0 || got[0].AssignedBy != "" {
		t.Fatalf("new face should be unassigned: %+v", got[0])
	}

	// Scan marker exists even for a no-face rescan, and replace is idempotent.
	if _, err := ReplaceFaces(db, "a.jpg", "sface", nil, 222); err != nil {
		t.Fatalf("rescan: %v", err)
	}
	got, err = GetFaces(db, "a.jpg", "sface")
	if err != nil {
		t.Fatalf("get after rescan: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("rescan should have replaced faces, got %d", len(got))
	}
	ok, err := HasFaceScan(db, "a.jpg", "sface")
	if err != nil || !ok {
		t.Fatalf("scan marker missing after no-face rescan: ok=%v err=%v", ok, err)
	}
}

func TestHasFaceScanIsModelKeyed(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	if _, err := ReplaceFaces(db, "a.jpg", "sface", nil, 1); err != nil {
		t.Fatal(err)
	}
	if ok, _ := HasFaceScan(db, "a.jpg", "sface"); !ok {
		t.Fatal("expected scan marker for sface")
	}
	if ok, _ := HasFaceScan(db, "a.jpg", "arcface"); ok {
		t.Fatal("scan marker leaked across models")
	}
	if ok, _ := HasFaceScan(db, "b.jpg", "sface"); ok {
		t.Fatal("scan marker leaked across paths")
	}
}

func TestFaceScansForPaths(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	_, _ = ReplaceFaces(db, "a.jpg", "photo-model", nil, 1)
	_, _ = ReplaceFaces(db, "b.jpg", "anime-model", nil, 1)
	_, _ = ReplaceFaces(db, "c.jpg", "other-model", nil, 1)

	got, err := FaceScansForPaths(db, []string{"photo-model", "anime-model"}, []string{"a.jpg", "b.jpg", "c.jpg", "d.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if !got["a.jpg"] || !got["b.jpg"] {
		t.Fatalf("candidate-model scans missed: %v", got)
	}
	// c.jpg was scanned only under a NON-candidate model — must not skip.
	if got["c.jpg"] || got["d.jpg"] {
		t.Fatalf("false positives: %v", got)
	}
	// Empty inputs are harmless.
	if empty, err := FaceScansForPaths(db, nil, []string{"a.jpg"}); err != nil || len(empty) != 0 {
		t.Fatalf("empty models: %v %v", empty, err)
	}
}

func TestLoadAllFacesAndGetByID(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	if _, err := ReplaceFaces(db, "a.jpg", "sface", []NewFace{{Score: 0.9, Vec: []float32{1, 2}}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := ReplaceFaces(db, "b.jpg", "sface", []NewFace{{Score: 0.8, Vec: []float32{3, 4}}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := ReplaceFaces(db, "c.jpg", "other", []NewFace{{Score: 0.7, Vec: []float32{5, 6}}}, 1); err != nil {
		t.Fatal(err)
	}

	all, err := LoadAllFaces(db, "sface")
	if err != nil {
		t.Fatalf("load all: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("got %d faces for sface, want 2 (model-keyed)", len(all))
	}

	f, ok, err := GetFaceByID(db, all[0].ID)
	if err != nil || !ok {
		t.Fatalf("get by id: ok=%v err=%v", ok, err)
	}
	if f.ID != all[0].ID || f.MediaPath != all[0].MediaPath {
		t.Fatalf("by-id mismatch: %+v vs %+v", f, all[0])
	}
	if _, ok, _ := GetFaceByID(db, 999999); ok {
		t.Fatal("expected miss for unknown id")
	}
}

func TestDeleteFacesForMedia(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	if _, err := ReplaceFaces(db, "a.jpg", "sface", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := ReplaceFaces(db, "a.jpg", "arcface", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1); err != nil {
		t.Fatal(err)
	}
	if err := DeleteFacesForMedia(db, "a.jpg"); err != nil {
		t.Fatalf("delete: %v", err)
	}
	for _, model := range []string{"sface", "arcface"} {
		if faces, _ := GetFaces(db, "a.jpg", model); len(faces) != 0 {
			t.Fatalf("faces remain for %s", model)
		}
		if ok, _ := HasFaceScan(db, "a.jpg", model); ok {
			t.Fatalf("scan marker remains for %s", model)
		}
	}
}

func TestCountFaceScans(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	_, _ = ReplaceFaces(db, "a.jpg", "sface", nil, 1)
	_, _ = ReplaceFaces(db, "b.jpg", "sface", nil, 1)
	_, _ = ReplaceFaces(db, "c.jpg", "other", nil, 1)
	n, err := CountFaceScans(db, "sface")
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("count = %d, want 2", n)
	}
}
