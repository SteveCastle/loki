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
	// Seed a second model's row directly: ReplaceFaces would supersede the
	// first model's data, and this test wants a genuinely multi-model path.
	if _, err := db.Exec(
		`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
		 VALUES ('a.jpg','arcface',0,0,1,1,0.9,x'0000803f')`,
	); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		`INSERT INTO face_scan (media_path, model, face_count, scanned_at) VALUES ('a.jpg','arcface',1,1)`,
	); err != nil {
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

// A scan is authoritative for the whole item: rescanning a path under a new
// recognizer (routing re-decided its domain) must remove the old
// recognizer's rows and scan marker, or they linger as permanently
// "ungrouped" ghost faces in the review UI.
func TestReplaceFacesSupersedesOtherModels(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	if _, err := ReplaceFaces(db, "a.jpg", "photo-model", []NewFace{{Score: 0.9, Vec: []float32{1}}}, 1); err != nil {
		t.Fatal(err)
	}
	// Another path's rows must be untouched by a.jpg's rescan.
	otherIDs, err := ReplaceFaces(db, "b.jpg", "photo-model", []NewFace{{Score: 0.8, Vec: []float32{1}}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReplaceFaces(db, "a.jpg", "anime-model", []NewFace{{Score: 0.95, Vec: []float32{2}}}, 2); err != nil {
		t.Fatal(err)
	}

	if faces, _ := GetFaces(db, "a.jpg", "photo-model"); len(faces) != 0 {
		t.Fatalf("superseded photo-model rows remain: %d", len(faces))
	}
	if ok, _ := HasFaceScan(db, "a.jpg", "photo-model"); ok {
		t.Fatal("superseded photo-model scan marker remains")
	}
	if faces, _ := GetFaces(db, "a.jpg", "anime-model"); len(faces) != 1 {
		t.Fatalf("anime-model rows = %d, want 1", len(faces))
	}
	if ok, _ := HasFaceScan(db, "a.jpg", "anime-model"); !ok {
		t.Fatal("anime-model scan marker missing")
	}
	if faces, _ := GetFaces(db, "b.jpg", "photo-model"); len(faces) != 1 || faces[0].ID != otherIDs[0] {
		t.Fatalf("unrelated path's faces disturbed: %+v", faces)
	}
}

// Pre-fix data repair: rows whose scan was superseded by a newer scan under
// another model are swept at schema init. Ties are left alone.
func TestCleanupSupersededFaces(t *testing.T) {
	db := newFaceDB(t)
	defer db.Close()
	seed := func(path, model string, scannedAt int64) {
		t.Helper()
		if _, err := db.Exec(
			// x'0000803f' = float32(1.0) little-endian — a decodable vector.
			`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
			 VALUES (?,?,0,0,1,1,0.9,x'0000803f')`, path, model,
		); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(
			`INSERT INTO face_scan (media_path, model, face_count, scanned_at) VALUES (?,?,1,?)`,
			path, model, scannedAt,
		); err != nil {
			t.Fatal(err)
		}
	}
	seed("stale.jpg", "photo-model", 1) // superseded by the anime scan below
	seed("stale.jpg", "anime-model", 2)
	seed("tie.jpg", "photo-model", 5) // identical timestamps: no winner, keep both
	seed("tie.jpg", "anime-model", 5)
	seed("solo.jpg", "photo-model", 1) // single model: never superseded

	n, err := CleanupSupersededFaces(db)
	if err != nil {
		t.Fatalf("cleanup: %v", err)
	}
	if n != 1 {
		t.Fatalf("removed = %d, want 1", n)
	}
	if faces, _ := GetFaces(db, "stale.jpg", "photo-model"); len(faces) != 0 {
		t.Fatal("superseded rows survived cleanup")
	}
	if ok, _ := HasFaceScan(db, "stale.jpg", "photo-model"); ok {
		t.Fatal("superseded scan marker survived cleanup")
	}
	if faces, _ := GetFaces(db, "stale.jpg", "anime-model"); len(faces) != 1 {
		t.Fatal("winning model's rows must survive")
	}
	for _, model := range []string{"photo-model", "anime-model"} {
		if faces, _ := GetFaces(db, "tie.jpg", model); len(faces) != 1 {
			t.Fatalf("tie rows must be kept (%s)", model)
		}
	}
	if faces, _ := GetFaces(db, "solo.jpg", "photo-model"); len(faces) != 1 {
		t.Fatal("single-model rows must be kept")
	}
	// Idempotent: a second sweep finds nothing.
	if n, err := CleanupSupersededFaces(db); err != nil || n != 0 {
		t.Fatalf("second sweep: n=%d err=%v", n, err)
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
