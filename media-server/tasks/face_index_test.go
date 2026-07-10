package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// embedindexNewForTest returns an empty index (alias so the fallback test
// reads clearly).
func embedindexNewForTest() embedindex.VectorIndex { return embedindex.New() }

func newFaceIndexDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	return db
}

// resetFaceIndex uninstalls any face index after the test.
func resetFaceIndex(t *testing.T) {
	t.Helper()
	t.Cleanup(func() { SetFaceIndexForModel(nil, "", nil) })
}

func seedFaces(t *testing.T, db *sql.DB, path, model string, vecs ...[]float32) []int64 {
	t.Helper()
	faces := make([]media.NewFace, 0, len(vecs))
	for i, v := range vecs {
		faces = append(faces, media.NewFace{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9 - float64(i)*0.01, Vec: v})
	}
	ids, err := media.ReplaceFaces(db, path, model, faces, 1)
	if err != nil {
		t.Fatalf("seed %s: %v", path, err)
	}
	return ids
}

func TestFaceIndexBuildAndSearch(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	idsA := seedFaces(t, db, "a.jpg", "m1", []float32{1, 0, 0}, []float32{0, 1, 0})
	idsB := seedFaces(t, db, "b.jpg", "m1", []float32{0.9, 0.1, 0})
	seedFaces(t, db, "c.jpg", "other-model", []float32{1, 0, 0})

	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	if idx.Len() != 3 {
		t.Fatalf("index has %d faces, want 3 (model-keyed)", idx.Len())
	}
	if len(pathKeys["a.jpg"]) != 2 || len(pathKeys["b.jpg"]) != 1 {
		t.Fatalf("pathKeys = %v", pathKeys)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	hits, err := SearchFacesByVector(db, "m1", []float32{1, 0, 0}, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 2 {
		t.Fatalf("got %d hits, want 2", len(hits))
	}
	if hits[0].FaceID != idsA[0] || hits[0].MediaPath != "a.jpg" {
		t.Fatalf("top hit = %+v, want face %d of a.jpg", hits[0], idsA[0])
	}
	if hits[1].FaceID != idsB[0] {
		t.Fatalf("second hit = %+v, want face %d of b.jpg", hits[1], idsB[0])
	}
	if hits[0].Score < 0.99 || hits[1].Score >= hits[0].Score {
		t.Fatalf("scores not descending/cosine: %v, %v", hits[0].Score, hits[1].Score)
	}
	// Hydration carries the bbox + model through.
	if hits[0].W != 0.2 || hits[0].Model != "m1" {
		t.Fatalf("hydrated hit missing fields: %+v", hits[0])
	}
}

func TestFaceSearchBruteForceFallback(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	ids := seedFaces(t, db, "a.jpg", "m1", []float32{0, 0, 1})

	// No index installed at all → brute force.
	hits, err := SearchFacesByVector(db, "m1", []float32{0, 0, 1}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].FaceID != ids[0] {
		t.Fatalf("brute-force hits = %+v", hits)
	}

	// Index for a DIFFERENT model installed → still brute force for m1.
	SetFaceIndexForModel(embedindexNewForTest(), "other", nil)
	hits, err = SearchFacesByVector(db, "m1", []float32{0, 0, 1}, 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].FaceID != ids[0] {
		t.Fatalf("model-mismatch fallback hits = %+v", hits)
	}
}

func TestFaceIndexReplacePathEvictsStaleFaces(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	seedFaces(t, db, "a.jpg", "m1", []float32{1, 0, 0})

	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	// Rescan a.jpg: now two different faces. DB replace + index replace.
	newFaces := []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.2, H: 0.2, Score: 0.9, Vec: []float32{0, 1, 0}},
		{X: 0.5, Y: 0.5, W: 0.2, H: 0.2, Score: 0.8, Vec: []float32{0, 0, 1}},
	}
	ids, err := media.ReplaceFaces(db, "a.jpg", "m1", newFaces, 2)
	if err != nil {
		t.Fatal(err)
	}
	faceIndexReplacePath("m1", "a.jpg", ids, newFaces)

	if got := FaceIndexSize(); got != 2 {
		t.Fatalf("index size = %d after replace, want 2", got)
	}
	// The old vector must be gone: searching it should now return the new
	// faces, not a perfect 1.0 match.
	hits, err := SearchFacesByVector(db, "m1", []float32{1, 0, 0}, 5)
	if err != nil {
		t.Fatal(err)
	}
	for _, h := range hits {
		if h.Score > 0.99 {
			t.Fatalf("stale face still indexed: %+v", h)
		}
	}

	// A wrong-model replace evicts the path (the scan wiped the whole item's
	// rows in the DB) but must NOT add the foreign model's vectors.
	faceIndexReplacePath("other-model", "a.jpg", []int64{999}, []media.NewFace{{Vec: []float32{1, 1, 1}}})
	if got := FaceIndexSize(); got != 0 {
		t.Fatalf("index size = %d after wrong-model replace, want 0 (path evicted, nothing added)", got)
	}
}

func TestFaceIndexDeletePath(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	seedFaces(t, db, "a.jpg", "m1", []float32{1, 0, 0})
	seedFaces(t, db, "b.jpg", "m1", []float32{0, 1, 0})

	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	FaceIndexDeletePath("a.jpg")
	if got := FaceIndexSize(); got != 1 {
		t.Fatalf("index size = %d after delete, want 1", got)
	}
	FaceIndexDeletePath("never-indexed.jpg") // no-op, no panic
	if got := FaceIndexSize(); got != 1 {
		t.Fatalf("no-op delete changed size: %d", got)
	}
}

func TestRebuildActiveFaceIndex(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	model := ActiveFaceModel()
	seedFaces(t, db, "a.jpg", model.ID, []float32{1, 0})

	gotModel, n, err := RebuildActiveFaceIndex(db, nil)
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != model.ID || n != 1 {
		t.Fatalf("rebuild = (%s, %d), want (%s, 1)", gotModel, n, model.ID)
	}
	if FaceIndexedModel() != model.ID || FaceIndexSize() != 1 {
		t.Fatalf("installed index = (%s, %d)", FaceIndexedModel(), FaceIndexSize())
	}
}

func TestFaceHitsToMediaHits(t *testing.T) {
	hits := []FaceHit{
		{FaceID: 1, MediaPath: "a.jpg", Score: 0.9},
		{FaceID: 2, MediaPath: "b.jpg", Score: 0.8},
		{FaceID: 3, MediaPath: "a.jpg", Score: 0.95}, // second face of a.jpg, better
	}
	out := FaceHitsToMediaHits(hits)
	if len(out) != 2 {
		t.Fatalf("got %d media hits, want 2", len(out))
	}
	if out[0].Path != "a.jpg" || out[0].Score != 0.95 {
		t.Fatalf("best-per-path not kept: %+v", out[0])
	}
	if out[1].Path != "b.jpg" {
		t.Fatalf("second = %+v", out[1])
	}
}

func TestRemoveItemsFromDBDeletesFaceRows(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	if _, err := db.Exec(`INSERT INTO media (path) VALUES ('a.jpg')`); err != nil {
		t.Fatal(err)
	}
	seedFaces(t, db, "a.jpg", "m1", []float32{1, 0})

	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	if _, err := media.RemoveItemsFromDB(t.Context(), db, []string{"a.jpg"}); err != nil {
		t.Fatal(err)
	}
	faces, err := media.GetFaces(db, "a.jpg", "m1")
	if err != nil || len(faces) != 0 {
		t.Fatalf("face rows survived removal: %v, %v", faces, err)
	}
	if ok, _ := media.HasFaceScan(db, "a.jpg", "m1"); ok {
		t.Fatal("face_scan marker survived removal")
	}
	// The registry-installed removal hook must have evicted the index entry.
	if got := FaceIndexSize(); got != 0 {
		t.Fatalf("index size = %d after media removal, want 0", got)
	}
}
