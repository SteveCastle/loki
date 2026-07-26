package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/media"
)

// After media.MovePath rewrites the DB, the live indexes are still keyed by
// the OLD path. The vector index would keep returning a path the media handler
// can no longer serve, and the face index's path→keys map would miss on the
// next eviction for either name.

func TestIndexRenamePathMovesTheVectorToTheNewPath(t *testing.T) {
	db := newFaceIndexDB(t)
	t.Cleanup(func() { SetVectorIndexForModel(nil, "") })

	const from = "/photos/old.jpg"
	const to = "/archive/new.jpg"
	const model = "m1"
	if err := media.UpsertEmbedding(db, from, model, []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}
	idx, err := BuildIndexFromDB(db, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	SetVectorIndexForModel(idx, model)

	// The move: DB first (what MovePath does), then the index re-key.
	if _, err := db.Exec(`UPDATE media_embedding SET media_path = ? WHERE media_path = ?`, to, from); err != nil {
		t.Fatal(err)
	}
	IndexRenamePath(db, from, to)

	hits, ok := indexSearch(model, []float32{1, 0, 0}, 5)
	if !ok {
		t.Fatal("no index installed")
	}
	if len(hits) != 1 {
		t.Fatalf("hits = %+v, want exactly one (the moved vector, not both keys)", hits)
	}
	if hits[0].Path != to {
		t.Errorf("hit path = %q, want %q", hits[0].Path, to)
	}
}

func TestIndexRenamePathEvictsWhenTheDestinationHasNoVector(t *testing.T) {
	db := newFaceIndexDB(t)
	t.Cleanup(func() { SetVectorIndexForModel(nil, "") })

	const from = "/photos/old.jpg"
	if err := media.UpsertEmbedding(db, from, "m1", []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}
	idx, err := BuildIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetVectorIndexForModel(idx, "m1")

	// No DB update: the destination has no row. The stale key must still go —
	// a hit on a path nothing can serve is worse than a missing hit.
	IndexRenamePath(db, from, "/archive/new.jpg")
	if n := IndexSize(); n != 0 {
		t.Errorf("index size = %d, want 0 (the old key survived)", n)
	}
}

func TestFaceIndexRenamePathReKeysThePathMap(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	const from = "a.jpg"
	const to = "moved/a.jpg"
	seedFaces(t, db, from, "m1", []float32{1, 0, 0}, []float32{0, 1, 0})
	seedFaces(t, db, "b.jpg", "m1", []float32{0, 0, 1})
	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	FaceIndexRenamePath(from, to)

	// Evicting the OLD path must now be a no-op: those faces live under the
	// new name and are still perfectly valid.
	FaceIndexDeletePath(from)
	if n := FaceIndexSize(); n != 3 {
		t.Fatalf("face index size = %d after evicting the old name, want 3", n)
	}
	// Evicting the NEW path drops exactly that item's two faces.
	FaceIndexDeletePath(to)
	if n := FaceIndexSize(); n != 1 {
		t.Errorf("face index size = %d after evicting the new name, want 1", n)
	}
}

func TestFaceIndexRenamePathIsANoOpForAnUnknownPath(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	seedFaces(t, db, "a.jpg", "m1", []float32{1, 0, 0})
	idx, pathKeys, err := BuildFaceIndexFromDB(db, "m1", nil)
	if err != nil {
		t.Fatal(err)
	}
	SetFaceIndexForModel(idx, "m1", pathKeys)

	FaceIndexRenamePath("never-indexed.jpg", "elsewhere.jpg")
	FaceIndexDeletePath("elsewhere.jpg")
	if n := FaceIndexSize(); n != 1 {
		t.Errorf("face index size = %d, want 1 (an unrelated rename disturbed it)", n)
	}
}
