package tasks

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
)

func writeMoveFixture(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0644); err != nil {
		t.Fatal(err)
	}
}

// The move task used to rewrite only media.path and the tag table, leaving
// embeddings, faces and the live vector index on the old path.
func TestMoveMediaItemRepointsLibraryIndexAndSidecar(t *testing.T) {
	db := newFaceIndexDB(t)
	t.Cleanup(func() { SetVectorIndexForModel(nil, "") })

	dir := t.TempDir()
	src := filepath.Join(dir, "in", "clip.mp4")
	dst := filepath.Join(dir, "out", "clip.mp4")
	writeMoveFixture(t, src, "video")
	writeMoveFixture(t, filepath.Join(dir, "in", "clip.vtt"), "WEBVTT")

	if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, src); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT OR IGNORE INTO category (label, weight) VALUES ('Subject', 0)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT OR IGNORE INTO tag (label, category_label, weight) VALUES ('sunset', 'Subject', 0)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		VALUES (?, 'sunset', 'Subject', 1, 0)`, src); err != nil {
		t.Fatal(err)
	}
	const model = "m1"
	if err := media.UpsertEmbedding(db, src, model, []float32{1, 0, 0}, 0); err != nil {
		t.Fatal(err)
	}
	idx, err := BuildIndexFromDB(db, model, nil)
	if err != nil {
		t.Fatal(err)
	}
	SetVectorIndexForModel(idx, model)

	res, err := moveMediaItem(context.Background(), db, src, dst)
	if err != nil {
		t.Fatalf("move: %v", err)
	}
	if res == nil || res.Items != 1 {
		t.Fatalf("result = %+v, want one library item moved", res)
	}

	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Errorf("source still on disk (err=%v)", err)
	}
	if b, err := os.ReadFile(dst); err != nil || string(b) != "video" {
		t.Errorf("destination = %q, %v", b, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "out", "clip.vtt")); err != nil {
		t.Errorf("transcript sidecar did not follow the video: %v", err)
	}

	for _, q := range []string{
		`SELECT COUNT(*) FROM media WHERE path = ?`,
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
		`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
	} {
		var atOld, atNew int
		_ = db.QueryRow(q, src).Scan(&atOld)
		_ = db.QueryRow(q, dst).Scan(&atNew)
		if atOld != 0 || atNew != 1 {
			t.Errorf("%s: old=%d new=%d, want 0/1", q, atOld, atNew)
		}
	}

	hits, ok := indexSearch(model, []float32{1, 0, 0}, 5, nil, embedindex.ScoreDefault)
	if !ok || len(hits) != 1 || hits[0].Path != dst {
		t.Errorf("vector index hits = %+v, want only %q", hits, dst)
	}
}

// A query that still matches what an earlier run already moved resolves each
// such item's destination to its own path — a no-op, not a failure.
func TestMoveMediaItemAlreadyInPlace(t *testing.T) {
	p := filepath.Join(t.TempDir(), "a.mp4")
	writeMoveFixture(t, p, "x")
	if _, err := moveMediaItem(context.Background(), nil, p, p); !errors.Is(err, errAlreadyInPlace) {
		t.Errorf("err = %v, want errAlreadyInPlace", err)
	}
	if _, err := os.Stat(p); err != nil {
		t.Errorf("file disturbed: %v", err)
	}
}

// When the library refuses the destination (another item owns that path) the
// file goes back where it was, so disk and library still agree.
func TestMoveMediaItemLibraryConflictMovesFileBack(t *testing.T) {
	db := newFaceIndexDB(t)
	dir := t.TempDir()
	src := filepath.Join(dir, "a.mp4")
	dst := filepath.Join(dir, "out", "a.mp4")
	writeMoveFixture(t, src, "x")
	for _, p := range []string{src, dst} {
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := moveMediaItem(context.Background(), db, src, dst); err == nil {
		t.Fatal("expected a library conflict error")
	}
	if _, err := os.Stat(src); err != nil {
		t.Errorf("source not restored: %v", err)
	}
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Errorf("destination left behind (err=%v)", err)
	}
}

func TestCopyFileToKeepsContentAndModTime(t *testing.T) {
	dir := t.TempDir()
	src := filepath.Join(dir, "src.bin")
	dst := filepath.Join(dir, "dst.bin")
	writeMoveFixture(t, src, "payload")
	old := time.Date(2019, 5, 1, 12, 0, 0, 0, time.UTC)
	if err := os.Chtimes(src, old, old); err != nil {
		t.Fatal(err)
	}
	st, _ := os.Stat(src)
	if err := copyFileTo(src, dst, st); err != nil {
		t.Fatal(err)
	}
	got, _ := os.Stat(dst)
	if b, _ := os.ReadFile(dst); string(b) != "payload" {
		t.Errorf("content = %q", b)
	}
	if !got.ModTime().Equal(old) {
		t.Errorf("mtime = %v, want %v", got.ModTime(), old)
	}
}

// os.Rename cannot cross volumes (every drive letter on Windows), which is
// why moves from C:/X:/F: into Z:\Generated failed. Opt-in because it needs a
// directory on a different volume from the temp dir:
//
//	LOKI_MOVE_XDEV_DIR=Z:\scratch go test ./tasks -run CrossVolume
func TestMoveFileCrossVolume(t *testing.T) {
	other := os.Getenv("LOKI_MOVE_XDEV_DIR")
	if other == "" {
		t.Skip("set LOKI_MOVE_XDEV_DIR to a directory on another volume")
	}
	src := filepath.Join(t.TempDir(), "xdev.bin")
	writeMoveFixture(t, src, "across")
	dst := filepath.Join(other, "loki-move-xdev-test.bin")
	os.Remove(dst)
	t.Cleanup(func() { os.Remove(dst) })

	if err := moveFile(src, dst); err != nil {
		t.Fatalf("moveFile: %v", err)
	}
	if _, err := os.Stat(src); !os.IsNotExist(err) {
		t.Errorf("source still present (err=%v)", err)
	}
	if b, err := os.ReadFile(dst); err != nil || string(b) != "across" {
		t.Errorf("destination = %q, %v", b, err)
	}
	if _, err := os.Stat(dst + ".loki-moving"); !os.IsNotExist(err) {
		t.Error("temp file left behind")
	}
}
