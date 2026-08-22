package tasks

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	_ "modernc.org/sqlite"
)

// absMediaPath mirrors the normalization the scanner applies before inserting.
func absMediaPath(t *testing.T, p string) string {
	t.Helper()
	abs, err := filepath.Abs(p)
	if err != nil {
		t.Fatal(err)
	}
	return filepath.FromSlash(abs)
}

// newIngestJob adds and claims an ingest job on a fresh single-connection
// in-memory queue DB. The single connection also proves the streaming loop
// never writes to the job row while an insert transaction is open.
func newIngestJob(t *testing.T, args []string, input string) (*jobqueue.Queue, *jobqueue.Job) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })

	q := jobqueue.NewQueueWithDB(db)
	id, err := q.AddJob("", "ingest", args, input, nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	return q, j
}

func TestIngestLocalStreamsRecursively(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub", "deep"), 0o755); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"c.png":                               "pngdata",
		"notmedia.txt":                        "text",
		filepath.Join("sub", "a.jpg"):         "jpegdata!",
		filepath.Join("sub", "deep", "b.mp4"): "mp4data",
	}
	for rel, content := range files {
		if err := os.WriteFile(filepath.Join(dir, rel), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	q, j := newIngestJob(t, []string{"--recursive"}, dir)
	var mu sync.Mutex

	// Pre-seed one file as already in the database (size 0 so the upsert's
	// backfill path is exercised too).
	if err := ensureMediaTableSchema(q.Db); err != nil {
		t.Fatal(err)
	}
	existing := absMediaPath(t, filepath.Join(dir, "c.png"))
	if _, err := q.Db.Exec(`INSERT INTO media (path, size) VALUES (?, 0)`, existing); err != nil {
		t.Fatal(err)
	}

	if err := ingestLocalTaskWithOptions(j, q, &mu, IngestOptions{}); err != nil {
		t.Fatalf("ingest: %v", err)
	}

	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Errorf("job state = %v; want Completed", got)
	}

	var count int
	if err := q.Db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Errorf("media rows = %d; want 3", count)
	}
	for _, rel := range []string{filepath.Join("sub", "a.jpg"), filepath.Join("sub", "deep", "b.mp4")} {
		p := absMediaPath(t, filepath.Join(dir, rel))
		var size int64
		if err := q.Db.QueryRow(`SELECT size FROM media WHERE path = ?`, p).Scan(&size); err != nil {
			t.Fatalf("row for %s: %v", p, err)
		}
		if size <= 0 {
			t.Errorf("size for %s = %d; want > 0", p, size)
		}
	}

	// Only the two NEW files are outputs for downstream workflow steps.
	if got := len(q.Jobs[j.ID].OutputFiles); got != 2 {
		t.Errorf("output files = %d; want 2 (%v)", got, q.Jobs[j.ID].OutputFiles)
	}

	// The progress bar lands full at the discovered-media total.
	if d, tot := q.Jobs[j.ID].ProgressDone, q.Jobs[j.ID].ProgressTotal; d != 3 || tot != 3 {
		t.Errorf("progress = %d/%d; want 3/3", d, tot)
	}
}

func TestIngestLocalNonRecursive(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "sub"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "top.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "sub", "nested.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}

	q, j := newIngestJob(t, nil, dir)
	var mu sync.Mutex
	if err := ingestLocalTaskWithOptions(j, q, &mu, IngestOptions{}); err != nil {
		t.Fatalf("ingest: %v", err)
	}

	var count int
	if err := q.Db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Errorf("media rows = %d; want 1 (nested file must be skipped)", count)
	}
}

func TestIngestLocalBadPathErrorsJob(t *testing.T) {
	q, j := newIngestJob(t, []string{"-r"}, filepath.Join(t.TempDir(), "does-not-exist"))
	var mu sync.Mutex
	if err := ingestLocalTaskWithOptions(j, q, &mu, IngestOptions{}); err == nil {
		t.Fatal("expected error for missing directory")
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateError {
		t.Errorf("job state = %v; want Error", got)
	}
}

func TestIngestLocalCanceled(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "a.jpg"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	q, j := newIngestJob(t, nil, dir)
	var mu sync.Mutex
	if err := q.CancelJob(j.ID); err != nil {
		t.Fatal(err)
	}
	err := ingestLocalTaskWithOptions(j, q, &mu, IngestOptions{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v; want context.Canceled", err)
	}
}

func TestMediaInsertBatchSizeBackfill(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if err := ensureMediaTableSchema(db); err != nil {
		t.Fatal(err)
	}

	b := newMediaInsertBatch(db)
	defer b.Discard()
	if err := b.Add("/x/a.jpg", 0); err != nil {
		t.Fatal(err)
	}
	if err := b.Add("/x/b.jpg", 7); err != nil {
		t.Fatal(err)
	}
	if n, err := b.Flush(); err != nil || n != 2 {
		t.Fatalf("Flush() = %d, %v; want 2, nil", n, err)
	}
	// Flushing again with nothing pending is a no-op.
	if n, err := b.Flush(); err != nil || n != 0 {
		t.Fatalf("empty Flush() = %d, %v; want 0, nil", n, err)
	}

	// A later real size backfills a 0/NULL size but never overwrites one.
	if err := insertMediaRecord(db, "/x/a.jpg", 5); err != nil {
		t.Fatal(err)
	}
	if err := insertMediaRecord(db, "/x/a.jpg", 9); err != nil {
		t.Fatal(err)
	}
	var size int64
	if err := db.QueryRow(`SELECT size FROM media WHERE path = '/x/a.jpg'`).Scan(&size); err != nil {
		t.Fatal(err)
	}
	if size != 5 {
		t.Errorf("size = %d; want 5 (backfilled once, not overwritten)", size)
	}
}
