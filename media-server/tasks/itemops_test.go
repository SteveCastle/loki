package tasks

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
	_ "modernc.org/sqlite"
)

// setupItemOpsDB creates the minimal media schema the unified runner and the
// simple metadata ops touch.
func setupItemOpsDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	// One connection: the runner reads/writes from multiple goroutines and a
	// pooled :memory: DB would give each connection its own empty database.
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(`CREATE TABLE media (
		path TEXT PRIMARY KEY,
		hash TEXT,
		size INTEGER,
		width INTEGER,
		height INTEGER,
		description TEXT,
		transcript TEXT
	)`); err != nil {
		t.Fatal(err)
	}
	return db
}

// writeTempMedia creates n small .jpg files on disk and inserts them into the
// media table, returning their paths.
func writeTempMedia(t *testing.T, db *sql.DB, n int) []string {
	t.Helper()
	dir := t.TempDir()
	paths := make([]string, 0, n)
	for i := 0; i < n; i++ {
		p := filepath.Join(dir, fmt.Sprintf("item-%d.jpg", i))
		if err := os.WriteFile(p, []byte("jpegdata"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
	}
	return paths
}

// newItemOpsJob adds a job to a fresh queue and claims it (so it is
// InProgress, as it would be when a runner invokes the task fn).
func newItemOpsJob(t *testing.T, db *sql.DB, command string, args []string, input string) (*jobqueue.Queue, *jobqueue.Job) {
	t.Helper()
	q := jobqueue.NewQueueWithDB(db)
	id, err := q.AddJob("", command, args, input, nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	return q, j
}

// registerTestOp registers a fake op that records processed paths and applies
// commits through the runner's committer. Returns the processed-path recorder.
func registerTestOp(t *testing.T, id string, opts ...func(*ItemOp)) *sync.Map {
	t.Helper()
	processed := &sync.Map{}
	op := ItemOp{
		ID:   id,
		Name: "Test Op " + id,
		Prepare: func(run *ItemRun) (*ItemProcessor, error) {
			return &ItemProcessor{
				Process: func(ctx context.Context, path, _ string) (*ItemCommit, error) {
					return &ItemCommit{
						Commit: func() error {
							processed.Store(id+"|"+path, true)
							return nil
						},
						Detail: "done",
					}, nil
				},
			}, nil
		},
	}
	for _, o := range opts {
		o(&op)
	}
	RegisterItemOp(op)
	t.Cleanup(func() {
		delete(itemOps, id)
		for i, oid := range itemOpsOrder {
			if oid == id {
				itemOpsOrder = append(itemOpsOrder[:i], itemOpsOrder[i+1:]...)
				break
			}
		}
	})
	return processed
}

func countStored(m *sync.Map) int {
	n := 0
	m.Range(func(_, _ any) bool { n++; return true })
	return n
}

// TestBuiltinOpsAreCombinable pins the full set of ops the "process" task can
// combine — faces included. A missing entry here means a per-item task
// silently fell out of the unified system.
func TestBuiltinOpsAreCombinable(t *testing.T) {
	want := []string{"describe", "transcribe", "hash", "dimensions", "embed", "autotag", "faces"}
	ids := ItemOpIDs()
	have := make(map[string]bool, len(ids))
	for _, id := range ids {
		have[id] = true
	}
	for _, id := range want {
		if !have[id] {
			t.Errorf("op %q is not registered as a combinable ItemOp", id)
		}
	}
	// And the process task must offer every registered op as a choice.
	choices := processTaskOptions()[0].Choices
	if len(choices) != len(ids) {
		t.Errorf("process --ops choices = %v, want all registered ops %v", choices, ids)
	}
}

func TestResolveJobItemsInputList(t *testing.T) {
	db := setupItemOpsDB(t)
	q := jobqueue.NewQueueWithDB(db)
	paths := writeTempMedia(t, db, 2)

	input := paths[0] + "\n--overwrite\n" + paths[1] + "\nnotes.txt"
	id, _ := q.AddJob("", "hash", nil, input, nil)
	j := q.GetJob(id)

	res, err := resolveJobItems(j, q)
	if err != nil {
		t.Fatal(err)
	}
	if res.FromQuery {
		t.Fatal("expected input-list resolution, got query")
	}
	// Flag token dropped, .txt filtered as non-media, both jpgs kept.
	if len(res.Paths) != 2 {
		t.Fatalf("expected 2 items, got %d: %v", len(res.Paths), res.Paths)
	}
}

func TestRunItemOpsProcessesAllItemsAndReportsProgress(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 3)
	processed := registerTestOp(t, "test-op-basic")

	q, j := newItemOpsJob(t, db, "test-op-basic", nil, strings.Join(paths, "\n"))
	if err := runItemOps(j, q, []string{"test-op-basic"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}

	if got := countStored(processed); got != 3 {
		t.Fatalf("expected 3 processed items, got %d", got)
	}
	if j.State != jobqueue.StateCompleted {
		t.Fatalf("expected Completed, got %v", j.State)
	}
	if j.ProgressTotal != 3 || j.ProgressDone != 3 {
		t.Fatalf("expected progress 3/3, got %d/%d", j.ProgressDone, j.ProgressTotal)
	}
	if len(j.OutputFiles) != 3 {
		t.Fatalf("expected 3 output files, got %d", len(j.OutputFiles))
	}
}

func TestRunItemOpsSkipExistingUnlessOverwrite(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 2)

	// Op reports the FIRST path as already having output.
	existing := paths[0]
	processed := registerTestOp(t, "test-op-skip", func(op *ItemOp) {
		base := op.Prepare
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			p, err := base(run)
			if err != nil {
				return nil, err
			}
			p.SkipExisting = func(path string) (bool, error) { return path == existing, nil }
			return p, nil
		}
	})

	q, j := newItemOpsJob(t, db, "test-op-skip", nil, strings.Join(paths, "\n"))
	if err := runItemOps(j, q, []string{"test-op-skip"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}
	if got := countStored(processed); got != 1 {
		t.Fatalf("without overwrite expected 1 processed, got %d", got)
	}
	if j.ProgressDone != 2 {
		t.Fatalf("skips still advance progress; got done=%d", j.ProgressDone)
	}

	// Same input with --overwrite processes both.
	processed2 := registerTestOp(t, "test-op-skip2", func(op *ItemOp) {
		base := op.Prepare
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			p, err := base(run)
			if err != nil {
				return nil, err
			}
			p.SkipExisting = func(path string) (bool, error) { return path == existing, nil }
			return p, nil
		}
	})
	q2, j2 := newItemOpsJob(t, db, "test-op-skip2", []string{"--overwrite"}, strings.Join(paths, "\n"))
	if err := runItemOps(j2, q2, []string{"test-op-skip2"}, false); err != nil {
		t.Fatalf("runItemOps overwrite: %v", err)
	}
	if got := countStored(processed2); got != 2 {
		t.Fatalf("with overwrite expected 2 processed, got %d", got)
	}
}

func TestRunItemOpsCombinedOpsAreAppliedPerItem(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 2)
	pa := registerTestOp(t, "test-op-a")
	pb := registerTestOp(t, "test-op-b")

	q, j := newItemOpsJob(t, db, "process", nil, strings.Join(paths, "\n"))
	if err := runItemOps(j, q, []string{"test-op-a", "test-op-b"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}
	if countStored(pa) != 2 || countStored(pb) != 2 {
		t.Fatalf("expected both ops applied to both items, got a=%d b=%d", countStored(pa), countStored(pb))
	}
	// One output registration per item, not per op.
	if len(j.OutputFiles) != 2 {
		t.Fatalf("expected 2 output files, got %d", len(j.OutputFiles))
	}
}

func TestRunItemOpsPausePreservesProgress(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 5)

	var q *jobqueue.Queue
	var jobID string
	processed := registerTestOp(t, "test-op-pause", func(op *ItemOp) {
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			return &ItemProcessor{
				Process: func(ctx context.Context, path, _ string) (*ItemCommit, error) {
					// Request pause during the first item: the feeder must stop
					// before the next item while this one's commit still lands.
					_ = q.RequestPause(jobID)
					return &ItemCommit{Commit: func() error { return nil }, Detail: "done"}, nil
				},
			}, nil
		}
	})
	_ = processed

	q2, j := newItemOpsJob(t, db, "test-op-pause", nil, strings.Join(paths, "\n"))
	q = q2
	jobID = j.ID

	err := runItemOps(j, q2, []string{"test-op-pause"}, false)
	if !errors.Is(err, jobqueue.ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", err)
	}
	if j.ProgressDone == 0 || j.ProgressDone >= j.ProgressTotal {
		t.Fatalf("expected partial progress, got %d/%d", j.ProgressDone, j.ProgressTotal)
	}

	// The runner would now transition the job to Paused; verify resume works.
	if err := q2.PauseJob(j.ID); err != nil {
		t.Fatalf("PauseJob: %v", err)
	}
	if j.State != jobqueue.StatePaused {
		t.Fatalf("expected Paused, got %v", j.State)
	}
	if err := q2.ResumeJob(j.ID); err != nil {
		t.Fatalf("ResumeJob: %v", err)
	}
	if j.State != jobqueue.StatePending {
		t.Fatalf("expected Pending after resume, got %v", j.State)
	}
	if j.ProgressDone == 0 {
		t.Fatal("resume must keep the persisted progress")
	}
}

// TestHashOpEndToEnd exercises a real op (hash) through the standalone task
// fn, verifying per-item DB writes land.
func TestHashOpEndToEnd(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 2)

	q, j := newItemOpsJob(t, db, "hash", nil, strings.Join(paths, "\n"))
	fn := makeItemOpTaskFn("hash")
	var mu sync.Mutex
	if err := fn(j, q, &mu); err != nil {
		t.Fatalf("hash task: %v", err)
	}

	for _, p := range paths {
		var hash sql.NullString
		var size sql.NullInt64
		if err := db.QueryRow(`SELECT hash, size FROM media WHERE path = ?`, p).Scan(&hash, &size); err != nil {
			t.Fatal(err)
		}
		if !hash.Valid || hash.String == "" || !size.Valid || size.Int64 == 0 {
			t.Fatalf("expected hash+size written for %s, got hash=%v size=%v", p, hash, size)
		}
	}
	if j.State != jobqueue.StateCompleted {
		t.Fatalf("expected Completed, got %v", j.State)
	}
}

// TestMetadataAliasMapsTypesToOps verifies the legacy task still works and
// routes --type values onto the split-out ops.
func TestMetadataAliasMapsTypesToOps(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 1)

	q, j := newItemOpsJob(t, db, "metadata", []string{"--type", "hash"}, paths[0])
	var mu sync.Mutex
	if err := metadataTask(j, q, &mu); err != nil {
		t.Fatalf("metadata alias: %v", err)
	}
	var hash sql.NullString
	if err := db.QueryRow(`SELECT hash FROM media WHERE path = ?`, paths[0]).Scan(&hash); err != nil {
		t.Fatal(err)
	}
	if !hash.Valid || hash.String == "" {
		t.Fatal("expected the legacy metadata task to run the hash op")
	}
}

// fakeS3Backend is an in-memory storage.Backend for exercising the s3://
// localize path without a real bucket.
type fakeS3Backend struct {
	files map[string][]byte
}

func (f *fakeS3Backend) List(ctx context.Context, path string) ([]storage.Entry, error) {
	return nil, nil
}
func (f *fakeS3Backend) Scan(ctx context.Context, path string, recursive bool) ([]storage.FileInfo, error) {
	return nil, nil
}
func (f *fakeS3Backend) Download(ctx context.Context, path string) (io.ReadCloser, error) {
	b, ok := f.files[path]
	if !ok {
		return nil, fmt.Errorf("no such object: %s", path)
	}
	return io.NopCloser(bytes.NewReader(b)), nil
}
func (f *fakeS3Backend) Upload(ctx context.Context, path string, r io.Reader, contentType string) error {
	return fmt.Errorf("not implemented")
}
func (f *fakeS3Backend) MediaURL(path string) (string, error) { return path, nil }
func (f *fakeS3Backend) Exists(ctx context.Context, path string) (bool, error) {
	_, ok := f.files[path]
	return ok, nil
}
func (f *fakeS3Backend) Contains(path string) bool { return strings.HasPrefix(path, "s3://tb/") }
func (f *fakeS3Backend) Root() storage.Entry {
	return storage.Entry{Name: "tb", Path: "s3://tb/", IsDir: true, Type: "s3"}
}

// TestItemOps_S3ItemsLocalized: an input-list item with an s3:// path must
// pass the existence gate (backend HEAD, not disk stat), be downloaded to a
// readable temp file for the op, keep its s3 path as the commit identity,
// and have the temp file removed afterwards.
func TestItemOps_S3ItemsLocalized(t *testing.T) {
	db := setupItemOpsDB(t)
	const s3path = "s3://tb/uploads/img.jpg"
	if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, s3path); err != nil {
		t.Fatal(err)
	}

	oldReg := storageReg
	SetStorageRegistry(storage.NewRegistry([]storage.Backend{
		&fakeS3Backend{files: map[string][]byte{s3path: []byte("s3-bytes")}},
	}))
	t.Cleanup(func() { storageReg = oldReg })

	var gotLocal string
	var gotBytes []byte
	processed := registerTestOp(t, "test-op-s3", func(op *ItemOp) {
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			return &ItemProcessor{
				Process: func(ctx context.Context, path, localPath string) (*ItemCommit, error) {
					gotLocal = localPath
					b, err := os.ReadFile(localPath)
					if err != nil {
						return nil, err
					}
					gotBytes = b
					return &ItemCommit{
						Commit: func() error {
							processedKey := "test-op-s3|" + path
							_ = processedKey
							return nil
						},
						Detail: "done",
					}, nil
				},
			}, nil
		}
	})
	_ = processed

	q, j := newItemOpsJob(t, db, "test-op-s3", nil, s3path)
	if err := runItemOps(j, q, []string{"test-op-s3"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}

	if gotLocal == "" || gotLocal == s3path {
		t.Fatalf("op did not receive a localized path (got %q)", gotLocal)
	}
	if string(gotBytes) != "s3-bytes" {
		t.Fatalf("localized file contents = %q, want s3-bytes", gotBytes)
	}
	if _, err := os.Stat(gotLocal); !os.IsNotExist(err) {
		t.Fatalf("temp file %s should be cleaned up after the run", gotLocal)
	}
	if j.State != jobqueue.StateCompleted {
		t.Fatalf("job state = %v, want completed", j.State)
	}
}
