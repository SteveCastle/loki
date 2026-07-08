package tasks

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
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
				Process: func(ctx context.Context, path string) (*ItemCommit, error) {
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
				Process: func(ctx context.Context, path string) (*ItemCommit, error) {
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
