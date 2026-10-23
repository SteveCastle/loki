package jobqueue

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

func openFileQueue(t *testing.T) (*Queue, *sql.DB) {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "q.sqlite")+"?_pragma=journal_mode=WAL")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	q := NewQueueWithDB(db)
	t.Cleanup(q.Close)
	return q, db
}

func storedStdout(t *testing.T, db *sql.DB, id string) string {
	t.Helper()
	var s string
	if err := db.QueryRow(`SELECT stdout FROM jobs WHERE id = ?`, id).Scan(&s); err != nil {
		t.Fatal(err)
	}
	return s
}

// A chatty job must not rewrite its whole row per line: 20k lines used to
// mean 20k INSERT OR REPLACEs of an ever-growing JSON array under q.mu.
func TestPushJobStdoutBurstIsCheap(t *testing.T) {
	q, _ := openFileQueue(t)
	id, _ := q.AddJob("", "wait", nil, "", nil)
	start := time.Now()
	for i := 0; i < 5000; i++ {
		_ = q.PushJobStdout(id, "a moderately long log line from a busy external tool")
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Fatalf("20k stdout lines took %v", d)
	}
}

func TestAppendsReachDBViaFlusherAndClose(t *testing.T) {
	q, db := openFileQueue(t)
	id, _ := q.AddJob("", "wait", nil, "", nil)
	_ = q.PushJobStdout(id, "line one")
	_ = q.RegisterOutputFile(id, "/out/a.jpg")

	deadline := time.Now().Add(3 * dirtyFlushInterval)
	for storedStdout(t, db, id) != `["line one"]` {
		if time.Now().After(deadline) {
			t.Fatalf("flusher never persisted stdout; row has %s", storedStdout(t, db, id))
		}
		time.Sleep(50 * time.Millisecond)
	}

	_ = q.PushJobStdout(id, "line two")
	q.Close()
	if got := storedStdout(t, db, id); got != `["line one","line two"]` {
		t.Fatalf("after Close stdout = %s", got)
	}
	var outs string
	_ = db.QueryRow(`SELECT output_files FROM jobs WHERE id = ?`, id).Scan(&outs)
	if outs != `["/out/a.jpg"]` {
		t.Fatalf("output_files = %s", outs)
	}
}

// State transitions persist synchronously, appended data included.
func TestCompleteJobPersistsPendingAppends(t *testing.T) {
	q, db := openFileQueue(t)
	id, _ := q.AddJob("", "wait", nil, "", nil)
	if _, err := q.ClaimJob(); err != nil {
		t.Fatal(err)
	}
	_ = q.PushJobStdout(id, "done soon")
	if err := q.CompleteJob(id); err != nil {
		t.Fatal(err)
	}
	if got := storedStdout(t, db, id); got != `["done soon"]` {
		t.Fatalf("stdout after CompleteJob = %s", got)
	}
}

// ClaimJob skips the finished prefix of the history, but must still find
// jobs appended later and jobs resumed in the middle of it.
func TestClaimJobLowWaterMark(t *testing.T) {
	q, _ := openFileQueue(t)
	var ids []string
	for i := 0; i < 5; i++ {
		id, _ := q.AddJob("", "wait", nil, "", nil)
		ids = append(ids, id)
	}
	for range ids {
		j, err := q.ClaimJob()
		if err != nil || j == nil {
			t.Fatalf("claim: %v %v", j, err)
		}
		_ = q.CompleteJob(j.ID)
	}
	if j, _ := q.ClaimJob(); j != nil {
		t.Fatalf("claimed %s from an all-finished queue", j.ID)
	}
	late, _ := q.AddJob("", "wait", nil, "", nil)
	if j, _ := q.ClaimJob(); j == nil || j.ID != late {
		t.Fatalf("appended job not claimed: %v", j)
	}
	// Pause the running job and resume it: it sits mid-history, behind the mark.
	if err := q.PauseJob(late); err != nil {
		t.Fatalf("pause: %v", err)
	}
	if _, err := q.ClaimJob(); err != nil {
		t.Fatal(err)
	}
	if err := q.ResumeJob(late); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if j, _ := q.ClaimJob(); j == nil || j.ID != late {
		t.Fatalf("resumed job not claimed: %v", j)
	}
}

func TestClearFinishedJobsLargeHistory(t *testing.T) {
	q, db := openFileQueue(t)
	for i := 0; i < 5000; i++ {
		id, _ := q.AddJob("", "wait", nil, "", nil)
		if i%2 == 0 {
			if j, _ := q.ClaimJob(); j != nil {
				_ = q.CompleteJob(j.ID)
			}
		}
		_ = id
	}
	start := time.Now()
	n, err := q.ClearFinishedJobs()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("cleared %d jobs in %v", n, time.Since(start))
	if d := time.Since(start); d > 5*time.Second {
		t.Fatalf("clearing %d jobs took %v", n, d)
	}
	if n != 2500 || len(q.JobOrder) != 2500 || len(q.Jobs) != 2500 {
		t.Fatalf("cleared %d, left order=%d jobs=%d", n, len(q.JobOrder), len(q.Jobs))
	}
	var rows int
	db.QueryRow(`SELECT COUNT(*) FROM jobs`).Scan(&rows)
	if rows != 2500 {
		t.Fatalf("db rows = %d, want 2500", rows)
	}
	if j, _ := q.ClaimJob(); j == nil {
		t.Fatal("pending jobs not claimable after clear")
	}
}
