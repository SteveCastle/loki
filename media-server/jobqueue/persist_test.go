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
	for i := 0; i < 20000; i++ {
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
