package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/runners"
	"github.com/stevecastle/shrike/storage"
	_ "modernc.org/sqlite"
)

// newJobControlEnv wires the real pipeline end-to-end: an on-disk sqlite DB
// with a media table, the job queue, the runner pool, and an HTTP mux with
// the create/pause/resume routes — everything a live server uses except auth
// and the tray.
func newJobControlEnv(t *testing.T) (*Dependencies, *http.ServeMux, *sql.DB) {
	t.Helper()
	dbPath := filepath.Join(t.TempDir(), "test.db")
	db, err := sql.Open("sqlite", sqliteDSN(dbPath))
	if err != nil {
		t.Fatal(err)
	}
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

	q := jobqueue.NewQueueWithDB(db)
	deps := &Dependencies{Queue: q, Storage: storage.NewRegistry(nil)}
	r := runners.New(q)
	t.Cleanup(r.Shutdown)

	mux := http.NewServeMux()
	mux.HandleFunc("/create", createJobHandler(deps))
	mux.HandleFunc("/job/{id}/pause", pauseJobHandler(deps))
	mux.HandleFunc("/job/{id}/resume", resumeJobHandler(deps))
	return deps, mux, db
}

func jobCtlPost(t *testing.T, mux *http.ServeMux, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var reader *strings.Reader
	if body != nil {
		b, _ := json.Marshal(body)
		reader = strings.NewReader(string(b))
	} else {
		reader = strings.NewReader("")
	}
	req := httptest.NewRequest(http.MethodPost, path, reader)
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	return rr
}

func waitForState(t *testing.T, q *jobqueue.Queue, id string, want jobqueue.JobState, timeout time.Duration) *jobqueue.Job {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		j := q.GetJob(id)
		if j != nil && j.State == want {
			return j
		}
		if time.Now().After(deadline) {
			state := jobqueue.JobState(-1)
			if j != nil {
				state = j.State
			}
			t.Fatalf("job %s never reached %v (state=%v)", id, want, state)
		}
		time.Sleep(25 * time.Millisecond)
	}
}

// TestCreateHashJobEndToEnd drives a real hash job through POST /create, the
// runner, and the unified item runner, asserting per-item DB writes and
// progress totals land.
func TestCreateHashJobEndToEnd(t *testing.T) {
	deps, mux, db := newJobControlEnv(t)

	dir := t.TempDir()
	var paths []string
	for i := 0; i < 3; i++ {
		p := filepath.Join(dir, fmt.Sprintf("f%d.jpg", i))
		if err := os.WriteFile(p, []byte("data-"+fmt.Sprint(i)), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
	}

	rr := jobCtlPost(t, mux, "/create", map[string]any{
		"input": `hash "` + strings.Join(paths, ",") + `"`,
	})
	if rr.Code != http.StatusCreated {
		t.Fatalf("create status = %d: %s", rr.Code, rr.Body.String())
	}
	var resp struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &resp)

	job := waitForState(t, deps.Queue, resp.ID, jobqueue.StateCompleted, 10*time.Second)
	if job.ProgressTotal != 3 || job.ProgressDone != 3 {
		t.Errorf("progress = %d/%d, want 3/3", job.ProgressDone, job.ProgressTotal)
	}
	for _, p := range paths {
		var hash sql.NullString
		if err := db.QueryRow(`SELECT hash FROM media WHERE path = ?`, p).Scan(&hash); err != nil {
			t.Fatal(err)
		}
		if !hash.Valid || hash.String == "" {
			t.Errorf("no hash written for %s", p)
		}
	}
}

// TestPauseResumeEndToEnd exercises the new pause/resume endpoints: a pending
// job (queued behind a running one in the same host bucket) pauses
// immediately, is skipped by the scheduler, and resumes to completion.
func TestPauseResumeEndToEnd(t *testing.T) {
	deps, mux, db := newJobControlEnv(t)

	p := filepath.Join(t.TempDir(), "one.jpg")
	if err := os.WriteFile(p, []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
		t.Fatal(err)
	}

	// The wait task occupies the localhost bucket (limit 1) for ~5s.
	rr := jobCtlPost(t, mux, "/create", map[string]any{"input": "wait blocker"})
	if rr.Code != http.StatusCreated {
		t.Fatalf("create wait status = %d: %s", rr.Code, rr.Body.String())
	}
	var waitResp struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &waitResp)
	waitForState(t, deps.Queue, waitResp.ID, jobqueue.StateInProgress, 5*time.Second)

	// The hash job shares the bucket, so it parks as pending.
	rr = jobCtlPost(t, mux, "/create", map[string]any{"input": `hash "` + p + `"`})
	var hashResp struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &hashResp)

	// Pause it while pending: immediate transition.
	if rr := jobCtlPost(t, mux, "/job/"+hashResp.ID+"/pause", nil); rr.Code != http.StatusOK {
		t.Fatalf("pause status = %d: %s", rr.Code, rr.Body.String())
	}
	waitForState(t, deps.Queue, hashResp.ID, jobqueue.StatePaused, 2*time.Second)

	// Paused jobs must not be claimed even after the blocker finishes.
	waitForState(t, deps.Queue, waitResp.ID, jobqueue.StateCompleted, 10*time.Second)
	time.Sleep(150 * time.Millisecond)
	if got := deps.Queue.GetJob(hashResp.ID).State; got != jobqueue.StatePaused {
		t.Fatalf("paused job was scheduled anyway: state=%v", got)
	}

	// Resume → runs → completes with its writes.
	if rr := jobCtlPost(t, mux, "/job/"+hashResp.ID+"/resume", nil); rr.Code != http.StatusOK {
		t.Fatalf("resume status = %d: %s", rr.Code, rr.Body.String())
	}
	waitForState(t, deps.Queue, hashResp.ID, jobqueue.StateCompleted, 10*time.Second)

	var hash sql.NullString
	if err := db.QueryRow(`SELECT hash FROM media WHERE path = ?`, p).Scan(&hash); err != nil {
		t.Fatal(err)
	}
	if !hash.Valid || hash.String == "" {
		t.Fatal("resumed job did not write its hash")
	}
}
