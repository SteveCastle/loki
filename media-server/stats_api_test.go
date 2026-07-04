package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

func newStatsTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	stmts := []string{
		`CREATE TABLE media (
			path TEXT PRIMARY KEY,
			description TEXT,
			transcript TEXT,
			size INTEGER,
			hash TEXT,
			width INTEGER,
			height INTEGER
		)`,
		`CREATE TABLE media_tag_by_category (
			media_path TEXT,
			tag_label TEXT,
			category_label TEXT,
			time_stamp REAL,
			created_at INTEGER,
			PRIMARY KEY(media_path, tag_label, category_label, time_stamp)
		)`,
		`CREATE TABLE media_embedding (
			media_path TEXT NOT NULL,
			model TEXT NOT NULL,
			dim INTEGER NOT NULL,
			vector BLOB NOT NULL,
			created_at INTEGER,
			PRIMARY KEY (media_path, model)
		)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatal(err)
		}
	}
	// Two images (one described) and one video without a transcript.
	if _, err := db.Exec(`INSERT INTO media (path, description) VALUES
		('/lib/a.jpg', 'a photo'),
		('/lib/b.jpg', NULL),
		('/lib/c.mp4', NULL)`); err != nil {
		t.Fatal(err)
	}
	return &Dependencies{DB: db}
}

func resetLibStats() {
	libStats.mu.Lock()
	libStats.snapshot = nil
	libStats.deltas = nil
	libStats.dirty = false
	libStats.computeMs = 0
	libStats.computing = false
	libStats.mu.Unlock()
}

func getStats(t *testing.T, deps *Dependencies) map[string]any {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/stats", nil)
	rr := httptest.NewRecorder()
	statsAPIHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, body = %s", rr.Code, rr.Body.String())
	}
	var out map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &out); err != nil {
		t.Fatalf("response not JSON: %v", err)
	}
	return out
}

// Before the first snapshot exists the handler must answer ready:false rather
// than blocking on a potentially minutes-long recount.
func TestStatsAPI_NotReadyBeforeFirstSnapshot(t *testing.T) {
	resetLibStats()
	deps := newStatsTestDeps(t)

	out := getStats(t, deps)
	if ready, _ := out["ready"].(bool); ready {
		t.Fatalf("expected ready=false before first snapshot, got %v", out)
	}

	// The handler kicked off a background recount against this test's DB;
	// wait for it so it can't race the DB close or a later test's state.
	deadline := time.Now().Add(5 * time.Second)
	for {
		libStats.mu.Lock()
		computing := libStats.computing
		libStats.mu.Unlock()
		if !computing {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("background stats compute did not finish")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// Task progress deltas must overlay the snapshot immediately (this is what
// moves the home page bars between recounts) and be retired — not double
// counted — once a recount that includes them installs.
func TestStatsAPI_DeltasOverlaySnapshotAndReconcile(t *testing.T) {
	resetLibStats()
	deps := newStatsTestDeps(t)

	libStats.mu.Lock()
	libStats.computing = true
	libStats.mu.Unlock()
	computeLibraryStats(deps)

	out := getStats(t, deps)
	if got := int(out["withDescription"].(float64)); got != 1 {
		t.Fatalf("snapshot withDescription = %d, want 1", got)
	}

	// A running task reports one more described item.
	applyStatsDelta(tasks.ProgressDescription, 1)
	out = getStats(t, deps)
	if got := int(out["withDescription"].(float64)); got != 2 {
		t.Fatalf("merged withDescription = %d, want 2 (snapshot 1 + delta 1)", got)
	}

	// Recount: the DB still says 1 (the delta was optimistic, e.g. an
	// overwrite re-run). Pre-recount deltas are retired, so the merged view
	// converges back to the database truth instead of double counting.
	libStats.mu.Lock()
	libStats.computing = true
	libStats.mu.Unlock()
	computeLibraryStats(deps)

	out = getStats(t, deps)
	if got := int(out["withDescription"].(float64)); got != 1 {
		t.Fatalf("post-recount withDescription = %d, want 1 (delta retired)", got)
	}
}

// The optimistic overlay must never push a counter past its denominator.
func TestStatsAPI_DeltaClampedToTotals(t *testing.T) {
	resetLibStats()
	deps := newStatsTestDeps(t)

	libStats.mu.Lock()
	libStats.computing = true
	libStats.mu.Unlock()
	computeLibraryStats(deps)

	applyStatsDelta(tasks.ProgressDescription, 999)
	out := getStats(t, deps)
	total := int(out["totalMedia"].(float64))
	if got := int(out["withDescription"].(float64)); got != total {
		t.Fatalf("clamped withDescription = %d, want totalMedia (%d)", got, total)
	}

	applyStatsDelta(tasks.ProgressTranscript, 999)
	out = getStats(t, deps)
	videos := int(out["totalVideos"].(float64))
	if got := int(out["videosWithTranscript"].(float64)); got != videos {
		t.Fatalf("clamped videosWithTranscript = %d, want totalVideos (%d)", got, videos)
	}
}
