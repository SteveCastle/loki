package jobqueue

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/stevecastle/shrike/renderer"
)

// Run-tracking contract: StartedAt = first claim (not queue time), kept
// through pause/resume and restarts; CompletedAt = end of the run on EVERY
// terminal transition; duration_ms derived from the two in the JSON.

func decodeJobJSON(t *testing.T, j Job) map[string]any {
	t.Helper()
	b, err := json.Marshal(j)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestStartedAtStampedOnClaimNotQueue(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id, _ := q.AddJob("", "wait", nil, "input", nil)
	if j := q.GetJob(id); !j.StartedAt.IsZero() || !j.CompletedAt.IsZero() {
		t.Fatalf("queued job must have no start/end, got %v / %v", j.StartedAt, j.CompletedAt)
	}
	if m := decodeJobJSON(t, *q.GetJob(id)); m["duration_ms"] != nil {
		t.Fatalf("duration_ms must be null before the job runs, got %v", m["duration_ms"])
	}

	before := time.Now()
	j, _ := q.ClaimJob()
	if j == nil || j.StartedAt.Before(before) {
		t.Fatalf("StartedAt must be set at claim, got %v (before %v)", j.StartedAt, before)
	}
	if !j.CompletedAt.IsZero() {
		t.Fatal("running job must have no CompletedAt")
	}
	if m := decodeJobJSON(t, *j); m["duration_ms"] != nil {
		t.Fatalf("running job duration_ms must be null, got %v", m["duration_ms"])
	}
}

func TestStartedAtSurvivesPauseResume(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	j := addAndClaim(t, q)
	start := j.StartedAt
	_ = q.RequestPause(j.ID, false)
	if err := q.PauseJob(j.ID); err != nil {
		t.Fatal(err)
	}
	if !j.CompletedAt.IsZero() {
		t.Fatal("pause is not terminal: CompletedAt must stay unset")
	}
	if err := q.ResumeJob(j.ID); err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Millisecond)
	j2, _ := q.ClaimJob()
	if j2 == nil || j2.ID != j.ID {
		t.Fatalf("resumed job not reclaimed: %v", j2)
	}
	if !j2.StartedAt.Equal(start) {
		t.Fatalf("StartedAt must be the first start (%v), got %v", start, j2.StartedAt)
	}
	if !j2.ClaimedAt.After(start) {
		t.Fatalf("ClaimedAt should track the new segment, got %v", j2.ClaimedAt)
	}
	if err := q.CompleteJob(j.ID); err != nil {
		t.Fatal(err)
	}
	d, ok := j2.RunDuration()
	if !ok || d != j2.CompletedAt.Sub(start) {
		t.Fatalf("RunDuration = %v,%v; want %v", d, ok, j2.CompletedAt.Sub(start))
	}
}

func TestCompletedAtOnEveryTerminalState(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	// completed
	j := addAndClaim(t, q)
	if err := q.CompleteJob(j.ID); err != nil {
		t.Fatal(err)
	}
	m := decodeJobJSON(t, *q.GetJob(j.ID))
	if j.CompletedAt.IsZero() || m["duration_ms"] == nil {
		t.Fatalf("completed: CompletedAt=%v duration_ms=%v", j.CompletedAt, m["duration_ms"])
	}
	if _, ok := m["started_at"]; !ok {
		t.Fatal("JSON must carry started_at")
	}

	// error
	j = addAndClaim(t, q)
	if err := q.ErrorJob(j.ID); err != nil {
		t.Fatal(err)
	}
	if j.CompletedAt.IsZero() || !j.CompletedAt.Equal(j.ErroredAt) {
		t.Fatalf("error: CompletedAt=%v ErroredAt=%v", j.CompletedAt, j.ErroredAt)
	}
	if _, ok := j.RunDuration(); !ok {
		t.Fatal("error: duration must be known")
	}

	// cancelled while running
	j = addAndClaim(t, q)
	if err := q.CancelJob(j.ID); err != nil {
		t.Fatal(err)
	}
	q.ReleaseCancelled(j.ID)
	if j.CompletedAt.IsZero() {
		t.Fatal("cancel (running): CompletedAt must be set")
	}
	if _, ok := j.RunDuration(); !ok {
		t.Fatal("cancel (running): duration must be known")
	}

	// cancelled before it ever ran: ended, but no start and no duration
	id, _ := q.AddJob("", "wait", nil, "x", []string{"missing-dep"})
	if err := q.CancelJob(id); err != nil {
		t.Fatal(err)
	}
	p := q.GetJob(id)
	if p.CompletedAt.IsZero() || !p.StartedAt.IsZero() {
		t.Fatalf("cancel (pending): CompletedAt=%v StartedAt=%v", p.CompletedAt, p.StartedAt)
	}
	if m := decodeJobJSON(t, *p); m["duration_ms"] != nil {
		t.Fatalf("never-run job duration_ms must be null, got %v", m["duration_ms"])
	}
}

func TestWorkflowDependentCancelStampsCompletedAt(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	ids, err := q.AddWorkflow(Workflow{Tasks: []WorkflowTask{
		{ID: "wf-a", Command: "wait", Input: "a"},
		{ID: "wf-b", Command: "wait", Dependencies: []string{"wf-a"}},
	}})
	if err != nil || len(ids) != 2 {
		t.Fatalf("AddWorkflow: %v %v", ids, err)
	}
	a, _ := q.ClaimJob()
	if a == nil || a.ID != "wf-a" {
		t.Fatalf("expected wf-a claimed, got %v", a)
	}
	if err := q.ErrorJob("wf-a"); err != nil {
		t.Fatal(err)
	}
	b := q.GetJob("wf-b")
	if b.State != StateCancelled || b.CompletedAt.IsZero() {
		t.Fatalf("dependent: state=%v CompletedAt=%v", b.State, b.CompletedAt)
	}
}

func TestCopyJobResetsRunTimes(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	j := addAndClaim(t, q)
	_ = q.CompleteJob(j.ID)
	newID, err := q.CopyJob(j.ID)
	if err != nil {
		t.Fatal(err)
	}
	c := q.GetJob(newID)
	if !c.StartedAt.IsZero() || !c.CompletedAt.IsZero() {
		t.Fatalf("copy must start fresh, got %v / %v", c.StartedAt, c.CompletedAt)
	}
}

func TestRunTimesPersistAcrossRestart(t *testing.T) {
	q, db := openFileQueue(t)

	done := addAndClaim(t, q)
	_ = q.CompleteJob(done.ID)
	running := addAndClaim(t, q) // still in progress at "crash"

	q2 := NewQueueWithDB(db)
	t.Cleanup(q2.Close)

	d := q2.GetJob(done.ID)
	if !d.StartedAt.Equal(done.StartedAt) || !d.CompletedAt.Equal(done.CompletedAt) {
		t.Fatalf("completed job times not persisted: %v/%v want %v/%v",
			d.StartedAt, d.CompletedAt, done.StartedAt, done.CompletedAt)
	}
	if _, ok := d.RunDuration(); !ok {
		t.Fatal("reloaded completed job must have a duration")
	}

	r := q2.GetJob(running.ID)
	if r.State != StatePending {
		t.Fatalf("interrupted job should be auto-resumed (pending), got %v", r.State)
	}
	if !r.StartedAt.Equal(running.StartedAt) {
		t.Fatalf("interrupted job must keep its original start %v, got %v", running.StartedAt, r.StartedAt)
	}
	if !r.ClaimedAt.IsZero() || !r.CompletedAt.IsZero() {
		t.Fatalf("resumed job: ClaimedAt=%v CompletedAt=%v", r.ClaimedAt, r.CompletedAt)
	}
}

// A jobs table from before started_at existed: the migration adds the column
// idempotently and the loader falls back to claimed_at / errored_at.
func TestLegacyJobsTableMigratesRunTimes(t *testing.T) {
	q, db := openFileQueue(t)
	q.Close()
	if _, err := db.Exec(`DROP TABLE jobs`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TABLE jobs (
		id TEXT PRIMARY KEY, command TEXT NOT NULL, arguments TEXT, input TEXT,
		original_input TEXT, host TEXT, stdout TEXT, dependencies TEXT,
		state INTEGER NOT NULL, created_at DATETIME NOT NULL, claimed_at DATETIME,
		completed_at DATETIME, errored_at DATETIME, job_order_position INTEGER)`); err != nil {
		t.Fatal(err)
	}
	created := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	claimed := created.Add(time.Minute)
	errored := claimed.Add(90 * time.Second)
	if _, err := db.Exec(`INSERT INTO jobs (id, command, arguments, input, original_input, host,
		stdout, dependencies, state, created_at, claimed_at, completed_at, errored_at, job_order_position)
		VALUES ('old', 'wait', '[]', '', '', 'localhost', '[]', '[]', ?, ?, ?, ?, ?, 0)`,
		int(StateError), created, claimed, time.Time{}, errored); err != nil {
		t.Fatal(err)
	}

	for i := 0; i < 2; i++ { // twice: the migration must be idempotent
		q2 := NewQueueWithDB(db)
		j := q2.GetJob("old")
		if j == nil {
			t.Fatalf("pass %d: legacy job not loaded", i)
		}
		if !j.StartedAt.Equal(claimed) {
			t.Fatalf("pass %d: StartedAt = %v, want claimed_at %v", i, j.StartedAt, claimed)
		}
		if !j.CompletedAt.Equal(errored) {
			t.Fatalf("pass %d: CompletedAt = %v, want errored_at %v", i, j.CompletedAt, errored)
		}
		if d, ok := j.RunDuration(); !ok || d != 90*time.Second {
			t.Fatalf("pass %d: RunDuration = %v,%v", i, d, ok)
		}
		// Re-save so the second pass reads a real started_at column value.
		q2.mu.Lock()
		_ = q2.saveJobToDB(j)
		q2.mu.Unlock()
		q2.Close()
	}
}

// The Go-template jobs pages show the run times: a ticking elapsed span for a
// running job, the final duration once it has ended.
func TestJobTemplatesShowRunTiming(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	j := addAndClaim(t, q)
	var row bytes.Buffer
	if err := renderer.Templates().ExecuteTemplate(&row, "jobRow", j); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(row.String(), `data-run-start="`) {
		t.Fatalf("running row must carry a live elapsed span:\n%s", row.String())
	}

	j.StartedAt = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)
	j.CompletedAt = j.StartedAt.Add(75 * time.Second)
	j.State = StateCompleted
	row.Reset()
	if err := renderer.Templates().ExecuteTemplate(&row, "jobRow", j); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(row.String(), "took 1m 15s") || strings.Contains(row.String(), "data-run-start") {
		t.Fatalf("finished row must show the duration and not tick:\n%s", row.String())
	}

	var hdr bytes.Buffer
	if err := renderer.Templates().ExecuteTemplate(&hdr, "detailHeader", j); err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`id="startedAt"`, `id="completedAt"`, "Duration:", "1m 15s"} {
		if !strings.Contains(hdr.String(), want) {
			t.Fatalf("detail header missing %q:\n%s", want, hdr.String())
		}
	}
}
