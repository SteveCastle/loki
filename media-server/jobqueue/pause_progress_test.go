package jobqueue

import (
	"testing"
)

func addAndClaim(t *testing.T, q *Queue) *Job {
	t.Helper()
	id, err := q.AddJob("", "wait", nil, "input", nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim: %v (%v)", err, j)
	}
	return j
}

func TestPauseRequestAndPauseJob(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)

	if q.PauseRequested(j.ID) {
		t.Fatal("no pause requested yet")
	}
	if err := q.RequestPause(j.ID); err != nil {
		t.Fatal(err)
	}
	if !q.PauseRequested(j.ID) {
		t.Fatal("pause request not visible")
	}
	// The task notices the flag and returns ErrPaused; the runner then calls:
	if err := q.PauseJob(j.ID); err != nil {
		t.Fatal(err)
	}
	if j.State != StatePaused {
		t.Fatalf("want Paused, got %v", j.State)
	}
	if q.PauseRequested(j.ID) {
		t.Fatal("pause request must clear on transition")
	}
	// Paused jobs must not be claimable.
	if claimed, _ := q.ClaimJob(); claimed != nil {
		t.Fatalf("paused job must not be claimed, got %v", claimed.ID)
	}
	// Running count must have been released.
	if q.RunningCounts[j.Host] != 0 {
		t.Fatalf("running count not released: %d", q.RunningCounts[j.Host])
	}
}

func TestRequestPauseOnPendingPausesImmediately(t *testing.T) {
	q := newTestQueue(t)
	id, _ := q.AddJob("", "wait", nil, "input", nil)
	if err := q.RequestPause(id); err != nil {
		t.Fatal(err)
	}
	if q.Jobs[id].State != StatePaused {
		t.Fatalf("pending job should pause immediately, got %v", q.Jobs[id].State)
	}
}

func TestResumeRequeuesAndKeepsProgress(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)
	_ = q.SetJobProgress(j.ID, 7, 20)
	_ = q.RequestPause(j.ID)
	_ = q.PauseJob(j.ID)

	if err := q.ResumeJob(j.ID); err != nil {
		t.Fatal(err)
	}
	if j.State != StatePending {
		t.Fatalf("want Pending after resume, got %v", j.State)
	}
	if j.ProgressDone != 7 || j.ProgressTotal != 20 {
		t.Fatalf("progress lost on resume: %d/%d", j.ProgressDone, j.ProgressTotal)
	}
	// Resumed job is claimable again.
	claimed, err := q.ClaimJob()
	if err != nil || claimed == nil || claimed.ID != j.ID {
		t.Fatalf("resume must make job claimable: %v (%v)", err, claimed)
	}
}

func TestCancelPausedJob(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)
	_ = q.RequestPause(j.ID)
	_ = q.PauseJob(j.ID)

	if err := q.CancelJob(j.ID); err != nil {
		t.Fatalf("cancelling a paused job must work: %v", err)
	}
	if j.State != StateCancelled {
		t.Fatalf("want Cancelled, got %v", j.State)
	}
	if q.RunningCounts[j.Host] != 0 {
		t.Fatalf("running count double-decremented or leaked: %d", q.RunningCounts[j.Host])
	}
}

func TestProgressPersistsAcrossReload(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)
	if err := q.SetJobProgress(j.ID, 5, 10); err != nil {
		t.Fatal(err)
	}
	// SetJobProgress writes a targeted UPDATE; a full row save keeps other
	// columns in sync at state transitions.
	_ = q.RequestPause(j.ID)
	_ = q.PauseJob(j.ID)

	// Simulate a restart: fresh queue over the same DB.
	q2 := NewQueueWithDB(q.Db)
	loaded := q2.GetJob(j.ID)
	if loaded == nil {
		t.Fatal("job not reloaded")
	}
	if loaded.State != StatePaused {
		t.Fatalf("paused state must survive restart, got %v", loaded.State)
	}
	if loaded.ProgressDone != 5 || loaded.ProgressTotal != 10 {
		t.Fatalf("progress must survive restart, got %d/%d", loaded.ProgressDone, loaded.ProgressTotal)
	}
}

func TestPausedStateJSONRoundTrip(t *testing.T) {
	b, err := StatePaused.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `"paused"` {
		t.Fatalf("got %s", b)
	}
	var s JobState
	if err := s.UnmarshalJSON(b); err != nil {
		t.Fatal(err)
	}
	if s != StatePaused {
		t.Fatalf("round trip failed: %v", s)
	}
}
