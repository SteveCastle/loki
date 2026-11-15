package jobqueue

import "testing"

// A preemptive pause stops a running job NOW (its context is cancelled) and,
// when the dying task reports error or cancel, parks it as Paused with a fresh
// context so Resume can run it again.
func TestPreemptivePauseParksInsteadOfFailing(t *testing.T) {
	for _, report := range []string{"error", "cancel"} {
		t.Run(report, func(t *testing.T) {
			q := newTestQueue(t)
			j := addAndClaim(t, q)
			if err := q.RequestPause(j.ID, true); err != nil {
				t.Fatal(err)
			}
			if j.Ctx.Err() == nil {
				t.Fatal("preemptive pause must cancel the run's context")
			}
			if j.State != StateInProgress || !j.PauseRequested || !q.PausePreempted(j.ID) {
				t.Fatalf("want in-progress pausing job, got state=%v requested=%v", j.State, j.PauseRequested)
			}
			// The task sees its context die and reports it the usual way.
			if report == "error" {
				_ = q.ErrorJob(j.ID)
			} else {
				_ = q.CancelJob(j.ID)
			}
			if j.State != StatePaused {
				t.Fatalf("want Paused, got %v", j.State)
			}
			if j.PauseRequested || q.PausePreempted(j.ID) {
				t.Fatal("pause flags must clear once parked")
			}
			if j.Ctx.Err() != nil {
				t.Fatal("parked job needs a fresh context for its resume")
			}
			if !j.CompletedAt.IsZero() {
				t.Fatal("a paused job has not ended")
			}
			if q.RunningCounts[j.Host] != 0 {
				t.Fatalf("running count not released: %d", q.RunningCounts[j.Host])
			}
			if err := q.ResumeJob(j.ID); err != nil {
				t.Fatal(err)
			}
			again, err := q.ClaimJob()
			if err != nil || again == nil || again.ID != j.ID || again.Ctx.Err() != nil {
				t.Fatalf("resumed job not runnable: %v %v", again, err)
			}
		})
	}
}

// An explicit user cancel during a preemptive pause really cancels.
func TestUserCancelOverridesPreemptivePause(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)
	_ = q.RequestPause(j.ID, true)
	if err := q.UserCancelJob(j.ID); err != nil {
		t.Fatal(err)
	}
	if j.State != StateCancelled {
		t.Fatalf("want Cancelled, got %v", j.State)
	}
}

// A graceful pause leaves the run alone until the task's item loop notices.
func TestGracefulPauseDoesNotCancelRun(t *testing.T) {
	q := newTestQueue(t)
	j := addAndClaim(t, q)
	_ = q.RequestPause(j.ID, false)
	if j.Ctx.Err() != nil {
		t.Fatal("graceful pause must not cancel the run")
	}
	if !j.PauseRequested {
		t.Fatal("pause_requested must be visible to UIs")
	}
	_ = q.ErrorJob(j.ID) // a real failure still fails
	if j.State != StateError {
		t.Fatalf("want Error, got %v", j.State)
	}
}
