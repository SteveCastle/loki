package tasks

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	_ "modernc.org/sqlite"
)

// A prepared op's Finalize hook runs once when the run completes successfully,
// so an op (faces) can queue follow-up work after all items are committed.
func TestRunItemOpsCallsFinalizeOnCompletion(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 2)
	var finalizeCalls int32
	registerTestOp(t, "test-op-finalize", func(op *ItemOp) {
		base := op.Prepare
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			p, err := base(run)
			if err != nil {
				return nil, err
			}
			p.Finalize = func() error { atomic.AddInt32(&finalizeCalls, 1); return nil }
			return p, nil
		}
	})

	q, j := newItemOpsJob(t, db, "test-op-finalize", nil, strings.Join(paths, "\n"))
	if err := runItemOps(j, q, []string{"test-op-finalize"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}
	if got := atomic.LoadInt32(&finalizeCalls); got != 1 {
		t.Fatalf("Finalize calls = %d, want 1", got)
	}
}

// Finalize must run while the job is still IN PROGRESS — before CompleteJob.
// Finalization can be minutes of every-core compute (the faces op's final full
// clustering pass); marking the job complete first left the server pegged with
// nothing visibly running, nothing holding the job's concurrency buckets, and
// no way to cancel.
func TestRunItemOpsFinalizeRunsBeforeComplete(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 1)

	var q *jobqueue.Queue
	var jobID string
	stateAtFinalize := make(chan jobqueue.JobState, 1)
	registerTestOp(t, "test-op-finalize-order", func(op *ItemOp) {
		base := op.Prepare
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			p, err := base(run)
			if err != nil {
				return nil, err
			}
			p.Finalize = func() error {
				for _, j := range q.GetJobs() {
					if j.ID == jobID {
						stateAtFinalize <- j.State
					}
				}
				return nil
			}
			return p, nil
		}
	})

	q2, j := newItemOpsJob(t, db, "test-op-finalize-order", nil, strings.Join(paths, "\n"))
	q = q2
	jobID = j.ID
	if err := runItemOps(j, q2, []string{"test-op-finalize-order"}, false); err != nil {
		t.Fatalf("runItemOps: %v", err)
	}
	select {
	case st := <-stateAtFinalize:
		if st != jobqueue.StateInProgress {
			t.Fatalf("job state at Finalize = %q, want in-progress (Finalize must precede CompleteJob)", st)
		}
	default:
		t.Fatal("Finalize never observed the job")
	}
	// And the job still completes afterward.
	for _, jj := range q2.GetJobs() {
		if jj.ID == jobID && jj.State != jobqueue.StateCompleted {
			t.Fatalf("final job state = %q, want completed", jj.State)
		}
	}
}

// Finalize must NOT run when the job pauses (or cancels/errors) — the run
// isn't complete, so follow-up work would be premature. The eventual resumed
// completion fires it.
func TestRunItemOpsSkipsFinalizeOnPause(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 5)

	var q *jobqueue.Queue
	var jobID string
	var finalizeCalls int32
	registerTestOp(t, "test-op-finalize-pause", func(op *ItemOp) {
		op.Prepare = func(run *ItemRun) (*ItemProcessor, error) {
			return &ItemProcessor{
				Process: func(ctx context.Context, path, _ string) (*ItemCommit, error) {
					_ = q.RequestPause(jobID)
					return &ItemCommit{Commit: func() error { return nil }}, nil
				},
				Finalize: func() error { atomic.AddInt32(&finalizeCalls, 1); return nil },
			}, nil
		}
	})

	q2, j := newItemOpsJob(t, db, "test-op-finalize-pause", nil, strings.Join(paths, "\n"))
	q = q2
	jobID = j.ID
	if err := runItemOps(j, q2, []string{"test-op-finalize-pause"}, false); !errors.Is(err, jobqueue.ErrPaused) {
		t.Fatalf("expected ErrPaused, got %v", err)
	}
	if got := atomic.LoadInt32(&finalizeCalls); got != 0 {
		t.Fatalf("Finalize must not run on pause; calls = %d", got)
	}
}

func countFacesClusterJobs(q *jobqueue.Queue) int {
	n := 0
	for _, j := range q.GetJobs() {
		if j.Command == "faces-cluster" {
			n++
		}
	}
	return n
}

// maybeQueueFaceClustering gates on new-face count and dedups against an
// already-pending cluster job.
func TestMaybeQueueFaceClustering(t *testing.T) {
	db := setupItemOpsDB(t)
	q := jobqueue.NewQueueWithDB(db)

	// Zero new faces: nothing to cluster.
	if _, queued := maybeQueueFaceClustering(q, 0); queued {
		t.Fatal("zero new faces must not queue a cluster job")
	}
	if n := countFacesClusterJobs(q); n != 0 {
		t.Fatalf("cluster jobs = %d, want 0", n)
	}

	// New faces, none pending: queue exactly one.
	if _, queued := maybeQueueFaceClustering(q, 3); !queued {
		t.Fatal("new faces must queue a cluster job")
	}
	if n := countFacesClusterJobs(q); n != 1 {
		t.Fatalf("cluster jobs = %d, want 1", n)
	}

	// New faces again while the first is still pending: no duplicate.
	if _, queued := maybeQueueFaceClustering(q, 5); queued {
		t.Fatal("must not queue a second cluster job while one is pending")
	}
	if n := countFacesClusterJobs(q); n != 1 {
		t.Fatalf("cluster jobs = %d, want 1 (deduped)", n)
	}
}
