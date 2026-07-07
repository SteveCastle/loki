package jobqueue

import (
	"testing"
)

// resourceTestResolver routes by command name for these tests:
//
//	"heavy-a"  → [gpu]
//	"heavy-b"  → [bucket-b, gpu]
//	"light"    → nil
func resourceTestResolver(command string, _ []string, _ string) []string {
	switch command {
	case "heavy-a":
		return []string{"gpu"}
	case "heavy-b":
		return []string{"bucket-b", "gpu"}
	default:
		return nil
	}
}

func withResourceResolver(t *testing.T) {
	t.Helper()
	SetResourceResolver(resourceTestResolver)
	t.Cleanup(func() { SetResourceResolver(nil) })
}

func TestSharedResourceBlocksOverlappingJobs(t *testing.T) {
	withResourceResolver(t)
	q := newTestQueue(t)

	// Different Host buckets (hostA/hostB via distinct commands → default
	// resolver gives "localhost"... so force different hosts via limits):
	// commands share only the "gpu" resource.
	idA, _ := q.AddJob("", "heavy-a", nil, "input-a", nil)
	idB, _ := q.AddJob("", "heavy-b", nil, "input-b", nil)
	// Put them in separate host buckets so only the shared resource contends.
	q.Jobs[idA].Host = "host-a"
	q.Jobs[idB].Host = "host-b"

	a, err := q.ClaimJob()
	if err != nil || a == nil || a.ID != idA {
		t.Fatalf("claim A: %v (%v)", err, a)
	}

	// B needs "gpu" which A holds — must NOT be claimable.
	if b, _ := q.ClaimJob(); b != nil {
		t.Fatalf("job B claimed while shared gpu resource was held (got %s)", b.ID)
	}

	if err := q.CompleteJob(idA); err != nil {
		t.Fatal(err)
	}
	// All of A's buckets released — B can now claim.
	b, err := q.ClaimJob()
	if err != nil || b == nil || b.ID != idB {
		t.Fatalf("claim B after release: %v (%v)", err, b)
	}
	// B holds host-b + bucket-b + gpu.
	for _, bucket := range []string{"host-b", "bucket-b", "gpu"} {
		if q.RunningCounts[bucket] != 1 {
			t.Errorf("bucket %q count = %d, want 1", bucket, q.RunningCounts[bucket])
		}
	}
}

func TestResourceCountsReleasedOnEveryTransition(t *testing.T) {
	withResourceResolver(t)

	assertAllZero := func(t *testing.T, q *Queue) {
		t.Helper()
		for bucket, n := range q.RunningCounts {
			if n != 0 {
				t.Errorf("bucket %q leaked: count = %d", bucket, n)
			}
		}
	}

	transitions := map[string]func(q *Queue, id string) error{
		"complete": func(q *Queue, id string) error { return q.CompleteJob(id) },
		"cancel":   func(q *Queue, id string) error { return q.CancelJob(id) },
		"error":    func(q *Queue, id string) error { return q.ErrorJob(id) },
		"pause": func(q *Queue, id string) error {
			if err := q.RequestPause(id); err != nil {
				return err
			}
			return q.PauseJob(id)
		},
		"remove": func(q *Queue, id string) error { return q.RemoveJob(id) },
	}
	for name, fn := range transitions {
		t.Run(name, func(t *testing.T) {
			q := newTestQueue(t)
			id, _ := q.AddJob("", "heavy-b", nil, "input", nil)
			j, err := q.ClaimJob()
			if err != nil || j == nil {
				t.Fatalf("claim: %v", err)
			}
			if err := fn(q, id); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			assertAllZero(t, q)
		})
	}
}

func TestResourcesResolvedForLegacyRowsOnReload(t *testing.T) {
	withResourceResolver(t)
	q := newTestQueue(t)
	id, _ := q.AddJob("", "heavy-b", nil, "input", nil)

	// Simulate a legacy row: blank out the persisted resources.
	if _, err := q.Db.Exec(`UPDATE jobs SET resources = NULL WHERE id = ?`, id); err != nil {
		t.Fatal(err)
	}

	q2 := NewQueueWithDB(q.Db)
	loaded := q2.GetJob(id)
	if loaded == nil {
		t.Fatal("job not reloaded")
	}
	if len(loaded.Resources) != 2 || loaded.Resources[0] != "bucket-b" || loaded.Resources[1] != "gpu" {
		t.Fatalf("legacy row resources not re-resolved: %v", loaded.Resources)
	}
}

func TestResourcesPersistAcrossReload(t *testing.T) {
	withResourceResolver(t)
	q := newTestQueue(t)
	id, _ := q.AddJob("", "heavy-a", nil, "input", nil)

	q2 := NewQueueWithDB(q.Db)
	loaded := q2.GetJob(id)
	if loaded == nil {
		t.Fatal("job not reloaded")
	}
	if len(loaded.Resources) != 1 || loaded.Resources[0] != "gpu" {
		t.Fatalf("resources lost on reload: %v", loaded.Resources)
	}
}
