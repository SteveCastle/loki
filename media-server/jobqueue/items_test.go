package jobqueue

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// itemsTestResolver mimics the shape of tasks.ResolveItems: nil when a query
// flag is present (membership unknown until claim), else the input parsed as
// a comma-separated path list.
func itemsTestResolver(command string, arguments []string, input string) []string {
	for _, a := range arguments {
		if strings.HasPrefix(a, "--query") {
			return nil
		}
	}
	var out []string
	for _, p := range strings.Split(input, ",") {
		if p = strings.TrimSpace(p); p != "" && !strings.HasPrefix(p, "--") {
			out = append(out, p)
		}
	}
	return out
}

func newItemsTestQueue(t *testing.T) *Queue {
	t.Helper()
	SetItemsResolver(itemsTestResolver)
	t.Cleanup(func() { SetItemsResolver(nil) })
	return NewQueue()
}

func pathIndexIDs(jobs []Job) []string {
	ids := make([]string, len(jobs))
	for i, j := range jobs {
		ids[i] = j.ID
	}
	return ids
}

// A path-list job is indexed at creation and drops out at every terminal
// transition.
func TestPathIndexLifecycle(t *testing.T) {
	q := newItemsTestQueue(t)
	p := filepath.FromSlash("/media/clip.mp4")

	id, err := q.AddJob("", "transcribe", []string{"--overwrite"}, p, nil)
	if err != nil {
		t.Fatal(err)
	}

	jobs := q.GetJobsForPath(p)
	if len(jobs) != 1 || jobs[0].ID != id {
		t.Fatalf("GetJobsForPath = %v, want [%s]", pathIndexIDs(jobs), id)
	}

	// Claiming (pending → in-progress) keeps the job indexed.
	claimed, err := q.ClaimJob()
	if err != nil || claimed == nil || claimed.ID != id {
		t.Fatalf("ClaimJob = %v, %v", claimed, err)
	}
	if got := q.GetJobsForPath(p); len(got) != 1 {
		t.Fatalf("in-progress job missing from index: %v", pathIndexIDs(got))
	}

	// Pausing keeps it; resuming keeps it.
	if err := q.PauseJob(id); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 1 || got[0].State != StatePaused {
		t.Fatalf("paused job missing from index: %v", pathIndexIDs(got))
	}
	if err := q.ResumeJob(id); err != nil {
		t.Fatal(err)
	}
	if _, err := q.ClaimJob(); err != nil {
		t.Fatal(err)
	}

	// Completion drops it.
	if err := q.CompleteJob(id); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 0 {
		t.Fatalf("completed job still indexed: %v", pathIndexIDs(got))
	}
}

func TestPathIndexCancelAndErrorDrop(t *testing.T) {
	q := newItemsTestQueue(t)
	p := "/media/a.jpg"

	cancelID, _ := q.AddJob("", "hash", nil, p, nil)
	if err := q.CancelJob(cancelID); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 0 {
		t.Fatalf("cancelled job still indexed: %v", pathIndexIDs(got))
	}

	errID, _ := q.AddJob("", "hash", nil, p, nil)
	if c, err := q.ClaimJob(); err != nil || c == nil || c.ID != errID {
		t.Fatalf("ClaimJob = %v, %v", c, err)
	}
	if err := q.ErrorJob(errID); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 0 {
		t.Fatalf("errored job still indexed: %v", pathIndexIDs(got))
	}
}

// Query jobs are not indexed at creation; SetJobItems (called by the task
// when its input resolves) is what makes them path-queryable.
func TestPathIndexQueryJobViaSetJobItems(t *testing.T) {
	q := newItemsTestQueue(t)
	p1, p2 := "/media/a.jpg", "/media/b.jpg"

	id, _ := q.AddJob("", "embed", []string{"--query64", "dGFnOmNhdA=="}, "", nil)
	if got := q.GetJobsForPath(p1); len(got) != 0 {
		t.Fatalf("unresolved query job should not be indexed: %v", pathIndexIDs(got))
	}

	if _, err := q.ClaimJob(); err != nil {
		t.Fatal(err)
	}
	q.SetJobItems(id, []string{p1, p2})

	for _, p := range []string{p1, p2} {
		if got := q.GetJobsForPath(p); len(got) != 1 || got[0].ID != id {
			t.Fatalf("GetJobsForPath(%s) = %v, want [%s]", p, pathIndexIDs(got), id)
		}
	}

	if err := q.CompleteJob(id); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p1); len(got) != 0 {
		t.Fatalf("completed query job still indexed: %v", pathIndexIDs(got))
	}
	// SetJobItems on a terminal job is a no-op.
	q.SetJobItems(id, []string{p1})
	if got := q.GetJobsForPath(p1); len(got) != 0 {
		t.Fatalf("SetJobItems resurrected a terminal job: %v", pathIndexIDs(got))
	}
}

func TestPathIndexRemoveAndClear(t *testing.T) {
	q := newItemsTestQueue(t)
	p := "/media/a.jpg"

	id, _ := q.AddJob("", "hash", nil, p, nil)
	if err := q.RemoveJob(id); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 0 {
		t.Fatalf("removed job still indexed: %v", pathIndexIDs(got))
	}

	id2, _ := q.AddJob("", "hash", nil, p, nil)
	if _, err := q.ClearNonRunningJobs(); err != nil {
		t.Fatal(err)
	}
	if got := q.GetJobsForPath(p); len(got) != 0 {
		t.Fatalf("cleared job %s still indexed: %v", id2, pathIndexIDs(got))
	}
}

// Jobs over the per-job cap are skipped entirely — the index must not
// balloon on a whole-library query job.
func TestPathIndexCap(t *testing.T) {
	q := newItemsTestQueue(t)

	id, _ := q.AddJob("", "embed", []string{"--query64", "eA=="}, "", nil)
	big := make([]string, maxIndexedItemsPerJob+1)
	for i := range big {
		big[i] = fmt.Sprintf("/media/%d.jpg", i)
	}
	if _, err := q.ClaimJob(); err != nil {
		t.Fatal(err)
	}
	q.SetJobItems(id, big)
	if got := q.GetJobsForPath(big[0]); len(got) != 0 {
		t.Fatalf("over-cap job was indexed: %v", pathIndexIDs(got))
	}
}

// On Windows the index is case-insensitive (NTFS paths are); elsewhere it is
// exact.
func TestPathIndexCaseFolding(t *testing.T) {
	q := newItemsTestQueue(t)
	p := filepath.Join("media", "Clip.MP4")

	if _, err := q.AddJob("", "transcribe", nil, p, nil); err != nil {
		t.Fatal(err)
	}
	got := q.GetJobsForPath(strings.ToLower(p))
	if runtime.GOOS == "windows" {
		if len(got) != 1 {
			t.Fatalf("windows lookup should be case-insensitive, got %v", pathIndexIDs(got))
		}
	} else if len(got) != 0 {
		t.Fatalf("non-windows lookup should be case-sensitive, got %v", pathIndexIDs(got))
	}
}
