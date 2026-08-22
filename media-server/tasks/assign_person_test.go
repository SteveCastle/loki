package tasks

import (
	"database/sql"
	"path/filepath"
	"strconv"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// newAssignPersonJob adds and claims an assign-person job on a fresh
// single-connection in-memory queue DB carrying the full media schema.
func newAssignPersonJob(t *testing.T, args []string, input string) (*jobqueue.Queue, *jobqueue.Job) {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })

	q := jobqueue.NewQueueWithDB(db)
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	id, err := q.AddJob("", "assign-person", args, input, nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	return q, j
}

func TestAssignPersonTaskBulk(t *testing.T) {
	dir := t.TempDir()
	// Pre-scanned targets, so no ONNX subprocess is needed. Absolute paths
	// because resolveJobItems absolutizes explicit path lists.
	similar := filepath.Join(dir, "similar.jpg") // has a face like Alice's
	crowd := filepath.Join(dir, "crowd.jpg")     // big stranger + small Alice-alike
	faceless := filepath.Join(dir, "faceless.jpg")

	q, j := newAssignPersonJob(t, nil, similar+"\n"+crowd+"\n"+faceless)
	model := ActiveFaceModel().ID

	alice, err := media.CreatePerson(q.Db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	seedIDs, err := media.ReplaceFaces(q.Db, filepath.Join(dir, "seed.jpg"), model, []media.NewFace{
		{X: 0.1, Y: 0.1, W: 0.3, H: 0.3, Score: 0.9, Vec: []float32{1, 0}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := media.AssignFace(q.Db, seedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}

	similarIDs, err := media.ReplaceFaces(q.Db, similar, model, []media.NewFace{
		{X: 0.2, Y: 0.2, W: 0.4, H: 0.4, Score: 0.9, Vec: []float32{0.99, 0.1}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	crowdIDs, err := media.ReplaceFaces(q.Db, crowd, model, []media.NewFace{
		{X: 0.0, Y: 0.0, W: 0.8, H: 0.8, Score: 0.95, Vec: []float32{0, 1}},
		{X: 0.7, Y: 0.7, W: 0.1, H: 0.1, Score: 0.85, Vec: []float32{0.98, 0.2}},
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := media.ReplaceFaces(q.Db, faceless, model, nil, 1); err != nil {
		t.Fatal(err)
	}

	// person-id arrives as an argument flag; the path list is the job input.
	j.Arguments = []string{"--person-id=" + strconv.FormatInt(alice, 10)}

	var mu sync.Mutex
	if err := assignPersonTask(j, q, &mu); err != nil {
		t.Fatalf("assign-person: %v", err)
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Errorf("job state = %v; want Completed", got)
	}

	f, _, err := media.GetFaceByID(q.Db, similarIDs[0])
	if err != nil {
		t.Fatal(err)
	}
	if f.PersonID != alice || f.AssignedBy != "user" {
		t.Errorf("similar face = %+v, want assigned to Alice by user", f)
	}
	// In the crowd shot the Alice-similar face wins despite being smaller.
	small, _, err := media.GetFaceByID(q.Db, crowdIDs[1])
	if err != nil {
		t.Fatal(err)
	}
	if small.PersonID != alice {
		t.Errorf("crowd similar face = %+v, want assigned to Alice", small)
	}
	big, _, err := media.GetFaceByID(q.Db, crowdIDs[0])
	if err != nil {
		t.Fatal(err)
	}
	if big.PersonID != 0 {
		t.Errorf("dissimilar face was assigned too: %+v", big)
	}
}

func TestAssignPersonTaskRejectsMissingPerson(t *testing.T) {
	q, j := newAssignPersonJob(t, []string{"--person-id=12345"}, "x.jpg")
	var mu sync.Mutex
	if err := assignPersonTask(j, q, &mu); err == nil {
		t.Fatal("expected an error for a missing person")
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateError {
		t.Errorf("job state = %v; want Error", got)
	}
}
