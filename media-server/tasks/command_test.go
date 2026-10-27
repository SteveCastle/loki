package tasks

import (
	"database/sql"
	"fmt"
	"os"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	_ "modernc.org/sqlite"
)

// fakeChattyMain is the test binary re-executed as a command that writes a
// burst of output on both pipes and exits at once, so its pipes still hold
// unread lines when the process ends.
func fakeChattyMain() int {
	for i := 0; i < 300; i++ {
		fmt.Fprintf(os.Stdout, "out %d\n", i)
		fmt.Fprintf(os.Stderr, "err %d\n", i)
	}
	fmt.Fprintln(os.Stdout, "LAST LINE")
	return 0
}

// A command that exits while its output is still buffered must complete
// with every line captured, not end in Error with its tail dropped (Wait
// used to run before the readers drained the pipes it closes).
func TestExecuteCommandCapturesAllOutput(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("LOKI_FAKE_CHATTY", "1")

	for run := 0; run < 10; run++ {
		db, err := sql.Open("sqlite", ":memory:")
		if err != nil {
			t.Fatal(err)
		}
		db.SetMaxOpenConns(1)
		q := jobqueue.NewQueueWithDB(db)
		id, err := q.AddJob("", exe, nil, "", nil)
		if err != nil {
			t.Fatal(err)
		}
		j, err := q.ClaimJob()
		if err != nil || j == nil || j.ID != id {
			t.Fatalf("claim: %v", err)
		}
		var mu sync.Mutex
		if err := executeCommand(j, q, &mu); err != nil {
			t.Fatalf("run %d: executeCommand: %v", run, err)
		}
		snap, lines, _ := q.JobSnapshot(id, -1)
		if snap.State != jobqueue.StateCompleted {
			t.Fatalf("run %d: state %v, want Completed", run, snap.State)
		}
		if len(lines) == 0 || !containsLine(lines, "LAST LINE") || len(lines) < 601 {
			t.Fatalf("run %d: captured %d lines (want 601 incl. LAST LINE)", run, len(lines))
		}
		db.Close()
	}
}

func containsLine(lines []string, want string) bool {
	for _, l := range lines {
		if l == want {
			return true
		}
	}
	return false
}
