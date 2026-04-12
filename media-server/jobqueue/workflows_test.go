package jobqueue

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func newTestQueue(t *testing.T) *Queue {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("failed to open in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return NewQueueWithDB(db)
}

func TestCreateAndGetWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{
		{ID: "a", Command: "cmd1", Input: "hello"},
		{ID: "b", Command: "cmd2", Dependencies: []string{"a"}},
	}

	saved, err := q.CreateWorkflow("my-workflow", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}
	if saved.Name != "my-workflow" {
		t.Errorf("expected name my-workflow, got %s", saved.Name)
	}

	got, err := q.GetWorkflow(saved.ID)
	if err != nil {
		t.Fatalf("GetWorkflow: %v", err)
	}
	if len(got.DAG) != 2 {
		t.Fatalf("expected 2 tasks, got %d", len(got.DAG))
	}
	if got.DAG[0].Command != "cmd1" {
		t.Errorf("expected cmd1, got %s", got.DAG[0].Command)
	}
	if len(got.DAG[1].Dependencies) != 1 || got.DAG[1].Dependencies[0] != "a" {
		t.Errorf("expected dependency on a, got %v", got.DAG[1].Dependencies)
	}
}

func TestListWorkflows(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "x", Command: "run"}}
	_, err := q.CreateWorkflow("zebra", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}
	_, err = q.CreateWorkflow("alpha", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	list, err := q.ListWorkflows()
	if err != nil {
		t.Fatalf("ListWorkflows: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("expected 2 workflows, got %d", len(list))
	}
	if list[0].Name != "alpha" || list[1].Name != "zebra" {
		t.Errorf("expected alphabetical order [alpha, zebra], got [%s, %s]", list[0].Name, list[1].Name)
	}
}

func TestUpdateWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "old"}}
	saved, err := q.CreateWorkflow("original", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	newDAG := []WorkflowTask{{ID: "b", Command: "new"}}
	err = q.UpdateWorkflow(saved.ID, "updated", newDAG)
	if err != nil {
		t.Fatalf("UpdateWorkflow: %v", err)
	}

	got, err := q.GetWorkflow(saved.ID)
	if err != nil {
		t.Fatalf("GetWorkflow: %v", err)
	}
	if got.Name != "updated" {
		t.Errorf("expected name updated, got %s", got.Name)
	}
	if got.DAG[0].Command != "new" {
		t.Errorf("expected command new, got %s", got.DAG[0].Command)
	}
}

func TestDeleteWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "cmd"}}
	saved, err := q.CreateWorkflow("to-delete", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	err = q.DeleteWorkflow(saved.ID)
	if err != nil {
		t.Fatalf("DeleteWorkflow: %v", err)
	}

	_, err = q.GetWorkflow(saved.ID)
	if err == nil {
		t.Fatal("expected error getting deleted workflow, got nil")
	}
}

func TestDeleteWorkflowNotFound(t *testing.T) {
	q := newTestQueue(t)

	err := q.DeleteWorkflow("nonexistent-id")
	if err == nil {
		t.Fatal("expected error deleting nonexistent workflow, got nil")
	}
}

func TestCreateWorkflowDuplicateName(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{{ID: "a", Command: "cmd"}}
	_, err := q.CreateWorkflow("dup", dag)
	if err != nil {
		t.Fatalf("first CreateWorkflow: %v", err)
	}

	_, err = q.CreateWorkflow("dup", dag)
	if err == nil {
		t.Fatal("expected error creating workflow with duplicate name, got nil")
	}
}

func TestValidateDAG(t *testing.T) {
	tests := []struct {
		name    string
		dag     []WorkflowTask
		wantErr bool
	}{
		{
			name:    "empty dag",
			dag:     []WorkflowTask{},
			wantErr: true,
		},
		{
			name:    "missing id",
			dag:     []WorkflowTask{{Command: "cmd"}},
			wantErr: true,
		},
		{
			name:    "missing command",
			dag:     []WorkflowTask{{ID: "a"}},
			wantErr: true,
		},
		{
			name: "duplicate id",
			dag: []WorkflowTask{
				{ID: "a", Command: "cmd1"},
				{ID: "a", Command: "cmd2"},
			},
			wantErr: true,
		},
		{
			name: "bad dep ref",
			dag: []WorkflowTask{
				{ID: "a", Command: "cmd", Dependencies: []string{"nonexistent"}},
			},
			wantErr: true,
		},
		{
			name: "valid dag",
			dag: []WorkflowTask{
				{ID: "a", Command: "cmd1"},
				{ID: "b", Command: "cmd2", Dependencies: []string{"a"}},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateDAG(tt.dag)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateDAG() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestRunWorkflow(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{
		{ID: "root", Command: "ingest", Input: "existing"},
		{ID: "child", Command: "process", Dependencies: []string{"root"}},
	}

	saved, err := q.CreateWorkflow("run-test", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	liveIDs, err := q.RunWorkflow(saved.ID, "runtime-input")
	if err != nil {
		t.Fatalf("RunWorkflow: %v", err)
	}

	if len(liveIDs) != 2 {
		t.Fatalf("expected 2 live IDs, got %d", len(liveIDs))
	}

	// Live IDs should differ from template IDs.
	if liveIDs[0] == "root" || liveIDs[1] == "child" {
		t.Error("live IDs should differ from template IDs")
	}

	// Root job should have input injected (appended with space).
	rootJob := q.GetJob(liveIDs[0])
	if rootJob == nil {
		t.Fatal("root job not found")
	}
	if rootJob.Input != "existing runtime-input" {
		t.Errorf("expected root input 'existing runtime-input', got %q", rootJob.Input)
	}

	// Child job should depend on root using live UUID.
	childJob := q.GetJob(liveIDs[1])
	if childJob == nil {
		t.Fatal("child job not found")
	}
	if len(childJob.Dependencies) != 1 || childJob.Dependencies[0] != liveIDs[0] {
		t.Errorf("expected child dependency on %s, got %v", liveIDs[0], childJob.Dependencies)
	}
}
