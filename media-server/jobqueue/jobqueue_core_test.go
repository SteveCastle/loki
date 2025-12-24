package jobqueue

import (
	"database/sql"
	"encoding/json"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// ============================================================================
// JobState Tests
// ============================================================================

func TestJobStateString(t *testing.T) {
	tests := []struct {
		state    JobState
		expected string
	}{
		{StatePending, "Pending"},
		{StateInProgress, "InProgress"},
		{StateCompleted, "Completed"},
		{StateCancelled, "Cancelled"},
		{StateError, "Error"},
		{JobState(99), "Unknown"},
	}

	for _, tt := range tests {
		got := tt.state.String()
		if got != tt.expected {
			t.Errorf("JobState(%d).String() = %q; want %q", tt.state, got, tt.expected)
		}
	}
}

func TestJobStateMarshalJSON(t *testing.T) {
	tests := []struct {
		state    JobState
		expected string
	}{
		{StatePending, `"pending"`},
		{StateInProgress, `"in_progress"`},
		{StateCompleted, `"completed"`},
		{StateCancelled, `"cancelled"`},
		{StateError, `"error"`},
		{JobState(99), `"unknown"`},
	}

	for _, tt := range tests {
		data, err := json.Marshal(tt.state)
		if err != nil {
			t.Errorf("JobState(%d).MarshalJSON() error = %v", tt.state, err)
			continue
		}
		if string(data) != tt.expected {
			t.Errorf("JobState(%d).MarshalJSON() = %s; want %s", tt.state, data, tt.expected)
		}
	}
}

func TestJobStateUnmarshalJSON(t *testing.T) {
	tests := []struct {
		json     string
		expected JobState
	}{
		{`"pending"`, StatePending},
		{`"in_progress"`, StateInProgress},
		{`"completed"`, StateCompleted},
		{`"cancelled"`, StateCancelled},
		{`"error"`, StateError},
		{`"unknown"`, StatePending}, // defaults to pending
		{`"invalid"`, StatePending}, // defaults to pending
	}

	for _, tt := range tests {
		var state JobState
		if err := json.Unmarshal([]byte(tt.json), &state); err != nil {
			t.Errorf("UnmarshalJSON(%s) error = %v", tt.json, err)
			continue
		}
		if state != tt.expected {
			t.Errorf("UnmarshalJSON(%s) = %d; want %d", tt.json, state, tt.expected)
		}
	}
}

// ============================================================================
// Queue Core Tests
// ============================================================================

func TestNewQueue(t *testing.T) {
	q := NewQueue()
	if q == nil {
		t.Fatal("NewQueue() returned nil")
	}
	if q.Jobs == nil {
		t.Error("NewQueue() Jobs map is nil")
	}
	if q.Signal == nil {
		t.Error("NewQueue() Signal channel is nil")
	}
	if q.HostLimits == nil {
		t.Error("NewQueue() HostLimits map is nil")
	}
	if q.RunningCounts == nil {
		t.Error("NewQueue() RunningCounts map is nil")
	}
}

func TestNewQueueWithDB(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}
	defer db.Close()

	q := NewQueueWithDB(db)
	if q == nil {
		t.Fatal("NewQueueWithDB() returned nil")
	}
	if q.Db != db {
		t.Error("NewQueueWithDB() did not set Db correctly")
	}

	// Verify jobs table was created
	var tableExists int
	err = db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='jobs'`).Scan(&tableExists)
	if err != nil {
		t.Errorf("Failed to check jobs table existence: %v", err)
	}
	if tableExists != 1 {
		t.Error("Jobs table was not created")
	}
}

func TestAddJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Test adding job with generated ID
	id, err := q.AddJob("", "test-command", []string{"arg1", "arg2"}, "test-input", nil)
	if err != nil {
		t.Fatalf("AddJob() error = %v", err)
	}
	if id == "" {
		t.Error("AddJob() returned empty ID")
	}

	job := q.GetJob(id)
	if job == nil {
		t.Fatal("GetJob() returned nil for added job")
	}
	if job.Command != "test-command" {
		t.Errorf("Job.Command = %q; want %q", job.Command, "test-command")
	}
	if len(job.Arguments) != 2 || job.Arguments[0] != "arg1" || job.Arguments[1] != "arg2" {
		t.Errorf("Job.Arguments = %v; want [arg1, arg2]", job.Arguments)
	}
	if job.Input != "test-input" {
		t.Errorf("Job.Input = %q; want %q", job.Input, "test-input")
	}
	if job.State != StatePending {
		t.Errorf("Job.State = %v; want StatePending", job.State)
	}
	if job.Ctx == nil {
		t.Error("Job.Ctx is nil")
	}
	if job.Cancel == nil {
		t.Error("Job.Cancel is nil")
	}
}

func TestAddJobWithCustomID(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	customID := "custom-job-id-123"
	id, err := q.AddJob(customID, "test-command", nil, "", nil)
	if err != nil {
		t.Fatalf("AddJob() error = %v", err)
	}
	if id != customID {
		t.Errorf("AddJob() returned ID %q; want %q", id, customID)
	}
}

func TestAddJobDuplicateID(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	customID := "duplicate-id"
	_, err := q.AddJob(customID, "test-command", nil, "", nil)
	if err != nil {
		t.Fatalf("First AddJob() error = %v", err)
	}

	_, err = q.AddJob(customID, "test-command", nil, "", nil)
	if err == nil {
		t.Error("Second AddJob() with same ID should return error")
	}
}

func TestAddJobWithDependencies(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add parent job
	parentID, _ := q.AddJob("parent", "test-command", nil, "parent-input", nil)

	// Add child job with dependency
	childID, err := q.AddJob("child", "test-command", nil, "child-input", []string{parentID})
	if err != nil {
		t.Fatalf("AddJob() with dependency error = %v", err)
	}

	childJob := q.GetJob(childID)
	if len(childJob.Dependencies) != 1 || childJob.Dependencies[0] != parentID {
		t.Errorf("Job.Dependencies = %v; want [%s]", childJob.Dependencies, parentID)
	}

	// Child should not be claimable while parent is pending
	claimedJob, _ := q.ClaimJob()
	if claimedJob == nil || claimedJob.ID != parentID {
		t.Errorf("ClaimJob() should claim parent first; got %v", claimedJob)
	}

	// Child still not claimable while parent is in progress
	childClaimAttempt, _ := q.ClaimJob()
	if childClaimAttempt != nil {
		t.Error("ClaimJob() should not claim child while parent is in progress")
	}

	// Complete parent
	q.CompleteJob(parentID)

	// Now child should be claimable
	childClaimed, _ := q.ClaimJob()
	if childClaimed == nil || childClaimed.ID != childID {
		t.Errorf("ClaimJob() should claim child after parent completes; got %v", childClaimed)
	}
}

// ============================================================================
// Workflow Tests
// ============================================================================

func TestAddWorkflow(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	workflow := Workflow{
		Tasks: []WorkflowTask{
			{ID: "task-1", Command: "command-1", Input: "input-1"},
			{ID: "task-2", Command: "command-2", Input: "input-2", Dependencies: []string{"task-1"}},
			{ID: "task-3", Command: "command-3", Input: "input-3", Dependencies: []string{"task-1", "task-2"}},
		},
	}

	ids, err := q.AddWorkflow(workflow)
	if err != nil {
		t.Fatalf("AddWorkflow() error = %v", err)
	}

	if len(ids) != 3 {
		t.Errorf("AddWorkflow() returned %d IDs; want 3", len(ids))
	}

	// Verify jobs were created
	for _, id := range ids {
		job := q.GetJob(id)
		if job == nil {
			t.Errorf("GetJob(%q) returned nil", id)
		}
	}

	// Verify dependencies
	task2 := q.GetJob("task-2")
	if len(task2.Dependencies) != 1 || task2.Dependencies[0] != "task-1" {
		t.Errorf("task-2 dependencies = %v; want [task-1]", task2.Dependencies)
	}

	task3 := q.GetJob("task-3")
	if len(task3.Dependencies) != 2 {
		t.Errorf("task-3 dependencies = %v; want [task-1, task-2]", task3.Dependencies)
	}
}

func TestAddWorkflowEmpty(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	workflow := Workflow{Tasks: []WorkflowTask{}}
	ids, err := q.AddWorkflow(workflow)
	if err != nil {
		t.Fatalf("AddWorkflow() error = %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("AddWorkflow() returned %d IDs; want 0", len(ids))
	}
}

// ============================================================================
// Job Operations Tests
// ============================================================================

func TestCopyJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add and complete original job
	originalID, _ := q.AddJob("", "test-command", []string{"arg1"}, "original-input", nil)
	originalJob := q.GetJob(originalID)
	originalJob.Stdout = []string{"output line 1", "output line 2"}

	// Copy the job
	copyID, err := q.CopyJob(originalID)
	if err != nil {
		t.Fatalf("CopyJob() error = %v", err)
	}
	if copyID == originalID {
		t.Error("CopyJob() returned same ID as original")
	}

	copyJob := q.GetJob(copyID)
	if copyJob == nil {
		t.Fatal("GetJob() returned nil for copied job")
	}

	// Verify copy properties
	if copyJob.Command != originalJob.Command {
		t.Errorf("Copy.Command = %q; want %q", copyJob.Command, originalJob.Command)
	}
	if copyJob.OriginalInput != originalJob.OriginalInput {
		t.Errorf("Copy.OriginalInput = %q; want %q", copyJob.OriginalInput, originalJob.OriginalInput)
	}
	if len(copyJob.Stdout) != 0 {
		t.Errorf("Copy.Stdout should be empty; got %v", copyJob.Stdout)
	}
	if copyJob.State != StatePending {
		t.Errorf("Copy.State = %v; want StatePending", copyJob.State)
	}
	if copyJob.CreatedAt.Before(originalJob.CreatedAt) {
		t.Error("Copy.CreatedAt should not be before original")
	}
}

func TestCopyJobNotFound(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	_, err := q.CopyJob("nonexistent")
	if err == nil {
		t.Error("CopyJob() should return error for nonexistent job")
	}
}

func TestRemoveJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test-command", nil, "", nil)

	err := q.RemoveJob(id)
	if err != nil {
		t.Fatalf("RemoveJob() error = %v", err)
	}

	job := q.GetJob(id)
	if job != nil {
		t.Error("GetJob() should return nil after RemoveJob()")
	}

	// Verify removed from database
	var count int
	db.QueryRow("SELECT COUNT(*) FROM jobs WHERE id = ?", id).Scan(&count)
	if count != 0 {
		t.Error("Job should be removed from database")
	}
}

func TestRemoveJobNotFound(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	err := q.RemoveJob("nonexistent")
	if err == nil {
		t.Error("RemoveJob() should return error for nonexistent job")
	}
}

func TestClearNonRunningJobs(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add jobs in various states
	pendingID, _ := q.AddJob("", "test", nil, "", nil)
	runningID, _ := q.AddJob("", "test", nil, "", nil)
	completedID, _ := q.AddJob("", "test", nil, "", nil)
	errorID, _ := q.AddJob("", "test", nil, "", nil)

	// Set states
	q.ClaimJob() // Claims first pending job (pendingID or runningID depending on order)

	// Re-setup to ensure specific states
	q2 := NewQueueWithDB(db)
	q2.Jobs[pendingID].State = StatePending
	q2.Jobs[runningID].State = StateInProgress
	q2.Jobs[completedID].State = StateCompleted
	q2.Jobs[errorID].State = StateError

	clearedCount, err := q2.ClearNonRunningJobs()
	if err != nil {
		t.Fatalf("ClearNonRunningJobs() error = %v", err)
	}

	// Should clear 3 jobs (pending, completed, error) but not running
	if clearedCount != 3 {
		t.Errorf("ClearNonRunningJobs() cleared %d; want 3", clearedCount)
	}

	// Running job should still exist
	if q2.GetJob(runningID) == nil {
		t.Error("Running job should not be cleared")
	}
}

func TestGetJobs(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add multiple jobs with slight delay to ensure different CreatedAt times
	q.AddJob("job-1", "command", nil, "", nil)
	time.Sleep(1 * time.Millisecond)
	q.AddJob("job-2", "command", nil, "", nil)
	time.Sleep(1 * time.Millisecond)
	q.AddJob("job-3", "command", nil, "", nil)

	jobs := q.GetJobs()
	if len(jobs) != 3 {
		t.Fatalf("GetJobs() returned %d jobs; want 3", len(jobs))
	}

	// Jobs should be returned in reverse order (newest first)
	if jobs[0].ID != "job-3" {
		t.Errorf("First job should be job-3 (newest); got %s", jobs[0].ID)
	}
	if jobs[2].ID != "job-1" {
		t.Errorf("Last job should be job-1 (oldest); got %s", jobs[2].ID)
	}
}

// ============================================================================
// Job State Transition Tests
// ============================================================================

func TestClaimJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "input", nil)

	job, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob() error = %v", err)
	}
	if job == nil {
		t.Fatal("ClaimJob() returned nil")
	}
	if job.ID != id {
		t.Errorf("ClaimJob() returned job %s; want %s", job.ID, id)
	}
	if job.State != StateInProgress {
		t.Errorf("Job state = %v; want StateInProgress", job.State)
	}
	if job.ClaimedAt.IsZero() {
		t.Error("Job.ClaimedAt should be set after claim")
	}
}

func TestClaimJobNoPending(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	job, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob() error = %v", err)
	}
	if job != nil {
		t.Error("ClaimJob() should return nil when no pending jobs")
	}
}

func TestCompleteJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)
	q.ClaimJob()

	err := q.CompleteJob(id)
	if err != nil {
		t.Fatalf("CompleteJob() error = %v", err)
	}

	job := q.GetJob(id)
	if job.State != StateCompleted {
		t.Errorf("Job state = %v; want StateCompleted", job.State)
	}
	if job.CompletedAt.IsZero() {
		t.Error("Job.CompletedAt should be set after completion")
	}
}

func TestCompleteJobNotInProgress(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)

	err := q.CompleteJob(id)
	if err == nil {
		t.Error("CompleteJob() should return error for pending job")
	}
}

func TestErrorJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)
	q.ClaimJob()

	err := q.ErrorJob(id)
	if err != nil {
		t.Fatalf("ErrorJob() error = %v", err)
	}

	job := q.GetJob(id)
	if job.State != StateError {
		t.Errorf("Job state = %v; want StateError", job.State)
	}
	if job.ErroredAt.IsZero() {
		t.Error("Job.ErroredAt should be set after error")
	}
}

func TestCancelJob(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)

	err := q.CancelJob(id)
	if err != nil {
		t.Fatalf("CancelJob() error = %v", err)
	}

	job := q.GetJob(id)
	if job.State != StateCancelled {
		t.Errorf("Job state = %v; want StateCancelled", job.State)
	}
}

func TestCancelJobInProgress(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)
	q.ClaimJob()

	err := q.CancelJob(id)
	if err != nil {
		t.Fatalf("CancelJob() error = %v", err)
	}

	job := q.GetJob(id)
	if job.State != StateCancelled {
		t.Errorf("Job state = %v; want StateCancelled", job.State)
	}
}

func TestPushJobStdout(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)

	q.PushJobStdout(id, "line 1")
	q.PushJobStdout(id, "line 2")

	job := q.GetJob(id)
	if len(job.Stdout) != 2 {
		t.Errorf("Job.Stdout length = %d; want 2", len(job.Stdout))
	}
	if job.Stdout[0] != "line 1" || job.Stdout[1] != "line 2" {
		t.Errorf("Job.Stdout = %v; want [line 1, line 2]", job.Stdout)
	}
}

// ============================================================================
// Database Persistence Tests
// ============================================================================

func TestDatabasePersistence(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()

	// Create queue and add jobs
	q1 := NewQueueWithDB(db)
	id1, _ := q1.AddJob("persist-1", "command-1", []string{"arg1"}, "input-1", nil)
	id2, _ := q1.AddJob("persist-2", "command-2", nil, "input-2", []string{id1})

	// Add some stdout
	q1.PushJobStdout(id1, "stdout line")

	// Claim and complete one job
	q1.ClaimJob()
	q1.CompleteJob(id1)

	// Create new queue from same database - simulates restart
	q2 := NewQueueWithDB(db)

	// Verify jobs were loaded
	job1 := q2.GetJob(id1)
	job2 := q2.GetJob(id2)

	if job1 == nil || job2 == nil {
		t.Fatal("Jobs were not persisted/loaded from database")
	}

	if job1.Command != "command-1" {
		t.Errorf("Loaded job1.Command = %q; want %q", job1.Command, "command-1")
	}
	if job1.State != StateCompleted {
		t.Errorf("Loaded job1.State = %v; want StateCompleted", job1.State)
	}
	if len(job1.Stdout) != 1 || job1.Stdout[0] != "stdout line" {
		t.Errorf("Loaded job1.Stdout = %v; want [stdout line]", job1.Stdout)
	}

	if len(job2.Dependencies) != 1 || job2.Dependencies[0] != id1 {
		t.Errorf("Loaded job2.Dependencies = %v; want [%s]", job2.Dependencies, id1)
	}
}

func TestDatabasePersistenceInProgressReset(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()

	// Create queue and claim a job (leave it in progress)
	q1 := NewQueueWithDB(db)
	id, _ := q1.AddJob("", "test", nil, "", nil)
	q1.ClaimJob()

	// Verify it's in progress
	job := q1.GetJob(id)
	if job.State != StateInProgress {
		t.Fatalf("Job should be in progress; got %v", job.State)
	}

	// Create new queue from same database - simulates crash recovery
	q2 := NewQueueWithDB(db)

	// Job should be reset to pending
	loadedJob := q2.GetJob(id)
	if loadedJob.State != StatePending {
		t.Errorf("In-progress job should be reset to pending on reload; got %v", loadedJob.State)
	}
}

func TestSetHostLimit(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	q.SetHostLimit("example.com", 5)

	// Verify limit is set
	q.mu.Lock()
	limit := q.getHostLimitLocked("example.com")
	q.mu.Unlock()

	if limit != 5 {
		t.Errorf("Host limit = %d; want 5", limit)
	}
}

func TestSaveAllJobsToDB(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add jobs
	q.AddJob("save-1", "command", nil, "", nil)
	q.AddJob("save-2", "command", nil, "", nil)

	// Manually clear database to test save
	db.Exec("DELETE FROM jobs")

	// Save all jobs
	err := q.SaveAllJobsToDB()
	if err != nil {
		t.Fatalf("SaveAllJobsToDB() error = %v", err)
	}

	// Verify jobs are in database
	var count int
	db.QueryRow("SELECT COUNT(*) FROM jobs").Scan(&count)
	if count != 2 {
		t.Errorf("Database has %d jobs; want 2", count)
	}
}
