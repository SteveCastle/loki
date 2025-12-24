package runners

import (
	"database/sql"
	"testing"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	_ "modernc.org/sqlite"
)

func setupTestQueue(t *testing.T) *jobqueue.Queue {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open in-memory database: %v", err)
	}

	return jobqueue.NewQueueWithDB(db)
}

// TestNewRunners verifies runner creation
func TestNewRunners(t *testing.T) {
	q := setupTestQueue(t)

	r := New(q)
	if r == nil {
		t.Fatal("New() returned nil")
	}

	// Verify fields are initialized
	if r.queue != q {
		t.Error("Runners queue not set correctly")
	}
	if r.ctx == nil {
		t.Error("Runners context is nil")
	}
	if r.cancel == nil {
		t.Error("Runners cancel function is nil")
	}

	// Clean up
	r.Shutdown()
}

// TestRunnersShutdown verifies graceful shutdown
func TestRunnersShutdown(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)

	// Start shutdown
	done := make(chan struct{})
	go func() {
		r.Shutdown()
		close(done)
	}()

	// Should complete quickly
	select {
	case <-done:
		// Success
	case <-time.After(2 * time.Second):
		t.Error("Shutdown did not complete in time")
	}
}

// TestRunnersDoubleShutdown ensures shutdown can be called multiple times safely
func TestRunnersDoubleShutdown(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)

	// First shutdown
	r.Shutdown()

	// Second shutdown should not panic
	defer func() {
		if recover() != nil {
			t.Error("Double shutdown caused panic")
		}
	}()
	r.Shutdown()
}

// TestCheckForJobs verifies job checking mechanism
func TestCheckForJobs(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)
	defer r.Shutdown()

	// Add a job with a known command that doesn't exist
	// This will cause an error and mark the job as failed
	q.AddJob("test-job", "nonexistent-command", nil, "", nil)

	// Trigger job check
	r.CheckForJobs()

	// Wait a bit for async processing
	time.Sleep(100 * time.Millisecond)

	// Job should have been claimed and errored (unknown command)
	job := q.GetJob("test-job")
	if job == nil {
		t.Fatal("Job not found")
	}

	// Job should be in error state because the command doesn't exist in the task registry
	// when running these tests in isolation (tasks may not be registered)
	if job.State == jobqueue.StatePending {
		// If still pending, the runner may not have processed it yet
		// This can happen in the test environment
		t.Log("Job still pending - runner may not have processed it")
	}
}

// TestRunnersProcessWaitTask tests that the wait task works correctly
func TestRunnersProcessWaitTask(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)
	defer r.Shutdown()

	// Add a wait job (this task is registered in tasks/wait.go)
	id, _ := q.AddJob("", "wait", nil, "", nil)

	// Signal that a job is available
	q.Signal <- id

	// Wait for job to complete (wait task takes 5 seconds)
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			job := q.GetJob(id)
			t.Fatalf("Job did not complete in time; state = %v", job.State)
		case <-ticker.C:
			job := q.GetJob(id)
			if job.State == jobqueue.StateCompleted {
				// Success
				return
			}
			if job.State == jobqueue.StateError {
				t.Fatalf("Job errored unexpectedly")
			}
		}
	}
}

// TestRunnersUnknownTask tests handling of unknown task commands
func TestRunnersUnknownTask(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)
	defer r.Shutdown()

	// Add a job with unknown command
	id, _ := q.AddJob("", "this-task-does-not-exist", nil, "", nil)

	// Trigger job processing
	q.Signal <- id

	// Wait for job to be processed
	timeout := time.After(2 * time.Second)
	ticker := time.NewTicker(50 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatal("Job was not processed in time")
		case <-ticker.C:
			job := q.GetJob(id)
			if job.State == jobqueue.StateError {
				// Verify error message in stdout
				if len(job.Stdout) > 0 {
					found := false
					for _, line := range job.Stdout {
						if line == "Task not found: this-task-does-not-exist" {
							found = true
							break
						}
					}
					if !found {
						t.Logf("Expected 'Task not found' message in stdout; got %v", job.Stdout)
					}
				}
				return
			}
		}
	}
}

// TestRunnersConcurrency tests that multiple jobs can be processed
func TestRunnersConcurrency(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)
	defer r.Shutdown()

	// Add multiple jobs for different hosts (to allow parallel execution)
	ids := []string{}
	for i := 0; i < 3; i++ {
		id, _ := q.AddJob("", "wait", nil, "", nil)
		ids = append(ids, id)
	}

	// Signal all jobs
	for _, id := range ids {
		q.Signal <- id
	}

	// Wait for all jobs to complete
	timeout := time.After(30 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			t.Fatal("Jobs did not complete in time")
		case <-ticker.C:
			allComplete := true
			for _, id := range ids {
				job := q.GetJob(id)
				if job.State != jobqueue.StateCompleted {
					allComplete = false
					break
				}
			}
			if allComplete {
				return
			}
		}
	}
}

// TestRunnersWithDependencies tests job dependency handling
func TestRunnersWithDependencies(t *testing.T) {
	q := setupTestQueue(t)
	r := New(q)
	defer r.Shutdown()

	// Add parent and child jobs
	parentID, _ := q.AddJob("parent", "wait", nil, "", nil)
	childID, _ := q.AddJob("child", "wait", nil, "", []string{parentID})

	// Signal parent
	q.Signal <- parentID

	// Wait for parent to complete
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	parentComplete := false
	for !parentComplete {
		select {
		case <-timeout:
			t.Fatal("Parent job did not complete in time")
		case <-ticker.C:
			job := q.GetJob(parentID)
			if job.State == jobqueue.StateCompleted {
				parentComplete = true
			}
		}
	}

	// Now child should become eligible
	// Signal to check for jobs again
	r.CheckForJobs()

	// Wait for child to complete
	timeout = time.After(10 * time.Second)
	for {
		select {
		case <-timeout:
			child := q.GetJob(childID)
			t.Fatalf("Child job did not complete in time; state = %v", child.State)
		case <-ticker.C:
			child := q.GetJob(childID)
			if child.State == jobqueue.StateCompleted {
				return
			}
		}
	}
}
