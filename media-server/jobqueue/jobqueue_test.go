package jobqueue

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func setupTestQueue(t *testing.T) *Queue {
	// Use in-memory database
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open in-memory database: %v", err)
	}

	q := NewQueueWithDB(db)
	return q
}

func TestGetHost(t *testing.T) {
	tests := []struct {
		command  string
		input    string
		expected string
	}{
		{"ingest", "http://example.com/video", "example.com"},
		{"ingest", "https://www.youtube.com/watch?v=123", "youtube.com"},
		{"ingest", "https://youtube.com/watch?v=123", "youtube.com"},
		{"ingest", "/local/path/to/file", "localhost"},
		{"ingest", "invalid-url", "localhost"},
		{"metadata", "/path/to/file", "localhost"},
		{"transcode", "http://example.com/file", "localhost"}, // Not an ingest command
	}

	for _, tt := range tests {
		got := getHost(tt.command, tt.input)
		if got != tt.expected {
			t.Errorf("getHost(%q, %q) = %q; want %q", tt.command, tt.input, got, tt.expected)
		}
	}
}

// TestRegisterOutputFiles checks the bulk variant matches what the one-at-a-time
// call produces (same OutputFiles/SourceFiles shape, persisted) — it exists only
// to avoid the quadratic per-path job rewrite, not to behave differently.
func TestRegisterOutputFiles(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id, _ := q.AddJob("", "remove", nil, "/a.jpg", nil)

	q.RegisterOutputFile(id, "/a.jpg")
	if err := q.RegisterOutputFiles(id, []string{"/b.jpg", "/c.jpg"}); err != nil {
		t.Fatalf("RegisterOutputFiles() error = %v", err)
	}

	job := q.GetJob(id)
	want := []string{"/a.jpg", "/b.jpg", "/c.jpg"}
	if len(job.OutputFiles) != len(want) {
		t.Fatalf("OutputFiles = %v, want %v", job.OutputFiles, want)
	}
	for i, p := range want {
		if job.OutputFiles[i] != p {
			t.Errorf("OutputFiles[%d] = %q, want %q", i, job.OutputFiles[i], p)
		}
	}
	// SourceFiles is read positionally against OutputFiles by downstream tasks,
	// so it must stay the same length.
	if len(job.SourceFiles) != len(want) {
		t.Errorf("SourceFiles has %d entries, want %d (parallel to OutputFiles)", len(job.SourceFiles), len(want))
	}

	// Empty input is a no-op, not an error, and must not touch the job.
	if err := q.RegisterOutputFiles(id, nil); err != nil {
		t.Errorf("RegisterOutputFiles(nil) error = %v", err)
	}
	if len(q.GetJob(id).OutputFiles) != len(want) {
		t.Errorf("RegisterOutputFiles(nil) changed OutputFiles")
	}

	if err := q.RegisterOutputFiles("no-such-job", []string{"/x.jpg"}); err == nil {
		t.Error("RegisterOutputFiles() on unknown job should error")
	}
}

func TestHostAssignment(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id1, _ := q.AddJob("", "ingest", nil, "http://test.com/1", nil)
	job1 := q.GetJob(id1)
	if job1.Host != "test.com" {
		t.Errorf("Job host = %q; want %q", job1.Host, "test.com")
	}

	id2, _ := q.AddJob("", "other", nil, "http://test.com/1", nil)
	job2 := q.GetJob(id2)
	if job2.Host != "localhost" {
		t.Errorf("Job host = %q; want %q", job2.Host, "localhost")
	}
}

func TestConcurrencyLimits(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	// Default limit is 1 for all hosts including localhost

	// Add 2 jobs for Host A
	idA1, _ := q.AddJob("", "ingest", nil, "http://host-a.com/1", nil)
	idA2, _ := q.AddJob("", "ingest", nil, "http://host-a.com/2", nil)

	// Add 1 job for Host B
	idB1, _ := q.AddJob("", "ingest", nil, "http://host-b.com/1", nil)

	// Claim first job (Host A)
	job, err := q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob failed: %v", err)
	}
	if job == nil {
		t.Fatal("Expected job, got nil")
	}
	if job.ID != idA1 {
		t.Errorf("Expected job %s, got %s", idA1, job.ID)
	}

	// Try to claim next job. Should skip A2 (limit reached) and pick B1.
	job, err = q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob failed: %v", err)
	}
	if job == nil {
		t.Fatal("Expected job B1, got nil")
	}
	if job.ID != idB1 {
		t.Errorf("Expected job %s (Host B), got %s (Host %s)", idB1, job.ID, job.Host)
	}

	// Try to claim again. Should be nil because A2 is blocked by A1.
	job, err = q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob failed: %v", err)
	}
	if job != nil {
		t.Errorf("Expected nil (A2 blocked), got job %s", job.ID)
	}

	// Complete A1
	q.CompleteJob(idA1)

	// Now A2 should be claimable
	job, err = q.ClaimJob()
	if err != nil {
		t.Fatalf("ClaimJob failed: %v", err)
	}
	if job == nil {
		t.Fatal("Expected job A2, got nil")
	}
	if job.ID != idA2 {
		t.Errorf("Expected job %s, got %s", idA2, job.ID)
	}
}

func TestLocalhostConcurrency(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	// Localhost limit is also 1 now

	idL1, _ := q.AddJob("", "metadata", nil, "/path/1", nil)
	idL2, _ := q.AddJob("", "metadata", nil, "/path/2", nil)

	// Claim L1
	job, _ := q.ClaimJob()
	if job == nil || job.ID != idL1 {
		t.Fatalf("Expected L1, got %v", job)
	}

	// Claim L2 -> Blocked
	job, _ = q.ClaimJob()
	if job != nil {
		t.Errorf("Expected nil (L2 blocked), got %s", job.ID)
	}

	// Complete L1
	q.CompleteJob(idL1)

	// Claim L2 -> Success
	job, _ = q.ClaimJob()
	if job == nil || job.ID != idL2 {
		t.Errorf("Expected L2, got %v", job)
	}
}

func TestErrorReleasesLock(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id1, _ := q.AddJob("", "ingest", nil, "http://host.com/1", nil)
	id2, _ := q.AddJob("", "ingest", nil, "http://host.com/2", nil)

	job, _ := q.ClaimJob() // Claim 1
	if job.ID != id1 {
		t.Fatal("Expected 1")
	}

	// Error 1
	q.ErrorJob(id1)

	// Claim 2 should succeed
	job, _ = q.ClaimJob()
	if job == nil || job.ID != id2 {
		t.Errorf("Expected 2, got %v", job)
	}
}

func TestCancelReleasesLock(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id1, _ := q.AddJob("", "ingest", nil, "http://host.com/1", nil)
	id2, _ := q.AddJob("", "ingest", nil, "http://host.com/2", nil)

	job, _ := q.ClaimJob() // Claim 1
	if job.ID != id1 {
		t.Fatal("Expected 1")
	}

	// Cancel 1 (must be in progress to release lock from running counts)
	q.CancelJob(id1)

	// Claim 2 should succeed
	job, _ = q.ClaimJob()
	if job == nil || job.ID != id2 {
		t.Errorf("Expected 2, got %v", job)
	}
}

func TestRemoveReleasesLock(t *testing.T) {
	q := setupTestQueue(t)
	defer q.Db.Close()

	id1, _ := q.AddJob("", "ingest", nil, "http://host.com/1", nil)
	id2, _ := q.AddJob("", "ingest", nil, "http://host.com/2", nil)

	job, _ := q.ClaimJob() // Claim 1
	if job.ID != id1 {
		t.Fatal("Expected 1")
	}

	// Remove 1 while running
	q.RemoveJob(id1)

	// Claim 2 should succeed
	job, _ = q.ClaimJob()
	if job == nil || job.ID != id2 {
		t.Errorf("Expected 2, got %v", job)
	}
}
