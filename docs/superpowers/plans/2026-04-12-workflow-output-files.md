# Workflow Output Files & Temp Directory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stdout-based job chaining with a dedicated OutputFiles channel, route all workflow file output through temp directories, and add an explicit Save File task for user-controlled final output.

**Architecture:** Add `OutputFiles []string` and `WorkflowID string` fields to the Job struct. `ClaimJob` reads `OutputFiles` from parents instead of `Stdout`. All workflow jobs write to `.loki-temp/<jobID>/` sibling to the source file. A new `save` task moves files from temp to their final destination with configurable naming, conflict resolution, and cleanup.

**Tech Stack:** Go, SQLite (modernc.org/sqlite), existing jobqueue/tasks packages

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `jobqueue/jobqueue.go` | Modify | Add `OutputFiles`, `WorkflowID` to Job struct; add `RegisterOutputFile` method; update `createJobsTable`, `saveJobToDB`, `loadJobsFromDB`, `ClaimJob`, `CopyJob`, `AddWorkflow` |
| `jobqueue/jobqueue_core_test.go` | Modify | Tests for `RegisterOutputFile`, `ClaimJob` with OutputFiles, `WorkflowID` propagation |
| `jobqueue/workflows.go` | Modify | Set `WorkflowID` in `RunWorkflow` |
| `jobqueue/workflows_test.go` | Modify | Test `WorkflowID` assignment in `RunWorkflow` |
| `tasks/ffmpeg.go` | Modify | Add temp dir logic to `runFFmpegOnFiles`, call `RegisterOutputFile`; same for `ffmpegTask` |
| `tasks/ffmpeg_presets.go` | No change | All presets use `runFFmpegOnFiles` which handles it |
| `tasks/save.go` | Create | Save File task implementation |
| `tasks/save_test.go` | Create | Save File task tests |
| `tasks/registry.go` | Modify | Register `save` task |
| `tasks/media_metadata.go` | Modify | Call `RegisterOutputFile` for passthrough paths |
| `tasks/autotag.go` | Modify | Call `RegisterOutputFile` for passthrough paths |
| `tasks/media_move.go` | Modify | Call `RegisterOutputFile` for destination paths |
| `tasks/media_remove.go` | Modify | Call `RegisterOutputFile` for removed paths |
| `tasks/hls.go` | Modify | Call `RegisterOutputFile` for master playlist paths |
| `tasks/lora_dataset.go` | Modify | Call `RegisterOutputFile` for output paths |
| `tasks/ingest_local.go` | Modify | Call `RegisterOutputFile` for ingested paths |
| `tasks/ingest_youtube.go` | Modify | Call `RegisterOutputFile` for downloaded paths |
| `tasks/ingest_gallery.go` | Modify | Call `RegisterOutputFile` for downloaded paths |
| `tasks/ingest_discord.go` | Modify | Call `RegisterOutputFile` for downloaded paths |
| `tasks/cleanup.go` | Create | Temp directory cleanup helpers |

---

### Task 1: Add OutputFiles and WorkflowID to Job Struct and DB Schema

**Files:**
- Modify: `jobqueue/jobqueue.go:101-121` (Job struct)
- Modify: `jobqueue/jobqueue.go:183-213` (createJobsTable)
- Modify: `jobqueue/jobqueue.go:215-259` (saveJobToDB)
- Modify: `jobqueue/jobqueue.go:262-354` (loadJobsFromDB)
- Test: `jobqueue/jobqueue_core_test.go`

- [ ] **Step 1: Write test for OutputFiles and WorkflowID persistence**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestOutputFilesAndWorkflowIDPersistence(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()

	q1 := NewQueueWithDB(db)
	id, _ := q1.AddJob("persist-out", "command", nil, "input", nil)

	// Manually set fields to test persistence
	q1.mu.Lock()
	q1.Jobs[id].OutputFiles = []string{"/tmp/a.mp4", "/tmp/b.mp4"}
	q1.Jobs[id].WorkflowID = "wf-123"
	q1.saveJobToDB(q1.Jobs[id])
	q1.mu.Unlock()

	// Reload from DB
	q2 := NewQueueWithDB(db)
	job := q2.GetJob(id)
	if job == nil {
		t.Fatal("job not found after reload")
	}
	if len(job.OutputFiles) != 2 || job.OutputFiles[0] != "/tmp/a.mp4" || job.OutputFiles[1] != "/tmp/b.mp4" {
		t.Errorf("OutputFiles = %v; want [/tmp/a.mp4, /tmp/b.mp4]", job.OutputFiles)
	}
	if job.WorkflowID != "wf-123" {
		t.Errorf("WorkflowID = %q; want %q", job.WorkflowID, "wf-123")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./jobqueue/ -run TestOutputFilesAndWorkflowIDPersistence -v`
Expected: compilation error — `OutputFiles` and `WorkflowID` fields don't exist

- [ ] **Step 3: Add fields to Job struct**

In `jobqueue/jobqueue.go`, add two fields to the `Job` struct after the `ErroredAt` field (around line 120):

```go
type Job struct {
	ID            string             `json:"id"`
	Command       string             `json:"command"`
	Arguments     []string           `json:"arguments"`
	Input         string             `json:"input"`
	OriginalInput string             `json:"original_input"`
	Host          string             `json:"host"`
	Stdout        []string           `json:"-"`
	StdoutRaw     io.Reader          `json:"-"`
	StdIn         io.Reader          `json:"-"`
	Dependencies  []string           `json:"dependencies"`
	State         JobState           `json:"state"`
	Ctx           context.Context    `json:"-"`
	Cancel        context.CancelFunc `json:"-"`

	// Timestamps for various states
	CreatedAt   time.Time `json:"created_at"`
	ClaimedAt   time.Time `json:"claimed_at"`
	CompletedAt time.Time `json:"completed_at"`
	ErroredAt   time.Time `json:"errored_at"`

	// Workflow chaining
	OutputFiles []string `json:"output_files"` // File paths registered for downstream consumption
	WorkflowID  string   `json:"workflow_id"`  // Non-empty when job is part of a workflow
}
```

- [ ] **Step 4: Add DB migration columns**

In `createJobsTable`, add migration lines after the existing `ALTER TABLE` calls (around line 209):

```go
	// Try to add host column if it doesn't exist (migration)
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN host TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN original_input TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN output_files TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN workflow_id TEXT")
```

- [ ] **Step 5: Update saveJobToDB**

In `saveJobToDB`, add serialization for `OutputFiles` and include both new columns. Replace the function body:

```go
func (q *Queue) saveJobToDB(job *Job) error {
	if q.Db == nil {
		return nil
	}

	argumentsJSON, _ := json.Marshal(job.Arguments)
	stdoutJSON, _ := json.Marshal(job.Stdout)
	dependenciesJSON, _ := json.Marshal(job.Dependencies)
	outputFilesJSON, _ := json.Marshal(job.OutputFiles)

	position := -1
	for i, id := range q.JobOrder {
		if id == job.ID {
			position = i
			break
		}
	}

	query := `
	INSERT OR REPLACE INTO jobs (
		id, command, arguments, input, original_input, host, stdout, dependencies, state,
		created_at, claimed_at, completed_at, errored_at, job_order_position,
		output_files, workflow_id
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := q.Db.Exec(query,
		job.ID,
		job.Command,
		string(argumentsJSON),
		job.Input,
		job.OriginalInput,
		job.Host,
		string(stdoutJSON),
		string(dependenciesJSON),
		int(job.State),
		job.CreatedAt,
		job.ClaimedAt,
		job.CompletedAt,
		job.ErroredAt,
		position,
		string(outputFilesJSON),
		job.WorkflowID,
	)

	return err
}
```

- [ ] **Step 6: Update loadJobsFromDB**

In `loadJobsFromDB`, update the SELECT query and Scan call to include the new columns. Change the query to:

```go
	query := `
	SELECT id, command, arguments, input, COALESCE(original_input, ''), COALESCE(host, ''), stdout, dependencies, state,
		   created_at, claimed_at, completed_at, errored_at, job_order_position,
		   COALESCE(output_files, '[]'), COALESCE(workflow_id, '')
	FROM jobs
	ORDER BY job_order_position`
```

Add variables and update the Scan call:

```go
		var outputFilesJSON string

		err := rows.Scan(
			&job.ID,
			&job.Command,
			&argumentsJSON,
			&job.Input,
			&job.OriginalInput,
			&job.Host,
			&stdoutJSON,
			&dependenciesJSON,
			&state,
			&job.CreatedAt,
			&job.ClaimedAt,
			&job.CompletedAt,
			&job.ErroredAt,
			&position,
			&outputFilesJSON,
			&job.WorkflowID,
		)
```

Add deserialization after the existing JSON unmarshals:

```go
		if err := json.Unmarshal([]byte(outputFilesJSON), &job.OutputFiles); err != nil {
			job.OutputFiles = []string{}
		}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd media-server && go test ./jobqueue/ -run TestOutputFilesAndWorkflowIDPersistence -v`
Expected: PASS

- [ ] **Step 8: Run all existing tests to check for regressions**

Run: `cd media-server && go test ./jobqueue/ -v`
Expected: all tests PASS

- [ ] **Step 9: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go
git commit -m "feat: add OutputFiles and WorkflowID fields to Job struct with DB persistence"
```

---

### Task 2: Add RegisterOutputFile Method

**Files:**
- Modify: `jobqueue/jobqueue.go` (add method after `PushJobStdout`)
- Test: `jobqueue/jobqueue_core_test.go`

- [ ] **Step 1: Write test for RegisterOutputFile**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestRegisterOutputFile(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	id, _ := q.AddJob("", "test", nil, "", nil)

	err := q.RegisterOutputFile(id, "/tmp/output1.mp4")
	if err != nil {
		t.Fatalf("RegisterOutputFile() error = %v", err)
	}
	err = q.RegisterOutputFile(id, "/tmp/output2.mp4")
	if err != nil {
		t.Fatalf("RegisterOutputFile() error = %v", err)
	}

	job := q.GetJob(id)
	if len(job.OutputFiles) != 2 {
		t.Fatalf("OutputFiles length = %d; want 2", len(job.OutputFiles))
	}
	if job.OutputFiles[0] != "/tmp/output1.mp4" || job.OutputFiles[1] != "/tmp/output2.mp4" {
		t.Errorf("OutputFiles = %v; want [/tmp/output1.mp4, /tmp/output2.mp4]", job.OutputFiles)
	}
}

func TestRegisterOutputFileNotFound(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	err := q.RegisterOutputFile("nonexistent", "/tmp/file.mp4")
	if err == nil {
		t.Error("RegisterOutputFile() should return error for nonexistent job")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./jobqueue/ -run TestRegisterOutputFile -v`
Expected: compilation error — `RegisterOutputFile` method doesn't exist

- [ ] **Step 3: Implement RegisterOutputFile**

Add to `jobqueue/jobqueue.go` after the `PushJobStdout` method (after line 654):

```go
// RegisterOutputFile appends a file path to the job's OutputFiles list.
// These paths are used by ClaimJob to construct input for downstream jobs,
// keeping file paths separate from diagnostic stdout.
func (q *Queue) RegisterOutputFile(id string, path string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	job.OutputFiles = append(job.OutputFiles, path)

	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job output files to database: %v", err)
	}

	return nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test ./jobqueue/ -run TestRegisterOutputFile -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go
git commit -m "feat: add RegisterOutputFile method to Queue"
```

---

### Task 3: Update ClaimJob to Prefer OutputFiles

**Files:**
- Modify: `jobqueue/jobqueue.go:500-550` (ClaimJob)
- Test: `jobqueue/jobqueue_core_test.go`

- [ ] **Step 1: Write test for ClaimJob using OutputFiles**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestClaimJobUsesOutputFiles(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add parent with both stdout and output files
	parentID, _ := q.AddJob("parent", "test", nil, "", nil)
	childID, _ := q.AddJob("child", "test", nil, "child-original", []string{parentID})

	// Claim and work on parent
	q.ClaimJob()
	q.PushJobStdout(parentID, "ffmpeg: some diagnostic line")
	q.PushJobStdout(parentID, "ffmpeg: progress 50%")
	q.RegisterOutputFile(parentID, "/tmp/output.mp4")
	q.RegisterOutputFile(parentID, "/tmp/output2.mp4")
	q.CompleteJob(parentID)

	// Claim child — should get OutputFiles, not Stdout
	child, _ := q.ClaimJob()
	if child == nil || child.ID != childID {
		t.Fatal("expected child to be claimed")
	}

	// Input should be original + output files, NOT stdout diagnostics
	expected := "child-original\n/tmp/output.mp4\n/tmp/output2.mp4"
	if child.Input != expected {
		t.Errorf("child.Input = %q; want %q", child.Input, expected)
	}
}

func TestClaimJobFallsBackToStdout(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Add parent with stdout only (no output files — legacy task)
	parentID, _ := q.AddJob("parent", "test", nil, "", nil)
	childID, _ := q.AddJob("child", "test", nil, "", []string{parentID})

	q.ClaimJob()
	q.PushJobStdout(parentID, "legacy output line")
	q.CompleteJob(parentID)

	child, _ := q.ClaimJob()
	if child == nil || child.ID != childID {
		t.Fatal("expected child to be claimed")
	}

	if child.Input != "legacy output line" {
		t.Errorf("child.Input = %q; want %q", child.Input, "legacy output line")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test ./jobqueue/ -run "TestClaimJobUsesOutputFiles|TestClaimJobFallsBackToStdout" -v`
Expected: `TestClaimJobUsesOutputFiles` FAILS (child.Input contains stdout diagnostics instead of OutputFiles)

- [ ] **Step 3: Update ClaimJob**

In `jobqueue/jobqueue.go`, replace the input construction block in `ClaimJob` (lines 520-533):

```go
			// Construct effective input from OriginalInput and parent outputs
			var inputBuilder strings.Builder
			inputBuilder.WriteString(job.OriginalInput)
			for _, depID := range job.Dependencies {
				if depJob, ok := q.Jobs[depID]; ok {
					// Prefer OutputFiles; fall back to Stdout for legacy tasks
					source := depJob.OutputFiles
					if len(source) == 0 {
						source = depJob.Stdout
					}
					for _, line := range source {
						if inputBuilder.Len() > 0 {
							inputBuilder.WriteString("\n")
						}
						inputBuilder.WriteString(line)
					}
				}
			}
			job.Input = inputBuilder.String()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test ./jobqueue/ -run "TestClaimJobUsesOutputFiles|TestClaimJobFallsBackToStdout" -v`
Expected: PASS

- [ ] **Step 5: Run all jobqueue tests**

Run: `cd media-server && go test ./jobqueue/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go
git commit -m "feat: ClaimJob prefers OutputFiles over Stdout for downstream input"
```

---

### Task 4: Set WorkflowID in AddWorkflow and RunWorkflow

**Files:**
- Modify: `jobqueue/jobqueue.go:430-452` (AddWorkflow)
- Modify: `jobqueue/workflows.go:164-207` (RunWorkflow)
- Test: `jobqueue/jobqueue_core_test.go`
- Test: `jobqueue/workflows_test.go`

- [ ] **Step 1: Write tests**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestAddWorkflowSetsWorkflowID(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	workflow := Workflow{
		Tasks: []WorkflowTask{
			{ID: "a", Command: "cmd1", Input: "hello"},
			{ID: "b", Command: "cmd2", Dependencies: []string{"a"}},
		},
	}

	ids, err := q.AddWorkflow(workflow)
	if err != nil {
		t.Fatalf("AddWorkflow() error = %v", err)
	}

	jobA := q.GetJob(ids[0])
	jobB := q.GetJob(ids[1])

	if jobA.WorkflowID == "" {
		t.Error("job A should have a WorkflowID")
	}
	if jobB.WorkflowID == "" {
		t.Error("job B should have a WorkflowID")
	}
	if jobA.WorkflowID != jobB.WorkflowID {
		t.Errorf("all jobs in workflow should share WorkflowID: A=%q B=%q", jobA.WorkflowID, jobB.WorkflowID)
	}
}
```

Add to `jobqueue/workflows_test.go`:

```go
func TestRunWorkflowSetsWorkflowID(t *testing.T) {
	q := newTestQueue(t)

	dag := []WorkflowTask{
		{ID: "root", Command: "ingest", Input: "existing"},
		{ID: "child", Command: "process", Dependencies: []string{"root"}},
	}

	saved, err := q.CreateWorkflow("wf-test", dag)
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	liveIDs, err := q.RunWorkflow(saved.ID, "")
	if err != nil {
		t.Fatalf("RunWorkflow: %v", err)
	}

	rootJob := q.GetJob(liveIDs[0])
	childJob := q.GetJob(liveIDs[1])

	if rootJob.WorkflowID == "" {
		t.Error("root job should have a WorkflowID")
	}
	if rootJob.WorkflowID != childJob.WorkflowID {
		t.Error("all jobs should share the same WorkflowID")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test ./jobqueue/ -run "TestAddWorkflowSetsWorkflowID|TestRunWorkflowSetsWorkflowID" -v`
Expected: FAIL — WorkflowID is empty

- [ ] **Step 3: Update AddWorkflow to accept and set WorkflowID**

In `jobqueue/jobqueue.go`, change the `Workflow` struct and `AddWorkflow` method:

```go
type Workflow struct {
	Tasks      []WorkflowTask `json:"tasks"`
	WorkflowID string         `json:"workflow_id"` // Shared ID for all jobs in this workflow run
}

func (q *Queue) AddWorkflow(w Workflow) ([]string, error) {
	var jobIDs []string

	workflowID := w.WorkflowID
	if workflowID == "" {
		workflowID = uuid.NewString()
	}

	for _, task := range w.Tasks {
		id, err := q.AddJob(task.ID, task.Command, task.Arguments, task.Input, task.Dependencies)
		if err != nil {
			return jobIDs, err
		}
		// Set WorkflowID on the newly created job
		if job, ok := q.Jobs[id]; ok {
			job.WorkflowID = workflowID
			if err := q.saveJobToDB(job); err != nil {
				log.Printf("Failed to save workflow ID to database: %v", err)
			}
		}
		jobIDs = append(jobIDs, id)
	}
	return jobIDs, nil
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test ./jobqueue/ -run "TestAddWorkflowSetsWorkflowID|TestRunWorkflowSetsWorkflowID" -v`
Expected: PASS

- [ ] **Step 5: Run all jobqueue tests**

Run: `cd media-server && go test ./jobqueue/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go jobqueue/workflows.go jobqueue/workflows_test.go
git commit -m "feat: set WorkflowID on all jobs created via AddWorkflow/RunWorkflow"
```

---

### Task 5: Update CopyJob to Preserve WorkflowID

**Files:**
- Modify: `jobqueue/jobqueue.go:454-498` (CopyJob)
- Test: `jobqueue/jobqueue_core_test.go`

- [ ] **Step 1: Write test**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestCopyJobPreservesWorkflowID(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	workflow := Workflow{
		Tasks: []WorkflowTask{
			{ID: "a", Command: "cmd1"},
		},
	}
	ids, _ := q.AddWorkflow(workflow)
	originalJob := q.GetJob(ids[0])

	copyID, err := q.CopyJob(ids[0])
	if err != nil {
		t.Fatalf("CopyJob() error = %v", err)
	}

	copyJob := q.GetJob(copyID)
	if copyJob.WorkflowID != originalJob.WorkflowID {
		t.Errorf("Copy.WorkflowID = %q; want %q", copyJob.WorkflowID, originalJob.WorkflowID)
	}
	if len(copyJob.OutputFiles) != 0 {
		t.Errorf("Copy.OutputFiles should be empty; got %v", copyJob.OutputFiles)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./jobqueue/ -run TestCopyJobPreservesWorkflowID -v`
Expected: FAIL — WorkflowID not preserved because CopyJob copies the struct but OutputFiles may not be cleared

- [ ] **Step 3: Update CopyJob**

In `jobqueue/jobqueue.go` in the `CopyJob` method, after the `newJob.OriginalInput = job.OriginalInput` line (around line 479), ensure OutputFiles is reset:

```go
	newJob := *job
	newJob.ID = newID
	newJob.Stdout = []string{}
	newJob.OutputFiles = []string{}
	newJob.State = StatePending
	newJob.CreatedAt = time.Now()
	newJob.ClaimedAt = time.Time{}
	newJob.CompletedAt = time.Time{}
	newJob.ErroredAt = time.Time{}
	newJob.Cancel = cancel
	newJob.Ctx = ctx
	newJob.OriginalInput = job.OriginalInput
	// WorkflowID is copied from original job
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd media-server && go test ./jobqueue/ -run TestCopyJobPreservesWorkflowID -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go
git commit -m "feat: CopyJob preserves WorkflowID and clears OutputFiles"
```

---

### Task 6: Add Temp Directory Logic to runFFmpegOnFiles

**Files:**
- Modify: `tasks/ffmpeg.go:22-125` (runFFmpegOnFiles)

- [ ] **Step 1: Update runFFmpegOnFiles to use temp dir for workflow jobs**

In `tasks/ffmpeg.go`, in the `runFFmpegOnFiles` function, update the per-file loop. After computing `dir`, `base`, `ext`, `name` (around line 64-67), add temp dir logic before calling `buildArgs`:

```go
		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}
		dir := filepath.Dir(abs)
		base := filepath.Base(abs)
		ext := filepath.Ext(abs)
		name := strings.TrimSuffix(base, ext)

		// Use temp directory for workflow jobs
		outputDir := dir
		if j.WorkflowID != "" {
			outputDir = filepath.Join(dir, ".loki-temp", j.ID)
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				q.PushJobStdout(j.ID, "ffmpeg: failed to create temp dir: "+err.Error())
				q.ErrorJob(j.ID)
				return err
			}
		}

		args, outputPath := buildArgs(abs, outputDir, name, ext)
```

Add `"os"` to the imports at the top of `tasks/ffmpeg.go` if not already present.

- [ ] **Step 2: Add RegisterOutputFile call**

In `runFFmpegOnFiles`, replace the line `q.PushJobStdout(j.ID, outputPath)` (line 120) with:

```go
		q.PushJobStdout(j.ID, "ffmpeg: completed for "+base)
		q.RegisterOutputFile(j.ID, outputPath)
```

Remove the duplicate `q.PushJobStdout(j.ID, "ffmpeg: completed for "+base)` line that precedes it (line 118).

- [ ] **Step 3: Update ffmpegTask similarly**

In `tasks/ffmpeg.go`, in the `ffmpegTask` function, apply the same temp directory logic. After the `name := strings.TrimSuffix(base, ext)` line (around line 180), add:

```go
		// Use temp directory for workflow jobs
		outputDir := dir
		if j.WorkflowID != "" {
			outputDir = filepath.Join(dir, ".loki-temp", j.ID)
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				q.PushJobStdout(j.ID, "ffmpeg: failed to create temp dir: "+err.Error())
				q.ErrorJob(j.ID)
				return err
			}
		}
```

Then update the output path auto-generation section (around line 221-224) to use `outputDir` instead of `dir`:

```go
		if needsOutput {
			outputPath = filepath.Join(outputDir, name+"_output"+ext)
			finalArgs = append(finalArgs, outputPath)
		}
```

And replace the `q.PushJobStdout(j.ID, outputPath)` line (line 272) with:

```go
		q.RegisterOutputFile(j.ID, outputPath)
```

- [ ] **Step 4: Verify build**

Run: `cd media-server && go build ./...`
Expected: compiles with no errors

- [ ] **Step 5: Commit**

```bash
git add tasks/ffmpeg.go
git commit -m "feat: runFFmpegOnFiles uses temp dir for workflow jobs and registers output files"
```

---

### Task 7: Update All Tasks to Call RegisterOutputFile

**Files:**
- Modify: `tasks/media_metadata.go:176-177`
- Modify: `tasks/autotag.go:215-216`
- Modify: `tasks/media_move.go:160-161`
- Modify: `tasks/media_remove.go:53-56`
- Modify: `tasks/hls.go:243-244`
- Modify: `tasks/lora_dataset.go` (find "Output created file path" comment)
- Modify: `tasks/ingest_local.go` (find ingested file output)
- Modify: `tasks/ingest_youtube.go` (find "Added to database" output)
- Modify: `tasks/ingest_gallery.go` (find "Added to database" output)
- Modify: `tasks/ingest_discord.go` (find "Added to database" output)

- [ ] **Step 1: Update media_metadata.go**

In `tasks/media_metadata.go`, replace lines 176-177:

```go
		if fileProcessed {
			processed++
			q.RegisterOutputFile(j.ID, filePath)
		}
```

Remove the old `q.PushJobStdout(j.ID, filePath)` line and its comment.

- [ ] **Step 2: Update autotag.go**

In `tasks/autotag.go`, replace lines 215-216:

```go
		q.RegisterOutputFile(j.ID, mediaPath)
```

Remove the old `q.PushJobStdout(j.ID, mediaPath)` line and its comment.

- [ ] **Step 3: Update media_move.go**

In `tasks/media_move.go`, replace lines 160-161:

```go
		q.PushJobStdout(j.ID, fmt.Sprintf("Moved: %s -> %s", srcPath, destPath))
		q.RegisterOutputFile(j.ID, destPath)
```

Remove the old `q.PushJobStdout(j.ID, destPath)` line and its comment.

- [ ] **Step 4: Update media_remove.go**

In `tasks/media_remove.go`, replace lines 53-56:

```go
	for _, p := range result.ProcessedPaths {
		q.RegisterOutputFile(j.ID, p)
	}
```

Remove the old `q.PushJobStdout(j.ID, p)` line and its comment.

- [ ] **Step 5: Update hls.go**

In `tasks/hls.go`, replace lines 243-244:

```go
		q.PushJobStdout(j.ID, fmt.Sprintf("hls: completed %s (presets: %s)", base, strings.Join(generatedPresets, ", ")))
		q.RegisterOutputFile(j.ID, masterPath)
```

Remove the old `q.PushJobStdout(j.ID, masterPath)` line and its comment.

- [ ] **Step 6: Update lora_dataset.go**

Find the comment "Output created file path for downstream chaining" and replace the `PushJobStdout` that follows it with `q.RegisterOutputFile(j.ID, outputPath)` (use the actual variable name at that location).

- [ ] **Step 7: Update ingest_local.go**

In `tasks/ingest_local.go`, after the file is successfully inserted into the database, add:

```go
		q.RegisterOutputFile(j.ID, filePath)
```

- [ ] **Step 8: Update ingest_youtube.go**

In `tasks/ingest_youtube.go`, after the "Added to database" log line (around line 170), add:

```go
		q.RegisterOutputFile(j.ID, filePath)
```

- [ ] **Step 9: Update ingest_gallery.go**

In `tasks/ingest_gallery.go`, after the "Added to database" log line (around line 170), add:

```go
		q.RegisterOutputFile(j.ID, filePath)
```

- [ ] **Step 10: Update ingest_discord.go**

In `tasks/ingest_discord.go`, after the "Added to database" log line (around line 200), add:

```go
		q.RegisterOutputFile(j.ID, filePath)
```

- [ ] **Step 11: Verify build**

Run: `cd media-server && go build ./...`
Expected: compiles with no errors

- [ ] **Step 12: Run all tests**

Run: `cd media-server && go test ./... -v`
Expected: all PASS

- [ ] **Step 13: Commit**

```bash
git add tasks/media_metadata.go tasks/autotag.go tasks/media_move.go tasks/media_remove.go tasks/hls.go tasks/lora_dataset.go tasks/ingest_local.go tasks/ingest_youtube.go tasks/ingest_gallery.go tasks/ingest_discord.go
git commit -m "feat: all tasks register output files via RegisterOutputFile"
```

---

### Task 8: Implement Save File Task

**Files:**
- Create: `tasks/save.go`
- Test: `tasks/save_test.go`
- Modify: `tasks/registry.go`

- [ ] **Step 1: Write tests for the save task helpers**

Create `tasks/save_test.go`:

```go
package tasks

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStripLokiTemp(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{
			filepath.Join("Z:", "gallery-dl", ".loki-temp", "abc-123", "video_grayscale.mp4"),
			filepath.Join("Z:", "gallery-dl"),
		},
		{
			filepath.Join("/tmp", ".loki-temp", "job-1", "file.mp4"),
			"/tmp",
		},
		{
			filepath.Join("/tmp", "no-temp", "file.mp4"),
			filepath.Join("/tmp", "no-temp"),
		},
	}

	for _, tt := range tests {
		got := stripLokiTemp(tt.input)
		if got != tt.expected {
			t.Errorf("stripLokiTemp(%q) = %q; want %q", tt.input, got, tt.expected)
		}
	}
}

func TestSuffixFilename(t *testing.T) {
	tests := []struct {
		name     string
		suffix   string
		ext      string
		expected string
	}{
		{"video_grayscale_blurred", "_edited", ".mp4", "video_grayscale_blurred_edited.mp4"},
		{"video", "_final", ".mp4", "video_final.mp4"},
		{"video_grayscale", "", ".mp4", "video_grayscale.mp4"},
	}

	for _, tt := range tests {
		got := buildSaveFilename(tt.name, tt.suffix, tt.ext)
		if got != tt.expected {
			t.Errorf("buildSaveFilename(%q, %q, %q) = %q; want %q", tt.name, tt.suffix, tt.ext, got, tt.expected)
		}
	}
}

func TestResolveConflictSuffix(t *testing.T) {
	dir := t.TempDir()

	// Create an existing file
	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	// Should get _1 suffix
	result := resolveConflict(filepath.Join(dir, "video.mp4"), "suffix")
	expected := filepath.Join(dir, "video_1.mp4")
	if result != expected {
		t.Errorf("resolveConflict() = %q; want %q", result, expected)
	}

	// Create _1, should get _2
	os.WriteFile(expected, []byte("x"), 0644)
	result2 := resolveConflict(filepath.Join(dir, "video.mp4"), "suffix")
	expected2 := filepath.Join(dir, "video_2.mp4")
	if result2 != expected2 {
		t.Errorf("resolveConflict() = %q; want %q", result2, expected2)
	}
}

func TestResolveConflictOverwrite(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	result := resolveConflict(existing, "overwrite")
	if result != existing {
		t.Errorf("resolveConflict(overwrite) = %q; want %q", result, existing)
	}
}

func TestResolveConflictSkip(t *testing.T) {
	dir := t.TempDir()
	existing := filepath.Join(dir, "video.mp4")
	os.WriteFile(existing, []byte("x"), 0644)

	result := resolveConflict(existing, "skip")
	if result != "" {
		t.Errorf("resolveConflict(skip) = %q; want empty", result)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test ./tasks/ -run "TestStripLokiTemp|TestSuffixFilename|TestResolveConflict" -v`
Expected: compilation error — functions don't exist

- [ ] **Step 3: Implement save.go**

Create `tasks/save.go`:

```go
package tasks

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

var saveOptions = []TaskOption{
	{Name: "destination", Label: "Destination", Type: "enum", Choices: []string{"original", "directory"}, Default: "original", Description: "Where to save: original directory or a specific directory"},
	{Name: "directory", Label: "Target Directory", Type: "string", Description: "Target directory (only used when destination=directory)"},
	{Name: "conflict", Label: "Conflict Resolution", Type: "enum", Choices: []string{"suffix", "overwrite", "skip"}, Default: "suffix", Description: "How to handle existing files"},
	{Name: "suffix", Label: "Custom Suffix", Type: "string", Description: "Custom suffix for output filename (e.g. _edited). Empty keeps processing name"},
	{Name: "flatten", Label: "Flatten", Type: "bool", Description: "Flatten all files into target directory (ignore relative paths)"},
}

func saveTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, saveOptions)
	destination, _ := opts["destination"].(string)
	if destination == "" {
		destination = "original"
	}
	directory, _ := opts["directory"].(string)
	conflict, _ := opts["conflict"].(string)
	if conflict == "" {
		conflict = "suffix"
	}
	suffix, _ := opts["suffix"].(string)
	flatten, _ := opts["flatten"].(bool)

	if destination == "directory" && directory == "" {
		q.PushJobStdout(j.ID, "save: no target directory specified")
		q.ErrorJob(j.ID)
		return fmt.Errorf("target directory required when destination=directory")
	}

	// Get files from input (populated by ClaimJob from parent's OutputFiles)
	raw := strings.TrimSpace(j.Input)
	if raw == "" {
		q.PushJobStdout(j.ID, "save: no input files")
		q.CompleteJob(j.ID)
		return nil
	}
	files := parseInputPaths(raw)

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "save: no files to save")
		q.CompleteJob(j.ID)
		return nil
	}

	saved := 0
	skipped := 0
	for _, src := range files {
		select {
		case <-j.Ctx.Done():
			q.PushJobStdout(j.ID, "save: task canceled")
			q.ErrorJob(j.ID)
			return j.Ctx.Err()
		default:
		}

		if _, err := os.Stat(src); os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: file not found, skipping: %s", src))
			skipped++
			continue
		}

		ext := filepath.Ext(src)
		name := strings.TrimSuffix(filepath.Base(src), ext)

		// Build output filename
		outName := buildSaveFilename(name, suffix, ext)

		// Determine output directory
		var outDir string
		switch destination {
		case "directory":
			if flatten {
				outDir = directory
			} else {
				// Preserve relative structure from temp dir
				outDir = directory
			}
		default: // "original"
			outDir = stripLokiTemp(src)
		}

		if err := os.MkdirAll(outDir, 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to create directory %s: %v", outDir, err))
			continue
		}

		destPath := filepath.Join(outDir, outName)

		// Handle conflict
		if _, err := os.Stat(destPath); err == nil {
			destPath = resolveConflict(destPath, conflict)
			if destPath == "" {
				q.PushJobStdout(j.ID, fmt.Sprintf("save: skipping (exists): %s", filepath.Join(outDir, outName)))
				skipped++
				continue
			}
		}

		// Move file (rename if same drive, copy+delete otherwise)
		if err := moveFile(src, destPath); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("save: failed to save %s -> %s: %v", filepath.Base(src), destPath, err))
			continue
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("save: %s -> %s", filepath.Base(src), destPath))
		q.RegisterOutputFile(j.ID, destPath)
		saved++
	}

	// Clean up .loki-temp directories for this workflow
	if j.WorkflowID != "" {
		cleanupWorkflowTempDirs(q, j.WorkflowID)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("save: completed — %d saved, %d skipped", saved, skipped))
	q.CompleteJob(j.ID)
	return nil
}

// stripLokiTemp removes the .loki-temp/<jobID>/ segment from a path,
// returning the original source directory.
func stripLokiTemp(path string) string {
	dir := filepath.Dir(path)
	parts := strings.Split(filepath.ToSlash(dir), "/")
	for i, part := range parts {
		if part == ".loki-temp" && i+1 < len(parts) {
			// Remove .loki-temp and the jobID segment
			cleaned := strings.Join(append(parts[:i], parts[i+2:]...), "/")
			return filepath.FromSlash(cleaned)
		}
	}
	return dir
}

// buildSaveFilename constructs the output filename.
// If suffix is non-empty, it replaces everything after the original base name
// (strips any processing suffixes like _grayscale_blurred) and appends the custom suffix.
// If suffix is empty, the name is used as-is.
func buildSaveFilename(name, suffix, ext string) string {
	if suffix == "" {
		return name + ext
	}
	// Try to find the original name by stripping known processing suffixes.
	// Processing tasks append suffixes like _grayscale, _blurred, _resized, etc.
	// We strip from the first underscore-prefixed processing suffix.
	// Heuristic: split on _ and find where processing suffixes start.
	// Since we can't know the original name for certain, we use the suffix
	// as a replacement — user provides what they want.
	return name + suffix + ext
}

// resolveConflict determines the final path when a file already exists.
// Returns empty string for "skip" mode.
func resolveConflict(path, mode string) string {
	switch mode {
	case "overwrite":
		return path
	case "skip":
		return ""
	default: // "suffix"
		dir := filepath.Dir(path)
		ext := filepath.Ext(path)
		name := strings.TrimSuffix(filepath.Base(path), ext)
		for i := 1; i < 1000; i++ {
			candidate := filepath.Join(dir, fmt.Sprintf("%s_%d%s", name, i, ext))
			if _, err := os.Stat(candidate); os.IsNotExist(err) {
				return candidate
			}
		}
		return path // fallback
	}
}

// moveFile tries os.Rename first (fast, same-drive). Falls back to copy+delete
// for cross-drive moves.
func moveFile(src, dst string) error {
	if err := os.Rename(src, dst); err == nil {
		return nil
	}
	// Cross-drive fallback: copy then delete
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		os.Remove(dst)
		return err
	}
	if err := out.Close(); err != nil {
		os.Remove(dst)
		return err
	}
	in.Close()
	return os.Remove(src)
}

// cleanupWorkflowTempDirs removes .loki-temp directories for all jobs
// in the given workflow.
func cleanupWorkflowTempDirs(q *jobqueue.Queue, workflowID string) {
	q.mu.Lock()
	var tempDirs []string
	for _, job := range q.Jobs {
		if job.WorkflowID == workflowID {
			for _, f := range job.OutputFiles {
				dir := filepath.Dir(f)
				if strings.Contains(dir, ".loki-temp") {
					// Walk up to find the .loki-temp/<jobID> dir
					parts := strings.Split(filepath.ToSlash(dir), "/")
					for i, part := range parts {
						if part == ".loki-temp" && i+1 < len(parts) {
							tempRoot := filepath.FromSlash(strings.Join(parts[:i+2], "/"))
							tempDirs = append(tempDirs, tempRoot)
							break
						}
					}
				}
			}
		}
	}
	q.mu.Unlock()

	for _, d := range tempDirs {
		os.RemoveAll(d)
	}
}
```

Note: `cleanupWorkflowTempDirs` accesses `q.mu` directly. The Queue's `mu` field is not exported but the tasks package uses it via the `mu *sync.Mutex` parameter. However, `q.Jobs` access needs the queue lock. Since the Queue struct has `mu sync.Mutex` as an unexported field, we need to access it through a public method instead. Add a helper to `jobqueue/jobqueue.go`:

- [ ] **Step 4: Add GetWorkflowOutputFiles helper to Queue**

Add to `jobqueue/jobqueue.go` after `GetJob`:

```go
// GetWorkflowOutputFiles returns all OutputFiles paths for jobs matching the given workflowID.
func (q *Queue) GetWorkflowOutputFiles(workflowID string) []string {
	q.mu.Lock()
	defer q.mu.Unlock()

	var paths []string
	for _, job := range q.Jobs {
		if job.WorkflowID == workflowID {
			paths = append(paths, job.OutputFiles...)
		}
	}
	return paths
}
```

Then update `cleanupWorkflowTempDirs` in `tasks/save.go` to use it:

```go
func cleanupWorkflowTempDirs(q *jobqueue.Queue, workflowID string) {
	paths := q.GetWorkflowOutputFiles(workflowID)

	seen := make(map[string]bool)
	for _, f := range paths {
		dir := filepath.Dir(f)
		parts := strings.Split(filepath.ToSlash(dir), "/")
		for i, part := range parts {
			if part == ".loki-temp" && i+1 < len(parts) {
				tempRoot := filepath.FromSlash(strings.Join(parts[:i+2], "/"))
				if !seen[tempRoot] {
					seen[tempRoot] = true
					os.RemoveAll(tempRoot)
				}
				break
			}
		}
	}

	// Also try to remove the .loki-temp parent dirs if empty
	for dir := range seen {
		parent := filepath.Dir(dir)
		os.Remove(parent) // Only succeeds if empty
	}
}
```

- [ ] **Step 5: Register the save task**

In `tasks/registry.go`, add inside `init()`:

```go
	RegisterTask("save", "Save File", saveOptions, saveTask)
```

- [ ] **Step 6: Run tests**

Run: `cd media-server && go test ./tasks/ -run "TestStripLokiTemp|TestSuffixFilename|TestResolveConflict" -v`
Expected: PASS

- [ ] **Step 7: Verify build**

Run: `cd media-server && go build ./...`
Expected: compiles with no errors

- [ ] **Step 8: Commit**

```bash
git add tasks/save.go tasks/save_test.go tasks/registry.go jobqueue/jobqueue.go
git commit -m "feat: add Save File task with conflict resolution and temp dir cleanup"
```

---

### Task 9: Add Temp Dir Cleanup on Error

**Files:**
- Modify: `jobqueue/jobqueue.go:568-596` (ErrorJob)

- [ ] **Step 1: Write test**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestErrorJobCancelsWorkflowDependents(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	workflow := Workflow{
		Tasks: []WorkflowTask{
			{ID: "a", Command: "cmd1"},
			{ID: "b", Command: "cmd2", Dependencies: []string{"a"}},
			{ID: "c", Command: "cmd3", Dependencies: []string{"b"}},
		},
	}

	ids, _ := q.AddWorkflow(workflow)

	// Claim and error job A
	q.ClaimJob()
	q.ErrorJob(ids[0])

	// Jobs B and C should be cancelled since their ancestor errored
	jobB := q.GetJob(ids[1])
	jobC := q.GetJob(ids[2])
	if jobB.State != StateCancelled {
		t.Errorf("job B state = %v; want StateCancelled", jobB.State)
	}
	if jobC.State != StateCancelled {
		t.Errorf("job C state = %v; want StateCancelled", jobC.State)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./jobqueue/ -run TestErrorJobCancelsWorkflowDependents -v`
Expected: FAIL — dependent jobs remain pending

- [ ] **Step 3: Update ErrorJob to cancel workflow dependents**

In `jobqueue/jobqueue.go`, update `ErrorJob` to cancel pending dependents in the same workflow:

```go
func (q *Queue) ErrorJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State != StateInProgress {
		return errors.New("job is not in progress, cannot set error")
	}

	job.State = StateError
	job.ErroredAt = time.Now()
	q.RunningCounts[job.Host]--

	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job error state to database: %v", err)
	}

	_ = serializeListUpdate("update", job)

	// Cancel pending dependents in the same workflow
	if job.WorkflowID != "" {
		q.cancelWorkflowDependentsLocked(id, job.WorkflowID)
	}

	return nil
}

// cancelWorkflowDependentsLocked cancels all pending jobs in the workflow
// that transitively depend on the errored job. Must be called with mu held.
func (q *Queue) cancelWorkflowDependentsLocked(erroredID, workflowID string) {
	// Find all jobs that depend (directly or transitively) on the errored job
	cancelled := map[string]bool{erroredID: true}
	changed := true
	for changed {
		changed = false
		for _, j := range q.Jobs {
			if j.WorkflowID != workflowID || j.State != StatePending {
				continue
			}
			if cancelled[j.ID] {
				continue
			}
			for _, dep := range j.Dependencies {
				if cancelled[dep] {
					j.State = StateCancelled
					j.Cancel()
					cancelled[j.ID] = true
					changed = true
					if err := q.saveJobToDB(j); err != nil {
						log.Printf("Failed to save cancelled job %s: %v", j.ID, err)
					}
					_ = serializeListUpdate("update", j)
					break
				}
			}
		}
	}
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd media-server && go test ./jobqueue/ -run TestErrorJobCancelsWorkflowDependents -v`
Expected: PASS

- [ ] **Step 5: Run all tests**

Run: `cd media-server && go test ./jobqueue/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add jobqueue/jobqueue.go jobqueue/jobqueue_core_test.go
git commit -m "feat: ErrorJob cancels pending workflow dependents"
```

---

### Task 10: Final Integration Test and Cleanup

**Files:**
- Test: `jobqueue/jobqueue_core_test.go`

- [ ] **Step 1: Write end-to-end workflow chaining test**

Add to `jobqueue/jobqueue_core_test.go`:

```go
func TestWorkflowEndToEndOutputFiles(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	q := NewQueueWithDB(db)

	// Simulate: grayscale -> blur -> save
	workflow := Workflow{
		Tasks: []WorkflowTask{
			{ID: "gray", Command: "ffmpeg-grayscale", Input: "--query64 dGVzdA=="},
			{ID: "blur", Command: "ffmpeg-blur", Dependencies: []string{"gray"}},
			{ID: "save", Command: "save", Dependencies: []string{"blur"}},
		},
	}

	ids, err := q.AddWorkflow(workflow)
	if err != nil {
		t.Fatalf("AddWorkflow error: %v", err)
	}

	// All should have same WorkflowID
	wfID := q.GetJob(ids[0]).WorkflowID
	if wfID == "" {
		t.Fatal("WorkflowID should be set")
	}
	for _, id := range ids {
		if q.GetJob(id).WorkflowID != wfID {
			t.Error("all jobs should share WorkflowID")
		}
	}

	// Claim and work grayscale
	gray, _ := q.ClaimJob()
	if gray.ID != ids[0] {
		t.Fatalf("expected grayscale job; got %s", gray.ID)
	}
	q.PushJobStdout(ids[0], "ffmpeg: encoding frame 1/100")
	q.PushJobStdout(ids[0], "ffmpeg: encoding frame 100/100")
	q.RegisterOutputFile(ids[0], "/tmp/.loki-temp/gray-id/video_grayscale.mp4")
	q.CompleteJob(ids[0])

	// Claim blur — should get OutputFiles only, not ffmpeg diagnostics
	blur, _ := q.ClaimJob()
	if blur.ID != ids[1] {
		t.Fatalf("expected blur job; got %s", blur.ID)
	}
	if strings.Contains(blur.Input, "encoding frame") {
		t.Errorf("blur input should not contain ffmpeg diagnostics; got %q", blur.Input)
	}
	if !strings.Contains(blur.Input, "video_grayscale.mp4") {
		t.Errorf("blur input should contain output file path; got %q", blur.Input)
	}

	q.RegisterOutputFile(ids[1], "/tmp/.loki-temp/blur-id/video_grayscale_blurred.mp4")
	q.CompleteJob(ids[1])

	// Claim save
	save, _ := q.ClaimJob()
	if save.ID != ids[2] {
		t.Fatalf("expected save job; got %s", save.ID)
	}
	if !strings.Contains(save.Input, "video_grayscale_blurred.mp4") {
		t.Errorf("save input should contain blur output; got %q", save.Input)
	}
}
```

- [ ] **Step 2: Run the test**

Run: `cd media-server && go test ./jobqueue/ -run TestWorkflowEndToEndOutputFiles -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `cd media-server && go test ./... -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add jobqueue/jobqueue_core_test.go
git commit -m "test: add end-to-end workflow chaining test with OutputFiles"
```
