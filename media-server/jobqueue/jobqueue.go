package jobqueue

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/stream"
)

// JobState represents the current state of a job in the queue.
type JobState int

type SerialziedEvent struct {
	UpdateType string `json:"updateType"`
	Job        Job    `json:"job"`
	HTML       string `json:"html"`
}

const (
	StatePending JobState = iota
	StateInProgress
	StateCompleted
	StateCancelled
	StateError
	StatePaused
)

// ErrPaused is returned by task functions that stopped at a pause point in
// response to RequestPause. The runner translates it into PauseJob rather
// than treating it as a failure. All per-item work completed before the pause
// point has already been committed, so a paused job can be resumed (or
// cancelled) without losing progress.
var ErrPaused = errors.New("job paused")

func (s JobState) String() string {
	switch s {
	case StatePending:
		return "Pending"
	case StateInProgress:
		return "InProgress"
	case StateCompleted:
		return "Completed"
	case StateCancelled:
		return "Cancelled"
	case StateError:
		return "Error"
	case StatePaused:
		return "Paused"
	default:
		return "Unknown"
	}
}

// MarshalJSON serializes JobState as a lowercase string for JSON.
func (s JobState) MarshalJSON() ([]byte, error) {
	var str string
	switch s {
	case StatePending:
		str = "pending"
	case StateInProgress:
		str = "in_progress"
	case StateCompleted:
		str = "completed"
	case StateCancelled:
		str = "cancelled"
	case StateError:
		str = "error"
	case StatePaused:
		str = "paused"
	default:
		str = "unknown"
	}
	return json.Marshal(str)
}

// UnmarshalJSON deserializes JobState from a string.
func (s *JobState) UnmarshalJSON(data []byte) error {
	var str string
	if err := json.Unmarshal(data, &str); err != nil {
		return err
	}

	switch str {
	case "pending":
		*s = StatePending
	case "in_progress":
		*s = StateInProgress
	case "completed":
		*s = StateCompleted
	case "cancelled":
		*s = StateCancelled
	case "error":
		*s = StateError
	case "paused":
		*s = StatePaused
	default:
		*s = StatePending
	}
	return nil
}

// Job represents an individual task in the queue.
type Job struct {
	ID            string   `json:"id"` // Unique identifier for the job
	Command       string   `json:"command"`
	Arguments     []string `json:"arguments"`
	Input         string   `json:"input"`
	OriginalInput string   `json:"original_input"`
	Host          string   `json:"host"`
	// Resources are ADDITIONAL concurrency buckets this job occupies besides
	// Host — one per machine resource its work actually consumes (e.g. a
	// combined job running embed+faces ops holds those buckets too, plus the
	// shared local-compute slot). A job is only claimed when EVERY bucket has
	// capacity, and it counts against all of them while running.
	Resources    []string           `json:"resources"`
	Stdout       []string           `json:"-"`
	StdoutRaw    io.Reader          `json:"-"` // Raw stdout stream
	StdIn        io.Reader          `json:"-"`
	Dependencies []string           `json:"dependencies"` // IDs of jobs that must complete before this one
	State        JobState           `json:"state"`
	Ctx          context.Context    `json:"-"`
	Cancel       context.CancelFunc `json:"-"`

	// Timestamps for various states
	CreatedAt   time.Time `json:"created_at"`
	ClaimedAt   time.Time `json:"claimed_at"`
	CompletedAt time.Time `json:"completed_at"`
	ErroredAt   time.Time `json:"errored_at"`

	// Workflow chaining
	OutputFiles []string `json:"output_files"` // File paths registered for downstream consumption
	SourceFiles []string `json:"source_files"` // Parallel to OutputFiles: the original source for each output
	WorkflowID  string   `json:"workflow_id"`  // Non-empty when job is part of a workflow

	// Item-level progress. Total is set once the job's input/query resolves to
	// a concrete item list; Done advances as items finish (including skips).
	// Both survive restarts so a resumed job renders where it left off.
	ProgressDone  int `json:"progress_done"`
	ProgressTotal int `json:"progress_total"`

	// Throttling bookkeeping for progress broadcasts/persists (not serialized).
	lastProgressBroadcast time.Time
	lastProgressPersist   time.Time
}

type WorkflowTask struct {
	ID           string   `json:"id"` // Internal ID for linking dependencies
	Command      string   `json:"command"`
	Arguments    []string `json:"arguments"`
	Input        string   `json:"input"`
	Dependencies []string `json:"dependencies"` // IDs of other tasks in this workflow
	PosX         float64  `json:"pos_x,omitempty"`
	PosY         float64  `json:"pos_y,omitempty"`
}

type Workflow struct {
	Tasks      []WorkflowTask `json:"tasks"`
	WorkflowID string         `json:"workflow_id"`
}

// Queue is a thread-safe structure that manages Jobs with dependencies.
type Queue struct {
	mu            sync.Mutex
	Jobs          map[string]*Job
	JobOrder      []string // Keep track of the order in which jobs are added
	Signal        chan string
	Db            *sql.DB // Database connection for persistence
	HostLimits    map[string]int
	RunningCounts map[string]int
	// pauseRequests holds job IDs asked to pause. Task item loops poll
	// PauseRequested between items and stop gracefully (returning ErrPaused)
	// so the current item's writes always land before the job parks.
	pauseRequests map[string]struct{}
}

// NewQueue initializes and returns a new Queue.
func NewQueue() *Queue {
	return &Queue{
		Jobs:          make(map[string]*Job),
		Signal:        make(chan string, 100),
		HostLimits:    make(map[string]int),
		RunningCounts: make(map[string]int),
		pauseRequests: make(map[string]struct{}),
	}
}

// NewQueueWithDB initializes and returns a new Queue with database support.
func NewQueueWithDB(db *sql.DB) *Queue {
	q := &Queue{
		Jobs:          make(map[string]*Job),
		Signal:        make(chan string, 100),
		Db:            db,
		HostLimits:    make(map[string]int),
		RunningCounts: make(map[string]int),
		pauseRequests: make(map[string]struct{}),
	}

	// Create the jobs table if it doesn't exist
	if err := q.createJobsTable(); err != nil {
		log.Printf("Failed to create jobs table: %v", err)
	}

	if err := q.createWorkflowsTable(); err != nil {
		log.Printf("Failed to create workflows table: %v", err)
	}

	// Load existing jobs from database
	if err := q.loadJobsFromDB(); err != nil {
		log.Printf("Failed to load jobs from database: %v", err)
	}

	return q
}

// createJobsTable creates the jobs table if it doesn't exist
func (q *Queue) createJobsTable() error {
	query := `
	CREATE TABLE IF NOT EXISTS jobs (
		id TEXT PRIMARY KEY,
		command TEXT NOT NULL,
		arguments TEXT, -- JSON array
		input TEXT,
		original_input TEXT,
		host TEXT,
		stdout TEXT, -- JSON array
		dependencies TEXT, -- JSON array
		state INTEGER NOT NULL,
		created_at DATETIME NOT NULL,
		claimed_at DATETIME,
		completed_at DATETIME,
		errored_at DATETIME,
		job_order_position INTEGER
	)`

	_, err := q.Db.Exec(query)
	if err != nil {
		return err
	}

	// Try to add host column if it doesn't exist (migration)
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN host TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN original_input TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN output_files TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN source_files TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN workflow_id TEXT")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN progress_done INTEGER")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN progress_total INTEGER")
	_, _ = q.Db.Exec("ALTER TABLE jobs ADD COLUMN resources TEXT")

	return nil
}

// saveJobToDB saves a single job to the database
func (q *Queue) saveJobToDB(job *Job) error {
	if q.Db == nil {
		return nil // No database connection
	}

	// Serialize arrays to JSON
	argumentsJSON, _ := json.Marshal(job.Arguments)
	stdoutJSON, _ := json.Marshal(job.Stdout)
	dependenciesJSON, _ := json.Marshal(job.Dependencies)
	outputFilesJSON, _ := json.Marshal(job.OutputFiles)
	sourceFilesJSON, _ := json.Marshal(job.SourceFiles)
	resourcesJSON, _ := json.Marshal(job.Resources)

	// Find position in job order
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
		output_files, source_files, workflow_id, progress_done, progress_total, resources
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

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
		string(sourceFilesJSON),
		job.WorkflowID,
		job.ProgressDone,
		job.ProgressTotal,
		string(resourcesJSON),
	)

	return err
}

// loadJobsFromDB loads all jobs from the database
func (q *Queue) loadJobsFromDB() error {
	if q.Db == nil {
		return nil // No database connection
	}

	query := `
	SELECT id, command, arguments, input, COALESCE(original_input, ''), COALESCE(host, ''), stdout, dependencies, state,
		   created_at, claimed_at, completed_at, errored_at, job_order_position,
		   COALESCE(output_files, '[]'), COALESCE(source_files, '[]'), COALESCE(workflow_id, ''),
		   COALESCE(progress_done, 0), COALESCE(progress_total, 0), COALESCE(resources, '')
	FROM jobs
	ORDER BY job_order_position`

	rows, err := q.Db.Query(query)
	if err != nil {
		return err
	}
	defer rows.Close()

	var resumedJobs []string

	for rows.Next() {
		var job Job
		var argumentsJSON, stdoutJSON, dependenciesJSON, outputFilesJSON, sourceFilesJSON, resourcesJSON string
		var state int
		var position int

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
			&sourceFilesJSON,
			&job.WorkflowID,
			&job.ProgressDone,
			&job.ProgressTotal,
			&resourcesJSON,
		)
		if err != nil {
			log.Printf("Error scanning job row: %v", err)
			continue
		}

		// Deserialize JSON arrays
		if err := json.Unmarshal([]byte(argumentsJSON), &job.Arguments); err != nil {
			job.Arguments = []string{}
		}
		if err := json.Unmarshal([]byte(stdoutJSON), &job.Stdout); err != nil {
			job.Stdout = []string{}
		}
		if err := json.Unmarshal([]byte(dependenciesJSON), &job.Dependencies); err != nil {
			job.Dependencies = []string{}
		}
		if err := json.Unmarshal([]byte(outputFilesJSON), &job.OutputFiles); err != nil {
			job.OutputFiles = []string{}
		}
		if err := json.Unmarshal([]byte(sourceFilesJSON), &job.SourceFiles); err != nil {
			job.SourceFiles = []string{}
		}
		if err := json.Unmarshal([]byte(resourcesJSON), &job.Resources); err != nil {
			job.Resources = nil
		}

		job.State = JobState(state)

		if job.Host == "" {
			job.Host = getHost(job.Command, job.Input)
		}
		// Legacy rows (or rows written before a resolver was registered) have
		// no resources; re-resolve so old queued jobs still respect the
		// machine-wide caps after an upgrade.
		if len(job.Resources) == 0 {
			job.Resources = getResources(job.Command, job.Arguments, job.Input)
		}

		// If job was in progress, reset it to pending so it can be resumed
		if job.State == StateInProgress {
			job.State = StatePending
			job.ClaimedAt = time.Time{} // Reset claimed time
			resumedJobs = append(resumedJobs, job.ID)
		}

		// Recreate context and cancel function
		ctx, cancel := context.WithCancel(context.Background())
		job.Ctx = ctx
		job.Cancel = cancel

		q.Jobs[job.ID] = &job
		q.JobOrder = append(q.JobOrder, job.ID)
	}

	if len(resumedJobs) > 0 {
		log.Printf("Resumed %d jobs that were in progress: %v", len(resumedJobs), resumedJobs)
		// Signal that jobs are available
		for _, jobID := range resumedJobs {
			select {
			case q.Signal <- jobID:
			default:
				// Channel full, skip
			}
		}
	}

	return rows.Err()
}

// removeJobFromDB removes a job from the database
func (q *Queue) removeJobFromDB(jobID string) error {
	if q.Db == nil {
		return nil // No database connection
	}

	_, err := q.Db.Exec("DELETE FROM jobs WHERE id = ?", jobID)
	return err
}

// SaveAllJobsToDB saves all current jobs to the database
func (q *Queue) SaveAllJobsToDB() error {
	if q.Db == nil {
		return nil // No database connection
	}

	q.mu.Lock()
	defer q.mu.Unlock()

	for _, job := range q.Jobs {
		if err := q.saveJobToDB(job); err != nil {
			log.Printf("Failed to save job %s to database: %v", job.ID, err)
		}
	}

	return nil
}

// AddJob adds a new job to the queue with the given dependencies.
// It generates a UUID for the job if not provided and returns it.
func (q *Queue) AddJob(id string, command string, arguments []string, input string, dependencies []string) (string, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	if id == "" {
		id = uuid.NewString()
	}
	if _, exists := q.Jobs[id]; exists {
		// Extremely unlikely to happen due to UUID uniqueness,
		// but we check for completeness.
		return "", errors.New("job with given ID already exists")
	}

	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{
		ID:            id,
		Input:         input,
		OriginalInput: input,
		Command:       command,
		Arguments:     arguments,
		Dependencies:  dependencies,
		State:         StatePending,
		Ctx:           ctx,
		Cancel:        cancel,
		CreatedAt:     time.Now(),
		Host:          getHost(command, input),
		Resources:     getResources(command, arguments, input),
	}
	q.Jobs[id] = job
	q.JobOrder = append(q.JobOrder, id)

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job to database: %v", err)
	}

	// Broadcast the new job to the Signal channel
	q.Signal <- id
	error := serializeListUpdate("create", job)
	if error != nil {
		return "", error
	}

	return id, nil
}

// Adds each job from the workflow.
// Does not acquire lock, so must be called from a function that does.
func (q *Queue) AddWorkflow(w Workflow) ([]string, error) {
	var jobIDs []string

	workflowID := w.WorkflowID
	if workflowID == "" {
		workflowID = uuid.NewString()
	}

	for _, task := range w.Tasks {
		// Ensure dependencies exist (basic check, could be improved)
		// Since we process a list, we assume the client sends a valid DAG or at least valid IDs.
		// If IDs are not provided by client, they should be generated there or here.
		// However, for DAG dependencies to work, IDs must be known.
		// So we assume the client provides IDs for tasks that are dependencies.
		// Or we could generate them here if not provided, but linking them up requires knowing which is which.
		// Let's assume the Workflow struct comes with pre-linked IDs if there are deps.
		// If ID is empty, AddJob will generate one, but then nothing can depend on it unless we return it.

		id, err := q.AddJob(task.ID, task.Command, task.Arguments, task.Input, task.Dependencies)
		if err != nil {
			return jobIDs, err
		}
		// Set WorkflowID on the newly created job
		q.mu.Lock()
		if job, ok := q.Jobs[id]; ok {
			job.WorkflowID = workflowID
			if err := q.saveJobToDB(job); err != nil {
				log.Printf("Failed to save workflow ID to database: %v", err)
			}
		}
		q.mu.Unlock()
		jobIDs = append(jobIDs, id)
	}
	return jobIDs, nil
}

func (q *Queue) CopyJob(id string) (string, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return "", errors.New("job not found")
	}

	newID := uuid.NewString()
	if _, exists := q.Jobs[newID]; exists {
		return "", errors.New("job with given ID already exists")
	}
	ctx, cancel := context.WithCancel(context.Background())

	newJob := *job
	newJob.ID = newID
	newJob.Stdout = []string{}
	newJob.OutputFiles = []string{}
	newJob.SourceFiles = []string{}
	newJob.State = StatePending
	newJob.ProgressDone = 0
	newJob.ProgressTotal = 0
	newJob.CreatedAt = time.Now()
	newJob.ClaimedAt = time.Time{}
	newJob.CompletedAt = time.Time{}
	newJob.ErroredAt = time.Time{}
	newJob.Cancel = cancel
	newJob.Ctx = ctx
	newJob.OriginalInput = job.OriginalInput
	// Host is copied from original job

	q.Jobs[newID] = &newJob
	q.JobOrder = append(q.JobOrder, newID)

	// Save to database
	if err := q.saveJobToDB(&newJob); err != nil {
		log.Printf("Failed to save copied job to database: %v", err)
	}

	// Broadcast the new job to the Signal channel
	q.Signal <- newID
	error := serializeListUpdate("create", &newJob)
	if error != nil {
		return "", error
	}

	return newID, nil
}

// ClaimJob tries to find a pending job whose dependencies are all completed,
// in FIFO order. If successful, it returns the job and marks it as InProgress.
// If no suitable job is found, it returns nil and no error.
func (q *Queue) ClaimJob() (*Job, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	for _, jobID := range q.JobOrder {
		job := q.Jobs[jobID]
		if job.State == StatePending && q.canClaim(job) {
			// Every bucket the job occupies (host + resources) must have a
			// free slot — resource-heavy jobs hold all the buckets their work
			// actually uses.
			if !q.hasCapacityLocked(job) {
				continue
			}

			job.State = StateInProgress
			job.ClaimedAt = time.Now()
			q.incRunningLocked(job)

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

			// Save to database
			if err := q.saveJobToDB(job); err != nil {
				log.Printf("Failed to save job state to database: %v", err)
			}

			err := serializeListUpdate("update", job)
			if err != nil {
				return nil, err
			}
			return job, nil
		}
	}

	// No claimable job found
	return nil, nil
}

// canClaim checks if a job's dependencies are all completed.
func (q *Queue) canClaim(job *Job) bool {
	for _, dep := range job.Dependencies {
		depJob, exists := q.Jobs[dep]
		if !exists {
			// If dependency doesn't exist, can't claim
			return false
		}
		if depJob.State != StateCompleted {
			// If any dependency is not completed, can't claim
			return false
		}
	}
	return true
}

// ErrorJob sets a job's state to error if it is currently in progress.
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
	q.decRunningLocked(job)
	delete(q.pauseRequests, id)

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job error state to database: %v", err)
	}

	err := serializeListUpdate("update", job)
	if err != nil {
		return nil
	}

	// Cancel pending dependents in the same workflow
	if job.WorkflowID != "" {
		q.cancelWorkflowDependentsLocked(id, job.WorkflowID)
	}

	return nil
}

// cancelWorkflowDependentsLocked cancels all pending jobs in the workflow
// that transitively depend on the errored job. Must be called with mu held.
func (q *Queue) cancelWorkflowDependentsLocked(erroredID, workflowID string) {
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

// CancelJob sets a job's state to cancelled if it is currently pending,
// in progress, or paused.
func (q *Queue) CancelJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State != StatePending && job.State != StateInProgress && job.State != StatePaused {
		return errors.New("job is not pending, in progress, or paused, cannot cancel")
	}
	job.Cancel()

	if job.State == StateInProgress {
		q.decRunningLocked(job)
	}

	job.State = StateCancelled
	delete(q.pauseRequests, id)

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job cancellation to database: %v", err)
	}

	err := serializeListUpdate("update", job)
	if err != nil {
		return err
	}

	return nil
}

// RequestPause asks a running or pending job to pause. A pending job pauses
// immediately (it just stops being claimable); an in-progress job keeps
// running until its task polls PauseRequested at the next item boundary and
// returns ErrPaused — so the current item's writes always complete first.
func (q *Queue) RequestPause(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	switch job.State {
	case StatePending:
		return q.pauseJobLocked(job)
	case StateInProgress:
		if q.pauseRequests == nil {
			q.pauseRequests = make(map[string]struct{})
		}
		q.pauseRequests[id] = struct{}{}
		return nil
	default:
		return errors.New("job is not pending or in progress, cannot pause")
	}
}

// PauseRequested reports whether a pause has been requested for the job.
// Task item loops poll this between items.
func (q *Queue) PauseRequested(id string) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	_, ok := q.pauseRequests[id]
	return ok
}

// PauseJob transitions an in-progress (or pending) job to Paused. Called by
// the runner when a task returns ErrPaused. Progress and all per-item writes
// are preserved; ResumeJob re-queues the job.
func (q *Queue) PauseJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}
	if job.State != StateInProgress && job.State != StatePending {
		return errors.New("job is not in progress or pending, cannot pause")
	}
	return q.pauseJobLocked(job)
}

// pauseJobLocked performs the Paused transition. Must be called with mu held.
func (q *Queue) pauseJobLocked(job *Job) error {
	if job.State == StateInProgress {
		q.decRunningLocked(job)
	}
	job.State = StatePaused
	delete(q.pauseRequests, job.ID)

	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job pause to database: %v", err)
	}
	return serializeListUpdate("update", job)
}

// ResumeJob re-queues a paused job. The task re-resolves its input when
// re-claimed; already-processed items are skipped via each task's
// skip-existing checks (and best-effort progress offsets for overwrite runs).
func (q *Queue) ResumeJob(id string) error {
	q.mu.Lock()

	job, exists := q.Jobs[id]
	if !exists {
		q.mu.Unlock()
		return errors.New("job not found")
	}
	if job.State != StatePaused {
		q.mu.Unlock()
		return errors.New("job is not paused, cannot resume")
	}

	job.State = StatePending
	job.ClaimedAt = time.Time{}
	delete(q.pauseRequests, id)

	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job resume to database: %v", err)
	}
	err := serializeListUpdate("update", job)
	q.mu.Unlock()

	// Signal outside the lock: ClaimJob (triggered by the runner) takes mu.
	select {
	case q.Signal <- id:
	default:
	}
	return err
}

// progressBroadcastInterval throttles per-item progress SSE broadcasts and DB
// persists — a fast op (hashing) can finish thousands of items per second and
// must not turn each into a broadcast plus a row rewrite. First and final
// updates always go out.
const progressBroadcastInterval = 250 * time.Millisecond

// SerializedProgress is the payload of the "progress" SSE event.
type SerializedProgress struct {
	UpdateType string `json:"updateType"` // always "progress"
	ID         string `json:"id"`
	Done       int    `json:"done"`
	Total      int    `json:"total"`
}

// SetJobProgress records item-level progress for a job and broadcasts it as a
// "progress" SSE event (throttled). Tasks call it once when the input/query
// resolves (done=0, total=N) and again as each item finishes.
func (q *Queue) SetJobProgress(id string, done, total int) error {
	q.mu.Lock()

	job, exists := q.Jobs[id]
	if !exists {
		q.mu.Unlock()
		return errors.New("job not found")
	}

	job.ProgressDone = done
	job.ProgressTotal = total

	now := time.Now()
	final := total > 0 && done >= total
	broadcast := final || done == 0 || now.Sub(job.lastProgressBroadcast) >= progressBroadcastInterval
	persist := final || done == 0 || now.Sub(job.lastProgressPersist) >= progressBroadcastInterval
	if broadcast {
		job.lastProgressBroadcast = now
	}
	if persist {
		job.lastProgressPersist = now
		// Targeted UPDATE: a full saveJobToDB rewrites the whole row (including
		// the stdout JSON blob) which is far too heavy per item.
		if q.Db != nil {
			if _, err := q.Db.Exec(`UPDATE jobs SET progress_done = ?, progress_total = ? WHERE id = ?`, done, total, id); err != nil {
				log.Printf("Failed to persist job progress: %v", err)
			}
		}
	}
	q.mu.Unlock()

	if !broadcast {
		return nil
	}
	payload, err := json.Marshal(SerializedProgress{
		UpdateType: "progress",
		ID:         id,
		Done:       done,
		Total:      total,
	})
	if err != nil {
		return err
	}
	stream.Broadcast(stream.Message{Type: "progress", Msg: string(payload)})
	return nil
}

// UpdateJobStdout updates the job's stdout with the given string.
func (q *Queue) PushJobStdout(id string, stdout string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	job.Stdout = append(job.Stdout, stdout)

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job stdout to database: %v", err)
	}

	err := serializeStdout(stdout, id)
	if err != nil {
		return nil
	}
	return nil
}

// RegisterOutputFile appends a file path to the job's OutputFiles list.
// These paths are used by ClaimJob to construct input for downstream jobs,
// keeping file paths separate from diagnostic stdout.
// RegisterOutputFile appends a file path to the job's OutputFiles list.
// An optional source argument records which original input file produced
// this output (kept in parallel in SourceFiles). Pass the source when the
// output was derived from a specific input so downstream tasks like "save"
// can recover the original filename.
func (q *Queue) RegisterOutputFile(id string, path string, source ...string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	job.OutputFiles = append(job.OutputFiles, path)
	src := ""
	if len(source) > 0 {
		src = source[0]
	}
	job.SourceFiles = append(job.SourceFiles, src)

	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job output files to database: %v", err)
	}

	return nil
}

// CompleteJob marks the specified job as completed if it is currently InProgress.
// Returns an error if the job does not exist, or if it's not in a valid state to be completed.
func (q *Queue) CompleteJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State != StateInProgress {
		return errors.New("job is not in progress, cannot complete")
	}

	job.State = StateCompleted
	job.CompletedAt = time.Now()
	q.decRunningLocked(job)
	delete(q.pauseRequests, id)

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job completion to database: %v", err)
	}

	err := serializeListUpdate("update", job)
	if err != nil {
		return nil
	}
	return nil
}

// GetJobs returns a slice of all jobs in the queue sorted by CreatedAt time in descending order.

func (q *Queue) GetJobs() []Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	length := len(q.Jobs)
	jobs := make([]Job, 0, length)
	for i := length - 1; i >= 0; i-- {
		jobs = append(jobs, *q.Jobs[q.JobOrder[i]])
	}
	return jobs
}

func (q *Queue) GetJob(id string) *Job {
	q.mu.Lock()
	defer q.mu.Unlock()
	job, exists := q.Jobs[id]
	if !exists {
		return nil
	}
	return job
}

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

// GetSourceMap returns a map from output path to the ultimate original
// source path, resolved through the entire workflow chain. For each
// output of the given job's parents, it follows SourceFiles pointers
// back through ancestor jobs until it reaches a source that isn't itself
// an output of any job in the workflow.
func (q *Queue) GetSourceMap(jobID string) map[string]string {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, ok := q.Jobs[jobID]
	if !ok {
		return nil
	}

	// Build a global output->source index across all workflow jobs.
	outputToSource := make(map[string]string)
	for _, j := range q.Jobs {
		if j.WorkflowID != job.WorkflowID || job.WorkflowID == "" {
			continue
		}
		for i, out := range j.OutputFiles {
			if i < len(j.SourceFiles) && j.SourceFiles[i] != "" {
				outputToSource[out] = j.SourceFiles[i]
			}
		}
	}

	// For each output of the direct parents, resolve to the original source.
	m := make(map[string]string)
	for _, depID := range job.Dependencies {
		dep, ok := q.Jobs[depID]
		if !ok {
			continue
		}
		for i, out := range dep.OutputFiles {
			src := out
			if i < len(dep.SourceFiles) && dep.SourceFiles[i] != "" {
				src = dep.SourceFiles[i]
			}
			// Walk back through the chain to the original.
			for depth := 0; depth < 100; depth++ {
				prev, ok := outputToSource[src]
				if !ok || prev == src {
					break
				}
				src = prev
			}
			m[out] = src
		}
	}
	return m
}

func (q *Queue) RemoveJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State == StateInProgress {
		q.decRunningLocked(job)
	}

	delete(q.pauseRequests, id)
	delete(q.Jobs, id)
	for i, jobId := range q.JobOrder {
		if jobId == id {
			q.JobOrder = append(q.JobOrder[:i], q.JobOrder[i+1:]...)
			break
		}
	}

	// Remove from database
	if err := q.removeJobFromDB(id); err != nil {
		log.Printf("Failed to remove job from database: %v", err)
	}

	err := serializeListUpdate("delete", &Job{ID: id})
	if err != nil {
		return err
	}
	return nil
}

// ClearNonRunningJobs removes all jobs that are not currently running (StateInProgress).
// This includes jobs in states: Pending, Completed, Cancelled, and Error.
// Returns the number of jobs cleared and any error that occurred.
func (q *Queue) ClearNonRunningJobs() (int, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	var clearedCount int
	var jobsToRemove []string

	// Collect IDs of jobs to remove (not currently running)
	for _, jobID := range q.JobOrder {
		job := q.Jobs[jobID]
		if job.State != StateInProgress {
			jobsToRemove = append(jobsToRemove, jobID)
		}
	}

	// Remove the jobs
	for _, jobID := range jobsToRemove {
		delete(q.Jobs, jobID)

		// Remove from job order
		for i, id := range q.JobOrder {
			if id == jobID {
				q.JobOrder = append(q.JobOrder[:i], q.JobOrder[i+1:]...)
				break
			}
		}

		// Remove from database
		if err := q.removeJobFromDB(jobID); err != nil {
			log.Printf("Failed to remove job %s from database: %v", jobID, err)
		}

		// Broadcast the delete event
		err := serializeListUpdate("delete", &Job{ID: jobID})
		if err != nil {
			return clearedCount, err
		}
		clearedCount++
	}

	return clearedCount, nil
}

type SerializedJob struct {
	UpdateType string `json:"updateType"`
	Job        Job    `json:"job"`
	HTML       string `json:"html"`
}

type SerializedStdout struct {
	UpdateType string `json:"updateType"`
	Line       string `json:"line"`
}

// serializeListUpdate serializes the given job and broadcasts it with the specified update type.
// It returns an error if template execution or JSON marshalling fails.
func serializeListUpdate(updateType string, job *Job) error {
	var html bytes.Buffer
	if err := renderer.Templates().ExecuteTemplate(&html, "jobRow", job); err != nil {
		return fmt.Errorf("error executing template: %v", err)
	}

	serializedEvent := SerializedJob{
		UpdateType: updateType,
		Job:        *job,
		HTML:       html.String(),
	}
	j, err := json.Marshal(serializedEvent)
	if err != nil {
		return fmt.Errorf("error marshalling event: %v", err)
	}

	stream.Broadcast(stream.Message{Type: updateType, Msg: string(j)})
	return nil
}

func serializeStdout(line string, id string) error {
	serializedEvent := SerializedStdout{
		UpdateType: "stdout",
		Line:       line,
	}

	j, err := json.Marshal(serializedEvent)
	if err != nil {
		return fmt.Errorf("error marshalling event: %v", err)
	}
	//Type should be in the format `stdout-<job-id>`
	stream.Broadcast(stream.Message{Type: "stdout-" + id, Msg: string(j)})
	return nil
}

// Helper methods

// HostResolverFunc maps a job's (command, input) pair to a concurrency
// bucket name. The bucket is then governed by HostLimits / RunningCounts;
// see ClaimJob. Inject one with SetHostResolver to keep task-specific
// policy out of jobqueue. The default resolver preserves the previous
// inline behavior so packages that don't override get the same result.
type HostResolverFunc func(command, input string) string

var hostResolver HostResolverFunc = defaultHostResolver

// SetHostResolver replaces the host-bucket resolver. Call once at startup
// (before any AddJob / loadJobsFromDB), typically with the tasks package's
// registry so per-task host policy stays alongside the tasks themselves.
func SetHostResolver(fn HostResolverFunc) {
	if fn == nil {
		hostResolver = defaultHostResolver
		return
	}
	hostResolver = fn
}

// defaultHostResolver mirrors the historical getHost behavior: URL hostname
// for "ingest", "localhost" for everything else. Kept here so existing
// jobqueue tests and any consumer that doesn't register a resolver still
// see the same buckets as before.
func defaultHostResolver(command, input string) string {
	if command == "ingest" {
		u, err := url.Parse(input)
		if err == nil && u.Host != "" {
			return strings.TrimPrefix(u.Hostname(), "www.")
		}
	}
	return "localhost"
}

// getHost is retained as a tiny shim so the existing internal call sites
// don't need to change shape.
func getHost(command, input string) string {
	return hostResolver(command, input)
}

// ResourceResolverFunc maps a job to the ADDITIONAL concurrency buckets it
// occupies besides its Host — one per machine resource the work consumes
// (GPU-bound model pools, the shared local-compute slot, ...). Arguments are
// included because composite jobs (e.g. `process --ops=...`) declare their
// workload there.
type ResourceResolverFunc func(command string, arguments []string, input string) []string

var resourceResolver ResourceResolverFunc

// SetResourceResolver installs the resource resolver. Call once at startup
// (alongside SetHostResolver, before any AddJob / loadJobsFromDB). nil
// disables it — jobs then occupy only their Host bucket, the historical
// behavior.
func SetResourceResolver(fn ResourceResolverFunc) {
	resourceResolver = fn
}

func getResources(command string, arguments []string, input string) []string {
	if resourceResolver == nil {
		return nil
	}
	return resourceResolver(command, arguments, input)
}

// jobBuckets returns the deduplicated set of concurrency buckets a job
// occupies: its Host plus any resolved Resources.
func jobBuckets(job *Job) []string {
	out := make([]string, 0, 1+len(job.Resources))
	seen := make(map[string]struct{}, 1+len(job.Resources))
	for _, b := range append([]string{job.Host}, job.Resources...) {
		if b == "" {
			continue
		}
		if _, dup := seen[b]; dup {
			continue
		}
		seen[b] = struct{}{}
		out = append(out, b)
	}
	return out
}

// hasCapacityLocked reports whether EVERY bucket the job occupies has a free
// slot. Must be called with mu held.
func (q *Queue) hasCapacityLocked(job *Job) bool {
	for _, b := range jobBuckets(job) {
		if q.RunningCounts[b] >= q.getHostLimitLocked(b) {
			return false
		}
	}
	return true
}

// incRunningLocked / decRunningLocked adjust the running counts for every
// bucket the job occupies. Must be called with mu held, exactly once per
// InProgress transition in each direction — all state transitions go through
// these so multi-bucket jobs can't leak capacity.
func (q *Queue) incRunningLocked(job *Job) {
	for _, b := range jobBuckets(job) {
		q.RunningCounts[b]++
	}
}

func (q *Queue) decRunningLocked(job *Job) {
	for _, b := range jobBuckets(job) {
		q.RunningCounts[b]--
	}
}

func (q *Queue) getHostLimitLocked(host string) int {
	if limit, ok := q.HostLimits[host]; ok {
		return limit
	}
	// Default limit for all hosts (including localhost)
	return 1
}

func (q *Queue) SetHostLimit(host string, limit int) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.HostLimits[host] = limit
}
