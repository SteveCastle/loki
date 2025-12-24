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
)

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
	default:
		*s = StatePending
	}
	return nil
}

// Job represents an individual task in the queue.
type Job struct {
	ID           string             `json:"id"` // Unique identifier for the job
	Command      string             `json:"command"`
	Arguments    []string           `json:"arguments"`
	Input        string             `json:"input"`
	Host         string             `json:"host"`
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
}

type Workflow struct {
	Command   string `json:"command"`
	Arguments []string
	Input     string     `json:"input"`
	Children  []Workflow `json:"children"`
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
}

// NewQueue initializes and returns a new Queue.
func NewQueue() *Queue {
	return &Queue{
		Jobs:          make(map[string]*Job),
		Signal:        make(chan string, 100),
		HostLimits:    make(map[string]int),
		RunningCounts: make(map[string]int),
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
	}

	// Create the jobs table if it doesn't exist
	if err := q.createJobsTable(); err != nil {
		log.Printf("Failed to create jobs table: %v", err)
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
		id, command, arguments, input, host, stdout, dependencies, state,
		created_at, claimed_at, completed_at, errored_at, job_order_position
	) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	_, err := q.Db.Exec(query,
		job.ID,
		job.Command,
		string(argumentsJSON),
		job.Input,
		job.Host,
		string(stdoutJSON),
		string(dependenciesJSON),
		int(job.State),
		job.CreatedAt,
		job.ClaimedAt,
		job.CompletedAt,
		job.ErroredAt,
		position,
	)

	return err
}

// loadJobsFromDB loads all jobs from the database
func (q *Queue) loadJobsFromDB() error {
	if q.Db == nil {
		return nil // No database connection
	}

	query := `
	SELECT id, command, arguments, input, COALESCE(host, ''), stdout, dependencies, state,
		   created_at, claimed_at, completed_at, errored_at, job_order_position
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
		var argumentsJSON, stdoutJSON, dependenciesJSON string
		var state int
		var position int

		err := rows.Scan(
			&job.ID,
			&job.Command,
			&argumentsJSON,
			&job.Input,
			&job.Host,
			&stdoutJSON,
			&dependenciesJSON,
			&state,
			&job.CreatedAt,
			&job.ClaimedAt,
			&job.CompletedAt,
			&job.ErroredAt,
			&position,
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

		job.State = JobState(state)

		if job.Host == "" {
			job.Host = getHost(job.Command, job.Input)
		}

		// If job was in progress, reset it to pending so it can be resumed
		if job.State == StateInProgress {
			job.State = StatePending
			job.ClaimedAt = time.Time{} // Reset claimed time
			resumedJobs = append(resumedJobs, job.ID)
		} else if job.State == StateInProgress {
			// This branch is unreachable due to above logic resetting it,
			// but if we were to keep it running, we'd update RunningCounts here.
			// Since we reset to Pending, we don't increment RunningCounts yet.
			// They will be picked up by ClaimJob.
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
// It generates a UUID for the job and returns it.
func (q *Queue) AddJob(command string, arguments []string, input string, dependencies []string) (string, error) {
	q.mu.Lock()
	defer q.mu.Unlock()
	id := uuid.NewString()
	if _, exists := q.Jobs[id]; exists {
		// Extremely unlikely to happen due to UUID uniqueness,
		// but we check for completeness.
		return "", errors.New("job with given ID already exists")
	}

	ctx, cancel := context.WithCancel(context.Background())
	job := &Job{
		ID:           id,
		Input:        input,
		Command:      command,
		Arguments:    arguments,
		Dependencies: dependencies,
		State:        StatePending,
		Ctx:          ctx,
		Cancel:       cancel,
		CreatedAt:    time.Now(),
		Host:         getHost(command, input),
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

// Recurses through the workflow and adds each job from the bottom up, adding dependencies as it goes.
// Dpes not acquire lock, so must be called from a function that does.
func (q *Queue) AddWorkflow(w Workflow) (string, error) {
	// Add all the children and accumulate thier ids for the dependencies

	dependencies := []string{}

	for _, child := range w.Children {
		id, err := q.AddWorkflow(child)
		if err != nil {
			return "", err
		}
		dependencies = append(dependencies, id)
	}
	return q.AddJob(w.Command, w.Arguments, w.Input, dependencies)
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
	newJob.State = StatePending
	newJob.CreatedAt = time.Now()
	newJob.ClaimedAt = time.Time{}
	newJob.CompletedAt = time.Time{}
	newJob.ErroredAt = time.Time{}
	newJob.Cancel = cancel
	newJob.Ctx = ctx
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
			// Check host limits
			limit := q.getHostLimitLocked(job.Host)
			if q.RunningCounts[job.Host] >= limit {
				continue
			}

			job.State = StateInProgress
			job.ClaimedAt = time.Now()
			q.RunningCounts[job.Host]++

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
	q.RunningCounts[job.Host]--

	// Save to database
	if err := q.saveJobToDB(job); err != nil {
		log.Printf("Failed to save job error state to database: %v", err)
	}

	err := serializeListUpdate("update", job)
	if err != nil {
		return nil
	}
	return nil
}

// CancelJob sets a job's state to cancelled if it is currently pending.
func (q *Queue) CancelJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State != StatePending && job.State != StateInProgress {
		return errors.New("job is not pending, or in progree, cannot cancel")
	}
	job.Cancel()

	if job.State == StateInProgress {
		q.RunningCounts[job.Host]--
	}

	job.State = StateCancelled

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
	q.RunningCounts[job.Host]--

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

func (q *Queue) RemoveJob(id string) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	job, exists := q.Jobs[id]
	if !exists {
		return errors.New("job not found")
	}

	if job.State == StateInProgress {
		q.RunningCounts[job.Host]--
	}

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

func getHost(command, input string) string {
	if command == "ingest" {
		u, err := url.Parse(input)
		if err == nil && u.Host != "" {
			return strings.TrimPrefix(u.Hostname(), "www.")
		}
	}
	return "localhost"
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
