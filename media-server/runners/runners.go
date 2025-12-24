package runners

import (
	"context"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/tasks"
)

// Runners manages a pool of concurrent job runners.
type Runners struct {
	queue   *jobqueue.Queue
	mu      sync.Mutex
	running int
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
}

// New creates a new Runners instance.
func New(queue *jobqueue.Queue) *Runners {
	ctx, cancel := context.WithCancel(context.Background())
	r := &Runners{
		queue:  queue,
		ctx:    ctx,
		cancel: cancel,
	}

	// Start a goroutine to listen to the signal channel.
	r.wg.Add(1)
	go func() {
		defer r.wg.Done()
		for {
			select {
			case <-r.ctx.Done():
				// Shutdown requested
				return
			case <-r.queue.Signal:
				// When a signal is received, attempt to pick up a new job.
				r.CheckForJobs()
			}
		}
	}()

	return r
}

// Shutdown stops the runners from accepting new jobs and waits for running jobs to complete.
func (r *Runners) Shutdown() {
	// Cancel the context to stop the signal listener
	r.cancel()
	// Wait for the signal listener goroutine to finish
	r.wg.Wait()
}

// CheckForJobs attempts to claim and run a new job if the runners are not at capacity.
// This can be called externally or triggered by signals.
func (r *Runners) CheckForJobs() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.tryFetchJobAndRun()
}

// runJob starts a single job in a separate goroutine. Once it completes,
// we decrement the running count and attempt to fetch the next job.
func (r *Runners) runJob(j *jobqueue.Job) {
	r.running++
	go func() {
		defer func() {
			r.mu.Lock()
			r.running--
			// After finishing this job, try fetching another one
			r.tryFetchJobAndRun()
			r.mu.Unlock()
		}()

		tasksMap := tasks.GetTasks()
		if task, exists := tasksMap[j.Command]; exists {
			// Ensure job state is finalized even if task forgets
			if err := task.Fn(j, r.queue, &r.mu); err != nil {
				// If context is canceled, prefer Cancelled state
				select {
				case <-j.Ctx.Done():
					_ = r.queue.CancelJob(j.ID)
				default:
					_ = r.queue.ErrorJob(j.ID)
				}
			}
		} else {
			// If the task is not found, we should mark the job as failed.
			r.queue.PushJobStdout(j.ID, "Task not found: "+j.Command)
			r.queue.ErrorJob(j.ID)
			return
		}
	}()
}

// tryFetchJobAndRun tries to fetch a new job if capacity allows.
func (r *Runners) tryFetchJobAndRun() {
	job, err := r.queue.ClaimJob()
	if err != nil || job == nil {
		// No job available or error encountered.
		return
	}

	r.runJob(job)
}
