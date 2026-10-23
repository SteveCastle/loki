package runners

import (
	"context"
	"errors"
	"fmt"
	"log"
	"runtime/debug"
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
			// The task has returned: a job cancelled mid-run can give back
			// its buckets now (before claiming the next job).
			r.queue.ReleaseCancelled(j.ID)
			r.mu.Lock()
			r.running--
			// After finishing this job, try fetching another one
			r.tryFetchJobAndRun()
			r.mu.Unlock()
		}()
		// A panicking task must fail its job, not take the whole server (and
		// every other running job) down with it.
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("task %s (job %s) panicked: %v\n%s", j.Command, j.ID, rec, debug.Stack())
				_ = r.queue.PushJobStdout(j.ID, fmt.Sprintf("Task panicked: %v", rec))
				_ = r.queue.ErrorJob(j.ID)
			}
		}()

		tasksMap := tasks.GetTasks()
		if task, exists := tasksMap[j.Command]; exists {
			err := task.Fn(j, r.queue, &r.mu)
			// Ensure job state is finalized even if the task forgets: a job
			// left InProgress would hold its concurrency bucket forever.
			defer r.finalizeIfRunning(j.ID, err)
			if err != nil {
				// A graceful pause is not a failure: park the job so it can
				// be resumed with its progress intact.
				if errors.Is(err, jobqueue.ErrPaused) {
					_ = r.queue.PauseJob(j.ID)
					return
				}
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

// finalizeIfRunning settles a job its task returned from without finishing.
// Every terminal transition is a no-op on a job that is no longer InProgress,
// so this only acts when the task really forgot.
func (r *Runners) finalizeIfRunning(id string, taskErr error) {
	snap, _, ok := r.queue.JobSnapshot(id, 0)
	if !ok || snap.State != jobqueue.StateInProgress {
		return
	}
	if taskErr != nil {
		_ = r.queue.ErrorJob(id)
		return
	}
	log.Printf("task %s (job %s) returned without finishing; marking completed", snap.Command, id)
	_ = r.queue.CompleteJob(id)
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
