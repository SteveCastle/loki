package tasks

import (
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

func waitFn(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx
	for i := 0; i < 5; i++ {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task was canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		case <-time.After(1 * time.Second):
			q.PushJobStdout(j.ID, "Waiting in task...")
		}
	}
	q.CompleteJob(j.ID)
	return nil
}
