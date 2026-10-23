package jobqueue

import (
	"log"
	"time"
)

// Deferred persistence for a job's append-only fields (stdout lines and
// output files).
//
// saveJobToDB rewrites the WHOLE job row: it re-marshals every stdout line
// and every output path. Doing that per appended line or per output made a
// chatty or large job quadratic (a process run over 1M items rewrote a row
// that grew to ~100 MB once per item) while holding q.mu, freezing every
// queue call and HTTP handler. Appends now only mark the job dirty, and a
// background flusher persists dirty jobs periodically. Every state
// transition still saves synchronously (which also clears the dirty mark),
// so terminal state and the full output list are always durable; only the
// last moments of a running job's log can be lost to a hard crash.

const (
	// dirtyFlushInterval is how often running jobs' appended stdout and
	// outputs reach the database.
	dirtyFlushInterval = time.Second
	// largeJobFlushInterval spaces out rewrites of a job whose row is big,
	// since each flush still re-serializes all of it.
	largeJobFlushInterval = 30 * time.Second
	largeJobThreshold     = 10000 // stdout lines + output files
)

func (q *Queue) markDirtyLocked(id string) {
	if q.Db == nil {
		return
	}
	if q.dirty == nil {
		q.dirty = make(map[string]struct{})
	}
	q.dirty[id] = struct{}{}
}

// noteSavedLocked is called by saveJobToDB: the row now holds everything.
func (q *Queue) noteSavedLocked(id string) {
	delete(q.dirty, id)
	if q.lastSaved == nil {
		q.lastSaved = make(map[string]time.Time)
	}
	q.lastSaved[id] = time.Now()
}

// startFlusher runs the periodic dirty-job flush until Close.
func (q *Queue) startFlusher() {
	q.flushStop = make(chan struct{})
	q.flushDone = make(chan struct{})
	go func() {
		defer close(q.flushDone)
		t := time.NewTicker(dirtyFlushInterval)
		defer t.Stop()
		for {
			select {
			case <-t.C:
				q.flushDirty(false)
			case <-q.flushStop:
				return
			}
		}
	}()
}

// flushDirty saves dirty jobs. Unless force is set, a large job is saved no
// more often than largeJobFlushInterval.
func (q *Queue) flushDirty(force bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	now := time.Now()
	for id := range q.dirty {
		job, ok := q.Jobs[id]
		if !ok {
			delete(q.dirty, id)
			continue
		}
		if !force && len(job.Stdout)+len(job.OutputFiles) > largeJobThreshold &&
			now.Sub(q.lastSaved[id]) < largeJobFlushInterval {
			continue
		}
		if err := q.saveJobToDB(job); err != nil {
			log.Printf("Failed to flush job %s to database: %v", id, err)
			delete(q.dirty, id) // don't spin on a closed/broken DB
		}
	}
}

// Close stops the background flusher and persists every pending append.
// Call it before closing the queue's database. Safe to call more than once.
func (q *Queue) Close() {
	q.closeOnce.Do(func() {
		if q.flushStop != nil {
			close(q.flushStop)
			<-q.flushDone
		}
		q.flushDirty(true)
	})
}
