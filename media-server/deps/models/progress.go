package models

import (
	"context"
	"sync"
	"time"
)

// InstallState enumerates the lifecycle of a model install.
type InstallState string

const (
	StateQueued      InstallState = "queued"
	StateDownloading InstallState = "downloading"
	StateVerifying   InstallState = "verifying"
	StateInstalled   InstallState = "installed"
	StateFailed      InstallState = "failed"
	StateCancelled   InstallState = "cancelled"
)

// Install is the current snapshot of one install attempt.
type Install struct {
	ID          string       `json:"id"`
	State       InstallState `json:"state"`
	CurrentFile string       `json:"current_file,omitempty"`
	BytesDone   int64        `json:"bytes_done"`
	BytesTotal  int64        `json:"bytes_total"`
	Error       string       `json:"error,omitempty"`
	StartedAt   time.Time    `json:"started_at"`
	UpdatedAt   time.Time    `json:"updated_at"`
}

type tracker struct {
	mu      sync.RWMutex
	rows    map[string]*Install
	cancels map[string]context.CancelFunc
	subs    map[chan Install]struct{}
}

var Tracker = &tracker{
	rows:    map[string]*Install{},
	cancels: map[string]context.CancelFunc{},
	subs:    map[chan Install]struct{}{},
}

// StartInstall launches an install for id in a goroutine. If one is already
// active for id, returns the existing one without starting another.
func (t *tracker) StartInstall(id string) (*Install, error) {
	t.mu.Lock()
	if row, ok := t.rows[id]; ok && (row.State == StateDownloading || row.State == StateQueued || row.State == StateVerifying) {
		t.mu.Unlock()
		return row, nil
	}
	row := &Install{ID: id, State: StateQueued, StartedAt: time.Now().UTC(), UpdatedAt: time.Now().UTC()}
	t.rows[id] = row
	ctx, cancel := context.WithCancel(context.Background())
	t.cancels[id] = cancel
	t.mu.Unlock()

	t.broadcast(*row)
	go t.run(ctx, id)
	return row, nil
}

// Cancel cancels the in-flight install for id if any.
func (t *tracker) Cancel(id string) bool {
	t.mu.Lock()
	cancel, ok := t.cancels[id]
	t.mu.Unlock()
	if !ok {
		return false
	}
	cancel()
	return true
}

// Snapshot returns the current Install for id, or false.
func (t *tracker) Snapshot(id string) (Install, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if row, ok := t.rows[id]; ok {
		return *row, true
	}
	return Install{}, false
}

// All returns a copy of every tracked Install.
func (t *tracker) All() []Install {
	t.mu.RLock()
	defer t.mu.RUnlock()
	out := make([]Install, 0, len(t.rows))
	for _, r := range t.rows {
		out = append(out, *r)
	}
	return out
}

// Subscribe returns a channel that receives every state transition. The
// returned cleanup func must be called to remove the subscription.
func (t *tracker) Subscribe() (<-chan Install, func()) {
	ch := make(chan Install, 16)
	t.mu.Lock()
	t.subs[ch] = struct{}{}
	t.mu.Unlock()
	return ch, func() {
		t.mu.Lock()
		delete(t.subs, ch)
		t.mu.Unlock()
		close(ch)
	}
}

func (t *tracker) broadcast(snap Install) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	for ch := range t.subs {
		// Drop messages on a slow consumer rather than block the producer.
		select {
		case ch <- snap:
		default:
		}
	}
}

func (t *tracker) update(id string, f func(*Install)) {
	t.mu.Lock()
	row, ok := t.rows[id]
	if !ok {
		t.mu.Unlock()
		return
	}
	f(row)
	row.UpdatedAt = time.Now().UTC()
	snap := *row
	t.mu.Unlock()
	t.broadcast(snap)
}

func (t *tracker) run(ctx context.Context, id string) {
	t.update(id, func(r *Install) { r.State = StateDownloading })
	progressFn := func(file string, done, total int64) {
		t.update(id, func(r *Install) {
			r.CurrentFile = file
			r.BytesDone = done
			r.BytesTotal = total
		})
	}
	err := InstallModel(ctx, id, progressFn)
	t.mu.Lock()
	delete(t.cancels, id)
	t.mu.Unlock()
	t.update(id, func(r *Install) {
		switch {
		case err == nil:
			r.State = StateInstalled
			r.Error = ""
		case ctx.Err() != nil:
			r.State = StateCancelled
		default:
			r.State = StateFailed
			r.Error = err.Error()
		}
	})
}
