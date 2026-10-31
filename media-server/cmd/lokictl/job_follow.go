package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// followDrain bounds how long a follower keeps reading after polling has
// already seen the job finish, waiting for the stream to deliver the job's
// last lines. The server broadcasts stdout and job updates through one
// ordered channel, so the terminal "update" event arrives after the final
// line — the drain normally ends on that event, well inside this bound.
var followDrain = 3 * time.Second

// followBufferMax caps the events held while the job id is still unknown
// (between attaching to the stream and the create response).
const followBufferMax = 10000

// jobFollower relays one job's stdout from /stream to w as plain lines.
//
// It attaches BEFORE the job exists: a fast job can print, or even finish,
// before the create call returns its id, and attaching afterwards would lose
// those lines. Events seen before SetID are buffered and replayed for the id.
// Reading stops on the job's own terminal "update" event, so the summary a
// task prints just before finishing is never cut off.
type jobFollower struct {
	w      io.Writer
	cancel context.CancelFunc
	exited chan struct{}

	mu      sync.Mutex
	id      string
	pending []sseEvent
	ended   bool
}

func startJobFollower(a *App, w io.Writer) *jobFollower {
	ctx, cancel := context.WithCancel(context.Background())
	f := &jobFollower{w: w, cancel: cancel, exited: make(chan struct{})}
	resp, err := a.Client.DoStream("GET", "/stream")
	if err != nil {
		fmt.Fprintf(a.ErrOut, `{"warning":"could not attach to job output stream: %s"}`+"\n", err)
		close(f.exited)
		return f
	}
	go func() {
		<-ctx.Done()
		resp.Body.Close()
	}()
	go func() {
		defer close(f.exited)
		defer resp.Body.Close()
		_ = readSSE(ctx, resp.Body, f.onEvent)
	}()
	return f
}

// SetID names the job to follow and replays what arrived before it was known.
func (f *jobFollower) SetID(id string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.id = id
	for _, ev := range f.pending {
		if !f.handle(ev) {
			break
		}
	}
	f.pending = nil
	if f.ended {
		f.cancel()
	}
}

// Finish waits up to drain for the stream to reach the job's end, then
// detaches. Call it once polling has seen a terminal state.
func (f *jobFollower) Finish(drain time.Duration) {
	select {
	case <-f.exited:
	case <-time.After(drain):
	}
	f.cancel()
	<-f.exited
}

func (f *jobFollower) onEvent(ev sseEvent) bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.id == "" {
		if (ev.Name == "update" || strings.HasPrefix(ev.Name, "stdout-")) && len(f.pending) < followBufferMax {
			f.pending = append(f.pending, ev)
		}
		return true
	}
	return f.handle(ev)
}

// handle prints the followed job's stdout and reports false once the job's
// terminal update has been seen. Caller holds f.mu.
func (f *jobFollower) handle(ev sseEvent) bool {
	if f.ended {
		return false
	}
	switch ev.Name {
	case "stdout-" + f.id:
		if line := stdoutLine(ev.Data); line != "" {
			fmt.Fprintln(f.w, line)
		}
	case "update":
		var u struct {
			Job struct {
				ID    string `json:"id"`
				State string `json:"state"`
			} `json:"job"`
		}
		if json.Unmarshal([]byte(ev.Data), &u) == nil && u.Job.ID == f.id && isTerminalState(u.Job.State) {
			f.ended = true
			return false
		}
	}
	return true
}

// stdoutLine unwraps the server's {"updateType":"stdout","line":...} payload,
// falling back to the raw data for anything else.
func stdoutLine(data string) string {
	var msg struct {
		Line *string `json:"line"`
	}
	if json.Unmarshal([]byte(data), &msg) == nil && msg.Line != nil {
		return *msg.Line
	}
	return data
}

// runTrackedJob is the --wait/--follow path shared by every command that
// starts a job. start creates (or re-queues) the job and returns its id; with
// follow, the stream is attached before start runs. Prints the final job JSON
// to stdout (the streamed lines go to stderr) and returns the waitForJob exit
// code: 0 completed, 3 error/cancelled/timed out.
func runTrackedJob(a *App, follow bool, timeout time.Duration, start func() (string, error)) int {
	var f *jobFollower
	if follow {
		f = startJobFollower(a, a.ErrOut)
	}
	id, err := start()
	if err != nil {
		if f != nil {
			f.Finish(0)
		}
		return a.Fail(err)
	}
	if f != nil {
		f.SetID(id)
	}
	job, code := waitForJob(a, id, timeout)
	if f != nil {
		drain := followDrain
		if job == nil || !isTerminalState(job.State) {
			drain = 0 // timed out or API error: nothing more to wait for
		}
		f.Finish(drain)
	}
	if job != nil {
		if pc := a.PrintJSON(job); code == 0 && pc != 0 {
			return pc
		}
	}
	return code
}
