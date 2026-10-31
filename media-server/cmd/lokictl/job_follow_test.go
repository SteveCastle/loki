package main

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// newFollowServer fakes a job that runs and finishes entirely on the stream:
// /stream sends its stdout lines and terminal update as soon as a client
// attaches — i.e. BEFORE /job/src/copy has returned the new id — then holds
// the connection open, so a follower only stops by recognising the update.
// /jobs/list reports in_progress for the first pendingCalls polls.
func newFollowServer(t *testing.T, pendingCalls int32, finalState string) *httptest.Server {
	t.Helper()
	var listCalls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/stream":
			w.Header().Set("Content-Type", "text/event-stream")
			fl := w.(http.Flusher)
			fmt.Fprint(w, "event: connected\ndata: hi\n\n")
			fmt.Fprint(w, `event: stdout-other`+"\n"+`data: {"updateType":"stdout","line":"not ours"}`+"\n\n")
			fmt.Fprint(w, `event: stdout-job-2`+"\n"+`data: {"updateType":"stdout","line":"Moved: a -> b"}`+"\n\n")
			fmt.Fprint(w, `event: stdout-job-2`+"\n"+`data: {"updateType":"stdout","line":"1 of 2 items were NOT moved:"}`+"\n\n")
			fmt.Fprint(w, `event: update`+"\n"+`data: {"updateType":"update","job":{"id":"job-2","state":"`+finalState+`"},"html":""}`+"\n\n")
			fmt.Fprint(w, `event: stdout-job-2`+"\n"+`data: {"updateType":"stdout","line":"after the end"}`+"\n\n")
			fl.Flush()
			<-r.Context().Done()
		case "/job/src/copy":
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{"id":"job-2","message":"Job copied successfully"}`))
		case "/job/job-2/resume":
			w.WriteHeader(http.StatusOK)
		case "/jobs/list":
			state := "in_progress"
			if atomic.AddInt32(&listCalls, 1) > pendingCalls {
				state = finalState
			}
			_, _ = w.Write([]byte(`[{"id":"job-2","command":"move","state":"` + state + `"}]`))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

func fastPolling(t *testing.T) {
	oldPoll, oldDrain := pollInterval, followDrain
	pollInterval = time.Millisecond
	followDrain = 2 * time.Second
	t.Cleanup(func() { pollInterval, followDrain = oldPoll, oldDrain })
}

func TestJobCopyFollowStreamsLinesAndExitsWithTheJob(t *testing.T) {
	fastPolling(t)
	srv := newFollowServer(t, 3, "error")
	a, out, errOut := appForServer(srv.URL)

	done := make(chan int, 1)
	go func() { done <- cmdJobCopy(a, []string{"src", "--follow"}) }()
	var code int
	select {
	case code = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("job copy --follow did not exit after the job finished")
	}

	if code != 3 {
		t.Errorf("exit = %d, want 3 for an errored job", code)
	}
	logs := errOut.String()
	// Lines the server sent before the copy returned the id are kept, the
	// JSON envelope is unwrapped, and other jobs' lines are not shown.
	if !strings.Contains(logs, "Moved: a -> b\n1 of 2 items were NOT moved:\n") {
		t.Errorf("stderr = %q, want the job's lines in order", logs)
	}
	for _, unwanted := range []string{"not ours", "after the end", `"updateType"`} {
		if strings.Contains(logs, unwanted) {
			t.Errorf("stderr contains %q: %q", unwanted, logs)
		}
	}
	if !strings.Contains(out.String(), `"id": "job-2"`) {
		t.Errorf("stdout = %s, want the final job JSON", out.String())
	}
}

func TestJobCopyWaitIsQuietUntilTheEnd(t *testing.T) {
	fastPolling(t)
	srv := newFollowServer(t, 1, "completed")
	a, out, errOut := appForServer(srv.URL)
	if code := cmdJobCopy(a, []string{"src", "--wait"}); code != 0 {
		t.Fatalf("exit = %d; stderr = %s", code, errOut.String())
	}
	if errOut.Len() != 0 {
		t.Errorf("--wait alone should not stream: %q", errOut.String())
	}
	if !strings.Contains(out.String(), `"state": "completed"`) {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestJobCopyWithoutFlagsPrintsTheNewID(t *testing.T) {
	srv := newFollowServer(t, 0, "completed")
	a, out, _ := appForServer(srv.URL)
	if code := cmdJobCopy(a, []string{"src"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(out.String(), `"id": "job-2"`) {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestJobCopyRejectsExtraArgs(t *testing.T) {
	a, _, _ := appForServer("http://127.0.0.1:1")
	if code := cmdJobCopy(a, []string{"src", "extra"}); code != 2 {
		t.Errorf("exit = %d, want usage error 2", code)
	}
	if code := cmdJobCopy(a, []string{"--follow"}); code != 2 {
		t.Errorf("exit = %d, want usage error 2 with no id", code)
	}
}

func TestJobResumeFollow(t *testing.T) {
	fastPolling(t)
	srv := newFollowServer(t, 2, "completed")
	a, _, errOut := appForServer(srv.URL)
	if code := cmdJobResume(a, []string{"job-2", "--follow"}); code != 0 {
		t.Fatalf("exit = %d; stderr = %s", code, errOut.String())
	}
	if !strings.Contains(errOut.String(), "Moved: a -> b") {
		t.Errorf("stderr = %q", errOut.String())
	}
}

func TestJobLogsExitsWhenTheJobFinishes(t *testing.T) {
	fastPolling(t)
	// Not terminal on the first lookup (cmdJobLogs refuses finished jobs).
	srv := newFollowServer(t, 2, "completed")
	a, out, errOut := appForServer(srv.URL)
	done := make(chan int, 1)
	go func() { done <- cmdJobLogs(a, []string{"job-2"}) }()
	select {
	case code := <-done:
		if code != 0 {
			t.Errorf("exit = %d; stderr = %s", code, errOut.String())
		}
	case <-time.After(5 * time.Second):
		t.Fatal("job logs kept streaming after the job finished")
	}
	if !strings.Contains(out.String(), "Moved: a -> b") {
		t.Errorf("stdout = %q", out.String())
	}
}

func TestStdoutLine(t *testing.T) {
	if got := stdoutLine(`{"updateType":"stdout","line":"hello"}`); got != "hello" {
		t.Errorf("got %q", got)
	}
	if got := stdoutLine("plain text"); got != "plain text" {
		t.Errorf("got %q", got)
	}
}
