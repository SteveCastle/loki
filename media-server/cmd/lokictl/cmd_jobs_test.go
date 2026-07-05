package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestBuildJobInput(t *testing.T) {
	got, err := buildJobInput("metadata", []string{"--type", "description", "C:/my dir/x.jpg"})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	want := `metadata --type description "C:/my dir/x.jpg"`
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}

	_, qerr := buildJobInput("t", []string{`has"quote`})
	if qerr == nil {
		t.Fatal("expected error for embedded quote")
	}
	if !strings.Contains(qerr.Error(), "--field") {
		t.Errorf("error should point at --field: %v", qerr)
	}
}

func TestSplitControlFlags(t *testing.T) {
	tokens, fields, wait, follow, timeout, err := splitControlFlags([]string{
		"--type", "description", "--wait", "--field", "prompt=describe it", "--timeout", "90s", "input.jpg",
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if !wait || follow {
		t.Errorf("wait=%v follow=%v", wait, follow)
	}
	if timeout != 90*time.Second {
		t.Errorf("timeout = %v", timeout)
	}
	if fields["prompt"] != "describe it" {
		t.Errorf("fields = %v", fields)
	}
	wantTokens := []string{"--type", "description", "input.jpg"}
	if len(tokens) != len(wantTokens) {
		t.Fatalf("tokens = %v", tokens)
	}
	for i := range wantTokens {
		if tokens[i] != wantTokens[i] {
			t.Errorf("tokens = %v, want %v", tokens, wantTokens)
			break
		}
	}
}

func TestReadSSE(t *testing.T) {
	stream := "event: connected\ndata: hi\n\n" +
		": keep-alive\n\n" +
		"event: stdout-abc\ndata: line one\ndata: line two\n\n"
	var got []sseEvent
	err := readSSE(context.Background(), strings.NewReader(stream), func(ev sseEvent) bool {
		got = append(got, ev)
		return true
	})
	if err != nil {
		t.Fatalf("err = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("events = %+v", got)
	}
	if got[0].Name != "connected" || got[0].Data != "hi" {
		t.Errorf("ev0 = %+v", got[0])
	}
	if got[1].Name != "stdout-abc" || got[1].Data != "line one\nline two" {
		t.Errorf("ev1 = %+v", got[1])
	}
}

// newJobServer fakes /create and /jobs/list: the job is pending for the
// first N list calls, then terminal with the given state.
func newJobServer(t *testing.T, pendingCalls int32, finalState string) (*httptest.Server, *int32) {
	t.Helper()
	var listCalls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/create":
			var req map[string]any
			_ = json.NewDecoder(r.Body).Decode(&req)
			if req["input"] == "" {
				t.Errorf("empty input in create request")
			}
			w.WriteHeader(http.StatusCreated)
			_, _ = w.Write([]byte(`{"id":"job-1"}`))
		case "/jobs/list":
			n := atomic.AddInt32(&listCalls, 1)
			state := "pending"
			if n > pendingCalls {
				state = finalState
			}
			_, _ = w.Write([]byte(`[{"id":"job-1","command":"wait","state":"` + state + `"}]`))
		default:
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &listCalls
}

func appForServer(url string) (*App, *bytes.Buffer, *bytes.Buffer) {
	out := &bytes.Buffer{}
	errOut := &bytes.Buffer{}
	return &App{
		Client: NewClient(url, "t", 5*time.Second),
		Out:    out,
		ErrOut: errOut,
	}, out, errOut
}

func TestJobRunWaitCompleted(t *testing.T) {
	old := pollInterval
	pollInterval = time.Millisecond
	defer func() { pollInterval = old }()

	srv, _ := newJobServer(t, 2, "completed")
	a, out, errOut := appForServer(srv.URL)
	code := cmdJobRun(a, []string{"wait", "5", "--wait"})
	if code != 0 {
		t.Fatalf("exit = %d; stderr = %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"state": "completed"`) {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestJobRunWaitErrorStateExits3(t *testing.T) {
	old := pollInterval
	pollInterval = time.Millisecond
	defer func() { pollInterval = old }()

	srv, _ := newJobServer(t, 0, "error")
	a, _, _ := appForServer(srv.URL)
	code := cmdJobRun(a, []string{"wait", "5", "--wait"})
	if code != 3 {
		t.Fatalf("exit = %d, want 3", code)
	}
}

func TestJobRunNoWaitPrintsID(t *testing.T) {
	srv, _ := newJobServer(t, 99, "pending")
	a, out, _ := appForServer(srv.URL)
	code := cmdJobRun(a, []string{"wait", "5"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(out.String(), "job-1") {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestJobClearNeedsYes(t *testing.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		_, _ = w.Write([]byte(`{"cleared_count":2}`))
	}))
	defer srv.Close()
	a, _, _ := appForServer(srv.URL)
	if code := cmdJobClear(a, nil); code != 2 {
		t.Errorf("without --yes exit = %d, want 2", code)
	}
	if called {
		t.Error("server hit without --yes")
	}
	if code := cmdJobClear(a, []string{"--yes"}); code != 0 {
		t.Errorf("with --yes exit = %d", code)
	}
	if !called {
		t.Error("server not hit with --yes")
	}
}

func TestJobListStateFilter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`[{"id":"a","state":"pending"},{"id":"b","state":"completed"}]`))
	}))
	defer srv.Close()
	a, out, _ := appForServer(srv.URL)
	if code := cmdJobList(a, []string{"--state", "completed"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if strings.Contains(out.String(), `"a"`) || !strings.Contains(out.String(), `"b"`) {
		t.Errorf("stdout = %s", out.String())
	}
}
