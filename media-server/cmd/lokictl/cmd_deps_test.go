package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// fastDeps makes the download wait poll in milliseconds for one test.
func fastDeps(t *testing.T) {
	t.Helper()
	poll, grace, restart := depsPoll, depsStatusGrace, depsRestartAfter
	depsPoll, depsStatusGrace, depsRestartAfter = time.Millisecond, 200*time.Millisecond, 5*time.Millisecond
	t.Cleanup(func() { depsPoll, depsStatusGrace, depsRestartAfter = poll, grace, restart })
}

// depsServer answers /download with 202 and /api/deps/status with the next of
// states (the last one repeats); a state "!" answers the status with a 503.
// It counts download requests.
func depsServer(t *testing.T, states ...string) (*httptest.Server, func() int) {
	t.Helper()
	var mu sync.Mutex
	downloads, n := 0, 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case strings.HasSuffix(r.URL.Path, "/download"):
			downloads++
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"id":"m1","state":"queued"}`))
		case r.URL.Path == "/api/deps/status":
			st := states[min(n, len(states)-1)]
			n++
			if st == "!" {
				http.Error(w, "restarting", http.StatusServiceUnavailable)
				return
			}
			_, _ = w.Write([]byte(st))
		default:
			t.Errorf("unexpected %s", r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, func() int { mu.Lock(); defer mu.Unlock(); return downloads }
}

// Real status rows: other dependencies carry object details (install hints,
// a user-supplied copy's source), the model in flight its progress.
const (
	otherDeps   = `{"id":"yt-dlp","category":"optional","name":"yt-dlp","state":"missing","detail":{"description":"Video downloader","commands":[{"os":"darwin","label":"Homebrew"}]}},{"id":"loki-retouch","category":"tool","name":"Retouch","state":"installed","detail":{"source":"path"},"source":"user"}`
	m1Installed = `[` + otherDeps + `,{"id":"m1","category":"model","name":"M","state":"installed"}]`
)

func m1(state string, extra string) string {
	return `[` + otherDeps + `,{"id":"m1","category":"model","name":"M","state":"` + state + `"` + extra + `}]`
}

// The bug: any dependency with an object detail broke every status read.
func TestDepsStatusReadsObjectDetails(t *testing.T) {
	srv, _ := depsServer(t, m1("downloading", `,"detail":{"id":"m1","state":"downloading","bytes_done":5,"bytes_total":10}`))
	a, _, _ := appForServer(srv.URL)
	items, err := fetchDeps(a)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 || items[1].Source != "user" {
		t.Fatalf("items %+v", items)
	}
	if done, total := items[2].progress(); done != 5 || total != 10 {
		t.Fatalf("progress %d/%d", done, total)
	}
	if done, total := items[0].progress(); done != 0 || total != 0 {
		t.Fatalf("a hint is not progress: %d/%d", done, total)
	}
}

func TestDepsDownloadWaitShowsProgressThenInstalled(t *testing.T) {
	fastDeps(t)
	srv, downloads := depsServer(t,
		m1("downloading", `,"detail":{"state":"downloading","bytes_done":10,"bytes_total":100}`),
		m1("downloading", `,"detail":{"state":"downloading","bytes_done":60,"bytes_total":100}`),
		m1Installed)
	a, out, errOut := appForServer(srv.URL)
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"installed"`) || !strings.Contains(errOut.String(), `"bytes_done":60`) || downloads() != 1 {
		t.Fatalf("out %s\nprogress %s\ndownloads %d", out.String(), errOut.String(), downloads())
	}
}

// A cancelled install is over: no waiting out the timeout.
func TestDepsDownloadWaitStopsWhenCancelled(t *testing.T) {
	fastDeps(t)
	srv, _ := depsServer(t, m1("downloading", ""), m1("cancelled", ""))
	a, out, _ := appForServer(srv.URL)
	start := time.Now()
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code != 1 {
		t.Fatalf("exit %d, want 1", code)
	}
	if time.Since(start) > 2*time.Second || !strings.Contains(out.String(), `"cancelled"`) {
		t.Fatalf("took %s, out %s", time.Since(start), out.String())
	}
}

func TestDepsDownloadWaitStopsWhenFailed(t *testing.T) {
	fastDeps(t)
	srv, _ := depsServer(t, m1("failed", `,"error":"checksum mismatch"`))
	a, out, _ := appForServer(srv.URL)
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code != 1 || !strings.Contains(out.String(), "checksum mismatch") {
		t.Fatalf("exit %d, out %s", code, out.String())
	}
}

// The server restarting mid-download: the status fails for a while, and the
// wait carries on.
func TestDepsDownloadWaitRidesOutStatusErrors(t *testing.T) {
	fastDeps(t)
	srv, _ := depsServer(t, m1("downloading", ""), "!", "!", "!", m1Installed)
	a, out, errOut := appForServer(srv.URL)
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if !strings.Contains(out.String(), `"installed"`) {
		t.Fatalf("out %s", out.String())
	}
}

// ...but not forever.
func TestDepsDownloadWaitGivesUpOnAStatusThatKeepsFailing(t *testing.T) {
	fastDeps(t)
	srv, _ := depsServer(t, "!")
	a, _, errOut := appForServer(srv.URL)
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code == 0 || !strings.Contains(errOut.String(), "kept failing") {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
}

// After a restart the server has forgotten the install: the model reads
// "missing" and nothing downloads it, so the wait asks again (it resumes).
func TestDepsDownloadWaitRestartsALostInstall(t *testing.T) {
	fastDeps(t)
	missing := m1("missing", "")
	states := []string{m1("downloading", "")}
	for i := 0; i < 50; i++ {
		states = append(states, missing)
	}
	srv, downloads := depsServer(t, append(states, m1Installed)...)
	a, out, errOut := appForServer(srv.URL)
	if code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "5s"}); code != 0 {
		t.Fatalf("exit %d: %s", code, errOut.String())
	}
	if downloads() < 2 || !strings.Contains(out.String(), `"installed"`) {
		t.Fatalf("downloads %d, out %s", downloads(), out.String())
	}
}
