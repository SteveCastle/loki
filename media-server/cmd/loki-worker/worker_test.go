package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// The test binary doubles as a fake engine in --worker mode (LOKI_FAKE_ENGINE=1):
//
//	-o OUT IN        copies IN to OUT (logs two step lines, prints OUT)
//	--sleep ...      runs until cancelled
//	--fail ...       fails with "boom"
//	--show-prompt    prints a prompt
//
// and answers loki-segment session requests (open / segment writing a mask).
func TestMain(m *testing.M) {
	if os.Getenv("LOKI_FAKE_ENGINE") == "1" {
		os.Exit(fakeEngine())
	}
	os.Exit(m.Run())
}

func fakeEngine() int {
	if len(os.Args) < 2 || os.Args[1] != "--worker" {
		// a plain command line (a light call)
		for _, a := range os.Args[1:] {
			if a == "--show-prompt" {
				fmt.Println("PROMPT")
				fmt.Fprintln(os.Stderr, "prompt only")
				return 0
			}
		}
		fmt.Fprintln(os.Stderr, "fake: error: only --show-prompt runs without --worker")
		return 1
	}
	out := json.NewEncoder(os.Stdout)
	_ = out.Encode(map[string]any{"event": "ready", "tool": "fake", "version": "9.9", "model": "Fake", "protocol": 1})
	var cancelled atomic.Bool
	var current atomic.Value
	current.Store("")
	reqs := make(chan map[string]any, 8)
	go func() {
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			var r map[string]any
			if json.Unmarshal(sc.Bytes(), &r) != nil {
				continue
			}
			if c, ok := r["cancel"].(string); ok {
				if c == current.Load().(string) {
					cancelled.Store(true)
				}
				continue
			}
			reqs <- r
		}
		close(reqs)
	}()
	opened := map[string]string{}
	loaded := false
	for r := range reqs {
		id, _ := r["id"].(string)
		current.Store(id)
		cancelled.Store(false)
		done := map[string]any{"id": id, "event": "done", "ok": true, "error": nil}
		var stdout []string
		cmd, _ := r["cmd"].(string)
		switch cmd {
		case "", "run":
			var args []string
			for _, a := range r["args"].([]any) {
				args = append(args, a.(string))
			}
			has := func(f string) bool {
				for _, a := range args {
					if a == f {
						return true
					}
				}
				return false
			}
			if !loaded {
				fmt.Fprintln(os.Stderr, "loading the fake model")
				loaded = true
			}
			switch {
			case has("--show-prompt"):
				stdout = append(stdout, "PROMPT line 1\nPROMPT line 2")
			case has("--fail"):
				fmt.Fprintln(os.Stderr, "about to fail")
				done["ok"], done["error"] = false, "boom"
			case has("--sleep"):
				for !cancelled.Load() {
					fmt.Fprintln(os.Stderr, "step 1/100")
					time.Sleep(20 * time.Millisecond)
				}
				done["ok"], done["error"], done["cancelled"] = false, "cancelled", true
			default:
				var o, in string
				for i := 0; i < len(args); i++ {
					if args[i] == "-o" && i+1 < len(args) {
						o = args[i+1]
						i++
					} else if !strings.HasPrefix(args[i], "-") {
						in = args[i]
					}
				}
				b, err := os.ReadFile(in)
				if err != nil {
					done["ok"], done["error"] = false, err.Error()
					break
				}
				fmt.Fprintln(os.Stderr, "step 1/2")
				fmt.Fprintf(os.Stderr, "reading %s\n", in)
				if strings.HasSuffix(o, "/") {
					o = filepath.Join(o, "result.txt")
				}
				_ = os.MkdirAll(filepath.Dir(o), 0755)
				_ = os.WriteFile(o, append([]byte("processed:"), b...), 0644)
				fmt.Fprintln(os.Stderr, "step 2/2")
				stdout = append(stdout, o)
			}
		case "session":
			sess, _ := r["session"].(string)
			req, _ := r["req"].(map[string]any)
			switch req["cmd"] {
			case "open":
				p, _ := req["path"].(string)
				if _, err := os.Stat(p); err != nil {
					done["ok"], done["error"] = false, err.Error()
				} else {
					opened[sess] = p
					done["reply"] = map[string]any{"ok": true, "width": 4, "height": 4}
				}
			case "segment":
				if opened[sess] == "" {
					done["ok"], done["error"] = false, "no image open: send {\"cmd\":\"open\",\"path\":...} first"
					break
				}
				dir, _ := req["out"].(string)
				_ = os.MkdirAll(dir, 0755)
				mask := filepath.Join(dir, "1.png")
				_ = os.WriteFile(mask, []byte("mask"), 0644)
				done["reply"] = map[string]any{"width": 4, "height": 4, "objects": []any{map[string]any{"id": 1, "mask": mask}}}
			}
		case "close-session":
			delete(opened, r["session"].(string))
		case "warm":
			fmt.Fprintln(os.Stderr, "warming")
		case "evict", "status":
		}
		for _, l := range stdout {
			_ = out.Encode(map[string]any{"id": id, "event": "stdout", "line": l})
		}
		done["resident"] = []any{map[string]any{"key": "fake model", "mb": 10}}
		fmt.Fprintf(os.Stderr, "%s %s\n", sentinel, id)
		_ = out.Encode(done)
		current.Store("")
	}
	return 0
}

const testToken = "tok"

func testWorker(t *testing.T, coexist string) (*worker, *httptest.Server) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	cfg := config{name: "test", mode: "http", dataDir: t.TempDir(), keep: time.Hour, token: testToken, coexist: coexist, engineEnv: []string{"LOKI_FAKE_ENGINE=1"}}
	tools := map[string]string{"loki-retouch": exe, "loki-segment": exe, "loki-dream": exe}
	w := newWorker(cfg, tools, []workerapi.GPU{{Index: 0, Name: "Fake GPU", TotalMB: 24000}})
	w.ready = true
	ctx, cancel := context.WithCancel(context.Background())
	go w.dispatch(ctx)
	srv := httptest.NewServer(w.routes())
	t.Cleanup(func() {
		srv.Close()
		cancel()
		shutdown(w)
	})
	return w, srv
}

func TestWorkerRunRoundTrip(t *testing.T) {
	w, srv := testWorker(t, "auto")
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	dir := t.TempDir()
	in := filepath.Join(dir, "my photo.png")
	if err := os.WriteFile(in, []byte("pixels"), 0644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "results", "out.png")
	for round := 0; round < 2; round++ {
		os.Remove(out)
		var logs []string
		plan := workerapi.PlanArgs([]string{"--steps", "2", "-o", out, in})
		res, err := workerapi.RunOnNode(context.Background(), n, plan, "retouch", false, workerapi.Hooks{Log: func(l string) { logs = append(logs, l) }})
		if err != nil {
			t.Fatalf("round %d: %v (logs %q)", round, err, logs)
		}
		if len(res.Stdout) != 1 || res.Stdout[0] != out {
			t.Fatalf("stdout %q, want %q", res.Stdout, out)
		}
		b, err := os.ReadFile(out)
		if err != nil || string(b) != "processed:pixels" {
			t.Fatalf("output %q %v", b, err)
		}
		// the engine's mention of its input comes back as the local path
		found := false
		for _, l := range logs {
			if strings.Contains(l, "reading "+in) {
				found = true
			}
		}
		if !found {
			t.Fatalf("logs do not map the input path back: %q", logs)
		}
	}
	// one engine process served both jobs (models stayed resident)
	if g := w.slots[0].engines["loki-retouch"].gen; g != 1 {
		t.Fatalf("engine started %d times, want 1", g)
	}
	info, err := n.Info(context.Background(), false)
	if err != nil || !info.Warm("loki-retouch") {
		t.Fatalf("info after a run should show resident models: %+v %v", info, err)
	}
}

func TestWorkerOutputFolder(t *testing.T) {
	_, srv := testWorker(t, "auto")
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	dir := t.TempDir()
	in := filepath.Join(dir, "in.txt")
	os.WriteFile(in, []byte("x"), 0644)
	outDir := filepath.Join(dir, "Generated")
	os.MkdirAll(outDir, 0755)
	res, err := workerapi.RunOnNode(context.Background(), n, workerapi.PlanArgs([]string{"-o", outDir, in}), "dream", false, workerapi.Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(outDir, "result.txt")
	if len(res.Stdout) != 1 || res.Stdout[0] != want {
		t.Fatalf("stdout %q, want %q", res.Stdout, want)
	}
	if _, err := os.Stat(want); err != nil {
		t.Fatal(err)
	}
}

func TestWorkerFailureAndAuth(t *testing.T) {
	_, srv := testWorker(t, "auto")
	bad := &workerapi.Node{Name: "test", URL: srv.URL, Token: "wrong"}
	if _, err := bad.Info(context.Background(), false); err != workerapi.ErrUnauthorized {
		t.Fatalf("wrong token: %v", err)
	}
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	_, err := workerapi.RunOnNode(context.Background(), n, workerapi.PlanArgs([]string{"--fail"}), "retouch", false, workerapi.Hooks{})
	var re *workerapi.RemoteError
	if err == nil || !strings.Contains(err.Error(), "boom") || !errorAs(err, &re) {
		t.Fatalf("failure: %v", err)
	}
	_, err = workerapi.RunOnNode(context.Background(), n, workerapi.PlanArgs([]string{"-o", "x"}), "reshoot", false, workerapi.Hooks{})
	if err == nil || !strings.Contains(err.Error(), "not available") {
		t.Fatalf("missing engine: %v", err)
	}
}

func errorAs(err error, re **workerapi.RemoteError) bool {
	r, ok := err.(*workerapi.RemoteError)
	if ok {
		*re = r
	}
	return ok
}

func TestWorkerCancel(t *testing.T) {
	w, srv := testWorker(t, "auto")
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := workerapi.RunOnNode(ctx, n, workerapi.PlanArgs([]string{"--sleep"}), "retouch", false, workerapi.Hooks{})
		errc <- err
	}()
	time.Sleep(700 * time.Millisecond)
	cancel()
	select {
	case err := <-errc:
		if err != context.Canceled {
			t.Fatalf("cancelled run: %v", err)
		}
	case <-time.After(20 * time.Second):
		t.Fatal("cancel did not return")
	}
	// the engine stopped the run and is free for the next job
	deadline := time.Now().Add(10 * time.Second)
	for w.slots[0].engines["loki-retouch"].State() != "idle" {
		if time.Now().After(deadline) {
			t.Fatalf("engine state %s", w.slots[0].engines["loki-retouch"].State())
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestWorkerLightJob(t *testing.T) {
	_, srv := testWorker(t, "auto")
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	// a light call runs as its own process, without --worker
	res, err := workerapi.RunOnNode(context.Background(), n, workerapi.PlanArgs([]string{"--show-prompt"}), "retouch", true, workerapi.Hooks{})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Stdout) != 1 || res.Stdout[0] != "PROMPT" {
		t.Fatalf("stdout %q", res.Stdout)
	}
}

func TestWorkerSessions(t *testing.T) {
	_, srv := testWorker(t, "never")
	n := &workerapi.Node{Name: "test", URL: srv.URL, Token: testToken}
	ctx := context.Background()
	id, err := n.OpenSession(ctx, "segment")
	if err != nil {
		t.Fatal(err)
	}
	pic := filepath.Join(t.TempDir(), "pic.png")
	os.WriteFile(pic, []byte("png"), 0644)
	r, err := n.CallSession(ctx, id, map[string]any{"cmd": "segment", "point": []int{1, 1}, "out": workerapi.SessionDir + "/out/1"}, time.Minute)
	if err != nil || r.Error == "" {
		t.Fatalf("segment before open should fail: %v %+v", err, r)
	}
	if err := n.PutSessionFile(ctx, id, "in/pic.png", pic); err != nil {
		t.Fatal(err)
	}
	if r, err = n.CallSession(ctx, id, map[string]any{"cmd": "open", "path": workerapi.SessionDir + "/in/pic.png"}, time.Minute); err != nil || r.Error != "" {
		t.Fatalf("open: %v %+v", err, r)
	}
	r, err = n.CallSession(ctx, id, map[string]any{"cmd": "segment", "point": []int{1, 1}, "out": workerapi.SessionDir + "/out/1"}, time.Minute)
	if err != nil || r.Error != "" {
		t.Fatalf("segment: %v %+v", err, r)
	}
	mask := r.Reply["objects"].([]any)[0].(map[string]any)["mask"].(string)
	if mask != workerapi.SessionDir+"/out/1/1.png" {
		t.Fatalf("mask path %q", mask)
	}
	if len(r.Files) != 1 || r.Files[0].Path != "out/1/1.png" {
		t.Fatalf("files %+v", r.Files)
	}
	var got []string
	if err := n.SessionFiles(ctx, id, "out/1", func(rel string, rd io.Reader) error {
		b, _ := io.ReadAll(rd)
		got = append(got, rel+"="+string(b))
		return nil
	}); err != nil || len(got) != 1 || got[0] != "out/1/1.png=mask" {
		t.Fatalf("files.tar: %v %q", err, got)
	}
	if err := n.CloseSession(ctx, id); err != nil {
		t.Fatal(err)
	}
	if _, err := n.CallSession(ctx, id, map[string]any{"cmd": "open"}, time.Minute); err == nil {
		t.Fatal("a closed session still answers")
	}
}

func TestWorkerPutOutputs(t *testing.T) {
	w, _ := testWorker(t, "auto")
	var mu sync.Mutex
	got := map[string]string{}
	store := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		got[r.URL.Path] = string(b)
		mu.Unlock()
	}))
	defer store.Close()
	src := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) { rw.Write([]byte("remote-bytes")) }))
	defer src.Close()
	spec := workerapi.JobSpec{Tool: "loki-retouch", Args: []string{"-o", "$JOB/out/0/r.png", "$JOB/in/0/a.png"},
		Inputs:  []workerapi.Input{{Path: "in/0/a.png", URL: src.URL + "/a"}},
		Outputs: &workerapi.OutputSink{PutURLs: []string{store.URL + "/slot0", store.URL + "/slot1"}}}
	j, err := w.newJob(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := stage(context.Background(), j.dir, &j.spec); err != nil {
		t.Fatal(err)
	}
	w.submit(j)
	<-j.done
	st := j.status(0)
	if st.State != workerapi.JobDone || len(st.Outputs) != 1 || st.Outputs[0].Slot != 0 {
		t.Fatalf("status %+v", st)
	}
	if got["/slot0"] != "processed:remote-bytes" {
		t.Fatalf("uploaded %q", got)
	}
}

// RunPod queue mode: take a job from the job-take webhook, run it, post the
// result (inline outputs) to job-done.
func TestRunPodQueueMode(t *testing.T) {
	w, _ := testWorker(t, "auto")
	spec := workerapi.JobSpec{Tool: "loki-retouch", Args: []string{"-o", "$JOB/out/0/r.png", "$JOB/in/0/a.png"},
		Inputs:  []workerapi.Input{{Path: "in/0/a.png", Data: []byte("inline")}},
		Outputs: &workerapi.OutputSink{Inline: true}}
	var mu sync.Mutex
	taken := false
	var result []byte
	var streamed int
	done := make(chan struct{})
	rp := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "rpkey" {
			rw.WriteHeader(401)
			return
		}
		mu.Lock()
		defer mu.Unlock()
		switch {
		case strings.HasPrefix(r.URL.Path, "/take/"):
			if taken {
				rw.WriteHeader(http.StatusNoContent)
				return
			}
			taken = true
			_ = json.NewEncoder(rw).Encode(map[string]any{"id": "rp1", "input": workerapi.RunPodInput{Spec: &spec}})
		case strings.HasPrefix(r.URL.Path, "/done/rp1"):
			b, _ := io.ReadAll(r.Body)
			if r.URL.Query().Get("isStream") == "true" {
				streamed++
				return
			}
			result = b
			close(done)
		}
	}))
	defer rp.Close()
	e := runpodEnv{getJob: rp.URL + "/take/$ID?x=1", postOutput: rp.URL + "/done/$ID", postStream: rp.URL + "/done/$ID", key: "rpkey", podID: "pod1", pingEvery: time.Hour}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go w.serveRunPod(ctx, e)
	select {
	case <-done:
	case <-time.After(30 * time.Second):
		t.Fatal("no result posted")
	}
	var body struct {
		Output workerapi.JobStatus `json:"output"`
	}
	if err := json.Unmarshal(result, &body); err != nil {
		t.Fatal(err)
	}
	st := body.Output
	if st.State != workerapi.JobDone || len(st.Outputs) != 1 || string(st.Outputs[0].Data) != "processed:inline" {
		t.Fatalf("result %+v", st)
	}
	if len(st.Stdout) != 1 || st.Stdout[0] != "$JOB/out/0/r.png" {
		t.Fatalf("stdout %q", st.Stdout)
	}
}

func TestPlaceholder(t *testing.T) {
	dir := filepath.Join(os.TempDir(), "lw", "jobs", "j1")
	cases := map[string]string{
		filepath.ToSlash(dir) + "/out/0/x.png": "$JOB/out/0/x.png",
		filepath.FromSlash(dir) + string(filepath.Separator) + "out" + string(filepath.Separator) + "x.png": "$JOB/out/x.png",
		"no paths here": "no paths here",
	}
	for in, want := range cases {
		if got := placeholder(in, dir, workerapi.JobDir); got != want {
			t.Errorf("placeholder(%q) = %q, want %q", in, got, want)
		}
	}
	// a JSON line (Windows paths are escaped there)
	js, _ := json.Marshal(map[string]string{"mask": filepath.Join(dir, "out", "1.png")})
	if got := placeholder(string(js), dir, workerapi.JobDir); !strings.Contains(got, `"$JOB/out/1.png"`) {
		t.Errorf("json placeholder: %s", got)
	}
}
