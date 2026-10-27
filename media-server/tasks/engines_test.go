package tasks

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/workerapi"
)

// fakeNode is a minimal loki-worker: it stages a job's uploads, runs the
// fake engine (the test binary, LOKI_FAKE_AI) with $JOB expanded, and serves
// the status and outputs like the real one.
type fakeNode struct {
	t     *testing.T
	dir   string
	bin   string
	mu    sync.Mutex
	jobs  map[string]*workerapi.JobStatus
	runs  atomic.Int32
	down  atomic.Bool
	srv   *httptest.Server
	warm  map[string]bool
	slots int
}

func newFakeNode(t *testing.T, slots int) *fakeNode {
	t.Helper()
	installFakeAI(t)
	bin, err := exec.LookPath("loki-retouch")
	if err != nil {
		t.Fatal(err)
	}
	n := &fakeNode{t: t, dir: t.TempDir(), bin: bin, jobs: map[string]*workerapi.JobStatus{}, warm: map[string]bool{}, slots: slots}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/info", func(w http.ResponseWriter, r *http.Request) {
		if n.down.Load() {
			http.Error(w, "down", http.StatusServiceUnavailable)
			return
		}
		info := workerapi.Info{Worker: "fake", Version: "1", Ready: true, Slots: n.slots}
		for _, e := range []string{"loki-retouch", "loki-reshoot", "loki-dream"} {
			ti := workerapi.ToolInfo{Name: e, Help: "--meta --model --show-prompt"}
			n.mu.Lock()
			if n.warm[e] {
				ti.Resident = []workerapi.Resident{{Key: "dit", MB: 1}}
			}
			n.mu.Unlock()
			info.Tools = append(info.Tools, ti)
		}
		_ = json.NewEncoder(w).Encode(info)
	})
	mux.HandleFunc("POST /v1/jobs", n.submit)
	mux.HandleFunc("GET /v1/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		n.mu.Lock()
		st := n.jobs[r.PathValue("id")]
		var cp workerapi.JobStatus
		if st != nil {
			cp = *st
		}
		n.mu.Unlock()
		if st == nil {
			http.NotFound(w, r)
			return
		}
		cp.LogNext = len(cp.Logs)
		_ = json.NewEncoder(w).Encode(cp)
	})
	mux.HandleFunc("GET /v1/jobs/{id}/outputs.tar", func(w http.ResponseWriter, r *http.Request) {
		n.mu.Lock()
		st := n.jobs[r.PathValue("id")]
		n.mu.Unlock()
		var rels []string
		for _, o := range st.Outputs {
			rels = append(rels, o.Path)
		}
		_ = workerapi.WriteTar(w, filepath.Join(n.dir, st.ID), rels)
	})
	mux.HandleFunc("DELETE /v1/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("POST /v1/jobs/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {})
	n.srv = httptest.NewServer(mux)
	t.Cleanup(n.srv.Close)
	return n
}

func (n *fakeNode) submit(w http.ResponseWriter, r *http.Request) {
	if n.down.Load() {
		http.Error(w, "down", http.StatusServiceUnavailable)
		return
	}
	mr, err := r.MultipartReader()
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	id := fmt.Sprintf("j%d", n.runs.Add(1))
	jd := filepath.Join(n.dir, id)
	var spec workerapi.JobSpec
	for {
		part, err := mr.NextPart()
		if err == io.EOF {
			break
		}
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if part.FormName() == "spec" {
			_ = json.NewDecoder(part).Decode(&spec)
			continue
		}
		if _, err := workerapi.WriteFileAtomic(filepath.Join(jd, filepath.FromSlash(part.FormName())), part); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
	}
	for _, d := range spec.Dirs {
		_ = os.MkdirAll(filepath.Join(jd, filepath.FromSlash(d)), 0755)
	}
	st := &workerapi.JobStatus{ID: id, Tool: spec.Tool, State: workerapi.JobRunning}
	n.mu.Lock()
	n.jobs[id] = st
	n.mu.Unlock()
	go func() {
		var args []string
		for _, a := range spec.Args {
			args = append(args, strings.ReplaceAll(a, workerapi.JobDir, filepath.ToSlash(jd)))
		}
		_ = os.MkdirAll(filepath.Join(jd, "out", "0"), 0755)
		cmd := exec.Command(n.bin, args...)
		cmd.Env = os.Environ()
		var so, se bytes.Buffer
		cmd.Stdout, cmd.Stderr = &so, &se
		err := cmd.Run()
		ph := func(s string) string {
			return strings.ReplaceAll(strings.ReplaceAll(s, filepath.ToSlash(jd), workerapi.JobDir), jd, workerapi.JobDir)
		}
		n.mu.Lock()
		defer n.mu.Unlock()
		for _, l := range strings.Split(strings.TrimSpace(se.String()), "\n") {
			st.Logs = append(st.Logs, ph(strings.TrimSpace(l)))
		}
		for _, l := range strings.Split(strings.TrimSpace(so.String()), "\n") {
			if l = strings.TrimSpace(l); l != "" {
				st.Stdout = append(st.Stdout, ph(filepath.ToSlash(l)))
			}
		}
		if err != nil {
			st.State, st.Error = workerapi.JobFailed, strings.TrimSpace(se.String())
			return
		}
		outs, _ := filepath.Glob(filepath.Join(jd, "out", "*", "*"))
		for _, o := range outs {
			rel, _ := filepath.Rel(jd, o)
			fi, _ := os.Stat(o)
			if fi != nil && !fi.IsDir() {
				st.Outputs = append(st.Outputs, workerapi.OutputFile{Path: filepath.ToSlash(rel), Size: fi.Size(), Slot: -1})
			}
		}
		st.State = workerapi.JobDone
		n.warm[spec.Tool] = true
	}()
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(st)
}

// usePools configures the dispatcher for a test and restores local after.
func usePools(t *testing.T, q *jobqueue.Queue, e appconfig.EnginesConfig) {
	t.Helper()
	ConfigureEngines(q, e)
	t.Cleanup(func() { ConfigureEngines(nil, appconfig.EnginesConfig{}) })
}

func TestEnginesDefaultIsLocal(t *testing.T) {
	ConfigureEngines(nil, appconfig.EnginesConfig{})
	for _, e := range workerapi.Engines {
		if !engineRunsLocalOnly(e) {
			t.Errorf("%s leaves this machine by default", e)
		}
	}
	if r := ResolveResources("retouch", nil, ""); len(r) != 1 || r[0] != HostBucketLocalCompute {
		t.Errorf("a local retouch holds %v", r)
	}
}

func TestEnginesRoutesAndPick(t *testing.T) {
	a, b := newFakeNode(t, 1), newFakeNode(t, 2)
	usePools(t, nil, appconfig.EnginesConfig{Strategy: "pool", Routes: map[string]string{"segment": "local"}, Pools: []appconfig.WorkerPool{
		{Name: "gpu", Kind: "nodes", Nodes: []appconfig.WorkerNode{{Name: "a", URL: a.srv.URL}, {Name: "b", URL: b.srv.URL}}},
	}})
	engineDisp.checkAll(context.Background())
	if !engineRunsLocalOnly("loki-segment") || engineRunsLocalOnly("loki-retouch") {
		t.Fatal("routes: segment should stay local, retouch go to the pool")
	}
	if r := ResolveResources("retouch", nil, ""); len(r) != 0 {
		t.Errorf("a pooled retouch holds local buckets %v", r)
	}
	// b has two slots, a one: three leases fit, the fourth waits
	ctx := context.Background()
	var leases []*engineLease
	for i := 0; i < 3; i++ {
		l, err := acquireEngine(ctx, "retouch", nil, nil)
		if err != nil || l.local() {
			t.Fatalf("lease %d: %v", i, err)
		}
		leases = append(leases, l)
	}
	got := make(chan *engineLease, 1)
	go func() {
		l, _ := acquireEngine(ctx, "retouch", nil, nil)
		got <- l
	}()
	select {
	case <-got:
		t.Fatal("a fourth lease did not wait for a free slot")
	case <-time.After(300 * time.Millisecond):
	}
	leases[0].Release()
	select {
	case l := <-got:
		l.Release()
	case <-time.After(5 * time.Second):
		t.Fatal("a released slot was not handed on")
	}
	for _, l := range leases[1:] {
		l.Release()
	}
	// warm affinity: with both idle, the node whose engine holds the models wins
	a.mu.Lock()
	a.warm["loki-dream"] = true
	a.mu.Unlock()
	engineDisp.checkAll(ctx)
	l, err := acquireEngine(ctx, "dream", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if l.t.node.URL != a.srv.URL {
		t.Errorf("picked %s, want the warm node", l.t.key)
	}
	l.Release()
}

func TestEnginesUnreachableAndFallback(t *testing.T) {
	a := newFakeNode(t, 1)
	a.down.Store(true)
	usePools(t, nil, appconfig.EnginesConfig{Strategy: "pool", Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "a", URL: a.srv.URL}}}}})
	engineDisp.checkAll(context.Background())
	if _, err := acquireEngine(context.Background(), "retouch", nil, nil); err == nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("an unreachable pool: %v", err)
	}
	usePools(t, nil, appconfig.EnginesConfig{Strategy: "pool", Fallback: true, Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "a", URL: a.srv.URL}}}}})
	engineDisp.checkAll(context.Background())
	l, err := acquireEngine(context.Background(), "retouch", nil, nil)
	if err != nil || !l.local() {
		t.Fatalf("fallback should lease the local GPU: %v", err)
	}
	l.Release()
}

// A retouch job routed to a pool runs on the worker and ends exactly like a
// local one: the result beside the original, in the library, progress full.
func TestRetouchTaskOnWorkerNode(t *testing.T) {
	node := newFakeNode(t, 1)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "my photo.png"), "png")
	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore", "--steps", "10", "--seed", "7"}, in)
	usePools(t, q, appconfig.EnginesConfig{Strategy: "pool", Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "fake", URL: node.srv.URL}}}}})
	// the engine is NOT on this machine's PATH: only the worker has it
	t.Setenv("PATH", t.TempDir())
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	want := filepath.Join(dir, "my photo_restored.png")
	if b, err := os.ReadFile(want); err != nil || string(b) != "fake output" {
		t.Fatalf("output %s: %q %v\n%s", want, b, err, jobLog(q, j.ID))
	}
	got := q.GetJob(j.ID)
	if got.State != jobqueue.StateCompleted || got.ProgressDone != got.ProgressTotal {
		t.Errorf("state %v progress %d/%d", got.State, got.ProgressDone, got.ProgressTotal)
	}
	if !mediaRowExists(t, db, want) {
		t.Error("the result is not in the library")
	}
	log := jobLog(q, j.ID)
	for _, s := range []string{"running on GPU worker gpu/fake", "retouch: step 10/10", "retouch: output " + want} {
		if !strings.Contains(log, s) {
			t.Errorf("log missing %q:\n%s", s, log)
		}
	}
	if engineDisp.targets[0].inflight != 0 {
		t.Error("the worker slot was not released")
	}
}

// A worker that refuses the job hands it to the next one.
func TestRemoteRetryOnAnotherNode(t *testing.T) {
	bad, good := newFakeNode(t, 1), newFakeNode(t, 1)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "a.png"), "png")
	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore", "--steps", "2"}, in)
	usePools(t, q, appconfig.EnginesConfig{Strategy: "pool", Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "bad", URL: bad.srv.URL}, {Name: "good", URL: good.srv.URL}}}}})
	engineDisp.checkAll(context.Background())
	bad.down.Store(true) // goes down after the health check: the submit fails
	// make "bad" the preferred pick
	bad.mu.Lock()
	bad.warm["loki-retouch"] = true
	bad.mu.Unlock()
	for _, tg := range engineDisp.targets {
		if tg.node.URL == bad.srv.URL {
			tg.mu.Lock()
			tg.info.Tools[0].Resident = []workerapi.Resident{{Key: "x", MB: 1}}
			tg.mu.Unlock()
		}
	}
	t.Setenv("PATH", t.TempDir())
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	if good.runs.Load() != 1 {
		t.Fatalf("the good node ran %d jobs\n%s", good.runs.Load(), jobLog(q, j.ID))
	}
	if !strings.Contains(jobLog(q, j.ID), "trying another worker") {
		t.Errorf("no retry note:\n%s", jobLog(q, j.ID))
	}
}

func TestGPUBucketSizedToPools(t *testing.T) {
	a, b := newFakeNode(t, 2), newFakeNode(t, 1)
	db := setupItemOpsDB(t)
	q, _ := newItemOpsJob(t, db, "retouch", nil, "x")
	usePools(t, q, appconfig.EnginesConfig{Strategy: "hybrid", Pools: []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{URL: a.srv.URL}, {URL: b.srv.URL}}}}})
	engineDisp.checkAll(context.Background())
	if got := q.HostLimit(HostBucketGPUDiffusion); got != 4 {
		t.Fatalf("gpu-diffusion limit %d, want 4 (local 1 + 2 + 1)", got)
	}
}
