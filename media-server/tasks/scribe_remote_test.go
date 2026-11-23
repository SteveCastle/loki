package tasks

import (
	"context"
	"encoding/json"
	"errors"
	"image"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/workerapi"
)

// scribeNode is a loki-worker serving only loki-scribe: it records each job's
// spec and uploads and answers with one engine result line, the way the real
// worker relays a `--json` run.
type scribeNode struct {
	srv     *httptest.Server
	mu      sync.Mutex
	specs   []workerapi.JobSpec
	uploads [][]string
}

func newScribeNode(t *testing.T, answer string) *scribeNode {
	t.Helper()
	n := &scribeNode{}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/info", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(workerapi.Info{Worker: "fake", Version: "1", Ready: true, Slots: 1,
			Tools: []workerapi.ToolInfo{{Name: "loki-scribe", Help: "--together --max-image-tokens --dry-multiplier --worker"}}})
	})
	mux.HandleFunc("POST /v1/jobs", func(w http.ResponseWriter, r *http.Request) {
		mr, err := r.MultipartReader()
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		var spec workerapi.JobSpec
		var parts []string
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
			parts = append(parts, part.FormName())
			_, _ = io.Copy(io.Discard, part)
		}
		n.mu.Lock()
		n.specs = append(n.specs, spec)
		n.uploads = append(n.uploads, parts)
		n.mu.Unlock()
		line, _ := json.Marshal(scribeResult{Index: 0, Text: answer, Finish: "stop"})
		w.WriteHeader(http.StatusAccepted)
		_ = json.NewEncoder(w).Encode(workerapi.JobStatus{ID: "j1", Tool: spec.Tool, State: workerapi.JobDone, Stdout: []string{string(line)}})
	})
	mux.HandleFunc("GET /v1/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {
		line, _ := json.Marshal(scribeResult{Index: 0, Text: answer, Finish: "stop"})
		_ = json.NewEncoder(w).Encode(workerapi.JobStatus{ID: r.PathValue("id"), Tool: "loki-scribe", State: workerapi.JobDone, Stdout: []string{string(line)}})
	})
	mux.HandleFunc("DELETE /v1/jobs/{id}", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("POST /v1/jobs/{id}/cancel", func(w http.ResponseWriter, r *http.Request) {})
	n.srv = httptest.NewServer(mux)
	t.Cleanup(n.srv.Close)
	return n
}

func writePNGForTest(t *testing.T, path string) string {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, image.NewRGBA(image.Rect(0, 0, 8, 8))); err != nil {
		t.Fatal(err)
	}
	return path
}

// Text work routed to a pool runs there: an Enhance request with two
// reference images reaches the worker as one --together run with both
// pictures uploaded, and its answer comes back.
func TestEnhancePromptOnWorkerNode(t *testing.T) {
	node := newScribeNode(t, "Optimized prompt: the key from <image2> on the hill of <image1>")
	usePools(t, nil, appconfig.EnginesConfig{
		Pools:  []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "fake", URL: node.srv.URL}}}},
		Routes: map[string]string{"loki-scribe": "pool"},
	})
	if InferenceHostIsLocal() {
		t.Fatal("loki-scribe routed to the pool still reports local")
	}
	// the engine is NOT on this machine's PATH: only the worker has it
	t.Setenv("PATH", t.TempDir())
	dir := t.TempDir()
	a := writePNGForTest(t, filepath.Join(dir, "hill.png"))
	b := writePNGForTest(t, filepath.Join(dir, "key.png"))
	res, err := EnhancePrompt(context.Background(), EnhanceRequest{
		Prompt: "put the key from <image2> on <image1>",
		Model:  "qwen-image-edit",
		Task:   "edit",
		References: []EnhanceReference{
			{Token: "<image1>", Kind: "image", Path: a},
			{Token: "<image2>", Kind: "image", Path: b},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Prompt != "the key from <image2> on the hill of <image1>" || len(res.MissingTokens) != 0 {
		t.Errorf("result = %+v", res)
	}
	if len(node.specs) != 1 {
		t.Fatalf("worker ran %d jobs; want 1", len(node.specs))
	}
	spec := node.specs[0]
	if spec.Tool != "loki-scribe" {
		t.Errorf("tool = %q", spec.Tool)
	}
	i := slices.Index(spec.Args, "--together")
	if i < 0 {
		t.Fatalf("no --together run: %q", spec.Args)
	}
	imgs := spec.Args[i+1:]
	if len(imgs) != 2 {
		t.Fatalf("images after --together = %q; want 2", imgs)
	}
	for _, p := range imgs {
		if !strings.HasPrefix(p, workerapi.JobDir) {
			t.Errorf("image arg %q is not a staged upload", p)
		}
	}
	if len(node.uploads[0]) != 2 {
		t.Errorf("uploaded %q; want both images", node.uploads[0])
	}
	if !slices.Contains(spec.Args, "--max-image-tokens") {
		t.Errorf("no vision budget: %q", spec.Args)
	}
}

// The Enhance button's rewrite is a queued job: it takes the text-model and
// local-compute buckets like a describe, reports its state by id while it
// waits and runs, and hands back the rewrite once it has finished.
func TestEnhancePromptQueuedJob(t *testing.T) {
	node := newScribeNode(t, "Prompt: a calm lake at dawn, mist over the water")
	usePools(t, nil, appconfig.EnginesConfig{
		Pools:  []appconfig.WorkerPool{{Name: "gpu", Nodes: []appconfig.WorkerNode{{Name: "fake", URL: node.srv.URL}}}},
		Routes: map[string]string{"loki-scribe": "pool"},
	})
	t.Setenv("PATH", t.TempDir())
	jobqueue.SetHostResolver(ResolveHost)
	jobqueue.SetResourceResolver(ResolveResources)
	t.Cleanup(func() {
		jobqueue.SetHostResolver(nil)
		jobqueue.SetResourceResolver(nil)
	})
	q := jobqueue.NewQueueWithDB(setupItemOpsDB(t))

	if _, err := QueueEnhanceJob(q, EnhanceRequest{Prompt: " ", Model: "flux2-klein"}); !errors.Is(err, ErrEnhanceInput) {
		t.Fatalf("empty prompt: %v; want ErrEnhanceInput", err)
	}
	id, err := QueueEnhanceJob(q, EnhanceRequest{Prompt: "a lake", Model: "flux2-klein", Task: "generate"})
	if err != nil {
		t.Fatal(err)
	}
	if st, ok := EnhanceJobStatus(q, id); !ok || st.State != "pending" {
		t.Fatalf("queued status = %+v %v; want pending", st, ok)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim: %v %v", j, err)
	}
	if j.Host != HostBucketScribe {
		t.Errorf("host = %q; want %q", j.Host, HostBucketScribe)
	}
	if err := enhancePromptTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatal(err)
	}
	st, ok := EnhanceJobStatus(q, id)
	if !ok || st.State != "completed" || st.Result == nil || st.Result.Prompt != "a calm lake at dawn, mist over the water" {
		t.Fatalf("finished status = %+v", st)
	}
	if got := q.GetJob(id); got.State != jobqueue.StateCompleted {
		t.Errorf("job state = %v", got.State)
	}
	if _, ok := EnhanceJobStatus(q, "nope"); ok {
		t.Error("an unknown id has a status")
	}
}

// A pending enhance job that outlived its server (nobody waits for it) is
// skipped without loading the text model.
func TestEnhanceJobWithoutWaiterIsSkipped(t *testing.T) {
	q := jobqueue.NewQueueWithDB(setupItemOpsDB(t))
	id, err := q.AddJob("", enhancePromptCommand, []string{"--request", `{"prompt":"x","model":"sdxl"}`}, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	j, _ := q.ClaimJob()
	if err := enhancePromptTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("ran a job nobody waits for")
	}
	if got := q.GetJob(id); got.State != jobqueue.StateError {
		t.Errorf("state = %v; want error", got.State)
	}
}
