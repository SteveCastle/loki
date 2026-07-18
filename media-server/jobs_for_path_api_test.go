package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/tasks"
)

// newJobsForPathEnv wires the create + for-path handlers over a real queue
// with the real tasks items resolver — but no runners, so created jobs stay
// pending and remain visible in the path index.
func newJobsForPathEnv(t *testing.T) (*Dependencies, *http.ServeMux) {
	t.Helper()
	jobqueue.SetItemsResolver(tasks.ResolveItems)
	t.Cleanup(func() { jobqueue.SetItemsResolver(nil) })

	deps := &Dependencies{Queue: jobqueue.NewQueue()}
	mux := http.NewServeMux()
	mux.HandleFunc("/create", createJobHandler(deps))
	mux.HandleFunc("/api/jobs/for-path", jobsForPathHandler(deps))
	return deps, mux
}

func getJobsForPath(t *testing.T, mux *http.ServeMux, path string) (int, []jobForPathSummary) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/jobs/for-path?path="+url.QueryEscape(path), nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		return rr.Code, nil
	}
	var resp struct {
		Jobs []jobForPathSummary `json:"jobs"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v: %s", err, rr.Body.String())
	}
	return rr.Code, resp.Jobs
}

// TestJobsForPathEndpoint drives the real flow the SPA uses: POST /create
// with a single-file transcribe job, then look the file up by path.
func TestJobsForPathEndpoint(t *testing.T) {
	_, mux := newJobsForPathEnv(t)

	p := filepath.Join(t.TempDir(), "episode.mp3")
	rr := jobCtlPost(t, mux, "/create", map[string]any{
		"input": `transcribe --overwrite "` + p + `"`,
	})
	if rr.Code != http.StatusCreated {
		t.Fatalf("create status = %d: %s", rr.Code, rr.Body.String())
	}
	var created struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &created)

	code, jobs := getJobsForPath(t, mux, p)
	if code != http.StatusOK {
		t.Fatalf("for-path status = %d", code)
	}
	if len(jobs) != 1 {
		t.Fatalf("got %d jobs, want 1", len(jobs))
	}
	j := jobs[0]
	if j.ID != created.ID || j.Command != "transcribe" || j.State != jobqueue.StatePending {
		t.Errorf("job = %+v, want id=%s command=transcribe state=pending", j, created.ID)
	}

	// A different path in the same directory matches nothing.
	if _, other := getJobsForPath(t, mux, filepath.Join(filepath.Dir(p), "other.mp3")); len(other) != 0 {
		t.Errorf("unrelated path returned %d jobs", len(other))
	}
}

func TestJobsForPathValidation(t *testing.T) {
	_, mux := newJobsForPathEnv(t)

	req := httptest.NewRequest(http.MethodGet, "/api/jobs/for-path", nil)
	rr := httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Errorf("missing path: status = %d, want 400", rr.Code)
	}

	req = httptest.NewRequest(http.MethodPost, "/api/jobs/for-path?path=x", nil)
	rr = httptest.NewRecorder()
	mux.ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST: status = %d, want 405", rr.Code)
	}
}

// TestJobsForPathQueryJobHidesUntilResolved documents the design: a query
// job is invisible to path lookups until its input resolves (SetJobItems).
func TestJobsForPathQueryJobHidesUntilResolved(t *testing.T) {
	deps, mux := newJobsForPathEnv(t)

	rr := jobCtlPost(t, mux, "/create", map[string]any{
		"input": "embed --query64 dGFnOmNhdA==",
	})
	if rr.Code != http.StatusCreated {
		t.Fatalf("create status = %d: %s", rr.Code, rr.Body.String())
	}
	var created struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &created)

	p := filepath.Join(t.TempDir(), "cat.jpg")
	if _, jobs := getJobsForPath(t, mux, p); len(jobs) != 0 {
		t.Fatalf("unresolved query job leaked into path lookup: %d jobs", len(jobs))
	}

	// Simulate the runner resolving the query to a concrete item list.
	if job, err := deps.Queue.ClaimJob(); err != nil || job == nil {
		t.Fatalf("ClaimJob = %v, %v", job, err)
	}
	deps.Queue.SetJobItems(created.ID, []string{p})

	if _, jobs := getJobsForPath(t, mux, p); len(jobs) != 1 || jobs[0].ID != created.ID {
		t.Fatalf("resolved query job not returned for its path")
	}
}
