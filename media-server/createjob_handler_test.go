package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/storage"
)

// newCreateJobTestDeps returns a minimal Dependencies wired for createJobHandler
// tests: in-memory Queue (no DB), nil storage registry. We do NOT execute the
// queued job — these tests only verify the request-decode → arguments path.
func newCreateJobTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	return &Dependencies{
		Queue:   jobqueue.NewQueue(),
		Storage: storage.NewRegistry(nil),
	}
}

func TestCreateJobHandler_AppendsPromptField(t *testing.T) {
	deps := newCreateJobTestDeps(t)
	body := map[string]any{
		"input": `metadata --type description --apply all --overwrite "C:/tmp/x.jpg"`,
		"fields": map[string]string{
			"prompt": `weird "quoted" override
on two lines`,
		},
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/create", strings.NewReader(string(b)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	createJobHandler(deps).ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body = %s", rr.Code, rr.Body.String())
	}
	var resp struct{ ID string }
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("response body not JSON: %v", err)
	}
	if resp.ID == "" {
		t.Fatal("expected non-empty job id in response")
	}

	job := deps.Queue.GetJob(resp.ID)
	if job == nil {
		t.Fatalf("queued job %q not found", resp.ID)
	}
	if job.Command != "metadata" {
		t.Errorf("Command = %q, want %q", job.Command, "metadata")
	}

	// Find the --prompt flag in Arguments and assert its value matches what
	// we sent, verbatim (quotes, newlines, all preserved).
	found := false
	for i := 0; i < len(job.Arguments)-1; i++ {
		if job.Arguments[i] == "--prompt" {
			got := job.Arguments[i+1]
			want := body["fields"].(map[string]string)["prompt"]
			if got != want {
				t.Errorf("prompt arg = %q, want %q", got, want)
			}
			found = true
			break
		}
	}
	if !found {
		t.Errorf("--prompt not found in Arguments: %v", job.Arguments)
	}
}

func TestCreateJobHandler_NoFieldsBehavesAsBefore(t *testing.T) {
	deps := newCreateJobTestDeps(t)
	body := map[string]any{
		"input": `metadata --type description --apply all --overwrite "C:/tmp/x.jpg"`,
	}
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/create", strings.NewReader(string(b)))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()

	createJobHandler(deps).ServeHTTP(rr, req)

	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201; body = %s", rr.Code, rr.Body.String())
	}
	var resp struct{ ID string }
	_ = json.Unmarshal(rr.Body.Bytes(), &resp)
	job := deps.Queue.GetJob(resp.ID)
	if job == nil {
		t.Fatalf("queued job not found")
	}
	for _, a := range job.Arguments {
		if a == "--prompt" {
			t.Errorf("unexpected --prompt in Arguments when fields omitted: %v", job.Arguments)
		}
	}
}
