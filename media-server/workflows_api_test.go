package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

// newWorkflowTestDeps wires a Dependencies with a DB-backed queue so the
// saved-workflow CRUD handlers (which persist to the workflows table) work.
func newWorkflowTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("failed to open in-memory db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return &Dependencies{Queue: jobqueue.NewQueueWithDB(db), DB: db}
}

func TestWorkflowsAPI_CreateListRunDelete(t *testing.T) {
	deps := newWorkflowTestDeps(t)

	// Create
	body := `{"name":"w1","dag":[{"id":"a","command":"wait","input":"1"}]}`
	req := httptest.NewRequest(http.MethodPost, "/workflows/create", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	workflowCreateHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("create status = %d, want 201; body = %s", rr.Code, rr.Body.String())
	}
	var created struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &created); err != nil {
		t.Fatalf("create response not JSON: %v", err)
	}
	if created.ID == "" || created.Name != "w1" {
		t.Fatalf("unexpected create response: %+v", created)
	}

	// List
	req = httptest.NewRequest(http.MethodGet, "/workflows", nil)
	rr = httptest.NewRecorder()
	workflowsListHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("list status = %d, want 200", rr.Code)
	}
	var list []struct {
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &list); err != nil {
		t.Fatalf("list response not JSON: %v; body = %s", err, rr.Body.String())
	}
	if len(list) != 1 || list[0].Name != "w1" {
		t.Fatalf("unexpected list: %+v", list)
	}

	// Run
	req = httptest.NewRequest(http.MethodPost, "/workflows/"+created.ID+"/run", strings.NewReader(`{"input":"2"}`))
	req.SetPathValue("id", created.ID)
	rr = httptest.NewRecorder()
	workflowRunHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("run status = %d, want 201; body = %s", rr.Code, rr.Body.String())
	}
	var run struct {
		IDs []string `json:"ids"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &run); err != nil {
		t.Fatalf("run response not JSON: %v", err)
	}
	if len(run.IDs) != 1 {
		t.Fatalf("expected 1 job id, got %v", run.IDs)
	}

	// Delete
	req = httptest.NewRequest(http.MethodDelete, "/workflows/"+created.ID, nil)
	req.SetPathValue("id", created.ID)
	rr = httptest.NewRecorder()
	workflowDetailHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusNoContent {
		t.Fatalf("delete status = %d, want 204; body = %s", rr.Code, rr.Body.String())
	}
}

func TestWorkflowsAPI_DetailGetAndUpdate(t *testing.T) {
	deps := newWorkflowTestDeps(t)

	wf, err := deps.Queue.CreateWorkflow("orig", []jobqueue.WorkflowTask{{ID: "a", Command: "wait"}})
	if err != nil {
		t.Fatalf("CreateWorkflow: %v", err)
	}

	// GET detail
	req := httptest.NewRequest(http.MethodGet, "/workflows/"+wf.ID, nil)
	req.SetPathValue("id", wf.ID)
	rr := httptest.NewRecorder()
	workflowDetailHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("get status = %d, want 200", rr.Code)
	}

	// PUT update
	body := `{"name":"renamed","dag":[{"id":"a","command":"wait","input":"3"}]}`
	req = httptest.NewRequest(http.MethodPut, "/workflows/"+wf.ID, strings.NewReader(body))
	req.SetPathValue("id", wf.ID)
	rr = httptest.NewRecorder()
	workflowDetailHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("update status = %d, want 200; body = %s", rr.Code, rr.Body.String())
	}
	var updated struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &updated); err != nil {
		t.Fatalf("update response not JSON: %v", err)
	}
	if updated.Name != "renamed" {
		t.Fatalf("update name = %q, want renamed", updated.Name)
	}
}
