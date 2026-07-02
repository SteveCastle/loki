package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadDAGFormats(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"bare.json":    `[{"id":"a","command":"wait"}]`,
		"wrapped.json": `{"dag":[{"id":"a","command":"wait"}]}`,
		"tasks.json":   `{"tasks":[{"id":"a","command":"wait"}]}`,
	} {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		dag, err := readDAG(p, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if len(dag) != 1 || dag[0]["id"] != "a" {
			t.Errorf("%s: dag = %v", name, dag)
		}
	}

	// stdin
	dag, err := readDAG("-", strings.NewReader(`[{"id":"s","command":"wait"}]`))
	if err != nil || len(dag) != 1 {
		t.Errorf("stdin dag = %v err = %v", dag, err)
	}

	if _, err := readDAG("-", strings.NewReader(`{"nope":true}`)); err == nil {
		t.Error("expected error for empty DAG")
	}
}

type recordedRequest struct {
	Method string
	Path   string
	Body   string
}

func newRecordingServer(t *testing.T, status int, respBody string) (*httptest.Server, *[]recordedRequest) {
	t.Helper()
	var reqs []recordedRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		reqs = append(reqs, recordedRequest{r.Method, r.URL.Path, string(b)})
		w.WriteHeader(status)
		_, _ = w.Write([]byte(respBody))
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs
}

func TestWorkflowCreate(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusCreated, `{"id":"wf1","name":"n","dag":[]}`)
	dagFile := filepath.Join(t.TempDir(), "dag.json")
	_ = os.WriteFile(dagFile, []byte(`[{"id":"a","command":"wait"}]`), 0o600)

	a, out, _ := appForServer(srv.URL)
	code := cmdWorkflowCreate(a, []string{"--name", "n", "--dag", dagFile})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if len(*reqs) != 1 {
		t.Fatalf("reqs = %+v", *reqs)
	}
	got := (*reqs)[0]
	if got.Method != "POST" || got.Path != "/workflows/create" {
		t.Errorf("request = %+v", got)
	}
	var body struct {
		Name string           `json:"name"`
		DAG  []map[string]any `json:"dag"`
	}
	if err := json.Unmarshal([]byte(got.Body), &body); err != nil || body.Name != "n" || len(body.DAG) != 1 {
		t.Errorf("body = %s", got.Body)
	}
	if !strings.Contains(out.String(), "wf1") {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestWorkflowDeleteNeedsYes(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusNoContent, "")
	a, _, _ := appForServer(srv.URL)
	if code := cmdWorkflowDelete(a, []string{"wf1"}); code != 2 {
		t.Errorf("without --yes exit = %d, want 2", code)
	}
	if len(*reqs) != 0 {
		t.Error("server hit without --yes")
	}
	if code := cmdWorkflowDelete(a, []string{"wf1", "--yes"}); code != 0 {
		t.Errorf("with --yes exit = %d", code)
	}
	if len(*reqs) != 1 || (*reqs)[0].Method != "DELETE" || (*reqs)[0].Path != "/workflows/wf1" {
		t.Errorf("reqs = %+v", *reqs)
	}
}

func TestWorkflowRun(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusCreated, `{"ids":["j1","j2"]}`)
	a, out, _ := appForServer(srv.URL)
	code := cmdWorkflowRun(a, []string{"wf1", "--input", "hello"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/workflows/wf1/run" || !strings.Contains(got.Body, `"hello"`) {
		t.Errorf("request = %+v", got)
	}
	if !strings.Contains(out.String(), "j1") {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestWorkflowRunAdhoc(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusCreated, `{"ids":["j1"]}`)
	dagFile := filepath.Join(t.TempDir(), "dag.json")
	_ = os.WriteFile(dagFile, []byte(`{"tasks":[{"id":"a","command":"wait","input":"1"}]}`), 0o600)
	a, _, _ := appForServer(srv.URL)
	code := cmdWorkflowRunAdhoc(a, []string{"--dag", dagFile})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Method != "POST" || got.Path != "/workflow" || !strings.Contains(got.Body, `"tasks"`) {
		t.Errorf("request = %+v", got)
	}
}

func TestWorkflowListOldServer404Hint(t *testing.T) {
	srv, _ := newRecordingServer(t, http.StatusNotFound, "404 page not found")
	a, _, errOut := appForServer(srv.URL)
	code := 0
	for _, c := range commands {
		if c.group == "workflow" && c.name == "list" {
			code = c.run(a, nil)
		}
	}
	if code != 1 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(errOut.String(), "build:server") {
		t.Errorf("stderr = %s", errOut.String())
	}
}
