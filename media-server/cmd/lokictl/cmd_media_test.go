package main

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
)

func TestBuildPredicates(t *testing.T) {
	preds := buildPredicates(
		stringList{"cat", "dog"}, stringList{"blurry"},
		"", "sunset", "", "", "beach at dusk", "OR",
	)
	want := []predicate{
		{Type: "tag", Value: "cat", Join: "OR"},
		{Type: "tag", Value: "dog", Join: "OR"},
		{Type: "tag", Value: "blurry", Exclude: true, Join: "OR"},
		{Type: "description", Value: "sunset", Join: "OR"},
		{Type: "visual", Value: "beach at dusk", Join: "OR"},
	}
	if len(preds) != len(want) {
		t.Fatalf("preds = %+v", preds)
	}
	for i := range want {
		if preds[i] != want[i] {
			t.Errorf("preds[%d] = %+v, want %+v", i, preds[i], want[i])
		}
	}
}

func TestMediaQuerySendsPredicates(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `[]`)
	a, _, _ := appForServer(srv.URL)
	code := cmdMediaQuery(a, []string{"--tag", "cat", "--exclude-tag", "dog", "--mode", "and"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Method != "POST" || got.Path != "/api/media/query" {
		t.Errorf("request = %+v", got)
	}
	var body struct {
		Predicates []predicate `json:"predicates"`
		Mode       string      `json:"mode"`
	}
	if err := json.Unmarshal([]byte(got.Body), &body); err != nil {
		t.Fatalf("body: %v", err)
	}
	if body.Mode != "AND" || len(body.Predicates) != 2 || !body.Predicates[1].Exclude {
		t.Errorf("body = %+v", body)
	}
}

func TestMediaQueryNoPredicatesIsUsageError(t *testing.T) {
	a, _, _ := appForServer("http://127.0.0.1:1")
	if code := cmdMediaQuery(a, nil); code != 2 {
		t.Errorf("exit = %d, want 2", code)
	}
}

func TestMediaSimilarEncodesPath(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `[]`)
	a, _, _ := appForServer(srv.URL)
	code := cmdMediaSimilar(a, []string{`C:\pics\a b.jpg`, "--limit", "5"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/media/similar" {
		t.Errorf("path = %q", got.Path)
	}
}

func TestMediaDescribe(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdMediaDescribe(a, []string{"C:/x.jpg", "--text", "a sunset"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/media/description" || !strings.Contains(got.Body, `"a sunset"`) {
		t.Errorf("request = %+v", got)
	}
}

func TestMediaDeleteNeedsYes(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdMediaDelete(a, []string{"C:/x.jpg"}); code != 2 {
		t.Errorf("without --yes exit = %d", code)
	}
	if len(*reqs) != 0 {
		t.Error("server hit without --yes")
	}
	if code := cmdMediaDelete(a, []string{"C:/x.jpg", "--yes"}); code != 0 {
		t.Errorf("with --yes exit = %d", code)
	}
}

func TestDBQueryCommand(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{"columns":["n"],"rows":[[1]],"row_count":1}`)
	a, out, _ := appForServer(srv.URL)
	code := cmdDBQuery(a, []string{"SELECT COUNT(*) AS n FROM media WHERE hash = ?", "--arg", "abc", "--limit", "10"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/db/query" {
		t.Errorf("path = %q", got.Path)
	}
	var body dbQueryBody
	if err := json.Unmarshal([]byte(got.Body), &body); err != nil {
		t.Fatal(err)
	}
	if body.Limit != 10 || len(body.Args) != 1 || body.Args[0] != "abc" || !strings.HasPrefix(body.SQL, "SELECT COUNT") {
		t.Errorf("body = %+v", body)
	}
	if !strings.Contains(out.String(), `"row_count"`) {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestDBSchemaWithTable(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{"columns":[],"rows":[]}`)
	a, _, _ := appForServer(srv.URL)
	var run func(a *App, args []string) int
	for _, c := range commands {
		if c.group == "db" && c.name == "schema" {
			run = c.run
		}
	}
	if code := run(a, []string{"media"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if !strings.Contains(got.Body, "name = ?") || !strings.Contains(got.Body, `"media"`) {
		t.Errorf("body = %s", got.Body)
	}
}
