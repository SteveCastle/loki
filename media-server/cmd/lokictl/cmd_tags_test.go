package main

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
)

func TestTagCreateSendsBody(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdTagCreate(a, []string{"sunset", "--category", "scene"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Method != "POST" || got.Path != "/api/tags" {
		t.Errorf("request = %+v", got)
	}
	var body tagBody
	_ = json.Unmarshal([]byte(got.Body), &body)
	if body.Label != "sunset" || body.CategoryLabel != "scene" {
		t.Errorf("body = %+v", body)
	}
}

func TestTagDeleteNeedsYesAndCategory(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdTagDelete(a, []string{"sunset", "--category", "scene"}); code != 2 {
		t.Errorf("without --yes exit = %d", code)
	}
	if len(*reqs) != 0 {
		t.Error("server hit without --yes")
	}
	if code := cmdTagDelete(a, []string{"sunset", "--category", "scene", "--yes"}); code != 0 {
		t.Errorf("with --yes exit = %d", code)
	}
	if (*reqs)[0].Method != "DELETE" {
		t.Errorf("method = %s", (*reqs)[0].Method)
	}
}

func TestTagAssign(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdTagAssign(a, []string{"C:/x.jpg", "sunset", "--category", "scene", "--timestamp", "12.5"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/assignments" {
		t.Errorf("path = %s", got.Path)
	}
	var body map[string]any
	_ = json.Unmarshal([]byte(got.Body), &body)
	if body["mediaPath"] != "C:/x.jpg" || body["tagLabel"] != "sunset" || body["categoryLabel"] != "scene" || body["timeStamp"] != 12.5 {
		t.Errorf("body = %v", body)
	}
}

func TestTagUnassignNestedBody(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdTagUnassign(a, []string{"C:/x.jpg", "sunset"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Method != "DELETE" || !strings.Contains(got.Body, `"tag_label":"sunset"`) {
		t.Errorf("request = %+v", got)
	}
}

func TestTaxonomyCategoryFlag(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `[]`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdTaxonomy(a, []string{"--category", "sc enes"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/taxonomy/tags" {
		t.Errorf("path = %s", got.Path)
	}
}

func TestBulkPathsStripsStdinBOM(t *testing.T) {
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	orig := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = orig }()
	w.WriteString("\ufeffC:/a.mp4\r\nC:/b.mp4\n")
	w.Close()
	paths, err := bulkPaths(nil, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 || paths[0] != "C:/a.mp4" || paths[1] != "C:/b.mp4" {
		t.Errorf("paths = %q", paths)
	}
}

func TestCategoryDeleteRefusesToUntagWithoutAllowNonEmpty(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{"label":"Anime","exists":true,"tags":0,"groups":0,"assignments":148,"media":140}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdCategoryDelete(a, []string{"Anime", "--yes"}); code != 2 {
		t.Fatalf("exit = %d, want 2", code)
	}
	for _, r := range *reqs {
		if r.Method == "DELETE" {
			t.Fatal("deleted a category that still owns assignments")
		}
	}
	if code := cmdCategoryDelete(a, []string{"Anime", "--yes", "--allow-nonempty"}); code != 0 {
		t.Fatalf("with --allow-nonempty exit = %d", code)
	}
	if last := (*reqs)[len(*reqs)-1]; last.Method != "DELETE" || last.Path != "/api/categories" {
		t.Errorf("last request = %+v", last)
	}
}

func TestCategoryDeleteRefusesWhenServerCannotPreview(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusNotFound, `404 page not found`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdCategoryDelete(a, []string{"Anime", "--yes", "--allow-nonempty"}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if len(*reqs) != 1 || (*reqs)[0].Method != "GET" {
		t.Fatalf("requests = %+v, want only the preview", *reqs)
	}
}

func TestCategoryDeleteEmptyNeedsOnlyYes(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{"label":"Empty","exists":true}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdCategoryDelete(a, []string{"Empty", "--dry-run"}); code != 0 || len(*reqs) != 1 {
		t.Fatalf("dry run exit = %d, requests = %d", code, len(*reqs))
	}
	if code := cmdCategoryDelete(a, []string{"Empty"}); code != 2 {
		t.Fatalf("without --yes exit = %d", code)
	}
	if code := cmdCategoryDelete(a, []string{"Empty", "--yes"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
}
