package main

import (
	"encoding/json"
	"net/http"
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
