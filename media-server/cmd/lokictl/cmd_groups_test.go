package main

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestGroupListEscapesCategory(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `[]`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdGroupList(a, []string{"--category", "TV Shows"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if got := (*reqs)[0]; got.Method != "GET" || got.Path != "/api/taxonomy/groups" {
		t.Errorf("request = %+v", got)
	}
}

func TestGroupCreateNeedsCategory(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdGroupCreate(a, []string{"Dirty Pair"}); code != 2 {
		t.Errorf("without --category exit = %d", code)
	}
	if code := cmdGroupCreate(a, []string{"Dirty Pair", "--category", "TV Shows"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	var body groupBody
	_ = json.Unmarshal([]byte(got.Body), &body)
	if got.Method != "POST" || got.Path != "/api/tag-groups" || body.Label != "Dirty Pair" || body.CategoryLabel != "TV Shows" {
		t.Errorf("request = %+v body = %+v", got, body)
	}
}

func TestGroupDeleteNeedsYes(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdGroupDelete(a, []string{"Dirty Pair", "--category", "TV Shows"}); code != 2 {
		t.Errorf("without --yes exit = %d", code)
	}
	if len(*reqs) != 0 {
		t.Error("server hit without --yes")
	}
	if code := cmdGroupDelete(a, []string{"Dirty Pair", "--category", "TV Shows", "--yes"}); code != 0 {
		t.Errorf("with --yes exit = %d", code)
	}
	if (*reqs)[0].Method != "DELETE" {
		t.Errorf("method = %s", (*reqs)[0].Method)
	}
}

func TestGroupAddCreatesThenGroupsEachTag(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdGroupAdd(a, []string{"Dirty Pair", "--category", "TV Shows", "Dirty Pair S01", "Dirty Pair S02"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if len(*reqs) != 3 {
		t.Fatalf("requests = %d, want 3", len(*reqs))
	}
	if (*reqs)[0].Path != "/api/tag-groups" {
		t.Errorf("first request = %+v", (*reqs)[0])
	}
	for i, want := range []string{"Dirty Pair S01", "Dirty Pair S02"} {
		got := (*reqs)[i+1]
		var body map[string]string
		_ = json.Unmarshal([]byte(got.Body), &body)
		if got.Method != "PUT" || got.Path != "/api/tags/group" || body["label"] != want || body["groupLabel"] != "Dirty Pair" {
			t.Errorf("request %d = %+v", i+1, got)
		}
	}
}

func TestGroupRemoveSendsEmptyGroup(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdGroupRemove(a, []string{"Dirty Pair S01"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	var body map[string]string
	_ = json.Unmarshal([]byte((*reqs)[0].Body), &body)
	if body["label"] != "Dirty Pair S01" || body["groupLabel"] != "" {
		t.Errorf("body = %v", body)
	}
}

func TestCategoryGroupingValidatesMode(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdCategoryGrouping(a, []string{"TV Shows", "maybe"}); code != 2 {
		t.Errorf("bad mode exit = %d", code)
	}
	if code := cmdCategoryGrouping(a, []string{"TV Shows", "on"}); code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if got := (*reqs)[0]; got.Path != "/api/categories/tag-grouping" {
		t.Errorf("request = %+v", got)
	}
}

func TestGroupHideAndView(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{}`)
	a, _, _ := appForServer(srv.URL)
	if code := cmdGroupHide(a, []string{"Dirty Pair", "--category", "TV Shows"}); code != 0 {
		t.Fatalf("hide exit = %d", code)
	}
	if code := cmdGroupView(a, []string{"Dirty Pair", "grid", "--category", "TV Shows"}); code != 2 {
		t.Errorf("bad mode exit = %d", code)
	}
	if code := cmdGroupView(a, []string{"Dirty Pair", "auto", "--category", "TV Shows"}); code != 0 {
		t.Fatalf("view exit = %d", code)
	}
	var hide, view map[string]any
	_ = json.Unmarshal([]byte((*reqs)[0].Body), &hide)
	_ = json.Unmarshal([]byte((*reqs)[1].Body), &view)
	if (*reqs)[0].Path != "/api/tag-groups/hidden" || hide["hidden"] != true {
		t.Errorf("hide request = %+v", (*reqs)[0])
	}
	// auto clears the override: no viewMode in the body, which the server reads as "".
	if (*reqs)[1].Path != "/api/tag-groups/view-mode" || view["viewMode"] != nil {
		t.Errorf("view request = %+v", (*reqs)[1])
	}
}
