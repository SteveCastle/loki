package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// newSwipeSimilarDeps seeds an in-memory library where similarity to
// /lib/a.jpg ranks b > orphan > c > d. The orphan has an embedding but no
// media row, so it must be filtered out of the ranked pages.
func newSwipeSimilarDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}

	for _, p := range []string{"/lib/a.jpg", "/lib/b.jpg", "/lib/c.jpg", "/lib/d.jpg"} {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatal(err)
		}
	}

	model := tasks.ActiveEmbedModel().ID
	vecs := map[string][]float32{
		"/lib/a.jpg":      {1, 0},
		"/lib/b.jpg":      {0.99, 0.14},
		"/lib/orphan.jpg": {0.9, 0.44}, // no media row
		"/lib/c.jpg":      {0.7, 0.71},
		"/lib/d.jpg":      {0, 1},
	}
	for p, v := range vecs {
		if err := media.UpsertEmbedding(db, p, model, v, 0); err != nil {
			t.Fatal(err)
		}
	}

	return &Dependencies{DB: db}
}

func getSwipeSimilar(t *testing.T, deps *Dependencies, url string) (media.APIResponse, *httptest.ResponseRecorder) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	if !maybeHandleSwipeSimilar(rec, req, deps) {
		t.Fatalf("expected similar mode to handle %s", url)
	}
	var resp media.APIResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v (body: %s)", err, rec.Body.String())
		}
	}
	return resp, rec
}

func TestSwipeSimilarIgnoresNonSimilarMode(t *testing.T) {
	deps := newSwipeSimilarDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/swipe/api?offset=0&limit=5", nil)
	if maybeHandleSwipeSimilar(httptest.NewRecorder(), req, deps) {
		t.Fatal("handled a request without mode=similar")
	}
}

func TestSwipeSimilarRequiresAnchor(t *testing.T) {
	deps := newSwipeSimilarDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/swipe/api?mode=similar", nil)
	rec := httptest.NewRecorder()
	if !maybeHandleSwipeSimilar(rec, req, deps) {
		t.Fatal("expected similar mode to be handled")
	}
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestSwipeSimilarRanksExcludesAnchorAndOrphans(t *testing.T) {
	deps := newSwipeSimilarDeps(t)
	resp, rec := getSwipeSimilar(t, deps, "/swipe/api?mode=similar&anchor=%2Flib%2Fa.jpg&offset=0&limit=2")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(resp.Items))
	}
	// Nearest-first, anchor and orphan excluded.
	if resp.Items[0].Path != "/lib/b.jpg" || resp.Items[1].Path != "/lib/c.jpg" {
		t.Fatalf("unexpected page: %s, %s", resp.Items[0].Path, resp.Items[1].Path)
	}
	if !resp.HasMore {
		t.Fatal("expected has_more=true, d.jpg remains")
	}
}

func TestSwipeSimilarSecondPageComposes(t *testing.T) {
	deps := newSwipeSimilarDeps(t)
	resp, rec := getSwipeSimilar(t, deps, "/swipe/api?mode=similar&anchor=%2Flib%2Fa.jpg&offset=2&limit=2")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) != 1 || resp.Items[0].Path != "/lib/d.jpg" {
		t.Fatalf("unexpected second page: %+v", resp.Items)
	}
	if resp.HasMore {
		t.Fatal("expected has_more=false at end of ranking")
	}
}
