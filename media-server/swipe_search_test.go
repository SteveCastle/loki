package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// ---- text mode ---------------------------------------------------------

// newSwipeTextDeps seeds an in-memory library where similarity to the query
// vector {1, 0} ranks a > b > orphan > c > d. The orphan has an embedding
// but no media row, so it must be filtered out of the ranked pages. The
// query vector is injected straight into the handler's text-vector cache —
// the real text encoder is a subprocess and has no place in a unit test.
func newSwipeTextDeps(t *testing.T, query string) *Dependencies {
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

	swipeTextVecMu.Lock()
	swipeTextVecs[query] = swipeTextVec{vec: []float32{1, 0}, model: model, at: time.Now()}
	swipeTextVecMu.Unlock()
	t.Cleanup(func() {
		swipeTextVecMu.Lock()
		delete(swipeTextVecs, query)
		swipeTextVecMu.Unlock()
	})

	return &Dependencies{DB: db}
}

func getSwipeSearch(t *testing.T, handler func(http.ResponseWriter, *http.Request, *Dependencies) bool, deps *Dependencies, url string) (media.APIResponse, *httptest.ResponseRecorder) {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, url, nil)
	rec := httptest.NewRecorder()
	if !handler(rec, req, deps) {
		t.Fatalf("expected the mode handler to handle %s", url)
	}
	var resp media.APIResponse
	if rec.Code == http.StatusOK {
		if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
			t.Fatalf("decode: %v (body: %s)", err, rec.Body.String())
		}
	}
	return resp, rec
}

func TestSwipeTextIgnoresOtherModes(t *testing.T) {
	deps := newSwipeTextDeps(t, "red bird")
	for _, url := range []string{"/swipe/api?offset=0&limit=5", "/swipe/api?mode=similar&anchor=x"} {
		req := httptest.NewRequest(http.MethodGet, url, nil)
		if maybeHandleSwipeTextSearch(httptest.NewRecorder(), req, deps) {
			t.Fatalf("handled %s without mode=text", url)
		}
	}
}

func TestSwipeTextRequiresQuery(t *testing.T) {
	deps := newSwipeTextDeps(t, "red bird")
	_, rec := getSwipeSearch(t, maybeHandleSwipeTextSearch, deps, "/swipe/api?mode=text&q=%20")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for blank q, got %d", rec.Code)
	}
}

func TestSwipeTextRanksAndFiltersOrphans(t *testing.T) {
	deps := newSwipeTextDeps(t, "red bird")
	resp, rec := getSwipeSearch(t, maybeHandleSwipeTextSearch, deps, "/swipe/api?mode=text&q=red+bird&offset=0&limit=3")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) != 3 {
		t.Fatalf("expected 3 items, got %d", len(resp.Items))
	}
	// Best-first; the orphan embedding never surfaces.
	want := []string{"/lib/a.jpg", "/lib/b.jpg", "/lib/c.jpg"}
	for i, w := range want {
		if resp.Items[i].Path != w {
			t.Fatalf("item %d = %s, want %s", i, resp.Items[i].Path, w)
		}
	}
	if !resp.HasMore {
		t.Fatal("expected has_more=true, d.jpg remains")
	}
}

func TestSwipeTextSecondPageComposes(t *testing.T) {
	deps := newSwipeTextDeps(t, "red bird")
	resp, rec := getSwipeSearch(t, maybeHandleSwipeTextSearch, deps, "/swipe/api?mode=text&q=red+bird&offset=3&limit=3")
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

// ---- face mode ---------------------------------------------------------

// newSwipeFaceDeps seeds stored faces under the active recognizer so that
// similarity to the (single) face in /lib/a.jpg ranks b > orphanface > c.
// b carries a second, much worse face — the handler must collapse it to
// one entry at its best rank. orphanface has a face row but no media row.
// Routing is pinned to "single" so no domain classification is attempted.
func newSwipeFaceDeps(t *testing.T) *Dependencies {
	t.Helper()
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })
	cfg := prev
	cfg.FaceRouting = "single"
	appconfig.Set(cfg)

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}

	for _, p := range []string{"/lib/a.jpg", "/lib/b.jpg", "/lib/c.jpg"} {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatal(err)
		}
	}

	model := tasks.ActiveFaceModel().ID
	face := func(vec []float32) media.NewFace {
		return media.NewFace{X: 0.1, Y: 0.1, W: 0.5, H: 0.5, Score: 0.99, Vec: vec}
	}
	seed := map[string][]media.NewFace{
		"/lib/a.jpg":          {face([]float32{1, 0})},
		"/lib/b.jpg":          {face([]float32{0.99, 0.14}), face([]float32{0, 1})},
		"/lib/orphanface.jpg": {face([]float32{0.9, 0.44})}, // no media row
		"/lib/c.jpg":          {face([]float32{0.7, 0.71})},
	}
	for p, faces := range seed {
		if _, err := media.ReplaceFaces(db, p, model, faces, 1); err != nil {
			t.Fatal(err)
		}
	}

	return &Dependencies{DB: db}
}

func TestSwipeFaceIgnoresOtherModes(t *testing.T) {
	deps := newSwipeFaceDeps(t)
	req := httptest.NewRequest(http.MethodGet, "/swipe/api?offset=0&limit=5", nil)
	if maybeHandleSwipeFace(httptest.NewRecorder(), req, deps) {
		t.Fatal("handled a request without mode=face")
	}
}

func TestSwipeFaceRequiresAnchor(t *testing.T) {
	deps := newSwipeFaceDeps(t)
	_, rec := getSwipeSearch(t, maybeHandleSwipeFace, deps, "/swipe/api?mode=face")
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", rec.Code)
	}
}

func TestSwipeFaceRanksCollapsesAndExcludesAnchor(t *testing.T) {
	deps := newSwipeFaceDeps(t)
	resp, rec := getSwipeSearch(t, maybeHandleSwipeFace, deps, "/swipe/api?mode=face&anchor=%2Flib%2Fa.jpg&offset=0&limit=1")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) != 1 || resp.Items[0].Path != "/lib/b.jpg" {
		t.Fatalf("unexpected first page: %+v", resp.Items)
	}
	if !resp.HasMore {
		t.Fatal("expected has_more=true, c.jpg remains")
	}
}

func TestSwipeFaceSecondPageComposes(t *testing.T) {
	deps := newSwipeFaceDeps(t)
	// Page 2: b's collapsed duplicate and the orphan face must not shift
	// offsets — the next real item is c, and the ranking ends there.
	resp, rec := getSwipeSearch(t, maybeHandleSwipeFace, deps, "/swipe/api?mode=face&anchor=%2Flib%2Fa.jpg&offset=1&limit=1")
	if rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body.String())
	}
	if len(resp.Items) != 1 || resp.Items[0].Path != "/lib/c.jpg" {
		t.Fatalf("unexpected second page: %+v", resp.Items)
	}
	if resp.HasMore {
		t.Fatal("expected has_more=false at end of ranking")
	}
}
