package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// newTreemapTestDeps extends the shared index test DB (media + media_embedding)
// with the tag and face tables the treemap metrics scan.
func newTreemapTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	deps := newIndexTestDeps(t)
	for _, stmt := range []string{
		`CREATE TABLE media_tag_by_category (
			media_path TEXT, tag_label TEXT, category_label TEXT,
			weight REAL, time_stamp REAL, created_at INTEGER)`,
		`CREATE TABLE face (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			media_path TEXT NOT NULL, model TEXT NOT NULL, vector BLOB)`,
	} {
		if _, err := deps.DB.Exec(stmt); err != nil {
			t.Fatalf("create table: %v", err)
		}
	}
	return deps
}

func seedTreemapMedia(t *testing.T, deps *Dependencies, paths ...string) {
	t.Helper()
	for _, p := range paths {
		if _, err := deps.DB.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatalf("insert media %q: %v", p, err)
		}
	}
}

type treemapResp struct {
	Metric     string       `json:"metric"`
	Path       string       `json:"path"`
	Total      int64        `json:"total"`
	TotalItems int64        `json:"totalItems"`
	Node       *treemapNode `json:"node"`
}

func getTreemap(t *testing.T, api *treemapAPI, query string) treemapResp {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/api/treemap"+query, nil)
	rr := httptest.NewRecorder()
	api.dataHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var resp treemapResp
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	return resp
}

func childByName(n *treemapNode, name string) *treemapNode {
	for _, c := range n.Children {
		if c.Name == name {
			return c
		}
	}
	return nil
}

func TestTreemapItemsAggregatesDirectories(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps,
		`V:\A\1.jpg`, `V:\A\2.jpg`, `V:\A\sub\3.jpg`, `V:\B\4.jpg`,
		`C:/x/5.jpg`, // mixed separators must coexist
	)
	api := newTreemapAPI(deps)

	resp := getTreemap(t, api, "?metric=items")
	if resp.Total != 5 || resp.Node.Count != 5 {
		t.Fatalf("expected 5 items at root, got total=%d count=%d", resp.Total, resp.Node.Count)
	}
	v := childByName(resp.Node, "V:")
	if v == nil || v.Count != 4 {
		t.Fatalf("expected V: with 4 items, got %+v", v)
	}
	a := childByName(v, "A")
	if a == nil || a.Count != 3 || a.Direct != 2 {
		t.Fatalf("expected A count=3 direct=2, got %+v", a)
	}
	if sub := childByName(a, "sub"); sub == nil || sub.Count != 1 || sub.Path != `V:\A\sub` {
		t.Fatalf("expected sub count=1 path=V:\\A\\sub, got %+v", sub)
	}
	if c := childByName(resp.Node, "C:"); c == nil || c.Count != 1 {
		t.Fatalf("expected C: with 1 item, got %+v", c)
	}

	// Drill into a directory via ?path= — full prefix keys must round-trip.
	drilled := getTreemap(t, api, "?metric=items&path="+`V:%5CA`)
	if drilled.Node.Count != 3 || drilled.Node.Direct != 2 || drilled.Node.Path != `V:\A` {
		t.Fatalf("drill-down mismatch: %+v", drilled.Node)
	}
}

func TestTreemapTagsCountsRowsAndCoverage(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps, `V:\A\1.jpg`, `V:\A\2.jpg`, `V:\A\3.jpg`)
	for i, tag := range []string{"cat", "dog", "bird"} {
		if _, err := deps.DB.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, time_stamp) VALUES (?, ?, '', ?)`,
			`V:\A\1.jpg`, tag, float64(i)); err != nil {
			t.Fatalf("insert tag: %v", err)
		}
	}
	if _, err := deps.DB.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, time_stamp) VALUES (?, 'cat', '', 0)`,
		`V:\A\2.jpg`); err != nil {
		t.Fatalf("insert tag: %v", err)
	}
	api := newTreemapAPI(deps)

	resp := getTreemap(t, api, "?metric=tags")
	// 4 tag rows across 2 of the 3 items.
	if resp.Total != 4 || resp.TotalItems != 3 {
		t.Fatalf("expected total=4 totalItems=3, got %d/%d", resp.Total, resp.TotalItems)
	}
	a := childByName(childByName(resp.Node, "V:"), "A")
	if a == nil || a.Count != 4 || a.Covered != 2 || a.ItemCount != 3 {
		t.Fatalf("expected A count=4 covered=2 itemCount=3, got %+v", a)
	}
}

func TestTreemapEmbedMetricFiltersByModel(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps, `V:\A\1.jpg`, `V:\A\2.jpg`)
	seedVizEmbedding(t, deps, `V:\A\1.jpg`, "model-a", []float32{1, 0})
	seedVizEmbedding(t, deps, `V:\A\1.jpg`, "model-b", []float32{0, 1})
	seedVizEmbedding(t, deps, `V:\A\2.jpg`, "model-b", []float32{1, 1})
	api := newTreemapAPI(deps)

	resp := getTreemap(t, api, "?metric=embed:model-a")
	if resp.Total != 1 {
		t.Fatalf("expected 1 row for model-a, got %d", resp.Total)
	}
	a := childByName(childByName(resp.Node, "V:"), "A")
	if a == nil || a.Count != 1 || a.Covered != 1 || a.ItemCount != 2 {
		t.Fatalf("expected A count=1 covered=1 itemCount=2, got %+v", a)
	}
}

func TestTreemapLimitCollapsesIntoOther(t *testing.T) {
	deps := newTreemapTestDeps(t)
	// big has 3 items; small1/small2/small3 have 1 each.
	seedTreemapMedia(t, deps,
		`r/big/1.jpg`, `r/big/2.jpg`, `r/big/3.jpg`,
		`r/small1/1.jpg`, `r/small2/1.jpg`, `r/small3/1.jpg`)
	api := newTreemapAPI(deps)

	resp := getTreemap(t, api, "?metric=items&path=r&limit=2")
	n := resp.Node
	if len(n.Children) != 2 {
		t.Fatalf("expected 2 children under limit, got %d", len(n.Children))
	}
	if n.Children[0].Name != "big" || n.Children[0].Count != 3 {
		t.Fatalf("expected big first, got %+v", n.Children[0])
	}
	if n.OtherDirs != 2 || n.OtherCount != 2 {
		t.Fatalf("expected otherDirs=2 otherCount=2, got %d/%d", n.OtherDirs, n.OtherCount)
	}
}

func TestTreemapDepthPrunes(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps, `a/b/c/d/1.jpg`)
	api := newTreemapAPI(deps)

	resp := getTreemap(t, api, "?metric=items&depth=1")
	a := childByName(resp.Node, "a")
	if a == nil {
		t.Fatal("missing child a")
	}
	if len(a.Children) != 0 {
		t.Fatalf("depth=1 should not serialize grandchildren, got %+v", a.Children)
	}
}

func TestTreemapUnknownMetricRejected(t *testing.T) {
	api := newTreemapAPI(newTreemapTestDeps(t))
	req := httptest.NewRequest(http.MethodGet, "/api/treemap?metric=bogus", nil)
	rr := httptest.NewRecorder()
	api.dataHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for unknown metric, got %d", rr.Code)
	}
}

func TestTreemapUnknownPath404(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps, `a/1.jpg`)
	api := newTreemapAPI(deps)
	req := httptest.NewRequest(http.MethodGet, "/api/treemap?metric=items&path=nope", nil)
	rr := httptest.NewRecorder()
	api.dataHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for unknown path, got %d", rr.Code)
	}
}

func TestTreemapMetricsEndpoint(t *testing.T) {
	deps := newTreemapTestDeps(t)
	seedTreemapMedia(t, deps, `a/1.jpg`)
	seedVizEmbedding(t, deps, `a/1.jpg`, "model-a", []float32{1})
	if _, err := deps.DB.Exec(
		`INSERT INTO face (media_path, model, vector) VALUES ('a/1.jpg', 'sface', x'00')`); err != nil {
		t.Fatalf("insert face: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/treemap/metrics", nil)
	rr := httptest.NewRecorder()
	treemapMetricsHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var resp struct {
		Metrics []struct {
			ID    string `json:"id"`
			Label string `json:"label"`
			Count int64  `json:"count"`
		} `json:"metrics"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("bad json: %v", err)
	}
	got := map[string]int64{}
	for _, m := range resp.Metrics {
		got[m.ID] = m.Count
	}
	for id, want := range map[string]int64{
		"items": 1, "tags": 0, "embed:model-a": 1, "face:sface": 1,
	} {
		if got[id] != want {
			t.Errorf("metric %s: got %d want %d (all: %v)", id, got[id], want, got)
		}
	}
}

func TestTreemapPageServed(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/viz/treemap", nil)
	rr := httptest.NewRecorder()
	treemapPageHandler().ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/html") {
		t.Errorf("content-type = %q", ct)
	}
	body := rr.Body.String()
	for _, want := range []string{"/api/treemap", "/api/treemap/metrics", "squarify"} {
		if !strings.Contains(body, want) {
			t.Errorf("page missing %q", want)
		}
	}
}
