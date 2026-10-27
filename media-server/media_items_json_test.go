package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// NULL columns are omitted, real zeros are kept — the shape the renderer's
// Item type has always received.
func TestMediaItemsJSONShape(t *testing.T) {
	deps := testDeps(t)
	if _, err := deps.DB.Exec(`UPDATE media SET elo = 0, battles = 0 WHERE path = '/photos/a.jpg'`); err != nil {
		t.Fatal(err)
	}
	rr := postJSON(lokiMediaHandler(deps), mediaRequest{})
	if rr.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rr.Code, rr.Body.String())
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q", ct)
	}
	var items []map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &items); err != nil {
		t.Fatalf("decode: %v\n%s", err, rr.Body.String())
	}
	byPath := map[string]map[string]any{}
	for _, it := range items {
		byPath[it["path"].(string)] = it
	}
	a, b := byPath["/photos/a.jpg"], byPath["/photos/b.jpg"]
	if a["elo"] != 0.0 || a["battles"] != 0.0 {
		t.Errorf("zero elo/battles must be present: %v", a)
	}
	if _, ok := b["elo"]; ok {
		t.Errorf("NULL elo must be omitted: %v", b)
	}
	if a["width"] != 1920.0 || a["height"] != 1080.0 || a["mtimeMs"] != 0.0 {
		t.Errorf("unexpected fields: %v", a)
	}
}

// A result far larger than one output chunk streams as one valid array.
func TestMediaQueryStreamsLargeResults(t *testing.T) {
	deps := testDeps(t)
	tx, _ := deps.DB.Begin()
	for i := 0; i < 20000; i++ {
		if _, err := tx.Exec(`INSERT INTO media (path, width, height) VALUES (?, 640, 480)`,
			fmt.Sprintf("/bulk/%06d-a-reasonably-long-file-name.jpg", i)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	for _, h := range []http.HandlerFunc{lokiMediaHandler(deps)} {
		rr := postJSON(h, mediaRequest{})
		var items []mediaItemJSON
		if err := json.Unmarshal(rr.Body.Bytes(), &items); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(items) != 20003 {
			t.Fatalf("got %d items, want 20003", len(items))
		}
		if rr.Body.Len() < 2*mediaArrayChunk {
			t.Fatalf("body %d bytes; test should span several chunks", rr.Body.Len())
		}
	}

	req := httptest.NewRequest(http.MethodPost, "/api/media/query", strings.NewReader(`{"predicates":[],"mode":"AND"}`))
	rec := httptest.NewRecorder()
	lokiMediaQueryHandler(deps)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("query status %d: %s", rec.Code, rec.Body.String())
	}
	var items []mediaItemJSON
	if err := json.Unmarshal(rec.Body.Bytes(), &items); err != nil {
		t.Fatalf("decode query: %v", err)
	}
	if len(items) != 20003 {
		t.Fatalf("query got %d items", len(items))
	}
}

func TestMediaHandlerEmptyResultIsArray(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiMediaHandler(deps), mediaRequest{Tags: []string{"no-such-tag"}, Mode: "EXCLUSIVE"})
	if got := strings.TrimSpace(rr.Body.String()); got != "[]" {
		t.Fatalf("empty result = %q, want []", got)
	}
}
