package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// TestMediaQueryVisualFiltersFirst pins the filter-first contract for all-AND
// composed queries: `similar:x AND tag:y` must return the most similar items
// WITHIN tag:y. Historically the handler took the global top-1000 by
// similarity and intersected afterward, so a tag:y match ranked 1001st or
// beyond silently vanished. The fixture builds exactly that shape: 1000
// untagged decoys sit between the anchor and the tagged item.
func TestMediaQueryVisualFiltersFirst(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	model := tasks.ActiveEmbedModel().ID

	insertMedia := func(path string, tagged bool) {
		t.Helper()
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, path); err != nil {
			t.Fatal(err)
		}
		if tagged {
			if _, err := db.Exec(
				`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
				 VALUES (?, 'keep', 'Subject', 1, 0)`, path,
			); err != nil {
				t.Fatal(err)
			}
		}
	}

	insertMedia("anchor.jpg", true)
	if err := media.UpsertEmbedding(db, "anchor.jpg", model,
		embedvec.Normalize([]float32{1, 0}), 0); err != nil {
		t.Fatal(err)
	}
	// The tagged item the old behavior lost: similarity 0 to the anchor, so it
	// ranks behind every decoy.
	insertMedia("wanted.jpg", true)
	if err := media.UpsertEmbedding(db, "wanted.jpg", model,
		embedvec.Normalize([]float32{0, 1}), 0); err != nil {
		t.Fatal(err)
	}
	// 1000 untagged decoys, every one more similar to the anchor than wanted.jpg
	// — enough to fill the visual candidate cap on their own. Embeddings only;
	// they don't need media rows for either assertion below.
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	stmt, err := tx.Prepare(
		`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?,?,?,?)`)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ {
		angle := 0.1 + 0.001*float64(i) // decoy cosines ~0.995..0.45, all > 0
		vec := embedvec.Normalize([]float32{float32(math.Cos(angle)), float32(math.Sin(angle))})
		if _, err := stmt.Exec(fmt.Sprintf("decoy-%04d.jpg", i), model, 2, embedvec.Encode(vec)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	runQuery := func(body string) []string {
		t.Helper()
		req := httptest.NewRequest(http.MethodPost, "/api/media/query", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		lokiMediaQueryHandler(deps)(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
		}
		var items []map[string]any
		if err := json.Unmarshal(rec.Body.Bytes(), &items); err != nil {
			t.Fatal(err)
		}
		paths := make([]string, 0, len(items))
		for _, it := range items {
			p, _ := it["path"].(string)
			paths = append(paths, p)
		}
		return paths
	}

	// Sanity: an unfiltered similar: query is still capped at the global
	// top-1000, which the decoys fill — wanted.jpg is out of reach.
	alone := runQuery(`{"predicates":[{"type":"similar","value":"anchor.jpg"}],"mode":"AND"}`)
	for _, p := range alone {
		if p == "wanted.jpg" {
			t.Fatalf("fixture broken: wanted.jpg inside the global top-1000: %v", alone)
		}
	}

	// The fix: ANDing tag:keep restricts the scan to the tagged items, so
	// wanted.jpg is now the anchor's nearest (only) neighbour within the set.
	both := runQuery(`{"predicates":[{"type":"similar","value":"anchor.jpg"},{"type":"tag","value":"keep","join":"AND"}],"mode":"AND"}`)
	if len(both) != 2 || both[0] != "anchor.jpg" || both[1] != "wanted.jpg" {
		t.Fatalf("expected [anchor.jpg wanted.jpg] (score-ordered), got %v", both)
	}

	// OR keeps resolve-then-compose: the visual side is still the global
	// top-1000 (all decoy embeddings, no media rows), unioned with tag:keep.
	// wanted.jpg arrives via the tag branch, not the scan.
	union := runQuery(`{"predicates":[{"type":"similar","value":"anchor.jpg"},{"type":"tag","value":"keep","join":"OR"}],"mode":"OR"}`)
	found := map[string]bool{}
	for _, p := range union {
		found[p] = true
	}
	if !found["anchor.jpg"] || !found["wanted.jpg"] {
		t.Fatalf("OR union missing expected items: %v", union)
	}
}

// TestMediaQueryVisualEmptyFilterSkipsScan verifies that when the SQL side of
// an all-AND query matches nothing, the visual predicate resolves to nothing
// (clause 1=0) without consulting the embedding backend.
func TestMediaQueryVisualEmptyFilterSkipsScan(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	model := tasks.ActiveEmbedModel().ID

	if _, err := db.Exec(`INSERT INTO media (path) VALUES ('anchor.jpg')`); err != nil {
		t.Fatal(err)
	}
	if err := media.UpsertEmbedding(db, "anchor.jpg", model,
		embedvec.Normalize([]float32{1, 0}), 0); err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/media/query", strings.NewReader(
		`{"predicates":[{"type":"similar","value":"anchor.jpg"},{"type":"tag","value":"no-such-tag","join":"AND"}],"mode":"AND"}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	lokiMediaQueryHandler(deps)(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", rec.Code, rec.Body.String())
	}
	var items []map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 0 {
		t.Fatalf("expected no items, got %v", items)
	}
}

// TestAllAndPredicates covers the connector classification the prefilter
// gates on: explicit Joins override mode, empty-value predicates don't
// participate, and the first valid predicate's own Join is ignored.
func TestAllAndPredicates(t *testing.T) {
	cases := []struct {
		name  string
		preds []Predicate
		mode  string
		want  bool
	}{
		{"all AND via mode", []Predicate{{Type: "similar", Value: "x"}, {Type: "tag", Value: "y"}}, "AND", true},
		{"all OR via mode", []Predicate{{Type: "similar", Value: "x"}, {Type: "tag", Value: "y"}}, "OR", false},
		{"explicit OR join overrides AND mode", []Predicate{{Type: "similar", Value: "x"}, {Type: "tag", Value: "y", Join: "OR"}}, "AND", false},
		{"explicit AND joins override OR mode", []Predicate{{Type: "similar", Value: "x"}, {Type: "tag", Value: "y", Join: "AND"}}, "OR", true},
		{"first predicate's own join ignored", []Predicate{{Type: "similar", Value: "x", Join: "OR"}, {Type: "tag", Value: "y", Join: "AND"}}, "AND", true},
		{"empty-value predicate skipped", []Predicate{{Type: "similar", Value: "x"}, {Type: "tag", Value: "", Join: "OR"}, {Type: "tag", Value: "y", Join: "AND"}}, "AND", true},
		{"single predicate", []Predicate{{Type: "similar", Value: "x"}}, "OR", true},
	}
	for _, tc := range cases {
		if got := allAndPredicates(tc.preds, tc.mode); got != tc.want {
			t.Errorf("%s: allAndPredicates = %v, want %v", tc.name, got, tc.want)
		}
	}
}
