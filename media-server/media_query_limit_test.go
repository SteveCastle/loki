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
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// TestMediaQueryVisualLimitWidens pins the per-chip candidate cap: a visual
// predicate resolves to its 1000 most similar items by default, `limit` widens
// that (the chip's "+1000" button), and a widened set larger than the bound
// IN-list size still runs (json_each) and keeps the score order.
func TestMediaQueryVisualLimitWidens(t *testing.T) {
	db := newFacesTestDB(t)
	deps := &Dependencies{DB: db}
	model := tasks.ActiveEmbedModel().ID

	const n = 2500
	tx, err := db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	media, err := tx.Prepare(`INSERT INTO media (path) VALUES (?)`)
	if err != nil {
		t.Fatal(err)
	}
	emb, err := tx.Prepare(`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?,?,?,?)`)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < n; i++ {
		path := fmt.Sprintf("item-%04d.jpg", i)
		// Monotonically less similar to the anchor (item-0000) as i grows.
		angle := 0.0005 * float64(i)
		vec := embedvec.Normalize([]float32{float32(math.Cos(angle)), float32(math.Sin(angle))})
		if _, err := media.Exec(path); err != nil {
			t.Fatal(err)
		}
		if _, err := emb.Exec(path, model, 2, embedvec.Encode(vec)); err != nil {
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

	if got := runQuery(`{"predicates":[{"type":"similar","value":"item-0000.jpg"}],"mode":"AND"}`); len(got) != 1000 {
		t.Fatalf("default cap: got %d results, want 1000", len(got))
	}
	widened := runQuery(`{"predicates":[{"type":"similar","value":"item-0000.jpg","limit":2000}],"mode":"AND"}`)
	if len(widened) != 2000 {
		t.Fatalf("limit 2000: got %d results", len(widened))
	}
	for i, p := range widened {
		if want := fmt.Sprintf("item-%04d.jpg", i); p != want {
			t.Fatalf("widened result %d = %s, want %s (score order)", i, p, want)
		}
	}
	if got := runQuery(`{"predicates":[{"type":"similar","value":"item-0000.jpg","limit":3000}],"mode":"AND"}`); len(got) != n {
		t.Fatalf("limit past the library: got %d results, want all %d", len(got), n)
	}
	// An excluded widened chip removes its whole set.
	if got := runQuery(`{"predicates":[{"type":"path","value":"item-"},{"type":"similar","value":"item-0000.jpg","limit":2000,"exclude":true,"join":"AND"}],"mode":"AND"}`); len(got) != n-2000 {
		t.Fatalf("excluded widened chip: got %d results, want %d", len(got), n-2000)
	}
}

func TestPredicateLimitClamps(t *testing.T) {
	if got := predicateLimit(Predicate{}, 1000); got != 1000 {
		t.Errorf("unset = %d", got)
	}
	if got := predicateLimit(Predicate{Limit: 5000}, 1000); got != 5000 {
		t.Errorf("5000 = %d", got)
	}
	if got := predicateLimit(Predicate{Limit: 1 << 30}, 1000); got != maxVisualCandidateLimit {
		t.Errorf("huge = %d", got)
	}
}
