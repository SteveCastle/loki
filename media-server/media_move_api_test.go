package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	_ "modernc.org/sqlite"
)

func postMove(t *testing.T, deps *Dependencies, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/api/media/move", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	lokiMediaMoveHandler(deps)(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec, out
}

// seedMoveItem gives a path a row in every table the move has to carry.
func seedMoveItem(t *testing.T, deps *Dependencies, path string) {
	t.Helper()
	for _, s := range []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO media (path) VALUES (?)`, []any{path}},
		{`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		  VALUES (?, 'sunset', 'Subject', 1, 0)`, []any{path}},
		{`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0000')`, []any{path}},
		{`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
		  VALUES (?, 'sface', 0.1, 0.1, 0.2, 0.2, 0.9, x'0000')`, []any{path}},
		{`INSERT INTO face_scan (media_path, model, face_count) VALUES (?, 'sface', 1)`, []any{path}},
		{`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, '/other.jpg', 1)`, []any{path}},
	} {
		if _, err := deps.DB.Exec(s.sql, s.args...); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}
}

func TestMediaMoveRepointsEveryReference(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	seedMoveItem(t, deps, "/photos/old.jpg")

	rec, out := postMove(t, deps, `{"from":"/photos/old.jpg","to":"/archive/new.jpg"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("move: %d %s", rec.Code, rec.Body.String())
	}
	if items, _ := out["items"].(float64); items != 1 {
		t.Errorf("items = %v, want 1 (%v)", out["items"], out)
	}
	rows, _ := out["rows"].(map[string]any)
	for _, key := range []string{
		"media.path", "media_tag_by_category.media_path", "media_embedding.media_path",
		"face.media_path", "face_scan.media_path", "battle.winner_path",
	} {
		if n, _ := rows[key].(float64); n != 1 {
			t.Errorf("rows[%q] = %v, want 1", key, rows[key])
		}
	}

	count := func(query string, args ...any) int {
		t.Helper()
		var n int
		if err := deps.DB.QueryRow(query, args...).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	if n := count(`SELECT COUNT(*) FROM media WHERE "path" = ?`, "/archive/new.jpg"); n != 1 {
		t.Errorf("media row not at the new path: %d", n)
	}
	if n := count(`SELECT COUNT(*) FROM face WHERE media_path = ?`, "/photos/old.jpg"); n != 0 {
		t.Errorf("faces still at the old path: %d", n)
	}
}

func TestMediaMovePrefixAndDryRun(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	seedMoveItem(t, deps, "/photos/2023/a.jpg")
	seedMoveItem(t, deps, "/photos/2023/b.jpg")
	seedMoveItem(t, deps, "/photos/2023extra/c.jpg")

	rec, out := postMove(t, deps,
		`{"from":"/photos/2023","to":"/archive/2023","prefix":true,"dryRun":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("dry run: %d %s", rec.Code, rec.Body.String())
	}
	// Segment-aligned: the lookalike folder is not included.
	if items, _ := out["items"].(float64); items != 2 {
		t.Errorf("items = %v, want 2 (%v)", out["items"], out)
	}
	if dry, _ := out["dryRun"].(bool); !dry {
		t.Error("response does not report the dry run")
	}
	var still int
	if err := deps.DB.QueryRow(
		`SELECT COUNT(*) FROM media WHERE "path" LIKE '/photos/2023/%'`).Scan(&still); err != nil {
		t.Fatal(err)
	}
	if still != 2 {
		t.Errorf("dry run moved rows: %d left at the source", still)
	}

	// The real run, same arguments minus dryRun.
	rec, out = postMove(t, deps, `{"from":"/photos/2023","to":"/archive/2023","prefix":true}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("move: %d %s", rec.Code, rec.Body.String())
	}
	if items, _ := out["items"].(float64); items != 2 {
		t.Errorf("items = %v, want 2", out["items"])
	}
	var moved, untouched int
	if err := deps.DB.QueryRow(
		`SELECT COUNT(*) FROM media WHERE "path" LIKE '/archive/2023/%'`).Scan(&moved); err != nil {
		t.Fatal(err)
	}
	if err := deps.DB.QueryRow(
		`SELECT COUNT(*) FROM media WHERE "path" = '/photos/2023extra/c.jpg'`).Scan(&untouched); err != nil {
		t.Fatal(err)
	}
	if moved != 2 || untouched != 1 {
		t.Errorf("moved = %d (want 2), lookalike folder rows = %d (want 1)", moved, untouched)
	}
}

func TestMediaMoveConflictIs409WithPaths(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	seedMoveItem(t, deps, "/photos/a.jpg")
	seedMoveItem(t, deps, "/photos/b.jpg")

	rec, out := postMove(t, deps, `{"from":"/photos/a.jpg","to":"/photos/b.jpg"}`)
	if rec.Code != http.StatusConflict {
		t.Fatalf("conflict: %d %s", rec.Code, rec.Body.String())
	}
	conflicts, _ := out["conflicts"].([]any)
	if len(conflicts) != 1 || conflicts[0] != "/photos/b.jpg" {
		t.Errorf("conflicts = %v (body %v)", conflicts, out)
	}
	// Nothing moved.
	var n int
	if err := deps.DB.QueryRow(
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, "/photos/a.jpg").Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("a refused move rewrote rows: %d", n)
	}
}

func TestMediaMoveRejectsBadArguments(t *testing.T) {
	deps := &Dependencies{DB: newFacesTestDB(t)}
	for _, body := range []string{
		`{"to":"/b.jpg"}`,
		`{"from":"/a.jpg"}`,
		`{"from":"/a.jpg","to":"/a.jpg"}`,
		`not json`,
	} {
		rec, _ := postMove(t, deps, body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %q → %d, want 400", body, rec.Code)
		}
	}
}
