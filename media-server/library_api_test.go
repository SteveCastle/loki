package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func newLibraryTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE media (
			path TEXT PRIMARY KEY, description TEXT, transcript TEXT,
			elo REAL, views INTEGER, wins INTEGER, losses INTEGER,
			battles INTEGER)`,
		`CREATE TABLE tag (label TEXT PRIMARY KEY, category_label TEXT, weight REAL)`,
		`CREATE TABLE media_tag_by_category (
			media_path TEXT, tag_label TEXT, category_label TEXT,
			weight REAL, time_stamp REAL, created_at INTEGER,
			PRIMARY KEY(media_path, tag_label, category_label, time_stamp))`,
		`CREATE TABLE battle (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			winner_path TEXT NOT NULL, loser_path TEXT NOT NULL,
			outcome REAL NOT NULL DEFAULT 1,
			winner_elo_before REAL, loser_elo_before REAL,
			winner_elo_after REAL, loser_elo_after REAL,
			created_at INTEGER)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("create table: %v", err)
		}
	}
	if _, err := db.Exec(`INSERT INTO media (path) VALUES ('a.jpg'), ('b.jpg')`); err != nil {
		t.Fatalf("seed media: %v", err)
	}
	return &Dependencies{DB: db}
}

func postLibraryJSON(t *testing.T, h http.HandlerFunc, target, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, target, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	return rr
}

func TestMediaTranscript(t *testing.T) {
	deps := newLibraryTestDeps(t)
	h := mediaTranscriptHandler(deps)

	rr := postLibraryJSON(t, h, "/api/media/transcript", `{"path":"a.jpg","transcript":"hello world"}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("set: status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var got string
	deps.DB.QueryRow(`SELECT transcript FROM media WHERE path = 'a.jpg'`).Scan(&got)
	if got != "hello world" {
		t.Errorf("transcript = %q", got)
	}

	// Clearing = setting empty.
	postLibraryJSON(t, h, "/api/media/transcript", `{"path":"a.jpg","transcript":""}`)
	deps.DB.QueryRow(`SELECT transcript FROM media WHERE path = 'a.jpg'`).Scan(&got)
	if got != "" {
		t.Errorf("cleared transcript = %q", got)
	}

	if rr = postLibraryJSON(t, h, "/api/media/transcript", `{"path":"nope.jpg","transcript":"x"}`); rr.Code != http.StatusNotFound {
		t.Errorf("unknown media: status = %d, want 404", rr.Code)
	}
	if rr = postLibraryJSON(t, h, "/api/media/transcript", `{"transcript":"x"}`); rr.Code != http.StatusBadRequest {
		t.Errorf("no path: status = %d, want 400", rr.Code)
	}
}

func TestMediaRating(t *testing.T) {
	deps := newLibraryTestDeps(t)
	h := mediaRatingHandler(deps)

	// Read-only request returns null fields.
	rr := postLibraryJSON(t, h, "/api/media/rating", `{"path":"a.jpg"}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("read: status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var read struct {
		Elo   *float64 `json:"elo"`
		Views *int64   `json:"views"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &read); err != nil || read.Elo != nil || read.Views != nil {
		t.Fatalf("read = %+v, err = %v, want nulls", read, err)
	}

	// Partial update touches only the provided fields.
	rr = postLibraryJSON(t, h, "/api/media/rating", `{"path":"a.jpg","elo":1600,"wins":3}`)
	if rr.Code != http.StatusOK {
		t.Fatalf("update: status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var upd struct {
		Elo    *float64 `json:"elo"`
		Views  *int64   `json:"views"`
		Wins   *int64   `json:"wins"`
		Losses *int64   `json:"losses"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &upd); err != nil {
		t.Fatalf("bad JSON: %v", err)
	}
	if upd.Elo == nil || *upd.Elo != 1600 || upd.Wins == nil || *upd.Wins != 3 {
		t.Errorf("updated = %+v", upd)
	}
	if upd.Views != nil || upd.Losses != nil {
		t.Errorf("untouched fields should stay null: %+v", upd)
	}

	if rr = postLibraryJSON(t, h, "/api/media/rating", `{"path":"nope.jpg","elo":1}`); rr.Code != http.StatusNotFound {
		t.Errorf("unknown media: status = %d, want 404", rr.Code)
	}
}

func TestTagsList(t *testing.T) {
	deps := newLibraryTestDeps(t)
	seed := []string{
		`INSERT INTO tag (label, category_label, weight) VALUES
			('sunset', 'scenes', 2), ('dog', 'animals', 1), ('unused', 'scenes', 0)`,
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES
			('a.jpg', 'sunset', 'scenes', 0, 0),
			('b.jpg', 'sunset', 'scenes', 0, 0),
			('a.jpg', 'dog', 'animals', 0, 0)`,
	}
	for _, s := range seed {
		if _, err := deps.DB.Exec(s); err != nil {
			t.Fatalf("seed: %v", err)
		}
	}

	req := httptest.NewRequest(http.MethodGet, "/api/tags/list", nil)
	rr := httptest.NewRecorder()
	tagsListHandler(deps).ServeHTTP(rr, req)
	var resp struct {
		Tags []struct {
			Label      string  `json:"label"`
			Category   string  `json:"category"`
			Weight     float64 `json:"weight"`
			MediaCount int     `json:"media_count"`
		} `json:"tags"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || len(resp.Tags) != 3 {
		t.Fatalf("tags = %+v, err = %v; body = %s", resp.Tags, err, rr.Body.String())
	}
	counts := map[string]int{}
	for _, tg := range resp.Tags {
		counts[tg.Label] = tg.MediaCount
	}
	if counts["sunset"] != 2 || counts["dog"] != 1 || counts["unused"] != 0 {
		t.Errorf("counts = %v", counts)
	}

	// Category filter
	req = httptest.NewRequest(http.MethodGet, "/api/tags/list?category=scenes", nil)
	rr = httptest.NewRecorder()
	tagsListHandler(deps).ServeHTTP(rr, req)
	resp.Tags = nil
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil || len(resp.Tags) != 2 {
		t.Fatalf("filtered tags = %+v, err = %v", resp.Tags, err)
	}
}

func TestDeleteAssignmentBulk(t *testing.T) {
	deps := newLibraryTestDeps(t)
	if _, err := deps.DB.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES
		('a.jpg', 'dog', 'animals', 0, 0),
		('b.jpg', 'dog', 'animals', 0, 0),
		('a.jpg', 'sunset', 'scenes', 0, 0)`); err != nil {
		t.Fatalf("seed: %v", err)
	}

	req := httptest.NewRequest(http.MethodDelete, "/api/assignments",
		strings.NewReader(`{"mediaPaths":["a.jpg","b.jpg"],"tag":{"tag_label":"dog"}}`))
	rr := httptest.NewRecorder()
	lokiDeleteAssignmentHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var n int
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = 'dog'`).Scan(&n)
	if n != 0 {
		t.Errorf("dog assignments left = %d, want 0", n)
	}
	deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category`).Scan(&n)
	if n != 1 {
		t.Errorf("total assignments = %d, want 1 (sunset untouched)", n)
	}
}
