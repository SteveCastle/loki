package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	_ "modernc.org/sqlite"

	"github.com/stevecastle/shrike/storage"
)

// setupTestDB creates an in-memory SQLite database with the schema and seed data.
func setupTestDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}

	stmts := []string{
		`CREATE TABLE category (
			label TEXT PRIMARY KEY,
			weight REAL
		)`,
		`CREATE TABLE tag (
			label TEXT PRIMARY KEY,
			category_label TEXT,
			weight REAL,
			FOREIGN KEY (category_label) REFERENCES category (label)
		)`,
		`CREATE TABLE media (
			path TEXT PRIMARY KEY,
			description TEXT,
			transcript TEXT,
			hash TEXT,
			size INTEGER,
			width INTEGER,
			height INTEGER,
			thumbnail_path_600 TEXT,
			thumbnail_path_1200 TEXT,
			elo REAL
		)`,
		`CREATE TABLE media_tag_by_category (
			media_path TEXT,
			tag_label TEXT,
			category_label TEXT,
			weight REAL,
			time_stamp REAL,
			created_at INTEGER,
			PRIMARY KEY(media_path, tag_label, category_label, time_stamp)
		)`,
		// Seed categories
		`INSERT INTO category (label, weight) VALUES ('Subject', 0)`,
		`INSERT INTO category (label, weight) VALUES ('Style', 1)`,
		// Seed tags
		`INSERT INTO tag (label, category_label, weight) VALUES ('sunset', 'Subject', 0)`,
		`INSERT INTO tag (label, category_label, weight) VALUES ('portrait', 'Subject', 1)`,
		`INSERT INTO tag (label, category_label, weight) VALUES ('moody', 'Style', 0)`,
		// Seed media
		`INSERT INTO media (path, description, width, height) VALUES ('/photos/a.jpg', 'A beautiful sunset photo', 1920, 1080)`,
		`INSERT INTO media (path, description, width, height) VALUES ('/photos/b.jpg', 'Portrait in the park', 1080, 1920)`,
		`INSERT INTO media (path, description, width, height) VALUES ('/photos/c.jpg', 'Mountain landscape', 3840, 2160)`,
		// Seed assignments
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES ('/photos/a.jpg', 'sunset', 'Subject', 0, 0)`,
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES ('/photos/a.jpg', 'moody', 'Style', 1, 0)`,
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES ('/photos/b.jpg', 'portrait', 'Subject', 0, 0)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatalf("exec %q: %v", s[:40], err)
		}
	}
	return db
}

func testDeps(t *testing.T) *Dependencies {
	t.Helper()
	return &Dependencies{
		DB:      setupTestDB(t),
		Storage: storage.NewRegistry(nil),
	}
}

func postJSON(handler http.HandlerFunc, body any) *httptest.ResponseRecorder {
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	return rr
}

func getJSON(handler http.HandlerFunc, path string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)
	return rr
}

// ---- Tests ----

func TestLokiTaxonomyHandler(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiTaxonomyHandler(deps), nil)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp map[string]struct {
		Label  string `json:"label"`
		Weight float64 `json:"weight"`
		Tags   []map[string]any `json:"tags"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(resp) != 2 {
		t.Errorf("expected 2 categories, got %d", len(resp))
	}
	totalTags := 0
	for _, cat := range resp {
		totalTags += len(cat.Tags)
	}
	if totalTags != 3 {
		t.Errorf("expected 3 tags, got %d", totalTags)
	}
}

func TestLokiMediaSearchHandler(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiMediaSearchHandler(deps), searchRequest{Description: "sunset"})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var items []map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &items); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 result, got %d", len(items))
	}
	if items[0]["path"] != "/photos/a.jpg" {
		t.Errorf("expected /photos/a.jpg, got %v", items[0]["path"])
	}
}

func TestLokiMediaHandler_NoTags(t *testing.T) {
	deps := testDeps(t)
	rr := postJSON(lokiMediaHandler(deps), mediaRequest{})

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var items []map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &items); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(items) != 3 {
		t.Errorf("expected 3 media items, got %d", len(items))
	}
}

func TestLokiMediaHandler_WithTags(t *testing.T) {
	deps := testDeps(t)

	// Exclusive mode: only a.jpg has both sunset and moody
	rr := postJSON(lokiMediaHandler(deps), mediaRequest{
		Tags: []string{"sunset", "moody"},
		Mode: "EXCLUSIVE",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	var items []map[string]any
	json.Unmarshal(rr.Body.Bytes(), &items)
	if len(items) != 1 {
		t.Fatalf("expected 1 result for exclusive, got %d", len(items))
	}
	if items[0]["path"] != "/photos/a.jpg" {
		t.Errorf("expected /photos/a.jpg, got %v", items[0]["path"])
	}

	// Inclusive mode: a.jpg has sunset, b.jpg has portrait — search for sunset OR portrait
	rr2 := postJSON(lokiMediaHandler(deps), mediaRequest{
		Tags: []string{"sunset", "portrait"},
		Mode: "INCLUSIVE",
	})
	if rr2.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr2.Code)
	}
	var items2 []map[string]any
	json.Unmarshal(rr2.Body.Bytes(), &items2)
	if len(items2) != 2 {
		t.Errorf("expected 2 results for inclusive, got %d", len(items2))
	}
}

func TestLokiCreateAndDeleteTag(t *testing.T) {
	deps := testDeps(t)

	// Create a new tag
	rr := postJSON(lokiCreateTagHandler(deps), tagRequest{
		Label:         "newTag",
		CategoryLabel: "Subject",
		Weight:        5,
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("create tag: expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	// Verify it exists
	var count int
	deps.DB.QueryRow("SELECT COUNT(*) FROM tag WHERE label = 'newTag'").Scan(&count)
	if count != 1 {
		t.Fatalf("expected tag to exist, count=%d", count)
	}

	// Delete the tag
	rr2 := postJSON(lokiDeleteTagHandler(deps), labelRequest{Label: "newTag"})
	if rr2.Code != http.StatusOK {
		t.Fatalf("delete tag: expected 200, got %d: %s", rr2.Code, rr2.Body.String())
	}

	deps.DB.QueryRow("SELECT COUNT(*) FROM tag WHERE label = 'newTag'").Scan(&count)
	if count != 0 {
		t.Errorf("expected tag to be deleted, count=%d", count)
	}
}

func TestLokiSettingsPutAndGet(t *testing.T) {
	// Reset global settings for test isolation
	lokiSettings = make(map[string]any)
	deps := testDeps(t)

	// Put a setting
	rr := postJSON(lokiSettingsPutHandler(deps), map[string]any{
		"key":   "theme",
		"value": "dark",
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("put setting: expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	// Get settings
	rr2 := getJSON(lokiSettingsGetHandler(deps), "/api/settings")
	if rr2.Code != http.StatusOK {
		t.Fatalf("get settings: expected 200, got %d: %s", rr2.Code, rr2.Body.String())
	}

	var settings map[string]any
	if err := json.Unmarshal(rr2.Body.Bytes(), &settings); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if settings["theme"] != "dark" {
		t.Errorf("expected theme=dark, got %v", settings["theme"])
	}
}
