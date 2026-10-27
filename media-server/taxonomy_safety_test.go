package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
)

// fkDeps opens a file library with foreign keys ON, the way the server's DSN
// does (db_dsn.go). The in-memory test DB leaves them off, which is how the
// category-rename failure went unnoticed.
func fkDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "lib.sqlite")+
		"?_pragma=busy_timeout=5000&_pragma=foreign_keys=ON&_pragma=journal_mode=WAL")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	for _, s := range []string{
		`INSERT INTO category (label, weight) VALUES ('Anime', 0), ('TV Shows', 1), ('Movies', 2)`,
		`INSERT INTO tag (label, category_label, weight) VALUES ('Cowboy Bebop', 'Anime', 0), ('Akira', 'Anime', 1)`,
		`INSERT INTO media (path) VALUES ('/a.jpg'), ('/b.jpg'), ('/c.jpg')`,
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES
			('/a.jpg', 'Cowboy Bebop', 'Anime', 0, 0), ('/b.jpg', 'Cowboy Bebop', 'Anime', 1, 0), ('/c.jpg', 'Akira', 'Anime', 0, 0)`,
	} {
		if _, err := db.Exec(s); err != nil {
			t.Fatalf("seed %q: %v", s, err)
		}
	}
	return &Dependencies{DB: db, Storage: storage.NewRegistry(nil)}
}

func countWhere(t *testing.T, db *sql.DB, q string, args ...any) int {
	t.Helper()
	var n int
	if err := db.QueryRow(q, args...).Scan(&n); err != nil {
		t.Fatalf("%s: %v", q, err)
	}
	return n
}


// The incident: tags moved out of a category, then the emptied category was
// deleted — which took every moved tag's assignments with it.
func TestTaxonomySafety_MoveThenDeleteCategoryKeepsAssignments(t *testing.T) {
	deps := fkDeps(t)

	mustOK(t, "move bebop", postJSON(lokiMoveTagHandler(deps), moveTagRequest{Label: "Cowboy Bebop", CategoryLabel: "TV Shows"}).Code, "")
	mustOK(t, "move akira", postJSON(lokiMoveTagHandler(deps), moveTagRequest{Label: "Akira", CategoryLabel: "Movies"}).Code, "")
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category WHERE category_label = 'Anime'`); n != 0 {
		t.Fatalf("%d assignments still filed under Anime after the move", n)
	}

	rr := getJSON(lokiCategoryImpactHandler(deps), "/api/categories/impact?label=Anime")
	mustOK(t, "impact", rr.Code, rr.Body.String())
	var im categoryImpact
	json.Unmarshal(rr.Body.Bytes(), &im)
	if !im.Exists || im.Tags != 0 || im.Assignments != 0 {
		t.Fatalf("impact of emptied category = %+v, want exists with nothing to remove", im)
	}

	rr = postJSON(lokiDeleteCategoryHandler(deps), labelRequest{Label: "Anime"})
	mustOK(t, "delete category", rr.Code, rr.Body.String())
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category`); n != 3 {
		t.Fatalf("assignments after deleting the emptied category = %d, want 3", n)
	}
}

// Even rows whose stored category_label has drifted (written before the move
// fix) survive deleting that category while their tag lives elsewhere.
func TestTaxonomySafety_DeleteCategoryIgnoresDriftedRows(t *testing.T) {
	deps := fkDeps(t)
	deps.DB.Exec(`UPDATE tag SET category_label = 'TV Shows' WHERE label = 'Cowboy Bebop'`) // pre-fix move

	rr := postJSON(lokiDeleteCategoryHandler(deps), labelRequest{Label: "Anime"})
	mustOK(t, "delete category", rr.Code, rr.Body.String())
	var im categoryImpact
	json.Unmarshal(rr.Body.Bytes(), &im)
	if im.Tags != 1 || im.Assignments != 1 {
		t.Fatalf("removed %+v, want only Akira's tag and its one assignment", im)
	}
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = 'Cowboy Bebop'`); n != 2 {
		t.Fatalf("Cowboy Bebop assignments = %d, want 2", n)
	}
}

func TestTaxonomySafety_RenameCategoryWithForeignKeys(t *testing.T) {
	deps := fkDeps(t)
	rr := postJSON(lokiRenameCategoryHandler(deps), renameRequest{Label: "Anime", NewLabel: "Japanese Animation"})
	mustOK(t, "rename", rr.Code, rr.Body.String())
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM tag WHERE category_label = 'Japanese Animation'`); n != 2 {
		t.Fatalf("tags under renamed category = %d, want 2", n)
	}
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category WHERE category_label = 'Japanese Animation'`); n != 3 {
		t.Fatalf("assignments under renamed category = %d, want 3", n)
	}
}

func TestTaxonomySafety_AssignmentUsesTagCategory(t *testing.T) {
	deps := fkDeps(t)

	// Omitted category: filed under the tag's own.
	rr := postJSON(lokiCreateAssignmentHandler(deps), assignmentRequest{MediaPath: "/c.jpg", TagLabel: "Cowboy Bebop"})
	mustOK(t, "assign", rr.Code, rr.Body.String())
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = '/c.jpg' AND tag_label = 'Cowboy Bebop' AND category_label = 'Anime'`); n != 1 {
		t.Fatal("assignment not filed under the tag's category")
	}

	// A different category is a conflict, not a second filing.
	rr = postJSON(lokiCreateAssignmentHandler(deps), assignmentRequest{MediaPath: "/a.jpg", TagLabel: "Akira", CategoryLabel: "Movies"})
	if rr.Code != http.StatusConflict {
		t.Fatalf("mismatched category: expected 409, got %d %s", rr.Code, rr.Body.String())
	}

	// A new tag is created in the requested category...
	rr = postJSON(lokiCreateAssignmentHandler(deps), assignmentRequest{MediaPaths: []string{"/a.jpg", "/b.jpg"}, TagLabel: "Twin Peaks", CategoryLabel: "TV Shows"})
	mustOK(t, "assign new tag", rr.Code, rr.Body.String())
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM tag WHERE label = 'Twin Peaks' AND category_label = 'TV Shows'`); n != 1 {
		t.Fatal("assigning a new tag didn't create it")
	}
	// ...which must exist, and must be named.
	rr = postJSON(lokiCreateAssignmentHandler(deps), assignmentRequest{MediaPath: "/a.jpg", TagLabel: "Seinfeld", CategoryLabel: "Nope"})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("unknown category: expected 400, got %d", rr.Code)
	}
	rr = postJSON(lokiCreateAssignmentHandler(deps), assignmentRequest{MediaPath: "/a.jpg", TagLabel: "Seinfeld"})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("new tag without category: expected 400, got %d", rr.Code)
	}
}

func TestTaxonomySafety_DeleteTagChecksCategory(t *testing.T) {
	deps := fkDeps(t)
	rr := postJSON(lokiDeleteTagHandler(deps), map[string]string{"label": "Akira", "categoryLabel": "Movies"})
	if rr.Code != http.StatusConflict {
		t.Fatalf("wrong category: expected 409, got %d", rr.Code)
	}
	if n := countWhere(t, deps.DB, `SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = 'Akira'`); n != 1 {
		t.Fatal("refused delete still removed assignments")
	}
	rr = postJSON(lokiDeleteTagHandler(deps), map[string]string{"label": "Akira", "categoryLabel": "Anime"})
	mustOK(t, "delete", rr.Code, rr.Body.String())
	var out struct {
		Assignments int `json:"assignments"`
	}
	json.Unmarshal(rr.Body.Bytes(), &out)
	if out.Assignments != 1 {
		t.Fatalf("reported %d removed assignments, want 1", out.Assignments)
	}
}
