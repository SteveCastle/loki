package media

import (
	"database/sql"
	"fmt"
	"testing"

	_ "modernc.org/sqlite"
)

// setupTagRegistryDB creates the `tag` registry table SuggestTagsWithCategories
// reads from and seeds it with `n` tags that all share a common substring, so a
// short prefix matches every one of them.
func setupTagRegistryDB(t *testing.T, n int) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if _, err := db.Exec(`CREATE TABLE tag (
		label TEXT PRIMARY KEY,
		category_label TEXT,
		weight REAL,
		preview TEXT,
		thumbnail_path_600 TEXT
	)`); err != nil {
		t.Fatalf("create tag table: %v", err)
	}
	for i := 0; i < n; i++ {
		if _, err := db.Exec(
			`INSERT INTO tag (label, category_label) VALUES (?, ?)`,
			fmt.Sprintf("cat_tag_%05d", i), "Suggested",
		); err != nil {
			t.Fatalf("insert tag %d: %v", i, err)
		}
	}
	return db
}

// A short substring against a large tag set must not return everything — the
// result is capped at the requested limit. This is the typeahead-freeze guard.
func TestSuggestTagsWithCategories_RespectsLimit(t *testing.T) {
	db := setupTagRegistryDB(t, 5000)

	tags, err := SuggestTagsWithCategories(db, "tag", 50)
	if err != nil {
		t.Fatalf("SuggestTagsWithCategories: %v", err)
	}
	if len(tags) != 50 {
		t.Fatalf("got %d tags, want 50 (capped)", len(tags))
	}
}

// A non-positive or oversized limit falls back to the default cap rather than
// returning the whole table.
func TestSuggestTagsWithCategories_ClampsLimit(t *testing.T) {
	db := setupTagRegistryDB(t, 5000)

	for _, lim := range []int{0, -1, 9999} {
		tags, err := SuggestTagsWithCategories(db, "tag", lim)
		if err != nil {
			t.Fatalf("limit=%d: %v", lim, err)
		}
		if len(tags) != 25 {
			t.Fatalf("limit=%d: got %d tags, want 25 (default clamp)", lim, len(tags))
		}
	}
}

// The prefix still filters: a substring that matches nothing returns nothing.
func TestSuggestTagsWithCategories_FiltersByPrefix(t *testing.T) {
	db := setupTagRegistryDB(t, 100)

	tags, err := SuggestTagsWithCategories(db, "no_such_substring", 50)
	if err != nil {
		t.Fatalf("SuggestTagsWithCategories: %v", err)
	}
	if len(tags) != 0 {
		t.Fatalf("got %d tags, want 0 for a non-matching prefix", len(tags))
	}
}
