package media

import (
	"database/sql"
	"testing"
)

// seedTaggedMedia inserts a small fixture: two files tagged "Exchange Student"
// and one untagged file, so we can prove a tag query selects a subset rather
// than the whole table.
func seedTaggedMedia(t *testing.T, db *sql.DB) {
	t.Helper()
	rows := []struct {
		path, tag string
	}{
		{"/lib/a.jpg", "Exchange Student"},
		{"/lib/b.jpg", "Exchange Student"},
		{"/lib/c.jpg", ""}, // untagged — must NOT be selected by the tag query
	}
	for _, r := range rows {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", r.path); err != nil {
			t.Fatalf("insert media %s: %v", r.path, err)
		}
		if r.tag != "" {
			if _, err := db.Exec(
				"INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)",
				r.path, r.tag, "people"); err != nil {
				t.Fatalf("insert tag for %s: %v", r.path, err)
			}
		}
	}
}

// TestGetPathsByQuery_QuotedMultiWordTag verifies that a properly quoted
// multi-word tag value selects exactly the tagged subset. This is the query the
// renderer now emits for tags containing spaces.
func TestGetPathsByQuery_QuotedMultiWordTag(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, `tag:"Exchange Student"`)
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 2 {
		t.Fatalf("quoted tag query selected %d paths, want 2 (%v)", len(paths), paths)
	}
}

// TestGetPathsByQuery_UnparseableQueryErrors is the regression guard for the
// reported bug: an unquoted multi-word tag (`tag:Exchange Student`) fails to
// parse. Previously this silently fell back to selecting the ENTIRE library,
// so a "describe these files" job ran against every item. It must now return an
// error so the calling job fails loudly instead.
func TestGetPathsByQuery_UnparseableQueryErrors(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, "tag:Exchange Student")
	if err == nil {
		t.Fatalf("expected an error for an unparseable query, got %d paths (%v)", len(paths), paths)
	}
	if paths != nil {
		t.Fatalf("expected no paths on parse failure, got %v", paths)
	}
}

// TestGetPathsByQuery_EmptyQuerySelectsAll confirms the legitimate "no filter"
// case still returns everything — only parse *failures* are treated as errors.
func TestGetPathsByQuery_EmptyQuerySelectsAll(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	seedTaggedMedia(t, db)

	paths, err := GetPathsByQuery(db, "")
	if err != nil {
		t.Fatalf("GetPathsByQuery() error = %v", err)
	}
	if len(paths) != 3 {
		t.Fatalf("empty query selected %d paths, want 3 (all) (%v)", len(paths), paths)
	}
}
