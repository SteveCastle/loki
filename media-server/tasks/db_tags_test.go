package tasks

import (
	"database/sql"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// setupTagDB creates the minimal tag + media_tag_by_category schema that
// insertTagsForFile touches, matching the real column/PK layout (the PK on
// media_tag_by_category is what duplicate tags collide on).
func setupTagDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })

	if _, err := db.Exec(`CREATE TABLE tag (
		label TEXT PRIMARY KEY,
		category_label TEXT,
		weight REAL,
		preview TEXT,
		thumbnail_path_600 TEXT
	)`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TABLE media_tag_by_category (
		media_path TEXT,
		tag_label TEXT,
		category_label TEXT,
		weight REAL,
		time_stamp REAL,
		created_at INTEGER,
		PRIMARY KEY(media_path, tag_label, category_label, time_stamp)
	)`); err != nil {
		t.Fatal(err)
	}
	return db
}

func countTagsForFile(t *testing.T, db *sql.DB, path string) int {
	t.Helper()
	var n int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, path,
	).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n
}

// Re-tagging a file that already carries some of the generated tags must not
// fail: pre-existing assignments are silently skipped and only the new tags
// are inserted.
func TestInsertTagsForFile_SkipsExistingInsertsNew(t *testing.T) {
	db := setupTagDB(t)
	const path = "/media/a.jpg"

	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "sunset", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("first insert failed: %v", err)
	}
	if got := countTagsForFile(t, db, path); got != 1 {
		t.Fatalf("after first insert: got %d tags, want 1", got)
	}

	// Second run overlaps on "sunset" (the duplicate that previously aborted
	// the whole job) and adds "beach".
	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "sunset", Category: "Suggested"},
		{Label: "beach", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("second insert (with a pre-existing tag) failed: %v", err)
	}

	if got := countTagsForFile(t, db, path); got != 2 {
		t.Fatalf("after second insert: got %d tags, want 2 (sunset + beach)", got)
	}

	var hasBeach int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ? AND tag_label = 'beach'`,
		path,
	).Scan(&hasBeach); err != nil {
		t.Fatal(err)
	}
	if hasBeach != 1 {
		t.Fatalf("expected the new 'beach' tag to be inserted, got %d", hasBeach)
	}
}

// insertTagsForFile stamps created_at with the tag-application time (Unix
// seconds) so tag-driven views can date-sort by when the tag was applied.
func TestInsertTagsForFile_StampsCreatedAt(t *testing.T) {
	db := setupTagDB(t)
	const path = "/media/a.jpg"
	before := time.Now().Unix()

	if err := insertTagsForFile(db, path, []TagInfo{
		{Label: "sunset", Category: "Suggested"},
	}); err != nil {
		t.Fatalf("insert failed: %v", err)
	}

	var createdAt sql.NullInt64
	if err := db.QueryRow(
		`SELECT created_at FROM media_tag_by_category WHERE media_path = ? AND tag_label = 'sunset'`,
		path,
	).Scan(&createdAt); err != nil {
		t.Fatal(err)
	}
	if !createdAt.Valid {
		t.Fatal("created_at was NULL; expected the apply time to be stamped")
	}
	if createdAt.Int64 < before {
		t.Fatalf("expected created_at >= %d (apply time), got %d", before, createdAt.Int64)
	}
}
