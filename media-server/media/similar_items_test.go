package media

import (
	"database/sql"
	"fmt"
	"testing"

	_ "modernc.org/sqlite"
)

func newSimilarItemsDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	return db
}

func TestFilterExistingMediaPathsDropsOrphansPreservesOrder(t *testing.T) {
	db := newSimilarItemsDB(t)
	for _, p := range []string{"/lib/a.jpg", "/lib/c.jpg"} {
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
			t.Fatalf("insert %s: %v", p, err)
		}
	}

	// b.jpg is an orphan (embedding without a media row) and must be dropped;
	// the survivors must keep their ranked order.
	got, err := FilterExistingMediaPaths(db, []string{"/lib/c.jpg", "/lib/b.jpg", "/lib/a.jpg"})
	if err != nil {
		t.Fatalf("filter: %v", err)
	}
	want := []string{"/lib/c.jpg", "/lib/a.jpg"}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

func TestFilterExistingMediaPathsChunks(t *testing.T) {
	db := newSimilarItemsDB(t)
	// More paths than one IN-list chunk to exercise the chunked loop.
	n := filterChunkSize + 50
	paths := make([]string, 0, n)
	for i := 0; i < n; i++ {
		p := fmt.Sprintf("/lib/%04d.jpg", i)
		paths = append(paths, p)
		if i%2 == 0 { // only even indexes exist in media
			if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", p); err != nil {
				t.Fatalf("insert %s: %v", p, err)
			}
		}
	}
	got, err := FilterExistingMediaPaths(db, paths)
	if err != nil {
		t.Fatalf("filter: %v", err)
	}
	if len(got) != (n+1)/2 {
		t.Fatalf("expected %d survivors, got %d", (n+1)/2, len(got))
	}
	// Order must be preserved across chunk boundaries.
	for i := 1; i < len(got); i++ {
		if got[i-1] >= got[i] {
			t.Fatalf("order not preserved: %s before %s", got[i-1], got[i])
		}
	}
}

func TestGetItemsByPathsPreservesOrderAndAttachesTags(t *testing.T) {
	db := newSimilarItemsDB(t)
	for _, p := range []string{"/lib/a.jpg", "/lib/b.jpg"} {
		if _, err := db.Exec(
			"INSERT INTO media (path, description, size) VALUES (?, ?, 100)", p, "desc-"+p,
		); err != nil {
			t.Fatalf("insert %s: %v", p, err)
		}
	}
	if _, err := db.Exec(
		"INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)",
		"/lib/b.jpg", "sunset", "Scenes",
	); err != nil {
		t.Fatalf("insert tag: %v", err)
	}

	// Request order is (b, missing, a): b first, the missing row skipped.
	items, err := GetItemsByPaths(db, []string{"/lib/b.jpg", "/lib/missing.jpg", "/lib/a.jpg"})
	if err != nil {
		t.Fatalf("get items: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}
	if items[0].Path != "/lib/b.jpg" || items[1].Path != "/lib/a.jpg" {
		t.Fatalf("order not preserved: %s, %s", items[0].Path, items[1].Path)
	}
	if len(items[0].Tags) != 1 || items[0].Tags[0].Label != "sunset" {
		t.Fatalf("expected sunset tag on b.jpg, got %v", items[0].Tags)
	}
	if items[1].Tags == nil || len(items[1].Tags) != 0 {
		t.Fatalf("expected empty (non-nil) tags on a.jpg, got %v", items[1].Tags)
	}
}
