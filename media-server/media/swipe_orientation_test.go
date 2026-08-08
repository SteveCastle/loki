package media

import (
	"testing"
)

// orientedRows is an untagged mixed-orientation library: 3 portrait,
// 3 landscape, 1 square, 1 with unknown dimensions. No row carries a tag —
// the swipe universe is the media table itself, so every row (except the
// unknown one under a restriction) must still be swipeable.
var orientedRows = []struct {
	path          string
	width, height interface{}
}{
	{"/o/port1.jpg", 1080, 1920},
	{"/o/port2.jpg", 720, 1280},
	{"/o/port3.jpg", 600, 800},
	{"/o/land1.jpg", 1920, 1080},
	{"/o/land2.jpg", 1280, 720},
	{"/o/land3.jpg", 800, 600},
	{"/o/square.jpg", 512, 512},
	{"/o/unknown.jpg", nil, nil},
}

func pathSet(items []MediaItem) map[string]bool {
	out := make(map[string]bool, len(items))
	for _, it := range items {
		out[it.Path] = true
	}
	return out
}

// TestSwipeOrientationSamplerPath pins the unfiltered-shuffle contract: the
// sampler universe is every media row (tags not required), and an orientation
// restriction keeps portrait+square / landscape+square while excluding items
// with unknown dimensions.
func TestSwipeOrientationSamplerPath(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	// The sampler is a package-global cache; isolate this test from any
	// snapshot another test built against its own DB, and clean up after.
	ResetRandomSampleCache()
	t.Cleanup(ResetRandomSampleCache)

	for _, r := range orientedRows {
		if _, err := db.Exec(
			`INSERT INTO media (path, width, height) VALUES (?, ?, ?)`,
			r.path, r.width, r.height,
		); err != nil {
			t.Fatalf("seed %s: %v", r.path, err)
		}
	}

	// Unrestricted: every row, tagged or not, known dimensions or not.
	all, _, err := GetRandomItems(db, 0, 50, "", 7, "")
	if err != nil {
		t.Fatalf("GetRandomItems: %v", err)
	}
	if len(all) != len(orientedRows) {
		t.Fatalf("unrestricted swipe pool has %d items, want all %d media rows (untagged included)", len(all), len(orientedRows))
	}

	portrait, _, err := GetRandomItems(db, 0, 50, "", 7, "portrait")
	if err != nil {
		t.Fatalf("GetRandomItems portrait: %v", err)
	}
	got := pathSet(portrait)
	want := map[string]bool{
		"/o/port1.jpg": true, "/o/port2.jpg": true, "/o/port3.jpg": true,
		"/o/square.jpg": true, // square displays fine either way
	}
	if len(got) != len(want) {
		t.Fatalf("portrait pool = %v, want %v", got, want)
	}
	for p := range want {
		if !got[p] {
			t.Fatalf("portrait pool missing %s (got %v)", p, got)
		}
	}

	landscape, _, err := GetRandomItems(db, 0, 50, "", 7, "landscape")
	if err != nil {
		t.Fatalf("GetRandomItems landscape: %v", err)
	}
	got = pathSet(landscape)
	for _, p := range []string{"/o/land1.jpg", "/o/land2.jpg", "/o/land3.jpg", "/o/square.jpg"} {
		if !got[p] {
			t.Fatalf("landscape pool missing %s (got %v)", p, got)
		}
	}
	if len(got) != 4 {
		t.Fatalf("landscape pool = %v, want exactly land1-3 + square", got)
	}
}

// TestSwipeOrientationSamplerPagination guards the offset contract under a
// restriction: the client's offset counts items it received, so filtered
// pages with a fixed seed must be disjoint and cover the restricted pool
// exactly — the filter runs while walking the shuffle, never after slicing.
func TestSwipeOrientationSamplerPagination(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	ResetRandomSampleCache()
	t.Cleanup(ResetRandomSampleCache)

	for _, r := range orientedRows {
		if _, err := db.Exec(
			`INSERT INTO media (path, width, height) VALUES (?, ?, ?)`,
			r.path, r.width, r.height,
		); err != nil {
			t.Fatalf("seed %s: %v", r.path, err)
		}
	}

	const seed = int64(4242)
	seen := map[string]bool{}
	offset := 0
	for {
		page, hasMore, err := GetRandomItems(db, offset, 2, "", seed, "portrait")
		if err != nil {
			t.Fatalf("page at offset %d: %v", offset, err)
		}
		for _, it := range page {
			if seen[it.Path] {
				t.Fatalf("item %s repeated across restricted pages (offset drift)", it.Path)
			}
			seen[it.Path] = true
		}
		offset += len(page)
		if !hasMore || len(page) == 0 {
			break
		}
		if offset > 50 {
			t.Fatal("pagination did not terminate")
		}
	}
	if len(seen) != 4 { // 3 portrait + square
		t.Fatalf("restricted pagination covered %d items, want 4: %v", len(seen), seen)
	}
}

// TestSwipeOrientationTagFilteredPath pins the tag-filtered swipe path: the
// orientation restriction combines with the tag universe before the shuffle.
func TestSwipeOrientationTagFilteredPath(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	ResetRandomSampleCache()
	t.Cleanup(ResetRandomSampleCache)

	for _, r := range orientedRows {
		if _, err := db.Exec(
			`INSERT INTO media (path, width, height) VALUES (?, ?, ?)`,
			r.path, r.width, r.height,
		); err != nil {
			t.Fatalf("seed %s: %v", r.path, err)
		}
	}
	// Tag one portrait, one landscape, and the unknown-dims item.
	for _, p := range []string{"/o/port1.jpg", "/o/land1.jpg", "/o/unknown.jpg"} {
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, 'pick', 'c')`, p,
		); err != nil {
			t.Fatalf("tag %s: %v", p, err)
		}
	}

	items, _, err := GetRandomItems(db, 0, 50, `tag:"pick"`, 7, "portrait")
	if err != nil {
		t.Fatalf("GetRandomItems: %v", err)
	}
	got := pathSet(items)
	if len(got) != 1 || !got["/o/port1.jpg"] {
		t.Fatalf("tag+portrait pool = %v, want only /o/port1.jpg", got)
	}

	// Unrestricted tag query still returns all tagged items.
	items, _, err = GetRandomItems(db, 0, 50, `tag:"pick"`, 7, "")
	if err != nil {
		t.Fatalf("GetRandomItems: %v", err)
	}
	if len(items) != 3 {
		t.Fatalf("tag pool has %d items, want 3", len(items))
	}
}

// TestSwipeOrientationGenericQueryPath pins the generic (non-tag-fast) SQL
// path: orientation joins the WHERE clause, and untagged items are included
// (the old always-require-a-tag filter is gone).
func TestSwipeOrientationGenericQueryPath(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()
	ResetRandomSampleCache()
	t.Cleanup(ResetRandomSampleCache)

	for _, r := range orientedRows {
		if _, err := db.Exec(
			`INSERT INTO media (path, width, height) VALUES (?, ?, ?)`,
			r.path, r.width, r.height,
		); err != nil {
			t.Fatalf("seed %s: %v", r.path, err)
		}
	}

	// filetype:image is not a pure tag query, so it takes the generic path.
	items, _, err := GetRandomItems(db, 0, 50, "filetype:image", 7, "landscape")
	if err != nil {
		t.Fatalf("GetRandomItems: %v", err)
	}
	got := pathSet(items)
	want := map[string]bool{
		"/o/land1.jpg": true, "/o/land2.jpg": true, "/o/land3.jpg": true,
		"/o/square.jpg": true,
	}
	if len(got) != len(want) {
		t.Fatalf("generic landscape pool = %v, want %v", got, want)
	}
	for p := range want {
		if !got[p] {
			t.Fatalf("generic landscape pool missing %s", p)
		}
	}
}

// TestFilterSwipePaths pins the shared candidate filter used by similar and
// feed modes: input order preserved, orphans dropped, orientation applied.
func TestFilterSwipePaths(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	for _, r := range orientedRows {
		if _, err := db.Exec(
			`INSERT INTO media (path, width, height) VALUES (?, ?, ?)`,
			r.path, r.width, r.height,
		); err != nil {
			t.Fatalf("seed %s: %v", r.path, err)
		}
	}

	in := []string{
		"/o/land2.jpg",
		"/o/ghost.jpg", // no media row — an orphan embedding
		"/o/port1.jpg",
		"/o/square.jpg",
		"/o/unknown.jpg",
		"/o/port3.jpg",
	}

	got, err := FilterSwipePaths(db, in, "portrait")
	if err != nil {
		t.Fatalf("FilterSwipePaths: %v", err)
	}
	want := []string{"/o/port1.jpg", "/o/square.jpg", "/o/port3.jpg"}
	if len(got) != len(want) {
		t.Fatalf("FilterSwipePaths = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("FilterSwipePaths order mismatch at %d: %v, want %v", i, got, want)
		}
	}

	// Unrestricted form only drops the orphan.
	got, err = FilterSwipePaths(db, in, "")
	if err != nil {
		t.Fatalf("FilterSwipePaths: %v", err)
	}
	if len(got) != 5 {
		t.Fatalf("unrestricted FilterSwipePaths kept %d paths, want 5 (orphan dropped): %v", len(got), got)
	}
}
