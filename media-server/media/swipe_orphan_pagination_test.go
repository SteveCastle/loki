package media

import (
	"fmt"
	"testing"
)

// TestSwipeFilterPaginationSkipsOrphansNoRepeats reproduces the swipe
// "loops over the same set of images before continuing" bug for tag-filtered
// swipe.
//
// Root cause: the fast tag path resolves candidate paths straight from
// media_tag_by_category, which can contain paths whose media row was deleted
// (orphans). assembleRandomItemsFromPaths drops those when the wide SELECT
// finds no media row, so a page returns FEWER items than the shuffle window it
// consumed. The swipe client advances its offset by the number of items it
// received (offset = items.length), while the server advances its shuffle
// window by `limit`. When the two diverge, consecutive windows overlap and the
// same real items get re-served over and over — pronounced when a tag has a
// high orphan rate (a real tag in the library has 105 orphans out of 114
// paths).
//
// The fix constrains the swipe universe to paths that actually have a media
// row, so every sampled path yields exactly one item and the client's
// offset stays aligned with the server's shuffle position.
func TestSwipeFilterPaginationSkipsOrphansNoRepeats(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	const tag = "loop"

	// 10 real paths: present in BOTH media and media_tag_by_category.
	real := make([]string, 0, 10)
	for i := 0; i < 10; i++ {
		p := fmt.Sprintf("/m/real-%02d.jpg", i)
		real = append(real, p)
		if _, err := db.Exec(
			`INSERT INTO media (path, description, size, hash, width, height) VALUES (?,?,?,?,?,?)`,
			p, nil, 1, p, 10, 10,
		); err != nil {
			t.Fatalf("insert media %s: %v", p, err)
		}
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?,?,?)`,
			p, tag, "cat",
		); err != nil {
			t.Fatalf("insert tag %s: %v", p, err)
		}
	}

	// 30 orphan paths: tagged but NO media row (deleted media, dangling tags).
	for i := 0; i < 30; i++ {
		p := fmt.Sprintf("/m/orphan-%02d.jpg", i)
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?,?,?)`,
			p, tag, "cat",
		); err != nil {
			t.Fatalf("insert orphan tag %s: %v", p, err)
		}
	}

	// Paginate exactly like the swipe client: stable seed, and the next offset
	// is the count of items received so far (offset = items.length).
	const seed = int64(42)
	const limit = 5
	seen := map[string]int{}
	offset := 0
	for iter := 0; iter < 100; iter++ {
		items, hasMore, err := GetRandomItems(db, offset, limit, `tag:"loop"`, seed)
		if err != nil {
			t.Fatalf("page at offset %d: %v", offset, err)
		}
		for _, it := range items {
			seen[it.Path]++
		}
		if !hasMore {
			break
		}
		if len(items) == 0 {
			// Server says "more" but returned nothing: the client's offset can
			// never catch up to the shuffle window — the swipe view wedges.
			t.Fatalf("page returned 0 items with hasMore=true at offset %d: pagination cannot progress", offset)
		}
		offset += len(items)
	}

	// Every real item must be served exactly once; orphans never appear.
	for p, n := range seen {
		if n != 1 {
			t.Errorf("path %s served %d times; swipe pagination must not repeat items", p, n)
		}
	}
	if len(seen) != len(real) {
		t.Errorf("saw %d distinct items, want %d real items (orphans must be excluded, reals must all appear once)", len(seen), len(real))
	}
}
