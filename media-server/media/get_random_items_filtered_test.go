package media

import (
	"fmt"
	"testing"
)

// TestGetRandomItemsFilteredPaginationNoRepeats guards the swipe pagination
// contract for the *filtered* code path (e.g. `tag:"X"` from the swipe UI's
// localStorage filter).
//
// The bug this test pins down: GetRandomItems used `ORDER BY RANDOM()`
// per request and ignored the session seed, so consecutive paginated
// calls produced *independent* random shuffles. The client's offset
// pointed into a different permutation each page, so items from page 1
// silently reappeared in page 2 — the "cycles of repeating media"
// reported in the swipe view.
//
// Contract: given a stable seed, two sequential pages of size N must
// together return 2N distinct items when the matching universe has at
// least 2N items.
func TestGetRandomItemsFilteredPaginationNoRepeats(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Insert 50 media rows, all tagged with "swipe". The swipe UI's
	// tag filter produces `tag:"swipe"` from selectedTags in localStorage,
	// so this mirrors the production path exactly.
	const total = 50
	for i := 0; i < total; i++ {
		path := fmt.Sprintf("/m/%03d.jpg", i)
		if _, err := db.Exec(
			`INSERT INTO media (path, description, size, hash, width, height) VALUES (?, ?, ?, ?, ?, ?)`,
			path, nil, 1024, fmt.Sprintf("h%03d", i), 100, 100,
		); err != nil {
			t.Fatalf("insert media %d: %v", i, err)
		}
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)`,
			path, "swipe", "swipe",
		); err != nil {
			t.Fatalf("insert tag %d: %v", i, err)
		}
	}

	const pageSize = 25
	const seed int64 = 42

	page1, _, err := GetRandomItems(db, 0, pageSize, `tag:"swipe"`, seed)
	if err != nil {
		t.Fatalf("page1: %v", err)
	}
	page2, _, err := GetRandomItems(db, pageSize, pageSize, `tag:"swipe"`, seed)
	if err != nil {
		t.Fatalf("page2: %v", err)
	}

	if len(page1) != pageSize {
		t.Fatalf("page1 expected %d items, got %d", pageSize, len(page1))
	}
	if len(page2) != pageSize {
		t.Fatalf("page2 expected %d items, got %d", pageSize, len(page2))
	}

	seen := make(map[string]int, total)
	for _, it := range page1 {
		seen[it.Path]++
	}
	for _, it := range page2 {
		seen[it.Path]++
	}
	var dupes []string
	for p, n := range seen {
		if n > 1 {
			dupes = append(dupes, fmt.Sprintf("%s(x%d)", p, n))
		}
	}
	if len(dupes) > 0 {
		t.Fatalf("filtered pagination must not repeat items across pages with a stable seed; duplicates: %v", dupes)
	}
	if len(seen) != 2*pageSize {
		t.Fatalf("page1 + page2 expected %d distinct items, got %d", 2*pageSize, len(seen))
	}
}

// TestGetRandomItemsFilteredSeedDeterministic guards that the same seed
// produces the same ordering across calls — needed for the swipe
// client's offset-based pagination contract.
func TestGetRandomItemsFilteredSeedDeterministic(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	const total = 40
	for i := 0; i < total; i++ {
		path := fmt.Sprintf("/m/%03d.jpg", i)
		if _, err := db.Exec(
			`INSERT INTO media (path, description, size, hash, width, height) VALUES (?, ?, ?, ?, ?, ?)`,
			path, nil, 1024, fmt.Sprintf("h%03d", i), 100, 100,
		); err != nil {
			t.Fatalf("insert media %d: %v", i, err)
		}
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)`,
			path, "swipe", "swipe",
		); err != nil {
			t.Fatalf("insert tag %d: %v", i, err)
		}
	}

	// Ask for a strict subset (limit < total) so the inner LIMIT actually
	// selects a window of the shuffle. With limit == total, the outer
	// IN-list returns everything and order can come back in PK order by
	// accident, masking non-determinism.
	const pageSize = 10
	first, _, err := GetRandomItems(db, 0, pageSize, `tag:"swipe"`, 1234)
	if err != nil {
		t.Fatalf("first: %v", err)
	}
	second, _, err := GetRandomItems(db, 0, pageSize, `tag:"swipe"`, 1234)
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if len(first) != len(second) {
		t.Fatalf("len mismatch: %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Path != second[i].Path {
			t.Fatalf("same seed must produce same order; differs at index %d: %q vs %q",
				i, first[i].Path, second[i].Path)
		}
	}
}
