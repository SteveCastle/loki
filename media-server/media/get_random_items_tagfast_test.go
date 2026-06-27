package media

import (
	"fmt"
	"sort"
	"testing"
)

// extractTagFilter must recognise exactly the "pure tag predicates joined by a
// single operator" shapes the swipe filter produces, and reject anything that
// would need the generic media-table query.
func TestExtractTagFilter(t *testing.T) {
	parseKind := func(q string) ([]string, tagFilterKind) {
		node, err := NewParser(q).Parse()
		if err != nil {
			t.Fatalf("parse %q: %v", q, err)
		}
		return extractTagFilter(node)
	}

	t.Run("single tag", func(t *testing.T) {
		labels, kind := parseKind(`tag:"swipe"`)
		if kind != tagFilterLeaf {
			t.Fatalf("kind = %v, want leaf", kind)
		}
		if len(labels) != 1 || labels[0] != "swipe" {
			t.Fatalf("labels = %v", labels)
		}
	})

	t.Run("OR union", func(t *testing.T) {
		labels, kind := parseKind(`tag:"a" OR tag:"b" OR tag:"c"`)
		if kind != tagFilterOr {
			t.Fatalf("kind = %v, want or", kind)
		}
		sort.Strings(labels)
		if fmt.Sprint(labels) != "[a b c]" {
			t.Fatalf("labels = %v", labels)
		}
	})

	t.Run("AND intersection", func(t *testing.T) {
		_, kind := parseKind(`tag:"a" AND tag:"b"`)
		if kind != tagFilterAnd {
			t.Fatalf("kind = %v, want and", kind)
		}
	})

	t.Run("implicit AND", func(t *testing.T) {
		_, kind := parseKind(`tag:"a" tag:"b"`)
		if kind != tagFilterAnd {
			t.Fatalf("kind = %v, want and", kind)
		}
	})

	rejects := map[string]string{
		"mixed AND/OR":  `tag:"a" OR (tag:"b" AND tag:"c")`,
		"non-tag path":  `path:"x"`,
		"NOT tag":       `NOT tag:"a"`,
		"tag wildcard":  `tag:"a*"`,
		"category":      `category:"x"`,
		"tag plus path": `tag:"a" AND path:"x"`,
		"tagcount":      `tagcount:>2`,
	}
	for name, q := range rejects {
		t.Run("reject "+name, func(t *testing.T) {
			_, kind := parseKind(q)
			if kind != tagFilterNone {
				t.Fatalf("kind = %v, want none for %q", kind, q)
			}
		})
	}
}

// The fast tag path must return the correct union / intersection sets — the
// same results the generic path would, just resolved straight from
// media_tag_by_category.
func TestGetRandomItemsTagUnionAndIntersection(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// m1: A,B   m2: A   m3: B   m4: (untagged-ish) C
	fixtures := map[string][]string{
		"/m/1.jpg": {"A", "B"},
		"/m/2.jpg": {"A"},
		"/m/3.jpg": {"B"},
		"/m/4.jpg": {"C"},
	}
	for path, tags := range fixtures {
		if _, err := db.Exec(
			`INSERT INTO media (path, description, size, hash, width, height) VALUES (?, ?, ?, ?, ?, ?)`,
			path, nil, 1, path, 10, 10,
		); err != nil {
			t.Fatalf("insert media %s: %v", path, err)
		}
		for _, tg := range tags {
			if _, err := db.Exec(
				`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)`,
				path, tg, "cat",
			); err != nil {
				t.Fatalf("insert tag %s/%s: %v", path, tg, err)
			}
		}
	}

	pathsOf := func(items []MediaItem) []string {
		out := make([]string, 0, len(items))
		for _, it := range items {
			out = append(out, it.Path)
		}
		sort.Strings(out)
		return out
	}

	// Union: A or B -> m1, m2, m3.
	union, _, err := GetRandomItems(db, 0, 50, `tag:"A" OR tag:"B"`, 7)
	if err != nil {
		t.Fatalf("union: %v", err)
	}
	if got := fmt.Sprint(pathsOf(union)); got != "[/m/1.jpg /m/2.jpg /m/3.jpg]" {
		t.Fatalf("union paths = %s", got)
	}

	// Intersection: A and B -> only m1.
	inter, _, err := GetRandomItems(db, 0, 50, `tag:"A" AND tag:"B"`, 7)
	if err != nil {
		t.Fatalf("intersection: %v", err)
	}
	if got := fmt.Sprint(pathsOf(inter)); got != "[/m/1.jpg]" {
		t.Fatalf("intersection paths = %s", got)
	}

	// Single tag -> m1, m2.
	single, _, err := GetRandomItems(db, 0, 50, `tag:"A"`, 7)
	if err != nil {
		t.Fatalf("single: %v", err)
	}
	if got := fmt.Sprint(pathsOf(single)); got != "[/m/1.jpg /m/2.jpg]" {
		t.Fatalf("single paths = %s", got)
	}
}
