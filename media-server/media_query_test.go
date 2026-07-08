// media-server/media_query_test.go
package main

import (
	"strings"
	"testing"
)

func norm(s string) string { return strings.Join(strings.Fields(s), " ") }

func TestBuildMediaQueryEmpty(t *testing.T) {
	sql, params := BuildMediaQuery(nil, "AND")
	if !strings.Contains(norm(sql), "FROM media") {
		t.Fatalf("expected base query, got %q", sql)
	}
	if strings.Contains(sql, "description") {
		t.Fatalf("description must not be selected: %q", sql)
	}
	if len(params) != 0 {
		t.Fatalf("expected no params, got %v", params)
	}
}

func TestBuildMediaQuerySingleTagDrivesFromIndex(t *testing.T) {
	// Single include-tag: drive FROM the indexed tag table, no media scan,
	// no EXISTS, no ORDER BY, description not selected.
	sql, params := BuildMediaQuery([]Predicate{{Type: "tag", Value: "portrait"}}, "AND")
	if !strings.Contains(sql, "FROM media_tag_by_category mtcw") {
		t.Fatalf("expected tag-table drive: %q", sql)
	}
	if !strings.Contains(sql, "WHERE mtcw.tag_label = ?") {
		t.Fatalf("expected tag_label filter: %q", sql)
	}
	if strings.Contains(sql, "EXISTS") {
		t.Fatalf("single tag should not use EXISTS: %q", sql)
	}
	if strings.Contains(sql, "ORDER BY") || strings.Contains(sql, "description") {
		t.Fatalf("no ORDER BY / no description expected: %q", sql)
	}
	if len(params) != 1 || params[0] != "portrait" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryLikeWrapping(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "path", Value: "a"},
		{Type: "description", Value: "b", Exclude: true},
		{Type: "hash", Value: "c"},
	}, "AND")
	if !strings.Contains(sql, "media.path LIKE ?") ||
		!strings.Contains(sql, "media.description NOT LIKE ?") || // filtered...
		!strings.Contains(sql, "media.hash LIKE ?") {
		t.Fatalf("bad like sql: %q", sql)
	}
	if strings.Contains(sql, "AS description") { // ...but not returned
		t.Fatalf("description must not be selected: %q", sql)
	}
	want := []string{"%a%", "%b%", "%c%"}
	for i := range want {
		if params[i] != want[i] {
			t.Fatalf("param %d = %q want %q", i, params[i], want[i])
		}
	}
}

func TestBuildMediaQueryFacesUngrouped(t *testing.T) {
	// faces:ungrouped — EXISTS over the face table for unassigned faces.
	sql, params := BuildMediaQuery([]Predicate{{Type: "faces", Value: "ungrouped"}}, "AND")
	if !strings.Contains(sql, "EXISTS (SELECT 1 FROM face f WHERE f.media_path = media.path AND COALESCE(f.person_id, 0) = 0)") {
		t.Fatalf("expected ungrouped-faces EXISTS: %q", sql)
	}
	if len(params) != 0 {
		t.Fatalf("expected no params, got %v", params)
	}

	// Excluded form inverts; combined with a tag it stays a conjunct.
	sql, _ = BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "portrait"},
		{Type: "faces", Value: "ungrouped", Exclude: true},
	}, "AND")
	if !strings.Contains(sql, "NOT EXISTS (SELECT 1 FROM face f") {
		t.Fatalf("expected NOT EXISTS for excluded faces predicate: %q", sql)
	}

	// Unknown values match nothing — never everything.
	sql, _ = BuildMediaQuery([]Predicate{{Type: "faces", Value: "bogus"}}, "AND")
	if !strings.Contains(sql, "1=0") {
		t.Fatalf("unknown faces value must match nothing: %q", sql)
	}
}

func TestBuildMediaQueryOrSetUsesInLookup(t *testing.T) {
	// OR-set of include-tags drives from an indexed tag_label IN, not a scan.
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a"}, {Type: "tag", Value: "b"},
	}, "OR")
	if !strings.Contains(sql, "FROM media_tag_by_category mtcw") ||
		!strings.Contains(sql, "WHERE mtcw.tag_label IN (?, ?)") {
		t.Fatalf("expected IN drive: %q", sql)
	}
	if strings.Contains(sql, "EXISTS") || strings.Contains(sql, "ORDER BY") {
		t.Fatalf("no EXISTS / ORDER BY expected: %q", sql)
	}
	if len(params) != 2 || params[0] != "a" || params[1] != "b" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryOrJoinUnions(t *testing.T) {
	// [a, b(OR)] reads "a OR b" — must be a UNION (the reported bug).
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a"},
		{Type: "tag", Value: "b", Join: "OR"},
	}, "AND")
	if !strings.Contains(sql, "WHERE mtcw.tag_label IN (?, ?)") {
		t.Fatalf("expected union IN, got %q", sql)
	}
	if strings.Contains(sql, "EXISTS") {
		t.Fatalf("union must not be an intersection: %q", sql)
	}
	if len(params) != 2 || params[0] != "a" || params[1] != "b" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryAllOrUnionsEveryTag(t *testing.T) {
	// First chip's join is ignored (base); all-OR connectors union a,b,c.
	sql, _ := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a", Join: "AND"},
		{Type: "tag", Value: "b", Join: "OR"},
		{Type: "tag", Value: "c", Join: "OR"},
	}, "AND")
	if !strings.Contains(sql, "WHERE mtcw.tag_label IN (?, ?, ?)") {
		t.Fatalf("expected union of all three: %q", sql)
	}
}

func TestBuildMediaQueryMixedConnectorsScan(t *testing.T) {
	// [a, b(OR), c(AND)] reads "(a OR b) AND c" — left-to-right media scan.
	sql, _ := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a"},
		{Type: "tag", Value: "b", Join: "OR"},
		{Type: "tag", Value: "c", Join: "AND"},
	}, "AND")
	n := norm(sql)
	if !strings.Contains(n, ") OR (") || !strings.Contains(n, ") AND (") {
		t.Fatalf("expected left-to-right (a OR b) AND c: %q", sql)
	}
}

func TestBuildMediaQueryCategoryDrivesFromIndex(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{{Type: "category", Value: "Artists"}}, "AND")
	if !strings.Contains(sql, "SELECT DISTINCT media_path FROM media_tag_by_category WHERE category_label = ?") {
		t.Fatalf("expected DISTINCT category drive: %q", sql)
	}
	if !strings.Contains(sql, "JOIN media ON media.path = cat.media_path") {
		t.Fatalf("expected join to media: %q", sql)
	}
	if !strings.Contains(sql, "NULL AS weight") {
		t.Fatalf("expected NULL tag columns: %q", sql)
	}
	if len(params) != 1 || params[0] != "Artists" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryTagDrivesOverCategory(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "t"}, {Type: "category", Value: "C"},
	}, "AND")
	if !strings.Contains(sql, "FROM media_tag_by_category mtcw") {
		t.Fatalf("expected tag drive: %q", sql)
	}
	if !strings.Contains(sql, "mtc.category_label = ?") {
		t.Fatalf("expected category EXISTS conjunct: %q", sql)
	}
	if len(params) != 2 || params[0] != "t" || params[1] != "C" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryAndCombo(t *testing.T) {
	// AND of two tags: drive from the first, second is a conjunct EXISTS.
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a"}, {Type: "tag", Value: "b"},
	}, "AND")
	if !strings.Contains(sql, "FROM media_tag_by_category mtcw") ||
		!strings.Contains(sql, "WHERE mtcw.tag_label = ?") ||
		!strings.Contains(sql, "AND (EXISTS") {
		t.Fatalf("expected drive + EXISTS conjunct: %q", sql)
	}
	if len(params) != 2 || params[0] != "a" || params[1] != "b" {
		t.Fatalf("bad params: %v", params)
	}
}

func TestBuildMediaQueryIncludeTagColumns(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{{Type: "tag", Value: "cat"}}, "AND")
	if !strings.Contains(sql, "mtcw.weight AS weight") ||
		!strings.Contains(sql, "mtcw.time_stamp AS time_stamp") {
		t.Fatalf("expected weight/timestamp columns: %q", sql)
	}
	if len(params) == 0 || params[0] != "cat" {
		t.Fatalf("expected drive param first: %v", params)
	}
}

// ---- Visual predicate tests ----

func TestBuildMediaQueryVisualInClause(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "visual", Value: "red car", Resolved: []string{"a.jpg", "b.jpg"}},
	}, "AND")
	n := norm(sql)
	if !strings.Contains(n, "media.path IN (?, ?)") {
		t.Fatalf("expected IN clause: %q", sql)
	}
	if len(params) != 2 || params[0] != "a.jpg" || params[1] != "b.jpg" {
		t.Fatalf("expected params [a.jpg b.jpg], got %v", params)
	}
}

func TestBuildMediaQueryVisualExclude(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "visual", Value: "red car", Resolved: []string{"a.jpg", "b.jpg"}, Exclude: true},
	}, "AND")
	n := norm(sql)
	if !strings.Contains(n, "media.path NOT IN (?, ?)") {
		t.Fatalf("expected NOT IN clause: %q", sql)
	}
	if len(params) != 2 || params[0] != "a.jpg" || params[1] != "b.jpg" {
		t.Fatalf("expected params [a.jpg b.jpg], got %v", params)
	}
}

func TestBuildMediaQueryVisualEmptyResolved(t *testing.T) {
	// Include with empty Resolved → matches nothing (1=0).
	sqlInc, _ := BuildMediaQuery([]Predicate{
		{Type: "visual", Value: "red car"},
	}, "AND")
	if !strings.Contains(sqlInc, "1=0") {
		t.Fatalf("empty include should produce 1=0: %q", sqlInc)
	}

	// Exclude with empty Resolved → removes nothing (1=1).
	sqlExc, _ := BuildMediaQuery([]Predicate{
		{Type: "visual", Value: "red car", Exclude: true},
	}, "AND")
	if !strings.Contains(sqlExc, "1=1") {
		t.Fatalf("empty exclude should produce 1=1: %q", sqlExc)
	}
}

func TestBuildMediaQueryVisualComposesWithTag(t *testing.T) {
	// Visual + tag in AND mode: tag drives the index, visual becomes an IN conjunct.
	predicates := []Predicate{
		{Type: "visual", Value: "x", Resolved: []string{"a.jpg"}},
		{Type: "tag", Value: "fav", Join: "AND"},
	}
	sql, params := BuildMediaQuery(predicates, "AND")
	n := norm(sql)
	if !strings.Contains(n, "media.path IN (?)") {
		t.Fatalf("expected visual IN clause: %q", sql)
	}
	if !strings.Contains(n, "tag_label") {
		t.Fatalf("expected tag reference: %q", sql)
	}
	foundA, foundFav := false, false
	for _, p := range params {
		if s, ok := p.(string); ok {
			switch s {
			case "a.jpg":
				foundA = true
			case "fav":
				foundFav = true
			}
		}
	}
	if !foundA || !foundFav {
		t.Fatalf("expected both 'a.jpg' and 'fav' in params, got %v", params)
	}
}

func TestSortItemsByScore(t *testing.T) {
	items := []map[string]any{
		{"path": "a"},
		{"path": "b"},
		{"path": "c"},
	}
	scores := map[string]float32{"a": 0.1, "b": 0.9, "c": 0.5}
	sortItemsByScore(items, scores)

	wantOrder := []string{"b", "c", "a"}
	for i, it := range items {
		if it["path"] != wantOrder[i] {
			t.Fatalf("position %d: got path %q, want %q", i, it["path"], wantOrder[i])
		}
		want32 := scores[wantOrder[i]]
		got, ok := it["score"].(float32)
		if !ok {
			t.Fatalf("item %q: score not float32: %T", wantOrder[i], it["score"])
		}
		if got != want32 {
			t.Fatalf("item %q: score = %v, want %v", wantOrder[i], got, want32)
		}
	}
}
