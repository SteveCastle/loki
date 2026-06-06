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

func TestBuildMediaQueryFaceted(t *testing.T) {
	// a(AND) drives; b/c(OR) become a conjunct OR-group of EXISTS.
	sql, params := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a", Join: "AND"},
		{Type: "tag", Value: "b", Join: "OR"},
		{Type: "tag", Value: "c", Join: "OR"},
	}, "AND")
	n := norm(sql)
	if !strings.Contains(n, "FROM media_tag_by_category mtcw") {
		t.Fatalf("expected tag-table drive: %q", sql)
	}
	if !strings.Contains(n, "WHERE mtcw.tag_label = ?") {
		t.Fatalf("expected drive filter: %q", sql)
	}
	if !strings.Contains(n, ") OR (") {
		t.Fatalf("expected OR group for b/c: %q", sql)
	}
	want := []any{"a", "b", "c"} // drive a, then EXISTS b, c
	if len(params) != len(want) {
		t.Fatalf("expected %d params, got %d: %v", len(want), len(params), params)
	}
	for i := range want {
		if params[i] != want[i] {
			t.Fatalf("param %d = %v want %v", i, params[i], want[i])
		}
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
