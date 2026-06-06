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
	if len(params) != 0 {
		t.Fatalf("expected no params, got %v", params)
	}
}

func TestBuildMediaQueryTagInclude(t *testing.T) {
	sql, params := BuildMediaQuery([]Predicate{{Type: "tag", Value: "portrait"}}, "AND")
	if !strings.Contains(sql, "EXISTS") || !strings.Contains(sql, "mtc.tag_label = ?") {
		t.Fatalf("bad tag sql: %q", sql)
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
		!strings.Contains(sql, "media.description NOT LIKE ?") ||
		!strings.Contains(sql, "media.hash LIKE ?") {
		t.Fatalf("bad like sql: %q", sql)
	}
	want := []string{"%a%", "%b%", "%c%"}
	for i := range want {
		if params[i] != want[i] {
			t.Fatalf("param %d = %q want %q", i, params[i], want[i])
		}
	}
}

func TestBuildMediaQueryOrJoin(t *testing.T) {
	sql, _ := BuildMediaQuery([]Predicate{
		{Type: "tag", Value: "a"}, {Type: "tag", Value: "b"},
	}, "OR")
	if !strings.Contains(norm(sql), ") OR (") {
		t.Fatalf("expected OR join: %q", sql)
	}
}
