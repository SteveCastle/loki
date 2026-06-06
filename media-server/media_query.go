// media-server/media_query.go
package main

import "strings"

// Predicate mirrors src/renderer/query/types.ts Predicate.
type Predicate struct {
	Type    string `json:"type"`    // tag|category|path|description|hash
	Value   string `json:"value"`
	Exclude bool   `json:"exclude"`
	Join    string `json:"join"` // "AND" | "OR" | "" (empty falls back to mode)
}

const baseColumns = "media.path, media.description, media.elo, media.height, media.width"

func clauseFor(p Predicate, params *[]any) string {
	like := "%" + p.Value + "%"
	switch p.Type {
	case "tag":
		*params = append(*params, p.Value)
		if p.Exclude {
			return "(NOT EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.tag_label = ?))"
		}
		return "(EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.tag_label = ?))"
	case "category":
		*params = append(*params, p.Value)
		if p.Exclude {
			return "(NOT EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.category_label = ?))"
		}
		return "(EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.category_label = ?))"
	case "path":
		*params = append(*params, like)
		if p.Exclude {
			return "(media.path NOT LIKE ?)"
		}
		return "(media.path LIKE ?)"
	case "description":
		*params = append(*params, like)
		if p.Exclude {
			return "(media.description NOT LIKE ?)"
		}
		return "(media.description LIKE ?)"
	case "hash":
		*params = append(*params, like)
		if p.Exclude {
			return "(media.hash NOT LIKE ?)"
		}
		return "(media.hash LIKE ?)"
	}
	return ""
}

// BuildMediaQuery returns SQL + params. Always selects 9 columns (the 4 tag
// columns are NULL unless there's an include-tag to LEFT JOIN on) so the
// handler can scan by position regardless of predicate mix.
func BuildMediaQuery(predicates []Predicate, mode string) (string, []any) {
	valid := []Predicate{}
	for _, p := range predicates {
		if p.Value != "" {
			valid = append(valid, p)
		}
	}

	// First INCLUDE tag drives the LEFT JOIN that surfaces weight/tag/timestamp.
	primaryTag := ""
	for _, p := range valid {
		if p.Type == "tag" && !p.Exclude {
			primaryTag = p.Value
			break
		}
	}

	params := []any{}
	var selectClause, order string
	if primaryTag != "" {
		selectClause = "SELECT " + baseColumns +
			", mtcw.weight AS weight, mtcw.tag_label AS tag_label, " +
			"mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at " +
			"FROM media LEFT JOIN media_tag_by_category mtcw " +
			"ON mtcw.media_path = media.path AND mtcw.tag_label = ?"
		params = append(params, primaryTag) // JOIN param first
		order = " ORDER BY mtcw.weight"
	} else {
		selectClause = "SELECT " + baseColumns +
			", NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at FROM media"
	}

	if len(valid) == 0 {
		return selectClause, params
	}

	defaultJoin := "AND"
	if mode == "OR" {
		defaultJoin = "OR"
	}
	joinOf := func(p Predicate) string {
		if p.Join == "AND" || p.Join == "OR" {
			return p.Join
		}
		return defaultJoin
	}
	andClauses := []string{}
	orClauses := []string{}
	for _, p := range valid {
		if joinOf(p) != "OR" {
			andClauses = append(andClauses, clauseFor(p, &params))
		}
	}
	for _, p := range valid {
		if joinOf(p) == "OR" {
			orClauses = append(orClauses, clauseFor(p, &params))
		}
	}
	pieces := append([]string{}, andClauses...)
	if len(orClauses) > 0 {
		pieces = append(pieces, "("+strings.Join(orClauses, " OR ")+")")
	}
	where := ""
	if len(pieces) > 0 {
		where = " WHERE " + strings.Join(pieces, " AND ")
	}
	return selectClause + where + order, params
}
