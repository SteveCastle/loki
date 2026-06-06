// media-server/media_query.go
package main

import "strings"

// Predicate mirrors src/renderer/query/types.ts Predicate.
type Predicate struct {
	Type    string `json:"type"`    // tag|category|path|description|hash
	Value   string `json:"value"`
	Exclude bool   `json:"exclude"`
}

const baseSelect = "SELECT media.path, media.description, media.elo, media.height, media.width FROM media"

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

// BuildMediaQuery returns the SQL and ordered params for the given predicates.
// mode AND -> AND join, OR -> OR join, EXCLUSIVE -> AND join (single by design).
func BuildMediaQuery(predicates []Predicate, mode string) (string, []any) {
	params := []any{}
	clauses := []string{}
	for _, p := range predicates {
		if p.Value == "" {
			continue
		}
		c := clauseFor(p, &params)
		if c != "" {
			clauses = append(clauses, c)
		}
	}
	if len(clauses) == 0 {
		return baseSelect, params
	}
	joiner := " AND "
	if mode == "OR" {
		joiner = " OR "
	}
	return baseSelect + " WHERE " + strings.Join(clauses, joiner), params
}
