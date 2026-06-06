// media-server/media_query.go
package main

import "strings"

// Predicate mirrors src/renderer/query/types.ts Predicate.
type Predicate struct {
	Type    string `json:"type"` // tag|category|path|description|hash
	Value   string `json:"value"`
	Exclude bool   `json:"exclude"`
	Join    string `json:"join"` // "AND" | "OR" | "" (empty falls back to mode)
}

// Columns returned for the library list. media.description is intentionally
// NOT selected (large text the list view doesn't use; the detail view fetches
// it on demand) — it's still available as a WHERE filter via clauseFor. The
// renderer sorts results, so queries emit no ORDER BY. Mirror of query-sql.ts.
const baseColumns = "media.path, media.elo, media.height, media.width"

// Aliased column list for the index-driven paths, matching the TS builder so
// the handler scans the same 8 columns by position regardless of path:
// path, elo, height, width, weight, tag_label, time_stamp, created_at.
const drivenColumns = "mtcw.media_path AS path, media.elo AS elo, media.height AS height, " +
	"media.width AS width, mtcw.weight AS weight, mtcw.tag_label AS tag_label, " +
	"mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at"

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

// facetedWhere combines predicates: AND-bucket clauses are required, the
// OR-bucket is grouped as "(a OR b ...)". Returns "" when there are none.
func facetedWhere(preds []Predicate, joinOf func(Predicate) string, params *[]any) string {
	andClauses := []string{}
	orClauses := []string{}
	for _, p := range preds {
		if joinOf(p) != "OR" {
			andClauses = append(andClauses, clauseFor(p, params))
		}
	}
	for _, p := range preds {
		if joinOf(p) == "OR" {
			orClauses = append(orClauses, clauseFor(p, params))
		}
	}
	pieces := append([]string{}, andClauses...)
	if len(orClauses) > 0 {
		pieces = append(pieces, "("+strings.Join(orClauses, " OR ")+")")
	}
	return strings.Join(pieces, " AND ")
}

// BuildMediaQuery returns SQL + params for the unified query. It drives tag
// queries from the indexed media_tag_by_category lookup (avoiding a full
// `media` scan) wherever semantics allow, mirroring src/main/query-sql.ts.
func BuildMediaQuery(predicates []Predicate, mode string) (string, []any) {
	valid := []Predicate{}
	for _, p := range predicates {
		if p.Value != "" {
			valid = append(valid, p)
		}
	}

	if len(valid) == 0 {
		return "SELECT " + baseColumns +
			", NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at FROM media", []any{}
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
	isIncludeTag := func(p Predicate) bool { return p.Type == "tag" && !p.Exclude }

	// Fast path: drive from a REQUIRED include-tag (AND-bucket, or the sole
	// predicate) via the indexed tag lookup instead of scanning `media`.
	driveIdx := -1
	for i := range valid {
		if isIncludeTag(valid[i]) && joinOf(valid[i]) != "OR" {
			driveIdx = i
			break
		}
	}
	if driveIdx == -1 && len(valid) == 1 && isIncludeTag(valid[0]) {
		driveIdx = 0
	}

	if driveIdx >= 0 {
		params := []any{valid[driveIdx].Value} // driving join param first
		rest := []Predicate{}
		for i := range valid {
			if i != driveIdx {
				rest = append(rest, valid[i])
			}
		}
		restWhere := facetedWhere(rest, joinOf, &params)
		extra := ""
		if restWhere != "" {
			extra = " AND " + restWhere
		}
		sql := "SELECT " + drivenColumns +
			" FROM media_tag_by_category mtcw LEFT JOIN media ON media.path = mtcw.media_path" +
			" WHERE mtcw.tag_label = ?" + extra
		return sql, params
	}

	// OR-set of include-tags: every predicate is an include-tag (an OR bucket)
	// — "media with ANY of these tags" = tag_label IN (...). Indexed lookup.
	allIncludeTags := true
	for _, p := range valid {
		if !isIncludeTag(p) {
			allIncludeTags = false
			break
		}
	}
	if allIncludeTags {
		params := []any{}
		placeholders := []string{}
		for _, p := range valid {
			params = append(params, p.Value)
			placeholders = append(placeholders, "?")
		}
		sql := "SELECT " + drivenColumns +
			" FROM media_tag_by_category mtcw LEFT JOIN media ON media.path = mtcw.media_path" +
			" WHERE mtcw.tag_label IN (" + strings.Join(placeholders, ", ") + ")"
		return sql, params
	}

	// Fallback: exclude-only, non-tag predicates, or an OR bucket mixing tags
	// with non-tags. Scan media; LEFT JOIN the first include-tag (if any) only
	// to surface weight/tag/timestamp columns.
	primaryTag := ""
	for _, p := range valid {
		if isIncludeTag(p) {
			primaryTag = p.Value
			break
		}
	}
	params := []any{}
	var selectClause string
	if primaryTag != "" {
		selectClause = "SELECT " + baseColumns +
			", mtcw.weight AS weight, mtcw.tag_label AS tag_label, mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at" +
			" FROM media LEFT JOIN media_tag_by_category mtcw ON mtcw.media_path = media.path AND mtcw.tag_label = ?"
		params = append(params, primaryTag)
	} else {
		selectClause = "SELECT " + baseColumns +
			", NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at FROM media"
	}
	restWhere := facetedWhere(valid, joinOf, &params)
	where := ""
	if restWhere != "" {
		where = " WHERE " + restWhere
	}
	return selectClause + where, params
}
