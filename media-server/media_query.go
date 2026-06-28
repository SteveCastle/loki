// media-server/media_query.go
package main

import "strings"

// Predicate mirrors src/renderer/query/types.ts Predicate.
type Predicate struct {
	Type     string   `json:"type"` // tag|category|path|description|hash|similar|visual
	Value    string   `json:"value"`
	Exclude  bool     `json:"exclude"`
	Join     string   `json:"join"`              // "AND" | "OR" | "" (empty falls back to mode)
	Resolved []string `json:"-"` // visual predicates (similar/visual): paths resolved by the handler before BuildMediaQuery
}

// Columns returned for the library list. media.description is intentionally
// NOT selected (large text the list view doesn't use; the detail view fetches
// it on demand) — it's still available as a WHERE filter via clauseFor. The
// renderer sorts results, so queries emit no ORDER BY. Mirror of query-sql.ts.
const baseColumns = "media.path, media.elo, media.height, media.width"

// Tag columns are NULL for category-driven / no-tag queries.
const nullTagCols = "NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at"

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
	case "similar", "visual":
		// Resolved is the path set produced by the handler (similarity search).
		// Empty set: an include matches nothing; an exclude removes nothing.
		if len(p.Resolved) == 0 {
			if p.Exclude {
				return "(1=1)"
			}
			return "(1=0)"
		}
		ph := make([]string, len(p.Resolved))
		for i, rp := range p.Resolved {
			ph[i] = "?"
			*params = append(*params, rp)
		}
		placeholders := strings.Join(ph, ", ")
		if p.Exclude {
			return "(media.path NOT IN (" + placeholders + "))"
		}
		return "(media.path IN (" + placeholders + "))"
	}
	return ""
}

func isIncludeTag(p Predicate) bool { return p.Type == "tag" && !p.Exclude }
func isIncludeCat(p Predicate) bool { return p.Type == "category" && !p.Exclude }

// andJoin joins clauses for the conjunct "rest" of an intersection fast path.
func andJoin(preds []Predicate, params *[]any) string {
	clauses := []string{}
	for _, p := range preds {
		clauses = append(clauses, clauseFor(p, params))
	}
	return strings.Join(clauses, " AND ")
}

// mediaScan is the always-correct path: scan media and combine clauses
// LEFT-TO-RIGHT with the given connectors (connectors[i] joins valid[i+1]),
// parenthesized left-associatively to match how the chips read. Surfaces tag
// columns via a LEFT JOIN on the first include-tag (if any).
func mediaScan(valid []Predicate, connectors []string) (string, []any) {
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
		selectClause = "SELECT " + baseColumns + ", " + nullTagCols + " FROM media"
	}
	expr := clauseFor(valid[0], &params)
	for i := 1; i < len(valid); i++ {
		expr = "(" + expr + " " + connectors[i-1] + " " + clauseFor(valid[i], &params) + ")"
	}
	return selectClause + " WHERE " + expr, params
}

// BuildMediaQuery returns SQL + params. Predicates combine LEFT-TO-RIGHT: the
// first is the base, each subsequent predicate's Join connects it to the
// running result. Homogeneous all-AND / all-OR get index-driven fast paths;
// mixed operators fall back to a correct media scan. Mirror of query-sql.ts.
func BuildMediaQuery(predicates []Predicate, mode string) (string, []any) {
	valid := []Predicate{}
	for _, p := range predicates {
		if p.Value != "" {
			valid = append(valid, p)
		}
	}

	if len(valid) == 0 {
		return "SELECT " + baseColumns + ", " + nullTagCols + " FROM media", []any{}
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

	// Connectors join each predicate after the first to the running result.
	connectors := []string{}
	for i := 1; i < len(valid); i++ {
		connectors = append(connectors, joinOf(valid[i]))
	}
	allOr := len(connectors) > 0
	for _, c := range connectors {
		if c != "OR" {
			allOr = false
		}
	}
	allAnd := true
	for _, c := range connectors {
		if c != "AND" {
			allAnd = false
		}
	}

	allTags := true
	allCats := true
	for _, p := range valid {
		if !isIncludeTag(p) {
			allTags = false
		}
		if !isIncludeCat(p) {
			allCats = false
		}
	}

	// ---- Union (all-OR) via an indexed IN lookup ----
	if allOr && allTags {
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
	if allOr && allCats {
		params := []any{}
		placeholders := []string{}
		for _, p := range valid {
			params = append(params, p.Value)
			placeholders = append(placeholders, "?")
		}
		sql := "SELECT media.path AS path, media.elo AS elo, media.height AS height, media.width AS width, " +
			nullTagCols +
			" FROM (SELECT DISTINCT media_path FROM media_tag_by_category WHERE category_label IN (" +
			strings.Join(placeholders, ", ") + ")) cat" +
			" JOIN media ON media.path = cat.media_path"
		return sql, params
	}

	// ---- Intersection (single predicate or all-AND): drive from an indexed
	//      tag/category with the rest as AND conjuncts ----
	if allAnd {
		driveIdx := -1
		for i := range valid {
			if isIncludeTag(valid[i]) {
				driveIdx = i
				break
			}
		}
		if driveIdx >= 0 {
			params := []any{valid[driveIdx].Value}
			rest := []Predicate{}
			for i := range valid {
				if i != driveIdx {
					rest = append(rest, valid[i])
				}
			}
			restWhere := andJoin(rest, &params)
			extra := ""
			if restWhere != "" {
				extra = " AND " + restWhere
			}
			sql := "SELECT " + drivenColumns +
				" FROM media_tag_by_category mtcw LEFT JOIN media ON media.path = mtcw.media_path" +
				" WHERE mtcw.tag_label = ?" + extra
			return sql, params
		}
		catIdx := -1
		for i := range valid {
			if isIncludeCat(valid[i]) {
				catIdx = i
				break
			}
		}
		if catIdx >= 0 {
			params := []any{valid[catIdx].Value}
			rest := []Predicate{}
			for i := range valid {
				if i != catIdx {
					rest = append(rest, valid[i])
				}
			}
			restWhere := andJoin(rest, &params)
			where := ""
			if restWhere != "" {
				where = " WHERE " + restWhere
			}
			sql := "SELECT media.path AS path, media.elo AS elo, media.height AS height, media.width AS width, " +
				nullTagCols +
				" FROM (SELECT DISTINCT media_path FROM media_tag_by_category WHERE category_label = ?) cat" +
				" JOIN media ON media.path = cat.media_path" + where
			return sql, params
		}
		// No drivable tag/category (paths/hash/excludes only) → AND media scan.
		return mediaScan(valid, connectors)
	}

	// ---- Mixed AND/OR operators → correct left-to-right media scan ----
	return mediaScan(valid, connectors)
}
