package media

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"unicode/utf8"
)

// Moving a path: the bookkeeping half of "I moved this file on disk".
//
// A media path is a foreign key spread across half the schema — tags, the
// embedding sidecar, face rows, scan markers, and the battle log all name it
// as a string. Renaming only `media.path` (or, as the viewer's old "Find
// Media" did, only the tag table) leaves the rest pointing at a path that no
// longer exists: the item keeps its row but loses its tags, its similarity
// vector, and the faces that made it show up under a person.
//
// So a move is all-or-nothing across every table, in one transaction, and it
// refuses to run when the destination is already taken rather than silently
// merging two items.

// pathColumn is one column somewhere in the schema that stores a media path.
type pathColumn struct {
	Table  string
	Column string
	// quoted is the column as it must appear in SQL ("path" is a keyword-ish
	// bare word in the media table's original DDL).
	quoted string
}

// movablePathColumns is the complete set of media-path references. Adding a
// table that stores a media path REQUIRES adding it here — a missing entry is
// a silently orphaned reference, which is exactly the bug this file exists to
// prevent.
var movablePathColumns = []pathColumn{
	{Table: "media", Column: "path", quoted: `"path"`},
	{Table: "media_tag_by_category", Column: "media_path", quoted: "media_path"},
	{Table: "media_embedding", Column: "media_path", quoted: "media_path"},
	{Table: "face", Column: "media_path", quoted: "media_path"},
	{Table: "face_scan", Column: "media_path", quoted: "media_path"},
	{Table: "battle", Column: "winner_path", quoted: "winner_path"},
	{Table: "battle", Column: "loser_path", quoted: "loser_path"},
}

// MoveOptions tunes a MovePath call.
type MoveOptions struct {
	// Prefix treats From and To as directory prefixes: every path under From
	// is rewritten to sit under To, keeping the sub-path. This is the
	// "I moved a whole folder" mode.
	Prefix bool
	// DryRun performs the entire move and rolls it back, so the reported
	// counts are exactly what a real run would change.
	DryRun bool
}

// MovedPath is one from→to rewrite, for previews and confirmation output.
type MovedPath struct {
	From string `json:"from"`
	To   string `json:"to"`
}

// movedPathSampleMax caps how many rewrites a result carries — a folder move
// can touch six figures of rows and the caller only wants a preview.
const movedPathSampleMax = 100

// MoveResult reports what a move touched, per table.
type MoveResult struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Prefix bool   `json:"prefix"`
	DryRun bool   `json:"dryRun"`
	// Items is the number of media rows whose path changed — the count a
	// human means by "how many files did this move".
	Items int64 `json:"items"`
	// Rows is rows updated per "table.column", including the zero entries so
	// the caller can see that a table was considered and had nothing to move.
	Rows map[string]int64 `json:"rows"`
	// Total is the sum of Rows.
	Total int64 `json:"total"`
	// Paths samples the rewrites (see Truncated).
	Paths     []MovedPath `json:"paths,omitempty"`
	Truncated bool        `json:"truncated,omitempty"`
}

// MoveConflictError reports destination paths that already belong to another
// media row. The move is refused entirely: merging two items is a different
// operation with different data loss, and guessing which one the caller wanted
// is not something a path rename should do.
type MoveConflictError struct {
	Conflicts []string
}

func (e *MoveConflictError) Error() string {
	if len(e.Conflicts) == 1 {
		return fmt.Sprintf("destination already exists in the library: %s", e.Conflicts[0])
	}
	return fmt.Sprintf(
		"%d destination paths already exist in the library (e.g. %s)",
		len(e.Conflicts), strings.Join(e.Conflicts[:min(3, len(e.Conflicts))], ", "),
	)
}

// trimTrailingSeparators strips trailing / and \ so "/photos/" and "/photos"
// describe the same folder. A bare root ("/" or "C:\") keeps its separator.
func trimTrailingSeparators(p string) string {
	trimmed := strings.TrimRight(p, `/\`)
	if trimmed == "" || strings.HasSuffix(trimmed, ":") {
		return p
	}
	return trimmed
}

// matchClause builds the WHERE fragment selecting the rows a move affects, plus
// its arguments. Prefix matching is SEGMENT-ALIGNED: "/a/foo" must not drag
// "/a/foobar" along with it, so a prefix only matches when the next character
// is a path separator. substr comparison is used rather than LIKE because
// SQLite's LIKE is case-insensitive for ASCII and would need % and _ escaped.
func matchClause(col, from string, prefix bool) (string, []any) {
	if !prefix {
		return col + " = ?", []any{from}
	}
	n := utf8.RuneCountInString(from) + 1 // +1 for the separator character
	return fmt.Sprintf(
			"(%s = ? OR substr(%s, 1, ?) = ? OR substr(%s, 1, ?) = ?)",
			col, col, col,
		), []any{
			from,
			n, from + "/",
			n, from + `\`,
		}
}

// rewriteExpr builds the SET expression producing the new path, and its args.
// In prefix mode the matched prefix is replaced and the remainder (separator
// included, so the stored separator style survives) is kept.
func rewriteExpr(col, from, to string, prefix bool) (string, []any) {
	if !prefix {
		return "?", []any{to}
	}
	// substr(col, len(from)+1) is the tail starting at the separator.
	return "? || substr(" + col + ", ?)", []any{to, utf8.RuneCountInString(from) + 1}
}

// MovePath rewrites every database reference to from so it points at to, in a
// single transaction. With opts.Prefix it moves a whole subtree. It never
// touches the filesystem — the caller has already moved the file(s); this
// makes the library agree.
//
// Returns a *MoveConflictError when a destination path is already in the
// library, and leaves the database untouched in that case.
func MovePath(ctx context.Context, db *sql.DB, from, to string, opts MoveOptions) (*MoveResult, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	from = trimTrailingSeparators(strings.TrimSpace(from))
	to = trimTrailingSeparators(strings.TrimSpace(to))
	if from == "" || to == "" {
		return nil, fmt.Errorf("both from and to paths are required")
	}
	if from == to {
		return nil, fmt.Errorf("from and to are the same path")
	}
	if opts.Prefix && isUnder(to, from) {
		// Moving /a into /a/b would rewrite the destination again on the same
		// pass and produce nonsense like /a/b/b.
		return nil, fmt.Errorf("destination %q is inside the source %q", to, from)
	}

	result := &MoveResult{
		From:   from,
		To:     to,
		Prefix: opts.Prefix,
		DryRun: opts.DryRun,
		Rows:   map[string]int64{},
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return nil, err
	}
	// Rolled back on every path that doesn't reach an explicit Commit,
	// including a dry run.
	defer tx.Rollback()

	// media_tag_by_category.media_path has a plain FOREIGN KEY to media(path)
	// with no ON UPDATE CASCADE, so there is no order in which the updates
	// below are individually legal: rewriting media.path first orphans the tag
	// rows, and rewriting the tag rows first points them at a path media does
	// not have yet. Deferring enforcement lets the whole set land and checks it
	// once at COMMIT, when every column agrees again. Per-transaction, and
	// reset by SQLite at COMMIT/ROLLBACK; issued on tx so it binds to this
	// transaction's connection (PRAGMA foreign_keys would be a no-op here).
	if _, err := tx.ExecContext(ctx, "PRAGMA defer_foreign_keys = ON"); err != nil {
		return nil, fmt.Errorf("deferring foreign keys: %w", err)
	}

	// The media rows being moved, and what they become — the preview, the
	// item count, and the conflict check all read from this.
	pairs, err := movingPairs(ctx, tx, from, to, opts.Prefix)
	if err != nil {
		return nil, err
	}
	result.Items = int64(len(pairs))
	if len(pairs) > movedPathSampleMax {
		result.Paths = pairs[:movedPathSampleMax]
		result.Truncated = true
	} else {
		result.Paths = pairs
	}

	if conflicts, err := moveConflicts(ctx, tx, pairs); err != nil {
		return nil, err
	} else if len(conflicts) > 0 {
		return nil, &MoveConflictError{Conflicts: conflicts}
	}

	for _, pc := range movablePathColumns {
		exists, err := tableExistsTx(ctx, tx, pc.Table)
		if err != nil {
			return nil, err
		}
		key := pc.Table + "." + pc.Column
		if !exists {
			// A viewer-only library has no face/embedding tables; skipping is
			// correct, and the absent key tells the caller it was skipped.
			continue
		}
		where, whereArgs := matchClause(pc.quoted, from, opts.Prefix)
		set, setArgs := rewriteExpr(pc.quoted, from, to, opts.Prefix)
		args := append(append([]any{}, setArgs...), whereArgs...)
		res, err := tx.ExecContext(ctx,
			fmt.Sprintf("UPDATE %s SET %s = %s WHERE %s", pc.Table, pc.quoted, set, where),
			args...,
		)
		if err != nil {
			return nil, fmt.Errorf("moving %s: %w", key, err)
		}
		n, _ := res.RowsAffected()
		result.Rows[key] = n
		result.Total += n
	}

	if opts.DryRun {
		return result, nil // deferred Rollback discards the work
	}
	if err := tx.Commit(); err != nil {
		return nil, err
	}
	if result.Items > 0 {
		// Moved paths may be in the swipe pool's cached sample.
		InvalidateRandomSampleCache()
	}
	return result, nil
}

// isUnder reports whether path sits inside dir (segment-aligned).
func isUnder(path, dir string) bool {
	return strings.HasPrefix(path, dir+"/") || strings.HasPrefix(path, dir+`\`)
}

// movingPairs lists the media rows the move affects and their new paths.
func movingPairs(ctx context.Context, tx *sql.Tx, from, to string, prefix bool) ([]MovedPath, error) {
	where, whereArgs := matchClause(`"path"`, from, prefix)
	set, setArgs := rewriteExpr(`"path"`, from, to, prefix)
	args := append(append([]any{}, setArgs...), whereArgs...)
	rows, err := tx.QueryContext(ctx,
		fmt.Sprintf(`SELECT "path", %s FROM media WHERE %s ORDER BY "path"`, set, where),
		args...,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var pairs []MovedPath
	for rows.Next() {
		var p MovedPath
		if err := rows.Scan(&p.From, &p.To); err != nil {
			return nil, err
		}
		pairs = append(pairs, p)
	}
	return pairs, rows.Err()
}

// moveConflicts returns destinations that already belong to a media row which
// is NOT itself part of this move.
func moveConflicts(ctx context.Context, tx *sql.Tx, pairs []MovedPath) ([]string, error) {
	if len(pairs) == 0 {
		return nil, nil
	}
	moving := make(map[string]bool, len(pairs))
	for _, p := range pairs {
		moving[p.From] = true
	}
	// Batched to stay under SQLite's parameter limit on a big folder move.
	const batch = 500
	var conflicts []string
	dests := make([]string, 0, len(pairs))
	for _, p := range pairs {
		dests = append(dests, p.To)
	}
	for i := 0; i < len(dests); i += batch {
		chunk := dests[i:min(i+batch, len(dests))]
		placeholders := strings.TrimSuffix(strings.Repeat("?,", len(chunk)), ",")
		args := make([]any, len(chunk))
		for j, d := range chunk {
			args[j] = d
		}
		rows, err := tx.QueryContext(ctx,
			fmt.Sprintf(`SELECT "path" FROM media WHERE "path" IN (%s)`, placeholders),
			args...,
		)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err != nil {
				rows.Close()
				return nil, err
			}
			// A destination that is itself moving out of the way is fine.
			if !moving[p] {
				conflicts = append(conflicts, p)
			}
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	return conflicts, nil
}

// tableExistsTx reports whether a table is present, so the move works against
// a viewer-only library that never had the media-server's face/embedding
// tables created.
func tableExistsTx(ctx context.Context, tx *sql.Tx, name string) (bool, error) {
	var one int
	err := tx.QueryRowContext(ctx,
		`SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?`, name,
	).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}
