package media

import (
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Duplicate candidates: a REVIEW feature, deliberately separate from the
// exact-bytes dedupe task (which merges and deletes on its own).
//
// The find-duplicates task walks the visual-embedding vectors and records
// every cluster of items whose similarity to a cluster anchor clears a
// threshold. Nothing is merged or deleted by the task: each cluster becomes a
// duplicate_group row the user reviews in the Duplicates panel, where they
// merge the copies they agree are copies (through media.MergeInto, the same
// merge behind the viewer's Merge action), dismiss a group that is only
// look-alikes, or drop a single member that doesn't belong.
//
// Like People, this is a special case of the tag system rather than a tag
// category: groups behave like tags in the library (the "dupe:<id>" query
// predicate filters to a group's members) but they are owned by these tables,
// never by the taxonomy, so no tag or category row is ever created for them.
//
// Invariants the rest of the code relies on:
//   - a media path belongs to at most ONE group (duplicate_member.media_path is
//     unique). A path already in a group — pending or dismissed, member or
//     excluded — is never regrouped, which is what makes re-runs incremental
//     and makes "dismiss" / "not a duplicate" stick;
//   - every member's score is its cosine similarity to the group's anchor
//     (the anchor itself scores 1). Groups are stars around an anchor, not
//     chains, so a group can't grow into a blob of transitively-similar items;
//   - a group with fewer than two member rows is meaningless and is pruned
//     whenever media rows are removed (PruneDuplicateGroups). A group whose
//     anchor path disappears is re-anchored on its best remaining member.

// Duplicate group statuses.
const (
	DuplicateStatusPending   = "pending"
	DuplicateStatusDismissed = "dismissed"
)

// DuplicateGroup is one cluster of duplicate candidates.
type DuplicateGroup struct {
	ID         int64   `json:"id"`
	Model      string  `json:"model"`
	Threshold  float64 `json:"threshold"` // similarity floor the run used, 0..1
	AnchorPath string  `json:"anchorPath"`
	Status     string  `json:"status"`
	CreatedAt  int64   `json:"createdAt"`
	UpdatedAt  int64   `json:"updatedAt"`
	// MemberCount counts every member row; ActiveCount excludes members the
	// user pulled out ("not a duplicate"). MinScore is the lowest active
	// member's similarity to the anchor — how tight the group is.
	MemberCount int               `json:"memberCount"`
	ActiveCount int               `json:"activeCount"`
	MinScore    float64           `json:"minScore"`
	Members     []DuplicateMember `json:"members,omitempty"`
}

// DuplicateMember is one item of a group.
type DuplicateMember struct {
	Path     string  `json:"path"`
	Score    float64 `json:"score"`
	Excluded bool    `json:"excluded"`
}

// DuplicateStats summarises the tables for the panel badge / storage page.
type DuplicateStats struct {
	Pending   int `json:"pending"`
	Dismissed int `json:"dismissed"`
	Members   int `json:"members"`
	// PendingItems counts active members of pending groups — the number of
	// library items still waiting on a decision.
	PendingItems int `json:"pendingItems"`
}

// initDuplicateSchema creates the duplicate-candidate tables. Called from
// InitializeSchema on every DB open, so it must stay idempotent.
func initDuplicateSchema(db *sql.DB) error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS duplicate_group (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			model       TEXT NOT NULL,
			threshold   REAL NOT NULL,
			anchor_path TEXT NOT NULL,
			status      TEXT NOT NULL DEFAULT 'pending',
			created_at  INTEGER,
			updated_at  INTEGER
		)`,
		`CREATE TABLE IF NOT EXISTS duplicate_member (
			group_id   INTEGER NOT NULL,
			media_path TEXT NOT NULL,
			score      REAL NOT NULL,
			excluded   INTEGER NOT NULL DEFAULT 0,
			added_at   INTEGER,
			PRIMARY KEY (group_id, media_path)
		)`,
		`CREATE UNIQUE INDEX IF NOT EXISTS idx_duplicate_member_path ON duplicate_member(media_path)`,
		`CREATE INDEX IF NOT EXISTS idx_duplicate_group_status ON duplicate_group(status)`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return fmt.Errorf("duplicate schema: %w", err)
		}
	}
	return nil
}

// CreateDuplicateGroup records a new group anchored on anchor with the given
// members (the anchor must be one of them; scores are similarities to it).
// Members that already belong to another group are skipped — the unique path
// index is the source of truth — and when fewer than two members remain the
// group is not created and 0 is returned.
func CreateDuplicateGroup(db *sql.DB, model string, threshold float64, anchor string, members []DuplicateMember) (int64, error) {
	if db == nil {
		return 0, fmt.Errorf("database connection not available")
	}
	anchor = strings.TrimSpace(anchor)
	if anchor == "" {
		return 0, fmt.Errorf("duplicate group anchor required")
	}
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()

	now := time.Now().Unix()
	res, err := tx.Exec(
		`INSERT INTO duplicate_group (model, threshold, anchor_path, status, created_at, updated_at)
		 VALUES (?, ?, ?, ?, ?, ?)`,
		model, threshold, anchor, DuplicateStatusPending, now, now)
	if err != nil {
		return 0, fmt.Errorf("create duplicate group: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return 0, err
	}
	added, err := insertDuplicateMembers(tx, id, members, now)
	if err != nil {
		return 0, err
	}
	if added < 2 {
		return 0, nil // rollback via defer: not a group
	}
	// The anchor must have landed (it could have belonged elsewhere already).
	var n int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM duplicate_member WHERE group_id = ? AND media_path = ?`, id, anchor).Scan(&n); err != nil {
		return 0, err
	}
	if n == 0 {
		return 0, nil
	}
	return id, tx.Commit()
}

// AddDuplicateMembers appends members to an existing group (scores are their
// similarity to the group's anchor). Paths already in any group are skipped.
// Returns how many rows were actually added.
func AddDuplicateMembers(db *sql.DB, groupID int64, members []DuplicateMember) (int, error) {
	if db == nil {
		return 0, fmt.Errorf("database connection not available")
	}
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	now := time.Now().Unix()
	added, err := insertDuplicateMembers(tx, groupID, members, now)
	if err != nil {
		return 0, err
	}
	if added > 0 {
		if _, err := tx.Exec(`UPDATE duplicate_group SET updated_at = ? WHERE id = ?`, now, groupID); err != nil {
			return 0, err
		}
	}
	return added, tx.Commit()
}

func insertDuplicateMembers(tx *sql.Tx, groupID int64, members []DuplicateMember, now int64) (int, error) {
	added := 0
	for _, m := range members {
		p := strings.TrimSpace(m.Path)
		if p == "" {
			continue
		}
		// INSERT OR IGNORE: the unique path index rejects a member that
		// already sits in another group, which is the "one group per item"
		// rule rather than an error.
		res, err := tx.Exec(
			`INSERT OR IGNORE INTO duplicate_member (group_id, media_path, score, excluded, added_at)
			 VALUES (?, ?, ?, 0, ?)`,
			groupID, p, m.Score, now)
		if err != nil {
			return added, fmt.Errorf("add duplicate member %s: %w", p, err)
		}
		if n, _ := res.RowsAffected(); n > 0 {
			added++
		}
	}
	return added, nil
}

// DuplicateAssignments returns every path that already belongs to a group
// (any status, excluded members included) mapped to its group id, plus each
// group's anchor path. The find-duplicates task uses it to skip grouped
// items and to let new items join an existing group through its anchor.
func DuplicateAssignments(db *sql.DB) (byPath map[string]int64, anchors map[int64]string, err error) {
	if db == nil {
		return nil, nil, fmt.Errorf("database connection not available")
	}
	byPath = map[string]int64{}
	anchors = map[int64]string{}
	rows, err := db.Query(`SELECT id, anchor_path FROM duplicate_group`)
	if err != nil {
		return nil, nil, err
	}
	for rows.Next() {
		var id int64
		var p string
		if err := rows.Scan(&id, &p); err != nil {
			rows.Close()
			return nil, nil, err
		}
		anchors[id] = p
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	rows, err = db.Query(`SELECT media_path, group_id FROM duplicate_member`)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var p string
		var id int64
		if err := rows.Scan(&p, &id); err != nil {
			return nil, nil, err
		}
		byPath[p] = id
	}
	return byPath, anchors, rows.Err()
}

const duplicateGroupColumns = `
	g.id, g.model, g.threshold, g.anchor_path, g.status,
	COALESCE(g.created_at, 0), COALESCE(g.updated_at, 0),
	(SELECT COUNT(*) FROM duplicate_member m WHERE m.group_id = g.id),
	(SELECT COUNT(*) FROM duplicate_member m WHERE m.group_id = g.id AND m.excluded = 0),
	COALESCE((SELECT MIN(m.score) FROM duplicate_member m WHERE m.group_id = g.id AND m.excluded = 0), 0)`

func scanDuplicateGroup(sc interface{ Scan(...any) error }) (DuplicateGroup, error) {
	var g DuplicateGroup
	err := sc.Scan(&g.ID, &g.Model, &g.Threshold, &g.AnchorPath, &g.Status,
		&g.CreatedAt, &g.UpdatedAt, &g.MemberCount, &g.ActiveCount, &g.MinScore)
	return g, err
}

// Group orderings for ListDuplicateGroups.
const (
	DuplicateSortNewest  = "newest"  // most recently found first (default)
	DuplicateSortMembers = "members" // biggest clusters (active members) first
)

// ListDuplicateGroups pages through groups of one status ("" = every
// status), ordered by sort (DuplicateSortNewest when empty), members
// included. total is the unpaged count.
func ListDuplicateGroups(db *sql.DB, status, sort string, limit, offset int) ([]DuplicateGroup, int, error) {
	if db == nil {
		return nil, 0, fmt.Errorf("database connection not available")
	}
	where := ""
	var args []any
	if status != "" {
		where = "WHERE g.status = ?"
		args = append(args, status)
	}
	var total int
	if err := db.QueryRow(`SELECT COUNT(*) FROM duplicate_group g `+where, args...).Scan(&total); err != nil {
		return nil, 0, err
	}
	if limit <= 0 {
		limit = 50
	}
	order := "g.id DESC"
	if sort == DuplicateSortMembers {
		order = "(SELECT COUNT(*) FROM duplicate_member m WHERE m.group_id = g.id AND m.excluded = 0) DESC, g.id DESC"
	}
	q := `SELECT ` + duplicateGroupColumns + ` FROM duplicate_group g ` + where +
		` ORDER BY ` + order + ` LIMIT ? OFFSET ?`
	rows, err := db.Query(q, append(args, limit, offset)...)
	if err != nil {
		return nil, 0, err
	}
	var out []DuplicateGroup
	for rows.Next() {
		g, err := scanDuplicateGroup(rows)
		if err != nil {
			rows.Close()
			return nil, 0, err
		}
		out = append(out, g)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, 0, err
	}
	for i := range out {
		members, err := DuplicateGroupMembers(db, out[i].ID)
		if err != nil {
			return nil, 0, err
		}
		out[i].Members = members
	}
	return out, total, nil
}

// GetDuplicateGroup returns one group with its members.
func GetDuplicateGroup(db *sql.DB, id int64) (DuplicateGroup, bool, error) {
	if db == nil {
		return DuplicateGroup{}, false, fmt.Errorf("database connection not available")
	}
	g, err := scanDuplicateGroup(db.QueryRow(
		`SELECT `+duplicateGroupColumns+` FROM duplicate_group g WHERE g.id = ?`, id))
	if err == sql.ErrNoRows {
		return DuplicateGroup{}, false, nil
	}
	if err != nil {
		return DuplicateGroup{}, false, err
	}
	g.Members, err = DuplicateGroupMembers(db, id)
	if err != nil {
		return DuplicateGroup{}, false, err
	}
	return g, true, nil
}

// DuplicateGroupMembers lists a group's members, anchor first, then by
// descending similarity (path ascending on ties), excluded members last.
func DuplicateGroupMembers(db *sql.DB, groupID int64) ([]DuplicateMember, error) {
	rows, err := db.Query(
		`SELECT m.media_path, m.score, m.excluded
		 FROM duplicate_member m
		 JOIN duplicate_group g ON g.id = m.group_id
		 WHERE m.group_id = ?
		 ORDER BY m.excluded ASC, (m.media_path = g.anchor_path) DESC, m.score DESC, m.media_path ASC`,
		groupID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []DuplicateMember{}
	for rows.Next() {
		var m DuplicateMember
		var excluded int
		if err := rows.Scan(&m.Path, &m.Score, &excluded); err != nil {
			return nil, err
		}
		m.Excluded = excluded != 0
		out = append(out, m)
	}
	return out, rows.Err()
}

// DuplicateGroupForPath returns the id of the group a path belongs to (0 when
// none) — the per-item "this has duplicate candidates" lookup.
func DuplicateGroupForPath(db *sql.DB, path string) (int64, error) {
	var id int64
	err := db.QueryRow(`SELECT group_id FROM duplicate_member WHERE media_path = ?`, path).Scan(&id)
	if err == sql.ErrNoRows {
		return 0, nil
	}
	return id, err
}

// SetDuplicateGroupStatus marks a group pending or dismissed.
func SetDuplicateGroupStatus(db *sql.DB, id int64, status string) error {
	if status != DuplicateStatusPending && status != DuplicateStatusDismissed {
		return fmt.Errorf("status must be %q or %q", DuplicateStatusPending, DuplicateStatusDismissed)
	}
	res, err := db.Exec(`UPDATE duplicate_group SET status = ?, updated_at = ? WHERE id = ?`,
		status, time.Now().Unix(), id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return fmt.Errorf("no duplicate group with id %d", id)
	}
	return nil
}

// SetDuplicateMemberExcluded pulls one member out of (or back into) a group's
// active set. The row stays, so the item is never regrouped with the same
// partners on the next run. When fewer than two active members remain the
// group has nothing left to merge and is dismissed automatically.
func SetDuplicateMemberExcluded(db *sql.DB, id int64, path string, excluded bool) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	flag := 0
	if excluded {
		flag = 1
	}
	res, err := tx.Exec(`UPDATE duplicate_member SET excluded = ? WHERE group_id = ? AND media_path = ?`,
		flag, id, path)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return fmt.Errorf("no member %q in duplicate group %d", path, id)
	}
	now := time.Now().Unix()
	var active int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM duplicate_member WHERE group_id = ? AND excluded = 0`, id).Scan(&active); err != nil {
		return err
	}
	if active < 2 {
		if _, err := tx.Exec(`UPDATE duplicate_group SET status = ?, updated_at = ? WHERE id = ?`,
			DuplicateStatusDismissed, now, id); err != nil {
			return err
		}
	} else if _, err := tx.Exec(`UPDATE duplicate_group SET updated_at = ? WHERE id = ?`, now, id); err != nil {
		return err
	}
	// Re-anchor when the anchor itself was excluded so scores keep meaning
	// "similarity to an active member".
	if excluded {
		if err := reanchorDuplicateGroups(tx); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// DeleteDuplicateGroup forgets a group entirely; its items become eligible
// for grouping again on the next run.
func DeleteDuplicateGroup(db *sql.DB, id int64) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	res, err := tx.Exec(`DELETE FROM duplicate_group WHERE id = ?`, id)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n == 0 {
		return fmt.Errorf("no duplicate group with id %d", id)
	}
	if _, err := tx.Exec(`DELETE FROM duplicate_member WHERE group_id = ?`, id); err != nil {
		return err
	}
	return tx.Commit()
}

// DeleteDuplicateGroups forgets every group of one status ("" = all).
// Returns the number of groups removed.
func DeleteDuplicateGroups(db *sql.DB, status string) (int, error) {
	if db == nil {
		return 0, fmt.Errorf("database connection not available")
	}
	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	defer tx.Rollback()
	var res sql.Result
	if status == "" {
		if _, err := tx.Exec(`DELETE FROM duplicate_member`); err != nil {
			return 0, err
		}
		res, err = tx.Exec(`DELETE FROM duplicate_group`)
	} else {
		if _, err := tx.Exec(
			`DELETE FROM duplicate_member WHERE group_id IN (SELECT id FROM duplicate_group WHERE status = ?)`,
			status); err != nil {
			return 0, err
		}
		res, err = tx.Exec(`DELETE FROM duplicate_group WHERE status = ?`, status)
	}
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return int(n), tx.Commit()
}

// GetDuplicateStats counts groups and members.
func GetDuplicateStats(db *sql.DB) (DuplicateStats, error) {
	var s DuplicateStats
	if db == nil {
		return s, fmt.Errorf("database connection not available")
	}
	err := db.QueryRow(`SELECT
		COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0),
		COALESCE(SUM(CASE WHEN status = 'dismissed' THEN 1 ELSE 0 END), 0)
		FROM duplicate_group`).Scan(&s.Pending, &s.Dismissed)
	if err != nil {
		return s, err
	}
	if err := db.QueryRow(`SELECT COUNT(*) FROM duplicate_member`).Scan(&s.Members); err != nil {
		return s, err
	}
	err = db.QueryRow(`SELECT COUNT(*) FROM duplicate_member m
		JOIN duplicate_group g ON g.id = m.group_id
		WHERE g.status = 'pending' AND m.excluded = 0`).Scan(&s.PendingItems)
	return s, err
}

// PruneDuplicateGroups repairs the tables after member rows disappeared
// (media removed or merged): groups left with fewer than two member rows
// are deleted, and a group whose anchor is gone is re-anchored on its best
// remaining active member. Safe to call on a DB without the tables.
func PruneDuplicateGroups(db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if err := pruneDuplicateGroupsTx(tx); err != nil {
		return err
	}
	return tx.Commit()
}

type execer interface {
	Exec(query string, args ...any) (sql.Result, error)
}

func pruneDuplicateGroupsTx(tx execer) error {
	if _, err := tx.Exec(`DELETE FROM duplicate_group WHERE
		(SELECT COUNT(*) FROM duplicate_member m WHERE m.group_id = duplicate_group.id) < 2`); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM duplicate_member WHERE
		group_id NOT IN (SELECT id FROM duplicate_group)`); err != nil {
		return err
	}
	return reanchorDuplicateGroups(tx)
}

func reanchorDuplicateGroups(tx execer) error {
	_, err := tx.Exec(`UPDATE duplicate_group SET anchor_path = COALESCE(
			(SELECT m.media_path FROM duplicate_member m
			 WHERE m.group_id = duplicate_group.id AND m.excluded = 0
			 ORDER BY m.score DESC, m.media_path ASC LIMIT 1),
			(SELECT m.media_path FROM duplicate_member m
			 WHERE m.group_id = duplicate_group.id
			 ORDER BY m.score DESC, m.media_path ASC LIMIT 1),
			anchor_path)
		WHERE NOT EXISTS (SELECT 1 FROM duplicate_member m
			WHERE m.group_id = duplicate_group.id AND m.media_path = duplicate_group.anchor_path AND m.excluded = 0)`)
	return err
}

// DuplicatePredicateSQL compiles a "dupe:" query value into a WHERE clause
// over pathExpr (the media path column of the outer query). Values:
//
//	<group id>  — the ACTIVE members of that group (excluded members are out)
//	pending     — active members of every pending group
//	any         — every grouped path, dismissed groups and excluded members too
//
// ok is false for anything else, so callers can match nothing rather than
// everything. Shared by both server query engines so they can't drift.
func DuplicatePredicateSQL(value, pathExpr string) (clause string, args []any, ok bool) {
	v := strings.ToLower(strings.TrimSpace(value))
	switch v {
	case "pending":
		return "EXISTS (SELECT 1 FROM duplicate_member dm JOIN duplicate_group dg ON dg.id = dm.group_id" +
			" WHERE dm.media_path = " + pathExpr + " AND dm.excluded = 0 AND dg.status = 'pending')", nil, true
	case "any", "all":
		return "EXISTS (SELECT 1 FROM duplicate_member dm WHERE dm.media_path = " + pathExpr + ")", nil, true
	}
	id, err := strconv.ParseInt(v, 10, 64)
	if err != nil || id <= 0 {
		return "", nil, false
	}
	return "EXISTS (SELECT 1 FROM duplicate_member dm WHERE dm.media_path = " + pathExpr +
		" AND dm.group_id = ? AND dm.excluded = 0)", []any{id}, true
}

// PendingDuplicateGroupIDs lists every pending group id, oldest first — the
// worklist of the merge-duplicates task (captured once up front so groups
// that fail to merge can't be revisited forever).
func PendingDuplicateGroupIDs(db *sql.DB) ([]int64, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	rows, err := db.Query(`SELECT id FROM duplicate_group WHERE status = ? ORDER BY id ASC`, DuplicateStatusPending)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []int64
	for rows.Next() {
		var id int64
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// DuplicateMemberFacts is what the keeper rule looks at for one member:
// stored dimensions (the "dimensions" op) and file size, 0 when unknown.
type DuplicateMemberFacts struct {
	Width, Height int64
	Size          int64
}

// LoadDuplicateMemberFacts reads width/height/size for paths from the media
// table, in batches. Paths without a row are simply absent from the map.
func LoadDuplicateMemberFacts(db *sql.DB, paths []string) (map[string]DuplicateMemberFacts, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	out := make(map[string]DuplicateMemberFacts, len(paths))
	const batch = 500
	for lo := 0; lo < len(paths); lo += batch {
		hi := min(lo+batch, len(paths))
		chunk := paths[lo:hi]
		ph := strings.TrimSuffix(strings.Repeat("?,", len(chunk)), ",")
		args := make([]any, len(chunk))
		for i, p := range chunk {
			args[i] = p
		}
		rows, err := db.Query(`SELECT "path", COALESCE(width, 0), COALESCE(height, 0), COALESCE("size", 0) FROM media WHERE "path" IN (`+ph+`)`, args...)
		if err != nil {
			return nil, err
		}
		for rows.Next() {
			var p string
			var f DuplicateMemberFacts
			if err := rows.Scan(&p, &f.Width, &f.Height, &f.Size); err != nil {
				rows.Close()
				return nil, err
			}
			out[p] = f
		}
		rows.Close()
		if err := rows.Err(); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// PreferredDuplicateKeeper is THE rule for which copy a merge keeps by
// default, shared by the merge endpoint, the bulk merge task, and the
// panel's default highlight: the highest resolution (width × height) wins;
// among equals, or when no member has dimensions, the largest file; then
// the anchor; then the shortest path (a stable tie-break). Only active
// members are candidates. Returns the anchor when the group has no active
// member at all.
func PreferredDuplicateKeeper(g DuplicateGroup, facts map[string]DuplicateMemberFacts) string {
	best := ""
	var bestPixels, bestSize int64
	better := func(p string, f DuplicateMemberFacts) bool {
		pixels := f.Width * f.Height
		if best == "" {
			return true
		}
		if pixels != bestPixels {
			return pixels > bestPixels
		}
		if f.Size != bestSize {
			return f.Size > bestSize
		}
		if (p == g.AnchorPath) != (best == g.AnchorPath) {
			return p == g.AnchorPath
		}
		if len(p) != len(best) {
			return len(p) < len(best)
		}
		return p < best
	}
	for _, m := range g.Members {
		if m.Excluded {
			continue
		}
		f := facts[m.Path]
		if better(m.Path, f) {
			best, bestPixels, bestSize = m.Path, f.Width*f.Height, f.Size
		}
	}
	if best == "" {
		return g.AnchorPath
	}
	return best
}

// DuplicateGroupKeeper loads the facts for g's members and applies
// PreferredDuplicateKeeper.
func DuplicateGroupKeeper(db *sql.DB, g DuplicateGroup) (string, map[string]DuplicateMemberFacts, error) {
	paths := make([]string, 0, len(g.Members))
	for _, m := range g.Members {
		paths = append(paths, m.Path)
	}
	facts, err := LoadDuplicateMemberFacts(db, paths)
	if err != nil {
		return g.AnchorPath, nil, err
	}
	return PreferredDuplicateKeeper(g, facts), facts, nil
}
