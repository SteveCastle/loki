package media

import (
	"context"
	"database/sql"
	"errors"
	"testing"
)

// seedMovable inserts one media item with a reference in every path-bearing
// table, so a test can assert that a move carried ALL of them.
func seedMovable(t *testing.T, db *sql.DB, path string) {
	t.Helper()
	stmts := []struct {
		sql  string
		args []any
	}{
		{`INSERT INTO media (path) VALUES (?)`, []any{path}},
		{`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		  VALUES (?, 'sunset', 'Subject', 1, 0)`, []any{path}},
		{`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0000')`, []any{path}},
		{`INSERT INTO face_scan (media_path, model, face_count) VALUES (?, 'sface', 1)`, []any{path}},
		{`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
		  VALUES (?, 'sface', 0.1, 0.1, 0.2, 0.2, 0.9, x'0000')`, []any{path}},
		{`INSERT INTO battle (winner_path, loser_path, outcome) VALUES (?, 'other.jpg', 1)`, []any{path}},
		{`INSERT INTO battle (winner_path, loser_path, outcome) VALUES ('other.jpg', ?, 1)`, []any{path}},
	}
	for _, s := range stmts {
		if _, err := db.Exec(s.sql, s.args...); err != nil {
			t.Fatalf("seed %q: %v", s.sql, err)
		}
	}
}

// countAt returns how many rows across every path-bearing table name the path.
func countAt(t *testing.T, db *sql.DB, path string) map[string]int {
	t.Helper()
	out := map[string]int{}
	queries := map[string]string{
		"media":                 `SELECT COUNT(*) FROM media WHERE "path" = ?`,
		"media_tag_by_category": `SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
		"media_embedding":       `SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
		"face":                  `SELECT COUNT(*) FROM face WHERE media_path = ?`,
		"face_scan":             `SELECT COUNT(*) FROM face_scan WHERE media_path = ?`,
		"battle":                `SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`,
	}
	for name, q := range queries {
		var n int
		args := []any{path}
		if name == "battle" {
			args = append(args, path)
		}
		if err := db.QueryRow(q, args...).Scan(&n); err != nil {
			t.Fatalf("count %s: %v", name, err)
		}
		out[name] = n
	}
	return out
}

func TestMovePathCarriesEveryReference(t *testing.T) {
	db := newPeopleDB(t)
	const from = "/photos/old.jpg"
	const to = "/archive/new.jpg"
	seedMovable(t, db, from)

	res, err := MovePath(context.Background(), db, from, to, MoveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Items != 1 {
		t.Errorf("items = %d, want 1", res.Items)
	}
	// Two battle columns are updated separately, one row each.
	wantRows := map[string]int64{
		"media.path":                       1,
		"media_tag_by_category.media_path": 1,
		"media_embedding.media_path":       1,
		"face.media_path":                  1,
		"face_scan.media_path":             1,
		"battle.winner_path":               1,
		"battle.loser_path":                1,
	}
	for key, want := range wantRows {
		if res.Rows[key] != want {
			t.Errorf("rows[%q] = %d, want %d (all: %v)", key, res.Rows[key], want, res.Rows)
		}
	}
	if res.Total != 7 {
		t.Errorf("total = %d, want 7", res.Total)
	}
	if len(res.Paths) != 1 || res.Paths[0].From != from || res.Paths[0].To != to {
		t.Errorf("paths = %+v", res.Paths)
	}

	for table, n := range countAt(t, db, from) {
		if n != 0 {
			t.Errorf("%s still references the old path (%d rows)", table, n)
		}
	}
	for table, n := range countAt(t, db, to) {
		want := 1
		if table == "battle" {
			want = 2
		}
		if n != want {
			t.Errorf("%s at the new path = %d, want %d", table, n, want)
		}
	}
}

// legacyFKDB builds a library with the OLD media_tag_by_category shape, which
// carries a plain FOREIGN KEY to media(path) with no ON UPDATE CASCADE. The
// current InitializeSchema omits that constraint, so a move against a fresh DB
// cannot exercise it — only libraries created by earlier versions can, and
// those are exactly the ones a move has to keep working on.
func legacyFKDB(t *testing.T) *sql.DB {
	t.Helper()
	db := newPeopleDB(t) // already opened with foreign_keys=ON
	stmts := []string{
		`DROP TABLE media_tag_by_category`,
		`CREATE TABLE media_tag_by_category (
			media_path TEXT,
			tag_label TEXT,
			category_label TEXT,
			weight REAL,
			time_stamp REAL,
			job_id INTEGER,
			created_at INTEGER,
			PRIMARY KEY (media_path, tag_label, category_label, time_stamp),
			FOREIGN KEY (media_path) REFERENCES media (path),
			FOREIGN KEY (tag_label) REFERENCES tag (label),
			FOREIGN KEY (category_label) REFERENCES category (label)
		)`,
		`INSERT INTO category (label) VALUES ('Subject')`,
		`INSERT INTO tag (label, category_label) VALUES ('sunset', 'Subject')`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			t.Fatalf("legacy schema %q: %v", s, err)
		}
	}
	return db
}

// A move rewrites media.path and media_tag_by_category.media_path in the same
// transaction. Under the legacy foreign key there is no ordering in which both
// statements are individually valid, so the move must defer enforcement to
// COMMIT. Without that, this fails with "FOREIGN KEY constraint failed (787)".
func TestMovePathUnderLegacyForeignKey(t *testing.T) {
	db := legacyFKDB(t)
	const from = "/photos/old.jpg"
	const to = "/archive/new.jpg"
	seedMovable(t, db, from)

	res, err := MovePath(context.Background(), db, from, to, MoveOptions{})
	if err != nil {
		t.Fatalf("move under legacy foreign key: %v", err)
	}
	if res.Rows["media.path"] != 1 || res.Rows["media_tag_by_category.media_path"] != 1 {
		t.Errorf("rows = %v, want media.path and the tag row both moved", res.Rows)
	}
	if n := countAt(t, db, from); n["media"] != 0 || n["media_tag_by_category"] != 0 {
		t.Errorf("old path still referenced: %v", n)
	}
	if n := countAt(t, db, to); n["media"] != 1 || n["media_tag_by_category"] != 1 {
		t.Errorf("new path = %v, want the media row and its tag", n)
	}
}

// The deferred check must still REJECT a genuinely dangling reference at
// COMMIT — deferring enforcement is not the same as disabling it.
func TestMovePathLegacyForeignKeyStillEnforcedAtCommit(t *testing.T) {
	db := legacyFKDB(t)
	if _, err := db.Exec(
		`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		 VALUES ('/nowhere/ghost.jpg', 'sunset', 'Subject', 1, 0)`,
	); err == nil {
		t.Fatal("insert naming a nonexistent media path was allowed")
	}
}

func TestMovePathPrefixMovesAFolderSegmentAligned(t *testing.T) {
	db := newPeopleDB(t)
	seedMovable(t, db, `/photos/2023/a.jpg`)
	seedMovable(t, db, `/photos/2023/sub/b.jpg`)
	// The classic prefix bug: /photos/2023 must not drag /photos/2023extra.
	seedMovable(t, db, `/photos/2023extra/c.jpg`)
	// A sibling folder is untouched.
	seedMovable(t, db, `/photos/2024/d.jpg`)

	res, err := MovePath(context.Background(), db, "/photos/2023", "/archive/2023", MoveOptions{Prefix: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Items != 2 {
		t.Fatalf("items = %d, want 2 (%+v)", res.Items, res.Paths)
	}
	for _, want := range []string{`/archive/2023/a.jpg`, `/archive/2023/sub/b.jpg`} {
		if countAt(t, db, want)["media"] != 1 {
			t.Errorf("%s was not moved", want)
		}
	}
	for _, untouched := range []string{`/photos/2023extra/c.jpg`, `/photos/2024/d.jpg`} {
		if countAt(t, db, untouched)["media"] != 1 {
			t.Errorf("%s should not have moved", untouched)
		}
	}
	// Sidecar rows followed, not just the media row.
	if n := countAt(t, db, `/archive/2023/sub/b.jpg`)["face"]; n != 1 {
		t.Errorf("faces did not follow the folder move: %d", n)
	}
}

func TestMovePathPrefixHandlesWindowsSeparators(t *testing.T) {
	db := newPeopleDB(t)
	seedMovable(t, db, `C:\media\shoot\a.jpg`)
	seedMovable(t, db, `C:\media\shoot-b\keep.jpg`) // not under "shoot"

	// A trailing separator on either side describes the same folder.
	res, err := MovePath(context.Background(), db, `C:\media\shoot\`, `D:\archive\shoot`, MoveOptions{Prefix: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Items != 1 {
		t.Fatalf("items = %d, want 1 (%+v)", res.Items, res.Paths)
	}
	if countAt(t, db, `D:\archive\shoot\a.jpg`)["media"] != 1 {
		t.Error("the backslash sub-path was not preserved")
	}
	if countAt(t, db, `C:\media\shoot-b\keep.jpg`)["media"] != 1 {
		t.Error("a sibling folder with a shared name prefix was moved")
	}
}

func TestMovePathDryRunChangesNothing(t *testing.T) {
	db := newPeopleDB(t)
	const from = "/photos/old.jpg"
	seedMovable(t, db, from)

	res, err := MovePath(context.Background(), db, from, "/archive/new.jpg", MoveOptions{DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	// The counts are the real ones — the work happened and was rolled back.
	if res.Items != 1 || res.Total != 7 {
		t.Errorf("dry run reported items=%d total=%d, want 1 and 7", res.Items, res.Total)
	}
	if !res.DryRun {
		t.Error("result does not report itself as a dry run")
	}
	if countAt(t, db, from)["media"] != 1 {
		t.Error("dry run moved the media row")
	}
	if countAt(t, db, "/archive/new.jpg")["media"] != 0 {
		t.Error("dry run wrote the destination")
	}
}

func TestMovePathRefusesAnOccupiedDestination(t *testing.T) {
	db := newPeopleDB(t)
	seedMovable(t, db, "/photos/a.jpg")
	seedMovable(t, db, "/photos/b.jpg")

	_, err := MovePath(context.Background(), db, "/photos/a.jpg", "/photos/b.jpg", MoveOptions{})
	var conflict *MoveConflictError
	if !errors.As(err, &conflict) {
		t.Fatalf("err = %v, want a MoveConflictError", err)
	}
	if len(conflict.Conflicts) != 1 || conflict.Conflicts[0] != "/photos/b.jpg" {
		t.Errorf("conflicts = %v", conflict.Conflicts)
	}
	// Refused means refused: nothing moved, including the sidecar tables.
	if countAt(t, db, "/photos/a.jpg")["media_tag_by_category"] != 1 {
		t.Error("a refused move still rewrote rows")
	}
}

func TestMovePathValidatesArguments(t *testing.T) {
	db := newPeopleDB(t)
	cases := []struct {
		name     string
		from, to string
		opts     MoveOptions
	}{
		{"empty from", "", "/b.jpg", MoveOptions{}},
		{"empty to", "/a.jpg", "  ", MoveOptions{}},
		{"same path", "/a.jpg", "/a.jpg", MoveOptions{}},
		{"same path after trimming separators", "/a/", "/a", MoveOptions{}},
		{"destination inside the source", "/a", "/a/b", MoveOptions{Prefix: true}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if _, err := MovePath(context.Background(), db, c.from, c.to, c.opts); err == nil {
				t.Error("want an error")
			}
		})
	}
}

func TestMovePathOnUnknownPathIsANoOp(t *testing.T) {
	db := newPeopleDB(t)
	seedMovable(t, db, "/photos/a.jpg")

	res, err := MovePath(context.Background(), db, "/photos/ghost.jpg", "/archive/ghost.jpg", MoveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Items != 0 || res.Total != 0 {
		t.Errorf("items=%d total=%d, want 0 and 0", res.Items, res.Total)
	}
	if countAt(t, db, "/photos/a.jpg")["media"] != 1 {
		t.Error("an unrelated item was touched")
	}
}

func TestMovePathSkipsTablesAViewerLibraryLacks(t *testing.T) {
	// A library the media-server has never opened has no face/embedding
	// tables; the move must do what it can rather than failing outright.
	db := newPeopleDB(t)
	seedMovable(t, db, "/photos/a.jpg")
	for _, table := range []string{"media_embedding", "face", "face_scan"} {
		if _, err := db.Exec(`DROP TABLE ` + table); err != nil {
			t.Fatal(err)
		}
	}

	res, err := MovePath(context.Background(), db, "/photos/a.jpg", "/archive/a.jpg", MoveOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.Rows["media.path"] != 1 || res.Rows["media_tag_by_category.media_path"] != 1 {
		t.Errorf("rows = %v", res.Rows)
	}
	// Absent tables are reported as absent (no key), not as "0 rows moved".
	if _, ok := res.Rows["face.media_path"]; ok {
		t.Error("a missing table was reported as considered")
	}
	// countAt would query the tables this test dropped.
	var moved int
	if err := db.QueryRow(`SELECT COUNT(*) FROM media WHERE "path" = ?`, "/archive/a.jpg").Scan(&moved); err != nil {
		t.Fatal(err)
	}
	if moved != 1 {
		t.Error("the media row did not move")
	}
}
