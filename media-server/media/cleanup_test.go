package media

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// fullSchemaDB opens a file-backed database with the complete server schema
// (every sidecar table the cleanup touches).
func fullSchemaDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "lib.sqlite")+"?_pragma=busy_timeout=5000&_pragma=foreign_keys=ON")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	return db
}

func mustExec(t *testing.T, db *sql.DB, q string, args ...any) {
	t.Helper()
	if _, err := db.Exec(q, args...); err != nil {
		t.Fatalf("%s: %v", q, err)
	}
}

func count(t *testing.T, db *sql.DB, q string, args ...any) int64 {
	t.Helper()
	var n int64
	if err := db.QueryRow(q, args...).Scan(&n); err != nil {
		t.Fatalf("%s: %v", q, err)
	}
	return n
}

func touch(t *testing.T, p string) string {
	t.Helper()
	if err := os.WriteFile(p, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	return p
}

// seedItem inserts a media row with one of everything that can reference it.
func seedItem(t *testing.T, db *sql.DB, p string) {
	t.Helper()
	mustExec(t, db, `INSERT INTO media ("path") VALUES (?)`, p)
	mustExec(t, db, `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES (?, 'tag', 'cat', 0, 0)`, p)
	mustExec(t, db, `INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0001')`, p)
	mustExec(t, db, `INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector) VALUES (?, 'm', 0, 0, 1, 1, 1, x'00')`, p)
	mustExec(t, db, `INSERT INTO face_scan (media_path, model, face_count) VALUES (?, 'm', 1)`, p)
	// Paired with itself so the row never names a path outside the library.
	mustExec(t, db, `INSERT INTO battle (winner_path, loser_path) VALUES (?, ?)`, p, p)
}

func TestCleanupLibraryFourPhases(t *testing.T) {
	db := fullSchemaDB(t)
	dir := t.TempDir()
	present := touch(t, filepath.Join(dir, "present.jpg"))
	missing := filepath.Join(dir, "missing.jpg")
	seedItem(t, db, present)
	seedItem(t, db, missing)
	// Curation assertions keyed by the missing item's face id.
	faceID := count(t, db, `SELECT id FROM face WHERE media_path = ?`, missing)
	presentFace := count(t, db, `SELECT id FROM face WHERE media_path = ?`, present)
	mustExec(t, db, `INSERT INTO person (name, cover_face_id) VALUES ('P', ?)`, faceID)
	mustExec(t, db, `INSERT INTO face_veto (face_id, person_id) VALUES (?, 1)`, faceID)
	mustExec(t, db, `INSERT INTO face_cannot_link (face_a, face_b) VALUES (?, ?)`, min64(faceID, presentFace), max64(faceID, presentFace))
	mustExec(t, db, `INSERT INTO face_group_ban (source_name) VALUES ('g')`)
	mustExec(t, db, `INSERT INTO face_group_ban_member (ban_id, face_id) VALUES (1, ?)`, faceID)
	// Dangling rows: a path that was never (or is no longer) in media.
	ghost := filepath.Join(dir, "ghost.mp4")
	mustExec(t, db, `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES (?, 'tag', 'cat', 0, 4.5)`, ghost)
	mustExec(t, db, `INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0001')`, ghost)
	mustExec(t, db, `INSERT INTO battle (winner_path, loser_path) VALUES (?, ?)`, present, ghost)

	var phases []CleanupPhase
	var logs []string
	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{
		Progress: func(p CleanupProgress) {
			if len(phases) == 0 || phases[len(phases)-1] != p.Phase {
				phases = append(phases, p.Phase)
			}
		},
		Log: func(s string) { logs = append(logs, s) },
	})
	if err != nil {
		t.Fatalf("CleanupLibrary: %v (logs %v)", err, logs)
	}
	wantPhases := []CleanupPhase{CleanupPhaseScan, CleanupPhaseVerify, CleanupPhaseRemove, CleanupPhaseOrphans}
	if len(phases) != len(wantPhases) {
		t.Errorf("phases = %v, want %v", phases, wantPhases)
	} else {
		for i := range phases {
			if phases[i] != wantPhases[i] {
				t.Errorf("phases = %v, want %v", phases, wantPhases)
				break
			}
		}
	}

	if res.MediaScanned != 2 || res.MissingFound != 1 || res.MediaRemoved != 1 {
		t.Errorf("scanned/missing/removed = %d/%d/%d, want 2/1/1", res.MediaScanned, res.MissingFound, res.MediaRemoved)
	}
	// Phase 3 (missing) + phase 4 (ghost) row counts.
	if res.TagsRemoved != 2 || res.EmbeddingsRemoved != 2 || res.FacesRemoved != 1 || res.FaceScansRemoved != 1 {
		t.Errorf("tags/embeddings/faces/scans = %d/%d/%d/%d, want 2/2/1/1", res.TagsRemoved, res.EmbeddingsRemoved, res.FacesRemoved, res.FaceScansRemoved)
	}
	if res.FaceAssertionsRemoved != 3 {
		t.Errorf("face assertions removed = %d, want 3 (veto, cannot-link, ban member)", res.FaceAssertionsRemoved)
	}
	// missing's own battle row + the present-vs-ghost row swept as dangling.
	if res.BattlesRemoved != 2 {
		t.Errorf("battles removed = %d, want 2", res.BattlesRemoved)
	}
	if res.OrphanPaths != 1 || res.OrphanPathsByTable["media_tag_by_category.media_path"] != 1 {
		t.Errorf("orphans = %d %v, want 1 swept via media_tag_by_category (then gone from the rest)", res.OrphanPaths, res.OrphanPathsByTable)
	}

	// The present item is untouched in every table.
	for _, q := range []string{
		`SELECT COUNT(*) FROM media WHERE "path" = ?`,
		`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
		`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
		`SELECT COUNT(*) FROM face WHERE media_path = ?`,
		`SELECT COUNT(*) FROM face_scan WHERE media_path = ?`,
		`SELECT COUNT(*) FROM battle WHERE winner_path = ? AND loser_path = winner_path`,
	} {
		if n := count(t, db, q, present); n != 1 {
			t.Errorf("%s (present) = %d, want 1", q, n)
		}
	}
	// Nothing references the missing item or the ghost anywhere.
	for _, p := range []string{missing, ghost} {
		for _, q := range []string{
			`SELECT COUNT(*) FROM media WHERE "path" = ?`,
			`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`,
			`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`,
			`SELECT COUNT(*) FROM face WHERE media_path = ?`,
			`SELECT COUNT(*) FROM face_scan WHERE media_path = ?`,
			`SELECT COUNT(*) FROM battle WHERE winner_path = ? OR loser_path = ?`,
		} {
			args := []any{p}
			if q[len(q)-len("loser_path = ?"):] == "loser_path = ?" {
				args = append(args, p)
			}
			if n := count(t, db, q, args...); n != 0 {
				t.Errorf("%s (%s) = %d, want 0", q, filepath.Base(p), n)
			}
		}
	}
	if n := count(t, db, `SELECT COUNT(*) FROM face_veto`) + count(t, db, `SELECT COUNT(*) FROM face_cannot_link`) + count(t, db, `SELECT COUNT(*) FROM face_group_ban_member`); n != 0 {
		t.Errorf("%d face assertions remain, want 0", n)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM person WHERE cover_face_id IS NULL`); n != 1 {
		t.Errorf("person cover not cleared")
	}
}

func TestCleanupLibraryDryRunDeletesNothing(t *testing.T) {
	db := fullSchemaDB(t)
	dir := t.TempDir()
	seedItem(t, db, touch(t, filepath.Join(dir, "present.jpg")))
	seedItem(t, db, filepath.Join(dir, "missing.jpg"))
	mustExec(t, db, `INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, x'0001')`, filepath.Join(dir, "ghost.jpg"))
	mustExec(t, db, `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES (?, 't', 'c', 0, 0)`, filepath.Join(dir, "ghost.jpg"))

	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{DryRun: true})
	if err != nil {
		t.Fatal(err)
	}
	if !res.DryRun || res.MediaRemoved != 1 || res.TagsRemoved != 0 {
		t.Errorf("dry run result = %+v", res)
	}
	// Dry-run orphan counts are per table: the ghost dangles in two.
	if res.OrphanPaths != 2 || res.OrphanPathsByTable["media_embedding.media_path"] != 1 || res.OrphanPathsByTable["media_tag_by_category.media_path"] != 1 {
		t.Errorf("orphan counts = %d %v", res.OrphanPaths, res.OrphanPathsByTable)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media`); n != 2 {
		t.Errorf("media rows = %d after dry run, want 2", n)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media_tag_by_category`); n != 3 {
		t.Errorf("tag rows = %d after dry run, want 3", n)
	}
}

func TestCleanupLibraryRecoversTransientlyMissing(t *testing.T) {
	db := fullSchemaDB(t)
	dir := t.TempDir()
	flaky := filepath.Join(dir, "flaky.jpg")
	gone := filepath.Join(dir, "gone.jpg")
	seedItem(t, db, flaky)
	seedItem(t, db, gone)

	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{
		// The share comes back between the scan and the verification.
		beforeVerify: func() { touch(t, flaky) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.MissingFound != 2 || res.Recovered != 1 || res.MediaRemoved != 1 {
		t.Errorf("missing/recovered/removed = %d/%d/%d, want 2/1/1", res.MissingFound, res.Recovered, res.MediaRemoved)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media WHERE "path" = ?`, flaky); n != 1 {
		t.Errorf("transiently-missing item was removed")
	}
}

func TestCleanupLibraryGuardHoldsBackMassMissing(t *testing.T) {
	db := fullSchemaDB(t)
	// Two configured roots: one healthy with a couple of deletions, one
	// where everything is missing (an empty bind mount, a renamed folder).
	healthy := t.TempDir()
	broken := t.TempDir()
	SetCleanupRootResolver(func(p string) (string, bool) {
		for _, r := range []string{healthy, broken} {
			if rel, err := filepath.Rel(r, p); err == nil && rel != ".." && !filepath.IsAbs(rel) && rel[:1] != "." {
				return r, true
			}
		}
		return "", false
	})
	defer SetCleanupRootResolver(nil)

	for i := 0; i < 30; i++ {
		p := filepath.Join(healthy, "h"+string(rune('a'+i%26))+string(rune('0'+i/26))+".jpg")
		if i < 27 {
			touch(t, p)
		}
		seedItem(t, db, p)
	}
	for i := 0; i < 25; i++ {
		seedItem(t, db, filepath.Join(broken, "b"+string(rune('a'+i))+".jpg"))
	}

	var logs []string
	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{Log: func(s string) { logs = append(logs, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if res.MediaRemoved != 3 {
		t.Errorf("removed %d, want 3 (the healthy root's deletions only); logs %v", res.MediaRemoved, logs)
	}
	if res.SkippedByGuard != 25 || len(res.GuardedRoots) != 1 || res.GuardedRoots[0].Root != broken {
		t.Errorf("guard: skipped %d, roots %+v", res.SkippedByGuard, res.GuardedRoots)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media WHERE "path" LIKE ?`, broken+"%"); n != 25 {
		t.Errorf("broken root has %d rows left, want 25", n)
	}

	// Raising the cap to 100 purges it deliberately.
	res, err = CleanupLibrary(context.Background(), db, CleanupOptions{MaxMissingPercent: 100})
	if err != nil {
		t.Fatal(err)
	}
	if res.MediaRemoved != 25 || res.SkippedByGuard != 0 {
		t.Errorf("with cap 100: removed %d, skipped %d", res.MediaRemoved, res.SkippedByGuard)
	}
}

func TestCleanupLibraryScope(t *testing.T) {
	db := fullSchemaDB(t)
	root := t.TempDir()
	inside := filepath.Join(root, "in")
	outside := filepath.Join(root, "out")
	for _, d := range []string{inside, outside} {
		if err := os.MkdirAll(d, 0755); err != nil {
			t.Fatal(err)
		}
	}
	seedItem(t, db, filepath.Join(inside, "missing.jpg"))
	seedItem(t, db, filepath.Join(outside, "missing.jpg"))
	mustExec(t, db, `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES (?, 't', 'c', 0, 0)`, filepath.ToSlash(filepath.Join(inside, "ghost.jpg")))
	mustExec(t, db, `INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp) VALUES (?, 't', 'c', 0, 0)`, filepath.Join(outside, "ghost.jpg"))

	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{Scope: inside})
	if err != nil {
		t.Fatal(err)
	}
	if res.MediaScanned != 1 || res.MediaRemoved != 1 || res.OrphanPaths != 1 {
		t.Errorf("scanned/removed/orphans = %d/%d/%d, want 1/1/1 (slash-spelled ghost included)", res.MediaScanned, res.MediaRemoved, res.OrphanPaths)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media WHERE "path" = ?`, filepath.Join(outside, "missing.jpg")); n != 1 {
		t.Errorf("item outside the scope was removed")
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, filepath.Join(outside, "ghost.jpg")); n != 1 {
		t.Errorf("dangling row outside the scope was swept")
	}
}

func TestCleanupLibraryInterruptBetweenChunks(t *testing.T) {
	db := fullSchemaDB(t)
	dir := t.TempDir()
	for i := 0; i < 6; i++ {
		seedItem(t, db, filepath.Join(dir, "m"+string(rune('a'+i))+".jpg"))
	}
	pause := errors.New("paused")
	calls := 0
	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{
		ScanBatch: 2,
		Interrupt: func() error {
			calls++
			if calls == 3 {
				return pause
			}
			return nil
		},
	})
	if !errors.Is(err, pause) {
		t.Fatalf("err = %v, want the interrupt error", err)
	}
	if res.MediaRemoved != 0 {
		t.Errorf("removed %d before the scan finished", res.MediaRemoved)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media`); n != 6 {
		t.Errorf("rows = %d, want 6 (interrupted during the scan)", n)
	}
}

func TestCleanupLibraryLeavesUnconfiguredBucketAlone(t *testing.T) {
	db := fullSchemaDB(t)
	SetRemoteExistsChecker(func(paths []string) map[string]bool {
		out := map[string]bool{}
		for _, p := range paths {
			out[p] = false // the bucket answers "not here" for everything
		}
		return out
	})
	defer SetRemoteExistsChecker(nil)
	SetRemoteRootChecker(func(root string) bool { return root == "s3://configured/" })
	defer SetRemoteRootChecker(nil)

	seedItem(t, db, "s3://configured/a.jpg")
	seedItem(t, db, "s3://forgotten/b.jpg")
	res, err := CleanupLibrary(context.Background(), db, CleanupOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if res.MediaRemoved != 1 || res.SkippedUnavailable != 1 || len(res.UnavailableRoots) != 1 || res.UnavailableRoots[0] != "s3://forgotten/" {
		t.Errorf("result = removed %d, skipped %d, roots %v", res.MediaRemoved, res.SkippedUnavailable, res.UnavailableRoots)
	}
	if n := count(t, db, `SELECT COUNT(*) FROM media WHERE "path" = 's3://forgotten/b.jpg'`); n != 1 {
		t.Errorf("item in the unconfigured bucket was removed")
	}
}
