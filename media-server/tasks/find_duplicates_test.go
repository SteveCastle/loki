package tasks

import (
	"database/sql"
	"encoding/base64"
	"math"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// newFindDuplicatesQueue builds an in-memory queue + DB seeded with the given
// path→vector embeddings under the active model (vectors are normalized by
// the seeding helper).
func newFindDuplicatesQueue(t *testing.T, vectors map[string][]float32) *jobqueue.Queue {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	q := jobqueue.NewQueueWithDB(db)
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	model := ActiveEmbedModel().ID
	for p, v := range vectors {
		if _, err := db.Exec(`INSERT OR IGNORE INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
		if err := media.UpsertEmbedding(db, p, model, unit(v), 0); err != nil {
			t.Fatal(err)
		}
	}
	return q
}

func unit(v []float32) []float32 {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	n = math.Sqrt(n)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / n)
	}
	return out
}

func runFindDuplicates(t *testing.T, q *jobqueue.Queue, args []string, input string) *jobqueue.Job {
	t.Helper()
	id, err := q.AddJob("", "find-duplicates", args, input, nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	var mu sync.Mutex
	if err := findDuplicatesTask(j, q, &mu); err != nil {
		t.Fatalf("find-duplicates: %v\n%s", err, strings.Join(q.Jobs[j.ID].Stdout, "\n"))
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Fatalf("job state = %v; want Completed\n%s", got, strings.Join(q.Jobs[j.ID].Stdout, "\n"))
	}
	return j
}

// groupsByAnchor reads the groups back as anchor → sorted member list.
func groupsByAnchor(t *testing.T, db *sql.DB) map[string][]string {
	t.Helper()
	groups, _, err := media.ListDuplicateGroups(db, "", "", 100, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string][]string{}
	for _, g := range groups {
		var members []string
		for _, m := range g.Members {
			members = append(members, filepath.Base(m.Path))
		}
		out[filepath.Base(g.AnchorPath)] = members
	}
	return out
}

// Two exact copies, one near copy (cosine ≈ 0.9998, i.e. a re-encode), one
// look-alike at ≈ 0.98, and unrelated items. In 8-dim space with a pinned
// seed the hashing is deterministic.
//
// Paths are absolute (under dir) because the path-list input is absolutized
// like every other bulk task's, and must round-trip to the stored spelling.
func duplicateFixture(dir string) map[string][]float32 {
	base := []float32{1, 0.5, 0.25, 0.125, 0, 0, 0, 0}
	near := []float32{1, 0.5, 0.25, 0.125, 0.02, 0, 0, 0}   // ≈ 0.9998
	alike := []float32{1, 0.5, 0.25, 0.125, 0.2, 0.1, 0, 0} // ≈ 0.98
	return map[string][]float32{
		filepath.Join(dir, "a.jpg"):     base,
		filepath.Join(dir, "b.jpg"):     base,
		filepath.Join(dir, "near.jpg"):  near,
		filepath.Join(dir, "alike.jpg"): alike,
		filepath.Join(dir, "x.jpg"):     {0, 0, 0, 0, 0, 1, 0.3, 0},
		filepath.Join(dir, "y.jpg"):     {0, 1, 0, 0, 0, 0, 0, 0.5},
		filepath.Join(dir, "z.jpg"):     {0.1, 0, 0.9, 0, 0, 0, 1, 0},
	}
}

func pinLSHSeed(t *testing.T) {
	t.Helper()
	prev := lshSeed
	lshSeed = func() int64 { return 42 }
	t.Cleanup(func() { lshSeed = prev })
}

func TestFindDuplicatesGroupsAtDefaultThresholdAndIsIncremental(t *testing.T) {
	pinLSHSeed(t)
	dir := t.TempDir()
	q := newFindDuplicatesQueue(t, duplicateFixture(dir))

	// Default: 100% → exact copies and the re-encode (0.9998 ≥ 0.9995) group
	// together; the 0.98 look-alike and unrelated items stay out.
	j := runFindDuplicates(t, q, nil, "")
	groups := groupsByAnchor(t, q.Db)
	if len(groups) != 1 {
		t.Fatalf("groups = %v; want exactly one", groups)
	}
	members, ok := groups["a.jpg"]
	if !ok {
		t.Fatalf("groups = %v; want anchored on a.jpg (first by path)", groups)
	}
	if strings.Join(members, ",") != "a.jpg,b.jpg,near.jpg" {
		t.Errorf("members = %v; want a, b, near (anchor first, then by score)", members)
	}
	log := strings.Join(q.Jobs[j.ID].Stdout, "\n")
	if !strings.Contains(log, "Review mode") || !strings.Contains(log, "1 new group(s)") {
		t.Errorf("log should announce review mode and one new group:\n%s", log)
	}
	// Nothing was merged or deleted.
	var mediaRows int
	_ = q.Db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&mediaRows)
	if mediaRows != 7 {
		t.Errorf("media rows = %d; want 7 (review only, nothing removed)", mediaRows)
	}

	// Second run: everything grouped is left alone, no new groups.
	j = runFindDuplicates(t, q, nil, "")
	if got := groupsByAnchor(t, q.Db); len(got) != 1 || len(got["a.jpg"]) != 3 {
		t.Errorf("second run changed groups: %v", got)
	}
	if log := strings.Join(q.Jobs[j.ID].Stdout, "\n"); !strings.Contains(log, "0 new group(s)") {
		t.Errorf("second run should report no new groups:\n%s", log)
	}

	// A new copy of a.jpg embedded later JOINS the existing group through its
	// anchor instead of seeding a second group.
	late := filepath.Join(dir, "late.jpg")
	if err := media.UpsertEmbedding(q.Db, late, ActiveEmbedModel().ID, unit(duplicateFixture(dir)[filepath.Join(dir, "a.jpg")]), 0); err != nil {
		t.Fatal(err)
	}
	if _, err := q.Db.Exec(`INSERT INTO media (path) VALUES (?)`, late); err != nil {
		t.Fatal(err)
	}
	j = runFindDuplicates(t, q, nil, "")
	got := groupsByAnchor(t, q.Db)
	if len(got) != 1 || strings.Join(got["a.jpg"], ",") != "a.jpg,b.jpg,late.jpg,near.jpg" {
		t.Errorf("late copy should join the existing group: %v", got)
	}
	if log := strings.Join(q.Jobs[j.ID].Stdout, "\n"); !strings.Contains(log, "1 item(s) joined existing groups") {
		t.Errorf("third run should report one join:\n%s", log)
	}

	// A dismissed group keeps its members out of future groups.
	groupsList, _, _ := media.ListDuplicateGroups(q.Db, "", "", 10, 0)
	if err := media.SetDuplicateGroupStatus(q.Db, groupsList[0].ID, media.DuplicateStatusDismissed); err != nil {
		t.Fatal(err)
	}
	runFindDuplicates(t, q, []string{"--threshold=97"}, "")
	got = groupsByAnchor(t, q.Db)
	if len(got) != 1 {
		// alike.jpg (0.98 to a) is now within threshold but a's group is
		// dismissed, and a is still that group's anchor → it joins there.
		t.Errorf("lowering the threshold must not seed a new group around dismissed members: %v", got)
	}
	if !strings.Contains(strings.Join(got["a.jpg"], ","), "alike.jpg") {
		t.Errorf("alike.jpg should have joined a.jpg's group at 97%%: %v", got)
	}
}

func TestFindDuplicatesScopeAndReset(t *testing.T) {
	pinLSHSeed(t)
	dir := t.TempDir()
	q := newFindDuplicatesQueue(t, duplicateFixture(dir))
	p := func(name string) string { return filepath.Join(dir, name) }

	// Path-list scope: only the listed items are compared (a/near), b is
	// not in scope and stays free.
	runFindDuplicates(t, q, nil, strings.Join([]string{p("a.jpg"), p("near.jpg"), p("x.jpg")}, "\n"))
	got := groupsByAnchor(t, q.Db)
	if len(got) != 1 || strings.Join(got["a.jpg"], ",") != "a.jpg,near.jpg" {
		t.Fatalf("scoped run groups = %v; want a+near only", got)
	}

	// Query scope (tag:dup = b, y): b.jpg joins a's group through the anchor
	// even though a is outside the run's scope; y stays alone.
	for _, name := range []string{"b.jpg", "y.jpg"} {
		if _, err := q.Db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
			VALUES (?, 'dup', 'Scene', 1, 0)`, p(name)); err != nil {
			t.Fatal(err)
		}
	}
	runFindDuplicates(t, q, []string{"--query64=" + base64.StdEncoding.EncodeToString([]byte("tag:dup"))}, "")
	got = groupsByAnchor(t, q.Db)
	if strings.Join(got["a.jpg"], ",") != "a.jpg,b.jpg,near.jpg" {
		t.Errorf("query-scoped run should let b.jpg join: %v", got)
	}

	// --reset forgets pending groups and rebuilds; the rebuilt group has the
	// same shape but a new id.
	before, _, _ := media.ListDuplicateGroups(q.Db, "", "", 10, 0)
	runFindDuplicates(t, q, []string{"--reset"}, "")
	after, _, _ := media.ListDuplicateGroups(q.Db, "", "", 10, 0)
	if len(after) != 1 || after[0].ID == before[0].ID {
		t.Errorf("--reset should rebuild the group under a new id: before=%v after=%v", before, after)
	}

	// Explicit, unknown directory → error state, nothing changed.
	id, _ := q.AddJob("", "find-duplicates", []string{"--target", filepath.Join(t.TempDir(), "missing")}, "", nil)
	j, _ := q.ClaimJob()
	var mu sync.Mutex
	if err := findDuplicatesTask(j, q, &mu); err == nil {
		t.Error("missing target directory should fail the job")
	}
	if q.Jobs[id].State != jobqueue.StateError {
		t.Errorf("job state = %v; want Errored", q.Jobs[id].State)
	}
}

func TestFindDuplicatesNoEmbeddingsCompletesGracefully(t *testing.T) {
	q := newFindDuplicatesQueue(t, nil)
	j := runFindDuplicates(t, q, nil, "")
	if log := strings.Join(q.Jobs[j.ID].Stdout, "\n"); !strings.Contains(log, "No embeddings found") {
		t.Errorf("empty library should say so:\n%s", log)
	}
}

func TestLSHParamsSizing(t *testing.T) {
	// Exact-duplicate threshold: few tables, high recall.
	bits, tables, recall := lshParams(50_000, duplicateThresholdScore(100))
	if bits < 12 || bits > 16 || tables < 1 || tables > 6 || recall < 0.98 {
		t.Errorf("100%% @ 50k: bits=%d tables=%d recall=%.3f", bits, tables, recall)
	}
	// Looser threshold needs more tables but stays capped.
	_, tables, recall = lshParams(2_000_000, duplicateThresholdScore(90))
	if tables > findDuplicatesMaxTables || recall < 0.9 {
		t.Errorf("90%% @ 2M: tables=%d recall=%.3f", tables, recall)
	}
	// Tiny inputs never go below the floor.
	bits, _, _ = lshParams(3, duplicateThresholdScore(100))
	if bits != 6 {
		t.Errorf("tiny n bits = %d; want 6", bits)
	}
	if s := duplicateThresholdScore(100); s < 0.9994 || s > 0.9996 {
		t.Errorf("100%% → %v; want ≈0.9995", s)
	}
	if s := duplicateThresholdScore(1000); s != duplicateThresholdScore(100) {
		t.Errorf("percent should clamp at 100: %v", s)
	}
}
