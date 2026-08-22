package tasks

import (
	"database/sql"
	"encoding/base64"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// newDedupeJob adds and claims a dedupe job on a fresh single-connection
// in-memory queue DB that also carries the full media schema, so merges can
// touch tags, embeddings, faces, and the battle log.
func newDedupeJob(t *testing.T, args []string, input string) (*jobqueue.Queue, *jobqueue.Job) {
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
	id, err := q.AddJob("", "dedupe", args, input, nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim job: %v (job=%v)", err, j)
	}
	return q, j
}

func mustExist(t *testing.T, path string, want bool) {
	t.Helper()
	_, err := os.Stat(path)
	if got := err == nil; got != want {
		t.Errorf("exists(%s) = %v, want %v (err=%v)", path, got, want, err)
	}
}

func TestDedupeMergesDuplicatesAndDeletesZeroByte(t *testing.T) {
	dir := t.TempDir()
	same := []byte("identical media bytes, long enough to matter")
	keep := filepath.Join(dir, "a.jpg")      // shortest path → keeper
	dup := filepath.Join(dir, "bbbb.jpg")    // duplicate of keep
	other := filepath.Join(dir, "other.jpg") // unique content, same size
	empty := filepath.Join(dir, "empty.bin")
	subDup := filepath.Join(dir, "sub", "deep.jpg") // out of scope: not recursive
	if err := os.MkdirAll(filepath.Dir(subDup), 0o755); err != nil {
		t.Fatal(err)
	}
	differ := append(append([]byte{}, same[:len(same)-1]...), 'X')
	for p, content := range map[string][]byte{
		keep: same, dup: same, other: differ, empty: {}, subDup: same,
	} {
		if err := os.WriteFile(p, content, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	q, j := newDedupeJob(t, []string{"--target", dir}, "")
	mustSeed := func(query string, args ...any) {
		t.Helper()
		if _, err := q.Db.Exec(query, args...); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range []string{keep, dup, empty} {
		mustSeed(`INSERT INTO media (path) VALUES (?)`, p)
	}
	mustSeed(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	          VALUES (?, 'sunset', 'Scene', 1, 0)`, keep)
	mustSeed(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	          VALUES (?, 'beach', 'Scene', 1, 0)`, dup)
	mustSeed(`INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'm1', 1, x'aa')`, dup)

	var mu sync.Mutex
	if err := dedupeTask(j, q, &mu); err != nil {
		t.Fatalf("dedupe: %v", err)
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Errorf("job state = %v; want Completed", got)
	}

	mustExist(t, keep, true)
	mustExist(t, dup, false)
	mustExist(t, other, true)
	mustExist(t, empty, false)
	mustExist(t, subDup, true) // non-recursive run leaves subdirectories alone

	count := func(query string, args ...any) int {
		t.Helper()
		var n int
		if err := q.Db.QueryRow(query, args...).Scan(&n); err != nil {
			t.Fatal(err)
		}
		return n
	}
	if n := count(`SELECT COUNT(*) FROM media WHERE path = ?`, dup); n != 0 {
		t.Errorf("dup media rows = %d, want 0", n)
	}
	if n := count(`SELECT COUNT(*) FROM media WHERE path = ?`, empty); n != 0 {
		t.Errorf("zero-byte media rows = %d, want 0", n)
	}
	if n := count(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, keep); n != 2 {
		t.Errorf("keeper tags = %d, want 2 (own + merged)", n)
	}
	if n := count(`SELECT COUNT(*) FROM media_embedding WHERE media_path = ?`, keep); n != 1 {
		t.Errorf("keeper embeddings = %d, want 1 (merged from dup)", n)
	}
	if n := count(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, dup); n != 0 {
		t.Errorf("dup tag rows = %d, want 0", n)
	}
}

func TestDedupeQueryInput(t *testing.T) {
	dir := t.TempDir()
	same := []byte("query-addressed duplicate bytes")
	a := filepath.Join(dir, "a.jpg")
	b := filepath.Join(dir, "bbbb.jpg")
	outside := filepath.Join(dir, "outside.jpg") // same bytes but not in the query
	for _, p := range []string{a, b, outside} {
		if err := os.WriteFile(p, same, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	query64 := base64.StdEncoding.EncodeToString([]byte("tag:dup"))
	q, j := newDedupeJob(t, []string{"--query64=" + query64}, "")
	for _, p := range []string{a, b, outside} {
		if _, err := q.Db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range []string{a, b} {
		if _, err := q.Db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		                        VALUES (?, 'dup', 'Scene', 1, 0)`, p); err != nil {
			t.Fatal(err)
		}
	}

	var mu sync.Mutex
	if err := dedupeTask(j, q, &mu); err != nil {
		t.Fatalf("dedupe: %v", err)
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Errorf("job state = %v; want Completed", got)
	}
	mustExist(t, a, true)  // shorter path wins within the query's scope
	mustExist(t, b, false)
	mustExist(t, outside, true) // identical bytes, but the query is the boundary
	var n int
	if err := q.Db.QueryRow(`SELECT COUNT(*) FROM media WHERE path = ?`, b).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 0 {
		t.Errorf("merged-away media rows = %d, want 0", n)
	}
}

func TestDedupePathListInput(t *testing.T) {
	dir := t.TempDir()
	same := []byte("selection duplicate bytes")
	a := filepath.Join(dir, "a.jpg")
	b := filepath.Join(dir, "bbbb.jpg")
	for _, p := range []string{a, b} {
		if err := os.WriteFile(p, same, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// The palette submits the discrete selection as a newline-joined list.
	q, j := newDedupeJob(t, nil, a+"\n"+b)
	for _, p := range []string{a, b} {
		if _, err := q.Db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := q.Db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	                        VALUES (?, 'beach', 'Scene', 1, 0)`, b); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	if err := dedupeTask(j, q, &mu); err != nil {
		t.Fatalf("dedupe: %v", err)
	}
	if got := q.Jobs[j.ID].State; got != jobqueue.StateCompleted {
		t.Errorf("job state = %v; want Completed", got)
	}
	mustExist(t, a, true)
	mustExist(t, b, false)
	var n int
	if err := q.Db.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, a).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("keeper tags = %d, want 1 (merged from deleted duplicate)", n)
	}
}

func TestDedupeRecursivePrefersLibraryKeeperAndDryRun(t *testing.T) {
	dir := t.TempDir()
	same := []byte("shared bytes across the tree")
	noLib := filepath.Join(dir, "x.jpg")
	inLib := filepath.Join(dir, "sub", "much-longer-name.jpg")
	if err := os.MkdirAll(filepath.Dir(inLib), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, p := range []string{noLib, inLib} {
		if err := os.WriteFile(p, same, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	q, j := newDedupeJob(t, []string{"--target", dir, "--recursive", "--dry-run"}, "")
	if _, err := q.Db.Exec(`INSERT INTO media (path) VALUES (?)`, inLib); err != nil {
		t.Fatal(err)
	}
	if _, err := q.Db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
	                        VALUES (?, 'sunset', 'Scene', 1, 0)`, inLib); err != nil {
		t.Fatal(err)
	}

	var mu sync.Mutex
	if err := dedupeTask(j, q, &mu); err != nil {
		t.Fatalf("dry run: %v", err)
	}
	mustExist(t, noLib, true)
	mustExist(t, inLib, true)

	// Real run: the library path wins even though the non-library path is
	// shorter, so the tag survives on a row queries can reach.
	id, err := q.AddJob("", "dedupe", []string{"--target", dir, "--recursive"}, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	j2, err := q.ClaimJob()
	if err != nil || j2 == nil || j2.ID != id {
		t.Fatalf("claim second job: %v (job=%v)", err, j2)
	}
	if err := dedupeTask(j2, q, &mu); err != nil {
		t.Fatalf("dedupe: %v", err)
	}
	mustExist(t, noLib, false)
	mustExist(t, inLib, true)
	var n int
	if err := q.Db.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, inLib).Scan(&n); err != nil {
		t.Fatal(err)
	}
	if n != 1 {
		t.Errorf("keeper tags = %d, want 1", n)
	}
}
