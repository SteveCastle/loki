package tasks

import (
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// seedMergeGroups builds a queue whose DB holds three pending groups with
// real files: a tight 100% pair, a looser 97% trio (one member excluded),
// and one dismissed group that must never be touched.
func seedMergeGroups(t *testing.T) (*jobqueue.Queue, string, map[string]int64) {
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
	dir := t.TempDir()
	mk := func(name string) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(`INSERT INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
		return p
	}
	ids := map[string]int64{}
	tight, err := media.CreateDuplicateGroup(db, "m", 0.9995, mk("t1.jpg"), []media.DuplicateMember{
		{Path: filepath.Join(dir, "t1.jpg"), Score: 1}, {Path: mk("t2.jpg"), Score: 0.9999},
	})
	if err != nil || tight == 0 {
		t.Fatal(err)
	}
	ids["tight"] = tight
	// t2 is the higher-resolution copy: the keeper rule must prefer it over
	// the anchor t1.
	if _, err := db.Exec(`UPDATE media SET width = 640, height = 480 WHERE path = ?`, filepath.Join(dir, "t1.jpg")); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE media SET width = 1920, height = 1080 WHERE path = ?`, filepath.Join(dir, "t2.jpg")); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
		VALUES (?, 'beach', 'Scene', 1, 0)`, filepath.Join(dir, "t2.jpg")); err != nil {
		t.Fatal(err)
	}
	loose, err := media.CreateDuplicateGroup(db, "m", 0.97, mk("l1.jpg"), []media.DuplicateMember{
		{Path: filepath.Join(dir, "l1.jpg"), Score: 1}, {Path: mk("l2.jpg"), Score: 0.975}, {Path: mk("l3.jpg"), Score: 0.972},
	})
	if err != nil || loose == 0 {
		t.Fatal(err)
	}
	ids["loose"] = loose
	if err := media.SetDuplicateMemberExcluded(db, loose, filepath.Join(dir, "l3.jpg"), true); err != nil {
		t.Fatal(err)
	}
	dismissed, err := media.CreateDuplicateGroup(db, "m", 0.9995, mk("d1.jpg"), []media.DuplicateMember{
		{Path: filepath.Join(dir, "d1.jpg"), Score: 1}, {Path: mk("d2.jpg"), Score: 0.9999},
	})
	if err != nil || dismissed == 0 {
		t.Fatal(err)
	}
	if err := media.SetDuplicateGroupStatus(db, dismissed, media.DuplicateStatusDismissed); err != nil {
		t.Fatal(err)
	}
	ids["dismissed"] = dismissed
	return q, dir, ids
}

func runMergeDuplicates(t *testing.T, q *jobqueue.Queue, args []string) string {
	t.Helper()
	id, err := q.AddJob("", "merge-duplicates", args, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != id {
		t.Fatalf("claim: %v", err)
	}
	var mu sync.Mutex
	if err := mergeDuplicatesTask(j, q, &mu); err != nil {
		t.Fatalf("merge-duplicates: %v\n%s", err, strings.Join(q.Jobs[id].Stdout, "\n"))
	}
	if q.Jobs[id].State != jobqueue.StateCompleted {
		t.Fatalf("state = %v", q.Jobs[id].State)
	}
	return strings.Join(q.Jobs[id].Stdout, "\n")
}

func exists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

func TestMergeDuplicatesDryRunTouchesNothing(t *testing.T) {
	q, dir, _ := seedMergeGroups(t)
	log := runMergeDuplicates(t, q, []string{"--dry-run"})
	if !strings.Contains(log, "2 group(s) would merge, deleting 2 file(s)") {
		t.Errorf("dry run summary wrong:\n%s", log)
	}
	for _, n := range []string{"t1.jpg", "t2.jpg", "l1.jpg", "l2.jpg", "l3.jpg", "d1.jpg", "d2.jpg"} {
		if !exists(filepath.Join(dir, n)) {
			t.Errorf("%s deleted by a dry run", n)
		}
	}
	s, _ := media.GetDuplicateStats(q.Db)
	if s.Pending != 2 || s.Dismissed != 1 {
		t.Errorf("stats changed by dry run: %+v", s)
	}
}

func TestMergeDuplicatesMergesPendingGroupsOnly(t *testing.T) {
	q, dir, ids := seedMergeGroups(t)
	log := runMergeDuplicates(t, q, nil)
	if !strings.Contains(log, "2 group(s) merged, 2 file(s) deleted") {
		t.Errorf("summary wrong:\n%s", log)
	}
	// Anchors kept, other active members deleted, excluded and dismissed untouched.
	for n, want := range map[string]bool{
		"t1.jpg": false, "t2.jpg": true, // t2 kept: higher resolution
		"l1.jpg": true, "l2.jpg": false, "l3.jpg": true,
		"d1.jpg": true, "d2.jpg": true,
	} {
		if got := exists(filepath.Join(dir, n)); got != want {
			t.Errorf("%s exists=%v; want %v", n, got, want)
		}
	}
	// The tight group is gone (nothing left to review); the loose group keeps
	// its anchor + excluded member as rows and is auto-dismissed (1 active).
	if _, found, _ := media.GetDuplicateGroup(q.Db, ids["tight"]); found {
		t.Error("fully merged group should be pruned")
	}
	if g, found, _ := media.GetDuplicateGroup(q.Db, ids["loose"]); !found || g.ActiveCount != 1 || g.MemberCount != 2 {
		t.Errorf("loose group after merge: found=%v %+v", found, g)
	}
	if g, found, _ := media.GetDuplicateGroup(q.Db, ids["dismissed"]); !found || g.MemberCount != 2 {
		t.Errorf("dismissed group touched: found=%v %+v", found, g)
	}
	// The kept copy (t2, which already had the tag) still has exactly it.
	var tags int
	_ = q.Db.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, filepath.Join(dir, "t2.jpg")).Scan(&tags)
	if tags != 1 {
		t.Errorf("keeper tags = %d; want 1", tags)
	}
	// Second run: nothing pending is mergeable.
	log = runMergeDuplicates(t, q, nil)
	if !strings.Contains(log, "No pending duplicate groups") && !strings.Contains(log, "0 group(s) merged") {
		t.Errorf("second run should find nothing:\n%s", log)
	}
}

func TestMergeDuplicatesMinSimilaritySkipsLooseGroups(t *testing.T) {
	q, dir, ids := seedMergeGroups(t)
	log := runMergeDuplicates(t, q, []string{"--min-similarity=99"})
	if !strings.Contains(log, "1 group(s) merged, 1 file(s) deleted") || !strings.Contains(log, "1 skipped below") {
		t.Errorf("summary wrong:\n%s", log)
	}
	if exists(filepath.Join(dir, "t1.jpg")) || !exists(filepath.Join(dir, "l2.jpg")) {
		t.Error("only the 100% group should have merged")
	}
	if g, found, _ := media.GetDuplicateGroup(q.Db, ids["loose"]); !found || g.Status != media.DuplicateStatusPending {
		t.Errorf("loose group should still be pending for review: %+v", g)
	}
}
