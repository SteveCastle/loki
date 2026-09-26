package media

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	_ "modernc.org/sqlite"
)

func newDuplicatesDB(t *testing.T) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	if err := InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return db
}

func seedMediaRows(t *testing.T, db *sql.DB, paths ...string) {
	t.Helper()
	for _, p := range paths {
		if _, err := db.Exec(`INSERT OR IGNORE INTO media (path) VALUES (?)`, p); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCreateDuplicateGroupEnforcesOneGroupPerPath(t *testing.T) {
	db := newDuplicatesDB(t)
	id, err := CreateDuplicateGroup(db, "m", 0.9995, "a.jpg", []DuplicateMember{
		{Path: "a.jpg", Score: 1}, {Path: "b.jpg", Score: 0.9999}, {Path: "c.jpg", Score: 0.9997},
	})
	if err != nil || id == 0 {
		t.Fatalf("create: id=%d err=%v", id, err)
	}
	// b.jpg is taken: a second group that only has b.jpg plus one new path
	// collapses to one member and is NOT created.
	id2, err := CreateDuplicateGroup(db, "m", 0.9995, "d.jpg", []DuplicateMember{
		{Path: "d.jpg", Score: 1}, {Path: "b.jpg", Score: 0.9999},
	})
	if err != nil {
		t.Fatal(err)
	}
	if id2 != 0 {
		t.Errorf("overlapping group created (id %d); want rejected", id2)
	}
	// A group whose anchor is already taken is rejected even with two free members.
	id3, err := CreateDuplicateGroup(db, "m", 0.9995, "a.jpg", []DuplicateMember{
		{Path: "a.jpg", Score: 1}, {Path: "e.jpg", Score: 0.9999}, {Path: "f.jpg", Score: 0.9999},
	})
	if err != nil {
		t.Fatal(err)
	}
	if id3 != 0 {
		t.Errorf("group with taken anchor created (id %d); want rejected", id3)
	}
	byPath, anchors, err := DuplicateAssignments(db)
	if err != nil {
		t.Fatal(err)
	}
	if len(byPath) != 3 || byPath["b.jpg"] != id {
		t.Errorf("assignments = %v; want a/b/c → %d", byPath, id)
	}
	if anchors[id] != "a.jpg" {
		t.Errorf("anchor = %q; want a.jpg", anchors[id])
	}

	added, err := AddDuplicateMembers(db, id, []DuplicateMember{{Path: "g.jpg", Score: 0.9996}, {Path: "c.jpg", Score: 0.5}})
	if err != nil {
		t.Fatal(err)
	}
	if added != 1 {
		t.Errorf("added = %d; want 1 (c.jpg already a member)", added)
	}
	g, found, err := GetDuplicateGroup(db, id)
	if err != nil || !found {
		t.Fatalf("get: found=%v err=%v", found, err)
	}
	if g.MemberCount != 4 || g.ActiveCount != 4 {
		t.Errorf("counts = %d/%d; want 4/4", g.MemberCount, g.ActiveCount)
	}
	if g.Members[0].Path != "a.jpg" {
		t.Errorf("first member = %q; want the anchor", g.Members[0].Path)
	}
	if g.MinScore < 0.9995 || g.MinScore > 0.9997 {
		t.Errorf("minScore = %v; want the lowest member score", g.MinScore)
	}
}

func TestExcludeMemberAutoDismissesAndReanchors(t *testing.T) {
	db := newDuplicatesDB(t)
	id, err := CreateDuplicateGroup(db, "m", 0.99, "a.jpg", []DuplicateMember{
		{Path: "a.jpg", Score: 1}, {Path: "b.jpg", Score: 0.995}, {Path: "c.jpg", Score: 0.992},
	})
	if err != nil || id == 0 {
		t.Fatal(err)
	}
	// Excluding the anchor re-anchors on the best remaining active member.
	if err := SetDuplicateMemberExcluded(db, id, "a.jpg", true); err != nil {
		t.Fatal(err)
	}
	g, _, _ := GetDuplicateGroup(db, id)
	if g.AnchorPath != "b.jpg" {
		t.Errorf("anchor after excluding a.jpg = %q; want b.jpg", g.AnchorPath)
	}
	if g.Status != DuplicateStatusPending || g.ActiveCount != 2 {
		t.Errorf("status/active = %s/%d; want pending/2", g.Status, g.ActiveCount)
	}
	if g.Members[len(g.Members)-1].Path != "a.jpg" || !g.Members[len(g.Members)-1].Excluded {
		t.Errorf("excluded member should sort last: %+v", g.Members)
	}
	// One active member left → nothing to merge → dismissed automatically.
	if err := SetDuplicateMemberExcluded(db, id, "c.jpg", true); err != nil {
		t.Fatal(err)
	}
	g, _, _ = GetDuplicateGroup(db, id)
	if g.Status != DuplicateStatusDismissed {
		t.Errorf("status = %s; want dismissed once < 2 active members remain", g.Status)
	}
	// The rows stay, so the paths are still "taken" for the next scan.
	byPath, _, _ := DuplicateAssignments(db)
	if len(byPath) != 3 {
		t.Errorf("assignments after exclusions = %d; want 3 (rows kept)", len(byPath))
	}
	if err := SetDuplicateMemberExcluded(db, id, "zzz.jpg", true); err == nil {
		t.Error("excluding a non-member should fail")
	}
	if err := SetDuplicateGroupStatus(db, id, DuplicateStatusPending); err != nil {
		t.Fatal(err)
	}
	if err := SetDuplicateGroupStatus(db, id+100, DuplicateStatusPending); err == nil {
		t.Error("status change on unknown group should fail")
	}
}

func TestDuplicateGroupsFollowMediaRemovalAndMoves(t *testing.T) {
	db := newDuplicatesDB(t)
	dir := t.TempDir()
	a, b, c := filepath.Join(dir, "a.jpg"), filepath.Join(dir, "b.jpg"), filepath.Join(dir, "c.jpg")
	seedMediaRows(t, db, a, b, c)
	id, err := CreateDuplicateGroup(db, "m", 0.99, a, []DuplicateMember{
		{Path: a, Score: 1}, {Path: b, Score: 0.995}, {Path: c, Score: 0.992},
	})
	if err != nil || id == 0 {
		t.Fatal(err)
	}
	// Removing the anchor's media row re-anchors the group on b (best score).
	if _, err := RemoveItemsFromDB(context.Background(), db, []string{a}); err != nil {
		t.Fatal(err)
	}
	g, found, err := GetDuplicateGroup(db, id)
	if err != nil || !found {
		t.Fatalf("group after removing anchor: found=%v err=%v", found, err)
	}
	if g.AnchorPath != b || g.MemberCount != 2 {
		t.Errorf("after removal: anchor=%q members=%d; want %q/2", g.AnchorPath, g.MemberCount, b)
	}
	// Moving a member follows it.
	moved := filepath.Join(dir, "moved.jpg")
	if err := os.WriteFile(c, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := MovePath(context.Background(), db, c, moved, MoveOptions{}); err != nil {
		t.Fatal(err)
	}
	g, _, _ = GetDuplicateGroup(db, id)
	var paths []string
	for _, m := range g.Members {
		paths = append(paths, m.Path)
	}
	if strings.Join(paths, ",") != b+","+moved {
		t.Errorf("members after move = %v; want [%s %s]", paths, b, moved)
	}
	// Removing one more leaves a single member: the group is pruned.
	if _, err := RemoveItemsFromDB(context.Background(), db, []string{moved}); err != nil {
		t.Fatal(err)
	}
	if _, found, _ := GetDuplicateGroup(db, id); found {
		t.Error("group with one member should be pruned on media removal")
	}
	var members int
	_ = db.QueryRow(`SELECT COUNT(*) FROM duplicate_member`).Scan(&members)
	if members != 0 {
		t.Errorf("orphan member rows = %d; want 0", members)
	}
}

func TestDuplicateStatsListAndWipe(t *testing.T) {
	db := newDuplicatesDB(t)
	mk := func(anchor string, others ...string) int64 {
		t.Helper()
		members := []DuplicateMember{{Path: anchor, Score: 1}}
		for _, o := range others {
			members = append(members, DuplicateMember{Path: o, Score: 0.999})
		}
		id, err := CreateDuplicateGroup(db, "m", 0.9995, anchor, members)
		if err != nil || id == 0 {
			t.Fatalf("create %s: id=%d err=%v", anchor, id, err)
		}
		return id
	}
	g1 := mk("a1", "a2", "a3", "a4")
	g2 := mk("b1", "b2", "b3")
	g3 := mk("c1", "c2")
	if err := SetDuplicateGroupStatus(db, g3, DuplicateStatusDismissed); err != nil {
		t.Fatal(err)
	}
	s, err := GetDuplicateStats(db)
	if err != nil {
		t.Fatal(err)
	}
	if s.Pending != 2 || s.Dismissed != 1 || s.Members != 9 || s.PendingItems != 7 {
		t.Errorf("stats = %+v; want pending 2, dismissed 1, members 9, pendingItems 7", s)
	}
	pending, total, err := ListDuplicateGroups(db, DuplicateStatusPending, "", 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	if total != 2 || len(pending) != 1 || pending[0].ID != g2 {
		t.Errorf("pending page = %d groups (total %d), first %d; want 1 (2), newest first (%d)", len(pending), total, pending[0].ID, g2)
	}
	if len(pending[0].Members) != 3 {
		t.Errorf("members loaded = %d; want 3", len(pending[0].Members))
	}
	// "members" order puts the biggest cluster first regardless of age.
	bySize, _, err := ListDuplicateGroups(db, DuplicateStatusPending, DuplicateSortMembers, 10, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(bySize) != 2 || bySize[0].ID != g1 || bySize[1].ID != g2 {
		t.Errorf("members sort = %v; want [%d %d] (4 members before 3)", bySize, g1, g2)
	}
	all, total, _ := ListDuplicateGroups(db, "", "", 10, 0)
	if total != 3 || len(all) != 3 {
		t.Errorf("all = %d/%d; want 3/3", len(all), total)
	}
	if gid, _ := DuplicateGroupForPath(db, "b3"); gid != g2 {
		t.Errorf("group for b3 = %d; want %d", gid, g2)
	}
	if gid, _ := DuplicateGroupForPath(db, "nope"); gid != 0 {
		t.Errorf("group for unknown path = %d; want 0", gid)
	}
	if err := DeleteDuplicateGroup(db, g1); err != nil {
		t.Fatal(err)
	}
	if err := DeleteDuplicateGroup(db, g1); err == nil {
		t.Error("deleting twice should fail")
	}
	n, err := DeleteDuplicateGroups(db, DuplicateStatusDismissed)
	if err != nil || n != 1 {
		t.Fatalf("wipe dismissed: n=%d err=%v", n, err)
	}
	n, err = DeleteDuplicateGroups(db, "")
	if err != nil || n != 1 {
		t.Fatalf("wipe all: n=%d err=%v", n, err)
	}
	s, _ = GetDuplicateStats(db)
	if s.Members != 0 || s.Pending != 0 {
		t.Errorf("stats after wipe = %+v; want empty", s)
	}
}

func TestDuplicatePredicateSQL(t *testing.T) {
	db := newDuplicatesDB(t)
	seedMediaRows(t, db, "a", "b", "c", "d", "e")
	g1, _ := CreateDuplicateGroup(db, "m", 0.99, "a", []DuplicateMember{{Path: "a", Score: 1}, {Path: "b", Score: 0.99}, {Path: "c", Score: 0.99}})
	g2, _ := CreateDuplicateGroup(db, "m", 0.99, "d", []DuplicateMember{{Path: "d", Score: 1}, {Path: "e", Score: 0.99}})
	if err := SetDuplicateMemberExcluded(db, g1, "c", true); err != nil {
		t.Fatal(err)
	}
	if err := SetDuplicateGroupStatus(db, g2, DuplicateStatusDismissed); err != nil {
		t.Fatal(err)
	}
	paths := func(value string) string {
		t.Helper()
		clause, args, ok := DuplicatePredicateSQL(value, "media.path")
		if !ok {
			return "<invalid>"
		}
		rows, err := db.Query(`SELECT path FROM media WHERE `+clause+` ORDER BY path`, args...)
		if err != nil {
			t.Fatalf("%s: %v", value, err)
		}
		defer rows.Close()
		var out []string
		for rows.Next() {
			var p string
			_ = rows.Scan(&p)
			out = append(out, p)
		}
		return strings.Join(out, ",")
	}
	if got := paths("pending"); got != "a,b" {
		t.Errorf("dupe:pending = %q; want a,b (active members of pending groups)", got)
	}
	if got := paths("any"); got != "a,b,c,d,e" {
		t.Errorf("dupe:any = %q; want every grouped path", got)
	}
	if got := paths(" " + strings.TrimSpace(itoa(g1)) + " "); got != "a,b" {
		t.Errorf("dupe:<id> = %q; want a,b", got)
	}
	if got := paths(itoa(g2)); got != "d,e" {
		t.Errorf("dupe:<dismissed id> = %q; want d,e (explicit id ignores status)", got)
	}
	for _, bad := range []string{"", "x", "0", "-1", "1.5"} {
		if got := paths(bad); got != "<invalid>" {
			t.Errorf("dupe:%q = %q; want invalid", bad, got)
		}
	}
}

func itoa(n int64) string { return strconv.FormatInt(n, 10) }

// TestMergeRepairsInterruptedMergeAndPrunesGroup covers the large-merge path:
// files already gone from disk (an earlier merge deleted them, then the
// request was abandoned before their rows were erased) are treated as
// deleted, every source's rows go in one batched erase, and the duplicate
// group that held them is pruned / re-anchored. Enough sources to span more
// than one removal batch, so the touched-group scratch table is reused.
func TestMergeRepairsInterruptedMergeAndPrunesGroup(t *testing.T) {
	db := newDuplicatesDB(t)
	dir := t.TempDir()
	keep := filepath.Join(dir, "keep.jpg")
	if err := os.WriteFile(keep, []byte("k"), 0o644); err != nil {
		t.Fatal(err)
	}
	seedMediaRows(t, db, keep)
	members := []DuplicateMember{{Path: keep, Score: 1}}
	var sources []string
	const n = 1203 // > 2 removal batches of 500
	for i := 0; i < n; i++ {
		p := filepath.Join(dir, fmt.Sprintf("copy-%04d.jpg", i))
		// Only every third copy still exists on disk; the rest were removed
		// by the interrupted run and must still be erased from the DB.
		if i%3 == 0 {
			if err := os.WriteFile(p, []byte("c"), 0o644); err != nil {
				t.Fatal(err)
			}
		}
		seedMediaRows(t, db, p)
		if _, err := db.Exec(`INSERT INTO media_tag_by_category (media_path, tag_label, category_label, weight, time_stamp)
			VALUES (?, 'copy', 'Scene', 1, 0)`, p); err != nil {
			t.Fatal(err)
		}
		members = append(members, DuplicateMember{Path: p, Score: 0.9999})
		sources = append(sources, p)
	}
	gid, err := CreateDuplicateGroup(db, "m", 0.9995, keep, members)
	if err != nil || gid == 0 {
		t.Fatalf("create group: id=%d err=%v", gid, err)
	}
	// A second, unrelated group must be untouched by the scoped prune.
	other, err := CreateDuplicateGroup(db, "m", 0.9995, "/o1.jpg", []DuplicateMember{{Path: "/o1.jpg", Score: 1}, {Path: "/o2.jpg", Score: 0.999}})
	if err != nil || other == 0 {
		t.Fatal(err)
	}

	// (The deletion phase runs under context.WithoutCancel so an abandoned
	// request cannot interrupt it; a pre-cancelled context would be refused
	// by the metadata transaction, correctly, so this exercises the repair.)
	res, err := MergeInto(context.Background(), db, keep, sources)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	if len(res.Deleted) != n || len(res.Failed) != 0 {
		t.Fatalf("deleted=%d failed=%d; want %d/0", len(res.Deleted), len(res.Failed), n)
	}
	if res.Tags != 1 {
		t.Errorf("tags gained = %d; want 1 (the shared 'copy' tag once)", res.Tags)
	}
	var rows int
	_ = db.QueryRow(`SELECT COUNT(*) FROM media`).Scan(&rows)
	if rows != 1 {
		t.Errorf("media rows = %d; want 1 (keeper only)", rows)
	}
	if _, found, _ := GetDuplicateGroup(db, gid); found {
		t.Error("fully merged group should be pruned")
	}
	if g, found, _ := GetDuplicateGroup(db, other); !found || g.MemberCount != 2 {
		t.Errorf("unrelated group touched: found=%v %+v", found, g)
	}
	for i := 0; i < n; i += 3 {
		if _, err := os.Stat(filepath.Join(dir, fmt.Sprintf("copy-%04d.jpg", i))); !os.IsNotExist(err) {
			t.Fatalf("copy %d should be deleted from disk", i)
		}
	}
}

func TestPreferredDuplicateKeeperRule(t *testing.T) {
	g := DuplicateGroup{AnchorPath: "/a.jpg", Members: []DuplicateMember{
		{Path: "/a.jpg", Score: 1}, {Path: "/b.jpg", Score: 0.999}, {Path: "/c.jpg", Score: 0.999}, {Path: "/x.jpg", Score: 0.999, Excluded: true},
	}}
	// Highest resolution wins, even over the anchor.
	facts := map[string]DuplicateMemberFacts{
		"/a.jpg": {Width: 800, Height: 600, Size: 900_000},
		"/b.jpg": {Width: 1920, Height: 1080, Size: 100_000},
		"/c.jpg": {Width: 1920, Height: 1080, Size: 400_000},
		"/x.jpg": {Width: 4000, Height: 3000, Size: 9_000_000}, // excluded: never a keeper
	}
	if k := PreferredDuplicateKeeper(g, facts); k != "/c.jpg" {
		t.Errorf("keeper = %s; want /c.jpg (top resolution, larger file)", k)
	}
	// No dimensions anywhere: the largest file.
	facts = map[string]DuplicateMemberFacts{"/a.jpg": {Size: 10}, "/b.jpg": {Size: 30}, "/c.jpg": {Size: 20}}
	if k := PreferredDuplicateKeeper(g, facts); k != "/b.jpg" {
		t.Errorf("keeper without dims = %s; want /b.jpg (largest file)", k)
	}
	// Nothing known at all: the anchor.
	if k := PreferredDuplicateKeeper(g, nil); k != "/a.jpg" {
		t.Errorf("keeper with no facts = %s; want the anchor", k)
	}
	// Ties beyond size: anchor first, then the shortest path.
	facts = map[string]DuplicateMemberFacts{"/a.jpg": {Size: 5}, "/b.jpg": {Size: 5}, "/c.jpg": {Size: 5}}
	if k := PreferredDuplicateKeeper(g, facts); k != "/a.jpg" {
		t.Errorf("tie keeper = %s; want the anchor", k)
	}
	g.AnchorPath = "/zz.jpg" // anchor gone (re-anchor pending): shortest path
	if k := PreferredDuplicateKeeper(g, facts); k != "/a.jpg" {
		t.Errorf("tie keeper without anchor = %s; want shortest path /a.jpg", k)
	}
	// Loaded facts come from the media table.
	db := newDuplicatesDB(t)
	if _, err := db.Exec(`INSERT INTO media (path, width, height, size) VALUES ('/p.jpg', 100, 50, 777)`); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadDuplicateMemberFacts(db, []string{"/p.jpg", "/missing.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if f := loaded["/p.jpg"]; f.Width != 100 || f.Height != 50 || f.Size != 777 {
		t.Errorf("loaded facts = %+v", f)
	}
	if _, ok := loaded["/missing.jpg"]; ok {
		t.Error("missing row should be absent")
	}
}
