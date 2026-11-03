package tasks

import (
	"errors"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/media"
)

// The per-item check seeds a star around the item from its ungrouped
// neighbours, joins an existing group through its anchor, and never touches
// the tables until the plan is applied.
func TestPlanDuplicatesForPath(t *testing.T) {
	dir := t.TempDir()
	q := newFindDuplicatesQueue(t, duplicateFixture(dir))
	db := q.Db
	p := func(n string) string { return filepath.Join(dir, n) }

	// Seed: a.jpg has b (identical) and near (≈ 0.9998) at 100%; alike is
	// below the floor.
	plan, err := PlanDuplicatesForPath(db, p("a.jpg"), 100)
	if err != nil {
		t.Fatal(err)
	}
	if plan.JoinGroup != 0 || len(plan.Members) != 3 || plan.Members[0].Path != p("a.jpg") || plan.Members[0].Score != 1 {
		t.Fatalf("seed plan = %+v; want a + b + near anchored on a", plan)
	}
	if id, _ := media.DuplicateGroupForPath(db, p("a.jpg")); id != 0 {
		t.Fatalf("planning must not write; a is in group %d", id)
	}
	// Lower threshold pulls in the look-alike.
	if plan, _ := PlanDuplicatesForPath(db, p("a.jpg"), 97); len(plan.Members) != 4 {
		t.Fatalf("97%% plan = %+v; want 4 members", plan)
	}

	gid, err := ApplyDuplicatePlan(db, p("a.jpg"), plan)
	if err != nil || gid == 0 {
		t.Fatalf("apply: gid=%d err=%v", gid, err)
	}
	g, _, _ := media.GetDuplicateGroup(db, gid)
	if g.AnchorPath != p("a.jpg") || g.ActiveCount != 3 {
		t.Fatalf("group = %+v", g)
	}

	// Join: at 97% alike now matches the existing group's anchor.
	plan, err = PlanDuplicatesForPath(db, p("alike.jpg"), 97)
	if err != nil {
		t.Fatal(err)
	}
	if plan.JoinGroup != gid || len(plan.Members) != 0 || plan.JoinScore < 0.97 {
		t.Fatalf("join plan = %+v; want join of group %d", plan, gid)
	}
	if got, err := ApplyDuplicatePlan(db, p("alike.jpg"), plan); err != nil || got != gid {
		t.Fatalf("apply join: %d %v", got, err)
	}
	if g, _, _ := media.GetDuplicateGroup(db, gid); g.ActiveCount != 4 {
		t.Fatalf("after join: %+v", g)
	}

	// Nothing near: no plan.
	if plan, err := PlanDuplicatesForPath(db, p("x.jpg"), 100); err != nil || plan.Found() {
		t.Fatalf("x plan = %+v err=%v; want nothing", plan, err)
	}
	// No embedding.
	if _, err := PlanDuplicatesForPath(db, p("missing.jpg"), 100); !errors.Is(err, ErrNoEmbedding) {
		t.Fatalf("missing: err=%v; want ErrNoEmbedding", err)
	}
}

// Neighbours already in some other group (not as its anchor) are left out
// of a new star — a path belongs to at most one group.
func TestPlanDuplicatesForPathSkipsGroupedNeighbours(t *testing.T) {
	dir := t.TempDir()
	q := newFindDuplicatesQueue(t, duplicateFixture(dir))
	db := q.Db
	p := func(n string) string { return filepath.Join(dir, n) }
	// b sits in a group anchored on an unrelated item.
	if gid, err := media.CreateDuplicateGroup(db, "m", 1, p("x.jpg"), []media.DuplicateMember{
		{Path: p("x.jpg"), Score: 1}, {Path: p("b.jpg"), Score: 0.5},
	}); err != nil || gid == 0 {
		t.Fatal(err)
	}
	plan, err := PlanDuplicatesForPath(db, p("a.jpg"), 100)
	if err != nil {
		t.Fatal(err)
	}
	if plan.JoinGroup != 0 || len(plan.Members) != 2 || plan.Members[1].Path != p("near.jpg") {
		t.Fatalf("plan = %+v; want a + near only", plan)
	}
}
