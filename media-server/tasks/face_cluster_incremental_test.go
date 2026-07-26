package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"math/rand"
	"sort"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// setDetScore pins a face's detection confidence so phase 2's
// strongest-first processing order is deterministic in these tests.
func setDetScore(t *testing.T, db *sql.DB, id int64, score float64) {
	t.Helper()
	if _, err := db.Exec(`UPDATE face SET det_score=? WHERE id=?`, score, id); err != nil {
		t.Fatalf("set det_score: %v", err)
	}
}

// grouping returns path→person-name for every assigned face, the observable
// outcome of a clustering run. Anonymous names are canonicalized to their
// member set so two runs that mint the same groups in a different order still
// compare equal.
func grouping(t *testing.T, db *sql.DB, model string) map[string]string {
	t.Helper()
	faces, err := media.LoadAllFaces(db, model)
	if err != nil {
		t.Fatal(err)
	}
	byPerson := map[int64][]string{}
	for _, f := range faces {
		if f.PersonID != 0 {
			byPerson[f.PersonID] = append(byPerson[f.PersonID], fmt.Sprintf("%s#%d", f.MediaPath, f.ID))
		}
	}
	out := map[string]string{}
	for _, members := range byPerson {
		sort.Strings(members)
		key := fmt.Sprint(members)
		for _, m := range members {
			out[m] = key
		}
	}
	return out
}

// buildIncrementalFixture seeds a library with one curated person, faces that
// should join it, two groups that should form, and noise that shouldn't.
func buildIncrementalFixture(t *testing.T, db *sql.DB) FaceModel {
	t.Helper()
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	if err := media.AssignFace(db, seedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}
	for i := range 6 {
		seedFaces(t, db, fmt.Sprintf("a%d.jpg", i), "m1", vecNear([]float32{1, 0, 0}, float32(i-3)*0.01))
	}
	for i := range 5 {
		seedFaces(t, db, fmt.Sprintf("b%d.jpg", i), "m1", vecNear([]float32{0, 1, 0}, float32(i-2)*0.01))
	}
	for i := range 4 {
		seedFaces(t, db, fmt.Sprintf("c%d.jpg", i), "m1", vecNear([]float32{0, 0, 1}, float32(i-2)*0.01))
	}
	// Isolated noise, one face per direction — never enough to form.
	seedFaces(t, db, "n0.jpg", "m1", []float32{0.7, 0.7, 0})
	seedFaces(t, db, "n1.jpg", "m1", []float32{0, 0.7, 0.7})
	return FaceModel{ID: "m1", MatchThreshold: 0.9}
}

func incrementalParams(flushEvery int) clusterParams {
	return clusterParams{
		joinThreshold: 0.9, formThreshold: 0.95, minQuality: 0.75,
		minCluster: 3, passes: 2, flushEvery: flushEvery,
	}
}

// Flushing as the pass runs is a change of WRITE TIMING only: the people it
// ends up with, and every statistic it reports, must be identical to the same
// pass committing once at the end.
func TestClusterFacesFlushEveryMatchesSingleCommit(t *testing.T) {
	run := func(flushEvery int) (clusterStats, map[string]string) {
		db := newFaceIndexDB(t)
		resetFaceIndex(t)
		model := buildIncrementalFixture(t, db)
		stats, err := clusterFaces(context.Background(), db, model, incrementalParams(flushEvery))
		if err != nil {
			t.Fatal(err)
		}
		return stats, grouping(t, db, "m1")
	}

	wantStats, wantGroups := run(1 << 20) // one commit at the very end
	if wantStats.NewPeople == 0 || wantStats.JoinedExisting == 0 {
		t.Fatalf("fixture is not exercising both phases: %+v", wantStats)
	}
	for _, flushEvery := range []int{1, 2, 3, 5, 7} {
		gotStats, gotGroups := run(flushEvery)
		if gotStats != wantStats {
			t.Fatalf("flushEvery=%d stats = %+v, want %+v", flushEvery, gotStats, wantStats)
		}
		if len(gotGroups) != len(wantGroups) {
			t.Fatalf("flushEvery=%d assigned %d faces, want %d", flushEvery, len(gotGroups), len(wantGroups))
		}
		for face, group := range wantGroups {
			if gotGroups[face] != group {
				t.Fatalf("flushEvery=%d: face %s grouped as %q, want %q", flushEvery, face, gotGroups[face], group)
			}
		}
	}
}

// The point of flushing: cancelling mid-pass keeps the groups already written,
// and re-running finishes the job from there instead of starting over.
func TestClusterFacesCancelKeepsFlushedProgress(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	model := buildIncrementalFixture(t, db)

	// Cancel as soon as the first flush has been reported.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	p := incrementalParams(1)
	ticks := 0
	p.progress = func(pr clusterProgress) {
		ticks++
		if ticks >= 2 {
			cancel()
		}
	}
	stats, err := clusterFaces(ctx, db, model, p)
	if err == nil {
		t.Fatal("want a cancellation error")
	}
	if !ctxCanceled(ctx) {
		t.Fatal("context should be canceled")
	}
	assignedAfterCancel := len(grouping(t, db, "m1"))
	if assignedAfterCancel <= 1 {
		t.Fatalf("cancel kept %d assigned face(s) — flushed work was lost (stats %+v)", assignedAfterCancel, stats)
	}

	// Resume: a plain re-run. It must finish the remaining backlog and end in
	// the same state a single uninterrupted run would have reached.
	if _, err := clusterFaces(context.Background(), db, model, incrementalParams(1)); err != nil {
		t.Fatal(err)
	}
	resumed := grouping(t, db, "m1")

	fresh := newFaceIndexDB(t)
	resetFaceIndex(t)
	freshModel := buildIncrementalFixture(t, fresh)
	if _, err := clusterFaces(context.Background(), fresh, freshModel, incrementalParams(1<<20)); err != nil {
		t.Fatal(err)
	}
	if want := len(grouping(t, fresh, "m1")); len(resumed) != want {
		t.Fatalf("cancel+resume assigned %d faces, an uninterrupted run assigns %d", len(resumed), want)
	}
}

func ctxCanceled(ctx context.Context) bool { return ctx.Err() != nil }

// Progress must be honest: never go backwards, never overshoot, land exactly
// on the workload, and only ever report counts that are already committed.
func TestClusterFacesProgressIsMonotonicAndCommitted(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)
	model := buildIncrementalFixture(t, db)

	p := incrementalParams(2)
	lastProcessed := -1
	phases := map[clusterPhase]bool{}
	var lastTick clusterProgress
	p.progress = func(pr clusterProgress) {
		phases[pr.Phase] = true
		if pr.Processed < lastProcessed {
			t.Fatalf("processed went backwards: %d after %d", pr.Processed, lastProcessed)
		}
		if pr.Processed > pr.Workload {
			t.Fatalf("processed %d exceeds workload %d", pr.Processed, pr.Workload)
		}
		if pr.Total > 0 && pr.Done > pr.Total {
			t.Fatalf("%s: done %d exceeds total %d", pr.Phase, pr.Done, pr.Total)
		}
		// Everything the tick claims is already in the database.
		if got := len(grouping(t, db, "m1")); got < pr.Stats.JoinedExisting+pr.Stats.NewlyClustered {
			t.Fatalf("%s: tick claims %d written, database holds %d",
				pr.Phase, pr.Stats.JoinedExisting+pr.Stats.NewlyClustered, got)
		}
		lastProcessed = pr.Processed
		lastTick = pr
	}
	stats, err := clusterFaces(context.Background(), db, model, p)
	if err != nil {
		t.Fatal(err)
	}
	if lastTick.Processed != lastTick.Workload {
		t.Fatalf("final tick processed %d of %d", lastTick.Processed, lastTick.Workload)
	}
	for _, want := range []clusterPhase{phaseJoin, phasePrescreen, phaseForm} {
		if !phases[want] {
			t.Fatalf("no %s progress was reported", want)
		}
	}
	if lastTick.Stats.NewPeople != stats.NewPeople || lastTick.Stats.JoinedExisting != stats.JoinedExisting {
		t.Fatalf("final tick %+v disagrees with returned stats %+v", lastTick.Stats, stats)
	}
}

// Progress must track work got through, not faces placed. The first cut
// counted only joins, so a library whose faces mostly DON'T group — the exact
// case where a run takes longest — showed a bar pinned near zero for the whole
// of phase 1 and then a jump at the end.
func TestClusterFacesProgressCountsUnplaceableFaces(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	// One curated person plus a pile of mutually dissimilar faces: nothing
	// joins it, nothing forms a group, so every candidate is "processed but
	// unplaceable".
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(7))
	randVec := func() []float32 {
		v := make([]float32, 64)
		for k := range v {
			v[k] = float32(rng.NormFloat64())
		}
		return v
	}
	seedIDs := seedFaces(t, db, "seed.jpg", "m1", randVec())
	if err := media.AssignFace(db, seedIDs[0], alice, "user"); err != nil {
		t.Fatal(err)
	}
	// Random 64-dim vectors are mutually near-orthogonal, so none of them
	// joins Alice and none of them groups with another.
	const junk = 40
	for i := range junk {
		seedFaces(t, db, fmt.Sprintf("junk%d.jpg", i), "m1", randVec())
	}

	model := FaceModel{ID: "m1", MatchThreshold: 0.99}
	p := incrementalParams(4)
	p.joinThreshold, p.formThreshold = 0.99, 0.999
	var midJoin, final clusterProgress
	p.progress = func(pr clusterProgress) {
		// Snapshot the last tick of the first join pass.
		if pr.Phase == phaseJoin && pr.Pass == 1 {
			midJoin = pr
		}
		final = pr
	}
	stats, err := clusterFaces(context.Background(), db, model, p)
	if err != nil {
		t.Fatal(err)
	}
	if stats.JoinedExisting != 0 || stats.NewPeople != 0 {
		t.Fatalf("fixture placed something, so it can't prove the point: %+v", stats)
	}
	if midJoin.Workload != junk {
		t.Fatalf("workload = %d, want %d candidates", midJoin.Workload, junk)
	}
	// Phase 1 is one of passes+2 visits per face, so finishing its first sweep
	// with nothing placed must still move the bar off zero by about that share.
	wantAtLeast := junk / (p.passes + 2)
	if midJoin.Processed < wantAtLeast {
		t.Fatalf("after a full join pass over %d unplaceable faces, processed = %d; want ≥ %d",
			junk, midJoin.Processed, wantAtLeast)
	}
	if final.Processed != junk {
		t.Fatalf("final processed = %d, want %d (every candidate accounted for)", final.Processed, junk)
	}
}

// The bar is a visit budget spent across the join passes, the prescreen and
// the formation scan, and candidates leave the pool by several different
// routes (placed, below the quality floor, passes converging early, no seeds
// to score against at all). Whichever route a run takes, the budget must come
// out exactly even — no stalling short of full, no relying on the clamp.
func TestClusterFacesProgressLandsOnWorkloadByEveryRoute(t *testing.T) {
	cases := []struct {
		name  string
		build func(t *testing.T, db *sql.DB)
		tune  func(p *clusterParams)
	}{
		{
			name: "no seeds at all (join passes never run)",
			build: func(t *testing.T, db *sql.DB) {
				for i := range 5 {
					seedFaces(t, db, fmt.Sprintf("b%d.jpg", i), "m1", vecNear([]float32{0, 1, 0}, float32(i-2)*0.01))
				}
			},
		},
		{
			name:  "everything joins (phase 2 gets nothing)",
			build: buildAliceAndFriends,
		},
		{
			name:  "all below the quality floor (phase 2 short-circuits)",
			build: buildAliceAndFriends,
			tune:  func(p *clusterParams) { p.minQuality = 0.99 },
		},
		{
			name:  "mixed: joins, a new group, and leftovers",
			build: func(t *testing.T, db *sql.DB) { buildIncrementalFixture(t, db) },
		},
		{
			name:  "mixed, single join pass",
			build: func(t *testing.T, db *sql.DB) { buildIncrementalFixture(t, db) },
			tune:  func(p *clusterParams) { p.passes = 1 },
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			db := newFaceIndexDB(t)
			resetFaceIndex(t)
			tc.build(t, db)
			p := incrementalParams(2)
			if tc.tune != nil {
				tc.tune(&p)
			}
			var final clusterProgress
			last := -1
			p.progress = func(pr clusterProgress) {
				if pr.Processed < last {
					t.Fatalf("processed went backwards: %d after %d", pr.Processed, last)
				}
				last = pr.Processed
				final = pr
			}
			if _, err := clusterFaces(context.Background(), db, FaceModel{ID: "m1", MatchThreshold: 0.9}, p); err != nil {
				t.Fatal(err)
			}
			if final.Workload == 0 {
				t.Fatal("fixture produced no candidates")
			}
			if final.Processed != final.Workload {
				t.Fatalf("ended at %d/%d — the visit budget did not come out even",
					final.Processed, final.Workload)
			}
		})
	}
}

// buildAliceAndFriends seeds one user-confirmed person plus faces that all
// join it, so phase 2 has nothing left to do.
func buildAliceAndFriends(t *testing.T, db *sql.DB) {
	t.Helper()
	alice, err := media.CreatePerson(db, "Alice")
	if err != nil {
		t.Fatal(err)
	}
	ids := seedFaces(t, db, "seed.jpg", "m1", []float32{1, 0, 0})
	if err := media.AssignFace(db, ids[0], alice, "user"); err != nil {
		t.Fatal(err)
	}
	for i := range 5 {
		seedFaces(t, db, fmt.Sprintf("a%d.jpg", i), "m1", vecNear([]float32{1, 0, 0}, float32(i-2)*0.01))
	}
}

// A group committed early is provisional: the dissolved-group ban only gains
// overlap as members are added, so a cluster that clears it at three members
// and trips it at four must be RETRACTED, leaving exactly the state a
// single end-of-run commit would have left.
func TestFlushedClusterRetractedWhenItGrowsIntoABan(t *testing.T) {
	db := newFaceIndexDB(t)
	resetFaceIndex(t)

	x := []float32{1, 0, 0}
	// Four faces the user grouped and then deleted — the ban's membership.
	banned := seedFaces(t, db, "banned.jpg", "m1", x, x, x, x)
	blob, err := media.CreatePerson(db, "Unknown #1")
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range banned {
		if err := media.AssignFace(db, id, blob, "auto"); err != nil {
			t.Fatal(err)
		}
	}
	if n, err := media.BanFaceGroup(db, blob, "Unknown #1"); err != nil || n != 4 {
		t.Fatalf("ban: n=%d err=%v", n, err)
	}
	if err := media.DeletePerson(db, blob); err != nil {
		t.Fatal(err)
	}
	// One unbanned face at the same identity.
	free := seedFaces(t, db, "free.jpg", "m1", x)

	// Processing order: banned[0], banned[1], free[0], banned[2], banned[3].
	// The cluster reaches three members (two of them banned — under the
	// overlap floor of 3) and is committed at the flush, then banned[2] pushes
	// the overlap to 3 and the ban binds.
	for i, id := range []int64{banned[0], banned[1], free[0], banned[2], banned[3]} {
		setDetScore(t, db, id, 0.9-float64(i)*0.01)
	}

	model := FaceModel{ID: "m1", MatchThreshold: 0.9}
	p := incrementalParams(3) // flush after the first three faces
	sawPerson := false
	lastProcessed := -1
	p.progress = func(pr clusterProgress) {
		if pr.Stats.NewPeople > 0 {
			sawPerson = true
		}
		// A retraction un-writes a person; the bar measures work got through,
		// so it must not walk backwards over one.
		if pr.Processed < lastProcessed {
			t.Fatalf("processed went backwards across a retraction: %d after %d", pr.Processed, lastProcessed)
		}
		lastProcessed = pr.Processed
	}
	stats, err := clusterFaces(context.Background(), db, model, p)
	if err != nil {
		t.Fatal(err)
	}
	if !sawPerson {
		t.Fatal("the cluster was never committed early — the test no longer exercises retraction")
	}
	if stats.NewPeople != 0 || stats.NewlyClustered != 0 {
		t.Fatalf("banned cluster survived: %+v", stats)
	}
	if stats.BanBlocked != 5 {
		t.Fatalf("ban-blocked = %d, want 5 (the whole cluster): %+v", stats.BanBlocked, stats)
	}
	if assigned := len(grouping(t, db, "m1")); assigned != 0 {
		t.Fatalf("%d face(s) still assigned after retraction", assigned)
	}
	people, err := media.GetPeople(db)
	if err != nil {
		t.Fatal(err)
	}
	if len(people) != 0 {
		t.Fatalf("retraction left person rows behind: %+v", people)
	}
}
