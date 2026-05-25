package media

import (
	"database/sql"
	"fmt"
	"testing"
	"time"
)

// TestSamplerQueryPathsSorted guards the swipe pagination contract.
//
// Background: a tag mutation invalidates the sampler cache and `runBuild`
// drops the cached shuffle. The next sample call reshuffles using the
// caller's session seed. Because the shuffle algorithm is input-order-
// sensitive, the new shuffle only matches the old one when queryPaths
// returns paths in the same order.
//
// Bug this guards against: queryPaths used to omit ORDER BY. SQLite gives
// no order guarantee for DISTINCT without ORDER BY, so on rebuild it could
// return the same set of paths in a different order. The reshuffle then
// produced a different permutation, and the swipe client's monotonic
// `offset` started re-emitting items it already showed — surfacing as
// "the swipe loops over the same set once before new items appear".
//
// The fix pins queryPaths to ORDER BY media_path so the input order is
// stable across rebuilds, which makes the deterministic shuffle stable
// for an unchanged universe.
func TestSamplerQueryPathsSorted(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Insert paths in non-sorted order to make sure we're not just
	// reading back insertion order.
	insertOrder := []string{
		"/m/zebra.jpg",
		"/m/alpha.jpg",
		"/m/middle.jpg",
		"/m/banana.jpg",
		"/m/yankee.jpg",
		"/m/charlie.jpg",
	}
	for _, p := range insertOrder {
		if _, err := db.Exec(
			`INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, 'x', 'c')`,
			p,
		); err != nil {
			t.Fatalf("seed insert: %v", err)
		}
	}

	s := &randomSampler{}
	got, err := s.queryPaths(db)
	if err != nil {
		t.Fatalf("queryPaths: %v", err)
	}

	for i := 1; i < len(got); i++ {
		if got[i-1] > got[i] {
			t.Fatalf("queryPaths must return paths in sorted order so the per-seed shuffle stays reproducible across rebuilds; got %v", got)
		}
	}
}

// TestSamplerShuffleStableForSameInput guards the lower-level invariant
// that runBuild relies on: same seed + same path slice (in the same order)
// produces the same shuffle. queryPaths supplies "same order" by sorting;
// this test pins down "same shuffle".
func TestSamplerShuffleStableForSameInput(t *testing.T) {
	const N = 100
	paths := make([]string, N)
	for i := 0; i < N; i++ {
		paths[i] = fmt.Sprintf("/m/%03d", i)
	}

	a := &randomSampler{paths: append([]string(nil), paths...)}
	b := &randomSampler{paths: append([]string(nil), paths...)}

	const seed = int64(12345)
	pageA, _ := a.sample(seed, 0, 30)
	pageB1, _ := b.sample(seed, 0, 30)
	pageB2, _ := b.sample(seed, 30, 30)

	for i := range pageA {
		if pageA[i] != pageB1[i] {
			t.Fatalf("same seed + same input must yield identical shuffle; pageA[%d]=%q pageB1[%d]=%q", i, pageA[i], i, pageB1[i])
		}
	}

	// And the second page from the same shuffle must be disjoint from the first.
	seen := make(map[string]bool, len(pageA))
	for _, p := range pageA {
		seen[p] = true
	}
	for _, p := range pageB2 {
		if seen[p] {
			t.Fatalf("page2 of a stable shuffle must not overlap page1; duplicate %q", p)
		}
	}
}

// TestSamplerEnsureBuiltSurfacesFirstBuildError guards the contract that
// callers can distinguish "the universe is empty" from "we couldn't build
// the universe". Before the fix, ensureBuilt logged the build error and
// returned nil; the sampler then served zero items, the API returned 200
// with an empty list, and the swipe UI showed "no matching media" — for
// what was really a transient SQL failure.
func TestSamplerEnsureBuiltSurfacesFirstBuildError(t *testing.T) {
	// Use a closed connection: every query fails. A real-world equivalent
	// is a DB swap that closed the old handle while a request was mid-flight.
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	db.Close()

	s := &randomSampler{}
	if err := s.ensureBuilt(db); err == nil {
		t.Fatal("ensureBuilt must surface the first-build error when there is no cache to fall back on")
	}
}

// TestSamplerStaleCacheSurvivesBackgroundFailure guards the stale-while-
// revalidate contract: if a background rebuild fails, callers keep the
// existing snapshot rather than getting a sudden error.
func TestSamplerStaleCacheSurvivesBackgroundFailure(t *testing.T) {
	s := &randomSampler{
		paths:   []string{"/m/a.jpg", "/m/b.jpg"},
		builtAt: time.Now().Add(-2 * samplerTTL), // TTL-stale
		stale:   true,
	}

	// Pass a closed DB so the background rebuild will fail.
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	db.Close()

	// With an existing snapshot, ensureBuilt must NOT block on the
	// background build and must NOT return an error: the stale snapshot is
	// the fallback. The background goroutine may finish before or after
	// this returns; either way the snapshot must still be sampleable.
	if err := s.ensureBuilt(db); err != nil {
		t.Fatalf("ensureBuilt with stale cache must swallow background-build errors; got %v", err)
	}

	// Give the background rebuild a moment to fail. It must not clobber
	// the snapshot, only mark `lastBuildErr`.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		done := s.buildCh == nil
		s.mu.Unlock()
		if done {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	got, total := s.sample(42, 0, 10)
	if total != 2 || len(got) != 2 {
		t.Fatalf("stale snapshot must still be sampleable after background build fails; got total=%d len=%d", total, len(got))
	}
}

// TestSamplerResetDropsCache guards the DB-swap contract: after Reset the
// next sample call must NOT return paths from the old snapshot, because
// the underlying database is now different and those paths may not exist
// in it.
func TestSamplerResetDropsCache(t *testing.T) {
	s := &randomSampler{
		paths: []string{"/old/a.jpg", "/old/b.jpg"},
	}
	if got, total := s.sample(1, 0, 10); total != 2 || len(got) != 2 {
		t.Fatalf("precondition: sampler should hold 2 paths; got total=%d len=%d", total, len(got))
	}
	s.Reset()
	if got, total := s.sample(1, 0, 10); total != 0 || len(got) != 0 {
		t.Fatalf("after Reset, sampler must hold zero paths so old-DB paths can't leak; got total=%d len=%d", total, len(got))
	}
}
