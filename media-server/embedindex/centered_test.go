package embedindex

import (
	"fmt"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
)

// seedCommonComponent fills idx with n vectors that all share a strong common
// direction (dim 0) plus a small per-item component (dim 1+i%spread), modeling
// the "every embedding is 'a photo'" bias mean-centering exists to remove.
// Returns the stored (normalized) vector of item 0.
func seedCommonComponent(idx VectorIndex, n, dim, spread int) []float32 {
	var first []float32
	for i := 0; i < n; i++ {
		v := make([]float32, dim)
		v[0] = 1
		v[1+i%spread] = 0.3
		idx.Add(fmt.Sprintf("item-%d", i), v)
		if i == 0 {
			first = embedvec.Normalize(v)
		}
	}
	return first
}

// TestCenteredDormantBelowGate verifies a centered index scores plain cosine
// (identical to New()) until it holds MinCenterCount vectors — a tiny
// library's mean is noise, not signal.
func TestCenteredDormantBelowGate(t *testing.T) {
	plain, centered := New(), NewCentered()
	seedCommonComponent(plain, MinCenterCount-1, 8, 4)
	q := seedCommonComponent(centered, MinCenterCount-1, 8, 4)

	hp := plain.Search(q, 5)
	hc := centered.Search(q, 5)
	if len(hp) != len(hc) {
		t.Fatalf("hit count mismatch: %d vs %d", len(hp), len(hc))
	}
	for i := range hp {
		if hp[i].Path != hc[i].Path || hp[i].Score != hc[i].Score {
			t.Fatalf("below gate, centered must equal plain: %+v vs %+v", hp[i], hc[i])
		}
	}
}

// TestCenteredSelfMatchStaysPerfect pins the key UX property centering must
// preserve: the query item itself still scores 1.0 and ranks first.
func TestCenteredSelfMatchStaysPerfect(t *testing.T) {
	idx := NewCentered()
	first := seedCommonComponent(idx, 2*MinCenterCount, 16, 8)
	hits := idx.Search(first, 3)
	if len(hits) == 0 || hits[0].Path != "item-0" {
		t.Fatalf("self item should rank first, got %+v", hits)
	}
	if hits[0].Score < 0.999 || hits[0].Score > 1.001 {
		t.Fatalf("self score should stay ~1.0 centered, got %v", hits[0].Score)
	}
}

// TestCenteredSuppressesHub verifies the point of centering: an item lying on
// the shared common component (a "hub" that plain cosine scores highly against
// EVERYTHING) collapses once the mean is subtracted.
func TestCenteredSuppressesHub(t *testing.T) {
	dim := 34
	build := func(idx VectorIndex) []float32 {
		q := seedCommonComponent(idx, 2*MinCenterCount, dim, 32)
		hub := make([]float32, dim)
		hub[0] = 1 // exactly the common direction
		idx.Add("hub", hub)
		return q
	}
	plainScore := func(idx VectorIndex, q []float32) float32 {
		for _, h := range idx.Search(q, idx.Len()) {
			if h.Path == "hub" {
				return h.Score
			}
		}
		t.Fatal("hub not found in results")
		return 0
	}
	pIdx, cIdx := New(), NewCentered()
	pq, cq := build(pIdx), build(cIdx)

	uncentered := plainScore(pIdx, pq)
	centered := plainScore(cIdx, cq)
	if uncentered < 0.9 {
		t.Fatalf("sanity: plain cosine should love the hub, got %v", uncentered)
	}
	if centered > 0.3 {
		t.Fatalf("centered index should suppress the hub, got %v (uncentered %v)", centered, uncentered)
	}
}

// TestCenteredRecentersAfterGrowth verifies the lazy recenter path: grow the
// index well past the drift threshold after the first centered search, search
// again, and the index must still behave (self-match 1.0, no stale-cnorm
// panic from the new slots).
func TestCenteredRecentersAfterGrowth(t *testing.T) {
	idx := NewCentered()
	seedCommonComponent(idx, MinCenterCount, 16, 8)
	q0 := embedvec.Normalize([]float32{1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	idx.Search(q0, 1) // first search: centers at MinCenterCount vectors

	for i := 0; i < MinCenterCount; i++ { // +100% drift >> the 10%+32 threshold
		v := make([]float32, 16)
		v[0] = 1
		v[8+i%8] = 0.5
		idx.Add(fmt.Sprintf("late-%d", i), v)
	}
	late := embedvec.Normalize([]float32{1, 0, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0, 0})
	hits := idx.Search(late, 2)
	if len(hits) == 0 || hits[0].Path != "late-0" {
		t.Fatalf("late item should rank first for its own vector, got %+v", hits)
	}
	if hits[0].Score < 0.999 {
		t.Fatalf("self score after recenter should be ~1.0, got %v", hits[0].Score)
	}
}

// TestCenteredAddDeleteKeepsSumConsistent exercises the running-sum
// bookkeeping through updates and swap-deletes: after churn, a fresh search
// still self-matches perfectly (a corrupted mean/cnorm would break that).
func TestCenteredAddDeleteKeepsSumConsistent(t *testing.T) {
	idx := NewCentered()
	seedCommonComponent(idx, 3*MinCenterCount, 16, 8)
	idx.Search(embedvec.Normalize([]float32{1, 0.3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 1)

	for i := 0; i < MinCenterCount; i++ {
		idx.Delete(fmt.Sprintf("item-%d", i*2))
	}
	// In-place vector update of a survivor.
	moved := make([]float32, 16)
	moved[0] = 1
	moved[15] = 0.9
	idx.Add("item-1", moved)

	hits := idx.Search(moved, 1)
	if len(hits) != 1 || hits[0].Path != "item-1" || hits[0].Score < 0.999 {
		t.Fatalf("after churn, updated item should self-match ~1.0, got %+v", hits)
	}
}

// TestSearchSharedRanksSharedConceptFirst is the football scenario at index
// level: three examples share dim 0, each with private scenery; the pure
// shared-concept item must outrank an item identical to one example, with the
// scenery-only item last among the scored set.
func TestSearchSharedRanksSharedConceptFirst(t *testing.T) {
	idx := New() // uncentered keeps the toy geometry exact
	q1 := []float32{1, 1, 0, 0}
	q2 := []float32{1, 0, 1, 0}
	q3 := []float32{1, 0, 0, 1}
	idx.Add("football", []float32{1, 0, 0, 0})
	idx.Add("example-1", q1)
	idx.Add("scenery-1", []float32{0, 1, 0, 0})

	hits, err := idx.SearchShared(SharedSpec{Pos: [][]float32{q1, q2, q3}}, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 || hits[0].Path != "football" {
		t.Fatalf("expected football first, got %+v", hits)
	}
	if hits[1].Path != "example-1" || hits[2].Path != "scenery-1" {
		t.Fatalf("expected example-1 then scenery-1, got %+v", hits)
	}
}

// TestSearchSharedHonorsAllowAndNegatives verifies allow-set restriction and
// negative steering at the index level.
func TestSearchSharedHonorsAllowAndNegatives(t *testing.T) {
	idx := New()
	idx.Add("a", []float32{1, 0.1, 0})
	idx.Add("b", []float32{1, -0.1, 0})
	idx.Add("bad", []float32{1, 0, 0.8})

	spec := SharedSpec{
		Pos:  [][]float32{{1, 0.2, 0}, {1, -0.2, 0}},
		Neg:  [][]float32{{0, 0, 1}},
		NegW: []float32{1},
	}
	hits, err := idx.SearchShared(spec, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 3 || hits[2].Path != "bad" {
		t.Fatalf("negative-aligned item should rank last, got %+v", hits)
	}

	only := map[string]struct{}{"b": {}}
	hits, err = idx.SearchShared(spec, 10, only)
	if err != nil {
		t.Fatal(err)
	}
	if len(hits) != 1 || hits[0].Path != "b" {
		t.Fatalf("allow set should restrict to b, got %+v", hits)
	}
	if hits, err = idx.SearchShared(spec, 10, map[string]struct{}{}); err != nil || hits != nil {
		t.Fatalf("empty allow should return nil, got %+v (%v)", hits, err)
	}
	if _, err = idx.SearchShared(SharedSpec{}, 10, nil); err == nil {
		t.Fatal("no positive terms: expected error")
	}
}
