package embedindex

import (
	"testing"

	"github.com/stevecastle/shrike/embedvec"
)

// crossModalQuery returns a unit vector that is orthogonal to the common
// direction the seeded library shares (dim 0) but leans toward one item's
// distinguishing component (dim 1). This models a SigLIP TEXT vector: it lives
// outside the stored image cone, so the library mean carries no information
// about it, yet it still ranks items by genuine similarity.
func crossModalQuery(dim int) []float32 {
	q := make([]float32, dim)
	q[1] = 1
	return q
}

// TestScorePlainBypassesCentering pins the fix for the text-search regression:
// a centered index asked for ScorePlain must produce exactly the plain-cosine
// ranking and scores of an uncentered index over the same vectors.
func TestScorePlainBypassesCentering(t *testing.T) {
	const dim, n = 8, MinCenterCount * 4
	plain, centered := New(), NewCentered()
	seedCommonComponent(plain, n, dim, 4)
	seedCommonComponent(centered, n, dim, 4)
	q := crossModalQuery(dim)

	want := plain.Search(q, 10)
	got := centered.SearchScored(q, 10, nil, ScorePlain)
	if len(want) != len(got) {
		t.Fatalf("hit count mismatch: %d vs %d", len(want), len(got))
	}
	for i := range want {
		if want[i].Path != got[i].Path || want[i].Score != got[i].Score {
			t.Fatalf("hit %d: ScorePlain %+v != uncentered %+v", i, got[i], want[i])
		}
	}
	// And the default mode must still center, or the flag would be a no-op and
	// this test would pass for the wrong reason.
	if def := centered.SearchScored(q, 10, nil, ScoreDefault); def[0].Score == got[0].Score {
		t.Fatalf("ScoreDefault scored plain cosine %v — centering is not active", def[0].Score)
	}
}

// TestScorePlainRanksCrossModalQueryCorrectly is the behavioural half: with a
// query drawn from OUTSIDE the stored distribution, centering ranks by how
// atypical each item is rather than by similarity to the query, so the true
// nearest neighbour is not first. ScorePlain puts it back on top.
func TestScorePlainRanksCrossModalQueryCorrectly(t *testing.T) {
	const dim = 8
	idx := NewCentered()
	// A library sharing a strong "is a photo" component on dim 0...
	for i := 0; i < MinCenterCount*4; i++ {
		v := make([]float32, dim)
		v[0] = 1
		v[2+i%4] = 0.3
		idx.Add(itemName(i), v)
	}
	// ...plus ONE item that genuinely matches the cross-modal query's direction
	// while still carrying the common component, so plain cosine prefers it but
	// its distance from the mean does not stand out.
	match := make([]float32, dim)
	match[0] = 1
	match[1] = 0.6
	idx.Add("match", match)

	q := crossModalQuery(dim)
	if hits := idx.SearchScored(q, 1, nil, ScorePlain); hits[0].Path != "match" {
		t.Fatalf("ScorePlain should rank the true nearest neighbour first, got %+v", hits[0])
	}
	// Sanity: plain cosine really does prefer "match" — the assertion above is
	// about scoring mode, not about the fixture being degenerate.
	if got := embedvec.Cosine(embedvec.Normalize(q), embedvec.Normalize(match)); got <= 0 {
		t.Fatalf("fixture broken: cos(query, match) = %v", got)
	}
}

// TestSearchSharedScoredPlainBypassesCentering is the shared-concept analogue:
// a multi-example query containing text must not be centered either.
func TestSearchSharedScoredPlainBypassesCentering(t *testing.T) {
	const dim, n = 8, MinCenterCount * 4
	plain, centered := New(), NewCentered()
	seedCommonComponent(plain, n, dim, 4)
	seedCommonComponent(centered, n, dim, 4)
	spec := SharedSpec{Pos: [][]float32{crossModalQuery(dim), {1, 0.4, 0, 0, 0, 0, 0, 0}}}

	want, err := plain.SearchShared(spec, 10, nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := centered.SearchSharedScored(spec, 10, nil, ScorePlain)
	if err != nil {
		t.Fatal(err)
	}
	if len(want) != len(got) {
		t.Fatalf("hit count mismatch: %d vs %d", len(want), len(got))
	}
	for i := range want {
		if want[i].Path != got[i].Path || want[i].Score != got[i].Score {
			t.Fatalf("hit %d: ScorePlain %+v != uncentered %+v", i, got[i], want[i])
		}
	}
}

// TestSearchScoredHonoursAllowSet guards the merged Search/SearchFiltered
// implementation: the allow set still restricts results in both modes.
func TestSearchScoredHonoursAllowSet(t *testing.T) {
	const dim, n = 8, MinCenterCount * 4
	idx := NewCentered()
	seedCommonComponent(idx, n, dim, 4)
	allow := map[string]struct{}{"item-7": {}, "item-9": {}}
	for _, s := range []Scoring{ScoreDefault, ScorePlain} {
		hits := idx.SearchScored(crossModalQuery(dim), 10, allow, s)
		if len(hits) != 2 {
			t.Fatalf("scoring %v: got %d hits, want 2", s, len(hits))
		}
		for _, h := range hits {
			if _, ok := allow[h.Path]; !ok {
				t.Fatalf("scoring %v: %q is outside the allow set", s, h.Path)
			}
		}
	}
}

func itemName(i int) string {
	return "bg-" + string(rune('a'+i%26)) + string(rune('a'+(i/26)%26)) + string(rune('a'+(i/676)%26))
}
