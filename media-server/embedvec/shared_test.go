package embedvec

import (
	"math"
	"testing"
)

func almostEq(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

// TestSharedQueryTwoExamplesIsMin pins the λ=1 identity: with two equal-weight
// examples and no projection (β=0), mean − std over {s₁,s₂} is exactly
// min(s₁,s₂) — the pure intersection semantics.
func TestSharedQueryTwoExamplesIsMin(t *testing.T) {
	q1 := Normalize([]float32{1, 0.5, 0})
	q2 := Normalize([]float32{1, -0.5, 0.2})
	sq, err := NewSharedQuery([][]float32{q1, q2}, nil, nil, nil, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, cand := range [][]float32{
		Normalize([]float32{1, 0, 0}),
		Normalize([]float32{0, 1, 0}),
		Normalize([]float32{0.3, -0.7, 0.6}),
		q1,
	} {
		s1, s2 := Cosine(q1, cand), Cosine(q2, cand)
		want := s1
		if s2 < s1 {
			want = s2
		}
		if got := sq.Score(cand); !almostEq(got, want, 1e-5) {
			t.Fatalf("score %v: got %v, want min(%v,%v)=%v", cand, got, s1, s2, want)
		}
	}
}

// TestSharedQueryFullProjectionCollapsesVariation pins β=1: the examples'
// difference direction is removed entirely, so an example itself scores the
// same 1.0 as the pure shared direction — everything the examples disagreed
// on has stopped mattering.
func TestSharedQueryFullProjectionCollapsesVariation(t *testing.T) {
	q1 := Normalize([]float32{1, 1, 0})  // shared x, differs +y
	q2 := Normalize([]float32{1, -1, 0}) // shared x, differs −y
	sq, err := NewSharedQuery([][]float32{q1, q2}, nil, nil, nil, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	shared := []float32{1, 0, 0}
	if got := sq.Score(shared); !almostEq(got, 1, 1e-5) {
		t.Fatalf("shared direction: got %v, want 1", got)
	}
	if got := sq.Score(q1); !almostEq(got, 1, 1e-5) {
		t.Fatalf("example itself under full projection: got %v, want 1", got)
	}
	if got := sq.Score([]float32{0, 0, 1}); !almostEq(got, 0, 1e-5) {
		t.Fatalf("orthogonal direction: got %v, want 0", got)
	}
}

// TestSharedQueryFootballScenario is the motivating case: several examples
// share one concept and each carries its own distractor. The pure shared
// candidate must beat a candidate identical to ONE example, which must beat a
// candidate matching only that example's distractor.
func TestSharedQueryFootballScenario(t *testing.T) {
	// dim 0 = "football"; dims 1-3 = per-example scenery.
	q1 := Normalize([]float32{1, 1, 0, 0})
	q2 := Normalize([]float32{1, 0, 1, 0})
	q3 := Normalize([]float32{1, 0, 0, 1})
	sq, err := NewSharedQuery([][]float32{q1, q2, q3}, nil, nil, nil, DefaultSharedBeta, DefaultSharedLambda)
	if err != nil {
		t.Fatal(err)
	}
	football := sq.Score([]float32{1, 0, 0, 0})
	oneExample := sq.Score(q1)
	sceneryOnly := sq.Score([]float32{0, 1, 0, 0})
	if !(football > oneExample) {
		t.Fatalf("football %v should beat one-example %v", football, oneExample)
	}
	if !(oneExample > sceneryOnly) {
		t.Fatalf("one-example %v should beat scenery-only %v", oneExample, sceneryOnly)
	}
}

// TestSharedQueryNegativePenalty verifies negatives subtract their weighted
// cosine: a candidate aligned with the negative term drops below an otherwise
// equal candidate that isn't.
func TestSharedQueryNegativePenalty(t *testing.T) {
	pos := [][]float32{Normalize([]float32{1, 0, 0})}
	neg := [][]float32{Normalize([]float32{0, 1, 0})}
	sq, err := NewSharedQuery(pos, nil, neg, []float32{0.5}, 0, 1)
	if err != nil {
		t.Fatal(err)
	}
	clean := Normalize([]float32{1, 0, 0.3})
	tainted := Normalize([]float32{1, 0.3, 0})
	if !(sq.Score(clean) > sq.Score(tainted)) {
		t.Fatalf("clean %v should beat tainted %v", sq.Score(clean), sq.Score(tainted))
	}
	// Exact form for a fully-aligned candidate: cos(pos,v) − 0.5·cos(neg,v).
	v := Normalize([]float32{1, 1, 0})
	want := Cosine(pos[0], v) - 0.5*Cosine(neg[0], v)
	if got := sq.Score(v); !almostEq(got, want, 1e-5) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// TestSharedQueryWeights verifies positive weights shift the aggregate toward
// the heavier example.
func TestSharedQueryWeights(t *testing.T) {
	q1 := Normalize([]float32{1, 0, 0})
	q2 := Normalize([]float32{0, 1, 0})
	heavy1, err := NewSharedQuery([][]float32{q1, q2}, []float32{1, 0.25}, nil, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	even, err := NewSharedQuery([][]float32{q1, q2}, nil, nil, nil, 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	cand := q1
	if !(heavy1.Score(cand) > even.Score(cand)) {
		t.Fatalf("q1-weighted %v should exceed even %v for candidate q1",
			heavy1.Score(cand), even.Score(cand))
	}
}

// TestSharedQuerySinglePositiveIsCosine verifies degradation: one positive,
// no negatives, any β/λ — plain cosine similarity.
func TestSharedQuerySinglePositiveIsCosine(t *testing.T) {
	q := Normalize([]float32{0.3, 0.7, 0.2})
	sq, err := NewSharedQuery([][]float32{q}, nil, nil, nil, DefaultSharedBeta, DefaultSharedLambda)
	if err != nil {
		t.Fatal(err)
	}
	for _, cand := range [][]float32{
		Normalize([]float32{1, 0, 0}),
		Normalize([]float32{-0.2, 0.5, 0.9}),
	} {
		if got, want := sq.Score(cand), Cosine(q, cand); !almostEq(got, want, 1e-5) {
			t.Fatalf("got %v, want cosine %v", got, want)
		}
	}
}

// TestSharedQueryDuplicateExamples verifies duplicate (zero-variation) inputs
// don't blow up: the difference vectors are ~zero and must be dropped from
// the basis rather than normalized into garbage.
func TestSharedQueryDuplicateExamples(t *testing.T) {
	q := Normalize([]float32{1, 2, 3})
	sq, err := NewSharedQuery([][]float32{q, q, q}, nil, nil, nil, DefaultSharedBeta, DefaultSharedLambda)
	if err != nil {
		t.Fatal(err)
	}
	got := sq.Score(q)
	if math.IsNaN(float64(got)) || !almostEq(got, 1, 1e-4) {
		t.Fatalf("identical examples, identical candidate: got %v, want 1", got)
	}
}

// TestSharedQueryErrors pins the constructor's validation.
func TestSharedQueryErrors(t *testing.T) {
	if _, err := NewSharedQuery(nil, nil, nil, nil, 0.5, 1); err == nil {
		t.Fatal("no positives: expected error")
	}
	v := []float32{1, 0}
	if _, err := NewSharedQuery([][]float32{v}, []float32{0}, nil, nil, 0.5, 1); err == nil {
		t.Fatal("zero positive weight: expected error")
	}
	if _, err := NewSharedQuery([][]float32{v, {1, 0, 0}}, nil, nil, nil, 0.5, 1); err == nil {
		t.Fatal("length mismatch: expected error")
	}
	if _, err := NewSharedQuery([][]float32{v}, nil, [][]float32{{1, 0}}, nil, 0.5, 1); err == nil {
		t.Fatal("negatives without weights: expected error")
	}
}
