package embedvec

import (
	"fmt"
	"math"
)

// Defaults for shared-concept ("must match all") multi-example queries.
const (
	// DefaultSharedBeta is the strength (0..1) with which the directions along
	// which the positive examples DIFFER from each other are attenuated before
	// scoring. The examples' differences span a small subspace (rank ≤ m−1);
	// down-weighting it focuses the score on what the examples have in common.
	// Keep it gentle: the same projection that damps a candidate's irrelevant
	// content also deletes what distinguishes an example-lookalike from the
	// pure shared concept, so a large β re-inflates lookalike scores until (at
	// β=1) an exact copy of ONE example ties the shared concept itself —
	// defeating the dispersion penalty. 0.25 measurably damps variation
	// content while preserving concept-first ordering even with 2 examples.
	DefaultSharedBeta = 0.25
	// DefaultSharedLambda scales the dispersion penalty in the aggregate
	// mean − λ·std over the per-example scores. λ=0 is plain averaging (the
	// centroid ranking); λ=1 makes the two-example case exactly min(s₁,s₂) and
	// approximates a soft minimum for larger sets: a candidate must score well
	// against EVERY example, not spectacularly against one.
	DefaultSharedLambda = 1.0
)

// SharedQuery scores candidates against SEVERAL positive examples at once with
// intersection ("must match all") semantics, unlike Combine, which averages the
// examples into one centroid vector. Because the dot product is linear, the
// centroid ranking equals the MEAN of the per-example cosines — it rewards a
// candidate extremely similar to one example as much as one moderately similar
// to all of them. SharedQuery instead:
//
//  1. attenuates the subspace spanned by the examples' mutual differences
//     (strength beta), so features the examples disagree on stop mattering;
//  2. aggregates the per-example cosines as weighted mean − lambda·std, so
//     dispersion across examples is penalized (a soft minimum);
//  3. subtracts each negative term's cosine scaled by its weight, matching the
//     steer-away semantics negative nodes have in blend mode.
//
// All vectors must share one embedding space. Inputs are normalized internally;
// candidates passed to Score must already be unit vectors.
type SharedQuery struct {
	pos     [][]float32 // unit positive examples
	posW    []float32   // per-example weights, normalized to sum 1
	neg     [][]float32 // unit negative terms
	negW    []float32   // negative weight magnitudes (not normalized)
	basis   [][]float32 // orthonormal basis of the examples' variation subspace
	gamma   float32     // 2β−β²: dot-product shrink factor of P=I−β·BBᵀ
	posNorm []float32   // ‖P·posᵢ‖ per example
	qb      [][]float32 // qb[i][j] = ⟨posᵢ, basisⱼ⟩
	lambda  float32
}

// NewSharedQuery builds a shared-concept scorer. pos must be non-empty; posW
// is per-example weight magnitudes (nil = equal). neg/negW (parallel, may be
// empty) are steer-away terms. beta is clamped to [0,1], lambda to ≥0; pass
// DefaultSharedBeta / DefaultSharedLambda absent a reason not to.
func NewSharedQuery(pos [][]float32, posW []float32, neg [][]float32, negW []float32, beta, lambda float32) (*SharedQuery, error) {
	if len(pos) == 0 {
		return nil, fmt.Errorf("embedvec: shared query has no positive terms")
	}
	if posW != nil && len(posW) != len(pos) {
		return nil, fmt.Errorf("embedvec: shared query: %d positives but %d weights", len(pos), len(posW))
	}
	if len(neg) != len(negW) {
		return nil, fmt.Errorf("embedvec: shared query: %d negatives but %d weights", len(neg), len(negW))
	}
	if beta < 0 {
		beta = 0
	}
	if beta > 1 {
		beta = 1
	}
	if lambda < 0 {
		lambda = 0
	}
	dim := len(pos[0])
	sq := &SharedQuery{
		pos:    make([][]float32, len(pos)),
		posW:   make([]float32, len(pos)),
		gamma:  beta * (2 - beta),
		lambda: lambda,
	}
	var wsum float64
	for i, v := range pos {
		if len(v) != dim {
			return nil, fmt.Errorf("embedvec: shared query length mismatch: %d vs %d", len(v), dim)
		}
		sq.pos[i] = Normalize(v)
		w := float32(1)
		if posW != nil {
			w = posW[i]
		}
		if w <= 0 {
			return nil, fmt.Errorf("embedvec: shared query positive weight must be > 0, got %g", w)
		}
		sq.posW[i] = w
		wsum += float64(w)
	}
	for i := range sq.posW {
		sq.posW[i] = float32(float64(sq.posW[i]) / wsum)
	}
	for i, v := range neg {
		if len(v) != dim {
			return nil, fmt.Errorf("embedvec: shared query length mismatch: %d vs %d", len(v), dim)
		}
		sq.neg = append(sq.neg, Normalize(v))
		w := negW[i]
		if w < 0 {
			w = -w
		}
		sq.negW = append(sq.negW, w)
	}

	// Variation subspace: orthonormalize the examples' offsets from their mean
	// (modified Gram-Schmidt, dropping near-zero residuals). Rank ≤ m−1.
	if beta > 0 && len(sq.pos) > 1 {
		mean := make([]float32, dim)
		for _, v := range sq.pos {
			for d, x := range v {
				mean[d] += x / float32(len(sq.pos))
			}
		}
		for _, v := range sq.pos {
			diff := make([]float32, dim)
			for d := range diff {
				diff[d] = v[d] - mean[d]
			}
			for _, b := range sq.basis {
				proj := Cosine(b, diff)
				for d := range diff {
					diff[d] -= proj * b[d]
				}
			}
			var n2 float64
			for _, x := range diff {
				n2 += float64(x) * float64(x)
			}
			if n2 < 1e-8 {
				continue
			}
			sq.basis = append(sq.basis, Normalize(diff))
		}
	}

	// Per-example projections onto the basis and projected norms ‖P·posᵢ‖.
	sq.qb = make([][]float32, len(sq.pos))
	sq.posNorm = make([]float32, len(sq.pos))
	for i, q := range sq.pos {
		sq.qb[i] = make([]float32, len(sq.basis))
		var b2 float64
		for j, b := range sq.basis {
			d := Cosine(q, b)
			sq.qb[i][j] = d
			b2 += float64(d) * float64(d)
		}
		n2 := 1 - float64(sq.gamma)*b2
		if n2 < 1e-6 {
			n2 = 1e-6
		}
		sq.posNorm[i] = float32(math.Sqrt(n2))
	}
	return sq, nil
}

// Score returns the shared-concept score of unit candidate v: weighted
// mean − λ·std of the per-example projected cosines, minus the weighted
// negative-term cosines. Higher is better; range is roughly [-1−Σnegw, 1].
// One-shot convenience — scoring many candidates should reuse one Scorer().
func (sq *SharedQuery) Score(v []float32) float32 {
	return sq.Scorer()(v)
}

// Scorer returns a scoring closure with private scratch buffers, so scan
// workers can score millions of candidates without per-candidate allocations.
// The closure is NOT safe for concurrent use; create one per goroutine.
func (sq *SharedQuery) Scorer() func(v []float32) float32 {
	vb := make([]float32, len(sq.basis))
	scores := make([]float64, len(sq.pos))
	return func(v []float32) float32 {
		// Candidate's components along the variation basis, and its norm after
		// the projection P (‖P·v‖² = 1 − γ·Σⱼ⟨bⱼ,v⟩² for unit v).
		var b2 float64
		for j, b := range sq.basis {
			d := Cosine(b, v)
			vb[j] = d
			b2 += float64(d) * float64(d)
		}
		vn2 := 1 - float64(sq.gamma)*b2
		if vn2 < 1e-6 {
			vn2 = 1e-6
		}
		vn := math.Sqrt(vn2)

		// Per-example projected cosines: ⟨P·q,P·v⟩ = ⟨q,v⟩ − γ·Σⱼ⟨q,bⱼ⟩⟨bⱼ,v⟩.
		var mean, m2 float64
		for i, q := range sq.pos {
			s := float64(Cosine(q, v))
			for j := range sq.basis {
				s -= float64(sq.gamma) * float64(sq.qb[i][j]) * float64(vb[j])
			}
			s /= float64(sq.posNorm[i]) * vn
			scores[i] = s
			mean += float64(sq.posW[i]) * s
		}
		for i := range scores {
			d := scores[i] - mean
			m2 += float64(sq.posW[i]) * d * d
		}
		out := mean - float64(sq.lambda)*math.Sqrt(m2)

		for i, n := range sq.neg {
			out -= float64(sq.negW[i]) * float64(Cosine(n, v))
		}
		return float32(out)
	}
}
