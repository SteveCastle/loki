package tasks

import (
	"context"
	"math/rand"
	"testing"

	"github.com/stevecastle/shrike/media"
)

// scoreAgainstSeedsRef is the one-candidate-at-a-time implementation the
// grouped kernel replaced. It is the reference for TestScoreAgainstSeeds-
// GroupedMatchesReference: the grouped version reorders only WHEN dot products
// are computed, never which seed contributes what, so the two must agree
// exactly — including on the curation constraints, which is the whole reason
// this path is worth pinning.
func scoreAgainstSeedsRef(unassigned []media.Face, seeds []seed, aggs map[int64]*personAgg, threshold float32, vetoes, cannot pairSet) []int64 {
	matches := make([]int64, len(unassigned))
	if len(unassigned) == 0 || len(seeds) == 0 {
		return matches
	}
	userMeanFloor := threshold - meanJoinSlack
	autoMeanFloor := threshold
	scores := map[int64]personScore{}
	forbidden := map[int64]bool{}
	for i := range unassigned {
		clear(scores)
		clear(forbidden)
		cl := cannot[unassigned[i].ID]
		for pid := range vetoes[unassigned[i].ID] {
			forbidden[pid] = true
		}
		for _, s := range seeds {
			if cl[s.id] {
				forbidden[s.personID] = true
				continue
			}
			if forbidden[s.personID] {
				continue
			}
			sc := dotf(unassigned[i].Vec, s.vec)
			if sc < threshold-corroborationSlack {
				continue
			}
			ps := scores[s.personID]
			if sc > ps.best {
				ps.best = sc
			}
			if s.user {
				ps.count += userSeedWeight
			} else {
				ps.count++
			}
			scores[s.personID] = ps
		}
		var bestPerson int64
		var best personScore
		for pid, ps := range scores {
			if forbidden[pid] || !acceptJoin(ps, threshold) {
				continue
			}
			a := aggs[pid]
			if a == nil {
				continue
			}
			mean := dot32(unassigned[i].Vec, a.sum) / a.n
			floor := autoMeanFloor
			if a.userN > 0 {
				mean = dot32(unassigned[i].Vec, a.userSum) / a.userN
				floor = userMeanFloor
			}
			if mean < floor {
				continue
			}
			if bestPerson == 0 || effectiveScore(ps) > effectiveScore(best) {
				bestPerson, best = pid, ps
			}
		}
		matches[i] = bestPerson
	}
	return matches
}

func TestScoreAgainstSeedsGroupedMatchesReference(t *testing.T) {
	cases := []struct {
		name                       string
		nCand, nSeed, people, dims int
		threshold                  float32
		vetoes, cannotLinks, users bool
	}{
		{"plain", 200, 500, 10, 64, 0.42, false, false, false},
		{"candidate count not a multiple of the group width", 203, 501, 7, 64, 0.42, false, false, false},
		{"seed count straddling the chunk boundary", 64, 257, 5, 64, 0.42, false, false, false},
		{"with user seeds (weighted corroboration + anchored mean)", 200, 500, 8, 64, 0.42, false, false, true},
		{"with vetoes", 200, 500, 8, 64, 0.42, true, false, true},
		{"with cannot-links", 200, 500, 8, 64, 0.42, false, true, true},
		{"everything at once, low threshold", 300, 800, 12, 32, 0.20, true, true, true},
		{"single candidate", 1, 300, 4, 64, 0.42, true, true, true},
		{"fewer candidates than the group width", 3, 300, 4, 64, 0.42, true, true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(len(tc.name)) * 31))
			// Real identity structure: each person's seeds sit tightly around
			// its own center. Without that the mean-similarity guard rejects
			// every join and the comparison is all zeros on both sides.
			centers := make([][]float32, tc.people)
			for i := range centers {
				centers[i] = randDimVec(rng, tc.dims)
			}
			nearCenter := func(c []float32, spread float32) []float32 {
				v := make([]float32, tc.dims)
				for k := range v {
					v[k] = c[k] + float32(rng.NormFloat64())*spread
				}
				return embedvecNormalize(v)
			}
			seeds := make([]seed, tc.nSeed)
			for i := range seeds {
				pid := i%tc.people + 1
				seeds[i] = seed{
					id:       int64(i + 1),
					vec:      nearCenter(centers[pid-1], 0.02),
					personID: int64(pid),
					user:     tc.users && i%5 == 0,
				}
			}
			cand := make([]media.Face, tc.nCand)
			for i := range cand {
				vec := randDimVec(rng, tc.dims) // junk by default
				if i%3 == 0 {
					vec = nearCenter(centers[i%tc.people], 0.03)
				}
				cand[i] = media.Face{ID: int64(1_000_000 + i), Vec: vec}
			}

			vetoes := pairSet{}
			cannot := pairSet{}
			// Offset off index 0 so candidate 0 stays joinable — the tiny cases
			// have no other candidate that can prove the paths agree.
			if tc.vetoes {
				for i := 1; i < len(cand); i += 4 {
					vetoes[cand[i].ID] = map[int64]bool{int64(i%tc.people + 1): true}
				}
			}
			if tc.cannotLinks {
				for i := 2; i < len(cand); i += 6 {
					cannot[cand[i].ID] = map[int64]bool{seeds[(i*3)%len(seeds)].id: true}
				}
			}
			aggs := buildPersonAggs(seeds)

			want := scoreAgainstSeedsRef(cand, seeds, aggs, tc.threshold, vetoes, cannot)
			got := scoreAgainstSeeds(context.Background(), cand, seeds, aggs, tc.threshold, vetoes, cannot)

			joins := 0
			for i := range want {
				if want[i] != got[i] {
					t.Fatalf("candidate %d: grouped = %d, reference = %d", i, got[i], want[i])
				}
				if want[i] != 0 {
					joins++
				}
			}
			if joins == 0 {
				t.Fatalf("no candidate joined anything — the case proves nothing")
			}
		})
	}
}

// dot4 must agree with four separate dotf calls, including on the mismatched
// -length contract (score 0) that keeps foreign-model vectors from matching.
func TestDot4MatchesScalarDot(t *testing.T) {
	rng := rand.New(rand.NewSource(17))
	for _, dims := range []int{1, 2, 3, 7, 8, 63, 64, 512} {
		b := randDimVec(rng, dims)
		a := [4][]float32{
			randDimVec(rng, dims), randDimVec(rng, dims),
			randDimVec(rng, dims), randDimVec(rng, dims),
		}
		s0, s1, s2, s3 := dot4(a[0], a[1], a[2], a[3], b)
		got := [4]float32{s0, s1, s2, s3}
		for i, want := range [4]float32{
			dotf(a[0], b), dotf(a[1], b), dotf(a[2], b), dotf(a[3], b),
		} {
			// Different summation order, so allow float32 rounding.
			if d := got[i] - want; d > 1e-5 || d < -1e-5 {
				t.Fatalf("dims=%d row %d: dot4 = %v, dotf = %v", dims, i, got[i], want)
			}
		}
	}
	// A row of the wrong width scores 0 and must not corrupt its neighbours.
	b := randDimVec(rng, 64)
	good := randDimVec(rng, 64)
	short := randDimVec(rng, 8)
	s0, s1, s2, s3 := dot4(good, short, good, good, b)
	if s1 != 0 {
		t.Fatalf("mismatched row scored %v, want 0", s1)
	}
	want := dotf(good, b)
	for i, got := range []float32{s0, s2, s3} {
		if got != want {
			t.Fatalf("row %d beside a mismatched row = %v, want %v", i, got, want)
		}
	}
}
