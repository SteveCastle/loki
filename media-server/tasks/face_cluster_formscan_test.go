package tasks

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

// randClusteredFaces builds faces in `groups` tight identity groups plus junk,
// which gives the formation loop a real mix of joinable and unjoinable faces.
func randClusteredFaces(t *testing.T, rng *rand.Rand, groups, perGroup, junk, dims int) []media.Face {
	t.Helper()
	unit := func(raw []float64) []float32 {
		var norm float64
		for _, x := range raw {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		v := make([]float32, len(raw))
		for k, x := range raw {
			v[k] = float32(x / norm)
		}
		return v
	}
	randRaw := func() []float64 {
		raw := make([]float64, dims)
		for k := range raw {
			raw[k] = rng.NormFloat64()
		}
		return raw
	}

	var out []media.Face
	id := int64(1)
	add := func(v []float32) {
		out = append(out, media.Face{
			ID: id, MediaPath: fmt.Sprintf("f%d.jpg", id),
			Score: 0.5 + rng.Float64()*0.5, Vec: v,
		})
		id++
	}
	for g := range groups {
		center := randRaw()
		for range perGroup {
			raw := make([]float64, dims)
			// Spread wide enough that group members land at a range of
			// similarities, so some join and some don't.
			for k := range raw {
				raw[k] = center[k] + rng.NormFloat64()*float64(1+g%3)*0.55
			}
			add(unit(raw))
		}
	}
	for range junk {
		add(unit(randRaw()))
	}
	rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
	return out
}

// formGreedyFull is the pre-optimization formation loop: every face scans
// EVERY cluster. It is the reference the restricted scan has to reproduce.
func formGreedyFull(faces []media.Face, threshold float32, cannot pairSet) []faceCluster {
	var clusters []faceCluster
	for _, f := range faces {
		bestIdx, bestScore := bestClusterIn(f, cannot[f.ID], clusters, 0, len(clusters))
		if bestIdx >= 0 && bestScore >= threshold {
			c := &clusters[bestIdx]
			for k := range c.sum {
				c.sum[k] += f.Vec[k]
			}
			c.members = append(c.members, f)
		} else {
			clusters = append(clusters, newFaceCluster(f))
		}
	}
	return clusters
}

func clusterShape(clusters []faceCluster) []string {
	out := make([]string, 0, len(clusters))
	for _, c := range clusters {
		ids := make([]int64, 0, len(c.members))
		for _, m := range c.members {
			ids = append(ids, m.ID)
		}
		out = append(out, fmt.Sprint(ids))
	}
	return out
}

// The candidate-restricted formation scan must be EXACT: a cluster can only be
// accepted when its mean-to-members clears the threshold, a mean never exceeds
// the best single member match, and every such member is an earlier face the
// prescreen recorded — so clusters holding no candidate can only ever score
// below the gate. This test pins that reasoning against the full scan over
// many shapes, including the cannot-link path (which prunes clusters) and
// thresholds low enough to make the neighbour lists dense.
func TestRestrictedFormScanMatchesFullScan(t *testing.T) {
	cases := []struct {
		name                         string
		groups, perGroup, junk, dims int
		threshold                    float32
		withCannotLinks              bool
	}{
		{"sparse, mostly junk", 4, 6, 200, 64, 0.55, false},
		{"dense groups", 12, 25, 60, 64, 0.35, false},
		{"very low threshold (long candidate lists)", 6, 20, 80, 32, 0.15, false},
		{"high threshold (almost all singletons)", 8, 10, 150, 64, 0.85, false},
		{"with cannot-links pruning clusters", 10, 14, 90, 64, 0.35, true},
		{"tiny dims, heavy collisions", 20, 10, 50, 8, 0.6, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(len(tc.name))))
			faces := randClusteredFaces(t, rng, tc.groups, tc.perGroup, tc.junk, tc.dims)

			cannot := pairSet{}
			if tc.withCannotLinks {
				for i := 0; i < len(faces); i += 7 {
					a := faces[i].ID
					b := faces[(i*3+1)%len(faces)].ID
					if a == b {
						continue
					}
					if cannot[a] == nil {
						cannot[a] = map[int64]bool{}
					}
					cannot[a][b] = true
				}
			}

			want := formGreedyFull(faces, tc.threshold, cannot)
			got := formGreedyRestricted(t, faces, tc.threshold, cannot)

			wantShape, gotShape := clusterShape(want), clusterShape(got)
			if len(wantShape) != len(gotShape) {
				t.Fatalf("restricted scan produced %d clusters, full scan %d",
					len(gotShape), len(wantShape))
			}
			for i := range wantShape {
				if wantShape[i] != gotShape[i] {
					t.Fatalf("cluster %d: restricted = %s, full = %s", i, gotShape[i], wantShape[i])
				}
			}
		})
	}
}

// formGreedyRestricted mirrors the production loop's candidate handling so the
// comparison above exercises the real prescreen output.
func formGreedyRestricted(t *testing.T, faces []media.Face, threshold float32, cannot pairSet) []faceCluster {
	t.Helper()
	cand := prescreenCandidates(context.Background(), faces, threshold, nil)
	var clusters []faceCluster
	clusterOf := make([]int32, len(faces))
	scan := make([]int32, 0, maxFormCandidates)
	seenGen := make([]int32, len(faces)+1)
	var gen int32
	for fi, f := range faces {
		near, overflow := cand.near[fi], cand.overflow[fi]
		if len(near) == 0 && !overflow {
			clusterOf[fi] = int32(len(clusters))
			clusters = append(clusters, newFaceCluster(f))
			continue
		}
		var bestIdx int
		var bestScore float32
		if overflow {
			bestIdx, bestScore = bestCluster(f, cannot[f.ID], clusters, 4)
		} else {
			gen++
			scan = scan[:0]
			for _, j := range near {
				ci := clusterOf[j]
				if seenGen[ci] != gen {
					seenGen[ci] = gen
					scan = append(scan, ci)
				}
			}
			bestIdx, bestScore = bestClusterAmong(f, cannot[f.ID], clusters, scan)
		}
		if bestIdx >= 0 && bestScore >= threshold {
			c := &clusters[bestIdx]
			for k := range c.sum {
				c.sum[k] += f.Vec[k]
			}
			c.members = append(c.members, f)
			clusterOf[fi] = int32(bestIdx)
		} else {
			clusterOf[fi] = int32(len(clusters))
			clusters = append(clusters, newFaceCluster(f))
		}
	}
	return clusters
}

// bestClusterAmong must reproduce a full ascending scan's lowest-index
// tie-break even though candidates do not arrive in cluster order.
func TestBestClusterAmongTieBreakMatchesAscendingScan(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	pool := randFaces(40, rng)
	clusters := make([]faceCluster, len(pool))
	for i, m := range pool {
		clusters[i] = newFaceCluster(m)
	}
	// Duplicate cluster 2 at the end so an exact tie spans two indices.
	clusters = append(clusters, newFaceCluster(pool[2]))

	all := make([]int32, len(clusters))
	for i := range all {
		all[i] = int32(i)
	}
	shuffled := append([]int32(nil), all...)
	rng.Shuffle(len(shuffled), func(i, j int) { shuffled[i], shuffled[j] = shuffled[j], shuffled[i] })

	queries := append(randFaces(20, rng), pool[2])
	for qi, q := range queries {
		wantIdx, wantScore := bestClusterIn(q, nil, clusters, 0, len(clusters))
		gotIdx, gotScore := bestClusterAmong(q, nil, clusters, shuffled)
		if wantIdx != gotIdx || wantScore != gotScore {
			t.Fatalf("query %d: among = (%d, %v), ascending scan = (%d, %v)",
				qi, gotIdx, gotScore, wantIdx, wantScore)
		}
	}
}
