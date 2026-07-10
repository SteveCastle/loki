package tasks

import (
	"math/rand"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
)

// randFaces builds n faces with random 64-dim unit vectors; every kth face is
// a near-duplicate of the previous one so similarities straddle the threshold.
func randFaces(n int, rng *rand.Rand) []media.Face {
	faces := make([]media.Face, n)
	for i := range faces {
		vec := make([]float32, 64)
		if i%5 == 4 {
			copy(vec, faces[i-1].Vec)
			vec[0] += 0.05
		} else {
			for k := range vec {
				vec[k] = float32(rng.NormFloat64())
			}
		}
		faces[i] = media.Face{ID: int64(i + 1), Vec: embedvec.Normalize(vec), Score: 0.9}
	}
	return faces
}

// dotf must agree with CosineSim on unit vectors and honor its
// mismatched-length contract.
func TestDotfMatchesCosineSimOnUnitVectors(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for _, f := range randFaces(200, rng) {
		for _, g := range randFaces(3, rng) {
			want := embedvec.CosineSim(f.Vec, g.Vec)
			got := dotf(f.Vec, g.Vec)
			if diff := got - want; diff > 1e-5 || diff < -1e-5 {
				t.Fatalf("dotf = %v, CosineSim = %v", got, want)
			}
		}
	}
	if dotf([]float32{1, 0}, []float32{1}) != 0 {
		t.Fatal("mismatched lengths must score 0, like CosineSim")
	}
}

// The prescreen must say exactly "some earlier face is ≥ threshold similar".
func TestPrescreenJoinableMatchesBruteForce(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	faces := randFaces(900, rng) // several row tiles' worth
	const threshold = 0.6
	got := prescreenJoinable(faces, threshold)
	for i := range faces {
		want := false
		for j := 0; j < i; j++ {
			if dotf(faces[i].Vec, faces[j].Vec) >= threshold {
				want = true
				break
			}
		}
		if got[i] != want {
			t.Fatalf("face %d: prescreen = %v, brute force = %v", i, got[i], want)
		}
	}
}

// The chunked parallel cluster scan must pick the same cluster (including the
// lowest-index-wins tie-break) as one serial full-range scan.
func TestBestClusterParallelMatchesSerial(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	memberPool := randFaces(5000, rng)
	clusters := make([]faceCluster, len(memberPool))
	for i, m := range memberPool {
		sum := make([]float32, len(m.Vec))
		copy(sum, m.Vec)
		clusters[i] = faceCluster{sum: sum, members: []media.Face{m}}
	}
	// Duplicate an early cluster at the end so a tie exists across chunks.
	dup := faceCluster{sum: clusters[3].sum, members: clusters[3].members}
	clusters = append(clusters, dup)

	queries := randFaces(50, rng)
	queries = append(queries, clusters[3].members[0]) // exact tie hit
	for qi, q := range queries {
		cl := map[int64]bool{memberPool[10].ID: true} // exercise cannot-links too
		serialIdx, serialScore := bestClusterIn(q, cl, clusters, 0, len(clusters))
		parIdx, parScore := bestCluster(q, cl, clusters, 16)
		if serialIdx != parIdx || serialScore != parScore {
			t.Fatalf("query %d: serial = (%d, %v), parallel = (%d, %v)",
				qi, serialIdx, serialScore, parIdx, parScore)
		}
	}
}
