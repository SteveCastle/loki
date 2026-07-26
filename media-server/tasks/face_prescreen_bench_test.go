package tasks

import (
	"context"
	"math/rand"
	"testing"

	"github.com/stevecastle/shrike/embedvec"
)

// BenchmarkPrescreenCandidates measures the pairwise sweep that dominates a
// large clustering run. It is bandwidth-bound rather than compute-bound: the
// working set is a tile of column vectors, so the tile size and the inner
// kernel's register reuse matter more than the raw arithmetic.
func BenchmarkPrescreenCandidates(b *testing.B) {
	rng := rand.New(rand.NewSource(5))
	faces := randFaces(20000, rng)
	for i := range faces {
		faces[i].Vec = randDimVec(rng, 512)
	}
	// Production packs before sweeping (see clusterFaces); benchmark the same
	// memory layout or this measures allocation scatter instead of the kernel.
	arena := packFaceVectors(faces)
	_ = arena
	b.ResetTimer()
	for range b.N {
		prescreenCandidates(context.Background(), faces, 0.47, nil)
	}
}

func randDimVec(rng *rand.Rand, dims int) []float32 {
	v := make([]float32, dims)
	for k := range v {
		v[k] = float32(rng.NormFloat64())
	}
	return embedvec.Normalize(v)
}
