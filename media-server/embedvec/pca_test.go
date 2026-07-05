package embedvec

import (
	"math"
	"math/rand"
	"testing"
)

// TestProjectPCA3RecoversPlantedStructure embeds a 3D point cloud into 32
// dimensions via a random rotation plus tiny noise, then checks that the PCA
// projection preserves pairwise distance ORDER (the property the visualization
// relies on: near things stay near, far things stay far).
func TestProjectPCA3RecoversPlantedStructure(t *testing.T) {
	const n, d = 200, 32
	rng := rand.New(rand.NewSource(7))

	// Three orthonormal directions in d-space.
	basis := make([][]float64, 3)
	for c := range basis {
		v := make([]float64, d)
		for j := range v {
			v[j] = rng.NormFloat64()
		}
		orthogonalize(v, basis[:c])
		normalize(v)
		basis[c] = v
	}

	// Points with variance 9/4/1 along the three planted axes + 0.01 noise.
	latent := make([][3]float64, n)
	vecs := make([][]float32, n)
	for i := range vecs {
		latent[i] = [3]float64{3 * rng.NormFloat64(), 2 * rng.NormFloat64(), rng.NormFloat64()}
		row := make([]float32, d)
		for j := 0; j < d; j++ {
			x := latent[i][0]*basis[0][j] + latent[i][1]*basis[1][j] + latent[i][2]*basis[2][j]
			row[j] = float32(x + 0.01*rng.NormFloat64())
		}
		vecs[i] = row
	}

	coords, variance := ProjectPCA3(vecs)
	if len(coords) != n {
		t.Fatalf("expected %d coords, got %d", n, len(coords))
	}

	// Nearly all variance lives in the planted 3D subspace.
	total := variance[0] + variance[1] + variance[2]
	if total < 0.99 {
		t.Errorf("expected ~all variance explained by 3 components, got %v (sum %v)", variance, total)
	}
	if !(variance[0] >= variance[1] && variance[1] >= variance[2]) {
		t.Errorf("components not in descending variance order: %v", variance)
	}

	// Distance-order preservation on random triples: if a is much closer to b
	// than to c in latent space, the same must hold in the projection.
	dist3 := func(p, q [3]float64) float64 {
		dx, dy, dz := p[0]-q[0], p[1]-q[1], p[2]-q[2]
		return math.Sqrt(dx*dx + dy*dy + dz*dz)
	}
	distP := func(p, q [3]float32) float64 {
		dx := float64(p[0] - q[0])
		dy := float64(p[1] - q[1])
		dz := float64(p[2] - q[2])
		return math.Sqrt(dx*dx + dy*dy + dz*dz)
	}
	checked := 0
	for trial := 0; trial < 500; trial++ {
		a, b, c := rng.Intn(n), rng.Intn(n), rng.Intn(n)
		db, dc := dist3(latent[a], latent[b]), dist3(latent[a], latent[c])
		if db*2 >= dc { // only judge clear-cut cases (2x margin)
			continue
		}
		checked++
		if distP(coords[a], coords[b]) >= distP(coords[a], coords[c]) {
			t.Fatalf("distance order not preserved: latent %v<%v but projected %v>=%v",
				db, dc, distP(coords[a], coords[b]), distP(coords[a], coords[c]))
		}
	}
	if checked < 50 {
		t.Fatalf("too few clear-cut triples checked (%d) — test data degenerate", checked)
	}
}

// TestProjectPCA3Deterministic pins that two runs over the same data produce
// identical output (no global rand, no map-order dependence).
func TestProjectPCA3Deterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	vecs := make([][]float32, 100)
	for i := range vecs {
		row := make([]float32, 16)
		for j := range row {
			row[j] = float32(rng.NormFloat64())
		}
		vecs[i] = row
	}
	c1, v1 := ProjectPCA3(vecs)
	c2, v2 := ProjectPCA3(vecs)
	if v1 != v2 {
		t.Fatalf("variance differs between runs: %v vs %v", v1, v2)
	}
	for i := range c1 {
		if c1[i] != c2[i] {
			t.Fatalf("coords differ at %d: %v vs %v", i, c1[i], c2[i])
		}
	}
}

// TestProjectPCA3Bounds verifies output fits [-1,1]^3 and touches the bound.
func TestProjectPCA3Bounds(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	vecs := make([][]float32, 50)
	for i := range vecs {
		row := make([]float32, 8)
		for j := range row {
			row[j] = float32(rng.NormFloat64() * 100) // large scale must not matter
		}
		vecs[i] = row
	}
	coords, _ := ProjectPCA3(vecs)
	maxAbs := 0.0
	for _, c := range coords {
		for _, x := range c {
			if a := math.Abs(float64(x)); a > maxAbs {
				maxAbs = a
			}
			if math.Abs(float64(x)) > 1+1e-6 {
				t.Fatalf("coordinate out of bounds: %v", c)
			}
		}
	}
	if maxAbs < 0.999 {
		t.Errorf("expected cloud scaled to touch the unit bound, max abs %v", maxAbs)
	}
}

// TestProjectPCA3Degenerate covers empty input and identical points.
func TestProjectPCA3Degenerate(t *testing.T) {
	if c, _ := ProjectPCA3(nil); len(c) != 0 {
		t.Fatalf("expected empty output for nil input")
	}
	same := [][]float32{{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}
	coords, variance := ProjectPCA3(same)
	if len(coords) != 3 {
		t.Fatalf("expected 3 coords, got %d", len(coords))
	}
	for _, c := range coords {
		if c != ([3]float32{}) {
			t.Errorf("identical points must project to origin, got %v", c)
		}
	}
	if variance != ([3]float64{}) {
		t.Errorf("identical points have zero variance, got %v", variance)
	}
}
