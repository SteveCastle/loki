package embedvec

import (
	"math"
	"runtime"
	"sync"
)

// ProjectPCA3 reduces high-dimensional vectors to 3D via principal component
// analysis, for visualizing the embedding space. It returns one [x,y,z] per
// input (uniformly scaled so the cloud fits in [-1,1]^3, preserving relative
// distances between axes) and the fraction of total variance each component
// explains.
//
// Components are found by power iteration with Gram-Schmidt deflation against
// a deterministic start vector, so repeated calls over the same data produce
// the same projection (no jitter between page reloads). The covariance
// matrix is never materialized: each iteration computes X^T(Xv) with
// on-the-fly mean-centering, parallelized across rows.
//
// All inputs must share one dimension; the caller filters mismatched rows.
func ProjectPCA3(vecs [][]float32) (coords [][3]float32, variance [3]float64) {
	n := len(vecs)
	coords = make([][3]float32, n)
	if n == 0 {
		return coords, variance
	}
	d := len(vecs[0])
	if d == 0 {
		return coords, variance
	}

	// Mean vector (float64 accumulation for stability).
	mean := make([]float64, d)
	for _, v := range vecs {
		for j, x := range v {
			mean[j] += float64(x)
		}
	}
	for j := range mean {
		mean[j] /= float64(n)
	}

	// Total variance = (1/n) Σ |x_i - mean|², the denominator for the
	// explained-variance fractions.
	var totalVar float64
	for _, v := range vecs {
		for j, x := range v {
			dx := float64(x) - mean[j]
			totalVar += dx * dx
		}
	}
	totalVar /= float64(n)
	if totalVar == 0 {
		return coords, variance // all points identical
	}

	comps := make([][]float64, 0, 3)
	eigen := make([]float64, 0, 3)
	for c := 0; c < 3; c++ {
		v := deterministicUnit(d, c)
		orthogonalize(v, comps)
		if normalize(v) == 0 {
			break
		}
		var lambda float64
		for iter := 0; iter < 100; iter++ {
			u := covTimes(vecs, mean, v)
			orthogonalize(u, comps)
			lambda = normalize(u)
			if lambda == 0 {
				break // no variance left in the orthogonal complement
			}
			done := math.Abs(dot(u, v)) > 1-1e-10
			copy(v, u)
			if done {
				break
			}
		}
		if lambda == 0 {
			break
		}
		comps = append(comps, v)
		eigen = append(eigen, lambda/float64(n))
	}

	// Project every (centered) point onto the components.
	meanDot := make([]float64, len(comps))
	for c, comp := range comps {
		meanDot[c] = dot(mean, comp)
	}
	maxAbs := 0.0
	for i, vec := range vecs {
		for c, comp := range comps {
			var s float64
			for j, x := range vec {
				s += float64(x) * comp[j]
			}
			s -= meanDot[c]
			coords[i][c] = float32(s)
			if a := math.Abs(s); a > maxAbs {
				maxAbs = a
			}
		}
	}
	// One uniform scale for all axes so relative spread between components
	// stays truthful (axis-independent scaling would exaggerate flat axes).
	if maxAbs > 0 {
		s := float32(1 / maxAbs)
		for i := range coords {
			coords[i][0] *= s
			coords[i][1] *= s
			coords[i][2] *= s
		}
	}
	for c := range eigen {
		variance[c] = eigen[c] / totalVar
	}
	return coords, variance
}

// covTimes returns X_c^T (X_c v) where X_c is the row-wise mean-centered data,
// without materializing X_c: y_i = <x_i,v> - <mean,v>, u = Σ y_i·x_i - (Σy_i)·mean.
// The row loop is parallelized across CPUs.
func covTimes(vecs [][]float32, mean, v []float64) []float64 {
	n, d := len(vecs), len(mean)
	mv := dot(mean, v)

	workers := runtime.NumCPU()
	if workers > n {
		workers = n
	}
	chunk := (n + workers - 1) / workers
	type partial struct {
		u    []float64
		ySum float64
	}
	// Partials are merged in fixed chunk order below — NOT completion order —
	// so float64 rounding is identical across runs and the projection is
	// deterministic.
	parts := make([]partial, (n+chunk-1)/chunk)
	var wg sync.WaitGroup
	for w, lo := 0, 0; lo < n; w, lo = w+1, lo+chunk {
		hi := lo + chunk
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			u := make([]float64, d)
			var ySum float64
			for i := lo; i < hi; i++ {
				row := vecs[i]
				var y float64
				for j, x := range row {
					y += float64(x) * v[j]
				}
				y -= mv
				ySum += y
				for j, x := range row {
					u[j] += y * float64(x)
				}
			}
			parts[w] = partial{u, ySum}
		}(w, lo, hi)
	}
	wg.Wait()

	u := make([]float64, d)
	var ySum float64
	for _, p := range parts {
		ySum += p.ySum
		for j, x := range p.u {
			u[j] += x
		}
	}
	for j := range u {
		u[j] -= ySum * mean[j]
	}
	return u
}

// deterministicUnit returns a reproducible pseudo-random direction for
// component c (a fixed LCG, no global rand), so projections are stable
// across calls and processes.
func deterministicUnit(d, c int) []float64 {
	v := make([]float64, d)
	state := uint64(0x9E3779B97F4A7C15 + uint64(c)*0xBF58476D1CE4E5B9)
	for j := range v {
		state = state*6364136223846793005 + 1442695040888963407
		v[j] = float64(int64(state>>11))/float64(1<<52) - 1 // roughly [-1,1)
	}
	return v
}

// orthogonalize removes from v its projection onto each vector in basis
// (assumed orthonormal).
func orthogonalize(v []float64, basis [][]float64) {
	for _, b := range basis {
		p := dot(v, b)
		for j := range v {
			v[j] -= p * b[j]
		}
	}
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// normalize scales v to unit length in place and returns its prior norm.
func normalize(v []float64) float64 {
	n := math.Sqrt(dot(v, v))
	if n == 0 {
		return 0
	}
	for j := range v {
		v[j] /= n
	}
	return n
}
