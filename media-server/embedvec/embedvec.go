// Package embedvec provides pure-Go serialization and similarity math for
// embedding vectors. It has no dependencies and no cgo, so it compiles into
// the CGO_ENABLED=0 server binary and is safe to use from handlers and tasks.
package embedvec

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Encode serializes a float32 slice to little-endian bytes for BLOB storage.
func Encode(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// Decode parses little-endian float32 bytes produced by Encode.
func Decode(b []byte) ([]float32, error) {
	if len(b)%4 != 0 {
		return nil, fmt.Errorf("embedvec: byte length %d is not a multiple of 4", len(b))
	}
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v, nil
}

// Normalize returns an L2-normalized copy. A zero vector is returned unchanged.
func Normalize(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		out := make([]float32, len(v))
		copy(out, v)
		return out
	}
	norm := math.Sqrt(sum)
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = float32(float64(x) / norm)
	}
	return out
}

// Blend returns the L2-normalized weighted combination (1-w)*a + w*b — the
// standard way to mix two same-space embeddings (e.g. a SigLIP 2 image vector
// and text vector) into a single query vector. Both inputs are normalized
// first so w is a true blend ratio even for legacy unnormalized rows. w is
// clamped to [0,1]. Errors when the lengths differ.
func Blend(a, b []float32, w float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("embedvec: blend length mismatch: %d vs %d", len(a), len(b))
	}
	if w < 0 {
		w = 0
	}
	if w > 1 {
		w = 1
	}
	a = Normalize(a)
	b = Normalize(b)
	out := make([]float32, len(a))
	for i := range a {
		out[i] = (1-w)*a[i] + w*b[i]
	}
	return Normalize(out), nil
}

// Combine returns the L2-normalized signed weighted sum Σ wᵢ·vᵢ of same-space
// embeddings — the N-way generalization of Blend for composite latent-space
// queries ("this image + that image + 'at night' − 'blurry'"). Each vector is
// normalized first so a weight is a true share regardless of input scale;
// NEGATIVE weights steer the query away from that concept. Errors on empty
// input, length mismatch, or when the weights cancel to a (near-)zero vector.
func Combine(vecs [][]float32, weights []float32) ([]float32, error) {
	if len(vecs) == 0 {
		return nil, fmt.Errorf("embedvec: combine: no vectors")
	}
	if len(weights) != len(vecs) {
		return nil, fmt.Errorf("embedvec: combine: %d vectors but %d weights", len(vecs), len(weights))
	}
	dim := len(vecs[0])
	out := make([]float32, dim)
	for i, v := range vecs {
		if len(v) != dim {
			return nil, fmt.Errorf("embedvec: combine length mismatch: %d vs %d", len(v), dim)
		}
		n := Normalize(v)
		w := weights[i]
		for k := range n {
			out[k] += w * n[k]
		}
	}
	var sum float64
	for _, x := range out {
		sum += float64(x) * float64(x)
	}
	if sum < 1e-12 {
		return nil, fmt.Errorf("embedvec: combine: weights cancel to a zero vector")
	}
	return Normalize(out), nil
}

// Cosine returns the dot product of a and b. When both are unit vectors this
// equals cosine similarity. Returns 0 if the lengths differ.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
	}
	return float32(dot)
}

// CosineSim returns true cosine similarity (dot product over the L2 norms),
// correct even when the inputs are not unit vectors. Returns 0 if the lengths
// differ or either vector is zero.
func CosineSim(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
