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
	inv := float32(1.0 / math.Sqrt(sum))
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x * inv
	}
	return out
}

// Cosine returns the dot product of a and b. When both are unit vectors this
// equals cosine similarity. Returns 0 if the lengths differ.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}
