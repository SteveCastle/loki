package embedvec

import (
	"math"
	"testing"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	in := []float32{0.5, -0.25, 1.0, 0.0}
	out, err := Decode(Encode(in))
	if err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("len mismatch: got %d want %d", len(out), len(in))
	}
	for i := range in {
		if out[i] != in[i] {
			t.Errorf("index %d: got %v want %v", i, out[i], in[i])
		}
	}
}

func TestDecodeRejectsRagged(t *testing.T) {
	if _, err := Decode([]byte{1, 2, 3}); err == nil {
		t.Fatal("expected error for non-multiple-of-4 length")
	}
}

func TestNormalizeUnitLength(t *testing.T) {
	v := Normalize([]float32{3, 4})
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("not unit length: sum of squares = %v", sum)
	}
}

func TestCosineIdentical(t *testing.T) {
	a := Normalize([]float32{1, 2, 3})
	if got := Cosine(a, a); math.Abs(float64(got)-1.0) > 1e-6 {
		t.Errorf("cosine of identical = %v, want ~1.0", got)
	}
}

func TestCosineLengthMismatch(t *testing.T) {
	if got := Cosine([]float32{1, 0}, []float32{1, 0, 0}); got != 0 {
		t.Errorf("mismatched lengths should return 0, got %v", got)
	}
}
