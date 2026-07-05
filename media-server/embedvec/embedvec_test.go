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

func TestBlendEndpoints(t *testing.T) {
	a := []float32{1, 0}
	b := []float32{0, 1}
	// w=0 → pure a; w=1 → pure b.
	got, err := Blend(a, b, 0)
	if err != nil {
		t.Fatalf("blend error: %v", err)
	}
	if math.Abs(float64(got[0])-1) > 1e-6 || math.Abs(float64(got[1])) > 1e-6 {
		t.Errorf("w=0 should return a: got %v", got)
	}
	got, err = Blend(a, b, 1)
	if err != nil {
		t.Fatalf("blend error: %v", err)
	}
	if math.Abs(float64(got[0])) > 1e-6 || math.Abs(float64(got[1])-1) > 1e-6 {
		t.Errorf("w=1 should return b: got %v", got)
	}
}

func TestBlendMidpointIsUnitLength(t *testing.T) {
	got, err := Blend([]float32{1, 0}, []float32{0, 1}, 0.5)
	if err != nil {
		t.Fatalf("blend error: %v", err)
	}
	var sum float64
	for _, x := range got {
		sum += float64(x) * float64(x)
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("blend not renormalized: sum of squares = %v", sum)
	}
	// Equidistant blend of orthogonal unit vectors → equal components.
	if math.Abs(float64(got[0]-got[1])) > 1e-6 {
		t.Errorf("even blend of orthogonal vectors should have equal components: %v", got)
	}
}

func TestBlendNormalizesInputs(t *testing.T) {
	// A legacy unnormalized image vector must not dominate by magnitude.
	got, err := Blend([]float32{10, 0}, []float32{0, 1}, 0.5)
	if err != nil {
		t.Fatalf("blend error: %v", err)
	}
	if math.Abs(float64(got[0]-got[1])) > 1e-6 {
		t.Errorf("inputs should be normalized before blending: %v", got)
	}
}

func TestBlendClampsWeight(t *testing.T) {
	got, err := Blend([]float32{1, 0}, []float32{0, 1}, 1.5)
	if err != nil {
		t.Fatalf("blend error: %v", err)
	}
	if math.Abs(float64(got[1])-1) > 1e-6 {
		t.Errorf("w>1 should clamp to pure b: got %v", got)
	}
}

func TestBlendLengthMismatch(t *testing.T) {
	if _, err := Blend([]float32{1, 0}, []float32{1, 0, 0}, 0.5); err == nil {
		t.Fatal("expected error for mismatched lengths")
	}
}

func TestNormalizeZeroVector(t *testing.T) {
	z := Normalize([]float32{0, 0, 0})
	for _, x := range z {
		if x != 0 {
			t.Fatalf("zero vector mutated: got %v", z)
		}
	}
}
