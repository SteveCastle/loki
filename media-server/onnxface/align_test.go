package onnxface

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// applyKnown maps a point through an explicit similarity: scale s, rotation
// theta (radians), then translation (tx, ty).
func applyKnown(s, theta, tx, ty, x, y float64) (float64, float64) {
	a := s * math.Cos(theta)
	b := s * math.Sin(theta)
	return a*x - b*y + tx, b*x + a*y + ty
}

func TestEstimateSimilarityIdentity(t *testing.T) {
	m, err := estimateSimilarity(alignTemplate[:], alignTemplate[:])
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(m.a-1) > 1e-9 || math.Abs(m.b) > 1e-9 || math.Abs(m.tx) > 1e-9 || math.Abs(m.ty) > 1e-9 {
		t.Fatalf("expected identity, got %+v", m)
	}
}

func TestEstimateSimilarityRecoversKnownTransform(t *testing.T) {
	// Build src by pushing the template through a known similarity; estimating
	// src→template must then recover its exact inverse (5 points, no noise).
	const s, theta, tx, ty = 2.5, 0.5, 40.0, -12.0
	var src [5][2]float32
	for i, p := range alignTemplate {
		x, y := applyKnown(s, theta, tx, ty, float64(p[0]), float64(p[1]))
		src[i] = [2]float32{float32(x), float32(y)}
	}
	m, err := estimateSimilarity(src[:], alignTemplate[:])
	if err != nil {
		t.Fatal(err)
	}
	for i, p := range src {
		gx, gy := m.apply(float64(p[0]), float64(p[1]))
		wx, wy := float64(alignTemplate[i][0]), float64(alignTemplate[i][1])
		if math.Abs(gx-wx) > 1e-3 || math.Abs(gy-wy) > 1e-3 {
			t.Fatalf("point %d: got (%.5f, %.5f), want (%.5f, %.5f)", i, gx, gy, wx, wy)
		}
	}
	// The recovered scale must be 1/s.
	gotScale := math.Hypot(m.a, m.b)
	if math.Abs(gotScale-1/s) > 1e-6 {
		t.Fatalf("scale: got %v, want %v", gotScale, 1/s)
	}
}

func TestSimilarityInvertRoundTrip(t *testing.T) {
	m := similarity{a: 0.8, b: 0.3, tx: 15, ty: -7}
	inv, err := m.invert()
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range [][2]float64{{0, 0}, {10, 20}, {-5, 3.5}} {
		fx, fy := m.apply(p[0], p[1])
		bx, by := inv.apply(fx, fy)
		if math.Abs(bx-p[0]) > 1e-9 || math.Abs(by-p[1]) > 1e-9 {
			t.Fatalf("round trip (%v,%v) -> (%v,%v)", p[0], p[1], bx, by)
		}
	}
}

func TestEstimateSimilarityDegenerate(t *testing.T) {
	same := [][2]float32{{5, 5}, {5, 5}, {5, 5}}
	if _, err := estimateSimilarity(same, same); err == nil {
		t.Fatal("expected error for coincident points")
	}
	if _, err := estimateSimilarity([][2]float32{{1, 2}}, [][2]float32{{1, 2}}); err == nil {
		t.Fatal("expected error for a single point")
	}
	if _, err := (similarity{}).invert(); err == nil {
		t.Fatal("expected error inverting a zero transform")
	}
}

// TestAlignFaceWarpsLandmarksToTemplate paints unique colors at five landmark
// positions (a scaled/rotated/translated copy of the template) in a synthetic
// source image, aligns, and verifies each color lands at its template
// coordinate in the 112×112 crop.
func TestAlignFaceWarpsLandmarksToTemplate(t *testing.T) {
	const s, theta, tx, ty = 3.0, 0.35, 120.0, 80.0
	colors := [5]color.NRGBA{
		{255, 0, 0, 255},
		{0, 255, 0, 255},
		{0, 0, 255, 255},
		{255, 255, 0, 255},
		{0, 255, 255, 255},
	}

	src := image.NewNRGBA(image.Rect(0, 0, 640, 640))
	var landmarks [5][2]float32
	for i, p := range alignTemplate {
		x, y := applyKnown(s, theta, tx, ty, float64(p[0]), float64(p[1]))
		landmarks[i] = [2]float32{float32(x), float32(y)}
		// Paint a blob comfortably larger than one warped output pixel (the
		// warp shrinks by ~1/s), so bilinear sampling reads a solid color.
		r := int(s * 2)
		for dy := -r; dy <= r; dy++ {
			for dx := -r; dx <= r; dx++ {
				src.SetNRGBA(int(x)+dx, int(y)+dy, colors[i])
			}
		}
	}

	dst, err := alignFace(src, landmarks)
	if err != nil {
		t.Fatal(err)
	}
	if got := dst.Bounds(); got.Dx() != AlignSize || got.Dy() != AlignSize {
		t.Fatalf("aligned crop is %v, want %dx%d", got, AlignSize, AlignSize)
	}
	for i, p := range alignTemplate {
		got := dst.NRGBAAt(int(p[0]+0.5), int(p[1]+0.5))
		want := colors[i]
		if colorDist(got, want) > 60 {
			t.Fatalf("landmark %d: color at template point = %v, want ≈ %v", i, got, want)
		}
	}
}

func colorDist(a, b color.NRGBA) float64 {
	dr := float64(a.R) - float64(b.R)
	dg := float64(a.G) - float64(b.G)
	db := float64(a.B) - float64(b.B)
	return math.Sqrt(dr*dr + dg*dg + db*db)
}

func TestSampleBilinearBorderIsBlack(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.SetNRGBA(x, y, color.NRGBA{255, 255, 255, 255})
		}
	}
	c := sampleBilinear(img, 4, 4, -10, -10)
	if c.R != 0 || c.G != 0 || c.B != 0 {
		t.Fatalf("outside sample = %v, want black", c)
	}
	c = sampleBilinear(img, 4, 4, 1.5, 1.5)
	if c.R != 255 || c.G != 255 || c.B != 255 {
		t.Fatalf("inside sample = %v, want white", c)
	}
}
