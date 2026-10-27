package onnxpose

import (
	"image"
	"image/color"
	"math"
	"testing"
)

func TestCropFor(t *testing.T) {
	// A tall person: padded 25%, widened to 3:4, centred on the box.
	c := CropFor(100, 50, 60, 200)
	if c.CX != 130 || c.CY != 150 || c.H != 250 || math.Abs(c.W-187.5) > 1e-9 {
		t.Fatalf("tall box crop %+v", c)
	}
	// A wide one: the height grows instead.
	c = CropFor(0, 0, 300, 100)
	if c.W != 375 || c.H != 500 {
		t.Fatalf("wide box crop %+v", c)
	}
}

// The input is the crop, normalized, channel-planar: a red left half and a
// blue right half land in the right planes and places.
func TestPreprocess(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 192, 256))
	for y := 0; y < 256; y++ {
		for x := 0; x < 192; x++ {
			c := color.RGBA{255, 0, 0, 255}
			if x >= 96 {
				c = color.RGBA{0, 0, 255, 255}
			}
			img.Set(x, y, c)
		}
	}
	// A crop exactly the image (undo the padding).
	in := Preprocess(img, Crop{CX: 96, CY: 128, W: 192, H: 256})
	n := InputW * InputH
	at := func(ch, u, v int) float32 { return in[ch*n+v*InputW+u]*std[ch] + mean[ch] }
	if r, b := at(0, 10, 100), at(2, 10, 100); math.Abs(float64(r-255)) > 1 || math.Abs(float64(b)) > 1 {
		t.Fatalf("left pixel r=%v b=%v", r, b)
	}
	if r, b := at(0, 180, 100), at(2, 180, 100); math.Abs(float64(r)) > 1 || math.Abs(float64(b-255)) > 1 {
		t.Fatalf("right pixel r=%v b=%v", r, b)
	}
}

// Each joint lands where its two classifications peak, mapped back through
// the crop, scored by the mean of the peaks.
func TestDecode(t *testing.T) {
	nx, ny := InputW*2, InputH*2
	sx := make([]float32, Joints*nx)
	sy := make([]float32, Joints*ny)
	for j := 0; j < Joints; j++ {
		sx[j*nx+96] = 0.8  // input x 48: a quarter across
		sy[j*ny+384] = 0.6 // input y 192: three quarters down
	}
	c := Crop{CX: 500, CY: 400, W: 300, H: 400}
	kp := Decode(sx, sy, c)
	if len(kp) != Joints {
		t.Fatalf("%d joints", len(kp))
	}
	if math.Abs(float64(kp[3].X-425)) > 0.01 || math.Abs(float64(kp[3].Y-500)) > 0.01 || math.Abs(float64(kp[3].Score-0.7)) > 1e-6 {
		t.Fatalf("joint %+v, want (425, 500) score 0.7", kp[3])
	}
}

func TestToOpenPose(t *testing.T) {
	coco := make([]Keypoint, Joints)
	for i := range coco {
		coco[i] = Keypoint{X: float32(i), Y: float32(100 + i), Score: 0.9}
	}
	coco[5] = Keypoint{X: 10, Y: 20, Score: 0.8} // left shoulder
	coco[6] = Keypoint{X: 30, Y: 40, Score: 0.5} // right shoulder
	op := ToOpenPose(coco)
	if len(op) != 18 {
		t.Fatalf("%d joints", len(op))
	}
	if op[1] != (Keypoint{X: 20, Y: 30, Score: 0.5}) {
		t.Fatalf("neck %+v", op[1])
	}
	// OpenPose's right shoulder is COCO's 6, its left ear COCO's 3.
	if op[2] != coco[6] || op[17] != coco[3] || op[10] != coco[16] {
		t.Fatalf("mapping %+v", op)
	}
}
