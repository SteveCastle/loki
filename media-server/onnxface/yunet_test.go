package onnxface

import (
	"math"
	"testing"
)

// buildRaw creates a zeroed raw output set for a padW×padH input, then lets
// the caller poke values in.
func buildRaw(padW, padH int) []yunetRaw {
	raws := make([]yunetRaw, 0, len(yunetStrides))
	for _, s := range yunetStrides {
		n := (padW / s) * (padH / s)
		raws = append(raws, yunetRaw{
			stride: s,
			cls:    make([]float32, n),
			obj:    make([]float32, n),
			bbox:   make([]float32, n*4),
			kps:    make([]float32, n*10),
		})
	}
	return raws
}

func TestDecodeYuNetSingleCell(t *testing.T) {
	const padW, padH = 64, 64
	raws := buildRaw(padW, padH)

	// One face at stride 8, cell (col=2, row=1).
	r := &raws[0]
	cols := padW / 8
	idx := 1*cols + 2
	r.cls[idx] = 0.9
	r.obj[idx] = 0.9 // score = sqrt(0.81) = 0.9
	r.bbox[idx*4+0] = 0.5
	r.bbox[idx*4+1] = 0.25
	r.bbox[idx*4+2] = float32(math.Log(4)) // w = 4*8 = 32
	r.bbox[idx*4+3] = float32(math.Log(2)) // h = 2*8 = 16
	for k := 0; k < 5; k++ {
		r.kps[idx*10+2*k] = float32(k) * 0.1
		r.kps[idx*10+2*k+1] = 0.5
	}

	dets := decodeYuNet(padW, padH, raws, 0.7)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1", len(dets))
	}
	d := dets[0]
	// center = ((2+0.5)*8, (1+0.25)*8) = (20, 10); box 32×16 → x=4, y=2.
	wantX, wantY := float32(4), float32(2)
	if !close32(d.X, wantX) || !close32(d.Y, wantY) || !close32(d.W, 32) || !close32(d.H, 16) {
		t.Fatalf("bbox = (%v,%v,%v,%v), want (%v,%v,32,16)", d.X, d.Y, d.W, d.H, wantX, wantY)
	}
	if !close32(d.Score, 0.9) {
		t.Fatalf("score = %v, want 0.9", d.Score)
	}
	for k := 0; k < 5; k++ {
		wantLX := (2 + float32(k)*0.1) * 8
		wantLY := float32((1 + 0.5) * 8)
		if !close32(d.Landmarks[k][0], wantLX) || !close32(d.Landmarks[k][1], wantLY) {
			t.Fatalf("landmark %d = %v, want (%v,%v)", k, d.Landmarks[k], wantLX, wantLY)
		}
	}
}

func TestDecodeYuNetThresholdAndClamp(t *testing.T) {
	const padW, padH = 32, 32
	raws := buildRaw(padW, padH)
	// Below threshold: sqrt(0.5*0.5) = 0.5 < 0.7 → dropped.
	raws[0].cls[0] = 0.5
	raws[0].obj[0] = 0.5
	// Out-of-range confidences must clamp to [0,1], not exceed 1.
	raws[1].cls[0] = 1.7
	raws[1].obj[0] = 2.0

	dets := decodeYuNet(padW, padH, raws, 0.7)
	if len(dets) != 1 {
		t.Fatalf("got %d detections, want 1 (clamped)", len(dets))
	}
	if dets[0].Score != 1 {
		t.Fatalf("clamped score = %v, want 1", dets[0].Score)
	}
}

func TestDecodeYuNetMalformedOutputSkipped(t *testing.T) {
	raws := []yunetRaw{{stride: 8, cls: []float32{1}, obj: []float32{1}, bbox: nil, kps: nil}}
	if dets := decodeYuNet(64, 64, raws, 0.5); len(dets) != 0 {
		t.Fatalf("malformed stride produced %d detections, want 0", len(dets))
	}
}

func TestNMS(t *testing.T) {
	dets := []Detection{
		{X: 0, Y: 0, W: 10, H: 10, Score: 0.8},
		{X: 1, Y: 1, W: 10, H: 10, Score: 0.9}, // overlaps the first heavily
		{X: 100, Y: 100, W: 10, H: 10, Score: 0.5},
	}
	kept := nms(dets, 0.3)
	if len(kept) != 2 {
		t.Fatalf("kept %d, want 2", len(kept))
	}
	if kept[0].Score != 0.9 || kept[1].Score != 0.5 {
		t.Fatalf("kept scores = %v, %v; want 0.9, 0.5 (score-descending)", kept[0].Score, kept[1].Score)
	}
}

func TestIoU(t *testing.T) {
	a := Detection{X: 0, Y: 0, W: 10, H: 10}
	b := Detection{X: 5, Y: 0, W: 10, H: 10} // overlap 50 of union 150
	if got := iou(a, b); !close32(got, 50.0/150.0) {
		t.Fatalf("iou = %v, want %v", got, 50.0/150.0)
	}
	c := Detection{X: 50, Y: 50, W: 10, H: 10}
	if got := iou(a, c); got != 0 {
		t.Fatalf("disjoint iou = %v, want 0", got)
	}
}

func close32(a, b float32) bool {
	return math.Abs(float64(a-b)) < 1e-4
}
