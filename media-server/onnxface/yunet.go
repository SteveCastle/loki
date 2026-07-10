package onnxface

import (
	"fmt"
	"math"
	"sort"
)

// yunetStrides are the three feature-map strides of the YuNet 2023mar head.
// For each stride s the model emits four tensors named cls_s, obj_s, bbox_s,
// kps_s over a (padH/s × padW/s) grid.
var yunetStrides = [3]int{8, 16, 32}

// yunetOutputNames returns the 12 output tensor names in the order the decode
// expects: cls, obj, bbox, kps per stride.
func yunetOutputNames() []string {
	names := make([]string, 0, 12)
	for _, s := range yunetStrides {
		names = append(names,
			fmt.Sprintf("cls_%d", s),
			fmt.Sprintf("obj_%d", s),
			fmt.Sprintf("bbox_%d", s),
			fmt.Sprintf("kps_%d", s),
		)
	}
	return names
}

// yunetRaw is the raw model output for one stride.
type yunetRaw struct {
	stride int
	cls    []float32 // [rows*cols] face-class probability
	obj    []float32 // [rows*cols] objectness probability
	bbox   []float32 // [rows*cols*4] center offset (cells) + log size (cells)
	kps    []float32 // [rows*cols*10] five landmark offsets (cells)
}

// decodeYuNet converts the raw per-stride maps into pixel-space detections on
// the padded input of size padW×padH, keeping only faces scoring at least
// scoreThreshold. This mirrors OpenCV's FaceDetectorYN decode: the anchor-free
// head predicts, per grid cell, a face probability (cls), an objectness (obj),
// a box center offset + log-size in cell units, and five landmark offsets in
// cell units. The detection score is sqrt(cls*obj).
func decodeYuNet(padW, padH int, raw []yunetRaw, scoreThreshold float32) []Detection {
	var dets []Detection
	for _, r := range raw {
		cols := padW / r.stride
		rows := padH / r.stride
		n := rows * cols
		if len(r.cls) < n || len(r.obj) < n || len(r.bbox) < 4*n || len(r.kps) < 10*n {
			continue // malformed output; skip the stride rather than panic
		}
		s := float32(r.stride)
		for row := 0; row < rows; row++ {
			for col := 0; col < cols; col++ {
				idx := row*cols + col
				cls := clamp01(r.cls[idx])
				obj := clamp01(r.obj[idx])
				score := float32(math.Sqrt(float64(cls * obj)))
				if score < scoreThreshold {
					continue
				}
				cx := (float32(col) + r.bbox[idx*4+0]) * s
				cy := (float32(row) + r.bbox[idx*4+1]) * s
				w := float32(math.Exp(float64(r.bbox[idx*4+2]))) * s
				h := float32(math.Exp(float64(r.bbox[idx*4+3]))) * s
				d := Detection{
					X:     cx - w/2,
					Y:     cy - h/2,
					W:     w,
					H:     h,
					Score: score,
				}
				for k := 0; k < 5; k++ {
					d.Landmarks[k][0] = (float32(col) + r.kps[idx*10+2*k]) * s
					d.Landmarks[k][1] = (float32(row) + r.kps[idx*10+2*k+1]) * s
				}
				dets = append(dets, d)
			}
		}
	}
	return dets
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// nms performs greedy non-maximum suppression: detections are sorted by score
// and any detection overlapping an already-kept one with IoU > iouThreshold is
// dropped. The returned slice is score-descending.
func nms(dets []Detection, iouThreshold float32) []Detection {
	sorted := make([]Detection, len(dets))
	copy(sorted, dets)
	sort.SliceStable(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })

	kept := sorted[:0]
	for _, d := range sorted {
		ok := true
		for _, k := range kept {
			if iou(d, k) > iouThreshold {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, d)
		}
	}
	out := make([]Detection, len(kept))
	copy(out, kept)
	return out
}

// iou computes intersection-over-union of two detections' boxes.
func iou(a, b Detection) float32 {
	x1 := maxf(a.X, b.X)
	y1 := maxf(a.Y, b.Y)
	x2 := minf(a.X+a.W, b.X+b.W)
	y2 := minf(a.Y+a.H, b.Y+b.H)
	iw := x2 - x1
	ih := y2 - y1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	union := a.W*a.H + b.W*b.H - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func maxf(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func minf(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
