package onnxface

// YOLO-style anime head detection (deepghs/anime_head_detection, MIT). The
// export is a standard single-class YOLOv8 head: input "images" [1,3,H,W]
// (RGB, 0..1, letterboxed), output "output0" [1, 5, N] — cx, cy, w, h in
// input pixels plus a sigmoid class score per anchor, no NMS baked in. Unlike
// YuNet the input is fully dynamic; we letterbox to a fixed square anyway so
// behavior is deterministic. Anime heads carry no usable landmarks, so
// detections from this path align via bbox expansion, not the 5-point warp.

const (
	yoloInputName  = "images"
	yoloOutputName = "output0"
)

// decodeYOLO converts a [1,5,N] YOLO output (planar: all cx, then all cy, …)
// into pixel-space detections on the letterboxed input, keeping anchors with
// score ≥ scoreThreshold. Landmarks are left zero.
func decodeYOLO(out []float32, scoreThreshold float32) []Detection {
	if len(out) < 5 {
		return nil
	}
	n := len(out) / 5
	var dets []Detection
	for i := 0; i < n; i++ {
		score := out[4*n+i]
		if score < scoreThreshold {
			continue
		}
		cx := out[0*n+i]
		cy := out[1*n+i]
		w := out[2*n+i]
		h := out[3*n+i]
		dets = append(dets, Detection{
			X:     cx - w/2,
			Y:     cy - h/2,
			W:     w,
			H:     h,
			Score: score,
		})
	}
	return dets
}
