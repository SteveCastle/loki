// Package onnxface implements local face detection and face-identity
// embeddings on top of ONNX Runtime.
//
// Detection uses YuNet (OpenCV Zoo, Apache-2.0): an anchor-free detector whose
// ONNX graph takes a dynamically-sized BGR image and emits per-stride
// classification/objectness/bbox/landmark maps that are decoded and NMS'd here
// in pure Go (yunet.go).
//
// Recognition embeds a face crop that has been aligned to the canonical
// ArcFace 112x112 five-landmark template via a similarity transform estimated
// from the detected landmarks (align.go). The recognizer is pluggable: SFace
// (OpenCV Zoo, Apache-2.0, 128-dim) is the shipped default, and
// bring-your-own research models (ArcFace/AdaFace exports, 512-dim) work by
// configuring tensor names, dimension, and preprocessing.
//
// The ONNX sessions live in pipeline.go (cgo builds only); everything else is
// pure Go so the correctness-critical decode + alignment math is unit-testable
// without ONNX Runtime or model files.
package onnxface

// Detection is one detected face in original-image pixel coordinates.
type Detection struct {
	// Bounding box (top-left corner + size), clamped to the image bounds.
	X, Y, W, H float32
	// Landmarks are the five YuNet keypoints in model output order:
	// right eye, left eye, nose tip, right mouth corner, left mouth corner
	// ("right eye" = the subject's right eye, which appears on the image's
	// left in a normal photo — matching the alignment template's point order).
	Landmarks [5][2]float32
	// Score is sqrt(cls*obj) in [0,1].
	Score float32
}

// Face is one detected face plus its identity embedding (raw, not yet
// L2-normalized — callers normalize).
type Face struct {
	Detection
	Vec []float32
}

// alignTemplate is the canonical ArcFace/SFace 112x112 destination template
// for the five landmarks, in the same point order as Detection.Landmarks.
// These are the exact constants OpenCV's FaceRecognizerSF aligns to.
var alignTemplate = [5][2]float32{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// AlignSize is the width/height of the aligned face crop the template targets.
const AlignSize = 112
