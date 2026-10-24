// Package onnxpose estimates a person's body pose with RTMPose (OpenMMLab
// mmpose, Apache-2.0) on ONNX Runtime: a top-down model that takes one
// person's box, crops it to 192x256 and answers the 17 COCO keypoints as SimCC
// coordinate classifications (simcc_x over 2x the input width, simcc_y over
// 2x its height). The Scene editor's mannequin is OpenPose's 18-joint layout,
// so ToOpenPose converts (the neck is the shoulders' midpoint).
//
// The session lives in session.go (cgo builds only); everything here is pure
// Go so the crop and decode math is unit-testable without ONNX Runtime.
package onnxpose

import (
	"image"
	"math"
)

const (
	// InputW, InputH: the model's input size.
	InputW = 192
	InputH = 256
	// splitRatio: SimCC bins per input pixel.
	splitRatio = 2.0
	// padding grows the person's box before cropping (mmpose's
	// TopDownGetBboxCenterScale default for RTMPose).
	padding = 1.25
	// Joints is the number of COCO body keypoints the model answers.
	Joints = 17
)

// RGB normalization (ImageNet, on the 0-255 scale).
var (
	mean = [3]float32{123.675, 116.28, 103.53}
	std  = [3]float32{58.395, 57.12, 57.375}
)

// Keypoint is one joint in image pixels with the model's confidence (about
// 0-1; below ~0.3 the joint is a guess).
type Keypoint struct {
	X, Y, Score float32
}

// Crop is the area of the image the model sees: its centre and size in
// image pixels, at the model's aspect.
type Crop struct {
	CX, CY, W, H float64
}

// CropFor is the crop for a person's box (x, y, w, h in image pixels): the box
// padded by 25% and grown to the model's 3:4 aspect about its centre.
func CropFor(x, y, w, h float64) Crop {
	c := Crop{CX: x + w/2, CY: y + h/2, W: w * padding, H: h * padding}
	if aspect := float64(InputW) / float64(InputH); c.W > c.H*aspect {
		c.H = c.W / aspect
	} else {
		c.W = c.H * aspect
	}
	return c
}

// Preprocess samples the crop out of img (bilinear, edges clamped) into the
// model's normalized RGB CHW input.
func Preprocess(img image.Image, c Crop) []float32 {
	b := img.Bounds()
	n := InputW * InputH
	out := make([]float32, 3*n)
	sample := func(x, y float64) [3]float32 {
		x = math.Max(float64(b.Min.X), math.Min(float64(b.Max.X-1), x))
		y = math.Max(float64(b.Min.Y), math.Min(float64(b.Max.Y-1), y))
		x0, y0 := int(math.Floor(x)), int(math.Floor(y))
		x1, y1 := min(x0+1, b.Max.X-1), min(y0+1, b.Max.Y-1)
		fx, fy := float32(x-float64(x0)), float32(y-float64(y0))
		var px [4][3]float32
		for k, p := range [4][2]int{{x0, y0}, {x1, y0}, {x0, y1}, {x1, y1}} {
			r, g, bl, _ := img.At(p[0], p[1]).RGBA()
			px[k] = [3]float32{float32(r >> 8), float32(g >> 8), float32(bl >> 8)}
		}
		var v [3]float32
		for ch := 0; ch < 3; ch++ {
			top := px[0][ch]*(1-fx) + px[1][ch]*fx
			bot := px[2][ch]*(1-fx) + px[3][ch]*fx
			v[ch] = top*(1-fy) + bot*fy
		}
		return v
	}
	for v := 0; v < InputH; v++ {
		for u := 0; u < InputW; u++ {
			x, y := c.toImage((float64(u)+0.5)*splitRatio, (float64(v)+0.5)*splitRatio)
			p := sample(x-0.5, y-0.5)
			i := v*InputW + u
			for ch := 0; ch < 3; ch++ {
				out[ch*n+i] = (p[ch] - mean[ch]) / std[ch]
			}
		}
	}
	return out
}

// toImage maps SimCC bin coordinates (input pixels x splitRatio) to image pixels.
func (c Crop) toImage(bx, by float64) (float64, float64) {
	return c.CX - c.W/2 + bx/splitRatio/InputW*c.W, c.CY - c.H/2 + by/splitRatio/InputH*c.H
}

// Decode turns the model's outputs (simcc_x: Joints x InputW*2, simcc_y:
// Joints x InputH*2) into keypoints in image pixels: each joint where its two
// classifications peak, scored by the mean of the peaks (mmpose's
// get_simcc_maximum).
func Decode(simccX, simccY []float32, c Crop) []Keypoint {
	nx, ny := int(InputW*splitRatio), int(InputH*splitRatio)
	out := make([]Keypoint, 0, Joints)
	for j := 0; j < Joints && (j+1)*nx <= len(simccX) && (j+1)*ny <= len(simccY); j++ {
		ix, vx := argmax(simccX[j*nx : (j+1)*nx])
		iy, vy := argmax(simccY[j*ny : (j+1)*ny])
		x, y := c.toImage(float64(ix), float64(iy))
		out = append(out, Keypoint{X: float32(x), Y: float32(y), Score: 0.5 * (vx + vy)})
	}
	return out
}

func argmax(v []float32) (int, float32) {
	best, bv := 0, float32(math.Inf(-1))
	for i, x := range v {
		if x > bv {
			best, bv = i, x
		}
	}
	return best, bv
}

// cocoToOpenPose: for each OpenPose joint (nose, neck, right shoulder, ...,
// left ear), the COCO keypoint it is; -1 = the neck, made from the shoulders.
var cocoToOpenPose = [18]int{0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3}

// ToOpenPose converts COCO-17 keypoints to OpenPose's 18 joints, the
// mannequin's layout: the neck is the midpoint of the shoulders, scored by the
// weaker of them.
func ToOpenPose(coco []Keypoint) []Keypoint {
	out := make([]Keypoint, 18)
	if len(coco) < Joints {
		return out
	}
	for j, c := range cocoToOpenPose {
		if c >= 0 {
			out[j] = coco[c]
			continue
		}
		l, r := coco[5], coco[6]
		out[j] = Keypoint{X: (l.X + r.X) / 2, Y: (l.Y + r.Y) / 2, Score: min(l.Score, r.Score)}
	}
	return out
}
