//go:build !cgo

package onnxpose

import (
	"errors"
	"image"
)

// ErrCGORequired is returned by inference in non-cgo builds.
var ErrCGORequired = errors.New("onnxpose: built without cgo; ONNX inference unavailable")

// Estimator mirrors the cgo type (session.go).
type Estimator struct{}

// NewEstimator always fails without cgo.
func NewEstimator(modelPath, ortLib string) (*Estimator, error) { return nil, ErrCGORequired }

// Estimate always fails without cgo.
func (e *Estimator) Estimate(img image.Image, x, y, w, h float64) ([]Keypoint, error) {
	return nil, ErrCGORequired
}

// Destroy does nothing without cgo.
func (e *Estimator) Destroy() {}
