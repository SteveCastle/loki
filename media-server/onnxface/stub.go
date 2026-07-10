//go:build !cgo

package onnxface

import (
	"errors"
	"image"
)

// ErrCGORequired is returned by every inference entry point in non-cgo builds.
// The pure-Go decode/align math still compiles and tests everywhere.
var ErrCGORequired = errors.New("onnxface: built without cgo; ONNX inference unavailable")

// DetectorConfig mirrors the cgo type (pipeline.go).
type DetectorConfig struct {
	ModelPath      string
	Kind           string
	InputSize      int
	ScoreThreshold float32
	NMSThreshold   float32
	MinSize        int
	ORTLib         string
	Provider       string
	Device         int
	Threads        int
}

// RecognizerConfig mirrors the cgo type (pipeline.go).
type RecognizerConfig struct {
	ModelPath  string
	Dim        int
	InputName  string
	OutputName string
	InputSize  int
	Mean, Std  [3]float32
	ColorOrder string
	ORTLib     string
	Provider   string
	Device     int
	Threads    int
}

// PipelineSpec mirrors the cgo type (pipeline.go).
type PipelineSpec struct {
	Detector        DetectorConfig
	Recognizer      RecognizerConfig
	Secondary       *RecognizerConfig
	Weight          float32
	SecondaryWeight float32
	Align           string
	CropExpand      float32
}

// Detector is unavailable without cgo.
type Detector struct{}

func NewDetector(cfg DetectorConfig) (*Detector, error)          { return nil, ErrCGORequired }
func (d *Detector) Detect(img image.Image) ([]Detection, error)  { return nil, ErrCGORequired }
func (d *Detector) Close() error                                 { return nil }

// Recognizer is unavailable without cgo.
type Recognizer struct{}

func NewRecognizer(cfg RecognizerConfig) (*Recognizer, error)          { return nil, ErrCGORequired }
func (r *Recognizer) Embed(aligned *image.NRGBA) ([]float32, error)    { return nil, ErrCGORequired }
func (r *Recognizer) Close() error                                     { return nil }

// Pipeline is unavailable without cgo.
type Pipeline struct {
	Det  *Detector
	Rec  *Recognizer
	Rec2 *Recognizer
}

func NewPipeline(spec PipelineSpec) (*Pipeline, error) {
	return nil, ErrCGORequired
}
func (p *Pipeline) Process(imagePath string) ([]Face, int, int, error) {
	return nil, 0, 0, ErrCGORequired
}
func (p *Pipeline) Close() error { return nil }
