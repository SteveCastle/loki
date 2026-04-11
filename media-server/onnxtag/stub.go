//go:build !cgo
// +build !cgo

// Package onnxtag provides image classification using ONNX models.
// This is a stub file for non-CGO builds where ONNX Runtime is not available.
package onnxtag

import (
	"errors"
)

// ErrCGORequired is returned when ONNX tagging is attempted without CGO support.
var ErrCGORequired = errors.New("onnxtag requires CGO support; rebuild with CGO_ENABLED=1")

// Options configures how the classifier runs.
// This must mirror the fields in the CGO build (onnxtag.go) so that
// cmd/onnxtag and other callers compile without CGO.
type Options struct {
	ORTSharedLibraryPath string
	InputName            string
	OutputName           string
	InputWidth           int
	InputHeight          int
	NormalizeMeanRGB     [3]float32
	NormalizeStddevRGB   [3]float32
	Labels               []string
	TopK                 int
	NumClasses           int
	Interpolation        string
	CropPct              float32
	CropMode             string
	SelectedClassNames   map[int]string
	InputLayout          string
	ColorOrder           string
	PixelRange           string
	PadToSquare          bool
	RatingIndices        []int
	GeneralIndices       []int
	CharacterIndices     []int
	GeneralThreshold     float32
	CharacterThreshold   float32
}

// DefaultOptions returns default Options.
func DefaultOptions() Options {
	return Options{}
}

// Result holds classification output for a single image.
type Result struct {
	Tags []Tag
}

// Tag represents a single classification tag.
type Tag struct {
	Label      string
	Category   string
	Confidence float32
}

// Classify returns an error indicating CGO is required.
func Classify(imagePath, modelPath string, opts Options) (Result, error) {
	return Result{}, ErrCGORequired
}

// ClassifyImage returns an error indicating CGO is required.
func ClassifyImage(modelPath, imagePath string, opts Options) ([]string, error) {
	return nil, ErrCGORequired
}

// ClassifyReader returns an error indicating CGO is required.
func ClassifyReader(r interface{}, modelPath string, opts Options) (Result, error) {
	return Result{}, ErrCGORequired
}

// ModelConfig holds configuration loaded from a model's config.json.
type ModelConfig struct{}

// LoadModelConfig returns an error indicating CGO is required.
func LoadModelConfig(configPath string) (*ModelConfig, error) {
	return nil, ErrCGORequired
}

// ApplyToOptions is a no-op in non-CGO builds.
func (mc *ModelConfig) ApplyToOptions(opts *Options) {}

// LoadLabels returns an error indicating CGO is required.
func LoadLabels(labelsPath string) ([]string, error) {
	return nil, ErrCGORequired
}

// LoadLabelsAndCategories returns an error indicating CGO is required.
func LoadLabelsAndCategories(labelsPath string) ([]string, []string, error) {
	return nil, nil, ErrCGORequired
}

// LoadSelectedTagsCSV returns an error indicating CGO is required.
func LoadSelectedTagsCSV(path string) (map[int]string, error) {
	return nil, ErrCGORequired
}

// LoadWdLabelsFromCSV returns an error indicating CGO is required.
func LoadWdLabelsFromCSV(path string) ([]string, error) {
	return nil, ErrCGORequired
}

// LoadWdCategoryIndicesWithOrder returns an error indicating CGO is required.
func LoadWdCategoryIndicesWithOrder(path string) (ratingIdx, generalIdx, characterIdx []int, err error) {
	return nil, nil, nil, ErrCGORequired
}
