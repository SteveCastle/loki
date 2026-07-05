package tasks

import (
	"fmt"
	"os"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
)

// FaceModel describes one face-identity recognizer: its identity, output
// dimensionality, tensor names, and aligned-crop preprocessing. It mirrors
// EmbedModel. Face vectors are stored keyed by ID in the face table, so
// recognizers coexist non-destructively.
//
// Two kinds of entry exist:
//   - built-in (shipped, Apache-2.0): resolved through the deps/models
//     download flow (DepID + File), e.g. SFace;
//   - bring-your-own (research-licensed ArcFace/AdaFace exports that must not
//     be shipped): declared in config (appconfig.ByoFaceModel) with an
//     absolute Path to a user-supplied ONNX file.
//
// Detection is always YuNet regardless of the recognizer.
type FaceModel struct {
	ID          string
	DisplayName string
	Dim         int

	// Built-in resolution (deps download flow). Empty for BYO models.
	DepID string // deps/models manifest ID
	File  string // rel path under the model dir

	// BYO resolution: absolute path to the ONNX file. Empty for built-ins.
	Path string
	BYO  bool

	// Recognizer I/O + preprocessing of the aligned 112×112 crop.
	InputName  string
	OutputName string
	Mean, Std  [3]float32 // per-channel RGB on the 0..255 pixel scale
	ColorOrder string     // "BGR" (SFace) or "RGB" (ArcFace-family)

	// MatchThreshold is the cosine similarity at/above which two faces count
	// as the same person (clustering / search grouping). Recognizers have
	// different score distributions, so this is per-model.
	MatchThreshold float32
}

// DefaultFaceModelID is the recognizer used when the config is empty or names
// an unknown model.
const DefaultFaceModelID = appconfig.DefaultFaceModel

// The YuNet detector's deps-manifest coordinates (shared by every recognizer).
const (
	FaceDetectorDepID = "yunet"
	FaceDetectorFile  = "model.onnx"
)

// defaultByoMatchThreshold is used when a BYO entry doesn't set one. ArcFace-
// family models typically land in the 0.35–0.45 same-person cosine band; 0.40
// is a conservative middle. BYO entries should still set it explicitly.
const defaultByoMatchThreshold = 0.40

// builtinFaceModels is keyed by model ID. Add new shipped recognizers here
// (they also need a deps/models manifest entry).
var builtinFaceModels = map[string]FaceModel{
	"sface": {
		ID:          "sface",
		DisplayName: "SFace (OpenCV Zoo — shipped default)",
		Dim:         128,
		DepID:       "sface",
		File:        "model.onnx",
		InputName:   "data",
		OutputName:  "fc1",
		Mean:        [3]float32{0, 0, 0},
		Std:         [3]float32{1, 1, 1},
		ColorOrder:  "BGR",
		// OpenCV's tuned same-identity cosine threshold for SFace.
		MatchThreshold: 0.363,
	},
}

// byoToFaceModel converts a config BYO declaration into a FaceModel, filling
// the SFace-convention defaults for anything unset.
func byoToFaceModel(b appconfig.ByoFaceModel) FaceModel {
	m := FaceModel{
		ID:             strings.TrimSpace(b.ID),
		DisplayName:    b.Name,
		Dim:            b.Dim,
		Path:           b.ModelPath,
		BYO:            true,
		InputName:      b.InputName,
		OutputName:     b.OutputName,
		ColorOrder:     b.ColorOrder,
		MatchThreshold: float32(b.MatchThreshold),
	}
	if m.DisplayName == "" {
		m.DisplayName = m.ID + " (bring-your-own)"
	}
	if m.InputName == "" {
		m.InputName = "data"
	}
	if m.OutputName == "" {
		m.OutputName = "fc1"
	}
	if m.ColorOrder == "" {
		m.ColorOrder = "BGR"
	}
	m.Mean = rgb3(b.Mean, [3]float32{0, 0, 0})
	m.Std = rgb3(b.Std, [3]float32{1, 1, 1})
	if m.MatchThreshold <= 0 {
		m.MatchThreshold = defaultByoMatchThreshold
	}
	return m
}

func rgb3(v []float64, def [3]float32) [3]float32 {
	if len(v) != 3 {
		return def
	}
	return [3]float32{float32(v[0]), float32(v[1]), float32(v[2])}
}

// FaceModelByID returns the recognizer for id — built-in first, then the
// config's BYO entries — and whether it exists. BYO entries missing an ID,
// path, or dimension are skipped (they can't run).
func FaceModelByID(id string) (FaceModel, bool) {
	if id == "" {
		return FaceModel{}, false
	}
	if m, ok := builtinFaceModels[id]; ok {
		return m, true
	}
	for _, b := range appconfig.Get().ByoFaceModels {
		if strings.TrimSpace(b.ID) == id && b.ModelPath != "" && b.Dim > 0 {
			return byoToFaceModel(b), true
		}
	}
	return FaceModel{}, false
}

// ActiveFaceModel returns the configured recognizer, falling back to the
// default when the config is empty or names an unknown/incomplete model.
func ActiveFaceModel() FaceModel {
	if m, ok := FaceModelByID(strings.TrimSpace(appconfig.Get().FaceModel)); ok {
		return m
	}
	return builtinFaceModels[DefaultFaceModelID]
}

// FaceModelList returns all selectable recognizers (built-ins first, then
// valid BYO entries) for the config UI.
func FaceModelList() []FaceModel {
	out := []FaceModel{builtinFaceModels[DefaultFaceModelID]}
	for _, b := range appconfig.Get().ByoFaceModels {
		if strings.TrimSpace(b.ID) == "" || b.ModelPath == "" || b.Dim <= 0 {
			continue
		}
		if _, clash := builtinFaceModels[strings.TrimSpace(b.ID)]; clash {
			continue // BYO may not shadow a built-in ID
		}
		out = append(out, byoToFaceModel(b))
	}
	return out
}

// FaceRecognizerPath resolves the on-disk ONNX path for a recognizer: BYO
// models point at their configured file; built-ins go through the deps
// download flow. The error is user-actionable.
func FaceRecognizerPath(m FaceModel) (string, error) {
	if m.BYO {
		if _, err := os.Stat(m.Path); err != nil {
			return "", fmt.Errorf("BYO face model %q: file not found at %s", m.ID, m.Path)
		}
		return m.Path, nil
	}
	p, err := deps.ModelPath(m.DepID, m.File)
	if err != nil || p == "" {
		return "", fmt.Errorf("%s not installed; install it from Dependencies", m.DisplayName)
	}
	return p, nil
}

// FaceDetectorPath resolves the YuNet detector ONNX path via the deps flow.
func FaceDetectorPath() (string, error) {
	p, err := deps.ModelPath(FaceDetectorDepID, FaceDetectorFile)
	if err != nil || p == "" {
		return "", fmt.Errorf("YuNet face detector not installed; install it from Dependencies")
	}
	return p, nil
}
