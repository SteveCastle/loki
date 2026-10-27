//go:build cgo

package onnxpose

import (
	"errors"
	"fmt"
	"image"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// Estimator is a loaded RTMPose session. Not safe for concurrent use.
type Estimator struct {
	session *ort.DynamicAdvancedSession
}

// NewEstimator loads an RTMPose ONNX model (input "input", outputs "simcc_x"
// and "simcc_y") on the CPU: one person's pose is a few milliseconds there,
// and it leaves the GPU to the generation engines.
func NewEstimator(modelPath, ortLib string) (*Estimator, error) {
	if modelPath == "" {
		return nil, errors.New("onnxpose: model path is required")
	}
	if ortLib != "" {
		ort.SetSharedLibraryPath(ortLib)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, err
		}
	}
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer so.Destroy()
	s, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input"}, []string{"simcc_x", "simcc_y"}, so)
	if err != nil {
		return nil, fmt.Errorf("onnxpose: loading %s: %w", modelPath, err)
	}
	return &Estimator{session: s}, nil
}

// Estimate is the pose of the person in box (x, y, w, h, image pixels) of img:
// the 17 COCO keypoints in image pixels.
func (e *Estimator) Estimate(img image.Image, x, y, w, h float64) ([]Keypoint, error) {
	c := CropFor(x, y, w, h)
	in, err := ort.NewTensor(ort.NewShape(1, 3, InputH, InputW), Preprocess(img, c))
	if err != nil {
		return nil, err
	}
	defer in.Destroy()
	outputs := []ort.Value{nil, nil}
	if err := e.session.Run([]ort.Value{in}, outputs); err != nil {
		return nil, err
	}
	data := make([][]float32, 2)
	for i, o := range outputs {
		if o == nil {
			return nil, errors.New("onnxpose: the model produced no output")
		}
		defer o.Destroy()
		t, ok := o.(*ort.Tensor[float32])
		if !ok {
			return nil, fmt.Errorf("onnxpose: unexpected output type %T", o)
		}
		data[i] = t.GetData()
	}
	return Decode(data[0], data[1], c), nil
}

// Destroy releases the session.
func (e *Estimator) Destroy() {
	if e != nil && e.session != nil {
		_ = e.session.Destroy()
	}
}
