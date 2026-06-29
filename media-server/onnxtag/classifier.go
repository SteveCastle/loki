//go:build cgo

package onnxtag

import (
	"errors"
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// ClassifierConfig configures a persistent Classifier.
type ClassifierConfig struct {
	ModelPath string
	Opts      Options // preprocessing, input/output names, labels, thresholds, category indices
	Provider  EmbedProvider
	Threads   int
	Device    int
	ORTLib    string
}

// Classifier holds an ONNX tagging session loaded ONCE and reused across many
// images, eliminating the per-image model reload of the old spawn-per-image
// autotag pipeline. NOT safe for concurrent use — one Classifier per worker.
// Close() must be called to release the session and the ONNX environment.
type Classifier struct {
	session *ort.DynamicAdvancedSession
	opts    Options
	envInit bool
}

// NewClassifier loads the model and prepares a reusable session. Labels,
// category indices, and thresholds come from cfg.Opts (loaded once by the
// caller) and are applied per image in Classify.
func NewClassifier(cfg ClassifierConfig) (*Classifier, error) {
	if cfg.Opts.InputName == "" || cfg.Opts.OutputName == "" {
		return nil, errors.New("onnxtag: input and output names must be provided")
	}
	if cfg.Opts.InputWidth <= 0 || cfg.Opts.InputHeight <= 0 {
		return nil, fmt.Errorf("onnxtag: invalid input size %dx%d", cfg.Opts.InputWidth, cfg.Opts.InputHeight)
	}

	if cfg.ORTLib != "" {
		ort.SetSharedLibraryPath(cfg.ORTLib)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	so, err := newSessionOptionsFor(cfg.Provider, cfg.Threads, cfg.Device)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, err
	}
	defer so.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{cfg.Opts.InputName},
		[]string{cfg.Opts.OutputName},
		so,
	)
	if err != nil {
		ort.DestroyEnvironment()
		return nil, err
	}

	return &Classifier{session: session, opts: cfg.Opts, envInit: true}, nil
}

// Classify runs the model on one image and returns its ranked tag strings
// (identical formatting to the one-shot ClassifyImage). A dynamic session lets
// the runtime allocate the [1,numClasses] output, so the class count is read
// from the model rather than pre-sized.
func (c *Classifier) Classify(imagePath string) ([]string, error) {
	imgTensor, err := loadImageAsTensor(imagePath, c.opts)
	if err != nil {
		return nil, err
	}
	defer imgTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := c.session.Run([]ort.Value{imgTensor}, outputs); err != nil {
		return nil, err
	}
	out := outputs[0]
	if out == nil {
		return nil, errors.New("onnxtag: model produced no output")
	}
	defer out.Destroy()

	tensor, ok := out.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("onnxtag: unexpected output type %T (want float32 tensor)", out)
	}
	src := tensor.GetData()
	scores := make([]float32, len(src))
	copy(scores, src) // copy out of the tensor backing memory before Destroy
	return scoresToTags(scores, c.opts), nil
}

// Close releases the session and the process-global ONNX environment.
func (c *Classifier) Close() error {
	var err error
	if c.session != nil {
		err = c.session.Destroy()
		c.session = nil
	}
	if c.envInit {
		ort.DestroyEnvironment()
		c.envInit = false
	}
	return err
}
