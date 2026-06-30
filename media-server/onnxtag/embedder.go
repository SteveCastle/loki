//go:build cgo

package onnxtag

import (
	"errors"
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// EmbedProvider selects the ONNX Runtime execution provider for an Embedder.
type EmbedProvider string

const (
	ProviderCPU      EmbedProvider = "cpu"
	ProviderDirectML EmbedProvider = "directml"
)

// EmbedderConfig configures a persistent Embedder.
type EmbedderConfig struct {
	ModelPath string
	Opts      Options // image preprocessing + input/output tensor names
	Dim       int
	Pooling   string // "" / "none" (output is [1,Dim]) or "cls" (token 0 of [1,N,Dim])
	Provider  EmbedProvider
	Threads   int    // CPU intra-op/inter-op threads; 0 = ORT default
	Device    int    // GPU device id for DirectML (0 = primary)
	ORTLib    string // path to the onnxruntime shared library (overrides global)
}

// Embedder holds an ONNX session that is loaded ONCE and reused across many
// images, eliminating the per-image model-reload that dominated the old
// spawn-per-image pipeline. It is NOT safe for concurrent use — create one
// Embedder per worker goroutine/process. Close() must be called to release the
// session and the (process-global) ONNX environment.
type Embedder struct {
	session *ort.DynamicAdvancedSession
	opts    Options
	dim     int
	pooling string
	envInit bool
}

// NewEmbedder loads the model and prepares a reusable session with the requested
// execution provider and thread settings.
func NewEmbedder(cfg EmbedderConfig) (*Embedder, error) {
	if cfg.Dim <= 0 {
		return nil, errors.New("onnxtag: Dim must be > 0")
	}
	if cfg.Opts.InputName == "" || cfg.Opts.OutputName == "" {
		return nil, errors.New("onnxtag: input and output names must be provided")
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

	return &Embedder{
		session: session,
		opts:    cfg.Opts,
		dim:     cfg.Dim,
		pooling: cfg.Pooling,
		envInit: true,
	}, nil
}

// newSessionOptionsFor builds ONNX SessionOptions for a provider + thread/device
// config. Shared by Embedder and Classifier. The caller owns Destroy().
func newSessionOptionsFor(provider EmbedProvider, threads, device int) (*ort.SessionOptions, error) {
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	switch provider {
	case ProviderDirectML:
		// DirectML requires sequential execution and no memory pattern (the
		// allocations are device-side). Intra-op threads don't apply to GPU
		// compute; preprocessing stays on the CPU caller side.
		_ = so.SetExecutionMode(ort.ExecutionModeSequential)
		_ = so.SetMemPattern(false)
		if err := so.AppendExecutionProviderDirectML(device); err != nil {
			so.Destroy()
			return nil, fmt.Errorf("directml provider unavailable: %w", err)
		}
	default: // CPU
		if threads > 0 {
			_ = so.SetIntraOpNumThreads(threads)
			_ = so.SetInterOpNumThreads(threads)
		}
	}
	return so, nil
}

// Embed runs the model on one image and returns the raw (un-normalized) vector.
// For both pooling modes the embedding is the first Dim floats of the output:
// "none" → the pooled [1,Dim] output is the vector; "cls" → token 0 of the
// [1,N,Dim] sequence is the first row. The caller L2-normalizes.
func (e *Embedder) Embed(imagePath string) ([]float32, error) {
	imgTensor, err := loadImageAsTensor(imagePath, e.opts)
	if err != nil {
		return nil, err
	}
	defer imgTensor.Destroy()

	outputs := []ort.Value{nil}
	if err := e.session.Run([]ort.Value{imgTensor}, outputs); err != nil {
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
	shape := tensor.GetShape()
	data := tensor.GetData()
	if len(shape) < 1 {
		return nil, fmt.Errorf("onnxtag: empty output shape")
	}
	lastDim := int(shape[len(shape)-1])
	if lastDim != e.dim {
		return nil, fmt.Errorf("onnxtag: output last dim %d != expected %d", lastDim, e.dim)
	}
	if len(data) < e.dim {
		return nil, fmt.Errorf("onnxtag: output has %d floats, need %d", len(data), e.dim)
	}
	vec := make([]float32, e.dim)
	copy(vec, data[:e.dim])
	return vec, nil
}

// Close releases the session and the process-global ONNX environment.
func (e *Embedder) Close() error {
	var err error
	if e.session != nil {
		err = e.session.Destroy()
		e.session = nil
	}
	if e.envInit {
		ort.DestroyEnvironment()
		e.envInit = false
	}
	return err
}
