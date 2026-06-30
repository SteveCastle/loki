//go:build cgo

package onnxtag

import (
	"errors"
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// EmbedImage runs an embedding model (e.g. a SigLIP image encoder) on imagePath
// and returns the raw output vector of length outputDim. Preprocessing reuses
// loadImageAsTensor, so the SigLIP normalization (mean/std 0.5, NCHW, RGB,
// 224x224) is configured through opts by the caller. The caller is responsible
// for L2-normalizing the result.
func EmbedImage(modelPath, imagePath string, opts Options, outputDim int) ([]float32, error) {
	if outputDim <= 0 {
		return nil, errors.New("onnxtag: outputDim must be > 0")
	}
	if opts.InputName == "" || opts.OutputName == "" {
		return nil, errors.New("onnxtag: input and output names must be provided")
	}

	if opts.ORTSharedLibraryPath != "" {
		ort.SetSharedLibraryPath(opts.ORTSharedLibraryPath)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	imgTensor, err := loadImageAsTensor(imagePath, opts)
	if err != nil {
		return nil, err
	}
	defer imgTensor.Destroy()

	outShape := ort.NewShape(1, int64(outputDim))
	out, err := ort.NewEmptyTensor[float32](outShape)
	if err != nil {
		return nil, err
	}
	defer out.Destroy()

	session, err := ort.NewAdvancedSession(
		modelPath,
		[]string{opts.InputName},
		[]string{opts.OutputName},
		[]ort.Value{imgTensor},
		[]ort.Value{out},
		nil,
	)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, err
	}
	src := out.GetData()
	vec := make([]float32, len(src))
	copy(vec, src) // copy out of the tensor's backing memory before Destroy
	return vec, nil
}

// EmbedImageCLS runs an embedding model whose output is a token *sequence*
// (shape [1, N, outputDim], e.g. a DINOv2 last_hidden_state) and returns the
// CLS token — token index 0 — as the embedding. Unlike EmbedImage it cannot
// pre-allocate the output (N depends on the image/patch size), so it uses a
// dynamic session and lets the runtime allocate the output tensor. The caller
// is responsible for L2-normalizing the result.
func EmbedImageCLS(modelPath, imagePath string, opts Options, outputDim int) ([]float32, error) {
	if outputDim <= 0 {
		return nil, errors.New("onnxtag: outputDim must be > 0")
	}
	if opts.InputName == "" || opts.OutputName == "" {
		return nil, errors.New("onnxtag: input and output names must be provided")
	}

	if opts.ORTSharedLibraryPath != "" {
		ort.SetSharedLibraryPath(opts.ORTSharedLibraryPath)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	imgTensor, err := loadImageAsTensor(imagePath, opts)
	if err != nil {
		return nil, err
	}
	defer imgTensor.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{opts.InputName},
		[]string{opts.OutputName},
		nil,
	)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	// Nil output → the runtime allocates it; we reclaim and read it below.
	outputs := []ort.Value{nil}
	if err := session.Run([]ort.Value{imgTensor}, outputs); err != nil {
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
	// Expect [1, N, outputDim] (or [N, outputDim]); the CLS token is row 0, i.e.
	// the first outputDim contiguous floats. Guard against a too-small output.
	if len(shape) < 2 {
		return nil, fmt.Errorf("onnxtag: expected a sequence output, got shape %v", shape)
	}
	lastDim := int(shape[len(shape)-1])
	if lastDim != outputDim {
		return nil, fmt.Errorf("onnxtag: output last dim %d != expected %d", lastDim, outputDim)
	}
	if len(data) < outputDim {
		return nil, fmt.Errorf("onnxtag: output has %d floats, need %d", len(data), outputDim)
	}
	vec := make([]float32, outputDim)
	copy(vec, data[:outputDim]) // CLS = token 0 = first row
	return vec, nil
}
