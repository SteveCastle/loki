//go:build cgo

package onnxtag

import (
	"errors"
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
