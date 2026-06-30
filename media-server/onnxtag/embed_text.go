//go:build cgo

package onnxtag

import (
	"errors"
	"os"

	"github.com/eliben/go-sentencepiece"
	ort "github.com/yalue/onnxruntime_go"
)

// EmbedText tokenizes text with the SigLIP 2 Gemma SentencePiece model and runs
// the text encoder, returning the raw output vector of length outputDim. The
// caller L2-normalizes. seqLen is the fixed sequence length (64 for SigLIP 2).
// opts.InputName/OutputName must be set (input_ids/pooler_output for SigLIP 2).
func EmbedText(textModelPath, tokenizerPath, text string, opts Options, outputDim, seqLen int) ([]float32, error) {
	if outputDim <= 0 || seqLen <= 0 {
		return nil, errors.New("onnxtag: outputDim and seqLen must be > 0")
	}
	if opts.InputName == "" || opts.OutputName == "" {
		return nil, errors.New("onnxtag: input and output names must be provided")
	}

	proc, err := sentencepiece.NewProcessorFromPath(tokenizerPath)
	if err != nil {
		return nil, err
	}
	ids := BuildTextInputIDs(proc, text, seqLen)

	if opts.ORTSharedLibraryPath != "" {
		ort.SetSharedLibraryPath(opts.ORTSharedLibraryPath)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	inTensor, err := ort.NewTensor(ort.NewShape(1, int64(seqLen)), ids)
	if err != nil {
		return nil, err
	}
	defer inTensor.Destroy()

	out, err := ort.NewEmptyTensor[float32](ort.NewShape(1, int64(outputDim)))
	if err != nil {
		return nil, err
	}
	defer out.Destroy()

	session, err := ort.NewAdvancedSession(textModelPath,
		[]string{opts.InputName}, []string{opts.OutputName},
		[]ort.Value{inTensor}, []ort.Value{out}, nil)
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
