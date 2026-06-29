//go:build !cgo

package onnxtag

// EmbedImage is unavailable without cgo. The real implementation lives in
// embed.go (//go:build cgo). ErrCGORequired is declared in stub.go.
func EmbedImage(modelPath, imagePath string, opts Options, outputDim int) ([]float32, error) {
	return nil, ErrCGORequired
}

// EmbedImageCLS is unavailable without cgo. The real implementation lives in
// embed.go (//go:build cgo).
func EmbedImageCLS(modelPath, imagePath string, opts Options, outputDim int) ([]float32, error) {
	return nil, ErrCGORequired
}

// EmbedText is unavailable without cgo. The real implementation lives in
// embed_text.go (//go:build cgo).
func EmbedText(textModelPath, tokenizerPath, text string, opts Options, outputDim, seqLen int) ([]float32, error) {
	return nil, ErrCGORequired
}
