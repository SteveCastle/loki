//go:build !cgo

package onnxtag

// EmbedImage is unavailable without cgo. The real implementation lives in
// embed.go (//go:build cgo). ErrCGORequired is declared in stub.go.
func EmbedImage(modelPath, imagePath string, opts Options, outputDim int) ([]float32, error) {
	return nil, ErrCGORequired
}
