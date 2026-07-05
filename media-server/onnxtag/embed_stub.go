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

// EmbedProvider mirrors the cgo type so non-cgo callers compile.
type EmbedProvider string

const (
	ProviderCPU      EmbedProvider = "cpu"
	ProviderDirectML EmbedProvider = "directml"
)

// EmbedderConfig mirrors the cgo type (embedder.go).
type EmbedderConfig struct {
	ModelPath string
	Opts      Options
	Dim       int
	Pooling   string
	Provider  EmbedProvider
	Threads   int
	Device    int
	ORTLib    string
}

// Embedder is unavailable without cgo. The real implementation lives in
// embedder.go (//go:build cgo).
type Embedder struct{}

// NewEmbedder returns ErrCGORequired in non-cgo builds.
func NewEmbedder(cfg EmbedderConfig) (*Embedder, error) { return nil, ErrCGORequired }

// Embed returns ErrCGORequired in non-cgo builds.
func (e *Embedder) Embed(imagePath string) ([]float32, error) { return nil, ErrCGORequired }

// Close is a no-op in non-cgo builds.
func (e *Embedder) Close() error { return nil }

// ClassifierConfig mirrors the cgo type (classifier.go).
type ClassifierConfig struct {
	ModelPath string
	Opts      Options
	Provider  EmbedProvider
	Threads   int
	Device    int
	ORTLib    string
}

// Classifier is unavailable without cgo. The real implementation lives in
// classifier.go (//go:build cgo).
type Classifier struct{}

// NewClassifier returns ErrCGORequired in non-cgo builds.
func NewClassifier(cfg ClassifierConfig) (*Classifier, error) { return nil, ErrCGORequired }

// Classify returns ErrCGORequired in non-cgo builds.
func (c *Classifier) Classify(imagePath string) ([]string, error) { return nil, ErrCGORequired }

// Close is a no-op in non-cgo builds.
func (c *Classifier) Close() error { return nil }
