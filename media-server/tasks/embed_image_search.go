package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"os"

	"github.com/stevecastle/shrike/deps"
)

// SearchByImage embeds an arbitrary image (e.g. a captured screen region) with
// the SigLIP 2 image encoder and returns the top-limit most similar media.
// Returns an error (not a panic) when the model or embed binary is absent.
func SearchByImage(ctx context.Context, db *sql.DB, image []byte, limit int) ([]SimilarHit, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("empty image")
	}
	imageModel, err := deps.ModelPath(EmbedModelID, "image_model.onnx")
	if err != nil {
		return nil, fmt.Errorf("image model not installed: %w", err)
	}
	if imageModel == "" {
		return nil, fmt.Errorf("image model not installed")
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed")
	}

	tmp, err := os.CreateTemp("", "region-*.png")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(image); err != nil {
		tmp.Close()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		return nil, err
	}

	vec, err := runEmbedSubprocess(ctx, embedBin, imageModel, ortLib, tmpPath, EmbedDim)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, EmbedModelID, vec, limit)
}
