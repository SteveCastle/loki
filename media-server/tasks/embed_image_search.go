package tasks

import (
	"context"
	"database/sql"
	"fmt"
	"os"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
)

// ImageQueryVectorForBytes embeds an arbitrary image (e.g. a captured screen
// region) with m's image encoder via a temp file and returns the vector.
// Returns an error (not a panic) when the model or embed binary is absent.
func ImageQueryVectorForBytes(ctx context.Context, m EmbedModel, image []byte) ([]float32, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("empty image")
	}
	imageModel, err := deps.ModelPath(m.ID, m.ImageModelFile)
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

	return runEmbedSubprocess(ctx, embedBin, imageModel, ortLib, tmpPath, m)
}

// SearchByImage embeds an arbitrary image (e.g. a captured screen region) with
// the active model's image encoder and returns the top-limit most similar media.
// Returns an error (not a panic) when the model or embed binary is absent.
func SearchByImage(ctx context.Context, db *sql.DB, image []byte, limit int) ([]SimilarHit, error) {
	model := ActiveEmbedModel()
	vec, err := ImageQueryVectorForBytes(ctx, model, image)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, model.ID, vec, limit)
}

// SearchByPathAndText blends the image embedding of a library item with a text
// embedding into one query vector ((1-w)*image + w*text, renormalized) and
// returns the top-limit most similar media. textWeight 0 = pure image,
// 1 = pure text. Both vectors come from the multimodal text-search model so
// they share one embedding space.
func SearchByPathAndText(ctx context.Context, db *sql.DB, path, text string, textWeight float32, limit int) ([]SimilarHit, error) {
	tvec, m, err := TextQueryVector(ctx, text)
	if err != nil {
		return nil, err
	}
	ivec, err := ImageQueryVectorForPath(ctx, db, m, path)
	if err != nil {
		return nil, err
	}
	blended, err := embedvec.Blend(ivec, tvec, textWeight)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, m.ID, blended, limit)
}

// SearchByImageAndText is SearchByPathAndText for raw image bytes (a captured
// screen region) instead of a library item.
func SearchByImageAndText(ctx context.Context, db *sql.DB, image []byte, text string, textWeight float32, limit int) ([]SimilarHit, error) {
	tvec, m, err := TextQueryVector(ctx, text)
	if err != nil {
		return nil, err
	}
	ivec, err := ImageQueryVectorForBytes(ctx, m, image)
	if err != nil {
		return nil, err
	}
	blended, err := embedvec.Blend(ivec, tvec, textWeight)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, m.ID, blended, limit)
}
