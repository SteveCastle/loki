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

// QueryTerm is one component of a composite latent-space query: a library
// item ("path"), an uploaded image ("image", e.g. a captured region), or free
// text ("text"). Weight is SIGNED — negative terms steer the query away from
// that concept.
type QueryTerm struct {
	Kind   string // "path" | "image" | "text"
	Value  string // media path or text, by Kind
	Image  []byte // raw image bytes when Kind == "image"
	Weight float32
}

// SearchByComposite combines any number of image/text terms into ONE query
// vector (normalized signed weighted sum — see embedvec.Combine) and returns
// the top-limit most similar media. With any text term present all terms are
// embedded with the multimodal text-search model so they share a space;
// otherwise the active embed model is used (matching plain similar: search).
func SearchByComposite(ctx context.Context, db *sql.DB, terms []QueryTerm, limit int) ([]SimilarHit, error) {
	if len(terms) == 0 {
		return nil, fmt.Errorf("composite query has no terms")
	}
	m := ActiveEmbedModel()
	for _, t := range terms {
		if t.Kind == "text" {
			m = TextSearchModel()
			break
		}
	}
	vecs := make([][]float32, 0, len(terms))
	weights := make([]float32, 0, len(terms))
	for _, t := range terms {
		var vec []float32
		var err error
		switch t.Kind {
		case "path":
			vec, err = ImageQueryVectorForPath(ctx, db, m, t.Value)
		case "image":
			vec, err = ImageQueryVectorForBytes(ctx, m, t.Image)
		case "text":
			// TextQueryVector always encodes with the text-search model, which
			// is what m resolved to whenever a text term exists.
			vec, _, err = TextQueryVector(ctx, t.Value)
		default:
			err = fmt.Errorf("unknown query term kind %q", t.Kind)
		}
		if err != nil {
			return nil, fmt.Errorf("composite term (%s %q): %w", t.Kind, truncateForLog(t.Value), err)
		}
		vecs = append(vecs, vec)
		weights = append(weights, t.Weight)
	}
	combined, err := embedvec.Combine(vecs, weights)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, m.ID, combined, limit)
}

func truncateForLog(s string) string {
	if len(s) > 80 {
		return s[:80] + "…"
	}
	return s
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
