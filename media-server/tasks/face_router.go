package tasks

// Face-domain routing: decide per media item whether it shows PHOTOGRAPHIC
// people (photo-trained detector/recognizer) or DRAWN characters (anime
// pipeline), so one `faces` job produces correct clusters for both.
//
// The classifier is a SigLIP text probe: the item's whole-image SigLIP
// embedding — already stored for most of the library by the embed task — is
// compared against two cached anchor vectors built from prompt sets ("an
// anime illustration…" vs "a photograph…"). For embedded items routing is a
// pure cosine (free); unembedded items are embedded on the fly, which also
// persists the vector for search. When the text encoder or embed binary is
// unavailable the router degrades to the active model with a warning.

import (
	"context"
	"database/sql"
	"fmt"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
)

// Anchor prompt sets. Several prompts per class, averaged, so no single
// phrasing dominates. These describe the IMAGE STYLE, not the content.
var (
	animeAnchorPrompts = []string{
		"an anime illustration of a character",
		"a cartoon drawing",
		"digital anime artwork, cel shaded",
	}
	photoAnchorPrompts = []string{
		"a photograph of a person",
		"a photo taken with a camera",
		"a realistic photograph",
	}
)

// FaceRoutingEnabled reports whether per-item domain routing is on
// ("auto", the default). "single" pins everything to the active model.
func FaceRoutingEnabled() bool {
	return appconfig.Get().FaceRouting != "single"
}

// anchorCache holds the encoded anchor vectors for one text-search model.
type anchorCache struct {
	model string // text model the anchors were encoded with
	anime []float32
	photo []float32
}

var (
	anchorMu     sync.Mutex
	cachedAnchor *anchorCache
	// anchorOverride lets tests inject anchors without a text encoder.
	anchorOverride *anchorCache
)

// SetAnchorOverrideForTest injects fixed anchor vectors (nil clears).
func SetAnchorOverrideForTest(anime, photo []float32) {
	anchorMu.Lock()
	defer anchorMu.Unlock()
	if anime == nil {
		anchorOverride = nil
		return
	}
	anchorOverride = &anchorCache{anime: anime, photo: photo}
}

// domainAnchors returns the (cached) anchor vectors, encoding them with the
// text-search model on first use.
func domainAnchors(ctx context.Context) (*anchorCache, error) {
	anchorMu.Lock()
	defer anchorMu.Unlock()
	if anchorOverride != nil {
		return anchorOverride, nil
	}
	textModel := TextSearchModel().ID
	if cachedAnchor != nil && cachedAnchor.model == textModel {
		return cachedAnchor, nil
	}
	encodeSet := func(prompts []string) ([]float32, error) {
		var sum []float32
		for _, prompt := range prompts {
			vec, _, err := TextQueryVector(ctx, prompt)
			if err != nil {
				return nil, err
			}
			if sum == nil {
				sum = make([]float32, len(vec))
			}
			for i := range vec {
				sum[i] += vec[i]
			}
		}
		return embedvec.Normalize(sum), nil
	}
	anime, err := encodeSet(animeAnchorPrompts)
	if err != nil {
		return nil, fmt.Errorf("encode anime anchors: %w", err)
	}
	photo, err := encodeSet(photoAnchorPrompts)
	if err != nil {
		return nil, fmt.Errorf("encode photo anchors: %w", err)
	}
	cachedAnchor = &anchorCache{model: textModel, anime: anime, photo: photo}
	return cachedAnchor, nil
}

// classifyVec assigns a domain from a whole-image embedding: whichever anchor
// is nearer wins. Pure function — the unit-testable core of the router.
func classifyVec(vec []float32, anchors *anchorCache) string {
	if embedvec.CosineSim(vec, anchors.anime) > embedvec.CosineSim(vec, anchors.photo) {
		return "anime"
	}
	return "photo"
}

// faceModelForDomain maps a domain to a recognizer: the active model when its
// domain matches (so a BYO photo recognizer keeps priority), otherwise the
// built-in for that domain.
func faceModelForDomain(domain string) FaceModel {
	active := ActiveFaceModel()
	if active.Domain == domain || (domain == "photo" && active.Domain == "") {
		return active
	}
	if domain == "anime" {
		if m, ok := FaceModelByID("anime-ccip"); ok {
			return m
		}
	}
	if m, ok := FaceModelByID(DefaultFaceModelID); ok {
		return m
	}
	return active
}

// routeVec picks the face model for one whole-image embedding.
func routeVec(vec []float32, anchors *anchorCache) FaceModel {
	return faceModelForDomain(classifyVec(vec, anchors))
}

// RoutedFaceModelForPath picks the face model for one media item. With
// routing off (or on any classification failure) it returns the active model.
// Items without a stored SigLIP embedding are embedded on the fly (and the
// vector persisted, so the next call is free).
func RoutedFaceModelForPath(ctx context.Context, db *sql.DB, path string) FaceModel {
	if !FaceRoutingEnabled() {
		return ActiveFaceModel()
	}
	anchors, err := domainAnchors(ctx)
	if err != nil {
		return ActiveFaceModel()
	}
	vec, err := ImageQueryVectorForPath(ctx, db, TextSearchModel(), path)
	if err != nil {
		return ActiveFaceModel()
	}
	return routeVec(vec, anchors)
}

// RoutedFaceModelForBytes picks the face model for an uploaded query image
// (region capture, search upload). Falls back to the active model.
func RoutedFaceModelForBytes(ctx context.Context, image []byte) FaceModel {
	if !FaceRoutingEnabled() {
		return ActiveFaceModel()
	}
	anchors, err := domainAnchors(ctx)
	if err != nil {
		return ActiveFaceModel()
	}
	vec, err := ImageQueryVectorForBytes(ctx, TextSearchModel(), image)
	if err != nil {
		return ActiveFaceModel()
	}
	return routeVec(vec, anchors)
}

// partitionPathsByModel groups paths by their routed face model, resolving
// stored embeddings in one batch and embedding stragglers on the fly. Paths
// whose classification fails land under the active model so a broken embed
// pipeline degrades to pre-routing behavior instead of dropping media.
// Returns model-ID → paths plus the models used, and how many items needed
// an on-the-fly embed (for job logging).
func partitionPathsByModel(ctx context.Context, db *sql.DB, paths []string) (map[string][]string, map[string]FaceModel, int, error) {
	groups := map[string][]string{}
	models := map[string]FaceModel{}
	add := func(m FaceModel, p string) {
		if _, ok := models[m.ID]; !ok {
			models[m.ID] = m
		}
		groups[m.ID] = append(groups[m.ID], p)
	}

	if !FaceRoutingEnabled() {
		m := ActiveFaceModel()
		for _, p := range paths {
			add(m, p)
		}
		return groups, models, 0, nil
	}
	anchors, err := domainAnchors(ctx)
	if err != nil {
		// No text encoder — degrade to single-model behavior.
		m := ActiveFaceModel()
		for _, p := range paths {
			add(m, p)
		}
		return groups, models, 0, err
	}

	embedModel := TextSearchModel()
	stored, err := media.GetEmbeddingsForPaths(db, embedModel.ID, paths)
	if err != nil {
		return nil, nil, 0, err
	}
	embeddedOnTheFly := 0
	for _, p := range paths {
		if ctx.Err() != nil {
			return nil, nil, embeddedOnTheFly, ctx.Err()
		}
		vec, ok := stored[p]
		if !ok {
			fresh, err := ImageQueryVectorForPath(ctx, db, embedModel, p)
			if err != nil {
				add(ActiveFaceModel(), p) // classification failed — degrade
				continue
			}
			embeddedOnTheFly++
			vec = fresh
		}
		add(routeVec(vec, anchors), p)
	}
	return groups, models, embeddedOnTheFly, nil
}
