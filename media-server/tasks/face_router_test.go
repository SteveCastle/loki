package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func withAnchors(t *testing.T, anime, photo []float32) {
	t.Helper()
	SetAnchorOverrideForTest(anime, photo)
	t.Cleanup(func() { SetAnchorOverrideForTest(nil, nil) })
}

func TestClassifyVec(t *testing.T) {
	anchors := &anchorCache{anime: []float32{1, 0}, photo: []float32{0, 1}}
	if got := classifyVec([]float32{0.9, 0.1}, anchors); got != "anime" {
		t.Fatalf("near-anime vec classified %q", got)
	}
	if got := classifyVec([]float32{0.1, 0.9}, anchors); got != "photo" {
		t.Fatalf("near-photo vec classified %q", got)
	}
}

func TestFaceModelForDomain(t *testing.T) {
	// Active photo model keeps priority for photo items (BYO stays in charge).
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "sface"
		c.FaceRouting = "auto"
	})
	if m := faceModelForDomain("photo"); m.ID != "sface" {
		t.Fatalf("photo → %s, want sface", m.ID)
	}
	// Drawn media is never scanned: it routes to the skip sentinel.
	if m := faceModelForDomain("anime"); m.ID != DrawnSkipModelID || !m.Skip {
		t.Fatalf("anime → %+v, want the drawn skip sentinel", m)
	}

	// A config still naming the retired anime recognizer falls back to the
	// photo default for photos, and drawn media stays skipped.
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "anime-ccip"
		c.FaceRouting = "auto"
	})
	if m := faceModelForDomain("photo"); m.ID != DefaultFaceModelID {
		t.Fatalf("photo under retired anime config → %s, want %s", m.ID, DefaultFaceModelID)
	}
	if m := faceModelForDomain("anime"); !m.Skip {
		t.Fatalf("anime under retired anime config → %s, want skip", m.ID)
	}
}

func TestPartitionPathsByModel(t *testing.T) {
	db := newFaceIndexDB(t)
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "sface"
		c.FaceRouting = "auto"
	})
	withAnchors(t, []float32{1, 0}, []float32{0, 1})

	// Stored whole-image embeddings under the text-search (siglip) model:
	// two anime-looking, one photo-looking.
	embedModel := TextSearchModel().ID
	for path, vec := range map[string][]float32{
		"a1.jpg": {0.95, 0.05},
		"a2.jpg": {0.9, 0.1},
		"p1.jpg": {0.1, 0.9},
	} {
		if err := media.UpsertEmbedding(db, path, embedModel, vec, 0); err != nil {
			t.Fatal(err)
		}
	}

	groups, models, onTheFly, err := partitionPathsByModel(t.Context(), db, []string{"a1.jpg", "a2.jpg", "p1.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if onTheFly != 0 {
		t.Fatalf("embedded on the fly = %d, want 0 (all stored)", onTheFly)
	}
	if len(groups[DrawnSkipModelID]) != 2 || len(groups["sface"]) != 1 {
		t.Fatalf("groups = %v", groups)
	}
	if m, ok := models[DrawnSkipModelID]; !ok || !m.Skip {
		t.Fatal("drawn skip sentinel missing from models map")
	}

	// A path with no stored embedding AND no decodable file degrades to the
	// active model instead of being dropped.
	groups, _, _, err = partitionPathsByModel(t.Context(), db, []string{"missing.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups["sface"]) != 1 {
		t.Fatalf("unclassifiable path not degraded to active model: %v", groups)
	}

	// Routing off → everything under the active model regardless of vectors.
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "sface"
		c.FaceRouting = "single"
	})
	groups, _, _, err = partitionPathsByModel(t.Context(), db, []string{"a1.jpg", "p1.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups["sface"]) != 2 {
		t.Fatalf("routing=single groups = %v", groups)
	}
}

// CachedRoutedFaceModelForPath is the status-endpoint router: it must route
// from cache + stored vectors when it can, and degrade to the active model —
// never spawn a subprocess — when it can't.
func TestCachedRoutedFaceModelForPath(t *testing.T) {
	db := newFaceIndexDB(t)
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "sface"
		c.FaceRouting = "auto"
	})

	// No cached anchors (and no text encoder in tests to build them): even an
	// embedded anime item degrades to the active model rather than encoding.
	embedModel := TextSearchModel().ID
	if err := media.UpsertEmbedding(db, "a.jpg", embedModel, []float32{0.95, 0.05}, 0); err != nil {
		t.Fatal(err)
	}
	if m := CachedRoutedFaceModelForPath(db, "a.jpg"); m.ID != "sface" {
		t.Fatalf("no anchors → %s, want active model", m.ID)
	}

	withAnchors(t, []float32{1, 0}, []float32{0, 1})
	if m := CachedRoutedFaceModelForPath(db, "a.jpg"); m.ID != DrawnSkipModelID {
		t.Fatalf("cached route → %s, want %s", m.ID, DrawnSkipModelID)
	}
	// No stored embedding → active model (never embeds on the fly).
	if m := CachedRoutedFaceModelForPath(db, "missing.jpg"); m.ID != "sface" {
		t.Fatalf("no stored vector → %s, want active model", m.ID)
	}

	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "sface"
		c.FaceRouting = "single"
	})
	if m := CachedRoutedFaceModelForPath(db, "a.jpg"); m.ID != "sface" {
		t.Fatalf("routing=single → %s, want active model", m.ID)
	}
}

func TestGetEmbeddingsForPaths(t *testing.T) {
	db := newFaceIndexDB(t)
	if err := media.UpsertEmbedding(db, "x.jpg", "m", []float32{1, 2}, 0); err != nil {
		t.Fatal(err)
	}
	got, err := media.GetEmbeddingsForPaths(db, "m", []string{"x.jpg", "y.jpg"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got["x.jpg"][1] != 2 {
		t.Fatalf("got %v", got)
	}
}
