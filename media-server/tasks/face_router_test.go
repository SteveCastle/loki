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
	if m := faceModelForDomain("anime"); m.ID != "anime-ccip" {
		t.Fatalf("anime → %s, want anime-ccip", m.ID)
	}

	// Active ANIME model: anime items keep it; photo items fall back to the
	// photo default rather than scanning photos with CCIP.
	setFaceConfig(t, func(c *appconfig.Config) {
		c.FaceModel = "anime-ccip"
		c.FaceRouting = "auto"
	})
	if m := faceModelForDomain("anime"); m.ID != "anime-ccip" {
		t.Fatalf("anime under anime-active → %s", m.ID)
	}
	if m := faceModelForDomain("photo"); m.ID != DefaultFaceModelID {
		t.Fatalf("photo under anime-active → %s, want %s", m.ID, DefaultFaceModelID)
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
	if len(groups["anime-ccip"]) != 2 || len(groups["sface"]) != 1 {
		t.Fatalf("groups = %v", groups)
	}
	if _, ok := models["anime-ccip"]; !ok {
		t.Fatal("anime model missing from models map")
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
