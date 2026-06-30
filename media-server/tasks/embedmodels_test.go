package tasks

import (
	"database/sql"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func TestEmbedModelByID(t *testing.T) {
	if _, ok := EmbedModelByID("siglip2-base-patch16-224"); !ok {
		t.Fatal("expected siglip2 to be registered")
	}
	if m, ok := EmbedModelByID("dinov2-base"); !ok {
		t.Fatal("expected dinov2-base to be registered")
	} else if m.Multimodal {
		t.Error("dinov2-base must be image-only (not multimodal)")
	} else if m.Pooling != "cls" {
		t.Errorf("dinov2-base pooling = %q, want cls", m.Pooling)
	}
	if _, ok := EmbedModelByID("nope"); ok {
		t.Error("unknown model should not resolve")
	}
}

func TestActiveEmbedModelFallback(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	// Unknown / empty configured model -> default (siglip2).
	for _, id := range []string{"", "does-not-exist"} {
		cfg := prev
		cfg.EmbeddingModel = id
		appconfig.Set(cfg)
		if got := ActiveEmbedModel().ID; got != DefaultEmbedModelID {
			t.Errorf("EmbeddingModel=%q: active = %q, want default %q", id, got, DefaultEmbedModelID)
		}
	}

	// Known model -> that model.
	cfg := prev
	cfg.EmbeddingModel = "dinov2-base"
	appconfig.Set(cfg)
	if got := ActiveEmbedModel().ID; got != "dinov2-base" {
		t.Errorf("active = %q, want dinov2-base", got)
	}
}

func TestTextSearchModelAlwaysMultimodal(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	// Active model image-only -> text search falls back to the default multimodal.
	cfg := prev
	cfg.EmbeddingModel = "dinov2-base"
	appconfig.Set(cfg)
	tm := TextSearchModel()
	if !tm.Multimodal {
		t.Fatalf("text search model %q is not multimodal", tm.ID)
	}
	if tm.ID != DefaultEmbedModelID {
		t.Errorf("text search model = %q, want %q", tm.ID, DefaultEmbedModelID)
	}

	// Active model multimodal -> use it.
	cfg.EmbeddingModel = "siglip2-base-patch16-224"
	appconfig.Set(cfg)
	if got := TextSearchModel().ID; got != "siglip2-base-patch16-224" {
		t.Errorf("text search model = %q, want siglip2", got)
	}
}

func TestEmbedModelListOrder(t *testing.T) {
	list := EmbedModelList()
	if len(list) < 2 {
		t.Fatalf("expected >=2 models, got %d", len(list))
	}
	if list[0].ID != "siglip2-base-patch16-224" {
		t.Errorf("first model = %q, want siglip2 (default) first", list[0].ID)
	}
}

func TestEmbedModelOverrideFromJob(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
		ok   bool
	}{
		{"equals form", []string{"--query64=abc", "--model=dinov2-base"}, "dinov2-base", true},
		{"space form", []string{"--model", "dinov2-base"}, "dinov2-base", true},
		{"absent", []string{"--query64=abc"}, "", false},
		{"empty value", []string{"--model="}, "", false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			j := &jobqueue.Job{Arguments: tc.args}
			got, ok := embedModelOverrideFromJob(j)
			if ok != tc.ok || got != tc.want {
				t.Errorf("got (%q,%v), want (%q,%v)", got, ok, tc.want, tc.ok)
			}
		})
	}
}

// TestSearchByVectorModelGuard verifies the live ANN index is only consulted for
// the model it was built for; a search for a different model falls back to
// brute-force over that model's stored vectors (not the wrong index).
func TestSearchByVectorModelGuard(t *testing.T) {
	db, _ := sql.Open("sqlite", ":memory:")
	defer db.Close()
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}

	const modelA = "siglip2-base-patch16-224"
	const modelB = "dinov2-base"

	// Index (model A) ranks "a-hit" first for a +x query.
	idx := embedindex.New()
	idx.Add("a-hit.jpg", embedvecNormalize([]float32{1, 0}))
	SetVectorIndexForModel(idx, modelA)
	t.Cleanup(func() { SetVectorIndex(nil) })

	// DB vectors for model B: a different winner for the same query.
	_ = media.UpsertEmbedding(db, "b-hit.jpg", modelB, embedvecNormalize([]float32{1, 0}), 0)
	_ = media.UpsertEmbedding(db, "b-miss.jpg", modelB, embedvecNormalize([]float32{-1, 0}), 0)

	q := embedvecNormalize([]float32{1, 0})

	// Same model as the index -> uses the index -> sees a-hit.jpg.
	hitsA, err := SearchByVector(db, modelA, q, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hitsA) != 1 || hitsA[0].Path != "a-hit.jpg" {
		t.Errorf("model A search = %+v, want a-hit.jpg from index", hitsA)
	}

	// Different model -> index skipped -> brute-force over model B's DB vectors.
	hitsB, err := SearchByVector(db, modelB, q, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(hitsB) != 1 || hitsB[0].Path != "b-hit.jpg" {
		t.Errorf("model B search = %+v, want b-hit.jpg from brute-force (index must be skipped)", hitsB)
	}
}
