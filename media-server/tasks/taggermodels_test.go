package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestActiveTaggerModelFallback(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	// Unknown / empty configured model -> default.
	for _, id := range []string{"", "does-not-exist"} {
		cfg := prev
		cfg.AutotagModel = id
		appconfig.Set(cfg)
		if got := ActiveTaggerModel().ID; got != DefaultTaggerModelID {
			t.Errorf("AutotagModel=%q: active = %q, want default %q", id, got, DefaultTaggerModelID)
		}
	}

	// Known model -> that model, with its files populated.
	cfg := prev
	cfg.AutotagModel = "wd-eva02-large-tagger-v3"
	appconfig.Set(cfg)
	m := ActiveTaggerModel()
	if m.ID != "wd-eva02-large-tagger-v3" {
		t.Fatalf("active = %q, want wd-eva02-large-tagger-v3", m.ID)
	}
	if m.ModelFile == "" || m.LabelsFile == "" || m.ConfigFile == "" {
		t.Errorf("tagger model files incomplete: %+v", m)
	}
}

func TestTaggerModelList(t *testing.T) {
	list := TaggerModelList()
	if len(list) < 1 {
		t.Fatal("expected at least one tagger model")
	}
	if list[0].ID != DefaultTaggerModelID {
		t.Errorf("first model = %q, want default %q first", list[0].ID, DefaultTaggerModelID)
	}
	if list[0].DisplayName == "" {
		t.Error("display name should be a human-friendly label, not empty")
	}
}
