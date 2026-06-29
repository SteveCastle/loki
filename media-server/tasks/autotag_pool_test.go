package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestTagsToTagInfos(t *testing.T) {
	in := []string{"1girl:0.99", "solo:0.87", "  ", "long_hair:0.40", "weird:tag:0.10"}
	got := tagsToTagInfos(in)
	// "weird:tag:0.10" → name is "weird:tag" (LastIndex strips only the score).
	want := []string{"1girl", "solo", "long_hair", "weird:tag"}
	if len(got) != len(want) {
		t.Fatalf("got %d tags, want %d (%+v)", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i].Label != w {
			t.Errorf("tag %d = %q, want %q", i, got[i].Label, w)
		}
		if got[i].Category != "Suggested" {
			t.Errorf("tag %d category = %q, want Suggested", i, got[i].Category)
		}
	}
}

func TestResolveAutotagResourcesIndependentFromEmbed(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	cfg := prev
	cfg.AutotagProvider = "cpu"
	cfg.AutotagPerformance = "custom"
	cfg.AutotagWorkers = 6
	cfg.AutotagThreadsPerWorker = 2
	// Different embedding settings must NOT affect autotag resolution.
	cfg.EmbeddingPerformance = "low"
	cfg.EmbeddingWorkers = 1
	appconfig.Set(cfg)

	if w, th := ResolveAutotagResources(); w != 6 || th != 2 {
		t.Errorf("autotag custom = (%d,%d), want (6,2)", w, th)
	}
	if AutotagProviderFromConfig() != "cpu" {
		t.Errorf("autotag provider = %q, want cpu", AutotagProviderFromConfig())
	}
}
