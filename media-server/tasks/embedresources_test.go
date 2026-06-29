package tasks

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestPresetResources(t *testing.T) {
	cases := []struct {
		preset      string
		cpus        int
		wantWorkers int
		wantThreads int
	}{
		{"low", 8, 1, 2},      // target=2 → 1 worker × 2 threads
		{"balanced", 8, 1, 4}, // target=4 → 1 × 4
		{"max", 8, 2, 4},      // target=7 → 2 × 4
		{"low", 4, 1, 1},      // target=1 → 1 × 1
		{"balanced", 16, 2, 4},// target=8 → 2 × 4
		{"max", 32, 8, 4},     // target=31 → 8 × 4
		{"balanced", 1, 1, 1}, // tiny host
	}
	for _, c := range cases {
		w, th := presetResources(c.preset, c.cpus)
		if w != c.wantWorkers || th != c.wantThreads {
			t.Errorf("presetResources(%q,%d) = (%d,%d), want (%d,%d)", c.preset, c.cpus, w, th, c.wantWorkers, c.wantThreads)
		}
		if w*th < 1 {
			t.Errorf("presetResources(%q,%d) product < 1", c.preset, c.cpus)
		}
	}
}

func TestResolveEmbedResourcesCustomAndDirectML(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })

	// Custom overrides are honored verbatim on CPU.
	cfg := prev
	cfg.EmbeddingProvider = "cpu"
	cfg.EmbeddingPerformance = "custom"
	cfg.EmbeddingWorkers = 3
	cfg.EmbeddingThreadsPerWorker = 5
	appconfig.Set(cfg)
	if w, th := ResolveEmbedResources(); w != 3 || th != 5 {
		t.Errorf("custom cpu = (%d,%d), want (3,5)", w, th)
	}

	// DirectML caps workers low (GPU is the shared resource); threads preserved.
	cfg.EmbeddingProvider = "directml"
	cfg.EmbeddingWorkers = 8
	cfg.EmbeddingThreadsPerWorker = 4
	appconfig.Set(cfg)
	if w, th := ResolveEmbedResources(); w != 2 || th != 4 {
		t.Errorf("custom directml = (%d,%d), want workers capped to 2, threads 4", w, th)
	}
}

func TestEmbedProviderFromConfig(t *testing.T) {
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })
	for in, want := range map[string]string{
		"":         "cpu",
		"cpu":      "cpu",
		"directml": "directml",
		"DirectML": "directml",
		"gpu":      "directml",
		"weird":    "cpu",
	} {
		cfg := prev
		cfg.EmbeddingProvider = in
		appconfig.Set(cfg)
		if got := EmbedProviderFromConfig(); got != want {
			t.Errorf("EmbedProviderFromConfig(%q) = %q, want %q", in, got, want)
		}
	}
}
