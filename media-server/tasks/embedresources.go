package tasks

import (
	"math"
	"runtime"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

// OnnxFileTimeout is the per-file processing timeout for local ONNX tasks
// (embed, autotag). Zero means disabled. A file exceeding it is skipped and the
// worker is restarted, so one bad file can't stall a whole job.
func OnnxFileTimeout() time.Duration {
	s := appconfig.Get().OnnxFileTimeoutSeconds
	if s <= 0 {
		return 0
	}
	return time.Duration(s) * time.Second
}

// normalizeProvider maps a raw provider string to "cpu" or "directml".
func normalizeProvider(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "directml", "dml", "gpu":
		return "directml"
	default:
		return "cpu"
	}
}

// EmbedProviderFromConfig returns the configured embedding execution provider.
func EmbedProviderFromConfig() string {
	return normalizeProvider(appconfig.Get().EmbeddingProvider)
}

// AutotagProviderFromConfig returns the configured auto-tagging execution provider.
func AutotagProviderFromConfig() string {
	return normalizeProvider(appconfig.Get().AutotagProvider)
}

// ResolveEmbedResources maps the configured performance preset (or custom
// overrides) to a concrete (workers, threads) pair, scaled to the host's logical
// CPU count. workers is the number of parallel embed worker processes; threads
// is the ONNX Runtime intra-op thread count per worker. Their product
// approximates the CPU footprint, so lower presets leave cores free for system
// responsiveness.
//
// On DirectML the GPU is the shared compute resource, so workers is capped low
// (the CPU only does image preprocessing); threads still applies to that.
func ResolveEmbedResources() (workers, threads int) {
	cfg := appconfig.Get()
	return resolveResources(cfg.EmbeddingPerformance, cfg.EmbeddingWorkers, cfg.EmbeddingThreadsPerWorker, EmbedProviderFromConfig())
}

// ResolveAutotagResources mirrors ResolveEmbedResources for the auto-tagger,
// reading the Autotag* config fields.
func ResolveAutotagResources() (workers, threads int) {
	cfg := appconfig.Get()
	return resolveResources(cfg.AutotagPerformance, cfg.AutotagWorkers, cfg.AutotagThreadsPerWorker, AutotagProviderFromConfig())
}

// resolveResources maps a performance preset (or custom overrides) + provider to
// a concrete (workers, threads) pair scaled to the host CPU count. Shared by the
// embed and autotag pools.
func resolveResources(performance string, customWorkers, customThreads int, provider string) (workers, threads int) {
	cpus := runtime.NumCPU()
	if cpus < 1 {
		cpus = 1
	}

	perf := strings.ToLower(strings.TrimSpace(performance))
	if perf == "custom" {
		workers, threads = presetResources("balanced", cpus) // defaults for unset fields
		if customWorkers > 0 {
			workers = customWorkers
		}
		if customThreads > 0 {
			threads = customThreads
		}
	} else {
		workers, threads = presetResources(perf, cpus)
	}

	// Clamp to sane bounds. Workers may exceed core count only if the user
	// explicitly set it (custom); presets never do.
	if workers < 1 {
		workers = 1
	}
	if threads < 1 {
		threads = 1
	}

	if provider == "directml" {
		// GPU does the heavy compute; many CPU workers just contend. Keep a
		// little overlap for preprocessing + submission.
		if workers > 2 {
			workers = 2
		}
	}
	return workers, threads
}

// presetResources computes (workers, threads) for a named preset given the
// logical CPU count. target = the number of cores the preset is allowed to use;
// it's split into a few threads per worker so multiple images run in parallel.
func presetResources(preset string, cpus int) (workers, threads int) {
	var target int
	switch preset {
	case "low":
		target = int(math.Round(float64(cpus) * 0.25))
	case "max":
		target = cpus - 1
	default: // "balanced"
		target = int(math.Round(float64(cpus) * 0.5))
	}
	if target < 1 {
		target = 1
	}

	t := target
	if t > 4 {
		t = 4 // diminishing returns past ~4 intra-op threads per session
	}
	w := int(math.Round(float64(target) / float64(t)))
	if w < 1 {
		w = 1
	}
	return w, t
}
