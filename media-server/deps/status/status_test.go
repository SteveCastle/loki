package status

import (
	"testing"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
)

func TestSnapshot_IncludesAllCategories(t *testing.T) {
	bundled.SetCachedStatusForTest([]bundled.Status{{ID: "ffmpeg", Name: "FFmpeg", State: "ready", Version: "7.1"}})
	t.Cleanup(func() { bundled.SetCachedStatusForTest(nil) })

	models.SetCachedStateForTest(map[string]models.ModelStatus{"fake-model": models.StatusInstalled})
	t.Cleanup(func() { models.SetCachedStateForTest(nil) })

	snap := Snapshot()
	var sawBundled, sawOptional, sawModel bool
	for _, s := range snap {
		switch s.Category {
		case "bundled":
			sawBundled = true
		case "optional":
			sawOptional = true
		case "model":
			sawModel = true
		}
	}
	if !sawBundled || !sawOptional || !sawModel {
		t.Errorf("missing categories: bundled=%v optional=%v model=%v", sawBundled, sawOptional, sawModel)
	}
}
