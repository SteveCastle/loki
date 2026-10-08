package status

import (
	"os"
	"path/filepath"
	"runtime"
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

func TestSnapshot_EngineOnPathCountsAsInstalled(t *testing.T) {
	dir := t.TempDir()
	name := "loki-retouch"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir)
	models.SetCachedStateForTest(map[string]models.ModelStatus{})
	t.Cleanup(func() { models.SetCachedStateForTest(nil) })

	byID := map[string]Item{}
	for _, s := range Snapshot() {
		byID[s.ID] = s
	}
	if got := byID["loki-retouch"]; got.State != string(models.StatusInstalled) || got.Path == "" {
		t.Errorf("loki-retouch on PATH: %+v", got)
	}
	if got := byID["loki-reshoot"]; got.State == string(models.StatusInstalled) {
		t.Errorf("loki-reshoot is not on PATH but reported installed: %+v", got)
	}
}
