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
	if got := byID["loki-retouch"]; got.State != string(models.StatusInstalled) || got.Path == "" || got.Source != SourceUser {
		t.Errorf("loki-retouch on PATH: %+v", got)
	}
	if got := byID["loki-reshoot"]; got.State == string(models.StatusInstalled) {
		t.Errorf("loki-reshoot is not on PATH but reported installed: %+v", got)
	}
}

func TestSnapshot_EngineModelsBesideThePathBinary(t *testing.T) {
	dir := t.TempDir()
	exe := "loki-reshoot"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	os.WriteFile(filepath.Join(dir, exe), []byte("x"), 0o755)
	t.Setenv("PATH", dir)
	t.Setenv("LOKI_MODELS", "")
	models.SetCachedStateForTest(map[string]models.ModelStatus{})
	t.Cleanup(func() { models.SetCachedStateForTest(nil) })

	state := func(id string) Item {
		for _, s := range Snapshot() {
			if s.ID == id {
				return s
			}
		}
		t.Fatalf("no %s", id)
		return Item{}
	}
	if got := state("minimax-h3-ref2va"); got.State == string(models.StatusInstalled) {
		t.Fatalf("no weights yet, but reported installed: %+v", got)
	}
	m, _ := models.Lookup("minimax-h3-ref2va")
	if err := os.MkdirAll(filepath.Join(dir, "models"), 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range m.EffectiveFiles() {
		os.WriteFile(filepath.Join(dir, "models", f.RelPath), []byte("x"), 0o644)
	}
	got := state("minimax-h3-ref2va")
	if got.State != string(models.StatusInstalled) || got.Path != filepath.Join(dir, "models") {
		t.Errorf("weights in models/ beside the binary: %+v", got)
	}
	if state("qwen-image-2.1").State == string(models.StatusInstalled) {
		t.Errorf("retouch weights are not there")
	}
}

// LOKI_MODELS is a path list: weights in any of its folders count.
func TestSnapshot_EngineModelsInLOKIMODELSPathList(t *testing.T) {
	dir := t.TempDir()
	exe := "loki-reshoot"
	if runtime.GOOS == "windows" {
		exe += ".exe"
	}
	os.WriteFile(filepath.Join(dir, exe), []byte("x"), 0o755)
	h3, fast := t.TempDir(), t.TempDir()
	t.Setenv("PATH", dir)
	t.Setenv("LOKI_MODELS", h3+string(os.PathListSeparator)+fast)
	models.SetCachedStateForTest(map[string]models.ModelStatus{})
	t.Cleanup(func() { models.SetCachedStateForTest(nil) })
	for id, d := range map[string]string{"minimax-h3-ref2va": h3, "fasth3-omni": fast} {
		m, _ := models.Lookup(id)
		for _, f := range m.EffectiveFiles() {
			os.WriteFile(filepath.Join(d, f.RelPath), []byte("x"), 0o644)
		}
	}
	got := map[string]Item{}
	for _, s := range Snapshot() {
		got[s.ID] = s
	}
	if s := got["minimax-h3-ref2va"]; s.State != string(models.StatusInstalled) || s.Path != h3 {
		t.Errorf("h3 weights in the first LOKI_MODELS folder: %+v", s)
	}
	if s := got["fasth3-omni"]; s.State != string(models.StatusInstalled) || s.Path != fast {
		t.Errorf("fast weights in the second LOKI_MODELS folder: %+v", s)
	}
}
