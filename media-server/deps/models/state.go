package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// ModelStatus is the simple per-model installed/missing summary.
type ModelStatus string

const (
	StatusInstalled ModelStatus = "installed"
	StatusMissing   ModelStatus = "missing"
)

var (
	stateCache   map[string]ModelStatus
	stateCacheMu sync.RWMutex
)

// SetCachedStateForTest overrides the cache. Tests only.
func SetCachedStateForTest(in map[string]ModelStatus) {
	stateCacheMu.Lock()
	defer stateCacheMu.Unlock()
	stateCache = in
}

// RebuildState walks the model directory and recomputes installed/missing
// for every manifest entry. Always succeeds; results are cached.
func RebuildState() map[string]ModelStatus {
	out := make(map[string]ModelStatus, len(Manifest))
	for _, m := range Manifest {
		out[m.ID] = statusFor(m)
	}
	persist(out)
	stateCacheMu.Lock()
	stateCache = out
	stateCacheMu.Unlock()
	return out
}

// Cached returns the most recent RebuildState result, rebuilding lazily if empty.
func Cached() map[string]ModelStatus {
	stateCacheMu.RLock()
	if stateCache != nil {
		out := make(map[string]ModelStatus, len(stateCache))
		for k, v := range stateCache {
			out[k] = v
		}
		stateCacheMu.RUnlock()
		return out
	}
	stateCacheMu.RUnlock()
	return RebuildState()
}

func statusFor(m Model) ModelStatus {
	dir := ModelDir(m.ID)
	if _, err := os.Stat(dir); err != nil {
		return StatusMissing
	}
	for _, f := range m.Files {
		if _, err := os.Stat(filepath.Join(dir, f.RelPath)); err != nil {
			return StatusMissing
		}
	}
	metaPath := filepath.Join(dir, ".meta.json")
	b, err := os.ReadFile(metaPath)
	if err != nil {
		return StatusMissing
	}
	var meta struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(b, &meta); err != nil {
		return StatusMissing
	}
	if meta.Version != m.Version {
		return StatusMissing
	}
	return StatusInstalled
}

// persist writes a cache of the derived state to <dataDir>/models/state.json.
// Failures are swallowed: the file is just a cache.
func persist(out map[string]ModelStatus) {
	root := filepath.Join(dataDir(), "models")
	_ = os.MkdirAll(root, 0o755)
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return
	}
	tmp := filepath.Join(root, "state.json.tmp")
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return
	}
	_ = os.Rename(tmp, filepath.Join(root, "state.json"))
}
