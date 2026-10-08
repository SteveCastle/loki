// Package status aggregates dep state across bundled / optional / model
// for the UI. It NEVER triggers installs; it only reads snapshots.
package status

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/optional"
)

type Item struct {
	ID          string `json:"id"`
	Category    string `json:"category"`
	Name        string `json:"name"`
	Feature     string `json:"feature,omitempty"`
	Description string `json:"description,omitempty"`
	State       string `json:"state"`
	Version     string `json:"version,omitempty"`
	SizeBytes   int64  `json:"size_bytes,omitempty"`
	Path        string `json:"path,omitempty"`
	Error       string `json:"error,omitempty"`
	Detail      any    `json:"detail,omitempty"`
}

func Snapshot() []Item {
	out := make([]Item, 0, 16)

	for _, b := range bundled.CachedStatus() {
		out = append(out, Item{
			ID: b.ID, Category: "bundled", Name: b.Name,
			State: b.State, Version: b.Version, Path: b.Path, Error: b.Error,
		})
	}
	// CachedDetect, not Detect: live detection spawns version subprocesses
	// (seconds each) and this endpoint is polled from many UI surfaces.
	for _, o := range optional.Manifest {
		s, _ := optional.CachedDetect(o.ID)
		state := "not_installed"
		if s.Installed {
			state = "installed"
		}
		out = append(out, Item{
			ID: s.ID, Category: "optional", Name: s.Name,
			Feature: o.Feature, Description: o.Description,
			State: state, Version: s.Version, Path: s.Path, Detail: s.Hint,
		})
	}
	cached := models.Cached()
	for _, m := range models.Manifest {
		state := string(cached[m.ID])
		if state == "" {
			state = string(models.StatusMissing)
		}
		path := ""
		if cached[m.ID] == models.StatusInstalled {
			path = models.ModelDir(m.ID)
		}
		item := Item{
			ID: m.ID, Category: m.EffectiveCategory(), Name: m.Name,
			Feature: m.Feature, Description: m.Description,
			State: state, SizeBytes: m.EffectiveSizeBytes(), Path: path,
		}
		if inst, ok := models.Tracker.Snapshot(m.ID); ok {
			item.State = string(inst.State)
			item.Detail = inst
			item.Error = inst.Error
			out = append(out, item)
			continue
		}
		// A user-configured faster-whisper binary satisfies the transcription
		// tool without the assisted download.
		if m.ID == "faster-whisper" && item.State == string(models.StatusMissing) {
			if p := strings.TrimSpace(appconfig.Get().FasterWhisperPath); p != "" {
				if _, err := os.Stat(p); err == nil {
					item.State = string(models.StatusInstalled)
					item.Path = p
					item.Detail = map[string]string{"source": "configured_path"}
				}
			}
		}
		// Model weights that a PATH-installed engine already has where it looks
		// for them (beside the executable, its models/ folder, $LOKI_MODELS).
		if m.EffectiveCategory() == "model" && item.State == string(models.StatusMissing) {
			if dir := engineModelsDir(m); dir != "" {
				item.State = string(models.StatusInstalled)
				item.Path = dir
				item.Detail = map[string]string{"source": "path"}
			}
		}
		// A loki-* engine already on PATH (a developer build, or installed by
		// hand) is used as-is by the tasks and never downloaded, so the UI must
		// not ask to download it or its models.
		if m.EffectiveCategory() == "tool" && strings.HasPrefix(m.ID, "loki-") && item.State == string(models.StatusMissing) {
			if p, err := exec.LookPath(m.ID); err == nil {
				item.State = string(models.StatusInstalled)
				item.Path = p
				item.Detail = map[string]string{"source": "path"}
			}
		}
		out = append(out, item)
	}
	return out
}

// engineModelsDir returns the folder holding every file of m when m belongs to
// a loki-* engine found on PATH, mirroring the engines' own model search
// order, or "" when the engine isn't on PATH or some file is absent.
func engineModelsDir(m models.Model) string {
	if len(m.Consumers) == 0 {
		return ""
	}
	exe, err := exec.LookPath("loki-" + m.Consumers[0])
	if err != nil {
		return ""
	}
	base := filepath.Dir(exe)
	dirs := []string{base, filepath.Join(base, "models")}
	if d := os.Getenv("LOKI_MODELS"); d != "" {
		dirs = append(dirs, d)
	}
	files := m.EffectiveFiles()
	for _, d := range dirs {
		all := len(files) > 0
		for _, f := range files {
			if _, err := os.Stat(filepath.Join(d, f.RelPath)); err != nil {
				all = false
				break
			}
		}
		if all {
			return d
		}
	}
	return ""
}
