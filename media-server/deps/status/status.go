// Package status aggregates dep state across bundled / optional / model
// for the UI. It NEVER triggers installs; it only reads snapshots.
package status

import (
	"os"
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
		out = append(out, item)
	}
	return out
}
