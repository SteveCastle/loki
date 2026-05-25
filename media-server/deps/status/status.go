// Package status aggregates dep state across bundled / optional / model
// for the UI. It NEVER triggers installs; it only reads snapshots.
package status

import (
	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/optional"
)

type Item struct {
	ID        string `json:"id"`
	Category  string `json:"category"`
	Name      string `json:"name"`
	State     string `json:"state"`
	Version   string `json:"version,omitempty"`
	SizeBytes int64  `json:"size_bytes,omitempty"`
	Path      string `json:"path,omitempty"`
	Error     string `json:"error,omitempty"`
	Detail    any    `json:"detail,omitempty"`
}

func Snapshot() []Item {
	out := make([]Item, 0, 16)

	for _, b := range bundled.CachedStatus() {
		out = append(out, Item{
			ID: b.ID, Category: "bundled", Name: b.Name,
			State: b.State, Version: b.Version, Path: b.Path, Error: b.Error,
		})
	}
	for _, o := range optional.Manifest {
		s, _ := optional.Detect(o.ID)
		state := "not_installed"
		if s.Installed {
			state = "installed"
		}
		out = append(out, Item{
			ID: s.ID, Category: "optional", Name: s.Name,
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
		if inst, ok := models.Tracker.Snapshot(m.ID); ok {
			state = string(inst.State)
			out = append(out, Item{
				ID: m.ID, Category: "model", Name: m.Name,
				State: state, SizeBytes: m.SizeBytes, Path: path,
				Detail: inst, Error: inst.Error,
			})
			continue
		}
		out = append(out, Item{
			ID: m.ID, Category: "model", Name: m.Name,
			State: state, SizeBytes: m.SizeBytes, Path: path,
		})
	}
	return out
}
