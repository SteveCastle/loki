// Package optional detects user-installed CLI tools on PATH and provides
// per-OS install hints. It NEVER installs anything.
package optional

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/platform"
)

var ErrUnknown = errors.New("optional: unknown tool id")

type Optional struct {
	ID          string
	Name        string
	Binary      string
	VersionArgs []string
	Feature     string // user-facing capability this unlocks
	Description string
	DocsURL     string
}

type Status struct {
	ID        string      `json:"id"`
	Name      string      `json:"name"`
	Installed bool        `json:"installed"`
	Path      string      `json:"path,omitempty"`
	Version   string      `json:"version,omitempty"`
	Hint      InstallHint `json:"hint"`
}

type InstallHint struct {
	Description string  `json:"description,omitempty"`
	Commands    []OSCmd `json:"commands,omitempty"`
	DocsURL     string  `json:"docs_url,omitempty"`
}

type OSCmd struct {
	OS      string `json:"os"`
	Label   string `json:"label"`
	Command string `json:"command"`
}

func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, o := range Manifest {
		out = append(out, o.ID)
	}
	return out
}

func Detect(id string) (Status, error) {
	entry, ok := lookup(id)
	if !ok {
		return Status{}, fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	s := Status{ID: entry.ID, Name: entry.Name, Hint: hintFor(entry)}
	path, err := exec.LookPath(entry.Binary)
	if err != nil {
		return s, nil
	}
	s.Installed = true
	s.Path = path
	if len(entry.VersionArgs) > 0 {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, path, entry.VersionArgs...)
		platform.HideSubprocessWindow(cmd)
		out, perr := cmd.CombinedOutput()
		if perr == nil {
			s.Version = firstLine(string(out))
		}
	}
	return s, nil
}

// Live detection is expensive: LookPath walks PATH (which may include slow
// network drives) and VersionArgs spawns a subprocess — yt-dlp and gallery-dl
// are Python entry points that take 1–3s just to start. /api/deps/status runs
// Detect for every manifest entry, and UI surfaces request it per-mount, so
// serving live detection per request let a burst of status calls take
// *minutes* to drain and starved the client's per-origin socket pool.
// CachedDetect serves a snapshot instead: stale-while-revalidate with a
// single-flight background refresh, so no request ever waits on a subprocess.
var (
	cacheMu    sync.Mutex
	cache      map[string]Status
	cachedAt   time.Time
	refreshing bool
)

const cacheTTL = 30 * time.Second

func detectAll() map[string]Status {
	out := make(map[string]Status, len(Manifest))
	for _, o := range Manifest {
		s, err := Detect(o.ID)
		if err != nil {
			continue
		}
		out[o.ID] = s
	}
	return out
}

func refreshCacheLocked() {
	if refreshing {
		return
	}
	refreshing = true
	go func() {
		all := detectAll()
		cacheMu.Lock()
		cache = all
		cachedAt = time.Now()
		refreshing = false
		cacheMu.Unlock()
	}()
}

// CachedDetect returns the cached status for one tool, kicking off a
// background refresh when the cache is stale or absent. It never blocks on
// detection: before the first refresh completes it reports the tool as not
// yet installed (with install hints populated), which the UI treats as a
// soft state. Call Warm at startup to make that window negligible.
func CachedDetect(id string) (Status, error) {
	entry, ok := lookup(id)
	if !ok {
		return Status{}, fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	cacheMu.Lock()
	defer cacheMu.Unlock()
	if cache == nil || time.Since(cachedAt) >= cacheTTL {
		refreshCacheLocked()
	}
	if s, ok := cache[id]; ok {
		return s, nil
	}
	return Status{ID: entry.ID, Name: entry.Name, Hint: hintFor(entry)}, nil
}

// Warm populates the detection cache in the background so the first
// /api/deps/status request after boot serves real states.
func Warm() {
	cacheMu.Lock()
	defer cacheMu.Unlock()
	refreshCacheLocked()
}

func lookup(id string) (Optional, bool) {
	for _, o := range Manifest {
		if o.ID == id {
			return o, true
		}
	}
	return Optional{}, false
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return strings.TrimSpace(s)
}
