// Package bundled resolves paths to native binaries shipped alongside the
// server executable. It never downloads; it only locates and validates.
package bundled

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Bundled is one binary or shared library bundled with the server release.
type Bundled struct {
	ID          string
	Name        string
	RelPath     string
	VersionArgs []string
}

// Status describes the runtime state of one bundled entry.
type Status struct {
	ID      string
	Name    string
	Path    string
	State   string // "ready" | "missing" | "broken"
	Version string
	Error   string
}

var (
	ErrUnknown = errors.New("bundled: unknown dependency id")
	errMissing = errors.New("bundled: file not found")

	execDirOverride string
	execDirOnce     sync.Once
	execDirCached   string
)

func IsMissing(err error) bool { return errors.Is(err, errMissing) }

func Resolve(id string) (string, error) {
	entry, ok := lookup(id)
	if !ok {
		return "", fmt.Errorf("%w: %q", ErrUnknown, id)
	}
	path := filepath.Join(execDir(), "bin", entry.RelPath)
	info, err := os.Stat(path)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("%w: %s (id=%s)", errMissing, path, id)
	}
	return path, nil
}

func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, b := range Manifest {
		out = append(out, b.ID)
	}
	return out
}

func lookup(id string) (Bundled, bool) {
	for _, b := range Manifest {
		if b.ID == id {
			return b, true
		}
	}
	return Bundled{}, false
}

func execDir() string {
	if execDirOverride != "" {
		return execDirOverride
	}
	execDirOnce.Do(func() {
		exe, err := os.Executable()
		if err != nil {
			execDirCached = "."
			return
		}
		execDirCached = filepath.Dir(exe)
	})
	return execDirCached
}
