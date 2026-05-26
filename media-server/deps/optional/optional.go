// Package optional detects user-installed CLI tools on PATH and provides
// per-OS install hints. It NEVER installs anything.
package optional

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/stevecastle/shrike/platform"
)

var ErrUnknown = errors.New("optional: unknown tool id")

type Optional struct {
	ID          string
	Name        string
	Binary      string
	VersionArgs []string
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
