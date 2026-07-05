// Package models implements the on-demand AI model downloader.
package models

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"runtime"
)

//go:embed manifest.json
var manifestJSON []byte

// Model is one downloadable model bundle (or standalone tool — see Category).
type Model struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Category    string   `json:"category,omitempty"` // "" == "model"; "tool" for downloadable binaries
	Feature     string   `json:"feature,omitempty"`  // user-facing capability this unlocks
	Description string   `json:"description"`
	Version     string   `json:"version"`
	Consumers   []string `json:"consumers"`
	SizeBytes   int64    `json:"size_bytes"`
	Files       []File   `json:"files"`
}

// File is one downloadable file within a model. When OS is set the file only
// applies to that GOOS. When Archive is set the download is an archive whose
// ArchiveMember gets extracted to RelPath (and the archive is discarded);
// SHA256 covers the archive as downloaded. An ArchiveMember ending in "/"
// extracts that whole subtree into RelPath as a directory (7z only); Exec
// then marks the extracted root-level binaries executable.
type File struct {
	URL           string `json:"url"`
	RelPath       string `json:"rel_path"`
	SHA256        string `json:"sha256"`
	OS            string `json:"os,omitempty"`
	SizeBytes     int64  `json:"size_bytes,omitempty"`
	Archive       string `json:"archive,omitempty"` // "zip" (single member) or "7z" (directory member)
	ArchiveMember string `json:"archive_member,omitempty"`
	Exec          bool   `json:"exec,omitempty"` // chmod +x after install
}

// EffectiveCategory normalizes the optional Category field.
func (m Model) EffectiveCategory() string {
	if m.Category == "" {
		return "model"
	}
	return m.Category
}

// EffectiveFiles returns the files applicable to the current OS.
func (m Model) EffectiveFiles() []File {
	out := make([]File, 0, len(m.Files))
	for _, f := range m.Files {
		if f.OS == "" || f.OS == runtime.GOOS {
			out = append(out, f)
		}
	}
	return out
}

// EffectiveSizeBytes sums per-file sizes for this OS when present, falling
// back to the model-level total.
func (m Model) EffectiveSizeBytes() int64 {
	var sum int64
	for _, f := range m.EffectiveFiles() {
		sum += f.SizeBytes
	}
	if sum > 0 {
		return sum
	}
	return m.SizeBytes
}

// Manifest is the parsed model registry.
var Manifest []Model

func init() {
	var doc struct {
		SchemaVersion int     `json:"schema_version"`
		Models        []Model `json:"models"`
	}
	if err := json.Unmarshal(manifestJSON, &doc); err != nil {
		panic(fmt.Sprintf("deps/models: manifest.json invalid: %v", err))
	}
	if doc.SchemaVersion != 1 {
		panic(fmt.Sprintf("deps/models: unsupported schema_version %d", doc.SchemaVersion))
	}
	Manifest = doc.Models
}

// Lookup returns the model with the given id.
func Lookup(id string) (Model, bool) {
	for _, m := range Manifest {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// IDs returns every model id in manifest order.
func IDs() []string {
	out := make([]string, 0, len(Manifest))
	for _, m := range Manifest {
		out = append(out, m.ID)
	}
	return out
}
