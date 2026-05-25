// Package models implements the on-demand AI model downloader.
package models

import (
	_ "embed"
	"encoding/json"
	"fmt"
)

//go:embed manifest.json
var manifestJSON []byte

// Model is one downloadable model bundle.
type Model struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Version     string   `json:"version"`
	Consumers   []string `json:"consumers"`
	SizeBytes   int64    `json:"size_bytes"`
	Files       []File   `json:"files"`
}

// File is one downloadable file within a model.
type File struct {
	URL     string `json:"url"`
	RelPath string `json:"rel_path"`
	SHA256  string `json:"sha256"`
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
