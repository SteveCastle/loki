package models

import (
	"net/url"
	"strings"
	"testing"
)

func TestManifest_ParsesAndHasModels(t *testing.T) {
	if len(Manifest) == 0 {
		t.Fatal("expected at least one model in manifest")
	}
	for _, m := range Manifest {
		if m.ID == "" {
			t.Error("empty model id")
		}
		if m.Version == "" {
			t.Errorf("model %s: empty version", m.ID)
		}
		if len(m.Files) == 0 {
			t.Errorf("model %s: no files", m.ID)
		}
		for _, f := range m.Files {
			if _, err := url.Parse(f.URL); err != nil || !strings.HasPrefix(f.URL, "http") {
				t.Errorf("model %s file %s: bad url %q", m.ID, f.RelPath, f.URL)
			}
			if f.RelPath == "" {
				t.Errorf("model %s: empty rel_path", m.ID)
			}
			if f.SHA256 == "" {
				t.Errorf("model %s file %s: empty sha256", m.ID, f.RelPath)
			}
		}
	}
}

func TestLookup(t *testing.T) {
	if _, ok := Lookup("nope"); ok {
		t.Error("expected !ok for unknown id")
	}
	m, ok := Lookup(Manifest[0].ID)
	if !ok || m.ID != Manifest[0].ID {
		t.Errorf("Lookup roundtrip failed")
	}
}
