package models

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRebuildState_DerivesFromFilesystem(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{
		{ID: "m1", Version: "1.0", Files: []File{{RelPath: "a", URL: "x", SHA256: "y"}}},
		{ID: "m2", Version: "2.0", Files: []File{{RelPath: "a", URL: "x", SHA256: "y"}}},
	}
	defer func() { Manifest = oldMan }()

	// Install m1 fully, leave m2 missing.
	if err := os.MkdirAll(ModelDir("m1"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	meta := []byte(`{"version":"1.0","installed_at":"2026-05-25T00:00:00Z","sha256_verified":true}`)
	if err := os.WriteFile(filepath.Join(ModelDir("m1"), ".meta.json"), meta, 0o644); err != nil {
		t.Fatal(err)
	}

	out := RebuildState()
	if out["m1"] != StatusInstalled {
		t.Errorf("m1 status=%q want installed", out["m1"])
	}
	if out["m2"] != StatusMissing {
		t.Errorf("m2 status=%q want missing", out["m2"])
	}
}

func TestRebuildState_PartialInstallIsMissing(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{{ID: "m1", Version: "1.0", Files: []File{
		{RelPath: "a"}, {RelPath: "b"},
	}}}
	defer func() { Manifest = oldMan }()

	if err := os.MkdirAll(ModelDir("m1"), 0o755); err != nil {
		t.Fatal(err)
	}
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("only-a"), 0o644)
	// Missing "b" and missing .meta.json → not installed.

	out := RebuildState()
	if out["m1"] != StatusMissing {
		t.Errorf("partial install must be missing, got %q", out["m1"])
	}
}

func TestRebuildState_VersionMismatchIsMissing(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	oldMan := Manifest
	Manifest = []Model{{ID: "m1", Version: "2.0", Files: []File{{RelPath: "a"}}}}
	defer func() { Manifest = oldMan }()

	_ = os.MkdirAll(ModelDir("m1"), 0o755)
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), "a"), []byte("data"), 0o644)
	old, _ := json.Marshal(map[string]any{"version": "1.0", "installed_at": "t", "sha256_verified": true})
	_ = os.WriteFile(filepath.Join(ModelDir("m1"), ".meta.json"), old, 0o644)

	out := RebuildState()
	if out["m1"] != StatusMissing {
		t.Errorf("version mismatch must be missing, got %q", out["m1"])
	}
}
