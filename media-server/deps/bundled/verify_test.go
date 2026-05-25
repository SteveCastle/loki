package bundled

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestVerifyAll_ReportsReadyForPresentBinaries(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "windows" {
		t.Skip("verify exec stub not portable to windows in unit test")
	}
	stub := filepath.Join(binDir, "ffmpeg")
	script := "#!/bin/sh\necho fake-version 1.2.3\n"
	if err := os.WriteFile(stub, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	old := Manifest
	Manifest = []Bundled{{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg", VersionArgs: []string{"--noop"}}}
	defer func() { Manifest = old }()

	prevExec := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = prevExec }()

	statuses := VerifyAll()
	if len(statuses) != 1 {
		t.Fatalf("want 1 status, got %d", len(statuses))
	}
	if statuses[0].State != "ready" {
		t.Errorf("state=%q error=%q want ready", statuses[0].State, statuses[0].Error)
	}
}

func TestVerifyAll_ReportsMissing(t *testing.T) {
	dir := t.TempDir()
	old := Manifest
	Manifest = []Bundled{{ID: "ffmpeg", Name: "FFmpeg", RelPath: "ffmpeg", VersionArgs: nil}}
	defer func() { Manifest = old }()
	prev := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = prev }()

	statuses := VerifyAll()
	if statuses[0].State != "missing" {
		t.Errorf("state=%q, want missing", statuses[0].State)
	}
}
