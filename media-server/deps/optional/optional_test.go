package optional

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func writeFakeBin(t *testing.T, dir, name, versionOutput string) string {
	t.Helper()
	if runtime.GOOS == "windows" {
		path := filepath.Join(dir, name+".bat")
		content := "@echo off\r\necho " + versionOutput + "\r\n"
		if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
			t.Fatal(err)
		}
		return path
	}
	path := filepath.Join(dir, name)
	content := "#!/bin/sh\necho " + versionOutput + "\n"
	if err := os.WriteFile(path, []byte(content), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func withPath(t *testing.T, dir string) {
	t.Helper()
	old := os.Getenv("PATH")
	sep := ":"
	if runtime.GOOS == "windows" {
		sep = ";"
	}
	if err := os.Setenv("PATH", dir+sep+old); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Setenv("PATH", old) })
}

func TestDetect_FindsBinaryOnPath(t *testing.T) {
	dir := t.TempDir()
	writeFakeBin(t, dir, "yt-dlp", "2026.05.01")
	withPath(t, dir)

	s, err := Detect("yt-dlp")
	if err != nil {
		t.Fatal(err)
	}
	if !s.Installed {
		t.Error("expected Installed=true")
	}
	if !strings.Contains(s.Version, "2026.05.01") {
		t.Errorf("Version=%q want 2026.05.01", s.Version)
	}
	if s.Hint.DocsURL == "" {
		t.Error("expected DocsURL populated even when installed")
	}
}

func TestDetect_ReturnsNotInstalledWithHint(t *testing.T) {
	t.Setenv("PATH", t.TempDir())

	s, err := Detect("yt-dlp")
	if err != nil {
		t.Fatal(err)
	}
	if s.Installed {
		t.Error("expected Installed=false")
	}
	if len(s.Hint.Commands) == 0 {
		t.Error("expected install commands")
	}
}

func TestDetect_UnknownIDIsError(t *testing.T) {
	_, err := Detect("not-a-tool")
	if err == nil {
		t.Fatal("expected error for unknown id")
	}
}
