package bundled

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestResolve_ReturnsPathRelativeToExecDir(t *testing.T) {
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	name := "ffmpeg"
	if runtime.GOOS == "windows" {
		name = "ffmpeg.exe"
	}
	if err := os.WriteFile(filepath.Join(binDir, name), []byte("fake"), 0o755); err != nil {
		t.Fatal(err)
	}

	old := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = old }()

	got, err := Resolve("ffmpeg")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := filepath.Join(binDir, name)
	if got != want {
		t.Errorf("Resolve = %q, want %q", got, want)
	}
}

func TestResolve_ReturnsErrMissingWhenFileAbsent(t *testing.T) {
	dir := t.TempDir()
	old := execDirOverride
	execDirOverride = dir
	defer func() { execDirOverride = old }()

	_, err := Resolve("ffmpeg")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
	if !IsMissing(err) {
		t.Errorf("expected IsMissing(err)=true, got: %v", err)
	}
}

func TestResolve_UnknownIDReturnsErrUnknown(t *testing.T) {
	_, err := Resolve("nope-not-a-dep")
	if err == nil {
		t.Fatal("expected error")
	}
	if IsMissing(err) {
		t.Error("unknown id should not be IsMissing")
	}
}
