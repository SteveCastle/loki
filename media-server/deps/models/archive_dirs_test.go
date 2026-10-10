package models

import (
	"archive/tar"
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/ulikunitz/xz"
)

func TestExtractZipDir(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "colmap.zip")
	writeTestZip(t, archive, map[string]string{
		"bin/colmap.exe":       "exe",
		"bin/cudart64_12.dll":  "dll",
		"bin/sub/plugin.dll":   "plug",
		"plugins/platforms/qw": "gui only",
		"COLMAP.bat":           "bat",
	})
	dst := filepath.Join(dir, "colmap")
	if err := extractZipDir(context.Background(), archive, "BIN/", dst, true, nil); err != nil {
		t.Fatalf("extractZipDir: %v", err)
	}
	for rel, want := range map[string]string{"colmap.exe": "exe", "cudart64_12.dll": "dll", "sub/plugin.dll": "plug"} {
		b, err := os.ReadFile(filepath.Join(dst, filepath.FromSlash(rel)))
		if err != nil || string(b) != want {
			t.Errorf("%s = %q, %v; want %q", rel, b, err, want)
		}
	}
	for _, stray := range []string{"COLMAP.bat", "plugins", "platforms"} {
		if _, err := os.Stat(filepath.Join(dst, stray)); !os.IsNotExist(err) {
			t.Errorf("%s outside the member prefix was extracted", stray)
		}
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial dir left behind")
	}
}

func writeTestTarXz(t *testing.T, path string, members map[string]string) {
	t.Helper()
	var tarBuf bytes.Buffer
	tw := tar.NewWriter(&tarBuf)
	for name, content := range members {
		if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0o755, Size: int64(len(content)), Typeflag: tar.TypeReg}); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	xw, err := xz.NewWriter(f)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := xw.Write(tarBuf.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err := xw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExtractTarXzMember(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "brush.tar.xz")
	writeTestTarXz(t, archive, map[string]string{
		"brush-app-x86_64-unknown-linux-gnu/README.md": "readme",
		"brush-app-x86_64-unknown-linux-gnu/brush_app": "ELF",
	})
	dst := filepath.Join(dir, "brush_app")
	if err := extractTarXzMember(archive, "brush-app-x86_64-unknown-linux-gnu/brush_app", dst); err != nil {
		t.Fatalf("extractTarXzMember: %v", err)
	}
	if b, err := os.ReadFile(dst); err != nil || string(b) != "ELF" {
		t.Errorf("extracted = %q, %v", b, err)
	}
	if err := extractTarXzMember(archive, "nope", filepath.Join(dir, "x")); err == nil {
		t.Errorf("missing member should fail")
	}
}
