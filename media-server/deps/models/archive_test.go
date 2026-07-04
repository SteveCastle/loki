package models

import (
	"archive/zip"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func writeTestZip(t *testing.T, path string, members map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	zw := zip.NewWriter(f)
	for name, content := range members {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestExtractZipMember(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "tool.zip")
	writeTestZip(t, archive, map[string]string{
		"Whisper-Faster/whisper-faster": "#!/bin/sh\necho hi\n",
		"license.txt":                   "MIT",
	})

	dst := filepath.Join(dir, "whisper-faster")
	if err := extractZipMember(archive, "Whisper-Faster/whisper-faster", dst); err != nil {
		t.Fatalf("extractZipMember: %v", err)
	}
	b, err := os.ReadFile(dst)
	if err != nil {
		t.Fatalf("read extracted: %v", err)
	}
	if string(b) != "#!/bin/sh\necho hi\n" {
		t.Errorf("extracted content = %q", b)
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial file left behind")
	}
}

func TestExtractZipMemberCaseInsensitive(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "tool.zip")
	writeTestZip(t, archive, map[string]string{"Dir/Tool.EXE": "bin"})

	dst := filepath.Join(dir, "tool.exe")
	if err := extractZipMember(archive, "dir/tool.exe", dst); err != nil {
		t.Fatalf("extractZipMember: %v", err)
	}
}

func TestExtractZipMemberMissing(t *testing.T) {
	dir := t.TempDir()
	archive := filepath.Join(dir, "tool.zip")
	writeTestZip(t, archive, map[string]string{"a.txt": "x"})

	if err := extractZipMember(archive, "nope", filepath.Join(dir, "out")); err == nil {
		t.Fatal("expected error for missing member")
	}
}

func TestEffectiveFilesFiltersByOS(t *testing.T) {
	m := Model{Files: []File{
		{RelPath: "any", SizeBytes: 1},
		{RelPath: "here", OS: runtime.GOOS, SizeBytes: 2},
		{RelPath: "other", OS: "plan9", SizeBytes: 4},
	}}
	eff := m.EffectiveFiles()
	if len(eff) != 2 {
		t.Fatalf("EffectiveFiles = %d entries; want 2", len(eff))
	}
	if got := m.EffectiveSizeBytes(); got != 3 {
		t.Errorf("EffectiveSizeBytes = %d; want 3", got)
	}
}

func TestEffectiveSizeBytesFallsBack(t *testing.T) {
	m := Model{SizeBytes: 42, Files: []File{{RelPath: "a"}}}
	if got := m.EffectiveSizeBytes(); got != 42 {
		t.Errorf("EffectiveSizeBytes = %d; want model-level 42", got)
	}
}

func TestManifestHasWhisperToolForThisOS(t *testing.T) {
	m, ok := Lookup("faster-whisper")
	if !ok {
		t.Fatal("faster-whisper missing from manifest")
	}
	if m.EffectiveCategory() != "tool" {
		t.Errorf("category = %q; want tool", m.EffectiveCategory())
	}
	eff := m.EffectiveFiles()
	if len(eff) != 1 {
		t.Fatalf("EffectiveFiles for %s = %d; want exactly 1", runtime.GOOS, len(eff))
	}
	if eff[0].Archive != "zip" || eff[0].ArchiveMember == "" || !eff[0].Exec {
		t.Errorf("whisper file entry incomplete: %+v", eff[0])
	}
}
