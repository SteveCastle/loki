package models

import (
	"archive/zip"
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
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

// testdata/tool7z-fixture.7z contains:
//
//	Tool-Dir/tool-binary        "#!/bin/sh\necho hi\n"
//	Tool-Dir/README.txt         "readme"
//	Tool-Dir/_data/lib.bin      "lib"
//	stray.txt                   "stray" (outside the extracted subtree)
const sevenZipFixture = "testdata/tool7z-fixture.7z"

func TestExtractSevenZipDir(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "xxl")
	if err := extractSevenZipDir(context.Background(), sevenZipFixture, "Tool-Dir/", dst, true, nil); err != nil {
		t.Fatalf("extractSevenZipDir: %v", err)
	}

	for rel, want := range map[string]string{
		"tool-binary":   "#!/bin/sh\necho hi\n",
		"README.txt":    "readme",
		"_data/lib.bin": "lib",
	} {
		b, err := os.ReadFile(filepath.Join(dst, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatalf("read %s: %v", rel, err)
		}
		if string(b) != want {
			t.Errorf("%s content = %q; want %q", rel, b, want)
		}
	}
	if _, err := os.Stat(filepath.Join(dst, "stray.txt")); !os.IsNotExist(err) {
		t.Errorf("stray.txt outside the member prefix was extracted")
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial dir left behind")
	}
	if runtime.GOOS != "windows" {
		info, err := os.Stat(filepath.Join(dst, "tool-binary"))
		if err != nil {
			t.Fatal(err)
		}
		if info.Mode()&0o111 == 0 {
			t.Errorf("root-level binary not marked executable: mode %v", info.Mode())
		}
	}
}

func TestExtractSevenZipDirCaseInsensitivePrefix(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "xxl")
	if err := extractSevenZipDir(context.Background(), sevenZipFixture, "tool-dir/", dst, false, nil); err != nil {
		t.Fatalf("extractSevenZipDir: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dst, "README.txt")); err != nil {
		t.Errorf("README.txt missing after case-insensitive extract: %v", err)
	}
}

func TestExtractSevenZipDirReplacesPreviousInstall(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "xxl")
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	leftover := filepath.Join(dst, "old-version-file")
	if err := os.WriteFile(leftover, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := extractSevenZipDir(context.Background(), sevenZipFixture, "Tool-Dir/", dst, false, nil); err != nil {
		t.Fatalf("extractSevenZipDir: %v", err)
	}
	if _, err := os.Stat(leftover); !os.IsNotExist(err) {
		t.Errorf("previous install contents survived the reinstall")
	}
	if _, err := os.Stat(filepath.Join(dst, "tool-binary")); err != nil {
		t.Errorf("new install incomplete: %v", err)
	}
}

func TestExtractSevenZipDirMissingPrefix(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "xxl")
	if err := extractSevenZipDir(context.Background(), sevenZipFixture, "No-Such-Dir/", dst, false, nil); err == nil {
		t.Fatal("expected error for missing member prefix")
	}
	if _, err := os.Stat(dst); !os.IsNotExist(err) {
		t.Errorf("dst dir created despite missing prefix")
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial dir left behind after failure")
	}
}

func TestExtractSevenZipDirReportsProgress(t *testing.T) {
	dir := t.TempDir()
	dst := filepath.Join(dir, "xxl")
	var last int64
	var name string
	progress := func(file string, done, total int64) {
		name = file
		if done > last {
			last = done
		}
		if total != 27 { // 18 + 6 + 3 bytes under Tool-Dir/
			t.Errorf("total = %d; want 27", total)
		}
	}
	if err := extractSevenZipDir(context.Background(), sevenZipFixture, "Tool-Dir/", dst, false, progress); err != nil {
		t.Fatalf("extractSevenZipDir: %v", err)
	}
	if last != 27 {
		t.Errorf("final progress = %d; want 27", last)
	}
	if name != "xxl (extracting)" {
		t.Errorf("progress name = %q", name)
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
	switch runtime.GOOS {
	case "windows", "linux":
		// XXL is a multi-file 7z bundle extracted as a directory.
		if eff[0].Archive != "7z" || !strings.HasSuffix(eff[0].ArchiveMember, "/") || !eff[0].Exec {
			t.Errorf("whisper file entry incomplete: %+v", eff[0])
		}
	default:
		// macOS keeps the legacy single-binary zip build (no XXL release).
		if eff[0].Archive != "zip" || eff[0].ArchiveMember == "" || !eff[0].Exec {
			t.Errorf("whisper file entry incomplete: %+v", eff[0])
		}
	}
}
