package models

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelDir_ComposesUnderDataDir(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	got := ModelDir("wd-eva02-large-tagger-v3")
	want := filepath.Join(dir, "models", "wd-eva02-large-tagger-v3")
	if got != want {
		t.Errorf("ModelDir = %q want %q", got, want)
	}
}

func TestPath_ReturnsErrNotInstalledWhenAbsent(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	_, err := Path("wd-eva02-large-tagger-v3", "model.onnx")
	if err == nil {
		t.Fatal("expected error")
	}
	if !IsNotInstalled(err) {
		t.Errorf("expected IsNotInstalled(err)=true, got %v", err)
	}
}

func TestPath_ReturnsAbsolutePathWhenPresent(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	mdir := ModelDir("wd-eva02-large-tagger-v3")
	if err := os.MkdirAll(mdir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(mdir, "model.onnx"), []byte("fake"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := Path("wd-eva02-large-tagger-v3", "model.onnx")
	if err != nil {
		t.Fatal(err)
	}
	want := filepath.Join(mdir, "model.onnx")
	if got != want {
		t.Errorf("Path = %q want %q", got, want)
	}
}

func TestAtomicWrite_RenamesAfterClose(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "out.bin")

	w, err := NewAtomicWriter(final)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("hello")); err != nil {
		t.Fatal(err)
	}
	// Before Commit, the final file should not exist.
	if _, err := os.Stat(final); !os.IsNotExist(err) {
		t.Errorf("final exists before commit: err=%v", err)
	}
	if err := w.Commit(); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(final)
	if err != nil || string(b) != "hello" {
		t.Fatalf("post-commit read: %q err=%v", b, err)
	}
}

func TestAtomicWrite_AbortCleansPartial(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "out.bin")
	w, err := NewAtomicWriter(final)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.Write([]byte("x")); err != nil {
		t.Fatal(err)
	}
	if err := w.Abort(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(final + ".partial"); !os.IsNotExist(err) {
		t.Errorf(".partial still exists after Abort")
	}
}
