package storage

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// setupTempDir creates a temporary directory tree for tests:
//
//	root/
//	  subdir/
//	    nested.jpg
//	  photo.jpg
//	  video.mp4
//	  readme.txt     (non-media, should be filtered)
//	  document.pdf   (non-media, should be filtered)
func setupTempDir(t *testing.T) string {
	t.Helper()
	root := t.TempDir()

	subdir := filepath.Join(root, "subdir")
	if err := os.MkdirAll(subdir, 0755); err != nil {
		t.Fatal(err)
	}

	writeFile := func(path, content string) {
		if err := os.WriteFile(path, []byte(content), 0644); err != nil {
			t.Fatal(err)
		}
	}

	writeFile(filepath.Join(root, "photo.jpg"), "img")
	writeFile(filepath.Join(root, "video.mp4"), "vid")
	writeFile(filepath.Join(root, "readme.txt"), "text")
	writeFile(filepath.Join(root, "document.pdf"), "pdf")
	writeFile(filepath.Join(subdir, "nested.jpg"), "img")

	return root
}

// --- List ---

func TestLocalBackend_List_ReturnsSubdirsAndMediaFiles(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	entries, err := b.List(context.Background(), root)
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}

	nameSet := map[string]bool{}
	for _, e := range entries {
		nameSet[e.Name] = true
	}

	// Must include the subdirectory and the two media files.
	for _, want := range []string{"subdir", "photo.jpg", "video.mp4"} {
		if !nameSet[want] {
			t.Errorf("expected %q in List results, got %v", want, entries)
		}
	}

	// Must exclude non-media files.
	for _, unwanted := range []string{"readme.txt", "document.pdf"} {
		if nameSet[unwanted] {
			t.Errorf("did not expect %q in List results", unwanted)
		}
	}
}

func TestLocalBackend_List_DirsFirst(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	entries, err := b.List(context.Background(), root)
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}

	if len(entries) == 0 {
		t.Fatal("expected non-empty entries")
	}

	// First entry must be a directory.
	if !entries[0].IsDir {
		t.Errorf("expected first entry to be a directory, got %+v", entries[0])
	}

	// All directories must appear before all files.
	seenFile := false
	for _, e := range entries {
		if seenFile && e.IsDir {
			t.Errorf("found directory %q after a file — expected dirs first", e.Name)
		}
		if !e.IsDir {
			seenFile = true
		}
	}
}

func TestLocalBackend_List_TypeField(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	entries, err := b.List(context.Background(), root)
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}

	for _, e := range entries {
		if e.Type != "local" {
			t.Errorf("entry %q has Type=%q, want \"local\"", e.Name, e.Type)
		}
	}
}

// --- Scan ---

func TestLocalBackend_Scan_NonRecursive(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	files, err := b.Scan(context.Background(), root, false)
	if err != nil {
		t.Fatalf("Scan returned error: %v", err)
	}

	nameSet := map[string]bool{}
	for _, f := range files {
		nameSet[filepath.Base(f.Path)] = true
	}

	// Should see top-level media files.
	for _, want := range []string{"photo.jpg", "video.mp4"} {
		if !nameSet[want] {
			t.Errorf("expected %q in Scan results", want)
		}
	}

	// Should NOT descend into subdir.
	if nameSet["nested.jpg"] {
		t.Error("non-recursive Scan should not return nested.jpg")
	}

	// Non-media files must be filtered.
	for _, unwanted := range []string{"readme.txt", "document.pdf"} {
		if nameSet[unwanted] {
			t.Errorf("did not expect %q in Scan results", unwanted)
		}
	}
}

func TestLocalBackend_Scan_Recursive(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	files, err := b.Scan(context.Background(), root, true)
	if err != nil {
		t.Fatalf("Scan returned error: %v", err)
	}

	nameSet := map[string]bool{}
	for _, f := range files {
		nameSet[filepath.Base(f.Path)] = true
	}

	// Should see both top-level and nested media files.
	for _, want := range []string{"photo.jpg", "video.mp4", "nested.jpg"} {
		if !nameSet[want] {
			t.Errorf("expected %q in recursive Scan results", want)
		}
	}

	// Non-media files must be filtered.
	for _, unwanted := range []string{"readme.txt", "document.pdf"} {
		if nameSet[unwanted] {
			t.Errorf("did not expect %q in recursive Scan results", unwanted)
		}
	}
}

// --- Contains ---

func TestLocalBackend_Contains_Inside(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	cases := []string{
		root,
		filepath.Join(root, "photo.jpg"),
		filepath.Join(root, "subdir"),
		filepath.Join(root, "subdir", "nested.jpg"),
	}
	for _, p := range cases {
		if !b.Contains(p) {
			t.Errorf("Contains(%q) = false, want true", p)
		}
	}
}

func TestLocalBackend_Contains_Outside(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	parent := filepath.Dir(root)
	outside := filepath.Join(parent, "other")

	cases := []string{
		parent,
		outside,
		"/tmp/some/other/path",
	}
	for _, p := range cases {
		if b.Contains(p) {
			t.Errorf("Contains(%q) = true, want false", p)
		}
	}
}

// --- MediaURL ---

func TestLocalBackend_MediaURL(t *testing.T) {
	b := NewLocalBackend("/media/root", "test")

	u, err := b.MediaURL("/media/root/photo.jpg")
	if err != nil {
		t.Fatalf("MediaURL returned error: %v", err)
	}

	if !strings.HasPrefix(u, "/media/file?path=") {
		t.Errorf("MediaURL = %q, want prefix \"/media/file?path=\"", u)
	}

	// The path must be URL-encoded in the query string.
	if strings.Contains(u, " ") {
		t.Errorf("MediaURL contains unencoded space: %q", u)
	}
}

func TestLocalBackend_MediaURL_EncodesSpecialChars(t *testing.T) {
	b := NewLocalBackend("/root", "test")

	u, err := b.MediaURL("/root/my file & more.jpg")
	if err != nil {
		t.Fatalf("MediaURL returned error: %v", err)
	}

	if strings.Contains(u, " ") || strings.Contains(u, "&") {
		t.Errorf("MediaURL did not encode special chars: %q", u)
	}
}

// --- Exists ---

func TestLocalBackend_Exists_ExistingFile(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	ok, err := b.Exists(context.Background(), filepath.Join(root, "photo.jpg"))
	if err != nil {
		t.Fatalf("Exists returned error: %v", err)
	}
	if !ok {
		t.Error("Exists = false for a file that exists")
	}
}

func TestLocalBackend_Exists_NonExistingFile(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "test")

	ok, err := b.Exists(context.Background(), filepath.Join(root, "ghost.jpg"))
	if err != nil {
		t.Fatalf("Exists returned unexpected error: %v", err)
	}
	if ok {
		t.Error("Exists = true for a file that does not exist")
	}
}

// --- Root ---

func TestLocalBackend_Root(t *testing.T) {
	root := setupTempDir(t)
	b := NewLocalBackend(root, "MyLabel")

	e := b.Root()
	if e.Path != root {
		t.Errorf("Root().Path = %q, want %q", e.Path, root)
	}
	if !e.IsDir {
		t.Error("Root().IsDir = false, want true")
	}
	if e.Type != "local" {
		t.Errorf("Root().Type = %q, want \"local\"", e.Type)
	}
	if e.Name != "MyLabel" {
		t.Errorf("Root().Name = %q, want \"MyLabel\"", e.Name)
	}
}
