package storage

import (
	"context"
	"io"
	"path/filepath"
	"strings"
	"testing"
)

// --- BackendFor ---

func TestRegistry_BackendFor_ReturnsCorrectBackend(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "backend1")
	b2 := NewLocalBackend(root2, "backend2")

	reg := NewRegistry([]Backend{b1, b2})

	path1 := filepath.Join(root1, "file.jpg")
	path2 := filepath.Join(root2, "file.mp4")

	got1 := reg.BackendFor(path1)
	if got1 != b1 {
		t.Errorf("BackendFor(%q) returned wrong backend", path1)
	}

	got2 := reg.BackendFor(path2)
	if got2 != b2 {
		t.Errorf("BackendFor(%q) returned wrong backend", path2)
	}
}

func TestRegistry_BackendFor_NilForUnknownPath(t *testing.T) {
	root := t.TempDir()
	b := NewLocalBackend(root, "test")
	reg := NewRegistry([]Backend{b})

	outside := filepath.Join(filepath.Dir(root), "unrelated", "file.jpg")
	got := reg.BackendFor(outside)
	if got != nil {
		t.Errorf("BackendFor unknown path = %v, want nil", got)
	}
}

func TestRegistry_BackendFor_FirstMatchWins(t *testing.T) {
	// Both backends claim the same root — the first one should win.
	root := t.TempDir()
	b1 := NewLocalBackend(root, "first")
	b2 := NewLocalBackend(root, "second")

	reg := NewRegistry([]Backend{b1, b2})

	got := reg.BackendFor(filepath.Join(root, "file.jpg"))
	if got != b1 {
		t.Error("expected first matching backend to win")
	}
}

// --- AllRoots ---

func TestRegistry_AllRoots_ReturnsAllRoots(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "label1")
	b2 := NewLocalBackend(root2, "label2")

	reg := NewRegistry([]Backend{b1, b2})

	roots := reg.AllRoots()
	if len(roots) != 2 {
		t.Fatalf("AllRoots returned %d entries, want 2", len(roots))
	}

	pathSet := map[string]bool{}
	for _, e := range roots {
		pathSet[e.Path] = true
	}

	for _, want := range []string{root1, root2} {
		if !pathSet[want] {
			t.Errorf("AllRoots missing entry for path %q", want)
		}
	}
}

func TestRegistry_AllRoots_EmptyRegistry(t *testing.T) {
	reg := NewRegistry(nil)
	roots := reg.AllRoots()
	if len(roots) != 0 {
		t.Errorf("AllRoots on empty registry = %v, want empty slice", roots)
	}
}

// --- Replace ---

func TestRegistry_Replace_SwapsBackends(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "b1")
	b2 := NewLocalBackend(root2, "b2")

	reg := NewRegistry([]Backend{b1})

	// Initially only b1 is registered.
	if reg.BackendFor(filepath.Join(root2, "file.jpg")) != nil {
		t.Error("expected nil before Replace")
	}

	// After Replace with b2, b2 should be found.
	reg.Replace([]Backend{b2})

	got := reg.BackendFor(filepath.Join(root2, "file.jpg"))
	if got != b2 {
		t.Error("expected b2 after Replace")
	}

	// b1 should no longer be found.
	if reg.BackendFor(filepath.Join(root1, "file.jpg")) != nil {
		t.Error("expected nil for b1 after Replace")
	}
}

// --- AllBackends ---

func TestRegistry_AllBackends_ReturnsCopy(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "b1")
	b2 := NewLocalBackend(root2, "b2")

	reg := NewRegistry([]Backend{b1, b2})

	backends := reg.AllBackends()
	if len(backends) != 2 {
		t.Fatalf("AllBackends returned %d, want 2", len(backends))
	}

	// Mutating the returned slice must not affect the registry.
	backends[0] = nil
	if reg.AllBackends()[0] == nil {
		t.Error("AllBackends returned a reference to the internal slice")
	}
}

// --- DefaultBackend ---

func TestRegistry_DefaultBackend_ExplicitDefault(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "b1")
	b2 := NewLocalBackend(root2, "b2")

	reg := NewRegistry([]Backend{b1, b2})
	reg.defaultIdx = 1 // mark b2 as default

	got := reg.DefaultBackend()
	if got != b2 {
		t.Error("DefaultBackend should return b2 (defaultIdx=1)")
	}
}

func TestRegistry_DefaultBackend_FallsBackToFirst(t *testing.T) {
	root := t.TempDir()
	b := NewLocalBackend(root, "only")
	reg := NewRegistry([]Backend{b})

	got := reg.DefaultBackend()
	if got != b {
		t.Error("DefaultBackend should return the first backend when no default is set")
	}
}

func TestRegistry_DefaultBackend_EmptyRegistry(t *testing.T) {
	reg := NewRegistry(nil)
	got := reg.DefaultBackend()
	if got != nil {
		t.Errorf("DefaultBackend on empty registry = %v, want nil", got)
	}
}

func TestRegistry_ReplaceWithDefault(t *testing.T) {
	root1 := t.TempDir()
	root2 := t.TempDir()

	b1 := NewLocalBackend(root1, "b1")
	b2 := NewLocalBackend(root2, "b2")

	reg := NewRegistry([]Backend{b1})
	reg.ReplaceWithDefault([]Backend{b1, b2}, 1)

	got := reg.DefaultBackend()
	if got != b2 {
		t.Error("DefaultBackend after ReplaceWithDefault should return b2")
	}
}

// --- Integration: Upload + Download + Exists via Registry ---

func TestRegistry_UploadDownloadExistsRoundTrip(t *testing.T) {
	root := t.TempDir()
	b := NewLocalBackend(root, "test")
	reg := NewRegistry([]Backend{b})

	ctx := context.Background()
	dest := filepath.Join(root, "uploads", "clip.mp4")
	content := "fake video bytes"

	// Upload through the backend obtained from registry.
	backend := reg.BackendFor(root)
	if backend == nil {
		t.Fatal("BackendFor root returned nil")
	}

	if err := backend.Upload(ctx, dest, strings.NewReader(content), "video/mp4"); err != nil {
		t.Fatalf("Upload: %v", err)
	}

	// Exists
	ok, err := backend.Exists(ctx, dest)
	if err != nil {
		t.Fatalf("Exists: %v", err)
	}
	if !ok {
		t.Fatal("Exists = false after Upload")
	}

	// Download
	rc, err := backend.Download(ctx, dest)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	defer rc.Close()

	data, err := io.ReadAll(rc)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if string(data) != content {
		t.Errorf("Download content = %q, want %q", string(data), content)
	}
}
