package tasks

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/storage"
)

// withStorageRegistry sets the package storageReg for the duration of a test
// and restores the previous value afterwards.
func withStorageRegistry(t *testing.T, r *storage.Registry) {
	t.Helper()
	prev := storageReg
	storageReg = r
	t.Cleanup(func() { storageReg = prev })
}

func TestResolveIngestDirLocalIsDirect(t *testing.T) {
	root := t.TempDir()
	reg := storage.NewRegistry([]storage.Backend{
		storage.NewLocalBackend(root, "local"),
	})
	withStorageRegistry(t, reg)

	target, err := resolveIngestDir("job-1", "downloads/")
	if err != nil {
		t.Fatalf("resolveIngestDir: %v", err)
	}
	if !target.direct {
		t.Fatalf("expected direct mode for a local backend")
	}
	want := filepath.Join(root, "downloads")
	if target.dir != want {
		t.Fatalf("dir = %q, want %q", target.dir, want)
	}
	// Direct mode targets the final location, not a temp staging dir.
	if strings.Contains(target.dir, "staging") {
		t.Fatalf("direct dir should not be under staging: %q", target.dir)
	}
	// cleanup must be a no-op: the download dir must survive it.
	target.cleanup()
	if _, err := os.Stat(target.dir); err != nil {
		t.Fatalf("direct download dir should survive cleanup: %v", err)
	}
}

func TestResolveIngestDirNoBackendStages(t *testing.T) {
	withStorageRegistry(t, nil)

	target, err := resolveIngestDir("job-2", "downloads/")
	if err != nil {
		t.Fatalf("resolveIngestDir: %v", err)
	}
	if target.direct {
		t.Fatalf("expected staging mode when no backend is configured")
	}
	if !strings.Contains(target.dir, "staging") {
		t.Fatalf("staging dir expected under a staging path: %q", target.dir)
	}
	// cleanup must remove the staging dir.
	target.cleanup()
	if _, err := os.Stat(target.dir); err == nil {
		t.Fatalf("staging dir should be removed by cleanup")
	}
}
