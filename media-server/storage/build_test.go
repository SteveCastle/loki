package storage

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestBuildRegistry_LocalRoots(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "local", Path: "/mnt/photos", Label: "Photos"},
		{Type: "local", Path: "/mnt/videos", Label: "Videos"},
	}

	reg, errs := BuildRegistry(roots)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	allRoots := reg.AllRoots()
	if len(allRoots) != 2 {
		t.Fatalf("expected 2 roots, got %d", len(allRoots))
	}
	if allRoots[0].Name != "Photos" {
		t.Errorf("root[0].Name = %q, want 'Photos'", allRoots[0].Name)
	}
}

func TestBuildRegistry_EmptyTypeDefaultsToLocal(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "", Path: "/data", Label: "Data"},
	}
	reg, errs := BuildRegistry(roots)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}
	if len(reg.AllRoots()) != 1 {
		t.Fatal("expected 1 root")
	}
}

func TestBuildRegistry_S3MissingBucket(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "local", Path: "/mnt/photos", Label: "Photos"},
		{Type: "s3", Label: "Bad S3"},
	}
	reg, errs := BuildRegistry(roots)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}
	if len(reg.AllRoots()) != 1 {
		t.Fatalf("expected 1 root (local only), got %d", len(reg.AllRoots()))
	}
}

func TestBuildRegistry_UnknownType(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "ftp", Label: "FTP"},
	}
	reg, errs := BuildRegistry(roots)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d", len(errs))
	}
	if len(reg.AllRoots()) != 0 {
		t.Fatal("expected 0 roots")
	}
}
