package main

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/media"
	_ "modernc.org/sqlite"
)

func openLibraryForRootsTest(t *testing.T, name string) *sql.DB {
	t.Helper()
	db, err := sql.Open("sqlite", sqliteDSN(filepath.Join(t.TempDir(), name)))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatal(err)
	}
	return db
}

// Switching libraries switches storage roots: each library keeps its own
// set, a library without one is seeded with the active roots, and a stored
// empty list is a deliberate clear rather than "unset".
func TestAdoptLibraryRootsFollowsDatabase(t *testing.T) {
	t.Setenv("LOWKEY_CONFIG_PATH", filepath.Join(t.TempDir(), "config.json"))
	t.Setenv("LOWKEY_DEFAULT_ROOT", "")
	fileRoots := []appconfig.StorageRoot{{Type: "local", Label: "File", Path: "/file"}}
	appconfig.Set(appconfig.Config{Roots: fileRoots})
	t.Cleanup(func() {
		currentConfig = appconfig.Config{}
		appconfig.ClearLibraryRoots()
	})

	libA := openLibraryForRootsTest(t, "a.sqlite")
	libB := openLibraryForRootsTest(t, "b.sqlite")

	// A library from before roots lived in the DB adopts config.json's roots
	// and stores them.
	if got := adoptLibraryRoots(libA); len(got) != 1 || got[0].Path != "/file" {
		t.Fatalf("seeded roots = %+v", got)
	}
	if stored, ok, err := loadLibraryRoots(libA); err != nil || !ok || stored[0].Path != "/file" {
		t.Fatalf("library A not seeded: %+v ok=%v err=%v", stored, ok, err)
	}

	rootsB := []appconfig.StorageRoot{{Type: "local", Label: "B", Path: "/b", Default: true}}
	if err := saveLibraryRoots(libB, rootsB); err != nil {
		t.Fatal(err)
	}
	if got := adoptLibraryRoots(libB); len(got) != 1 || got[0].Path != "/b" {
		t.Fatalf("library B roots = %+v", got)
	}
	if r := appconfig.Get().Roots; len(r) != 1 || r[0].Path != "/b" {
		t.Fatalf("in-memory config not switched: %+v", r)
	}
	if r := currentConfig.Roots; len(r) != 1 || r[0].Path != "/b" {
		t.Fatalf("currentConfig not switched: %+v", r)
	}

	// Re-reading config.json (every config GET does) must not revert to the
	// file's roots.
	if cfg, _, err := appconfig.Load(); err != nil || len(cfg.Roots) != 1 || cfg.Roots[0].Path != "/b" {
		t.Fatalf("Load after adopt: roots = %+v err=%v", cfg.Roots, err)
	}
	if r := appconfig.Get().Roots; len(r) != 1 || r[0].Path != "/b" {
		t.Fatalf("in-memory roots reverted by Load: %+v", r)
	}

	// Back to A: its own roots, not B's.
	if got := adoptLibraryRoots(libA); len(got) != 1 || got[0].Path != "/file" {
		t.Fatalf("library A roots after switching back = %+v", got)
	}

	// An explicit clear survives a round trip.
	if err := saveLibraryRoots(libB, nil); err != nil {
		t.Fatal(err)
	}
	if got := adoptLibraryRoots(libB); len(got) != 0 {
		t.Fatalf("cleared library roots = %+v", got)
	}
}

func TestPersistLibraryRootsSkipsUnchanged(t *testing.T) {
	t.Setenv("LOWKEY_CONFIG_PATH", filepath.Join(t.TempDir(), "config.json"))
	t.Cleanup(appconfig.ClearLibraryRoots)
	db := openLibraryForRootsTest(t, "lib.sqlite")
	roots := []appconfig.StorageRoot{{Type: "local", Label: "Env", Path: "/env"}}

	// An unchanged echo (e.g. env-injected roots posted back by the config
	// page) must not be written to the library.
	if err := persistLibraryRoots(db, roots, roots); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := loadLibraryRoots(db); ok {
		t.Fatal("unchanged roots were written to the library")
	}

	edited := []appconfig.StorageRoot{{Type: "local", Label: "Edited", Path: "/edited"}}
	if err := persistLibraryRoots(db, edited, roots); err != nil {
		t.Fatal(err)
	}
	if got, ok, _ := loadLibraryRoots(db); !ok || got[0].Path != "/edited" {
		t.Fatalf("edited roots not stored: %+v", got)
	}
}

// Databases created before library_settings existed read as "no roots".
func TestLoadLibraryRootsWithoutTable(t *testing.T) {
	db, err := sql.Open("sqlite", sqliteDSN(filepath.Join(t.TempDir(), "old.sqlite")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if _, ok, err := loadLibraryRoots(db); err != nil || ok {
		t.Fatalf("ok=%v err=%v, want no roots and no error", ok, err)
	}
}
