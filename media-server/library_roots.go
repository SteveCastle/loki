package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"reflect"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/storage"
)

// Storage roots belong to the library, not the install: every media path in a
// library lives under that library's roots, so they are stored in the library
// database (library_settings) and switching databases switches roots.
// config.json's "roots" is only the seed for a library that has none yet.
const libraryRootsKey = "storage_roots"

// loadLibraryRoots returns the roots stored in db; ok is false when the
// library has never stored any. A stored empty list is a deliberate clear.
func loadLibraryRoots(db *sql.DB) (roots []appconfig.StorageRoot, ok bool, err error) {
	raw, ok, err := media.GetLibrarySetting(db, libraryRootsKey)
	if err != nil || !ok {
		return nil, false, err
	}
	if err := json.Unmarshal([]byte(raw), &roots); err != nil {
		return nil, false, fmt.Errorf("decode library storage roots: %w", err)
	}
	if roots == nil {
		roots = []appconfig.StorageRoot{}
	}
	return roots, true, nil
}

func saveLibraryRoots(db *sql.DB, roots []appconfig.StorageRoot) error {
	if roots == nil {
		roots = []appconfig.StorageRoot{}
	}
	raw, err := json.Marshal(roots)
	if err != nil {
		return err
	}
	return media.SetLibrarySetting(db, libraryRootsKey, string(raw))
}

// persistLibraryRoots writes edited roots to the library and makes them the
// active library roots. Unchanged roots are
// skipped so an env-injected set (LOWKEY_ROOTS) echoed back by the config page
// is never baked into the database.
func persistLibraryRoots(db *sql.DB, next, prev []appconfig.StorageRoot) error {
	if db == nil || reflect.DeepEqual(next, prev) {
		return nil
	}
	if err := saveLibraryRoots(db, next); err != nil {
		return err
	}
	if !appconfig.EnvRootsActive() {
		appconfig.SetLibraryRoots(next)
	}
	return nil
}

// adoptLibraryRoots makes db's roots the active set (in-memory config and
// currentConfig) and returns them. A library without stored roots is seeded
// with the currently active ones, which at boot are config.json's. Env roots
// still win at runtime and are never written to the library.
func adoptLibraryRoots(db *sql.DB) []appconfig.StorageRoot {
	defer func() { currentConfig.Roots = appconfig.Get().Roots }()
	if appconfig.EnvRootsActive() {
		return appconfig.Get().Roots
	}
	roots, ok, err := loadLibraryRoots(db)
	if err != nil {
		log.Printf("Warning: could not read storage roots from library database, keeping current roots: %v", err)
		return appconfig.Get().Roots
	}
	if !ok {
		roots = appconfig.Get().Roots
		if err := saveLibraryRoots(db, roots); err != nil {
			log.Printf("Warning: could not store storage roots in library database: %v", err)
		} else {
			log.Printf("Library had no storage roots; seeded it with the current %d root(s)", len(roots))
		}
	} else {
		log.Printf("Using %d storage root(s) from the library database", len(roots))
	}
	appconfig.SetLibraryRoots(roots)
	return appconfig.Get().Roots
}

// applyStorageRoots rebuilds the live storage registry in place (tasks and the
// cleanup resolvers hold the same *Registry).
func applyStorageRoots(reg *storage.Registry, roots []appconfig.StorageRoot) {
	if reg == nil {
		return
	}
	newReg, errs := storage.BuildRegistry(roots)
	for _, err := range errs {
		log.Printf("Warning: storage backend init error: %v", err)
	}
	reg.ReplaceWithDefault(newReg.AllBackends(), newReg.DefaultIdx())
}
