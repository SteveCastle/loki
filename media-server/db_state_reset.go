package main

import (
	"database/sql"
	"log"
	"sync"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
)

// resetDBDerivedState drops every piece of in-memory state that was derived
// from the previous database and schedules rebuilds against newDB. Called by
// switchDatabase (all platform mains) right after deps.DB/deps.Queue point at
// the new handles.
//
// Contract: everything old-DB-derived must be gone by the time this returns —
// a request racing the swap may get an empty or slow answer (brute-force
// similarity, empty swipe page, stats "ready:false") but never rows from the
// old library. Rebuilds run in the background so the swap request stays fast.
func resetDBDerivedState(newDB *sql.DB) {
	// Vector indexes (similarity + faces): a nil index makes searches
	// brute-force against the current deps.DB — correct, just slower —
	// until the background rebuild installs the new-DB index.
	tasks.SetVectorIndexForModel(nil, "")
	tasks.SetFaceIndexForModel(nil, "", nil)

	// Home-page stats: the snapshot and progress deltas count the old
	// library; the generation bump also discards any recount in flight.
	resetLibraryStats()

	// Swipe: the random-sampler universe is old-DB paths, and the For-You
	// feed engine caches a taste profile, session feeds, and — critically —
	// the old *sql.DB itself, which is about to be closed.
	media.ResetRandomSampleCache()
	resetSwipeFeedEngine(newDB)

	// Pending CLI device-auth codes are bound to old-DB usernames; redeeming
	// one now would mint an API key for whoever shares that name in the new
	// database.
	resetCLIAuthCodes()

	// SPA per-session state (filters, selected paths) references old-DB
	// paths. Settings are user preferences and survive the swap.
	lokiSession = make(map[string]any)

	// Warm the caches the next request would otherwise pay for.
	media.WarmRandomSampleCache(newDB)
	warmLibraryStats(deps)
	go rebuildIndexesAfterSwap(newDB)
}

// swapRebuildMu serialises index rebuilds across rapid consecutive swaps so a
// slow rebuild against database A can never install after — and clobber — the
// rebuild against database B.
var swapRebuildMu sync.Mutex

// rebuildIndexesAfterSwap rebuilds the media and face vector indexes from
// newDB, skipping installation when another swap has superseded newDB (the
// newer swap's goroutine owns the indexes then).
func rebuildIndexesAfterSwap(newDB *sql.DB) {
	swapRebuildMu.Lock()
	defer swapRebuildMu.Unlock()

	superseded := func() bool { return deps.DB != newDB }

	if superseded() {
		return
	}
	model := tasks.ActiveEmbedModel()
	idx, err := tasks.BuildIndexFromDB(newDB, model.ID, nil)
	switch {
	case err != nil:
		log.Printf("embedding index rebuild after database switch failed (model %s): %v", model.ID, err)
	case superseded():
		// A newer swap landed while we were scanning; leave the index alone.
	default:
		tasks.SetVectorIndexForModel(idx, model.ID)
		log.Printf("embedding index rebuilt after database switch: %d vectors (model %s)", idx.Len(), model.ID)
	}

	if superseded() {
		return
	}
	faceModel := tasks.ActiveFaceModel()
	fidx, pathKeys, err := tasks.BuildFaceIndexFromDB(newDB, faceModel.ID, nil)
	switch {
	case err != nil:
		log.Printf("face index rebuild after database switch failed (model %s): %v", faceModel.ID, err)
	case superseded():
	default:
		tasks.SetFaceIndexForModel(fidx, faceModel.ID, pathKeys)
		log.Printf("face index rebuilt after database switch: %d faces (model %s)", fidx.Len(), faceModel.ID)
	}
}
