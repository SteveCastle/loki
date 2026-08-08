package main

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/tasks"
	_ "modernc.org/sqlite"
)

// TestResetDBDerivedState is the DB hot-swap contract: after switchDatabase
// calls resetDBDerivedState, nothing derived from the old database may be
// served — the vector/face indexes, stats snapshot, CLI auth codes, and SPA
// session state are dropped synchronously — and the background rebuilds must
// land with NEW-database contents.
func TestResetDBDerivedState(t *testing.T) {
	t.Setenv("LOWKEY_CONFIG_PATH", filepath.Join(t.TempDir(), "config.json"))
	appconfig.Set(appconfig.Config{JWTSecret: "test-secret"})

	// The new database: one media item with an embedding for the active model.
	// File-backed, not :memory: — the background rebuilds run on their own
	// pool connections, and modernc gives every :memory: connection a
	// separate empty database.
	newDB, err := sql.Open("sqlite", sqliteDSN(filepath.Join(t.TempDir(), "new.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { newDB.Close() })
	if err := media.InitializeSchema(newDB); err != nil {
		t.Fatal(err)
	}
	embedModel := tasks.ActiveEmbedModel().ID
	if _, err := newDB.Exec(`INSERT INTO media (path) VALUES ('/new/x.jpg')`); err != nil {
		t.Fatal(err)
	}
	if err := media.UpsertEmbedding(newDB, "/new/x.jpg", embedModel, []float32{0.6, 0.8}, 0); err != nil {
		t.Fatal(err)
	}

	// Old-database residue in the process globals. The old vector index holds
	// TWO entries so a rebuilt one (exactly one entry, from newDB) is
	// distinguishable from a survivor.
	oldIdx := embedindex.New()
	oldIdx.Add("/old/a.jpg", []float32{1, 0})
	oldIdx.Add("/old/b.jpg", []float32{0, 1})
	tasks.SetVectorIndexForModel(oldIdx, embedModel)
	oldFaces := embedindex.New()
	oldFaces.Add("1", []float32{1, 0})
	tasks.SetFaceIndexForModel(oldFaces, tasks.ActiveFaceModel().ID, map[string][]string{"/old/a.jpg": {"1"}})
	libStats.mu.Lock()
	libStats.snapshot = &statsAPIResponse{Ready: true, TotalMedia: 999}
	libStats.deltas = map[string]int{"description": 3}
	libStats.mu.Unlock()
	if _, err := mintCLIAuthCode("old-db-user", "challenge", "key"); err != nil {
		t.Fatal(err)
	}
	lokiSession["filters"] = "old-library-state"

	oldDeps := deps
	deps = &Dependencies{DB: newDB}
	// Registered after the newDB close above, so (LIFO) this runs first:
	// globals are detached from the test DB before it closes.
	t.Cleanup(func() {
		deps = oldDeps
		tasks.SetVectorIndexForModel(nil, "")
		tasks.SetFaceIndexForModel(nil, "", nil)
		resetLibStats()
		media.ResetRandomSampleCache()
	})

	resetDBDerivedState(newDB)

	// Synchronous guarantees: old-DB state is unreachable the moment the call
	// returns. (The rebuilds may or may not have landed yet, so the index can
	// legitimately hold 0 or 1 vectors here — but never the old two.)
	if n := tasks.IndexSize(); n >= 2 {
		t.Fatalf("old vector index still installed (%d vectors)", n)
	}
	cliAuthCodes.Lock()
	pending := len(cliAuthCodes.m)
	cliAuthCodes.Unlock()
	if pending != 0 {
		t.Fatalf("expected pending CLI auth codes to be dropped, %d remain", pending)
	}
	if len(lokiSession) != 0 {
		t.Fatalf("expected SPA session state to be cleared, got %v", lokiSession)
	}
	libStats.mu.Lock()
	snap := libStats.snapshot
	libStats.mu.Unlock()
	if snap != nil && snap.TotalMedia == 999 {
		t.Fatal("old stats snapshot survived the reset")
	}

	// Eventual guarantees: the background rebuilds install NEW-database state.
	deadline := time.Now().Add(10 * time.Second)
	for {
		vectorsOK := tasks.IndexedModel() == embedModel && tasks.IndexSize() == 1
		facesOK := tasks.FaceIndexedModel() == tasks.ActiveFaceModel().ID && tasks.FaceIndexSize() == 0
		libStats.mu.Lock()
		snap = libStats.snapshot
		computing := libStats.computing
		libStats.mu.Unlock()
		statsOK := snap != nil && !computing
		if vectorsOK && facesOK && statsOK {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("rebuilds did not land: vectors(model=%q size=%d) faces(model=%q size=%d) stats(snapshot=%v computing=%v)",
				tasks.IndexedModel(), tasks.IndexSize(), tasks.FaceIndexedModel(), tasks.FaceIndexSize(), snap != nil, computing)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snap.TotalMedia != 1 {
		t.Fatalf("rebuilt stats snapshot TotalMedia = %d, want 1 (new database)", snap.TotalMedia)
	}
}
