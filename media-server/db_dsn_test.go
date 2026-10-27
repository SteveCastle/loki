package main

import (
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite"
)

func TestSqliteDSNOpensInWALMode(t *testing.T) {
	db, err := sql.Open("sqlite", sqliteDSN(filepath.Join(t.TempDir(), "lib.sqlite")))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var mode string
	if err := db.QueryRow(`PRAGMA journal_mode`).Scan(&mode); err != nil {
		t.Fatal(err)
	}
	if mode != "wal" {
		t.Fatalf("journal_mode = %q, want wal", mode)
	}
	var sync, busy int
	_ = db.QueryRow(`PRAGMA synchronous`).Scan(&sync)
	_ = db.QueryRow(`PRAGMA busy_timeout`).Scan(&busy)
	if sync != 1 || busy != 5000 {
		t.Fatalf("synchronous = %d (want 1/NORMAL), busy_timeout = %d (want 5000)", sync, busy)
	}
}
