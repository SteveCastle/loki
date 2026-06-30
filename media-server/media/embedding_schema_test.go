package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

func TestInitializeSchemaCreatesEmbeddingTable(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("InitializeSchema: %v", err)
	}
	var name string
	err = db.QueryRow(
		`SELECT name FROM sqlite_master WHERE type='table' AND name='media_embedding'`,
	).Scan(&name)
	if err != nil {
		t.Fatalf("media_embedding table not found: %v", err)
	}
	// Insert + read back a row to confirm columns.
	if _, err := db.Exec(
		`INSERT INTO media_embedding (media_path, model, dim, vector, created_at) VALUES (?,?,?,?,?)`,
		"a.jpg", "m", 2, []byte{1, 2, 3, 4, 5, 6, 7, 8}, 0,
	); err != nil {
		t.Fatalf("insert: %v", err)
	}
}
