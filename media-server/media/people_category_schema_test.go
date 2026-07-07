package media

import (
	"database/sql"
	"testing"

	_ "modernc.org/sqlite"
)

// The People taxonomy category must exist on every freshly initialized DB,
// even before any face has been clustered — otherwise the taxonomy shows no
// People tab until the first person is named. InitializeSchema seeds it.
func TestInitializeSchemaSeedsPeopleCategory(t *testing.T) {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("InitializeSchema: %v", err)
	}
	var n int
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM category WHERE label = ?`, PeopleCategory,
	).Scan(&n); err != nil {
		t.Fatalf("query category: %v", err)
	}
	if n != 1 {
		t.Fatalf("People category rows = %d, want 1", n)
	}

	// Re-running must be idempotent (no duplicate, no error) — InitializeSchema
	// runs on every boot and on every DB swap.
	if err := InitializeSchema(db); err != nil {
		t.Fatalf("InitializeSchema (rerun): %v", err)
	}
	if err := db.QueryRow(
		`SELECT COUNT(*) FROM category WHERE label = ?`, PeopleCategory,
	).Scan(&n); err != nil {
		t.Fatalf("query category (rerun): %v", err)
	}
	if n != 1 {
		t.Fatalf("People category rows after rerun = %d, want 1", n)
	}
}
