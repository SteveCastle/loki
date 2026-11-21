package media

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// library_settings holds settings that belong to a library rather than to the
// server install (storage roots: the media paths in a library only make sense
// against that library's roots), so switching databases switches them too.

func ensureLibrarySettingsTable(db *sql.DB) error {
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS library_settings (
			"key" TEXT PRIMARY KEY,
			value TEXT NOT NULL
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create library_settings table: %w", err)
	}
	return nil
}

// GetLibrarySetting returns the stored value for key; ok is false when the
// library has never stored one (including databases that predate the table).
func GetLibrarySetting(db *sql.DB, key string) (value string, ok bool, err error) {
	err = db.QueryRow(`SELECT value FROM library_settings WHERE "key" = ?`, key).Scan(&value)
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		if strings.Contains(err.Error(), "no such table") {
			return "", false, nil
		}
		return "", false, err
	}
	return value, true, nil
}

// SetLibrarySetting stores value under key, creating the table if needed.
func SetLibrarySetting(db *sql.DB, key, value string) error {
	if err := ensureLibrarySettingsTable(db); err != nil {
		return err
	}
	_, err := db.Exec(`INSERT INTO library_settings ("key", value) VALUES (?, ?)
		ON CONFLICT("key") DO UPDATE SET value = excluded.value`, key, value)
	return err
}
