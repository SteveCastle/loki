package media

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	_ "modernc.org/sqlite"
)

// TestFormatBytes tests the FormatBytes function
func TestFormatBytes(t *testing.T) {
	tests := []struct {
		name     string
		bytes    int64
		expected string
	}{
		{"Zero bytes", 0, "0 B"},
		{"Bytes", 512, "512.0 B"},
		{"Kilobytes", 1024, "1.0 KB"},
		{"Megabytes", 1024 * 1024, "1.0 MB"},
		{"Gigabytes", 1024 * 1024 * 1024, "1.0 GB"},
		{"Terabytes", 1024 * 1024 * 1024 * 1024, "1.0 TB"},
		{"Mixed KB", 1536, "1.5 KB"},
		{"Mixed MB", 2.5 * 1024 * 1024, "2.5 MB"},
		{"Large number", 999 * 1024 * 1024, "999.0 MB"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatBytes(tt.bytes)
			if result != tt.expected {
				t.Errorf("FormatBytes(%d) = %s, want %s", tt.bytes, result, tt.expected)
			}
		})
	}
}

// TestRemoteExistsRouting verifies s3:// paths are answered by the wired
// remote checker (and default to existing when none is wired) while local
// paths still go through os.Stat.
func TestRemoteExistsRouting(t *testing.T) {
	// No checker wired: remote paths must report as existing — "unknown"
	// must not render as missing or filter items out of samplers.
	SetRemoteExistsChecker(nil)
	if !CheckFileExists("s3://bucket/unknown.jpg") {
		t.Error("remote path should default to existing when no checker is wired")
	}

	SetRemoteExistsChecker(func(paths []string) map[string]bool {
		out := make(map[string]bool, len(paths))
		for _, p := range paths {
			out[p] = p == "s3://bucket/yes.jpg"
		}
		return out
	})
	defer SetRemoteExistsChecker(nil)

	tmpFile, err := os.CreateTemp("", "remote_exists_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()
	missingLocal := filepath.Join(filepath.Dir(tmpFile.Name()), "definitely-missing-98431.txt")

	got := CheckFilesExistConcurrent([]string{
		"s3://bucket/yes.jpg", "s3://bucket/no.jpg", tmpFile.Name(), missingLocal,
	})
	want := map[string]bool{
		"s3://bucket/yes.jpg": true,
		"s3://bucket/no.jpg":  false,
		tmpFile.Name():        true,
		missingLocal:          false,
	}
	for p, exp := range want {
		if got[p] != exp {
			t.Errorf("CheckFilesExistConcurrent[%s] = %v, want %v", p, got[p], exp)
		}
	}

	if !CheckFileExists("s3://bucket/yes.jpg") || CheckFileExists("s3://bucket/no.jpg") {
		t.Error("CheckFileExists should route s3:// paths through the remote checker")
	}
}

// TestCheckFileExists tests the CheckFileExists function
func TestCheckFileExists(t *testing.T) {
	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "test_file_*.txt")
	if err != nil {
		t.Fatalf("Failed to create temporary file: %v", err)
	}
	defer os.Remove(tmpFile.Name())
	tmpFile.Close()

	tests := []struct {
		name     string
		path     string
		expected bool
	}{
		{"Existing file", tmpFile.Name(), true},
		{"Non-existent file", "/path/that/does/not/exist", false},
		{"Empty path", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CheckFileExists(tt.path)
			if result != tt.expected {
				t.Errorf("CheckFileExists(%s) = %v, want %v", tt.path, result, tt.expected)
			}
		})
	}
}

// TestCheckFilesExistConcurrent tests the CheckFilesExistConcurrent function
func TestCheckFilesExistConcurrent(t *testing.T) {
	// Create temporary files
	tmpDir, err := os.MkdirTemp("", "test_concurrent_*")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create some test files
	existingFiles := []string{
		filepath.Join(tmpDir, "file1.txt"),
		filepath.Join(tmpDir, "file2.txt"),
		filepath.Join(tmpDir, "file3.txt"),
	}

	for _, file := range existingFiles {
		if err := os.WriteFile(file, []byte("test content"), 0644); err != nil {
			t.Fatalf("Failed to create test file %s: %v", file, err)
		}
	}

	nonExistentFiles := []string{
		filepath.Join(tmpDir, "missing1.txt"),
		filepath.Join(tmpDir, "missing2.txt"),
	}

	allFiles := append(existingFiles, nonExistentFiles...)

	// Test with all files
	result := CheckFilesExistConcurrent(allFiles)

	if len(result) != len(allFiles) {
		t.Errorf("CheckFilesExistConcurrent returned %d results, want %d", len(result), len(allFiles))
	}

	// Check existing files
	for _, file := range existingFiles {
		if exists, found := result[file]; !found || !exists {
			t.Errorf("CheckFilesExistConcurrent: file %s should exist but got exists=%v, found=%v", file, exists, found)
		}
	}

	// Check non-existent files
	for _, file := range nonExistentFiles {
		if exists, found := result[file]; !found || exists {
			t.Errorf("CheckFilesExistConcurrent: file %s should not exist but got exists=%v, found=%v", file, exists, found)
		}
	}

	// Test with empty slice
	emptyResult := CheckFilesExistConcurrent([]string{})
	if len(emptyResult) != 0 {
		t.Errorf("CheckFilesExistConcurrent with empty slice should return empty map, got %d items", len(emptyResult))
	}
}

// TestMediaItemMarshalJSON tests the custom JSON marshaling for MediaItem
func TestMediaItemMarshalJSON(t *testing.T) {
	tests := []struct {
		name     string
		item     MediaItem
		expected map[string]interface{}
	}{
		{
			name: "All fields valid",
			item: MediaItem{
				Path:        "/path/to/file.jpg",
				Description: sql.NullString{String: "Test description", Valid: true},
				Size:        sql.NullInt64{Int64: 1024, Valid: true},
				Hash:        sql.NullString{String: "abc123", Valid: true},
				Width:       sql.NullInt64{Int64: 1920, Valid: true},
				Height:      sql.NullInt64{Int64: 1080, Valid: true},
				Tags:        []MediaTag{{Label: "test", Category: "category"}},
				Exists:      true,
			},
			expected: map[string]interface{}{
				"path":        "/path/to/file.jpg",
				"description": "Test description",
				"size":        int64(1024),
				"hash":        "abc123",
				"width":       int64(1920),
				"height":      int64(1080),
				"tags":        []interface{}{map[string]interface{}{"label": "test", "category": "category"}},
				"exists":      true,
			},
		},
		{
			name: "Nullable fields invalid",
			item: MediaItem{
				Path:        "/path/to/file.jpg",
				Description: sql.NullString{Valid: false},
				Size:        sql.NullInt64{Valid: false},
				Hash:        sql.NullString{Valid: false},
				Width:       sql.NullInt64{Valid: false},
				Height:      sql.NullInt64{Valid: false},
				Tags:        []MediaTag{},
				Exists:      false,
			},
			expected: map[string]interface{}{
				"path":        "/path/to/file.jpg",
				"description": nil,
				"size":        nil,
				"hash":        nil,
				"width":       nil,
				"height":      nil,
				"tags":        []interface{}{},
				"exists":      false,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.item)
			if err != nil {
				t.Errorf("json.Marshal() error = %v", err)
				return
			}

			var result map[string]interface{}
			if err := json.Unmarshal(data, &result); err != nil {
				t.Errorf("json.Unmarshal() error = %v", err)
				return
			}

			// Compare each field
			for key, expectedValue := range tt.expected {
				if actualValue, exists := result[key]; !exists {
					t.Errorf("Missing field %s in JSON output", key)
				} else {
					// Handle special case for tags slice comparison
					if key == "tags" {
						if !compareTags(actualValue, expectedValue) {
							t.Errorf("Field %s = %v, want %v", key, actualValue, expectedValue)
						}
					} else if key == "size" || key == "width" || key == "height" {
						// Handle numeric fields that might be unmarshaled as float64
						if expectedValue != nil {
							if actualFloat, ok := actualValue.(float64); ok {
								if expectedInt, ok := expectedValue.(int64); ok {
									if int64(actualFloat) != expectedInt {
										t.Errorf("Field %s = %v, want %v", key, actualValue, expectedValue)
									}
								}
							} else if !reflect.DeepEqual(actualValue, expectedValue) {
								t.Errorf("Field %s = %v, want %v", key, actualValue, expectedValue)
							}
						} else if actualValue != nil {
							t.Errorf("Field %s = %v, want %v", key, actualValue, expectedValue)
						}
					} else if !reflect.DeepEqual(actualValue, expectedValue) {
						t.Errorf("Field %s = %v, want %v", key, actualValue, expectedValue)
					}
				}
			}
		})
	}
}

// compareTags is a helper function to compare tag slices in JSON
func compareTags(actual, expected interface{}) bool {
	actualSlice, ok1 := actual.([]interface{})
	expectedSlice, ok2 := expected.([]interface{})

	if !ok1 || !ok2 {
		return false
	}

	if len(actualSlice) != len(expectedSlice) {
		return false
	}

	for i, actualTag := range actualSlice {
		expectedTag := expectedSlice[i]
		if !reflect.DeepEqual(actualTag, expectedTag) {
			return false
		}
	}

	return true
}

// setupTestDB creates a test database with schema
func setupTestDB(t *testing.T) *sql.DB {
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("Failed to open test database: %v", err)
	}

	// Create media table
	mediaTableQuery := `
		CREATE TABLE media (
			path TEXT PRIMARY KEY,
			description TEXT,
			transcript TEXT,
			size INTEGER,
			hash TEXT,
			width INTEGER,
			height INTEGER
		)
	`
	if _, err := db.Exec(mediaTableQuery); err != nil {
		t.Fatalf("Failed to create media table: %v", err)
	}

	// Create media_tag_by_category table
	tagTableQuery := `
		CREATE TABLE media_tag_by_category (
			media_path TEXT,
			tag_label TEXT,
			category_label TEXT,
			FOREIGN KEY (media_path) REFERENCES media(path)
		)
	`
	if _, err := db.Exec(tagTableQuery); err != nil {
		t.Fatalf("Failed to create media_tag_by_category table: %v", err)
	}

	// Create media_embedding table (required by RemoveItemsFromDB).
	if _, err := db.Exec(`
		CREATE TABLE media_embedding (
			media_path TEXT NOT NULL,
			model      TEXT NOT NULL,
			dim        INTEGER NOT NULL,
			vector     BLOB NOT NULL,
			created_at INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (media_path, model)
		)
	`); err != nil {
		t.Fatalf("Failed to create media_embedding table: %v", err)
	}

	// Create face + face_scan tables (required by RemoveItemsFromDB).
	if _, err := db.Exec(`
		CREATE TABLE face (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			media_path  TEXT NOT NULL,
			model       TEXT NOT NULL,
			frame_ts    REAL NOT NULL DEFAULT 0,
			bbox_x      REAL NOT NULL,
			bbox_y      REAL NOT NULL,
			bbox_w      REAL NOT NULL,
			bbox_h      REAL NOT NULL,
			det_score   REAL NOT NULL,
			vector      BLOB NOT NULL,
			person_id   INTEGER,
			assigned_by TEXT,
			created_at  INTEGER
		)
	`); err != nil {
		t.Fatalf("Failed to create face table: %v", err)
	}
	if _, err := db.Exec(`
		CREATE TABLE face_scan (
			media_path TEXT NOT NULL,
			model      TEXT NOT NULL,
			face_count INTEGER NOT NULL DEFAULT 0,
			scanned_at INTEGER,
			PRIMARY KEY (media_path, model)
		)
	`); err != nil {
		t.Fatalf("Failed to create face_scan table: %v", err)
	}
	if _, err := db.Exec(`
		CREATE TABLE person (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			name          TEXT UNIQUE,
			cover_face_id INTEGER,
			created_at    INTEGER
		)
	`); err != nil {
		t.Fatalf("Failed to create person table: %v", err)
	}

	return db
}

// TestGetTags tests the GetTags function
func TestGetTags(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Insert test data
	testData := []struct {
		path     string
		tagLabel string
		category string
	}{
		{"/path/to/file1.jpg", "landscape", "composition"},
		{"/path/to/file1.jpg", "nature", "subject"},
		{"/path/to/file2.jpg", "portrait", "composition"},
		{"/path/to/file3.jpg", "urban", "subject"},
	}

	for _, data := range testData {
		_, err := db.Exec("INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)",
			data.path, data.tagLabel, data.category)
		if err != nil {
			t.Fatalf("Failed to insert test data: %v", err)
		}
	}

	// Test getting tags for existing paths
	result, err := GetTags(db, []string{"/path/to/file1.jpg", "/path/to/file2.jpg"})
	if err != nil {
		t.Fatalf("GetTags() error = %v", err)
	}

	if len(result) != 2 {
		t.Errorf("GetTags() returned %d paths, want 2", len(result))
	}

	// Check file1 tags
	file1Tags, exists := result["/path/to/file1.jpg"]
	if !exists {
		t.Error("GetTags() missing tags for file1.jpg")
	} else if len(file1Tags) != 2 {
		t.Errorf("GetTags() file1.jpg has %d tags, want 2", len(file1Tags))
	}

	// Check file2 tags
	file2Tags, exists := result["/path/to/file2.jpg"]
	if !exists {
		t.Error("GetTags() missing tags for file2.jpg")
	} else if len(file2Tags) != 1 {
		t.Errorf("GetTags() file2.jpg has %d tags, want 1", len(file2Tags))
	}

	// Test with empty slice
	emptyResult, err := GetTags(db, []string{})
	if err != nil {
		t.Errorf("GetTags() with empty slice error = %v", err)
	}
	if len(emptyResult) != 0 {
		t.Errorf("GetTags() with empty slice returned %d items, want 0", len(emptyResult))
	}

	// Test with non-existent paths
	nonExistentResult, err := GetTags(db, []string{"/non/existent/path.jpg"})
	if err != nil {
		t.Errorf("GetTags() with non-existent path error = %v", err)
	}
	if len(nonExistentResult) != 0 {
		t.Errorf("GetTags() with non-existent path returned %d items, want 0", len(nonExistentResult))
	}
}

// TestRemoveItemsFromDB tests the RemoveItemsFromDB function
func TestRemoveItemsFromDB(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Insert test data
	testPaths := []string{
		"/path/to/file1.jpg",
		"/path/to/file2.jpg",
		"/path/to/file3.jpg",
	}

	for _, path := range testPaths {
		_, err := db.Exec("INSERT INTO media (path, description, size) VALUES (?, ?, ?)",
			path, "Test description", 1024)
		if err != nil {
			t.Fatalf("Failed to insert test media: %v", err)
		}

		_, err = db.Exec("INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)",
			path, "test", "category")
		if err != nil {
			t.Fatalf("Failed to insert test tag: %v", err)
		}

		_, err = db.Exec("INSERT INTO media_embedding (media_path, model, dim, vector) VALUES (?, 'siglip2', 2, ?)",
			path, []byte{0, 1})
		if err != nil {
			t.Fatalf("Failed to insert test embedding: %v", err)
		}
	}

	// Test removing items
	ctx := context.Background()
	result, err := RemoveItemsFromDB(ctx, db, []string{testPaths[0], testPaths[1]})
	if err != nil {
		t.Fatalf("RemoveItemsFromDB() error = %v", err)
	}

	if result.MediaItemsRemoved != 2 {
		t.Errorf("RemoveItemsFromDB() removed %d media items, want 2", result.MediaItemsRemoved)
	}

	if result.TagsRemoved != 2 {
		t.Errorf("RemoveItemsFromDB() removed %d tags, want 2", result.TagsRemoved)
	}

	// Verify items are actually removed
	var count int
	err = db.QueryRow("SELECT COUNT(*) FROM media").Scan(&count)
	if err != nil {
		t.Fatalf("Failed to count remaining media items: %v", err)
	}
	if count != 1 {
		t.Errorf("Expected 1 remaining media item, got %d", count)
	}

	// Embeddings for the removed paths must go with them; the survivor keeps its row.
	for _, tc := range []struct {
		path string
		want int
	}{
		{testPaths[0], 0},
		{testPaths[1], 0},
		{testPaths[2], 1},
	} {
		if err := db.QueryRow("SELECT COUNT(*) FROM media_embedding WHERE media_path = ?", tc.path).Scan(&count); err != nil {
			t.Fatalf("Failed to count embeddings for %s: %v", tc.path, err)
		}
		if count != tc.want {
			t.Errorf("Expected %d embedding rows for %s after removal, got %d", tc.want, tc.path, count)
		}
	}

	// Test with empty slice
	emptyResult, err := RemoveItemsFromDB(ctx, db, []string{})
	if err != nil {
		t.Errorf("RemoveItemsFromDB() with empty slice error = %v", err)
	}
	if emptyResult.MediaItemsRemoved != 0 {
		t.Errorf("RemoveItemsFromDB() with empty slice removed %d items, want 0", emptyResult.MediaItemsRemoved)
	}

	// Test with nil database
	_, err = RemoveItemsFromDB(ctx, nil, []string{"test"})
	if err == nil {
		t.Error("RemoveItemsFromDB() with nil db should return error")
	}
}

// TestRemoveItemsFromDBStreamReportsCommittedBatches asserts the streaming
// contract the remove job's progress bar depends on: onBatch fires once per
// batch (not once at the end), Done advances monotonically to Total, and the
// rows are ALREADY gone when the callback runs — a caller may report the batch
// as durable progress.
func TestRemoveItemsFromDBStreamReportsCommittedBatches(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	const n = 1200 // > 2 batches at the 500-path batch size
	paths := make([]string, n)
	for i := range paths {
		paths[i] = fmt.Sprintf("/lib/file%04d.jpg", i)
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", paths[i]); err != nil {
			t.Fatalf("Failed to insert test media: %v", err)
		}
		if _, err := db.Exec("INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, 'test', 'category')", paths[i]); err != nil {
			t.Fatalf("Failed to insert test tag: %v", err)
		}
	}

	var batches []RemovalBatch
	result, err := RemoveItemsFromDBStream(context.Background(), db, paths, func(b RemovalBatch) {
		batches = append(batches, b)

		// The batch's rows must be committed by the time we're told about it.
		var remaining int
		if err := db.QueryRow("SELECT COUNT(*) FROM media WHERE path >= ? AND path <= ?", b.Paths[0], b.Paths[len(b.Paths)-1]).Scan(&remaining); err != nil {
			t.Errorf("count after batch: %v", err)
		}
		if remaining != 0 {
			t.Errorf("batch ending at %d: %d rows still present, want 0 (batch not committed before callback)", b.Done, remaining)
		}
	})
	if err != nil {
		t.Fatalf("RemoveItemsFromDBStream() error = %v", err)
	}

	if len(batches) != 3 {
		t.Fatalf("got %d batch callbacks, want 3 (500+500+200)", len(batches))
	}
	prev := 0
	for i, b := range batches {
		if b.Total != n {
			t.Errorf("batch %d: Total = %d, want %d", i, b.Total, n)
		}
		if b.Done <= prev {
			t.Errorf("batch %d: Done = %d, not advancing past %d", i, b.Done, prev)
		}
		if b.Done != prev+len(b.Paths) {
			t.Errorf("batch %d: Done = %d, want %d for %d paths", i, b.Done, prev+len(b.Paths), len(b.Paths))
		}
		if b.MediaItemsRemoved != int64(b.Done) {
			t.Errorf("batch %d: MediaItemsRemoved = %d, want running total %d", i, b.MediaItemsRemoved, b.Done)
		}
		prev = b.Done
	}
	if prev != n {
		t.Errorf("final Done = %d, want %d", prev, n)
	}
	if result.MediaItemsRemoved != n {
		t.Errorf("MediaItemsRemoved = %d, want %d", result.MediaItemsRemoved, n)
	}
}

// TestRemoveItemsFromDBWithEnforcedForeignKeys removes from a schema where the
// sidecar tables DO declare foreign keys to media(path) and enforcement is on —
// the shape of the Electron client's dream-x.sqlite, which carries FKs the Go
// schema never creates. Without the in-transaction deferral, one out-of-order
// DELETE aborts the whole batch with "FOREIGN KEY constraint failed".
func TestRemoveItemsFromDBWithEnforcedForeignKeys(t *testing.T) {
	db, err := sql.Open("sqlite", "file:fkremove?mode=memory&cache=shared&_pragma=foreign_keys=ON")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	defer db.Close()

	for _, stmt := range []string{
		`CREATE TABLE media (path TEXT PRIMARY KEY)`,
		`CREATE TABLE media_tag_by_category (
			media_path TEXT, tag_label TEXT, category_label TEXT,
			FOREIGN KEY (media_path) REFERENCES media(path))`,
		`CREATE TABLE media_embedding (
			media_path TEXT NOT NULL, model TEXT NOT NULL, dim INTEGER NOT NULL,
			vector BLOB NOT NULL, created_at INTEGER,
			PRIMARY KEY (media_path, model),
			FOREIGN KEY (media_path) REFERENCES media(path))`,
		`CREATE TABLE face (
			id INTEGER PRIMARY KEY AUTOINCREMENT, media_path TEXT NOT NULL,
			model TEXT NOT NULL, frame_ts REAL NOT NULL DEFAULT 0,
			bbox_x REAL NOT NULL, bbox_y REAL NOT NULL, bbox_w REAL NOT NULL,
			bbox_h REAL NOT NULL, det_score REAL NOT NULL, vector BLOB NOT NULL,
			person_id INTEGER, assigned_by TEXT, created_at INTEGER,
			FOREIGN KEY (media_path) REFERENCES media(path))`,
		`CREATE TABLE face_scan (
			media_path TEXT NOT NULL, model TEXT NOT NULL,
			face_count INTEGER NOT NULL DEFAULT 0, scanned_at INTEGER,
			PRIMARY KEY (media_path, model),
			FOREIGN KEY (media_path) REFERENCES media(path))`,
		`CREATE TABLE person (
			id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE,
			cover_face_id INTEGER, created_at INTEGER,
			FOREIGN KEY (cover_face_id) REFERENCES face(id))`,
		`INSERT INTO media (path) VALUES ('/lib/a.jpg')`,
		`INSERT INTO media_tag_by_category VALUES ('/lib/a.jpg', 'test', 'category')`,
		`INSERT INTO media_embedding VALUES ('/lib/a.jpg', 'siglip2', 2, x'0001', 0)`,
		`INSERT INTO face (media_path, model, bbox_x, bbox_y, bbox_w, bbox_h, det_score, vector)
			VALUES ('/lib/a.jpg', 'sface', 0, 0, 1, 1, 1, x'0001')`,
		`INSERT INTO face_scan VALUES ('/lib/a.jpg', 'sface', 1, 0)`,
		`INSERT INTO person (name, cover_face_id) VALUES ('Someone', 1)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("setup %q: %v", stmt, err)
		}
	}

	// Sanity: enforcement really is on for this connection.
	var fk int
	if err := db.QueryRow("PRAGMA foreign_keys").Scan(&fk); err != nil || fk != 1 {
		t.Fatalf("foreign_keys pragma = %d (err %v), want 1", fk, err)
	}

	result, err := RemoveItemsFromDB(context.Background(), db, []string{"/lib/a.jpg"})
	if err != nil {
		t.Fatalf("RemoveItemsFromDB() error = %v", err)
	}
	if result.MediaItemsRemoved != 1 || result.TagsRemoved != 1 {
		t.Errorf("removed %d media / %d tags, want 1 / 1", result.MediaItemsRemoved, result.TagsRemoved)
	}

	for _, table := range []string{"media", "media_tag_by_category", "media_embedding", "face", "face_scan"} {
		var count int
		if err := db.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&count); err != nil {
			t.Fatalf("count %s: %v", table, err)
		}
		if count != 0 {
			t.Errorf("%s has %d rows after removal, want 0", table, count)
		}
	}
	// The person survives with its cover cleared, rather than being deleted.
	var cover sql.NullInt64
	if err := db.QueryRow("SELECT cover_face_id FROM person WHERE name = 'Someone'").Scan(&cover); err != nil {
		t.Fatalf("read person: %v", err)
	}
	if cover.Valid {
		t.Errorf("cover_face_id = %d, want NULL", cover.Int64)
	}
}

// TestRemoveItemsFromDBStreamCancelKeepsCommittedWork asserts that cancelling
// mid-run keeps every committed batch removed and reports the partial counts —
// the property that lets the remove job be paused and resumed instead of
// restarted.
func TestRemoveItemsFromDBStreamCancelKeepsCommittedWork(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	const n = 1200
	paths := make([]string, n)
	for i := range paths {
		paths[i] = fmt.Sprintf("/lib/file%04d.jpg", i)
		if _, err := db.Exec("INSERT INTO media (path) VALUES (?)", paths[i]); err != nil {
			t.Fatalf("Failed to insert test media: %v", err)
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	calls := 0
	result, err := RemoveItemsFromDBStream(ctx, db, paths, func(b RemovalBatch) {
		calls++
		cancel() // stop after the first committed batch
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context.Canceled", err)
	}
	if calls != 1 {
		t.Errorf("got %d batch callbacks after cancel, want 1", calls)
	}
	if result.MediaItemsRemoved != 500 {
		t.Errorf("MediaItemsRemoved = %d, want 500", result.MediaItemsRemoved)
	}

	var remaining int
	if err := db.QueryRow("SELECT COUNT(*) FROM media").Scan(&remaining); err != nil {
		t.Fatalf("count remaining: %v", err)
	}
	if remaining != n-500 {
		t.Errorf("%d media rows remain, want %d (committed batch must survive cancellation)", remaining, n-500)
	}
}

// TestGetNonExistentItems tests the GetNonExistentItems function
func TestGetNonExistentItems(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Create temporary directory and files
	tmpDir, err := os.MkdirTemp("", "test_non_existent_*")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create one existing file
	existingFile := filepath.Join(tmpDir, "existing.jpg")
	if err := os.WriteFile(existingFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Insert test data - mix of existing and non-existing files
	testPaths := []string{
		existingFile,
		filepath.Join(tmpDir, "missing1.jpg"),
		filepath.Join(tmpDir, "missing2.jpg"),
	}

	for _, path := range testPaths {
		_, err := db.Exec("INSERT INTO media (path, description, size) VALUES (?, ?, ?)",
			path, "Test description", 1024)
		if err != nil {
			t.Fatalf("Failed to insert test media: %v", err)
		}
	}

	// Test getting non-existent items
	ctx := context.Background()
	nonExistentPaths, err := GetNonExistentItems(ctx, db)
	if err != nil {
		t.Fatalf("GetNonExistentItems() error = %v", err)
	}

	// Should find 2 non-existent files
	if len(nonExistentPaths) != 2 {
		t.Errorf("GetNonExistentItems() found %d non-existent files, want 2", len(nonExistentPaths))
	}

	// Sort for consistent comparison
	sort.Strings(nonExistentPaths)
	expectedPaths := []string{
		filepath.Join(tmpDir, "missing1.jpg"),
		filepath.Join(tmpDir, "missing2.jpg"),
	}
	sort.Strings(expectedPaths)

	if !reflect.DeepEqual(nonExistentPaths, expectedPaths) {
		t.Errorf("GetNonExistentItems() = %v, want %v", nonExistentPaths, expectedPaths)
	}

	// Test with cancelled context
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = GetNonExistentItems(cancelledCtx, db)
	if err != context.Canceled {
		t.Errorf("GetNonExistentItems() with cancelled context should return context.Canceled, got %v", err)
	}
}

// TestStreamingCleanupNonExistentItems tests the StreamingCleanupNonExistentItems function
func TestStreamingCleanupNonExistentItems(t *testing.T) {
	db := setupTestDB(t)
	defer db.Close()

	// Create temporary directory and files
	tmpDir, err := os.MkdirTemp("", "test_streaming_cleanup_*")
	if err != nil {
		t.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	// Create one existing file
	existingFile := filepath.Join(tmpDir, "existing.jpg")
	if err := os.WriteFile(existingFile, []byte("test"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Insert test data - mix of existing and non-existing files
	testPaths := []string{
		existingFile,
		filepath.Join(tmpDir, "missing1.jpg"),
		filepath.Join(tmpDir, "missing2.jpg"),
	}

	for _, path := range testPaths {
		_, err := db.Exec("INSERT INTO media (path, description, size) VALUES (?, ?, ?)",
			path, "Test description", 1024)
		if err != nil {
			t.Fatalf("Failed to insert test media: %v", err)
		}
	}

	// Test streaming cleanup with progress callback
	var progressCalls []struct {
		found   int
		removed int
	}

	progressCallback := func(found, removed int) {
		progressCalls = append(progressCalls, struct {
			found   int
			removed int
		}{found, removed})
	}

	ctx := context.Background()
	result, err := StreamingCleanupNonExistentItems(ctx, db, progressCallback)
	if err != nil {
		t.Fatalf("StreamingCleanupNonExistentItems() error = %v", err)
	}

	if result.MediaItemsRemoved != 2 {
		t.Errorf("StreamingCleanupNonExistentItems() removed %d media items, want 2", result.MediaItemsRemoved)
	}

	// Verify progress callback was called
	if len(progressCalls) == 0 {
		t.Error("StreamingCleanupNonExistentItems() progress callback was not called")
	}

	// Verify only the existing file remains
	var remainingCount int
	err = db.QueryRow("SELECT COUNT(*) FROM media").Scan(&remainingCount)
	if err != nil {
		t.Fatalf("Failed to count remaining media items: %v", err)
	}
	if remainingCount != 1 {
		t.Errorf("Expected 1 remaining media item, got %d", remainingCount)
	}

	// Test with cancelled context
	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err = StreamingCleanupNonExistentItems(cancelledCtx, db, nil)
	if err != context.Canceled {
		t.Errorf("StreamingCleanupNonExistentItems() with cancelled context should return context.Canceled, got %v", err)
	}
}

// Benchmark tests for performance-critical functions
func BenchmarkFormatBytes(b *testing.B) {
	sizes := []int64{0, 1024, 1024 * 1024, 1024 * 1024 * 1024}

	for i := 0; i < b.N; i++ {
		FormatBytes(sizes[i%len(sizes)])
	}
}

func BenchmarkCheckFilesExistConcurrent(b *testing.B) {
	// Create test files
	tmpDir, err := os.MkdirTemp("", "bench_test_*")
	if err != nil {
		b.Fatalf("Failed to create temporary directory: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	var testPaths []string
	for i := 0; i < 100; i++ {
		if i%2 == 0 {
			// Create existing file
			path := filepath.Join(tmpDir, fmt.Sprintf("file%d.txt", i))
			os.WriteFile(path, []byte("test"), 0644)
			testPaths = append(testPaths, path)
		} else {
			// Add non-existent file
			testPaths = append(testPaths, filepath.Join(tmpDir, fmt.Sprintf("missing%d.txt", i)))
		}
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		CheckFilesExistConcurrent(testPaths)
	}
}
