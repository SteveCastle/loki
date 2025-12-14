package media

import (
	"context"
	"database/sql"
	"encoding/json"
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

// TestParseSearchQuery tests the parseSearchQuery function
func TestParseSearchQuery(t *testing.T) {
	tests := []struct {
		name        string
		query       string
		expected    *SearchQuery
		expectError bool
	}{
		{
			name:        "Empty query",
			query:       "",
			expected:    nil,
			expectError: false,
		},
		{
			name:        "Whitespace only",
			query:       "   ",
			expected:    nil,
			expectError: false,
		},
		{
			name:  "Simple path search",
			query: `path:"video.mp4"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "path", Operator: "=", Value: "video.mp4", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Path with wildcard",
			query: `path:"*.jpg"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "path", Operator: "LIKE", Value: "%.jpg", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Size greater than",
			query: `size:>1000000`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "size", Operator: ">", Value: "1000000", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Size less than",
			query: `size:<500000`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "size", Operator: "<", Value: "500000", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Size range",
			query: `size:>1000000 AND size:<10000000`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "size", Operator: ">", Value: "1000000", Negate: false},
					{Column: "size", Operator: "<", Value: "10000000", Negate: false},
				},
				Logic: []string{"AND"},
			},
		},
		{
			name:  "NOT condition",
			query: `NOT tag:"portrait"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "tag", Operator: "=", Value: "portrait", Negate: true},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Complex query with AND/OR",
			query: `path:"*.jpg" AND tag:"landscape" OR category:"nature"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "path", Operator: "LIKE", Value: "%.jpg", Negate: false},
					{Column: "tag", Operator: "=", Value: "landscape", Negate: false},
					{Column: "category", Operator: "=", Value: "nature", Negate: false},
				},
				Logic: []string{"AND", "OR"},
			},
		},
		{
			name:  "Unix pathdir",
			query: `pathdir:"/home/user/photos/"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "pathdir", Operator: "PATHDIR", Value: "UNIX:/home/user/photos/", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Windows pathdir",
			query: `pathdir:"C:\Users\User\Photos\"`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "pathdir", Operator: "PATHDIR", Value: "WIN:C:\\Users\\User\\Photos\\", Negate: false},
				},
				Logic: []string{},
			},
		},
		{
			name:  "Exists condition",
			query: `exists:true`,
			expected: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "exists", Operator: "=", Value: "true", Negate: false},
				},
				Logic: []string{},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseSearchQuery(tt.query)

			if tt.expectError {
				if err == nil {
					t.Errorf("parseSearchQuery(%s) expected error but got none", tt.query)
				}
				return
			}

			if err != nil {
				t.Errorf("parseSearchQuery(%s) unexpected error: %v", tt.query, err)
				return
			}

			if !compareSearchQuery(result, tt.expected) {
				t.Errorf("parseSearchQuery(%s) = %+v, want %+v", tt.query, result, tt.expected)
			}
		})
	}
}

// TestBuildWhereClause tests the buildWhereClause function
func TestBuildWhereClause(t *testing.T) {
	tests := []struct {
		name              string
		searchQuery       *SearchQuery
		expectedWhere     string
		expectedArgsCount int
		expectedTagJoin   bool
		expectedExistsLen int
	}{
		{
			name:              "Nil query",
			searchQuery:       nil,
			expectedWhere:     "",
			expectedArgsCount: 0,
			expectedTagJoin:   false,
			expectedExistsLen: 0,
		},
		{
			name: "Simple path condition",
			searchQuery: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "path", Operator: "=", Value: "test.jpg", Negate: false},
				},
			},
			expectedWhere:     "WHERE m.path = ?",
			expectedArgsCount: 1,
			expectedTagJoin:   false,
			expectedExistsLen: 0,
		},
		{
			name: "Tag condition requiring join",
			searchQuery: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "tag", Operator: "=", Value: "landscape", Negate: false},
				},
			},
			expectedWhere:     "WHERE mtbc.tag_label = ?",
			expectedArgsCount: 1,
			expectedTagJoin:   true,
			expectedExistsLen: 0,
		},
		{
			name: "Exists condition",
			searchQuery: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "exists", Operator: "=", Value: "true", Negate: false},
				},
			},
			expectedWhere:     "",
			expectedArgsCount: 0,
			expectedTagJoin:   false,
			expectedExistsLen: 1,
		},
		{
			name: "Mixed conditions",
			searchQuery: &SearchQuery{
				Conditions: []SearchCondition{
					{Column: "path", Operator: "LIKE", Value: "%.jpg", Negate: false},
					{Column: "size", Operator: ">", Value: "1000000", Negate: false},
				},
				Logic: []string{"AND"},
			},
			expectedWhere:     "WHERE m.path LIKE ? AND m.size > ?",
			expectedArgsCount: 2,
			expectedTagJoin:   false,
			expectedExistsLen: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			whereClause, args, needsTagJoin, existsConditions := buildWhereClause(tt.searchQuery)

			if whereClause != tt.expectedWhere {
				t.Errorf("buildWhereClause() whereClause = %q, want %q", whereClause, tt.expectedWhere)
			}

			if len(args) != tt.expectedArgsCount {
				t.Errorf("buildWhereClause() args count = %d, want %d", len(args), tt.expectedArgsCount)
			}

			if needsTagJoin != tt.expectedTagJoin {
				t.Errorf("buildWhereClause() needsTagJoin = %v, want %v", needsTagJoin, tt.expectedTagJoin)
			}

			if len(existsConditions) != tt.expectedExistsLen {
				t.Errorf("buildWhereClause() existsConditions count = %d, want %d", len(existsConditions), tt.expectedExistsLen)
			}
		})
	}
}

// TestEvaluateExistsConditions tests the evaluateExistsConditions function
func TestEvaluateExistsConditions(t *testing.T) {
	tests := []struct {
		name       string
		item       MediaItem
		conditions []SearchCondition
		expected   bool
	}{
		{
			name:       "No conditions",
			item:       MediaItem{Exists: true},
			conditions: []SearchCondition{},
			expected:   true,
		},
		{
			name: "Exists true condition matches",
			item: MediaItem{Exists: true},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "true", Negate: false},
			},
			expected: true,
		},
		{
			name: "Exists true condition doesn't match",
			item: MediaItem{Exists: false},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "true", Negate: false},
			},
			expected: false,
		},
		{
			name: "Exists false condition matches",
			item: MediaItem{Exists: false},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "false", Negate: false},
			},
			expected: true,
		},
		{
			name: "NOT exists condition",
			item: MediaItem{Exists: true},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "false", Negate: true},
			},
			expected: true,
		},
		{
			name: "Multiple conditions - all match",
			item: MediaItem{Exists: true},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "true", Negate: false},
				{Column: "exists", Operator: "=", Value: "false", Negate: true},
			},
			expected: true,
		},
		{
			name: "Multiple conditions - one doesn't match",
			item: MediaItem{Exists: false},
			conditions: []SearchCondition{
				{Column: "exists", Operator: "=", Value: "true", Negate: false},
				{Column: "exists", Operator: "=", Value: "false", Negate: true},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := evaluateExistsConditions(tt.item, tt.conditions)
			if result != tt.expected {
				t.Errorf("evaluateExistsConditions() = %v, want %v", result, tt.expected)
			}
		})
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

// compareSearchQuery is a helper function to compare SearchQuery structs
func compareSearchQuery(actual, expected *SearchQuery) bool {
	// Both nil
	if actual == nil && expected == nil {
		return true
	}

	// One nil, one not
	if actual == nil || expected == nil {
		return false
	}

	// Compare conditions
	if len(actual.Conditions) != len(expected.Conditions) {
		return false
	}

	for i, actualCondition := range actual.Conditions {
		expectedCondition := expected.Conditions[i]
		if actualCondition.Column != expectedCondition.Column ||
			actualCondition.Operator != expectedCondition.Operator ||
			actualCondition.Value != expectedCondition.Value ||
			actualCondition.Negate != expectedCondition.Negate {
			return false
		}
	}

	// Compare logic
	if len(actual.Logic) != len(expected.Logic) {
		return false
	}

	for i, actualLogic := range actual.Logic {
		if actualLogic != expected.Logic[i] {
			return false
		}
	}

	return true
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

func BenchmarkParseSearchQuery(b *testing.B) {
	queries := []string{
		`path:"*.jpg"`,
		`size:>1000000 AND size:<10000000`,
		`tag:"landscape" OR category:"nature"`,
		`NOT exists:false AND path:"*.png"`,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		parseSearchQuery(queries[i%len(queries)])
	}
}
