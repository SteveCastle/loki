package tasks

import (
	"database/sql"
	"fmt"

	"github.com/stevecastle/shrike/media"
)

// TagInfo represents a tag with its category for the auto-tagging system
type TagInfo struct {
	Label    string
	Category string
}

// EnsureCategoryExists inserts the category if it doesn't already exist.
// The category table is expected to have columns: label, weight
// This is a wrapper around media.EnsureCategoryExists for backwards compatibility
func EnsureCategoryExists(db *sql.DB, label string, weight int) error {
	return media.EnsureCategoryExists(db, label, weight)
}

// getAllAvailableTags fetches all unique tags and their categories from the database
func getAllAvailableTags(db *sql.DB) ([]TagInfo, error) {
	query := `
		SELECT DISTINCT tag_label, category_label 
		FROM media_tag_by_category 
		ORDER BY category_label, tag_label`
	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query available tags: %w", err)
	}
	defer rows.Close()
	var tags []TagInfo
	for rows.Next() {
		var tag TagInfo
		if err := rows.Scan(&tag.Label, &tag.Category); err != nil {
			return nil, fmt.Errorf("failed to scan tag row: %w", err)
		}
		tags = append(tags, tag)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating over tag rows: %w", err)
	}
	return tags, nil
}

// getExistingTagsForFile checks if a file already has tags
func getExistingTagsForFile(db *sql.DB, filePath string) ([]TagInfo, error) {
	rows, err := db.Query(`
		SELECT tag_label, category_label 
		FROM media_tag_by_category 
		WHERE media_path = ?`, filePath)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var tags []TagInfo
	for rows.Next() {
		var tag TagInfo
		if err := rows.Scan(&tag.Label, &tag.Category); err != nil {
			return nil, err
		}
		tags = append(tags, tag)
	}
	return tags, nil
}

// removeExistingTagsForFile removes all existing tags for a file
func removeExistingTagsForFile(db *sql.DB, filePath string) error {
	_, err := db.Exec(`DELETE FROM media_tag_by_category WHERE media_path = ?`, filePath)
	return err
}

// insertTagsForFile inserts tags for a file into the database
func insertTagsForFile(db *sql.DB, filePath string, tags []TagInfo) error {
	if err := EnsureTagsExist(db, tags); err != nil {
		return err
	}
	stmt := `INSERT INTO media_tag_by_category (media_path, tag_label, category_label) VALUES (?, ?, ?)`
	for _, t := range tags {
		if _, err := db.Exec(stmt, filePath, t.Label, t.Category); err != nil {
			return fmt.Errorf("failed to insert tag %s/%s: %w", t.Category, t.Label, err)
		}
	}
	return nil
}

// EnsureTagsExist inserts any missing tags into the tag table.
// The tag table is expected to have columns: label, category_label
// This is a wrapper around media.EnsureTagsExist for backwards compatibility
func EnsureTagsExist(db *sql.DB, tags []TagInfo) error {
	if len(tags) == 0 {
		return nil
	}
	// Convert tasks.TagInfo to media.TagInfo
	mediaTags := make([]media.TagInfo, len(tags))
	for i, t := range tags {
		mediaTags[i] = media.TagInfo{
			Label:    t.Label,
			Category: t.Category,
		}
	}
	return media.EnsureTagsExist(db, mediaTags)
}
