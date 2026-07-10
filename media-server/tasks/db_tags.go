package tasks

import (
	"database/sql"
	"fmt"
	"time"

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

// hasSuggestedTags reports whether a file already carries any ONNX-suggested
// tags — the skip-existing marker for the autotag op (a re-run without
// --overwrite skips files the tagger already processed).
func hasSuggestedTags(db *sql.DB, filePath string) (bool, error) {
	var one int
	err := db.QueryRow(`SELECT 1 FROM media_tag_by_category WHERE media_path = ? AND category_label = ? LIMIT 1`,
		filePath, suggestedCategory).Scan(&one)
	if err == sql.ErrNoRows {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	return true, nil
}

// removeSuggestedTagsForFile removes only the ONNX-suggested tags for a file,
// leaving user-applied tags in other categories intact. Used by the autotag
// op's --overwrite path (e.g. re-tagging under a different tagger model).
func removeSuggestedTagsForFile(db *sql.DB, filePath string) error {
	_, err := db.Exec(`DELETE FROM media_tag_by_category WHERE media_path = ? AND category_label = ?`,
		filePath, suggestedCategory)
	if err != nil {
		return err
	}
	media.InvalidateRandomSampleCache()
	return nil
}

// insertTagsForFile inserts tags for a file into the database
func insertTagsForFile(db *sql.DB, filePath string, tags []TagInfo) error {
	if err := EnsureTagsExist(db, tags); err != nil {
		return err
	}
	// time_stamp is the in-media offset, not a wall clock; 0 means "tags the
	// media in general". Always write 0 here so auto-tagged rows match the 0
	// convention used everywhere else (AddTag, createAssignment, etc.).
	//
	// INSERT OR IGNORE: a tag the file already carries collides with the
	// (media_path, tag_label, category_label, time_stamp) primary key. Re-tagging
	// a file (e.g. running the ONNX tagger again) must not abort the whole job on
	// that collision — silently skip the pre-existing assignment and insert only
	// the genuinely new tags.
	//
	// created_at records WHEN the tag was applied (Unix seconds, matching
	// createAssignment/AddTag), so tag-driven views can date-sort by application
	// time. Previously auto-tagged rows left this NULL → they all read as time 0.
	createdAt := time.Now().Unix()
	stmt := `INSERT OR IGNORE INTO media_tag_by_category (media_path, tag_label, category_label, time_stamp, created_at) VALUES (?, ?, ?, 0, ?)`
	inserted := 0
	for _, t := range tags {
		res, err := db.Exec(stmt, filePath, t.Label, t.Category, createdAt)
		if err != nil {
			return fmt.Errorf("failed to insert tag %s/%s: %w", t.Category, t.Label, err)
		}
		// RowsAffected is 0 when the row was an ignored duplicate, so this counts
		// only tags actually added — keeping the cache invalidation below honest.
		if n, err := res.RowsAffected(); err == nil {
			inserted += int(n)
		}
	}
	// New tag rows may make a previously-untagged path eligible for the
	// swipe pool. Without this, auto-tagged paths never appear there until
	// the sampler's TTL expires.
	if inserted > 0 {
		media.InvalidateRandomSampleCache()
		// The "with tags" coverage stat counts items with ≥1 tag, so it only
		// advances when this file went from untagged to tagged. If every tag
		// the file now carries was inserted just now, this call was that
		// transition. (PK-indexed count — one cheap lookup per tagged file.)
		var total int
		if err := db.QueryRow(
			`SELECT COUNT(*) FROM media_tag_by_category WHERE media_path = ?`, filePath,
		).Scan(&total); err == nil && total == inserted {
			notifyProgress(ProgressTags, 1)
		}
	}
	return nil
}

// resolveTagCategory queries the tag table for the category of an existing tag.
// Returns "General" if the tag is not found or has no category.
func resolveTagCategory(db *sql.DB, tagLabel string) string {
	var category sql.NullString
	err := db.QueryRow(`SELECT category_label FROM tag WHERE label = ?`, tagLabel).Scan(&category)
	if err != nil || !category.Valid || category.String == "" {
		return "General"
	}
	return category.String
}

// resolveTagCategories fills in empty Category fields using the database.
// Tags with an explicit category are left unchanged.
func resolveTagCategories(db *sql.DB, tags []TagInfo) []TagInfo {
	resolved := make([]TagInfo, len(tags))
	for i, t := range tags {
		resolved[i] = t
		if resolved[i].Category == "" {
			resolved[i].Category = resolveTagCategory(db, t.Label)
		}
	}
	return resolved
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
