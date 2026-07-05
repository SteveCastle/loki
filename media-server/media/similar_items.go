package media

import (
	"database/sql"
	"strings"

	"github.com/stevecastle/shrike/querylog"
)

// filterChunkSize bounds the IN-list size per query. The similar-mode
// candidate window grows with scroll depth, so unlike the swipe sampler's
// fixed limit≤50 lookups this list can reach thousands of paths.
const filterChunkSize = 500

// FilterExistingMediaPaths returns the subset of paths that have a row in the
// media table, preserving the input order. Embeddings can outlive their media
// row (deleted/moved files), and letting such orphans into an offset-paginated
// ranking desyncs the client's offset from the server's — the same failure
// mode as the swipe tag-sampler orphan bug — so ranked candidates must be
// filtered before slicing a page out of them.
func FilterExistingMediaPaths(db *sql.DB, paths []string) ([]string, error) {
	if len(paths) == 0 {
		return nil, nil
	}

	exists := make(map[string]struct{}, len(paths))
	for start := 0; start < len(paths); start += filterChunkSize {
		end := start + filterChunkSize
		if end > len(paths) {
			end = len(paths)
		}
		chunk := paths[start:end]

		placeholders := strings.Repeat("?,", len(chunk))
		placeholders = placeholders[:len(placeholders)-1]
		query := `SELECT path FROM media WHERE path IN (` + placeholders + `)`
		args := make([]interface{}, len(chunk))
		for i, p := range chunk {
			args[i] = p
		}

		stop := querylog.Start("FilterExistingMediaPaths", query, args)
		rows, err := db.Query(query, args...)
		if err != nil {
			stop(-1, err)
			return nil, err
		}
		rowCount := 0
		for rows.Next() {
			var p string
			if err := rows.Scan(&p); err != nil {
				rows.Close()
				stop(rowCount, err)
				return nil, err
			}
			rowCount++
			exists[p] = struct{}{}
		}
		rows.Close()
		stop(rowCount, nil)
	}

	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if _, ok := exists[p]; ok {
			out = append(out, p)
		}
	}
	return out, nil
}

// GetItemsByPaths returns full MediaItems (metadata, tags, on-disk existence)
// for the given paths, preserving the input order. Paths without a media row
// are skipped. Callers that page over a ranked path list should filter with
// FilterExistingMediaPaths BEFORE slicing the page so skipped rows can't
// shift pagination offsets.
func GetItemsByPaths(db *sql.DB, paths []string) ([]MediaItem, error) {
	if len(paths) == 0 {
		return []MediaItem{}, nil
	}

	placeholders := strings.Repeat("?,", len(paths))
	placeholders = placeholders[:len(placeholders)-1]
	query := `SELECT m.path, m.description, m.size, m.hash, m.width, m.height FROM media m WHERE m.path IN (` + placeholders + `)`
	args := make([]interface{}, len(paths))
	for i, p := range paths {
		args[i] = p
	}

	stop := querylog.Start("GetItemsByPaths", query, args)
	rows, err := db.Query(query, args...)
	if err != nil {
		stop(-1, err)
		return nil, err
	}
	defer rows.Close()

	// SQLite's IN-list returns rows in PK order; re-emit in input order.
	byPath := make(map[string]MediaItem, len(paths))
	rowCount := 0
	for rows.Next() {
		var item MediaItem
		if err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height); err != nil {
			stop(rowCount, err)
			return nil, err
		}
		rowCount++
		if item.Size.Valid {
			item.FormattedSize = FormatBytes(item.Size.Int64)
		} else {
			item.FormattedSize = "Unknown"
		}
		byPath[item.Path] = item
	}
	stop(rowCount, nil)

	items := make([]MediaItem, 0, len(paths))
	mediaPaths := make([]string, 0, len(paths))
	for _, p := range paths {
		if it, ok := byPath[p]; ok {
			items = append(items, it)
			mediaPaths = append(mediaPaths, p)
		}
	}

	tagMap, err := GetTags(db, mediaPaths)
	if err != nil {
		return nil, err
	}
	for i := range items {
		if tags, ok := tagMap[items[i].Path]; ok {
			items[i].Tags = tags
		} else {
			items[i].Tags = []MediaTag{}
		}
	}

	existenceMap := CheckFilesExistConcurrent(mediaPaths)
	for i := range items {
		items[i].Exists = existenceMap[items[i].Path]
	}

	return items, nil
}
