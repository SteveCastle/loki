package media

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/querylog"
)

// MediaItem represents a row from the media table
type MediaItem struct {
	Path           string         `json:"path"`
	Description    sql.NullString `json:"description"`
	Size           sql.NullInt64  `json:"size"`
	Hash           sql.NullString `json:"hash"`
	Width          sql.NullInt64  `json:"width"`
	Height         sql.NullInt64  `json:"height"`
	FormattedSize  string         `json:"-"`
	Tags           []MediaTag     `json:"tags"`
	Exists         bool           `json:"exists"`
	DuplicateCount int64          `json:"duplicateCount"`
}

// MediaTag represents a tag with its category
type MediaTag struct {
	Label    string `json:"label"`
	Category string `json:"category"`
}

// MarshalJSON implements custom JSON marshaling for MediaItem
func (m MediaItem) MarshalJSON() ([]byte, error) {
	type Alias MediaItem
	return json.Marshal(&struct {
		*Alias
		Description *string `json:"description"`
		Size        *int64  `json:"size"`
		Hash        *string `json:"hash"`
		Width       *int64  `json:"width"`
		Height      *int64  `json:"height"`
	}{
		Alias: (*Alias)(&m),
		Description: func() *string {
			if m.Description.Valid {
				return &m.Description.String
			} else {
				return nil
			}
		}(),
		Size: func() *int64 {
			if m.Size.Valid {
				return &m.Size.Int64
			} else {
				return nil
			}
		}(),
		Hash: func() *string {
			if m.Hash.Valid {
				return &m.Hash.String
			} else {
				return nil
			}
		}(),
		Width: func() *int64 {
			if m.Width.Valid {
				return &m.Width.Int64
			} else {
				return nil
			}
		}(),
		Height: func() *int64 {
			if m.Height.Valid {
				return &m.Height.Int64
			} else {
				return nil
			}
		}(),
	})
}

// TemplateData represents data for the media template
type TemplateData struct {
	MediaItems         []MediaItem `json:"media_items"`
	Offset             int         `json:"offset"`
	HasMore            bool        `json:"has_more"`
	TotalCount         int         `json:"total_count"`
	SearchQuery        string      `json:"search_query"`
	DefaultOllamaModel string      `json:"default_ollama_model"`
}

// APIResponse represents the JSON response for the API endpoint
type APIResponse struct {
	Items      []MediaItem `json:"items"`
	HasMore    bool        `json:"has_more"`
	TotalCount int         `json:"total_count"`
}

// formatBytes converts bytes to human readable format
func FormatBytes(bytes int64) string {
	if bytes == 0 {
		return "0 B"
	}
	const unit = 1024
	sizes := []string{"B", "KB", "MB", "GB", "TB"}
	i := 0
	b := float64(bytes)
	for b >= unit && i < len(sizes)-1 {
		b /= unit
		i++
	}
	return fmt.Sprintf("%.1f %s", b, sizes[i])
}

// Remote (s3://) existence is answered by the storage layer, wired in at
// startup via SetRemoteExistsChecker. When no checker is wired, remote paths
// report as existing: "unknown" must not render as missing in the browser or
// filter items out of the existence-aware samplers.
var (
	remoteExistsMu      sync.RWMutex
	remoteExistsChecker func(paths []string) map[string]bool
)

// SetRemoteExistsChecker installs the bulk existence checker used for
// remote (s3://) paths.
func SetRemoteExistsChecker(fn func(paths []string) map[string]bool) {
	remoteExistsMu.Lock()
	remoteExistsChecker = fn
	remoteExistsMu.Unlock()
}

// IsRemotePath reports whether the media path lives in remote storage
// rather than on a local filesystem.
func IsRemotePath(path string) bool {
	return strings.HasPrefix(path, "s3://")
}

func checkRemoteExists(paths []string) map[string]bool {
	remoteExistsMu.RLock()
	fn := remoteExistsChecker
	remoteExistsMu.RUnlock()

	out := make(map[string]bool, len(paths))
	if fn == nil {
		for _, p := range paths {
			out[p] = true
		}
		return out
	}
	res := fn(paths)
	for _, p := range paths {
		if v, ok := res[p]; ok {
			out[p] = v
		} else {
			out[p] = true
		}
	}
	return out
}

// CheckFileExists checks if a file exists at the given path
// Returns true if the file exists, false otherwise
func CheckFileExists(path string) bool {
	if IsRemotePath(path) {
		return checkRemoteExists([]string{path})[path]
	}
	_, err := os.Stat(path)
	return !os.IsNotExist(err)
}

// statConcurrency bounds the parallel os.Stat fan-out. Stats are I/O-bound so
// modest parallelism helps, but one-goroutine-per-path (the old behavior)
// stampedes network shares with a thousand concurrent SMB round-trips per
// batch and wins nothing on local disks past a few dozen workers.
const statConcurrency = 64

// CheckFilesExistConcurrent checks file existence for multiple paths concurrently
// This is more efficient than checking files sequentially when dealing with many files
func CheckFilesExistConcurrent(paths []string) map[string]bool {
	if len(paths) == 0 {
		return make(map[string]bool)
	}

	// Remote paths go to the storage layer in one bulk call (it bounds its
	// own network fan-out); only local paths hit the stat worker pool.
	var local, remote []string
	for _, p := range paths {
		if IsRemotePath(p) {
			remote = append(remote, p)
		} else {
			local = append(local, p)
		}
	}

	remoteDone := make(chan map[string]bool, 1)
	go func() { remoteDone <- checkRemoteExists(remote) }()

	// Workers write disjoint indexes, so no locking is needed.
	exists := make([]bool, len(local))
	if len(local) > 0 {
		workers := statConcurrency
		if workers > len(local) {
			workers = len(local)
		}
		indexes := make(chan int)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range indexes {
					exists[i] = CheckFileExists(local[i])
				}
			}()
		}
		for i := range local {
			indexes <- i
		}
		close(indexes)
		wg.Wait()
	}

	existenceMap := make(map[string]bool, len(paths))
	for i, p := range local {
		existenceMap[p] = exists[i]
	}
	for p, v := range <-remoteDone {
		existenceMap[p] = v
	}
	return existenceMap
}

// GetTags fetches all tags for a list of media paths
func GetTags(db *sql.DB, mediaPaths []string) (map[string][]MediaTag, error) {
	if len(mediaPaths) == 0 {
		return make(map[string][]MediaTag), nil
	}

	// Create placeholders for IN clause
	placeholders := make([]string, len(mediaPaths))
	args := make([]interface{}, len(mediaPaths))
	for i, path := range mediaPaths {
		placeholders[i] = "?"
		args[i] = path
	}

	query := fmt.Sprintf(`
		SELECT media_path, tag_label, category_label
		FROM media_tag_by_category
		WHERE media_path IN (%s)
		ORDER BY media_path, category_label, tag_label
	`, strings.Join(placeholders, ","))

	stop := querylog.Start("GetTags", query, args)
	rows, err := db.Query(query, args...)
	if err != nil {
		stop(-1, err)
		return nil, err
	}
	defer rows.Close()

	tagMap := make(map[string][]MediaTag)
	rowCount := 0
	for rows.Next() {
		var mediaPath, tagLabel, categoryLabel string
		if err := rows.Scan(&mediaPath, &tagLabel, &categoryLabel); err != nil {
			stop(rowCount, err)
			return nil, err
		}
		rowCount++

		tag := MediaTag{
			Label:    tagLabel,
			Category: categoryLabel,
		}

		tagMap[mediaPath] = append(tagMap[mediaPath], tag)
	}

	stop(rowCount, nil)
	return tagMap, nil
}

// GetRandomItems fetches media items in a randomized order with pagination and optional search
// The randomization is seeded per-session to maintain consistency during scrolling
// This function is designed for the TikTok-like swipe view
// Only items with at least one tag are included
func GetRandomItems(db *sql.DB, offset, limit int, searchQuery string, seed int64) ([]MediaItem, bool, error) {
	// Fast path: no search filter (the dominant swipe case). Use the
	// in-memory sampler — see random_sampler.go. ORDER BY RANDOM() over the
	// full tagged set scaled to ~7 seconds on a real library; the sampler
	// path is essentially constant time after a one-time cache build.
	if strings.TrimSpace(searchQuery) == "" {
		return getRandomItemsFromSampler(db, offset, limit, seed)
	}

	// Use a deterministic but shuffled ordering based on a hash of the path
	// This provides consistent pagination while appearing random
	// You can later modify this to use different algorithms (trending, recent, AI-curated, etc.)
	baseQuery := `SELECT DISTINCT m.path, m.description, m.size, m.hash, m.width, m.height FROM media m`
	// Order by a hash of the path with session seed to get pseudo-random but consistent ordering
	// Uses SQLite's built-in functions for a deterministic shuffle
	// The seed changes per page load but remains consistent during pagination within a session
	// Simplified randomization: purely random every request
	// This means items may repeat and the order is not stable across pagination pages
	// but provides the most "shuffle-like" experience
	orderBy := ` ORDER BY RANDOM()`

	var whereClause string
	var args []interface{}
	var rootNode Node

	// Parse search query if provided
	if strings.TrimSpace(searchQuery) != "" {
		parser := NewParser(searchQuery)
		var err error
		rootNode, err = parser.Parse()
		if err != nil {
			log.Printf("Search query parsing failed: %v", err)
			rootNode = nil
		}
	}

	// Fast path for pure tag union/intersection queries — the swipe filter
	// case (`tag:"X"` or `tag:"X" OR tag:"Y"`). Resolve matching paths straight
	// from media_tag_by_category by tag_label instead of scanning the whole
	// media table with a correlated EXISTS per row. Downstream shuffle/paginate
	// is shared with the generic path so behaviour is identical.
	if labels, kind := extractTagFilter(rootNode); kind != tagFilterNone {
		allPaths, err := getTaggedPathsByLabels(db, labels, kind == tagFilterAnd)
		if err != nil {
			return nil, false, err
		}
		if len(allPaths) == 0 {
			return nil, false, nil
		}
		return assembleRandomItemsFromPaths(db, allPaths, offset, limit, seed)
	}

	// Build WHERE clause
	if rootNode != nil {
		sqlPart, sqlArgs := rootNode.ToSQL()
		if sqlPart != "" {
			whereClause = "WHERE " + sqlPart
			args = sqlArgs
		}
		// If duplicates condition is used, order by hash first to cluster duplicates together
		if rootNode.HasDuplicates() {
			orderBy = ` ORDER BY m.hash, RANDOM()`
		}
	}

	// Always require items to have at least one tag
	hasTagsFilter := `EXISTS (SELECT 1 FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path)`
	if whereClause == "" {
		whereClause = "WHERE " + hasTagsFilter
	} else {
		whereClause = whereClause + " AND " + hasTagsFilter
	}

	// If there are exists conditions, we need to implement existence-aware pagination
	if rootNode != nil && rootNode.HasExists() {
		return getRandomItemsWithExistenceFilter(db, baseQuery, whereClause, args, orderBy, offset, limit, rootNode)
	}

	// Hash-clustered duplicates view stays on the SQL path: it wants items
	// grouped by hash, not shuffled into a stable random order. Everything
	// else (the dominant tag-filtered swipe case) goes through the Go-side
	// seeded shuffle below.
	if rootNode != nil && rootNode.HasDuplicates() {
		return getRandomItemsFilteredSQL(db, whereClause, args, orderBy, offset, limit)
	}

	// Fetch all matching paths in a stable order and shuffle in Go keyed by
	// the session seed. The previous implementation used `ORDER BY RANDOM()`
	// per request, which produced a fresh independent shuffle on every page
	// — so consecutive paginated calls (offset=0,limit=1 then offset=1,limit=30
	// then offset=31,limit=30…) sliced windows out of *different* permutations
	// of the same universe, and items reappeared across pages (the swipe
	// "cycles of repeating media" bug). A deterministic per-seed shuffle on
	// a fixed input order — mirroring the unfiltered sampler in
	// random_sampler.go — produces a stable session-wide permutation that the
	// client can paginate without overlap.
	pathQuery := `SELECT m.path FROM media m ` + whereClause + ` ORDER BY m.path`
	stopPaths := querylog.Start("GetRandomItems.filtered.paths", pathQuery, args)
	pathRows, err := db.Query(pathQuery, args...)
	if err != nil {
		stopPaths(-1, err)
		return nil, false, err
	}
	allPaths := make([]string, 0, 1024)
	for pathRows.Next() {
		var p string
		if err := pathRows.Scan(&p); err != nil {
			pathRows.Close()
			stopPaths(len(allPaths), err)
			return nil, false, err
		}
		allPaths = append(allPaths, p)
	}
	pathRows.Close()
	if err := pathRows.Err(); err != nil {
		stopPaths(len(allPaths), err)
		return nil, false, err
	}
	stopPaths(len(allPaths), nil)

	if len(allPaths) == 0 {
		return nil, false, nil
	}

	return assembleRandomItemsFromPaths(db, allPaths, offset, limit, seed)
}

// getTaggedPathsByLabels resolves the media paths carrying the given tag labels
// straight from media_tag_by_category, ordered by path for a stable shuffle
// input. This avoids the generic path's full media-table scan with a correlated
// EXISTS per row — the swipe filter only needs tag_label. With intersect=false
// (union) it returns paths having *any* of the labels; with intersect=true it
// returns only paths having *all* of them.
func getTaggedPathsByLabels(db *sql.DB, labels []string, intersect bool) ([]string, error) {
	// De-duplicate labels (and drop empties) so the IN-list and the
	// intersection HAVING count are correct.
	seen := make(map[string]struct{}, len(labels))
	uniq := make([]string, 0, len(labels))
	for _, l := range labels {
		if l == "" {
			continue
		}
		if _, ok := seen[l]; ok {
			continue
		}
		seen[l] = struct{}{}
		uniq = append(uniq, l)
	}
	if len(uniq) == 0 {
		return nil, nil
	}

	placeholders := strings.Repeat("?,", len(uniq))
	placeholders = placeholders[:len(placeholders)-1]
	args := make([]interface{}, 0, len(uniq)+1)
	for _, l := range uniq {
		args = append(args, l)
	}

	// EXISTS(media) is load-bearing for pagination, not just a tidiness filter:
	// media_tag_by_category can hold paths whose media row was deleted (dangling
	// tags). Such a path produces no item from the downstream wide SELECT, so
	// including it makes a page return fewer items than the shuffle window it
	// consumed. The swipe client advances its offset by items received while the
	// server advances by `limit`; the two then diverge and windows overlap,
	// re-serving the same items — the "swipe loops over the same set" bug, acute
	// for high-orphan tags. Constraining the universe to paths with a media row
	// keeps the 1:1 path→item mapping the client's offset assumes.
	var query string
	if intersect && len(uniq) > 1 {
		query = `SELECT media_path FROM media_tag_by_category WHERE tag_label IN (` +
			placeholders + `) AND EXISTS (SELECT 1 FROM media m WHERE m.path = media_path)` +
			` GROUP BY media_path HAVING COUNT(DISTINCT tag_label) = ? ORDER BY media_path`
		args = append(args, len(uniq))
	} else {
		query = `SELECT DISTINCT media_path FROM media_tag_by_category WHERE tag_label IN (` +
			placeholders + `) AND EXISTS (SELECT 1 FROM media m WHERE m.path = media_path)` +
			` ORDER BY media_path`
	}

	stop := querylog.Start("GetRandomItems.tagfast.paths", query, args)
	rows, err := db.Query(query, args...)
	if err != nil {
		stop(-1, err)
		return nil, err
	}
	defer rows.Close()

	paths := make([]string, 0, 1024)
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			stop(len(paths), err)
			return nil, err
		}
		paths = append(paths, p)
	}
	if err := rows.Err(); err != nil {
		stop(len(paths), err)
		return nil, err
	}
	stop(len(paths), nil)
	return paths, nil
}

// assembleRandomItemsFromPaths shuffles allPaths deterministically by seed,
// paginates [offset, offset+limit), fetches the full media rows for that page
// via PK point lookups, attaches tags, and checks file existence. Shared by the
// generic filtered path and the fast tag-only path so both behave identically.
func assembleRandomItemsFromPaths(db *sql.DB, allPaths []string, offset, limit int, seed int64) ([]MediaItem, bool, error) {
	if len(allPaths) == 0 {
		return nil, false, nil
	}

	// seed == 0 means "fresh randomness this session" — matches the sampler's
	// contract so the unfiltered and filtered swipe paths behave the same.
	effectiveSeed := seed
	if effectiveSeed == 0 {
		effectiveSeed = time.Now().UnixNano()
	}
	rng := rand.New(rand.NewSource(effectiveSeed))
	rng.Shuffle(len(allPaths), func(i, j int) { allPaths[i], allPaths[j] = allPaths[j], allPaths[i] })

	if offset < 0 {
		offset = 0
	}
	if offset >= len(allPaths) {
		return nil, false, nil
	}
	end := offset + limit + 1
	if end > len(allPaths) {
		end = len(allPaths)
	}
	picked := allPaths[offset:end]
	hasMore := len(picked) > limit
	if hasMore {
		picked = picked[:limit]
	}

	placeholders := strings.Repeat("?,", len(picked))
	placeholders = placeholders[:len(placeholders)-1]
	wideQuery := `SELECT m.path, m.description, m.size, m.hash, m.width, m.height FROM media m WHERE m.path IN (` + placeholders + `)`
	wideArgs := make([]interface{}, len(picked))
	for i, p := range picked {
		wideArgs[i] = p
	}

	stop := querylog.Start("GetRandomItems.filtered.wide", wideQuery, wideArgs)
	rows, err := db.Query(wideQuery, wideArgs...)
	if err != nil {
		stop(-1, err)
		return nil, false, err
	}
	defer rows.Close()

	// SQLite's IN-list returns rows in PK order, not the shuffle order. Build
	// a path→item map and re-emit in `picked` order so the client sees items
	// in the deterministic shuffle order — needed for stable offset-based
	// pagination across calls within a session.
	byPath := make(map[string]MediaItem, len(picked))
	rowCount := 0
	for rows.Next() {
		var item MediaItem
		if err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height); err != nil {
			stop(rowCount, err)
			return nil, false, err
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

	items := make([]MediaItem, 0, len(picked))
	mediaPaths := make([]string, 0, len(picked))
	for _, p := range picked {
		if it, ok := byPath[p]; ok {
			items = append(items, it)
			mediaPaths = append(mediaPaths, p)
		}
	}

	// Fetch tags for all media items
	// For performance, skip tags if only requesting 1 item (initial fast load)
	if limit > 1 || len(items) > 1 {
		tagMap, err := GetTags(db, mediaPaths)
		if err != nil {
			log.Printf("Error fetching media tags: %v", err)
		} else {
			for i := range items {
				if tags, exists := tagMap[items[i].Path]; exists {
					items[i].Tags = tags
				} else {
					items[i].Tags = []MediaTag{}
				}
			}
		}
	} else {
		// For single-item fast loads, initialize empty tags
		for i := range items {
			items[i].Tags = []MediaTag{}
		}
	}

	// Check file existence for all media items concurrently
	existenceMap := CheckFilesExistConcurrent(mediaPaths)
	for i := range items {
		if exists, found := existenceMap[items[i].Path]; found {
			items[i].Exists = exists
		} else {
			items[i].Exists = false
		}
	}

	return items, hasMore, nil
}

// getRandomItemsFilteredSQL is the legacy SQL-shuffled path. Used only when
// the search query includes `duplicates:` — that mode wants items clustered
// by hash so duplicate sets show together, which the Go-side seeded shuffle
// in GetRandomItems would scramble. Repeats across pages are an inherent
// property of `ORDER BY RANDOM()` pagination but the duplicates view doesn't
// rely on stable session-wide pagination.
//
// Performance note: ORDER BY RANDOM() forces SQLite to materialize and sort
// the entire matching set. Doing it on the full SELECT means the sort buffer
// holds wide rows for every match. The inner-query split below keeps the
// random sort against a narrow path-only projection, then fetches wide rows
// for just the chosen paths.
func getRandomItemsFilteredSQL(db *sql.DB, whereClause string, args []interface{}, orderBy string, offset, limit int) ([]MediaItem, bool, error) {
	limitClause := ` LIMIT ? OFFSET ?`
	innerQuery := `SELECT m.path FROM media m ` + whereClause + orderBy + limitClause
	innerArgs := append([]interface{}{}, args...)
	innerArgs = append(innerArgs, limit+1, offset)

	query := `SELECT m.path, m.description, m.size, m.hash, m.width, m.height FROM media m WHERE m.path IN (` + innerQuery + `)`

	stop := querylog.Start("GetRandomItems.duplicates", query, innerArgs)
	rows, err := db.Query(query, innerArgs...)
	if err != nil {
		stop(-1, err)
		return nil, false, err
	}
	defer rows.Close()

	var items []MediaItem
	var mediaPaths []string
	rowCount := 0
	for rows.Next() {
		var item MediaItem
		if err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height); err != nil {
			stop(rowCount, err)
			return nil, false, err
		}
		rowCount++
		if item.Size.Valid {
			item.FormattedSize = FormatBytes(item.Size.Int64)
		} else {
			item.FormattedSize = "Unknown"
		}
		items = append(items, item)
		mediaPaths = append(mediaPaths, item.Path)
	}
	stop(rowCount, nil)

	hasMore := len(items) > limit
	if hasMore {
		items = items[:limit]
		mediaPaths = mediaPaths[:limit]
	}

	if limit > 1 || len(items) > 1 {
		tagMap, err := GetTags(db, mediaPaths)
		if err != nil {
			log.Printf("Error fetching media tags: %v", err)
		} else {
			for i := range items {
				if tags, exists := tagMap[items[i].Path]; exists {
					items[i].Tags = tags
				} else {
					items[i].Tags = []MediaTag{}
				}
			}
		}
	} else {
		for i := range items {
			items[i].Tags = []MediaTag{}
		}
	}

	existenceMap := CheckFilesExistConcurrent(mediaPaths)
	for i := range items {
		if exists, found := existenceMap[items[i].Path]; found {
			items[i].Exists = exists
		} else {
			items[i].Exists = false
		}
	}

	return items, hasMore, nil
}

// getRandomItemsWithExistenceFilter handles randomized pagination when exists conditions are present
// Note: whereClause passed to this function should already include the "has tags" filter from GetRandomItems
func getRandomItemsWithExistenceFilter(db *sql.DB, baseQuery, whereClause string, whereArgs []interface{}, orderBy string, offset, limit int, filter Node) ([]MediaItem, bool, error) {
	const batchSize = 100
	var allMatchingItems []MediaItem

	for len(allMatchingItems) < limit {
		limitClause := ` LIMIT ?`
		var query string
		var args []interface{}

		// whereClause should always be non-empty here since GetRandomItems adds the has-tags filter
		query = baseQuery + " " + whereClause + orderBy + limitClause
		args = append(whereArgs, batchSize)

		stop := querylog.Start("getRandomItemsWithExistenceFilter", query, args)
		rows, err := db.Query(query, args...)
		if err != nil {
			stop(-1, err)
			return nil, false, err
		}

		var batchItems []MediaItem
		var batchPaths []string
		rowCount := 0
		for rows.Next() {
			var item MediaItem
			err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height)
			if err != nil {
				stop(rowCount, err)
				rows.Close()
				return nil, false, err
			}
			rowCount++

			if item.Size.Valid {
				item.FormattedSize = FormatBytes(item.Size.Int64)
			} else {
				item.FormattedSize = "Unknown"
			}

			batchItems = append(batchItems, item)
			batchPaths = append(batchPaths, item.Path)
		}
		rows.Close()
		stop(rowCount, nil)

		if len(batchItems) == 0 {
			break
		}

		tagMap, err := GetTags(db, batchPaths)
		if err != nil {
			log.Printf("Error fetching media tags: %v", err)
		} else {
			for i := range batchItems {
				if tags, exists := tagMap[batchItems[i].Path]; exists {
					batchItems[i].Tags = tags
				} else {
					batchItems[i].Tags = []MediaTag{}
				}
			}
		}

		existenceMap := CheckFilesExistConcurrent(batchPaths)
		for i := range batchItems {
			if exists, found := existenceMap[batchItems[i].Path]; found {
				batchItems[i].Exists = exists
			} else {
				batchItems[i].Exists = false
			}
		}

		for _, item := range batchItems {
			if filter.Evaluate(item) {
				allMatchingItems = append(allMatchingItems, item)
			}
		}

		// For random mode, we don't offset the DB query, but we might want to ensure we don't just return the same items
		// if the batch loop runs multiple times (though RANDOM() helps).
		// dbOffset += batchSize // Removed for pure random mode

		if len(batchItems) < batchSize {
			break
		}
	}

	// For random mode, we don't really care about offsets into the result set, we just want 'limit' items.
	// We'll take the first 'limit' items from what we found.
	totalMatching := len(allMatchingItems)
	hasMore := totalMatching >= limit // Simplified hasMore logic for random mode

	startIdx := 0
	endIdx := limit
	if endIdx > totalMatching {
		endIdx = totalMatching
	}

	var resultItems []MediaItem
	if startIdx < endIdx {
		resultItems = allMatchingItems[startIdx:endIdx]
	}

	return resultItems, hasMore, nil
}

// GetItems fetches media items from the database with pagination and search
func GetItems(db *sql.DB, offset, limit int, searchQuery string) ([]MediaItem, int, bool, error) {
	baseQuery := `SELECT DISTINCT m.path, m.description, m.size, m.hash, m.width, m.height FROM media m`
	orderBy := ` ORDER BY m.path`

	var whereClause string
	var args []interface{}
	var rootNode Node

	// Parse search query if provided
	if strings.TrimSpace(searchQuery) != "" {
		parser := NewParser(searchQuery)
		var err error
		rootNode, err = parser.Parse()
		if err != nil {
			// If parsing fails, ignore search and return all results
			log.Printf("Search query parsing failed: %v", err)
			rootNode = nil
		}
	}

	// Build WHERE clause
	if rootNode != nil {
		sqlPart, sqlArgs := rootNode.ToSQL()
		if sqlPart != "" {
			whereClause = "WHERE " + sqlPart
			args = sqlArgs
		}
		// If duplicates condition is used, order by hash first to cluster duplicates together
		if rootNode.HasDuplicates() {
			orderBy = ` ORDER BY m.hash, m.path`
		}
	}

	// If there are exists conditions, we need to implement existence-aware pagination
	if rootNode != nil && rootNode.HasExists() {
		items, hasMore, err := getItemsWithExistenceFilter(db, baseQuery, whereClause, args, orderBy, offset, limit, rootNode)
		return items, -1, hasMore, err
	}

	// Calculate total count for standard queries
	countQuery := "SELECT COUNT(*) FROM media m " + whereClause
	var totalCount int
	stopCount := querylog.Start("GetItems.count", countQuery, args)
	if err := db.QueryRow(countQuery, args...).Scan(&totalCount); err != nil {
		stopCount(-1, err)
		log.Printf("Error calculating total count: %v", err)
		totalCount = -1
	} else {
		stopCount(1, nil)
	}

	// Standard pagination for stable sorting
	limitClause := ` LIMIT ? OFFSET ?`
	var query string

	// Construct full query
	query = baseQuery + " " + whereClause + orderBy + limitClause
	// Copy args for the main query since we used them for count query
	queryArgs := append([]interface{}{}, args...)
	queryArgs = append(queryArgs, limit+1, offset)

	stop := querylog.Start("GetItems", query, queryArgs)
	rows, err := db.Query(query, queryArgs...)
	if err != nil {
		stop(-1, err)
		return nil, 0, false, err
	}
	defer rows.Close()

	var items []MediaItem
	var mediaPaths []string
	rowCount := 0
	for rows.Next() {
		var item MediaItem
		err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height)
		if err != nil {
			stop(rowCount, err)
			return nil, 0, false, err
		}
		rowCount++

		// Handle nullable size field
		if item.Size.Valid {
			item.FormattedSize = FormatBytes(item.Size.Int64)
		} else {
			item.FormattedSize = "Unknown"
		}

		items = append(items, item)
		mediaPaths = append(mediaPaths, item.Path)
	}
	stop(rowCount, nil)

	hasMore := len(items) > limit
	if hasMore {
		items = items[:limit]           // Remove the extra item
		mediaPaths = mediaPaths[:limit] // Also trim the paths list
	}

	// Fetch tags for all media items
	tagMap, err := GetTags(db, mediaPaths)
	if err != nil {
		log.Printf("Error fetching media tags: %v", err)
		// Continue without tags rather than failing completely
	} else {
		// Populate tags for each item
		for i := range items {
			if tags, exists := tagMap[items[i].Path]; exists {
				items[i].Tags = tags
			} else {
				items[i].Tags = []MediaTag{} // Empty slice instead of nil
			}
		}
	}

	// Check file existence for all media items concurrently
	existenceMap := CheckFilesExistConcurrent(mediaPaths)

	// Populate existence information for each item
	for i := range items {
		if exists, found := existenceMap[items[i].Path]; found {
			items[i].Exists = exists
		} else {
			items[i].Exists = false // Default to false if check failed
		}
	}

	return items, totalCount, hasMore, nil
}

// GetItemByPath fetches a single media item by its path
func GetItemByPath(db *sql.DB, path string) (*MediaItem, error) {
	query := `SELECT path, description, size, hash, width, height FROM media WHERE path = ?`

	var item MediaItem
	err := db.QueryRow(query, path).Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Item not found
		}
		return nil, err
	}

	// Handle nullable size field
	if item.Size.Valid {
		item.FormattedSize = FormatBytes(item.Size.Int64)
	} else {
		item.FormattedSize = "Unknown"
	}

	// Fetch tags for this item
	tagMap, err := GetTags(db, []string{item.Path})
	if err != nil {
		log.Printf("Error fetching tags for item %s: %v", item.Path, err)
		item.Tags = []MediaTag{} // Empty slice instead of nil
	} else {
		if tags, exists := tagMap[item.Path]; exists {
			item.Tags = tags
		} else {
			item.Tags = []MediaTag{} // Empty slice instead of nil
		}
	}

	// Check file existence
	item.Exists = CheckFileExists(item.Path)

	return &item, nil
}

// getItemsWithExistenceFilter handles pagination when exists conditions are present
func getItemsWithExistenceFilter(db *sql.DB, baseQuery, whereClause string, whereArgs []interface{}, orderBy string, offset, limit int, filter Node) ([]MediaItem, bool, error) {
	const batchSize = 100 // Fetch items in batches
	var allMatchingItems []MediaItem
	var dbOffset = 0

	// Keep fetching batches until we have enough matching items or run out of data
	for len(allMatchingItems) < offset+limit {
		// Construct batch query
		limitClause := ` LIMIT ? OFFSET ?`
		var query string
		var args []interface{}

		if whereClause != "" {
			query = baseQuery + " " + whereClause + orderBy + limitClause
			args = append(whereArgs, batchSize, dbOffset)
		} else {
			query = baseQuery + orderBy + limitClause
			args = []interface{}{batchSize, dbOffset}
		}

		stop := querylog.Start("getItemsWithExistenceFilter", query, args)
		rows, err := db.Query(query, args...)
		if err != nil {
			stop(-1, err)
			return nil, false, err
		}

		var batchItems []MediaItem
		var batchPaths []string
		rowCount := 0
		for rows.Next() {
			var item MediaItem
			err := rows.Scan(&item.Path, &item.Description, &item.Size, &item.Hash, &item.Width, &item.Height)
			if err != nil {
				stop(rowCount, err)
				rows.Close()
				return nil, false, err
			}
			rowCount++

			// Handle nullable size field
			if item.Size.Valid {
				item.FormattedSize = FormatBytes(item.Size.Int64)
			} else {
				item.FormattedSize = "Unknown"
			}

			batchItems = append(batchItems, item)
			batchPaths = append(batchPaths, item.Path)
		}
		rows.Close()
		stop(rowCount, nil)

		// If no more items from database, break
		if len(batchItems) == 0 {
			break
		}

		// Fetch tags for batch items
		tagMap, err := GetTags(db, batchPaths)
		if err != nil {
			log.Printf("Error fetching media tags: %v", err)
		} else {
			for i := range batchItems {
				if tags, exists := tagMap[batchItems[i].Path]; exists {
					batchItems[i].Tags = tags
				} else {
					batchItems[i].Tags = []MediaTag{}
				}
			}
		}

		// Check file existence for batch items
		existenceMap := CheckFilesExistConcurrent(batchPaths)
		for i := range batchItems {
			if exists, found := existenceMap[batchItems[i].Path]; found {
				batchItems[i].Exists = exists
			} else {
				batchItems[i].Exists = false
			}
		}

		// Filter batch items by exists conditions
		for _, item := range batchItems {
			if filter.Evaluate(item) {
				allMatchingItems = append(allMatchingItems, item)
			}
		}

		// Move to next batch
		dbOffset += batchSize

		// If we got fewer items than batch size, we've reached the end
		if len(batchItems) < batchSize {
			break
		}
	}

	// Apply pagination to filtered results
	totalMatching := len(allMatchingItems)

	// Check if there are more items beyond our current page
	hasMore := totalMatching > offset+limit

	// Extract the requested page
	startIdx := offset
	if startIdx > totalMatching {
		startIdx = totalMatching
	}

	endIdx := startIdx + limit
	if endIdx > totalMatching {
		endIdx = totalMatching
	}

	var resultItems []MediaItem
	if startIdx < endIdx {
		resultItems = allMatchingItems[startIdx:endIdx]
	}

	return resultItems, hasMore, nil
}

// RemovalResult contains the results of a database removal operation
type RemovalResult struct {
	MediaItemsRemoved int64
	TagsRemoved       int64
	ProcessedPaths    []string
	Errors            []error
	// SkippedUnavailable counts missing-on-disk items that were NOT removed
	// because their whole volume is offline (see StreamingCleanupNonExistentItems).
	SkippedUnavailable int64
	// UnavailableRoots lists the offline volume roots behind SkippedUnavailable.
	UnavailableRoots []string
}

// mediaRemovalHook is invoked with each batch of paths whose database rows
// (media + tags + embeddings) were just deleted and committed. The tasks
// package registers a hook that evicts the paths from the in-memory vector
// index — without it, similarity search keeps returning deleted items until
// the next index rebuild. Set once at init (tasks/registry.go) before any
// jobs run; nil means no-op.
var mediaRemovalHook func(paths []string)

// SetMediaRemovalHook registers the post-removal callback. Call during
// package initialization only — it is read without synchronization.
func SetMediaRemovalHook(fn func(paths []string)) {
	mediaRemovalHook = fn
}

// RemoveItemsFromDB removes media items and their associated tags from the database
// This function is designed to be reusable across different parts of the application
func RemoveItemsFromDB(ctx context.Context, db *sql.DB, paths []string) (*RemovalResult, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}

	if len(paths) == 0 {
		return &RemovalResult{}, nil
	}

	// Clean and validate paths
	var validPaths []string
	for _, path := range paths {
		cleanPath := strings.TrimSpace(path)
		if cleanPath != "" {
			validPaths = append(validPaths, cleanPath)
		}
	}

	if len(validPaths) == 0 {
		return &RemovalResult{}, nil
	}

	result := &RemovalResult{
		ProcessedPaths: validPaths,
	}

	// Process in batches to avoid SQL parameter limits (SQLite limit is typically 999)
	const batchSize = 500 // Use 500 to be safe
	totalMediaRemoved := int64(0)
	totalTagsRemoved := int64(0)

	for i := 0; i < len(validPaths); i += batchSize {
		// Check if context was cancelled
		select {
		case <-ctx.Done():
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, ctx.Err()
		default:
		}

		end := i + batchSize
		if end > len(validPaths) {
			end = len(validPaths)
		}

		batch := validPaths[i:end]

		// Start a database transaction for this batch
		tx, err := db.BeginTx(ctx, nil)
		if err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("failed to start transaction for batch: %w", err))
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, err
		}

		// Build parameterized query for this batch
		placeholders := make([]string, len(batch))
		args := make([]interface{}, len(batch))
		for j, path := range batch {
			placeholders[j] = "?"
			args[j] = path
		}

		// First, remove related records from media_tag_by_category table
		tagQuery := fmt.Sprintf(`
			DELETE FROM media_tag_by_category
			WHERE media_path IN (%s)
		`, strings.Join(placeholders, ","))

		tagResult, err := tx.ExecContext(ctx, tagQuery, args...)
		if err != nil {
			tx.Rollback()
			result.Errors = append(result.Errors, fmt.Errorf("failed to remove media tags for batch: %w", err))
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, err
		}

		batchTagsRemoved, _ := tagResult.RowsAffected()
		totalTagsRemoved += batchTagsRemoved

		// Then remove the main media records
		mediaQuery := fmt.Sprintf(`
			DELETE FROM media
			WHERE path IN (%s)
		`, strings.Join(placeholders, ","))

		mediaResult, err := tx.ExecContext(ctx, mediaQuery, args...)
		if err != nil {
			tx.Rollback()
			result.Errors = append(result.Errors, fmt.Errorf("failed to remove media items for batch: %w", err))
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, err
		}

		batchMediaRemoved, _ := mediaResult.RowsAffected()
		totalMediaRemoved += batchMediaRemoved

		// Remove embeddings for the same batch (visual-similarity sidecar table).
		if _, err := tx.ExecContext(ctx, fmt.Sprintf(`DELETE FROM media_embedding WHERE media_path IN (%s)`, strings.Join(placeholders, ",")), args...); err != nil {
			tx.Rollback()
			result.Errors = append(result.Errors, fmt.Errorf("failed to remove embeddings for batch: %w", err))
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, err
		}

		// Remove face rows + scan markers (face-identity sidecar tables). The
		// registered removal hook evicts them from the in-memory face index.
		// Person covers pointing at the doomed faces are cleared first so they
		// don't dangle (GetPeople falls back to the person's best face).
		for _, stmt := range []string{
			`UPDATE person SET cover_face_id = NULL WHERE cover_face_id IN (SELECT id FROM face WHERE media_path IN (%s))`,
			`DELETE FROM face WHERE media_path IN (%s)`,
			`DELETE FROM face_scan WHERE media_path IN (%s)`,
		} {
			if _, err := tx.ExecContext(ctx, fmt.Sprintf(stmt, strings.Join(placeholders, ",")), args...); err != nil {
				tx.Rollback()
				result.Errors = append(result.Errors, fmt.Errorf("failed to remove face rows for batch: %w", err))
				result.MediaItemsRemoved = totalMediaRemoved
				result.TagsRemoved = totalTagsRemoved
				return result, err
			}
		}

		// Commit this batch
		if err := tx.Commit(); err != nil {
			result.Errors = append(result.Errors, fmt.Errorf("failed to commit transaction for batch: %w", err))
			result.MediaItemsRemoved = totalMediaRemoved
			result.TagsRemoved = totalTagsRemoved
			return result, err
		}

		// Rows are gone from the DB; let the registered hook evict them from
		// derived in-memory state (vector index). Paths that weren't present
		// are harmless no-ops there.
		if mediaRemovalHook != nil {
			mediaRemovalHook(batch)
		}
	}

	result.MediaItemsRemoved = totalMediaRemoved
	result.TagsRemoved = totalTagsRemoved
	// Removed media may have been in the swipe pool — invalidate the cache.
	if totalMediaRemoved > 0 || totalTagsRemoved > 0 {
		InvalidateRandomSampleCache()
	}
	return result, nil
}

// GetNonExistentItems retrieves all media items from the database that don't exist in the file system
// This function processes all items in batches to avoid memory issues with large databases
func GetNonExistentItems(ctx context.Context, db *sql.DB) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}

	const batchSize = 1000 // Process items in batches to manage memory
	var nonExistentPaths []string
	offset := 0

	for {
		// Check if context was cancelled
		select {
		case <-ctx.Done():
			return nonExistentPaths, ctx.Err()
		default:
		}

		// Get a batch of media items
		query := `SELECT path FROM media ORDER BY path LIMIT ? OFFSET ?`
		rows, err := db.QueryContext(ctx, query, batchSize, offset)
		if err != nil {
			return nonExistentPaths, fmt.Errorf("failed to query media items: %w", err)
		}

		var batchPaths []string
		for rows.Next() {
			var path string
			if err := rows.Scan(&path); err != nil {
				rows.Close()
				return nonExistentPaths, fmt.Errorf("failed to scan media path: %w", err)
			}
			batchPaths = append(batchPaths, path)
		}
		rows.Close()

		// If no more items, we're done
		if len(batchPaths) == 0 {
			break
		}

		// Check file existence for this batch
		existenceMap := CheckFilesExistConcurrent(batchPaths)

		// Collect non-existent paths
		for path, exists := range existenceMap {
			if !exists {
				nonExistentPaths = append(nonExistentPaths, path)
			}
		}

		// Move to next batch
		offset += batchSize

		// If we got fewer items than batch size, we've reached the end
		if len(batchPaths) < batchSize {
			break
		}
	}

	return nonExistentPaths, nil
}

// volumeRoot returns the filesystem root that must exist for a "file is
// missing" classification to be trusted: `C:\` for drive paths, `\\host\share\`
// for UNC paths. Empty string means the path carries no volume (relative or
// unix-style) and no guard applies.
func volumeRoot(path string) string {
	vol := filepath.VolumeName(path)
	if vol == "" {
		return ""
	}
	return vol + string(filepath.Separator)
}

// StreamingCleanupNonExistentItems finds and removes non-existent media items in streaming batches
// This avoids memory issues and provides progress feedback during the operation
//
// Offline-volume guard: a path on an unmounted drive or unreachable network
// share stats exactly like a deleted file, so without a guard, running cleanup
// while one volume is offline would silently purge that volume's entire
// library (tags, descriptions, transcripts — unrecoverable). A missing file is
// therefore only treated as orphaned when its volume root still exists;
// otherwise it is counted in SkippedUnavailable and left alone. Root existence
// is checked once per volume per run.
func StreamingCleanupNonExistentItems(ctx context.Context, db *sql.DB, progressCallback func(found, removed int)) (*RemovalResult, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}

	const batchSize = 1000      // Process items in batches to manage memory
	const removeBatchSize = 500 // Remove in smaller batches to avoid SQL limits

	result := &RemovalResult{}
	totalFound := 0
	totalRemoved := 0
	rootAvailable := map[string]bool{} // volume root → exists (cached per run)

	// Use cursor-based pagination to avoid skipping items when deletions occur.
	// Using OFFSET-based pagination with deletions would skip items because
	// deleting N items shifts all subsequent items up by N positions.
	var lastPath string

	for {
		// Check if context was cancelled
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		// Get a batch of media items using cursor-based pagination
		// This ensures we don't skip items when deletions occur
		var rows *sql.Rows
		var err error
		if lastPath == "" {
			query := `SELECT path FROM media ORDER BY path LIMIT ?`
			rows, err = db.QueryContext(ctx, query, batchSize)
		} else {
			query := `SELECT path FROM media WHERE path > ? ORDER BY path LIMIT ?`
			rows, err = db.QueryContext(ctx, query, lastPath, batchSize)
		}
		if err != nil {
			return result, fmt.Errorf("failed to query media items: %w", err)
		}

		var batchPaths []string
		for rows.Next() {
			var path string
			if err := rows.Scan(&path); err != nil {
				rows.Close()
				return result, fmt.Errorf("failed to scan media path: %w", err)
			}
			batchPaths = append(batchPaths, path)
		}
		if err := rows.Err(); err != nil {
			rows.Close()
			return result, fmt.Errorf("error iterating media rows: %w", err)
		}
		rows.Close()

		// If no more items, we're done
		if len(batchPaths) == 0 {
			break
		}

		// Track the last path for cursor-based pagination
		// We need to remember the last path BEFORE any deletions
		lastPath = batchPaths[len(batchPaths)-1]

		// Check file existence for this batch
		existenceMap := CheckFilesExistConcurrent(batchPaths)

		// Collect non-existent paths from this batch, skipping any whose
		// volume is offline (see the guard note in the function comment).
		var nonExistentPaths []string
		for path, exists := range existenceMap {
			if exists {
				continue
			}
			if root := volumeRoot(path); root != "" {
				available, checked := rootAvailable[root]
				if !checked {
					_, statErr := os.Stat(root)
					available = statErr == nil
					rootAvailable[root] = available
					if !available {
						result.UnavailableRoots = append(result.UnavailableRoots, root)
					}
				}
				if !available {
					result.SkippedUnavailable++
					continue
				}
			}
			nonExistentPaths = append(nonExistentPaths, path)
		}

		totalFound += len(nonExistentPaths)

		// Remove non-existent items if any found
		if len(nonExistentPaths) > 0 {
			// Process removals in smaller batches to avoid SQL parameter limits
			for i := 0; i < len(nonExistentPaths); i += removeBatchSize {
				end := i + removeBatchSize
				if end > len(nonExistentPaths) {
					end = len(nonExistentPaths)
				}

				removeBatch := nonExistentPaths[i:end]

				// Remove this batch
				batchResult, err := RemoveItemsFromDB(ctx, db, removeBatch)
				if err != nil {
					result.Errors = append(result.Errors, err)
					return result, err
				}

				// Accumulate counts only — retaining every removed path for
				// the whole run costs unbounded memory on large cleanups and
				// no caller of the streaming variant uses the paths.
				result.MediaItemsRemoved += batchResult.MediaItemsRemoved
				result.TagsRemoved += batchResult.TagsRemoved
				result.Errors = append(result.Errors, batchResult.Errors...)

				totalRemoved += len(removeBatch)

				// Call progress callback if provided
				if progressCallback != nil {
					progressCallback(totalFound, totalRemoved)
				}
			}
		}

		// If we got fewer items than batch size, we've reached the end
		if len(batchPaths) < batchSize {
			break
		}
	}

	return result, nil
}

// -----------------------------------------------------------------------------
// Suggestions for typeahead search
// -----------------------------------------------------------------------------

// escapeLikePattern escapes special LIKE characters so that user input is treated
// as a literal substring. It escapes %, _ and the escape character itself (\).
func escapeLikePattern(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

// SuggestFilters returns supported filter keys for the search syntax
func SuggestFilters() []string {
	return []string{
		"path",
		"description",
		"size",
		"hash",
		"width",
		"height",
		"tag",
		"category",
		"tags",
		"tagcount",
		"exists",
		"pathdir",
	}
}

// SuggestTagLabels returns distinct tag labels matching a prefix (case-insensitive)
func SuggestTagLabels(db *sql.DB, prefix string, limit int) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}
	like := "%" + escapeLikePattern(strings.TrimSpace(prefix)) + "%"
	rows, err := db.Query(`
        SELECT DISTINCT tag_label
        FROM media_tag_by_category
        WHERE tag_label COLLATE NOCASE LIKE ? ESCAPE '\'
        ORDER BY tag_label
        LIMIT ?
    `, like, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// SuggestTagsWithCategories returns tags with their categories, grouped and
// ordered, capped at `limit` rows. The cap matters: a library with 100k+ tags
// matches tens of thousands of them on a short substring, and returning every
// match froze the typeahead (huge JSON payload + one DOM node per tag in the
// dropdown). `limit` is clamped to a sane range; pass the request's limit.
func SuggestTagsWithCategories(db *sql.DB, prefix string, limit int) ([]MediaTag, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}
	like := "%" + escapeLikePattern(strings.TrimSpace(prefix)) + "%"
	// Source the list from the `tag` registry (one row per tag) rather than
	// `media_tag_by_category` (one row per assignment). With many media and
	// up to ~10k tags, the assignment table holds millions of rows; doing
	// `SELECT DISTINCT` over it forced a full scan and dedupe just to
	// recover the same handful of distinct tags. The `tag` table is bounded
	// by the actual tag count and is cheap to scan even unfiltered.
	//
	// Both the Electron client and Go server insert into `tag` whenever an
	// assignment is created (taxonomy.ts uses `INSERT … ON CONFLICT DO
	// NOTHING`; media.go uses EnsureTagsExist), so this is in sync with the
	// set of tags actually in use.
	rows, err := db.Query(`
        SELECT label, COALESCE(category_label, '')
        FROM tag
        WHERE label COLLATE NOCASE LIKE ? ESCAPE '\'
        ORDER BY category_label, label
        LIMIT ?
    `, like, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []MediaTag
	for rows.Next() {
		var tag MediaTag
		if err := rows.Scan(&tag.Label, &tag.Category); err != nil {
			return nil, err
		}
		out = append(out, tag)
	}
	return out, nil
}

// SuggestCategoryLabels returns distinct category labels matching a prefix (case-insensitive)
func SuggestCategoryLabels(db *sql.DB, prefix string, limit int) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}
	like := "%" + escapeLikePattern(strings.TrimSpace(prefix)) + "%"
	rows, err := db.Query(`
        SELECT DISTINCT category_label
        FROM media_tag_by_category
        WHERE category_label COLLATE NOCASE LIKE ? ESCAPE '\'
        ORDER BY category_label
        LIMIT ?
    `, like, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []string
	for rows.Next() {
		var s string
		if err := rows.Scan(&s); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// EnsureCategoryExists inserts the category if it doesn't already exist.
// The category table is expected to have columns: label, weight
func EnsureCategoryExists(db *sql.DB, label string, weight int) error {
	label = strings.TrimSpace(label)
	if label == "" {
		return fmt.Errorf("EnsureCategoryExists: empty label")
	}
	_, err := db.Exec(`INSERT OR IGNORE INTO category (label, weight) VALUES (?, ?)`, label, weight)
	if err != nil {
		return fmt.Errorf("EnsureCategoryExists: insert %s: %w", label, err)
	}
	return nil
}

// TagInfo represents a tag with its category
type TagInfo struct {
	Label    string
	Category string
}

// EnsureTagsExist inserts any missing tags into the tag table.
// The tag table uses label as primary key with category_label, weight, preview, and thumbnail_path_600
func EnsureTagsExist(db *sql.DB, tags []TagInfo) error {
	if len(tags) == 0 {
		return nil
	}
	tx, err := db.Begin()
	if err != nil {
		return fmt.Errorf("EnsureTagsExist: begin tx: %w", err)
	}
	defer tx.Rollback()
	// Insert or update: if tag exists, update category_label; otherwise insert new tag
	insertSQL := `INSERT INTO tag (label, category_label, weight, preview, thumbnail_path_600)
	              VALUES (?, ?, NULL, NULL, NULL)
	              ON CONFLICT(label) DO UPDATE SET category_label = excluded.category_label`
	for _, t := range tags {
		if strings.TrimSpace(t.Label) == "" {
			continue
		}
		if _, err := tx.Exec(insertSQL, t.Label, t.Category); err != nil {
			return fmt.Errorf("EnsureTagsExist: insert %s/%s: %w", t.Category, t.Label, err)
		}
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("EnsureTagsExist: commit: %w", err)
	}
	return nil
}

// AddTag adds a tag with category to a media item
// Ensures the category and tag exist in their respective tables before assignment
func AddTag(db *sql.DB, mediaPath, tagLabel, categoryLabel string) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}
	if mediaPath == "" || tagLabel == "" || categoryLabel == "" {
		return fmt.Errorf("mediaPath, tagLabel, and categoryLabel are required")
	}

	// Check if tag already exists for this media
	var count int
	err := db.QueryRow(`
		SELECT COUNT(*) FROM media_tag_by_category
		WHERE media_path = ? AND tag_label = ? AND category_label = ?
	`, mediaPath, tagLabel, categoryLabel).Scan(&count)
	if err != nil {
		return fmt.Errorf("failed to check existing tag: %w", err)
	}

	// If tag already exists, no need to insert
	if count > 0 {
		return nil
	}

	// Ensure the category exists in the category table (default weight: 0)
	if err := EnsureCategoryExists(db, categoryLabel, 0); err != nil {
		return fmt.Errorf("failed to ensure category exists: %w", err)
	}

	// Ensure the tag exists in the tag table
	tagInfo := []TagInfo{{Label: tagLabel, Category: categoryLabel}}
	if err := EnsureTagsExist(db, tagInfo); err != nil {
		return fmt.Errorf("failed to ensure tag exists: %w", err)
	}

	// Insert the tag assignment with current timestamp
	now := time.Now()

	createdAt := now.Unix()

	_, err = db.Exec(`
		INSERT INTO media_tag_by_category (media_path, tag_label, category_label, time_stamp, weight,  created_at)
		VALUES (?, ?, ?, 0, 0, ?)
	`, mediaPath, tagLabel, categoryLabel, createdAt)
	if err != nil {
		return fmt.Errorf("failed to insert tag: %w", err)
	}

	// New tag may make a previously-untagged media path eligible for the
	// random swipe pool. Mark the cache stale so the next sample sees it.
	InvalidateRandomSampleCache()
	return nil
}

// RemoveTag removes a tag from a media item
func RemoveTag(db *sql.DB, mediaPath, tagLabel, categoryLabel string) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}
	if mediaPath == "" || tagLabel == "" || categoryLabel == "" {
		return fmt.Errorf("mediaPath, tagLabel, and categoryLabel are required")
	}

	_, err := db.Exec(`
		DELETE FROM media_tag_by_category
		WHERE media_path = ? AND tag_label = ? AND category_label = ?
	`, mediaPath, tagLabel, categoryLabel)
	if err != nil {
		return fmt.Errorf("failed to remove tag: %w", err)
	}

	// Removed tag may strip a path from the swipe pool entirely (if it had
	// no other tags). Mark the cache stale.
	InvalidateRandomSampleCache()
	return nil
}

// HasTag checks if a media item has a specific tag in a category
func HasTag(db *sql.DB, mediaPath, tagLabel, categoryLabel string) (bool, error) {
	if db == nil {
		return false, fmt.Errorf("database connection not available")
	}

	var count int
	err := db.QueryRow(`
		SELECT COUNT(*) FROM media_tag_by_category
		WHERE media_path = ? AND tag_label = ? AND category_label = ?
	`, mediaPath, tagLabel, categoryLabel).Scan(&count)
	if err != nil {
		return false, fmt.Errorf("failed to check tag: %w", err)
	}

	return count > 0, nil
}

// SuggestPaths returns distinct media paths matching a prefix (case-insensitive for ASCII)
// Includes both full file paths and directory paths (ending with *) if they match the query
func SuggestPaths(db *sql.DB, prefix string, limit int) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}

	// Fetch a larger sample to allow for directory extraction and filtering
	fetchLimit := limit * 5
	like := "%" + escapeLikePattern(strings.TrimSpace(prefix)) + "%"

	rows, err := db.Query(`
        SELECT DISTINCT path
        FROM media
        WHERE path LIKE ? ESCAPE '\'
        ORDER BY path
        LIMIT ?
    `, like, fetchLimit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var filePaths []string
	dirSet := make(map[string]struct{})
	lowerPrefix := strings.ToLower(strings.TrimSpace(prefix))

	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}

		// Add the file path itself
		filePaths = append(filePaths, p)

		// Extract and check directory
		d := filepath.Dir(p)
		if d == "." || d == "" {
			continue
		}

		// Normalize separator logic matches SuggestPathDirs
		sep := "/"
		if strings.Contains(d, "\\") {
			sep = "\\"
		}

		// If the directory name itself matches the search query (substring)
		if strings.Contains(strings.ToLower(d), lowerPrefix) {
			// Ensure trailing separator
			if !strings.HasSuffix(d, sep) {
				d += sep
			}
			// Append wildcard
			d += "*"
			dirSet[d] = struct{}{}
		}
	}

	// Combine results - directories first, then files
	var dirPaths []string
	for d := range dirSet {
		dirPaths = append(dirPaths, d)
	}
	sort.Strings(dirPaths)
	// filePaths are already somewhat sorted from DB, but let's ensure consistent Go sorting
	sort.Strings(filePaths)

	// Deduplicate and apply limit
	seen := make(map[string]struct{})
	var limitedResults []string

	// Add directories first
	for _, s := range dirPaths {
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		limitedResults = append(limitedResults, s)
		if len(limitedResults) >= limit {
			break
		}
	}

	// Add files if space remains
	if len(limitedResults) < limit {
		for _, s := range filePaths {
			if _, ok := seen[s]; ok {
				continue
			}
			seen[s] = struct{}{}
			limitedResults = append(limitedResults, s)
			if len(limitedResults) >= limit {
				break
			}
		}
	}

	return limitedResults, nil
}

// SuggestPathDirs returns directory suggestions based on stored paths.
// It computes directory prefixes in Go to avoid complex SQL.
func SuggestPathDirs(db *sql.DB, prefix string, limit int) ([]string, error) {
	if db == nil {
		return nil, fmt.Errorf("database connection not available")
	}
	if limit <= 0 || limit > 200 {
		limit = 25
	}
	like := "%" + escapeLikePattern(strings.TrimSpace(prefix)) + "%"
	// Fetch a larger sample of matching paths then reduce to unique directories
	sampleLimit := limit * 10
	if sampleLimit < 100 {
		sampleLimit = 100
	}
	rows, err := db.Query(`
        SELECT DISTINCT path
        FROM media
        WHERE path LIKE ? ESCAPE '\'
        ORDER BY path
        LIMIT ?
    `, like, sampleLimit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	dirSet := make(map[string]struct{})
	for rows.Next() {
		var p string
		if err := rows.Scan(&p); err != nil {
			return nil, err
		}
		// Normalize to OS-specific separators present in data. Keep both cases.
		d := filepath.Dir(p)
		if d == "." || d == "" {
			continue
		}
		// Ensure trailing separator for pathdir consistency
		if strings.Contains(d, "\\") {
			if !strings.HasSuffix(d, "\\") {
				d += "\\"
			}
		} else {
			if !strings.HasSuffix(d, "/") {
				d += "/"
			}
		}
		// Filter again by substring (case-insensitive) to match input style
		if prefix == "" || strings.Contains(strings.ToLower(d), strings.ToLower(prefix)) {
			dirSet[d] = struct{}{}
		}
	}

	// Collect up to limit results in lexical order
	var out []string
	for k := range dirSet {
		out = append(out, k)
	}
	// Simple insertion sort for small slices
	for i := 1; i < len(out); i++ {
		j := i
		for j > 0 && out[j-1] > out[j] {
			out[j-1], out[j] = out[j], out[j-1]
			j--
		}
	}
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// GetPathsByQuery retrieves all media paths matching a search query efficiently
// This only fetches paths in a single query (no tags, no file existence checks, no pagination overhead)
func GetPathsByQuery(db *sql.DB, searchQuery string) ([]string, error) {
	baseQuery := `SELECT DISTINCT m.path FROM media m`
	orderBy := ` ORDER BY m.path`

	var whereClause string
	var args []interface{}
	var rootNode Node

	// Parse search query if provided
	if strings.TrimSpace(searchQuery) != "" {
		parser := NewParser(searchQuery)
		var err error
		rootNode, err = parser.Parse()
		if err != nil {
			// Unlike GetItems (which powers the browse UI and can safely degrade
			// to showing everything), this function feeds batch tasks that ACT on
			// the result — describe, move, remove, autotag, transcode. Silently
			// selecting the whole library on a parse error is catastrophic there
			// (e.g. an unquoted multi-word tag like `tag:Exchange Student`). Fail
			// loudly so the calling job errors instead of running on everything.
			log.Printf("Search query parsing failed: %v", err)
			return nil, fmt.Errorf("invalid selection query %q: %w", searchQuery, err)
		}
	}

	// Build WHERE clause
	if rootNode != nil {
		sqlPart, sqlArgs := rootNode.ToSQL()
		if sqlPart != "" {
			whereClause = "WHERE " + sqlPart
			args = sqlArgs
		}
		// If duplicates condition is used, order by hash first to cluster duplicates together
		if rootNode.HasDuplicates() {
			orderBy = ` ORDER BY m.hash, m.path`
		}
	}

	// If there are exists conditions, we need to fall back to the slower path
	// since exists conditions require checking file system
	if rootNode != nil && rootNode.HasExists() {
		// Fall back to paginated approach for exists conditions
		return getPathsByQueryWithExistence(db, searchQuery)
	}

	// Construct full query (no pagination - get all at once)
	var query string

	if whereClause != "" {
		query = baseQuery + " " + whereClause + orderBy
	} else {
		query = baseQuery + orderBy
	}

	stop := querylog.Start("GetPathsByQuery", query, args)
	rows, err := db.Query(query, args...)
	if err != nil {
		stop(-1, err)
		return nil, err
	}
	defer rows.Close()

	var paths []string
	rowCount := 0
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			stop(rowCount, err)
			return nil, err
		}
		rowCount++
		paths = append(paths, path)
	}
	stop(rowCount, nil)

	return paths, nil
}

// getPathsByQueryWithExistence handles queries with exists conditions (requires file system checks)
func getPathsByQueryWithExistence(db *sql.DB, searchQuery string) ([]string, error) {
	const batchSize = 1000
	offset := 0
	var paths []string
	for {
		items, _, hasMore, err := GetItems(db, offset, batchSize, searchQuery)
		if err != nil {
			return nil, err
		}
		for _, it := range items {
			paths = append(paths, it.Path)
		}
		if !hasMore {
			break
		}
		offset += batchSize
	}
	return paths, nil
}

// InitializeSchema creates the media database schema if it doesn't exist
// This matches the schema from the Lowkey Media Viewer application
func InitializeSchema(db *sql.DB) error {
	if db == nil {
		return fmt.Errorf("database connection not available")
	}

	// Create category table
	_, err := db.Exec(`
		CREATE TABLE IF NOT EXISTS category (
			label TEXT PRIMARY KEY,
			weight REAL,
			description TEXT,
			tag_view_mode TEXT
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create category table: %w", err)
	}

	// Idempotent migrations for older databases that pre-date these columns.
	// Errors are ignored — they fire when the column already exists.
	_, _ = db.Exec(`ALTER TABLE category ADD COLUMN description TEXT`)
	_, _ = db.Exec(`ALTER TABLE category ADD COLUMN tag_view_mode TEXT`)

	// Create tag table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS tag (
			label TEXT PRIMARY KEY,
			category_label TEXT,
			weight REAL,
			preview BLOB,
			thumbnail_path_600 INTEGER,
			FOREIGN KEY (category_label) REFERENCES category (label)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create tag table: %w", err)
	}

	// Create media_tag_by_category table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS media_tag_by_category (
			media_path TEXT,
			tag_label TEXT,
			category_label TEXT,
			weight REAL,
			time_stamp REAL,
			created_at INTEGER,
			PRIMARY KEY(media_path, tag_label, category_label, time_stamp)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create media_tag_by_category table: %w", err)
	}

	// Create cache table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS cache (
			"key" TEXT,
			files TEXT
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create cache table: %w", err)
	}

	// Create media table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS media (
			"path" TEXT,
			description TEXT,
			transcript TEXT,
			preview BLOB,
			thumbnail_path_600 TEXT,
			thumbnail_path_1200 TEXT,
			elo REAL,
			views INTEGER,
			wins INTEGER,
			losses INTEGER,
			battles INTEGER,
			"size" INTEGER,
			hash TEXT,
			width INTEGER,
			height INTEGER,
			CONSTRAINT MEDIA_PK PRIMARY KEY ("path")
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create media table: %w", err)
	}

	// Idempotent migration for databases that pre-date the battles counter
	// (errors fire when the column already exists — same pattern as the
	// category migrations above).
	_, _ = db.Exec(`ALTER TABLE media ADD COLUMN battles INTEGER`)

	// Create battle table — append-only log of battle-mode votes. The elo
	// column on media is a derived cache; this log is the source of truth,
	// enabling recomputation, rematch suppression, and per-item match counts
	// (used for K decay). outcome is the winner_path score: 1 win, 0.5 draw.
	// Mirrored in the Electron viewer's initDB (src/main/database.ts) since
	// both processes open the same database.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS battle (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			winner_path TEXT NOT NULL,
			loser_path TEXT NOT NULL,
			outcome REAL NOT NULL DEFAULT 1,
			winner_elo_before REAL,
			loser_elo_before REAL,
			winner_elo_after REAL,
			loser_elo_after REAL,
			created_at INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create battle table: %w", err)
	}
	// Match-count lookups filter on either side of the pairing.
	if _, err := db.Exec(
		`CREATE INDEX IF NOT EXISTS idx_battle_winner ON battle(winner_path)`,
	); err != nil {
		log.Printf("warning: failed to create idx_battle_winner: %v", err)
	}
	if _, err := db.Exec(
		`CREATE INDEX IF NOT EXISTS idx_battle_loser ON battle(loser_path)`,
	); err != nil {
		log.Printf("warning: failed to create idx_battle_loser: %v", err)
	}

	// Create users table
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			created_at INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create users table: %w", err)
	}

	// Create api_keys table (long-lived credentials for lokictl / automation;
	// plaintext keys are never stored, only their SHA-256 hash)
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS api_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			name TEXT NOT NULL,
			key_hash TEXT UNIQUE NOT NULL,
			prefix TEXT NOT NULL,
			created_at INTEGER,
			last_used_at INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create api_keys table: %w", err)
	}

	// Index on tag_label for the typed-query hot paths. The composite PK on
	// (media_path, tag_label, ...) only helps queries that lead with
	// media_path; a standalone tag_label index is needed so `WHERE
	// tag_label = ?` lookups don't full-scan.
	//
	// We deliberately don't add a separate index on media_path: the PK's
	// leading column already covers those lookups, and adding a redundant
	// one contended with the Electron client and produced SQLITE_BUSY on
	// startup when both processes had the DB open.
	//
	// Errors are logged but never returned — index creation is a perf
	// optimisation, never a correctness requirement.
	if _, err := db.Exec(
		`CREATE INDEX IF NOT EXISTS idx_mtc_tag_label ON media_tag_by_category(tag_label)`,
	); err != nil {
		log.Printf("warning: failed to create idx_mtc_tag_label (will retry on next start): %v", err)
	}

	// Create media_embedding table (visual similarity search). Sidecar table,
	// model-keyed so re-embedding with a new model is non-destructive and
	// multiple models can coexist. vector is little-endian float32 (see
	// embedvec.Encode), L2-normalized.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS media_embedding (
			media_path TEXT NOT NULL,
			model      TEXT NOT NULL,
			dim        INTEGER NOT NULL,
			vector     BLOB NOT NULL,
			created_at INTEGER,
			PRIMARY KEY (media_path, model)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create media_embedding table: %w", err)
	}
	if _, err := db.Exec(
		`CREATE INDEX IF NOT EXISTS idx_media_embedding_model ON media_embedding(model)`,
	); err != nil {
		log.Printf("warning: failed to create idx_media_embedding_model: %v", err)
	}

	// Face identity tables (face detection/recognition feature). Decided up
	// front because they're hard to reverse:
	//   - bbox coordinates are RELATIVE ([0,1] of the image dimensions) so
	//     video overlays and multi-frame scanning need no migration;
	//   - frame_ts is the video timestamp the face was sampled from (0 for
	//     images and single-frame scans);
	//   - vectors are keyed by model so recognizers coexist non-destructively
	//     (shipped SFace 128-dim vs BYO ArcFace-family 512-dim);
	//   - assigned_by records whether person_id came from clustering ('auto')
	//     or the user ('user') — reclustering must never overwrite 'user'.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face (
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
	`)
	if err != nil {
		return fmt.Errorf("failed to create face table: %w", err)
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS person (
			id            INTEGER PRIMARY KEY AUTOINCREMENT,
			name          TEXT UNIQUE,
			cover_face_id INTEGER,
			created_at    INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create person table: %w", err)
	}
	// Seed the People taxonomy category so the People tab is present on every
	// DB from first boot, not only after the first person is named. Naming a
	// person creates this row lazily (ensurePersonTag), but a DB that has never
	// clustered faces would otherwise have no People category at all. Idempotent
	// (INSERT OR IGNORE) — this runs on every startup and every DB swap.
	if _, err := db.Exec(
		`INSERT OR IGNORE INTO category (label, weight) VALUES (?, 0)`, PeopleCategory,
	); err != nil {
		return fmt.Errorf("failed to seed People category: %w", err)
	}
	// face_scan marks media already scanned under a model so no-face media
	// isn't rescanned on every run (absence of face rows can't distinguish
	// "no faces" from "never scanned").
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face_scan (
			media_path TEXT NOT NULL,
			model      TEXT NOT NULL,
			face_count INTEGER NOT NULL DEFAULT 0,
			scanned_at INTEGER,
			PRIMARY KEY (media_path, model)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create face_scan table: %w", err)
	}
	// Human curation assertions for face clustering. Two complementary shapes,
	// because they survive different lifecycles:
	//   - face_veto ("this face is NEVER person P") is the direct, cheap check,
	//     but a veto against an anonymous "Unknown #N" dies with the person row
	//     when a reset dissolves it;
	//   - face_cannot_link ("these two faces are NEVER the same person",
	//     face_a < face_b normalized) is recorded against the group's exemplar
	//     faces at rejection time, so the assertion still holds when the same
	//     visual cluster re-forms under a fresh person id.
	// Both are enforced by every clustering path and only ever removed by a
	// contradicting USER action (assigning the face to that person, merging).
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face_veto (
			face_id    INTEGER NOT NULL,
			person_id  INTEGER NOT NULL,
			created_at INTEGER,
			PRIMARY KEY (face_id, person_id)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create face_veto table: %w", err)
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face_cannot_link (
			face_a     INTEGER NOT NULL,
			face_b     INTEGER NOT NULL,
			created_at INTEGER,
			PRIMARY KEY (face_a, face_b)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create face_cannot_link table: %w", err)
	}
	// Dissolved-group tombstones: when the user deletes a group (keeping its
	// faces), the membership is snapshotted so no automatic clustering pass
	// may reunite the majority of it — the same nonsense blob can't simply
	// re-form. Keyed by face ids, so a ban outlives every person id.
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face_group_ban (
			id          INTEGER PRIMARY KEY AUTOINCREMENT,
			source_name TEXT,
			created_at  INTEGER
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create face_group_ban table: %w", err)
	}
	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS face_group_ban_member (
			ban_id  INTEGER NOT NULL,
			face_id INTEGER NOT NULL,
			PRIMARY KEY (ban_id, face_id)
		)
	`)
	if err != nil {
		return fmt.Errorf("failed to create face_group_ban_member table: %w", err)
	}
	for _, stmt := range []string{
		`CREATE INDEX IF NOT EXISTS idx_face_media_path ON face(media_path)`,
		`CREATE INDEX IF NOT EXISTS idx_face_model ON face(model)`,
		`CREATE INDEX IF NOT EXISTS idx_face_person ON face(person_id)`,
		`CREATE INDEX IF NOT EXISTS idx_face_veto_person ON face_veto(person_id)`,
		`CREATE INDEX IF NOT EXISTS idx_face_cannot_link_b ON face_cannot_link(face_b)`,
		`CREATE INDEX IF NOT EXISTS idx_face_group_ban_member_face ON face_group_ban_member(face_id)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			log.Printf("warning: failed to create face index (will retry on next start): %v", err)
		}
	}

	// One-time repair for DBs written before ReplaceFaces claimed the whole
	// item per scan: drop face rows a newer scan under another model
	// superseded. These ghosts appear as permanently "ungrouped" faces in the
	// People review UI. Runs before the face index is built, so no live index
	// eviction is needed. Best-effort — a failure only delays the repair.
	if n, err := CleanupSupersededFaces(db); err != nil {
		log.Printf("warning: superseded-face cleanup failed (will retry on next start): %v", err)
	} else if n > 0 {
		log.Printf("faces: removed %d stale face row(s) superseded by newer scans under another model", n)
	}

	log.Println("Database schema initialized successfully")
	return nil
}
