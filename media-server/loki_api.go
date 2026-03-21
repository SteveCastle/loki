package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func formatFileSize(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d Bytes", bytes)
	}
}

// ---- Request/Response types ----

type mediaRequest struct {
	Tags []string `json:"tags"`
	Mode string   `json:"mode"`
}

type searchRequest struct {
	Description   string   `json:"description"`
	Tags          []string `json:"tags"`
	FilteringMode string   `json:"filteringMode"`
}

type pathRequest struct {
	Path string `json:"path"`
}

type mediaPreviewRequest struct {
	Path      string `json:"path"`
	Cache     any    `json:"cache"`
	TimeStamp int    `json:"timeStamp"`
}

type gifMetadataResponse struct {
	FrameCount int     `json:"frameCount"`
	Duration   float64 `json:"duration"`
}

type mediaMetadataResponse struct {
	Width    int     `json:"width"`
	Height   int     `json:"height"`
	Duration float64 `json:"duration,omitempty"`
}

type descriptionRequest struct {
	Path        string `json:"path"`
	Description string `json:"description"`
}

// ---- Helper ----

func writeJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func readJSON(r *http.Request, v any) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func httpError(w http.ResponseWriter, msg string, code int) {
	http.Error(w, fmt.Sprintf(`{"error":%q}`, msg), code)
}

// ---- Media handlers ----

// scanMediaItem scans a row from a media query with tag join into a map
// matching the Electron Item type: { path, weight, mtimeMs, timeStamp, elo, tagLabel, height, width }
func scanMediaItem(rows *sql.Rows) (map[string]any, error) {
	var path string
	var tagLabel, categoryLabel sql.NullString
	var weight, timeStamp sql.NullFloat64
	var createdAt sql.NullInt64
	var height, width sql.NullInt64
	var elo sql.NullFloat64
	err := rows.Scan(&path, &tagLabel, &categoryLabel, &weight, &timeStamp, &createdAt, &height, &width, &elo)
	if err != nil {
		return nil, err
	}
	item := map[string]any{
		"path":    path,
		"mtimeMs": createdAt.Int64,
	}
	if tagLabel.Valid {
		item["tagLabel"] = tagLabel.String
	}
	if weight.Valid {
		item["weight"] = weight.Float64
	}
	if timeStamp.Valid {
		item["timeStamp"] = timeStamp.Float64
	}
	if height.Valid {
		item["height"] = height.Int64
	}
	if width.Valid {
		item["width"] = width.Int64
	}
	if elo.Valid {
		item["elo"] = elo.Float64
	}
	return item, nil
}

func lokiMediaHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req mediaRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		if len(req.Tags) == 0 {
			// No tags: return all media with full item fields
			rows, err := deps.DB.Query(`
				SELECT m.path, '' AS tag_label, '' AS category_label,
					0 AS weight, 0 AS time_stamp, 0 AS created_at,
					m.height, m.width, m.elo
				FROM media m ORDER BY m.path`)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer rows.Close()

			var items []map[string]any
			for rows.Next() {
				item, err := scanMediaItem(rows)
				if err != nil {
					continue
				}
				items = append(items, item)
			}
			if items == nil {
				items = []map[string]any{}
			}
			writeJSON(w, items)
			return
		}

		// With tags: filter by tag assignments, return full item fields
		mode := strings.ToUpper(req.Mode)
		if mode == "" {
			mode = "EXCLUSIVE"
		}

		placeholders := make([]string, len(req.Tags))
		args := make([]any, len(req.Tags))
		for i, t := range req.Tags {
			placeholders[i] = "?"
			args[i] = t
		}
		ph := strings.Join(placeholders, ",")

		var query string
		if mode == "EXCLUSIVE" {
			query = fmt.Sprintf(`
				SELECT mtbc.media_path, mtbc.tag_label, mtbc.category_label,
					mtbc.weight, mtbc.time_stamp, mtbc.created_at,
					m.height, m.width, m.elo
				FROM media_tag_by_category mtbc
				LEFT JOIN media m ON m.path = mtbc.media_path
				WHERE mtbc.tag_label IN (%s)
				GROUP BY mtbc.media_path
				HAVING COUNT(DISTINCT mtbc.tag_label) = ?
				ORDER BY mtbc.media_path`, ph)
			args = append(args, len(req.Tags))
		} else {
			query = fmt.Sprintf(`
				SELECT mtbc.media_path, mtbc.tag_label, mtbc.category_label,
					mtbc.weight, mtbc.time_stamp, mtbc.created_at,
					m.height, m.width, m.elo
				FROM media_tag_by_category mtbc
				LEFT JOIN media m ON m.path = mtbc.media_path
				WHERE mtbc.tag_label IN (%s)
				ORDER BY mtbc.media_path`, ph)
		}

		rows, err := deps.DB.Query(query, args...)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var items []map[string]any
		for rows.Next() {
			item, err := scanMediaItem(rows)
			if err != nil {
				continue
			}
			items = append(items, item)
		}
		if items == nil {
			items = []map[string]any{}
		}
		writeJSON(w, items)
	}
}

func lokiMediaSearchHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req searchRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		query := `SELECT m.path, '' AS tag_label, '' AS category_label,
			0 AS weight, 0 AS time_stamp, 0 AS created_at,
			m.height, m.width, m.elo
			FROM media m WHERE m.description LIKE ? ORDER BY m.path`
		rows, err := deps.DB.Query(query, "%"+req.Description+"%")
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var items []map[string]any
		for rows.Next() {
			item, err := scanMediaItem(rows)
			if err != nil {
				continue
			}
			items = append(items, item)
		}
		if items == nil {
			items = []map[string]any{}
		}
		writeJSON(w, items)
	}
}

func lokiMediaMetadataHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		var width, height, size sql.NullInt64
		var description, transcript, hash sql.NullString
		deps.DB.QueryRow(
			"SELECT width, height, size, description, transcript, hash FROM media WHERE path = ?",
			req.Path,
		).Scan(&width, &height, &size, &description, &transcript, &hash)

		fileMetadata := map[string]any{
			"size":     "0 Bytes",
			"modified": "",
			"width":    0,
			"height":   0,
		}
		if width.Valid {
			fileMetadata["width"] = int(width.Int64)
		}
		if height.Valid {
			fileMetadata["height"] = int(height.Int64)
		}
		if size.Valid {
			fileMetadata["size"] = formatFileSize(size.Int64)
		}

		resp := map[string]any{
			"fileMetadata": fileMetadata,
			"hash":         "",
		}
		if description.Valid {
			resp["description"] = description.String
		}
		if transcript.Valid {
			resp["transcript"] = transcript.String
		}
		if hash.Valid {
			resp["hash"] = hash.String
		}
		writeJSON(w, resp)
	}
}

func lokiMediaTagsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		rows, err := deps.DB.Query(`
			SELECT tag_label, category_label, weight, time_stamp
			FROM media_tag_by_category
			WHERE media_path = ?`, req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var tags []map[string]any
		for rows.Next() {
			var tagLabel, catLabel string
			var weight, timeStamp float64
			rows.Scan(&tagLabel, &catLabel, &weight, &timeStamp)
			tags = append(tags, map[string]any{
				"tag_label":      tagLabel,
				"category_label": catLabel,
				"weight":         weight,
				"time_stamp":     timeStamp,
			})
		}
		if tags == nil {
			tags = []map[string]any{}
		}
		writeJSON(w, map[string]any{"tags": tags})
	}
}

func lokiUpdateDescriptionHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req descriptionRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		_, err := deps.DB.Exec("UPDATE media SET description = ? WHERE path = ?", req.Description, req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{})
	}
}

func lokiMediaPreviewHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req mediaPreviewRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		// Return the thumbnail path if it exists in DB
		var thumbPath sql.NullString
		cacheStr, ok := req.Cache.(string)
		if !ok || cacheStr == "" {
			// cache is false or not a string — no thumbnail to look up
			writeJSON(w, nil)
			return
		}
		cache := cacheStr
		// Whitelist cache column names to prevent SQL injection
		switch cache {
		case "thumbnail_path_100", "thumbnail_path_600", "thumbnail_path_1200":
			// valid
		default:
			cache = "thumbnail_path_600"
		}
		deps.DB.QueryRow(
			fmt.Sprintf("SELECT %s FROM media WHERE path = ?", cache),
			req.Path,
		).Scan(&thumbPath)

		if thumbPath.Valid && thumbPath.String != "" {
			// Check if the thumbnail file actually exists on disk
			if _, err := os.Stat(thumbPath.String); err == nil {
				writeJSON(w, thumbPath.String)
				return
			}
		}

		// No thumbnail in DB or file missing — generate one asynchronously
		dbPath := currentConfig.DBPath
		if dbPath == "" {
			writeJSON(w, nil)
			return
		}
		basePath := filepath.Dir(dbPath)
		go func() {
			generated, err := generateThumbnail(req.Path, basePath, cache, req.TimeStamp)
			if err != nil {
				log.Printf("Thumbnail generation failed for %s: %v", req.Path, err)
				return
			}
			// Store the generated path in the DB
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				generated, req.Path,
			)
			log.Printf("Generated thumbnail for %s → %s", req.Path, generated)
		}()
		writeJSON(w, nil)
	}
}

func lokiMediaDeleteHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		// Delete from database
		deps.DB.Exec("DELETE FROM media_tag_by_category WHERE media_path = ?", req.Path)
		deps.DB.Exec("DELETE FROM media WHERE path = ?", req.Path)
		writeJSON(w, map[string]string{})
	}
}

func lokiGifMetadataHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		// Use ffprobe to get gif metadata
		cmd := exec.Command("ffprobe", "-v", "error", "-count_frames",
			"-select_streams", "v:0", "-show_entries",
			"stream=nb_read_frames,duration", "-of", "json", req.Path)
		out, err := cmd.Output()
		if err != nil {
			writeJSON(w, nil)
			return
		}

		var probe struct {
			Streams []struct {
				NbReadFrames string `json:"nb_read_frames"`
				Duration     string `json:"duration"`
			} `json:"streams"`
		}
		if err := json.Unmarshal(out, &probe); err != nil || len(probe.Streams) == 0 {
			writeJSON(w, nil)
			return
		}

		frames, _ := strconv.Atoi(probe.Streams[0].NbReadFrames)
		dur, _ := strconv.ParseFloat(probe.Streams[0].Duration, 64)
		writeJSON(w, gifMetadataResponse{FrameCount: frames, Duration: dur})
	}
}

// ---- Taxonomy types ----

type tagRequest struct {
	Label         string  `json:"label"`
	CategoryLabel string  `json:"categoryLabel"`
	Weight        float64 `json:"weight"`
}

type renameRequest struct {
	Label    string `json:"label"`
	NewLabel string `json:"newLabel"`
}

type moveTagRequest struct {
	Label         string `json:"label"`
	CategoryLabel string `json:"categoryLabel"`
}

type orderRequest struct {
	Labels []string `json:"labels"`
}

type weightRequest struct {
	Label  string  `json:"label"`
	Weight float64 `json:"weight"`
}

type updateTimestampRequest struct {
	MediaPath    string  `json:"mediaPath"`
	TagLabel     string  `json:"tagLabel"`
	OldTimestamp float64 `json:"oldTimestamp"`
	NewTimestamp float64 `json:"newTimestamp"`
}

type removeTimestampRequest struct {
	MediaPath string  `json:"mediaPath"`
	TagLabel  string  `json:"tagLabel"`
	Timestamp float64 `json:"timestamp"`
}

type assignmentRequest struct {
	MediaPaths      []string `json:"mediaPaths"`
	MediaPath       string   `json:"mediaPath"`
	TagLabel        string   `json:"tagLabel"`
	CategoryLabel   string   `json:"categoryLabel"`
	TimeStamp       float64  `json:"timeStamp"`
	ApplyTagPreview bool     `json:"applyTagPreview"`
}

type deleteAssignmentRequest struct {
	MediaPath string `json:"mediaPath"`
	Tag       struct {
		TagLabel  string  `json:"tag_label"`
		TimeStamp float64 `json:"time_stamp"`
	} `json:"tag"`
}

type assignmentWeightRequest struct {
	MediaPath      string  `json:"mediaPath"`
	TagLabel       string  `json:"tagLabel"`
	Weight         float64 `json:"weight"`
	MediaTimeStamp float64 `json:"mediaTimeStamp"`
}

type labelRequest struct {
	Label string `json:"label"`
}

// ---- Taxonomy handlers ----

func lokiTaxonomyHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Must return the same shape as the Electron app:
		// { "categoryLabel": { label: string, weight: number, tags: [{label, category, weight}] } }
		rows, err := deps.DB.Query(`
			SELECT
				c.label AS category_label,
				c.weight AS category_weight,
				COALESCE(t.label, '') AS tag_label,
				COALESCE(t.category_label, '') AS tag_category,
				COALESCE(t.weight, 0) AS tag_weight
			FROM category c
			LEFT JOIN tag t ON c.label = t.category_label
			ORDER BY c.weight, t.weight`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		type tagInfo struct {
			Label    string  `json:"label"`
			Category string  `json:"category"`
			Weight   float64 `json:"weight"`
		}
		type categoryInfo struct {
			Label  string    `json:"label"`
			Weight float64   `json:"weight"`
			Tags   []tagInfo `json:"tags"`
		}

		catMap := make(map[string]*categoryInfo)
		catOrder := []string{}

		for rows.Next() {
			var catLabel, tagLabel, tagCategory string
			var catWeight, tagWeight float64
			rows.Scan(&catLabel, &catWeight, &tagLabel, &tagCategory, &tagWeight)

			cat, exists := catMap[catLabel]
			if !exists {
				cat = &categoryInfo{Label: catLabel, Weight: catWeight, Tags: []tagInfo{}}
				catMap[catLabel] = cat
				catOrder = append(catOrder, catLabel)
			}
			if tagLabel != "" {
				cat.Tags = append(cat.Tags, tagInfo{
					Label: tagLabel, Category: tagCategory, Weight: tagWeight,
				})
			}
		}

		result := make(map[string]any)
		for _, label := range catOrder {
			result[label] = catMap[label]
		}
		writeJSON(w, result)
	}
}

func lokiCreateTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req tagRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		_, err := deps.DB.Exec(
			"INSERT INTO tag (label, category_label, weight) VALUES (?, ?, ?)",
			req.Label, req.CategoryLabel, req.Weight)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{"label": req.Label})
	}
}

func lokiRenameTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req renameRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		tx.Exec("UPDATE tag SET label = ? WHERE label = ?", req.NewLabel, req.Label)
		tx.Exec("UPDATE media_tag_by_category SET tag_label = ? WHERE tag_label = ?", req.NewLabel, req.Label)
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

func lokiMoveTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req moveTagRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		deps.DB.Exec("UPDATE tag SET category_label = ? WHERE label = ?", req.CategoryLabel, req.Label)
		writeJSON(w, map[string]string{})
	}
}

func lokiDeleteTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req labelRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		tx.Exec("DELETE FROM media_tag_by_category WHERE tag_label = ?", req.Label)
		tx.Exec("DELETE FROM tag WHERE label = ?", req.Label)
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

func lokiOrderTagsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req orderRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for i, label := range req.Labels {
			tx.Exec("UPDATE tag SET weight = ? WHERE label = ?", float64(i), label)
		}
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

func lokiUpdateTagWeightHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req weightRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		deps.DB.Exec("UPDATE tag SET weight = ? WHERE label = ?", req.Weight, req.Label)
		writeJSON(w, map[string]string{})
	}
}

func lokiUpdateTimestampHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req updateTimestampRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		deps.DB.Exec(`UPDATE media_tag_by_category SET time_stamp = ?
			WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
			req.NewTimestamp, req.MediaPath, req.TagLabel, req.OldTimestamp)
		writeJSON(w, map[string]string{})
	}
}

func lokiRemoveTimestampHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req removeTimestampRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		// Check if there's already a tag without timestamp
		var count int
		deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category
			WHERE media_path = ? AND tag_label = ? AND time_stamp = 0`,
			req.MediaPath, req.TagLabel).Scan(&count)

		if count > 0 {
			// Already has non-timestamped entry, just delete the timestamped one
			deps.DB.Exec(`DELETE FROM media_tag_by_category
				WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
				req.MediaPath, req.TagLabel, req.Timestamp)
		} else {
			// Set timestamp to 0
			deps.DB.Exec(`UPDATE media_tag_by_category SET time_stamp = 0
				WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
				req.MediaPath, req.TagLabel, req.Timestamp)
		}
		writeJSON(w, map[string]string{})
	}
}

func lokiTagPreviewHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req labelRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		// Return the thumbnail_path_600 from the tag table (a single string, like Electron)
		var thumbPath sql.NullString
		deps.DB.QueryRow(`SELECT thumbnail_path_600 FROM tag WHERE label = ?`, req.Label).Scan(&thumbPath)
		if thumbPath.Valid && thumbPath.String != "" {
			writeJSON(w, thumbPath.String)
		} else {
			writeJSON(w, nil)
		}
	}
}

func lokiTagCountHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req labelRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		var count int
		deps.DB.QueryRow(`
			SELECT COUNT(DISTINCT media_path) FROM media_tag_by_category
			WHERE tag_label = ?`, req.Label).Scan(&count)
		writeJSON(w, map[string]int{"count": count})
	}
}

// ---- Category handlers ----

func lokiCreateCategoryHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req tagRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		_, err := deps.DB.Exec(
			"INSERT INTO category (label, weight) VALUES (?, ?)",
			req.Label, req.Weight)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]string{"label": req.Label})
	}
}

func lokiRenameCategoryHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req renameRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		tx.Exec("UPDATE category SET label = ? WHERE label = ?", req.NewLabel, req.Label)
		tx.Exec("UPDATE tag SET category_label = ? WHERE category_label = ?", req.NewLabel, req.Label)
		tx.Exec("UPDATE media_tag_by_category SET category_label = ? WHERE category_label = ?", req.NewLabel, req.Label)
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

func lokiDeleteCategoryHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req labelRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		tx.Exec("DELETE FROM media_tag_by_category WHERE category_label = ?", req.Label)
		tx.Exec("DELETE FROM tag WHERE category_label = ?", req.Label)
		tx.Exec("DELETE FROM category WHERE label = ?", req.Label)
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

// ---- Assignment handlers ----

func lokiCreateAssignmentHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req assignmentRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		paths := req.MediaPaths
		if len(paths) == 0 && req.MediaPath != "" {
			paths = []string{req.MediaPath}
		}
		if len(paths) == 0 || req.TagLabel == "" {
			httpError(w, "missing required fields", http.StatusBadRequest)
			return
		}
		tx, err := deps.DB.Begin()
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Get current count for weight calculation
		var count int
		deps.DB.QueryRow("SELECT COUNT(*) FROM media_tag_by_category WHERE tag_label = ?", req.TagLabel).Scan(&count)
		for _, p := range paths {
			tx.Exec(`INSERT INTO media_tag_by_category
				(media_path, tag_label, category_label, weight, time_stamp, created_at)
				VALUES (?, ?, ?, ?, ?, strftime('%s','now'))
				ON CONFLICT(media_path, tag_label, category_label, time_stamp) DO NOTHING`,
				p, req.TagLabel, req.CategoryLabel, count, req.TimeStamp)
			count++
		}
		tx.Commit()
		writeJSON(w, map[string]string{})
	}
}

func lokiDeleteAssignmentHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req deleteAssignmentRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		tagLabel := req.Tag.TagLabel
		timeStamp := req.Tag.TimeStamp
		if timeStamp != 0 {
			// Delete specific timestamped assignment
			deps.DB.Exec(`DELETE FROM media_tag_by_category
				WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
				req.MediaPath, tagLabel, timeStamp)
		} else {
			// Delete all assignments for this tag
			deps.DB.Exec(`DELETE FROM media_tag_by_category
				WHERE media_path = ? AND tag_label = ?`,
				req.MediaPath, tagLabel)
		}
		writeJSON(w, map[string]string{})
	}
}

func lokiUpdateAssignmentWeightHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req assignmentWeightRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.MediaTimeStamp != 0 {
			deps.DB.Exec(`UPDATE media_tag_by_category SET weight = ?
				WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
				req.Weight, req.MediaPath, req.TagLabel, req.MediaTimeStamp)
		} else {
			deps.DB.Exec(`UPDATE media_tag_by_category SET weight = ?
				WHERE media_path = ? AND tag_label = ?`,
				req.Weight, req.MediaPath, req.TagLabel)
		}
		writeJSON(w, map[string]string{})
	}
}

// ---- Thumbnail handlers ----

func lokiThumbnailsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		var thumb600, thumb1200 sql.NullString
		deps.DB.QueryRow(
			"SELECT thumbnail_path_600, thumbnail_path_1200 FROM media WHERE path = ?",
			req.Path).Scan(&thumb600, &thumb1200)

		type thumbInfo struct {
			Cache  string `json:"cache"`
			Path   string `json:"path"`
			Exists bool   `json:"exists"`
			Size   int64  `json:"size"`
		}

		var results []thumbInfo
		for _, t := range []struct {
			cache string
			path  sql.NullString
		}{
			{"thumbnail_path_600", thumb600},
			{"thumbnail_path_1200", thumb1200},
		} {
			if t.path.Valid && t.path.String != "" {
				info, err := os.Stat(t.path.String)
				exists := err == nil
				var size int64
				if exists {
					size = info.Size()
				}
				results = append(results, thumbInfo{
					Cache: t.cache, Path: t.path.String,
					Exists: exists, Size: size,
				})
			}
		}
		if results == nil {
			results = []thumbInfo{}
		}
		writeJSON(w, results)
	}
}

func lokiRegenerateThumbnailHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path      string `json:"path"`
			Cache     string `json:"cache"`
			TimeStamp int    `json:"timeStamp"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		// Whitelist cache column names
		cache := req.Cache
		switch cache {
		case "thumbnail_path_100", "thumbnail_path_600", "thumbnail_path_1200":
			// valid
		default:
			cache = "thumbnail_path_600"
		}

		dbPath := currentConfig.DBPath
		if dbPath == "" {
			httpError(w, "no database configured", http.StatusInternalServerError)
			return
		}
		basePath := filepath.Dir(dbPath)

		// Delete existing thumbnail if present
		oldPath := getThumbnailPath(req.Path, basePath, cache, req.TimeStamp)
		os.Remove(oldPath)

		generated, err := generateThumbnail(req.Path, basePath, cache, req.TimeStamp)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Update DB
		deps.DB.Exec(
			fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
			generated, req.Path,
		)

		writeJSON(w, generated)
	}
}

// ---- Settings & Session handlers ----

// In-memory settings store (persisted to a JSON file)
var lokiSettings = make(map[string]any)
var lokiSession = make(map[string]any)

func lokiSettingsGetHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, lokiSettings)
	}
}

func lokiSettingsPutHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Accept either {key, value} for single setting or {key1: val1, key2: val2} for batch
		var raw map[string]any
		if err := readJSON(r, &raw); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if key, ok := raw["key"].(string); ok {
			// Single setting: {key: "foo", value: "bar"}
			lokiSettings[key] = raw["value"]
		} else {
			// Batch: {key1: val1, key2: val2, ...}
			for k, v := range raw {
				lokiSettings[k] = v
			}
		}
		writeJSON(w, map[string]string{})
	}
}

func lokiSessionGetAllHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, lokiSession)
	}
}

func lokiSessionGetHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Extract key from path: /api/session/{key}
		key := strings.TrimPrefix(r.URL.Path, "/api/session/")
		if key == "" {
			writeJSON(w, lokiSession)
			return
		}
		writeJSON(w, lokiSession[key])
	}
}

func lokiSessionPutHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		key := strings.TrimPrefix(r.URL.Path, "/api/session/")
		if key == "" {
			// Bulk update
			var updates map[string]any
			if err := readJSON(r, &updates); err != nil {
				httpError(w, "bad request", http.StatusBadRequest)
				return
			}
			for k, v := range updates {
				lokiSession[k] = v
			}
		} else {
			var value any
			if err := readJSON(r, &value); err != nil {
				httpError(w, "bad request", http.StatusBadRequest)
				return
			}
			lokiSession[key] = value
		}
		writeJSON(w, map[string]string{})
	}
}

func lokiSessionDeleteHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		lokiSession = make(map[string]any)
		writeJSON(w, map[string]string{})
	}
}

func lokiSessionDeleteKeysHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Keys []string `json:"keys"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		for _, k := range req.Keys {
			delete(lokiSession, k)
		}
		writeJSON(w, map[string]string{})
	}
}

// ---- Database handler ----

func lokiDBLoadHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req pathRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		// Database switching: update the deps.DB to point to new file
		// For now, just acknowledge
		writeJSON(w, map[string]string{})
	}
}

// ---- SPA handler ----

func lokiSPAHandler(spaFS fs.FS) http.HandlerFunc {
	fileServer := http.FileServer(http.FS(spaFS))
	return func(w http.ResponseWriter, r *http.Request) {
		// Check if the file exists in the embedded FS
		reqPath := strings.TrimPrefix(r.URL.Path, "/app/")
		if reqPath == "" {
			reqPath = "index.html"
		}
		if f, err := spaFS.Open(reqPath); err == nil {
			f.Close()
			// Serve the static file
			http.StripPrefix("/app/", fileServer).ServeHTTP(w, r)
			return
		}
		// Fallback: serve index.html for SPA routing
		indexData, err := fs.ReadFile(spaFS, "index.html")
		if err != nil {
			http.Error(w, "index.html not found", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexData)
	}
}
