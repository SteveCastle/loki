package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

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
	Cache     string `json:"cache"`
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

func lokiMediaHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req mediaRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		if len(req.Tags) == 0 {
			// No tags: return all media
			rows, err := deps.DB.Query("SELECT path FROM media ORDER BY path")
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			defer rows.Close()

			var items []map[string]any
			for rows.Next() {
				var path string
				rows.Scan(&path)
				items = append(items, map[string]any{"path": path})
			}
			if items == nil {
				items = []map[string]any{}
			}
			writeJSON(w, items)
			return
		}

		// With tags: filter by tag assignments
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
			// All tags must match
			query = fmt.Sprintf(`
				SELECT DISTINCT m.path
				FROM media m
				JOIN media_tag_by_category mtbc ON m.path = mtbc.media_path
				WHERE mtbc.tag_label IN (%s)
				GROUP BY m.path
				HAVING COUNT(DISTINCT mtbc.tag_label) = ?
				ORDER BY m.path`, ph)
			args = append(args, len(req.Tags))
		} else {
			// Any tag matches
			query = fmt.Sprintf(`
				SELECT DISTINCT m.path
				FROM media m
				JOIN media_tag_by_category mtbc ON m.path = mtbc.media_path
				WHERE mtbc.tag_label IN (%s)
				ORDER BY m.path`, ph)
		}

		rows, err := deps.DB.Query(query, args...)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var items []map[string]any
		for rows.Next() {
			var path string
			rows.Scan(&path)
			items = append(items, map[string]any{"path": path})
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

		query := `SELECT path FROM media WHERE description LIKE ? ORDER BY path`
		rows, err := deps.DB.Query(query, "%"+req.Description+"%")
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		var items []map[string]any
		for rows.Next() {
			var path string
			rows.Scan(&path)
			items = append(items, map[string]any{"path": path})
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

		// Try DB first
		var width, height sql.NullInt64
		deps.DB.QueryRow("SELECT width, height FROM media WHERE path = ?", req.Path).Scan(&width, &height)

		resp := mediaMetadataResponse{}
		if width.Valid {
			resp.Width = int(width.Int64)
		}
		if height.Valid {
			resp.Height = int(height.Int64)
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
				"tagLabel":      tagLabel,
				"categoryLabel": catLabel,
				"weight":        weight,
				"timeStamp":     timeStamp,
			})
		}
		if tags == nil {
			tags = []map[string]any{}
		}
		writeJSON(w, tags)
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
		cache := req.Cache
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
			writeJSON(w, thumbPath.String)
		} else {
			writeJSON(w, nil)
		}
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
		// Thumbnail regeneration requires ffmpeg — stub for now
		httpError(w, "thumbnail regeneration not yet implemented in web API", http.StatusNotImplemented)
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
		var req struct {
			Key   string `json:"key"`
			Value any    `json:"value"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		lokiSettings[req.Key] = req.Value
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

func lokiSPAHandler(staticDir string) http.HandlerFunc {
	fs := http.FileServer(http.Dir(staticDir))
	return func(w http.ResponseWriter, r *http.Request) {
		// Check if the file exists in the static dir
		filePath := filepath.Join(staticDir, strings.TrimPrefix(r.URL.Path, "/app/"))
		if _, err := os.Stat(filePath); err == nil {
			// Serve the static file
			http.StripPrefix("/app/", fs).ServeHTTP(w, r)
			return
		}
		// Fallback: serve index.html for SPA routing
		http.ServeFile(w, r, filepath.Join(staticDir, "index.html"))
	}
}
