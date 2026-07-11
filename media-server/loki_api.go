package main

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/storage"
	"github.com/stevecastle/shrike/tasks"
)

// redirectToPresigned sends the browser to a presigned URL for an s3://
// object, matching mediaFileHandler's serving strategy.
func redirectToPresigned(w http.ResponseWriter, r *http.Request, backend storage.Backend, path string) {
	u, err := backend.MediaURL(path)
	if err != nil {
		log.Printf("Failed to generate presigned URL for %s: %v", path, err)
		http.Error(w, "Failed to generate media URL", http.StatusInternalServerError)
		return
	}
	http.Redirect(w, r, u, http.StatusFound)
}

// wireRemoteExistence gives the media package a way to answer existence for
// s3:// paths (browser Exists flag, existence-filtered samplers). The
// registry is updated in place on config reload, so capturing it once at
// startup is safe. Network errors report as existing — "unknown" must not
// render as missing.
func wireRemoteExistence(reg *storage.Registry) {
	const remoteExistsConcurrency = 16
	media.SetRemoteExistsChecker(func(paths []string) map[string]bool {
		out := make(map[string]bool, len(paths))
		if len(paths) == 0 {
			return out
		}
		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer cancel()
		sem := make(chan struct{}, remoteExistsConcurrency)
		var mu sync.Mutex
		var wg sync.WaitGroup
		for _, p := range paths {
			backend := reg.BackendFor(p)
			if backend == nil {
				out[p] = false
				continue
			}
			wg.Add(1)
			go func(p string, b storage.Backend) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()
				exists, err := b.Exists(ctx, p)
				mu.Lock()
				out[p] = exists || err != nil
				mu.Unlock()
			}(p, backend)
		}
		wg.Wait()
		return out
	})
}

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
	Path      string  `json:"path"`
	Cache     any     `json:"cache"`
	TimeStamp float64 `json:"timeStamp"`
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

func lokiSimilarHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		path := r.URL.Query().Get("path")
		if path == "" {
			httpError(w, "path is required", http.StatusBadRequest)
			return
		}
		limit := 50
		if l := r.URL.Query().Get("limit"); l != "" {
			if n, err := strconv.Atoi(l); err == nil && n > 0 {
				limit = n
			}
		}
		hits, err := tasks.SimilarByPathOrEmbed(r.Context(), deps.DB, tasks.ActiveEmbedModel().ID, path, limit)
		if err != nil {
			log.Printf("similar search failed (path=%q model=%q): %v", path, tasks.ActiveEmbedModel().ID, err)
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, hits)
	}
}

func lokiVisualSearchHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query().Get("q")
		if q == "" {
			httpError(w, "q is required", http.StatusBadRequest)
			return
		}
		limit := 50
		if l := r.URL.Query().Get("limit"); l != "" {
			if n, err := strconv.Atoi(l); err == nil && n > 0 {
				limit = n
			}
		}
		hits, err := tasks.SearchByText(r.Context(), deps.DB, q, limit)
		if err != nil {
			log.Printf("visual (text) search failed (q=%q): %v", q, err)
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, hits)
	}
}

// decodeImageDataURL decodes a base64 image payload, with or without a
// `data:<mime>;base64,` prefix — the renderer's clip predicates carry a PNG
// data URL so the same value can double as the chip thumbnail.
func decodeImageDataURL(s string) ([]byte, error) {
	if strings.HasPrefix(s, "data:") {
		i := strings.Index(s, ",")
		if i < 0 {
			return nil, errors.New("malformed data URL")
		}
		s = s[i+1:]
	}
	return base64.StdEncoding.DecodeString(s)
}

// blendTextWeight resolves a blended predicate's text share: clamped to [0,1],
// defaulting to an even 0.5 when the client sent text without a weight.
func blendTextWeight(w *float64) float32 {
	if w == nil {
		return 0.5
	}
	v := *w
	if v < 0 {
		v = 0
	}
	if v > 1 {
		v = 1
	}
	return float32(v)
}

// compositeTerms builds the term list for a multi-node similarity predicate:
// the base term (the predicate's own value, weight 1) plus every blend node,
// each with its signed weight (0..1 magnitude, negated by Negative). The
// legacy Text field folds in as one more text term for mixed old/new clients.
func compositeTerms(base tasks.QueryTerm, p Predicate) ([]tasks.QueryTerm, error) {
	terms := []tasks.QueryTerm{base}
	if t := strings.TrimSpace(p.Text); t != "" {
		terms = append(terms, tasks.QueryTerm{Kind: "text", Value: t, Weight: blendTextWeight(p.TextWeight)})
	}
	for _, n := range p.Nodes {
		w := float32(1)
		if n.Weight != nil {
			w = blendTextWeight(n.Weight) // same 0..1 clamp
		}
		if w == 0 {
			continue // zero-weight nodes contribute nothing
		}
		if n.Negative {
			w = -w
		}
		switch n.Kind {
		case "image":
			terms = append(terms, tasks.QueryTerm{Kind: "path", Value: n.Value, Weight: w})
		case "clip":
			img, err := decodeImageDataURL(n.Value)
			if err != nil {
				return nil, fmt.Errorf("blend node clip: %w", err)
			}
			terms = append(terms, tasks.QueryTerm{Kind: "image", Image: img, Weight: w})
		case "text":
			if v := strings.TrimSpace(n.Value); v != "" {
				terms = append(terms, tasks.QueryTerm{Kind: "text", Value: v, Weight: w})
			}
		default:
			return nil, fmt.Errorf("unknown blend node kind %q", n.Kind)
		}
	}
	return terms, nil
}

// sortItemsByScore orders items (each a map with "path") by descending score
// from scoreByPath, attaching item["score"]. Stable for equal scores.
func sortItemsByScore(items []map[string]any, scoreByPath map[string]float32) {
	for _, it := range items {
		p, _ := it["path"].(string)
		it["score"] = scoreByPath[p]
	}
	sort.SliceStable(items, func(a, b int) bool {
		pa, _ := items[a]["path"].(string)
		pb, _ := items[b]["path"].(string)
		return scoreByPath[pa] > scoreByPath[pb]
	})
}

func lokiMediaQueryHandler(deps *Dependencies) http.HandlerFunc {
	type queryRequest struct {
		Predicates []Predicate `json:"predicates"`
		Mode       string      `json:"mode"`
	}
	// Candidate cap for visual predicates: we pull the top-N most similar paths
	// from the ANN/brute-force search, then compose them with the other SQL
	// predicates. A composite like `visual:x AND tag:y` therefore only considers
	// the top-N by similarity — a `y` match ranked beyond N is not returned.
	// 1000 balances recall vs. the SQL IN-list size (well under SQLite's 32766 var cap).
	const visualCandidateLimit = 1000

	return func(w http.ResponseWriter, r *http.Request) {
		var req queryRequest
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		// Resolve visual predicates (similar/visual/clip) into path sets before
		// BuildMediaQuery, which is pure and cannot call the model.
		scoreByPath := map[string]float32{}
		hasVisual := false
		for i := range req.Predicates {
			pt := req.Predicates[i].Type
			val := req.Predicates[i].Value
			if (pt == "similar" || pt == "visual" || pt == "clip" || pt == "face") && val != "" {
				hasVisual = true
				var hits []tasks.SimilarHit
				var err error
				// An image predicate carrying text becomes a blended query: one
				// combined vector ((1-w)*image + w*text) cosine-scanned once,
				// rather than two independently-resolved path sets.
				blendText := strings.TrimSpace(req.Predicates[i].Text)
				hasNodes := len(req.Predicates[i].Nodes) > 0
				switch pt {
				case "similar":
					if hasNodes {
						var terms []tasks.QueryTerm
						terms, err = compositeTerms(tasks.QueryTerm{Kind: "path", Value: val, Weight: 1}, req.Predicates[i])
						if err == nil {
							hits, err = tasks.SearchByComposite(r.Context(), deps.DB, terms, visualCandidateLimit)
						}
					} else if blendText != "" {
						hits, err = tasks.SearchByPathAndText(r.Context(), deps.DB, val, blendText, blendTextWeight(req.Predicates[i].TextWeight), visualCandidateLimit)
					} else {
						hits, err = tasks.SimilarByPathOrEmbed(r.Context(), deps.DB, tasks.ActiveEmbedModel().ID, val, visualCandidateLimit)
					}
				case "clip":
					// A captured screen region: the value is a PNG data URL.
					var image []byte
					if image, err = decodeImageDataURL(val); err == nil {
						if hasNodes {
							var terms []tasks.QueryTerm
							terms, err = compositeTerms(tasks.QueryTerm{Kind: "image", Image: image, Weight: 1}, req.Predicates[i])
							if err == nil {
								hits, err = tasks.SearchByComposite(r.Context(), deps.DB, terms, visualCandidateLimit)
							}
						} else if blendText != "" {
							hits, err = tasks.SearchByImageAndText(r.Context(), deps.DB, image, blendText, blendTextWeight(req.Predicates[i].TextWeight), visualCandidateLimit)
						} else {
							hits, err = tasks.SearchByImage(r.Context(), deps.DB, image, visualCandidateLimit)
						}
					}
				case "face":
					// Face identity: the value is a library path ("find this
					// person") or a captured-region PNG data URL. Matches by
					// face embedding, collapsed to one hit per media item.
					var faceHits []tasks.FaceHit
					if strings.HasPrefix(val, "data:") {
						var image []byte
						if image, err = decodeImageDataURL(val); err == nil {
							faceHits, err = tasks.SearchFacesByImage(r.Context(), deps.DB, image, visualCandidateLimit)
						}
					} else {
						faceHits, err = tasks.SearchFacesByMediaPath(r.Context(), deps.DB, val, visualCandidateLimit)
					}
					hits = tasks.FaceHitsToMediaHits(faceHits)
				default: // "visual": free-text → image search, composable like similar/clip
					if hasNodes {
						var terms []tasks.QueryTerm
						terms, err = compositeTerms(tasks.QueryTerm{Kind: "text", Value: val, Weight: 1}, req.Predicates[i])
						if err == nil {
							hits, err = tasks.SearchByComposite(r.Context(), deps.DB, terms, visualCandidateLimit)
						}
					} else {
						hits, err = tasks.SearchByText(r.Context(), deps.DB, val, visualCandidateLimit)
					}
				}
				if err != nil {
					// A clip/face value can be a multi-hundred-KB data URL — log its size, not the payload.
					logVal := val
					if pt == "clip" || strings.HasPrefix(val, "data:") {
						logVal = fmt.Sprintf("<%s %d bytes>", pt, len(val))
					}
					log.Printf("query: %s predicate %q failed (model=%q): %v", pt, logVal, tasks.ActiveEmbedModel().ID, err)
					httpError(w, err.Error(), http.StatusInternalServerError)
					return
				}
				paths := make([]string, 0, len(hits))
				for _, h := range hits {
					paths = append(paths, h.Path)
					// Merge scores with MAX so multi-predicate composites keep the best.
					if s, ok := scoreByPath[h.Path]; !ok || h.Score > s {
						scoreByPath[h.Path] = h.Score
					}
				}
				req.Predicates[i].Resolved = paths
			}
		}

		querySQL, params := BuildMediaQuery(req.Predicates, req.Mode)
		rows, err := deps.DB.Query(querySQL, params...)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		items := []map[string]any{}
		for rows.Next() {
			var path string
			var elo sql.NullFloat64
			var height, width sql.NullInt64
			var weight sql.NullFloat64
			var tagLabel sql.NullString
			var timeStamp sql.NullFloat64
			var createdAt sql.NullInt64
			// 8 columns, matching BuildMediaQuery — description is filtered but
			// never selected: path, elo, height, width, weight, tag_label,
			// time_stamp, created_at.
			if err := rows.Scan(&path, &elo, &height, &width,
				&weight, &tagLabel, &timeStamp, &createdAt); err != nil {
				continue
			}
			item := map[string]any{"path": path, "mtimeMs": createdAt.Int64}
			if elo.Valid {
				item["elo"] = elo.Float64
			}
			if height.Valid {
				item["height"] = height.Int64
			}
			if width.Valid {
				item["width"] = width.Int64
			}
			if weight.Valid {
				item["weight"] = weight.Float64
			}
			if tagLabel.Valid {
				item["tagLabel"] = tagLabel.String
			}
			if timeStamp.Valid {
				item["timeStamp"] = timeStamp.Float64
			}
			items = append(items, item)
		}

		if hasVisual {
			sortItemsByScore(items, scoreByPath)
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
		// Reads/generates a thumbnail from req.Path — scope it for anon.
		if !mediaReadAllowed(deps, r, req.Path) {
			httpError(w, "path is not within any configured storage root", http.StatusForbidden)
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
			if strings.HasPrefix(thumbPath.String, "s3://") {
				backend := deps.Storage.BackendFor(thumbPath.String)
				if backend != nil {
					exists, _ := backend.Exists(r.Context(), thumbPath.String)
					if exists {
						writeJSON(w, thumbPath.String)
						return
					}
				}
			} else if _, err := os.Stat(thumbPath.String); err == nil {
				writeJSON(w, thumbPath.String)
				return
			}
		}

		// Generate thumbnail — S3 or local
		if strings.HasPrefix(req.Path, "s3://") {
			backend := deps.Storage.BackendFor(req.Path)
			if backend == nil {
				writeJSON(w, nil)
				return
			}
			s3b, ok := backend.(*storage.S3Backend)
			if !ok {
				writeJSON(w, nil)
				return
			}
			generated, err := generateS3ThumbnailThrottled(r.Context(), req.Path, s3b, cache, req.TimeStamp)
			if err != nil {
				log.Printf("S3 thumbnail generation failed for %s: %v", req.Path, err)
				httpError(w, fmt.Sprintf("S3 thumbnail generation failed: %v", err), http.StatusInternalServerError)
				return
			}
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				generated, req.Path,
			)
			log.Printf("Generated S3 thumbnail for %s → %s", req.Path, generated)
			writeJSON(w, generated)
			return
		}

		// Local thumbnail generation
		dbPath := currentConfig.DBPath
		if dbPath == "" {
			writeJSON(w, nil)
			return
		}
		basePath := filepath.Dir(dbPath)

		// Check if thumbnail already exists on disk (e.g. generated by the
		// Electron app which doesn't store the path in the DB).
		expectedPath := getThumbnailPath(req.Path, basePath, cache, req.TimeStamp)
		if _, err := os.Stat(expectedPath); err == nil {
			// Thumbnail exists on disk — store in DB for future lookups and return
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				expectedPath, req.Path,
			)
			writeJSON(w, expectedPath)
			return
		}

		generated, err := generateThumbnailThrottled(req.Path, basePath, cache, req.TimeStamp)
		if err != nil {
			log.Printf("Thumbnail generation failed for %s: %v", req.Path, err)
			httpError(w, fmt.Sprintf("thumbnail generation failed: %v", err), http.StatusInternalServerError)
			return
		}
		deps.DB.Exec(
			fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
			generated, req.Path,
		)
		log.Printf("Generated thumbnail for %s → %s", req.Path, generated)
		writeJSON(w, generated)
	}
}

// mediaThumbnailHandler serves thumbnail bytes directly via GET.
// It generates the thumbnail on-demand if it doesn't exist yet.
// Query params: path (required), cache (default thumbnail_path_600), ts (optional timestamp for video).
func mediaThumbnailHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		rawPath := getRawQueryParam(r.URL.RawQuery, "path")
		if rawPath == "" {
			http.Error(w, "Missing path parameter", http.StatusBadRequest)
			return
		}
		filePath, err := url.PathUnescape(rawPath)
		if err != nil {
			http.Error(w, "Invalid path encoding", http.StatusBadRequest)
			return
		}
		filePath = strings.TrimSpace(filePath)
		if filePath == "" {
			http.Error(w, "Empty file path", http.StatusBadRequest)
			return
		}

		// Validate cache parameter
		cache := r.URL.Query().Get("cache")
		switch cache {
		case "thumbnail_path_100", "thumbnail_path_600", "thumbnail_path_1200":
			// valid
		default:
			cache = "thumbnail_path_600"
		}

		// Parse optional timestamp (float to match Electron's behavior)
		timeStamp := 0.0
		if tsStr := r.URL.Query().Get("ts"); tsStr != "" {
			if v, err := strconv.ParseFloat(tsStr, 64); err == nil {
				timeStamp = v
			}
		}

		// A thumbnail recorded in the DB wins — for S3 media it's an
		// s3:// object we can only find through the DB, and the POST
		// preview handler stores its results there.
		var dbThumb sql.NullString
		deps.DB.QueryRow(
			fmt.Sprintf("SELECT %s FROM media WHERE path = ?", cache),
			filePath,
		).Scan(&dbThumb)
		if dbThumb.Valid && strings.HasPrefix(dbThumb.String, "s3://") {
			if backend := deps.Storage.BackendFor(dbThumb.String); backend != nil {
				if exists, _ := backend.Exists(r.Context(), dbThumb.String); exists {
					redirectToPresigned(w, r, backend, dbThumb.String)
					return
				}
			}
		}

		// S3 source: generate to the bucket (downloads the object, renders
		// with ffmpeg, uploads the thumb) and redirect to a presigned URL.
		if strings.HasPrefix(filePath, "s3://") {
			backend := deps.Storage.BackendFor(filePath)
			if backend == nil {
				http.Error(w, "No storage backend for path", http.StatusNotFound)
				return
			}
			s3b, ok := backend.(*storage.S3Backend)
			if !ok {
				http.Error(w, "Path is not on S3 storage", http.StatusBadRequest)
				return
			}
			generated, genErr := generateS3ThumbnailThrottled(r.Context(), filePath, s3b, cache, timeStamp)
			if genErr != nil {
				log.Printf("S3 thumbnail generation failed for %s: %v", filePath, genErr)
				http.Error(w, "Thumbnail generation failed", http.StatusInternalServerError)
				return
			}
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				generated, filePath,
			)
			redirectToPresigned(w, r, backend, generated)
			return
		}

		// Non-admin requesters may only thumbnail local files inside a
		// configured storage root.
		if !pathAllowedForRequest(deps, r, filePath) {
			http.Error(w, "path is not within any configured storage root", http.StatusForbidden)
			return
		}

		dbPath := currentConfig.DBPath
		if dbPath == "" {
			http.Error(w, "No database configured", http.StatusInternalServerError)
			return
		}
		basePath := filepath.Dir(dbPath)

		// Check if thumbnail already exists on disk (DB-recorded local path
		// first — it may have been generated with different inputs — then
		// the deterministic computed path).
		thumbPath := ""
		if dbThumb.Valid && dbThumb.String != "" && !strings.HasPrefix(dbThumb.String, "s3://") {
			if _, err := os.Stat(dbThumb.String); err == nil {
				thumbPath = dbThumb.String
			}
		}
		if thumbPath == "" {
			thumbPath = getThumbnailPath(filePath, basePath, cache, timeStamp)
		}
		if _, err := os.Stat(thumbPath); err != nil {
			// Not on disk — check if the source file exists before generating
			if !media.CheckFileExists(filePath) {
				http.Error(w, "Source file not found", http.StatusNotFound)
				return
			}

			// Generate the thumbnail
			generated, genErr := generateThumbnailThrottled(filePath, basePath, cache, timeStamp)
			if genErr != nil {
				log.Printf("Thumbnail generation failed for %s: %v", filePath, genErr)
				http.Error(w, "Thumbnail generation failed", http.StatusInternalServerError)
				return
			}
			thumbPath = generated

			// Store in DB for future lookups
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				thumbPath, filePath,
			)
		}

		// Serve the thumbnail file
		fileInfo, err := os.Stat(thumbPath)
		if err != nil {
			http.Error(w, "Thumbnail not found after generation", http.StatusInternalServerError)
			return
		}

		ext := strings.ToLower(filepath.Ext(thumbPath))
		contentType := getContentType(ext)
		w.Header().Set("Content-Type", contentType)

		// Thumbnails are content-addressed (SHA256 hash) — cache aggressively
		w.Header().Set("Cache-Control", "public, max-age=86400, immutable")
		etag := fmt.Sprintf(`"%s-%d"`, filepath.Base(thumbPath), fileInfo.Size())
		w.Header().Set("ETag", etag)

		if r.Header.Get("If-None-Match") == etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}

		http.ServeFile(w, r, thumbPath)
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
		deps.DB.Exec("DELETE FROM media_embedding WHERE media_path = ?", req.Path)
		// Path removed — drop it from the swipe sampler and ANN index.
		media.InvalidateRandomSampleCache()
		tasks.IndexDelete(req.Path)
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
		// ffprobe runs on req.Path (a subprocess that speaks network
		// protocols) — scope it for anon so it can't probe arbitrary files
		// or URLs.
		if !mediaReadAllowed(deps, r, req.Path) {
			httpError(w, "path is not within any configured storage root", http.StatusForbidden)
			return
		}
		// Use ffprobe to get gif metadata
		cmd := exec.Command(depspkg.MustBundled("ffprobe"), "-v", "error", "-count_frames",
			"-select_streams", "v:0", "-show_entries",
			"stream=nb_read_frames,duration", "-of", "json", req.Path)
		platform.HideSubprocessWindow(cmd)
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
	// MediaPaths (bulk) takes precedence over MediaPath when non-empty.
	MediaPaths []string `json:"mediaPaths"`
	MediaPath  string   `json:"mediaPath"`
	Tag        struct {
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

// Returns the category list only (no tags). Pairs with lokiTaxonomyTagsHandler
// so the renderer can lazy-load tags one category at a time.
func lokiCategoriesHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := deps.DB.Query(`
			SELECT label,
			       COALESCE(weight, 0) AS weight,
			       COALESCE(description, '') AS description,
			       COALESCE(tag_view_mode, 'card') AS tag_view_mode
			FROM category
			ORDER BY weight`)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		defer rows.Close()

		type categoryInfo struct {
			Label       string  `json:"label"`
			Weight      float64 `json:"weight"`
			Description string  `json:"description"`
			TagViewMode string  `json:"tagViewMode"`
		}
		result := []categoryInfo{}
		for rows.Next() {
			var c categoryInfo
			if err := rows.Scan(&c.Label, &c.Weight, &c.Description, &c.TagViewMode); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			result = append(result, c)
		}
		writeJSON(w, result)
	}
}

// Returns tag rows. With ?category=<label> returns just that category's tags;
// without the param returns every tag (used by the search box on first focus).
func lokiTaxonomyTagsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		category := r.URL.Query().Get("category")

		var rows *sql.Rows
		var err error
		if category != "" {
			rows, err = deps.DB.Query(`
				SELECT label,
				       COALESCE(category_label, '') AS category_label,
				       COALESCE(weight, 0) AS weight
				FROM tag
				WHERE category_label = ?
				ORDER BY weight`, category)
		} else {
			rows, err = deps.DB.Query(`
				SELECT label,
				       COALESCE(category_label, '') AS category_label,
				       COALESCE(weight, 0) AS weight
				FROM tag
				ORDER BY category_label, weight`)
		}
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
		result := []tagInfo{}
		for rows.Next() {
			var t tagInfo
			if err := rows.Scan(&t.Label, &t.Category, &t.Weight); err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			result = append(result, t)
		}
		writeJSON(w, result)
	}
}

func lokiTaxonomyHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Must return the same shape as the Electron app:
		// { "categoryLabel": { label: string, weight: number, tags: [{label, category, weight}] } }
		rows, err := deps.DB.Query(`
			SELECT
				c.label AS category_label,
				c.weight AS category_weight,
				COALESCE(c.tag_view_mode, '') AS category_tag_view_mode,
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
			Label       string    `json:"label"`
			Weight      float64   `json:"weight"`
			TagViewMode string    `json:"tagViewMode"`
			Tags        []tagInfo `json:"tags"`
		}

		catMap := make(map[string]*categoryInfo)
		catOrder := []string{}

		for rows.Next() {
			var catLabel, catTagViewMode, tagLabel, tagCategory string
			var catWeight, tagWeight float64
			rows.Scan(&catLabel, &catWeight, &catTagViewMode, &tagLabel, &tagCategory, &tagWeight)

			cat, exists := catMap[catLabel]
			if !exists {
				mode := catTagViewMode
				if mode == "" {
					mode = "card"
				}
				cat = &categoryInfo{Label: catLabel, Weight: catWeight, TagViewMode: mode, Tags: []tagInfo{}}
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
		// Cascade may have removed paths from the swipe pool.
		media.InvalidateRandomSampleCache()
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
		// Either branch can shift this path's row out of the swipe sampler's view.
		media.InvalidateRandomSampleCache()
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

func lokiCategoryCountHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		category := r.URL.Query().Get("category")
		var count int
		// Optional cap: COUNT(DISTINCT media_path) walks a table row per index
		// entry, which takes 20+ seconds on the huge autotag "Suggested"
		// category. The SPA's suggestion badge passes a cap so the count stops
		// early; callers that need the exact count (lokictl) omit it.
		if cap, err := strconv.Atoi(r.URL.Query().Get("cap")); err == nil && cap > 0 {
			deps.DB.QueryRow(`SELECT COUNT(*) FROM (
				SELECT DISTINCT media_path FROM media_tag_by_category
				WHERE category_label = ? LIMIT ?)`, category, cap).Scan(&count)
		} else {
			deps.DB.QueryRow(`SELECT COUNT(DISTINCT media_path) FROM media_tag_by_category WHERE category_label = ?`, category).Scan(&count)
		}
		writeJSON(w, count)
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

func lokiUpdateCategoryTagViewModeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Label string `json:"label"`
			Mode  string `json:"mode"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.Label == "" {
			httpError(w, "missing label", http.StatusBadRequest)
			return
		}
		if _, err := deps.DB.Exec(
			"UPDATE category SET tag_view_mode = ? WHERE label = ?",
			req.Mode, req.Label,
		); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
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
		// Cascade may have removed paths from the swipe pool.
		media.InvalidateRandomSampleCache()
		writeJSON(w, map[string]string{})
	}
}

// ---- Assignment handlers ----

// resolveThumbnailPath returns a servable thumbnail path for mediaPath,
// generating one (and recording it on the media row) when none exists yet.
// Mirrors the POST /api/media/preview get-or-generate flow for both local
// and s3:// media. cache must be a whitelisted thumbnail column name.
func resolveThumbnailPath(ctx context.Context, deps *Dependencies, mediaPath, cache string, timeStamp float64) (string, error) {
	var dbThumb sql.NullString
	deps.DB.QueryRow(
		fmt.Sprintf("SELECT %s FROM media WHERE path = ?", cache),
		mediaPath,
	).Scan(&dbThumb)
	if dbThumb.Valid && dbThumb.String != "" {
		if strings.HasPrefix(dbThumb.String, "s3://") {
			if b := deps.Storage.BackendFor(dbThumb.String); b != nil {
				if ok, _ := b.Exists(ctx, dbThumb.String); ok {
					return dbThumb.String, nil
				}
			}
		} else if _, err := os.Stat(dbThumb.String); err == nil {
			return dbThumb.String, nil
		}
	}

	if strings.HasPrefix(mediaPath, "s3://") {
		backend := deps.Storage.BackendFor(mediaPath)
		s3b, ok := backend.(*storage.S3Backend)
		if backend == nil || !ok {
			return "", fmt.Errorf("no S3 backend for %s", mediaPath)
		}
		generated, err := generateS3ThumbnailThrottled(ctx, mediaPath, s3b, cache, timeStamp)
		if err != nil {
			return "", err
		}
		deps.DB.Exec(
			fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
			generated, mediaPath,
		)
		return generated, nil
	}

	dbPath := currentConfig.DBPath
	if dbPath == "" {
		return "", fmt.Errorf("no database configured")
	}
	basePath := filepath.Dir(dbPath)
	generated := getThumbnailPath(mediaPath, basePath, cache, timeStamp)
	if _, err := os.Stat(generated); err != nil {
		var genErr error
		generated, genErr = generateThumbnailThrottled(mediaPath, basePath, cache, timeStamp)
		if genErr != nil {
			return "", genErr
		}
	}
	deps.DB.Exec(
		fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
		generated, mediaPath,
	)
	return generated, nil
}

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
		// New assignments may add previously-untagged paths to the swipe pool.
		media.InvalidateRandomSampleCache()

		// Shift-drag (or the tag-preview toggle): the tagged media's
		// thumbnail becomes the tag's preview image, mirroring the
		// Electron main-process behavior (src/main/taxonomy.ts).
		if req.ApplyTagPreview {
			thumb, err := resolveThumbnailPath(
				r.Context(), deps, paths[0], "thumbnail_path_600", req.TimeStamp,
			)
			if err != nil {
				log.Printf("apply tag preview for %q: %v", req.TagLabel, err)
			} else {
				deps.DB.Exec(
					"UPDATE tag SET thumbnail_path_600 = ? WHERE label = ?",
					thumb, req.TagLabel,
				)
			}
		}
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
		paths := req.MediaPaths
		if len(paths) == 0 {
			paths = []string{req.MediaPath}
		}
		for _, p := range paths {
			if timeStamp != 0 {
				// Delete specific timestamped assignment
				deps.DB.Exec(`DELETE FROM media_tag_by_category
					WHERE media_path = ? AND tag_label = ? AND time_stamp = ?`,
					p, tagLabel, timeStamp)
			} else {
				// Delete all assignments for this tag
				deps.DB.Exec(`DELETE FROM media_tag_by_category
					WHERE media_path = ? AND tag_label = ?`,
					p, tagLabel)
			}
		}
		// Person tags mirror face assignments: once an item no longer carries
		// the person's tag at any timestamp, removing it means "this person is
		// not in this item" — discard their faces from the group (veto +
		// cannot-links) so clustering can't put them back.
		if person, found, err := media.GetPersonByName(deps.DB, tagLabel); err == nil && found {
			rejected := false
			for _, p := range paths {
				var remaining int
				deps.DB.QueryRow(`SELECT COUNT(*) FROM media_tag_by_category
					WHERE media_path = ? AND tag_label = ?`, p, tagLabel).Scan(&remaining)
				if remaining > 0 {
					continue // a timestamped copy survives; still tagged
				}
				if ids, err := media.RejectPersonFacesOnMedia(deps.DB, p, person.ID); err == nil && len(ids) > 0 {
					rejected = true
				}
			}
			if rejected {
				broadcastPeopleChanged()
			}
		}
		// Removing the path's last tag drops it from the swipe pool.
		media.InvalidateRandomSampleCache()
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
			Path      string  `json:"path"`
			Cache     string  `json:"cache"`
			TimeStamp float64 `json:"timeStamp"`
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

		generated, err := generateThumbnailThrottled(req.Path, basePath, cache, req.TimeStamp)
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
			// The bundle files are NOT content-hashed (renderer.js keeps its
			// name across releases) and embedded files carry no modtime, so
			// any heuristic caching serves a stale app after a server
			// upgrade. Force revalidation on every load.
			w.Header().Set("Cache-Control", "no-cache")
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
		w.Header().Set("Cache-Control", "no-cache")
		w.Write(indexData)
	}
}
