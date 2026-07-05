package main

import (
	"context"
	"database/sql"
	"errors"
	"image"
	_ "image/gif"
	"image/jpeg"
	_ "image/png"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/tasks"
	_ "golang.org/x/image/bmp"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// buildFaceIndexAtStartup builds the in-memory face vector index from all
// stored faces for the active recognizer (best-effort, non-fatal — searches
// fall back to brute-force DB scans without it). Called from each platform's
// main file right after the embedding index build.
func buildFaceIndexAtStartup(db *sql.DB) {
	log.Printf("Building face search index…")
	if model, n, err := tasks.RebuildActiveFaceIndex(db, nil); err == nil {
		log.Printf("face index loaded: %d faces (model %s)", n, model)
	} else {
		log.Printf("face index unavailable (model %s), using brute-force: %v", model, err)
	}
}

// RegisterFacesRoutes wires the face-identity API onto mux. Called from each
// platform's main file alongside the other API routes.
//
//	POST /api/media/search/face      — image body → media ranked by face match
//	GET  /api/faces?path=            — stored faces for one media item
//	GET  /api/faces/{id}/similar     — media ranked by similarity to one face
//	GET  /media/facecrop?id=&size=   — JPEG crop of a stored face
func RegisterFacesRoutes(mux *http.ServeMux, deps *Dependencies) {
	mux.HandleFunc("/api/media/search/face", renderer.ApplyMiddlewares(lokiFaceSearchHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces", renderer.ApplyMiddlewares(facesForPathHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/faces/{id}/similar", renderer.ApplyMiddlewares(similarFacesHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/facecrop", renderer.ApplyMiddlewares(faceCropHandler(deps), renderer.RoleAdmin))
	RegisterPeopleRoutes(mux, deps)
}

// faceHitItems converts ranked face hits into the flat media-item shape the
// renderer's grids expect (path + score + metadata), collapsed to one item per
// media path (best face wins) with that face attached under "face".
func faceHitItems(db *sql.DB, hits []tasks.FaceHit) ([]map[string]any, error) {
	items, err := enrichScoredItems(db, tasks.FaceHitsToMediaHits(hits))
	if err != nil {
		return nil, err
	}
	bestByPath := make(map[string]tasks.FaceHit, len(hits))
	for _, h := range hits {
		if cur, seen := bestByPath[h.MediaPath]; !seen || h.Score > cur.Score {
			bestByPath[h.MediaPath] = h
		}
	}
	for _, item := range items {
		path, _ := item["path"].(string)
		if h, ok := bestByPath[path]; ok {
			item["faceId"] = h.FaceID
			item["face"] = map[string]any{
				"x": h.X, "y": h.Y, "w": h.W, "h": h.H,
				"frameTs":  h.FrameTS,
				"personId": h.PersonID,
			}
		}
	}
	return items, nil
}

// searchLimit parses ?limit with a default and cap. The limit counts FACES
// searched, so after collapsing to media items the result may be smaller.
func searchLimit(r *http.Request, def, max int) int {
	if v := r.URL.Query().Get("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			if n > max {
				return max
			}
			return n
		}
	}
	return def
}

func lokiFaceSearchHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			httpError(w, "use POST with an image body", http.StatusMethodNotAllowed)
			return
		}
		body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 32<<20)) // 32 MB cap
		if err != nil {
			if errors.As(err, new(*http.MaxBytesError)) {
				httpError(w, "image too large (max 32 MB)", http.StatusRequestEntityTooLarge)
			} else {
				httpError(w, "image body required", http.StatusBadRequest)
			}
			return
		}
		if len(body) == 0 {
			httpError(w, "image body required", http.StatusBadRequest)
			return
		}
		hits, err := tasks.SearchFacesByImage(r.Context(), deps.DB, body, searchLimit(r, 100, 500))
		if err != nil {
			if errors.Is(err, tasks.ErrNoFaceInQuery) {
				httpError(w, err.Error(), http.StatusUnprocessableEntity)
			} else {
				httpError(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		items, err := faceHitItems(deps.DB, hits)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, items)
	}
}

func facesForPathHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		path := r.URL.Query().Get("path")
		if path == "" {
			httpError(w, "path query parameter required", http.StatusBadRequest)
			return
		}
		model := tasks.ActiveFaceModel()
		faces, err := media.GetFaces(deps.DB, path, model.ID)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		scanned, err := media.HasFaceScan(deps.DB, path, model.ID)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		out := make([]map[string]any, 0, len(faces))
		for _, f := range faces {
			out = append(out, map[string]any{
				"id": f.ID,
				"x":  f.X, "y": f.Y, "w": f.W, "h": f.H,
				"score":      f.Score,
				"frameTs":    f.FrameTS,
				"personId":   f.PersonID,
				"assignedBy": f.AssignedBy,
			})
		}
		writeJSON(w, map[string]any{
			"model":   model.ID,
			"scanned": scanned,
			"faces":   out,
		})
	}
}

func similarFacesHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		id, err := strconv.ParseInt(r.PathValue("id"), 10, 64)
		if err != nil {
			httpError(w, "invalid face id", http.StatusBadRequest)
			return
		}
		hits, err := tasks.SimilarFacesByID(deps.DB, id, searchLimit(r, 100, 500))
		if err != nil {
			if strings.Contains(err.Error(), "no face with id") {
				httpError(w, err.Error(), http.StatusNotFound)
			} else {
				httpError(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		items, err := faceHitItems(deps.DB, hits)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, items)
	}
}

// faceCropMargin expands the stored bbox on every side (fraction of the bbox's
// larger edge) so crops show the whole head, not a tight face rectangle.
const faceCropMargin = 0.35

func faceCropHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			httpError(w, "use GET", http.StatusMethodNotAllowed)
			return
		}
		id, err := strconv.ParseInt(r.URL.Query().Get("id"), 10, 64)
		if err != nil {
			httpError(w, "id query parameter required", http.StatusBadRequest)
			return
		}
		size := 160
		if v := r.URL.Query().Get("size"); v != "" {
			if n, perr := strconv.Atoi(v); perr == nil && n >= 32 && n <= 1024 {
				size = n
			}
		}

		f, ok, err := media.GetFaceByID(deps.DB, id)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		if !ok {
			httpError(w, "no such face", http.StatusNotFound)
			return
		}

		img, err := decodeFaceSource(r.Context(), deps.DB, f.MediaPath)
		if err != nil {
			httpError(w, err.Error(), http.StatusNotFound)
			return
		}
		crop := cropFace(img, f, faceCropMargin, size)

		w.Header().Set("Content-Type", "image/jpeg")
		// A face id's crop never changes (rescans mint new ids), so let
		// clients cache aggressively.
		w.Header().Set("Cache-Control", "private, max-age=86400")
		_ = jpeg.Encode(w, crop, &jpeg.Options{Quality: 88})
	}
}

// decodeFaceSource decodes the media item the face belongs to. Images decode
// directly. Videos re-extract the SAME deterministic midpoint frame the scan
// analyzed (via ffmpeg), so the stored relative bbox lines up exactly. Only
// when frame extraction fails does it fall back to the stored 600px thumbnail
// — same aspect ratio, but potentially a different frame, so it's a
// last-resort approximation rather than the primary path.
func decodeFaceSource(ctx context.Context, db *sql.DB, mediaPath string) (image.Image, error) {
	if img, err := decodeImageFile(mediaPath); err == nil {
		return img, nil
	}
	if framePath, tempFrame, err := tasks.ExtractFrameForMedia(ctx, mediaPath); err == nil {
		img, derr := decodeImageFile(framePath)
		if tempFrame != "" {
			_ = os.Remove(tempFrame)
		}
		if derr == nil {
			return img, nil
		}
	}
	var thumb sql.NullString
	if err := db.QueryRow(`SELECT thumbnail_path_600 FROM media WHERE path = ?`, mediaPath).Scan(&thumb); err != nil || !thumb.Valid || thumb.String == "" {
		return nil, errors.New("media not decodable and no thumbnail available")
	}
	img, err := decodeImageFile(thumb.String)
	if err != nil {
		return nil, errors.New("thumbnail not decodable")
	}
	return img, nil
}

// decodeImageFile opens and decodes one image file.
func decodeImageFile(path string) (image.Image, error) {
	fh, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	img, _, err := image.Decode(fh)
	return img, err
}

// cropFace cuts the face's (relative) bbox out of img with a margin and
// scales the result so its longer edge is maxSide.
func cropFace(img image.Image, f media.Face, margin float64, maxSide int) image.Image {
	b := img.Bounds()
	iw, ih := float64(b.Dx()), float64(b.Dy())

	// Absolute bbox + margin, clamped to the image.
	mx := f.W * iw
	if f.H*ih > mx {
		mx = f.H * ih
	}
	pad := mx * margin
	x0 := f.X*iw - pad
	y0 := f.Y*ih - pad
	x1 := (f.X+f.W)*iw + pad
	y1 := (f.Y+f.H)*ih + pad
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > iw {
		x1 = iw
	}
	if y1 > ih {
		y1 = ih
	}
	cw, ch := x1-x0, y1-y0
	if cw < 1 || ch < 1 { // degenerate bbox — return a scaled full image
		x0, y0, cw, ch = 0, 0, iw, ih
	}

	scale := float64(maxSide) / cw
	if ch > cw {
		scale = float64(maxSide) / ch
	}
	if scale > 1 {
		scale = 1 // never upscale
	}
	dw := int(cw*scale + 0.5)
	dh := int(ch*scale + 0.5)
	if dw < 1 {
		dw = 1
	}
	if dh < 1 {
		dh = 1
	}

	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	src := image.Rect(b.Min.X+int(x0), b.Min.Y+int(y0), b.Min.X+int(x1), b.Min.Y+int(y1))
	draw.CatmullRom.Scale(dst, dst.Bounds(), img, src, draw.Src, nil)
	return dst
}
