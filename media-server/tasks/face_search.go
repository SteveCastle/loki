package tasks

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
)

// ErrNoFaceInQuery is returned when a query image contains no detectable
// face. HTTP handlers map it to 422 so the UI can show a friendly message.
var ErrNoFaceInQuery = errors.New("no face detected in query image")

// FaceHit is one ranked face-similarity result: the matching stored face plus
// its media item and cosine score.
type FaceHit struct {
	FaceID    int64   `json:"faceId"`
	MediaPath string  `json:"path"`
	Score     float32 `json:"score"`
	// Relative bbox of the matched face within its media item.
	X        float64 `json:"x"`
	Y        float64 `json:"y"`
	W        float64 `json:"w"`
	H        float64 `json:"h"`
	FrameTS  float64 `json:"frameTs"`
	PersonID int64   `json:"personId,omitempty"`
	Model    string  `json:"model"`
}

// buildFacesOneShotArgs assembles `embed --faces --image=...` arguments for a
// single query image (the one-shot sibling of buildFacesServeArgs).
func buildFacesOneShotArgs(detectorPath, recognizerPath, secondaryPath, ortLib, imagePath string, m FaceModel) []string {
	args := append(faceModelArgs(m, detectorPath, recognizerPath, secondaryPath),
		"--image="+imagePath,
	)
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	return args
}

// FacesInImageFile runs the face pipeline on one image file and returns the
// detected faces with their normalized embeddings, under the given recognizer.
func FacesInImageFile(ctx context.Context, m FaceModel, imagePath string) ([]media.NewFace, error) {
	detectorPath, err := FaceDetectorPathFor(m)
	if err != nil {
		return nil, err
	}
	recognizerPath, err := FaceRecognizerPath(m)
	if err != nil {
		return nil, err
	}
	secondaryPath, err := FaceSecondaryPath(m)
	if err != nil {
		return nil, err
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed")
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")

	cmd := exec.CommandContext(ctx, embedBin, buildFacesOneShotArgs(detectorPath, recognizerPath, secondaryPath, ortLib, imagePath, m)...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, embedSubprocessError(err)
	}
	return parseFacesLine(strings.TrimSpace(string(out)))
}

// FacesInImageBytes runs the face pipeline on an arbitrary image (uploaded
// query, captured region) via a temp file and returns the detected faces with
// their normalized embeddings, under the given recognizer.
func FacesInImageBytes(ctx context.Context, m FaceModel, image []byte) ([]media.NewFace, error) {
	if len(image) == 0 {
		return nil, fmt.Errorf("empty image")
	}
	tmp, err := os.CreateTemp("", "facequery-*.png")
	if err != nil {
		return nil, err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(image); err != nil {
		tmp.Close()
		return nil, err
	}
	if err := tmp.Close(); err != nil {
		return nil, err
	}
	return FacesInImageFile(ctx, m, tmpPath)
}

// FacesForPathOrScan returns the stored faces of a library item under the
// active recognizer, scanning it on the fly (and persisting the result +
// updating the live index) when it hasn't been scanned yet — so "find this
// person" works from any media item, not only already-scanned ones.
func FacesForPathOrScan(ctx context.Context, db *sql.DB, path string) ([]media.Face, FaceModel, error) {
	m := ActiveFaceModel()
	scanned, err := media.HasFaceScan(db, path, m.ID)
	if err != nil {
		return nil, m, err
	}
	if scanned {
		faces, err := media.GetFaces(db, path, m.ID)
		return faces, m, err
	}

	imagePath, tempFrame, err := extractFrameForFile(ctx, path, 0)
	if err != nil {
		return nil, m, fmt.Errorf("extract frame from %q: %w", path, err)
	}
	fresh, err := FacesInImageFile(ctx, m, imagePath)
	if tempFrame != "" {
		_ = os.Remove(tempFrame)
	}
	if err != nil {
		return nil, m, err
	}
	ids, err := media.ReplaceFaces(db, path, m.ID, fresh, time.Now().Unix())
	if err != nil {
		return nil, m, err
	}
	faceIndexReplacePath(m.ID, path, ids, fresh)
	autoAssignNewFaces(db, m, ids, fresh)
	faces, err := media.GetFaces(db, path, m.ID)
	return faces, m, err
}

// SearchFacesByMediaPath finds the stored faces most similar to the LARGEST
// face in a library item (scanning it on the fly when needed).
func SearchFacesByMediaPath(ctx context.Context, db *sql.DB, path string, limit int) ([]FaceHit, error) {
	faces, m, err := FacesForPathOrScan(ctx, db, path)
	if err != nil {
		return nil, err
	}
	if len(faces) == 0 {
		return nil, ErrNoFaceInQuery
	}
	best := faces[0]
	for _, f := range faces[1:] {
		if f.W*f.H > best.W*best.H {
			best = f
		}
	}
	return SearchFacesByVector(db, m.ID, best.Vec, limit)
}

// FaceQueryVectorForBytes returns the embedding of the LARGEST face in the
// query image (the natural "search for this person" interpretation of a photo
// with several people) plus the recognizer it belongs to. Errors when the
// image contains no detectable face.
func FaceQueryVectorForBytes(ctx context.Context, image []byte) ([]float32, FaceModel, error) {
	m := ActiveFaceModel()
	faces, err := FacesInImageBytes(ctx, m, image)
	if err != nil {
		return nil, m, err
	}
	if len(faces) == 0 {
		return nil, m, ErrNoFaceInQuery
	}
	best := faces[0]
	for _, f := range faces[1:] {
		if f.W*f.H > best.W*best.H {
			best = f
		}
	}
	return best.Vec, m, nil
}

// SearchFacesByVector returns the top-limit most similar stored faces to query
// under model, using the installed face index when present or brute-force
// cosine over all stored faces otherwise. Hits are hydrated with their face
// rows (bbox, media path, person).
func SearchFacesByVector(db *sql.DB, model string, query []float32, limit int) ([]FaceHit, error) {
	if raw, ok := faceIndexSearch(model, query, limit); ok {
		ids := make([]int64, 0, len(raw))
		scores := make(map[int64]float32, len(raw))
		for _, h := range raw {
			id, err := strconv.ParseInt(h.Path, 10, 64)
			if err != nil {
				continue // foreign key in the index; skip defensively
			}
			ids = append(ids, id)
			scores[id] = h.Score
		}
		hits, err := hydrateFaceHits(db, ids, scores, model)
		if err != nil {
			return nil, err
		}
		return hits, nil
	}

	// Brute force over the DB (no index, or index holds another model).
	all, err := media.LoadAllFaces(db, model)
	if err != nil {
		return nil, err
	}
	hits := make([]FaceHit, 0, len(all))
	for _, f := range all {
		hits = append(hits, faceToHit(f, embedvec.CosineSim(query, f.Vec)))
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

// SearchFacesByImage finds the stored faces most similar to the largest face
// in an uploaded image, under the active recognizer.
func SearchFacesByImage(ctx context.Context, db *sql.DB, image []byte, limit int) ([]FaceHit, error) {
	vec, m, err := FaceQueryVectorForBytes(ctx, image)
	if err != nil {
		return nil, err
	}
	return SearchFacesByVector(db, m.ID, vec, limit)
}

// SimilarFacesByID returns the stored faces most similar to face id, INCLUDING
// the query face itself (cosine 1.0, ranked first — same convention as
// SimilarByPath). The search runs under the query face's own model.
func SimilarFacesByID(db *sql.DB, faceID int64, limit int) ([]FaceHit, error) {
	f, ok, err := media.GetFaceByID(db, faceID)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("no face with id %d", faceID)
	}
	return SearchFacesByVector(db, f.Model, f.Vec, limit)
}

// hydrateFaceHits loads the face rows for ids and pairs them with scores,
// sorted score-descending. Faces deleted since the index was built are
// silently dropped.
func hydrateFaceHits(db *sql.DB, ids []int64, scores map[int64]float32, model string) ([]FaceHit, error) {
	hits := make([]FaceHit, 0, len(ids))
	for _, id := range ids {
		f, ok, err := media.GetFaceByID(db, id)
		if err != nil {
			return nil, err
		}
		if !ok || f.Model != model {
			continue
		}
		hits = append(hits, faceToHit(f, scores[id]))
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	return hits, nil
}

func faceToHit(f media.Face, score float32) FaceHit {
	return FaceHit{
		FaceID:    f.ID,
		MediaPath: f.MediaPath,
		Score:     score,
		X:         f.X, Y: f.Y, W: f.W, H: f.H,
		FrameTS:  f.FrameTS,
		PersonID: f.PersonID,
		Model:    f.Model,
	}
}

// FaceHitsToMediaHits collapses face hits to one hit per media item (best
// face score wins), preserving score order — the shape media-grid consumers
// expect.
func FaceHitsToMediaHits(hits []FaceHit) []SimilarHit {
	best := make(map[string]float32, len(hits))
	order := make([]string, 0, len(hits))
	for _, h := range hits {
		if cur, seen := best[h.MediaPath]; !seen {
			best[h.MediaPath] = h.Score
			order = append(order, h.MediaPath)
		} else if h.Score > cur {
			best[h.MediaPath] = h.Score
		}
	}
	out := make([]SimilarHit, 0, len(order))
	for _, p := range order {
		out = append(out, SimilarHit{Path: p, Score: best[p]})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	return out
}
