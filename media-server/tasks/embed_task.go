package tasks

import (
	"context"
	"database/sql"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedindex"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
)

// EmbedModelID is the active embedding model; vectors are stored keyed by it.
const EmbedModelID = "siglip-base-patch16-224"

// EmbedDim is the SigLIP base embedding dimension. Confirm against the chosen
// ONNX export (Task 7) and update if different.
const EmbedDim = 768

// SimilarHit is one ranked similarity result.
type SimilarHit struct {
	Path  string  `json:"path"`
	Score float32 `json:"score"`
}

// -----------------------------------------------------------------------------
// Package-level ANN index — serialised behind a single mutex so concurrent
// embed workers can call indexAdd without data-racing on the HNSW graph.
// -----------------------------------------------------------------------------

var (
	vectorIndexMu sync.Mutex
	vectorIndex   embedindex.VectorIndex
)

// SetVectorIndex installs the active ANN index (nil disables it → brute-force).
func SetVectorIndex(idx embedindex.VectorIndex) {
	vectorIndexMu.Lock()
	vectorIndex = idx
	vectorIndexMu.Unlock()
}

// indexSearch runs a locked ANN search. ok is false when no index is installed.
func indexSearch(query []float32, k int) ([]embedindex.SearchHit, bool) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return nil, false
	}
	return vectorIndex.Search(query, k), true
}

// indexAdd inserts one vector into the active index under lock (no-op if none).
func indexAdd(path string, vec []float32) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex != nil {
		vectorIndex.Add(path, vec)
	}
}

// BuildIndexFromDB constructs an ANN index from all stored vectors for model.
func BuildIndexFromDB(db *sql.DB, model string) (embedindex.VectorIndex, error) {
	all, err := media.LoadAllEmbeddings(db, model)
	if err != nil {
		return nil, err
	}
	idx := embedindex.New()
	for _, e := range all {
		idx.Add(e.Path, e.Vec)
	}
	return idx, nil
}

// SimilarByPath returns the top-limit most similar media to path. When an ANN
// index is installed it uses that; otherwise it falls back to brute-force
// cosine over all stored embeddings. The query path is always excluded.
func SimilarByPath(db *sql.DB, model, path string, limit int) ([]SimilarHit, error) {
	query, ok, err := media.GetEmbedding(db, path, model)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("no embedding for %q (model %q)", path, model)
	}

	// Fast path: ANN index is installed.
	if raw, ok := indexSearch(query, limit+1); ok {
		hits := make([]SimilarHit, 0, limit)
		for _, h := range raw {
			if h.Path == path {
				continue
			}
			hits = append(hits, SimilarHit{Path: h.Path, Score: h.Score})
			if len(hits) == limit {
				break
			}
		}
		return hits, nil
	}

	// Slow path: brute-force cosine over all stored embeddings.
	all, err := media.LoadAllEmbeddings(db, model)
	if err != nil {
		return nil, err
	}
	hits := make([]SimilarHit, 0, len(all))
	for _, e := range all {
		if e.Path == path {
			continue
		}
		hits = append(hits, SimilarHit{Path: e.Path, Score: embedvec.Cosine(query, e.Vec)})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

func shouldSkipEmbed(db *sql.DB, path, model string) bool {
	ok, err := media.HasEmbedding(db, path, model)
	return err == nil && ok
}

// runEmbedSubprocess invokes embed.exe for one image and returns the decoded,
// already-L2-normalized vector.
func runEmbedSubprocess(ctx context.Context, embedBin, model, ortLib, imagePath string, dim int) ([]float32, error) {
	args := []string{
		"--model=" + model,
		"--image=" + imagePath,
		fmt.Sprintf("--dim=%d", dim),
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	line := strings.TrimSpace(string(out))
	raw, err := base64.StdEncoding.DecodeString(line)
	if err != nil {
		return nil, fmt.Errorf("decode base64 vector: %w", err)
	}
	return embedvec.Decode(raw)
}

func embedTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var paths []string
	fromQuery := false
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("Using query: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "Failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		paths = mediaPaths
		fromQuery = true
		q.PushJobStdout(j.ID, fmt.Sprintf("Query matched %d items", len(paths)))
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "No image path provided")
			q.CompleteJob(j.ID)
			return nil
		}
		paths = parseInputPaths(raw)
		q.PushJobStdout(j.ID, fmt.Sprintf("Processing %d files from input", len(paths)))
	}
	if len(paths) == 0 {
		q.PushJobStdout(j.ID, "No files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	// Resolve model + runtime + binary (deps first, like autotag).
	imageModel, _ := deps.ModelPath(EmbedModelID, "image_model.onnx")
	if imageModel == "" {
		q.PushJobStdout(j.ID, "SigLIP model not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("model %s not installed", EmbedModelID)
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		q.PushJobStdout(j.ID, "embed binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("embed binary not installed")
	}

	processed, skipped := 0, 0
	for idx, mediaPath := range paths {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return ctx.Err()
		default:
		}

		if shouldSkipEmbed(q.Db, mediaPath, EmbedModelID) {
			skipped++
			continue
		}
		if !fromQuery {
			if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
				q.PushJobStdout(j.ID, fmt.Sprintf("Skipping (not found): %s", filepath.Base(mediaPath)))
				skipped++
				continue
			}
		}

		imagePath := mediaPath
		var tempFramePath string
		switch strings.ToLower(filepath.Ext(mediaPath)) {
		case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".gif":
			framePath, err := extractVideoFrame(ctx, mediaPath, "")
			if err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("  Failed to extract frame: %v", err))
				skipped++
				continue
			}
			tempFramePath = framePath
			imagePath = framePath
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("[%d/%d] Embedding: %s", idx+1, len(paths), filepath.Base(mediaPath)))
		vec, err := runEmbedSubprocess(ctx, embedBin, imageModel, ortLib, imagePath, EmbedDim)
		if tempFramePath != "" {
			_ = os.Remove(tempFramePath)
		}
		if err != nil {
			q.PushJobStdout(j.ID, "  embed failed: "+err.Error())
			skipped++
			continue
		}
		if err := media.UpsertEmbedding(q.Db, mediaPath, EmbedModelID, vec, 0); err != nil {
			q.PushJobStdout(j.ID, "  Failed to store embedding: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		// Incrementally insert into the ANN index so newly-embedded media
		// becomes immediately searchable without a restart.
		indexAdd(mediaPath, embedvec.Normalize(vec))
		processed++
		q.RegisterOutputFile(j.ID, mediaPath)
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d embedded, %d skipped", processed, skipped))
	q.CompleteJob(j.ID)
	return nil
}
