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
// SigLIP 2 (google/siglip2-base-patch16-224): same image preprocessing as v1
// base (224x224, RGB, NCHW, mean/std 0.5 -> [-1,1]), but the text encoder uses
// the Gemma multilingual SentencePiece tokenizer (Phase 3). Re-embedding to a
// different model just changes this key — storage is model-keyed, non-destructive.
const EmbedModelID = "siglip2-base-patch16-224"

// EmbedDim is the SigLIP 2 base embedding dimension (768). Other variants:
// large=1024, so400m=1152, giant=1536. Confirm against the chosen ONNX export
// (deferred Task 7) and update if a different variant is selected.
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

// IndexDelete removes a path from the active index under lock (no-op if none).
// It is exported so loki_api.go (package main) can call it on single-item deletion.
func IndexDelete(path string) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex != nil {
		vectorIndex.Delete(path)
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
		idx.Add(e.Path, embedvec.Normalize(e.Vec))
	}
	return idx, nil
}

// SearchByVector returns the top-limit most similar media to query using the
// installed ANN index (when present) or brute-force cosine over all stored
// embeddings. No self-exclusion is performed — callers that need it (e.g.
// SimilarByPath) must filter afterward.
func SearchByVector(db *sql.DB, model string, query []float32, limit int) ([]SimilarHit, error) {
	if raw, ok := indexSearch(query, limit); ok {
		hits := make([]SimilarHit, 0, len(raw))
		for _, h := range raw {
			hits = append(hits, SimilarHit{Path: h.Path, Score: h.Score})
		}
		return hits, nil
	}
	all, err := media.LoadAllEmbeddings(db, model)
	if err != nil {
		return nil, err
	}
	hits := make([]SimilarHit, 0, len(all))
	for _, e := range all {
		hits = append(hits, SimilarHit{Path: e.Path, Score: embedvec.Cosine(query, e.Vec)})
	}
	sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
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

	// Request limit+1 so there is room to drop the self entry.
	all, err := SearchByVector(db, model, query, limit+1)
	if err != nil {
		return nil, err
	}
	hits := make([]SimilarHit, 0, limit)
	for _, h := range all {
		if h.Path == path {
			continue
		}
		hits = append(hits, h)
		if len(hits) == limit {
			break
		}
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

// runEmbedTextSubprocess invokes embed.exe for one text query and returns the
// decoded, already-L2-normalized vector. It mirrors runEmbedSubprocess but
// passes --text/--text-model/--tokenizer instead of --model/--image.
func runEmbedTextSubprocess(ctx context.Context, embedBin, textModel, tokenizer, ortLib, text string, dim int) ([]float32, error) {
	args := []string{
		"--text=" + text,
		"--text-model=" + textModel,
		"--tokenizer=" + tokenizer,
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

// SearchByText encodes text via the SigLIP 2 text encoder subprocess and
// returns the top-limit most similar media by cosine similarity. Returns an
// error (not a panic) when the model, tokenizer, or embed binary is absent.
func SearchByText(ctx context.Context, db *sql.DB, text string, limit int) ([]SimilarHit, error) {
	textModel, err := deps.ModelPath(EmbedModelID, "text_model.onnx")
	if err != nil || textModel == "" {
		return nil, fmt.Errorf("text model not installed: %w", err)
	}
	tokenizer, err := deps.ModelPath(EmbedModelID, "tokenizer.model")
	if err != nil || tokenizer == "" {
		return nil, fmt.Errorf("tokenizer not installed: %w", err)
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed")
	}
	vec, err := runEmbedTextSubprocess(ctx, embedBin, textModel, tokenizer, ortLib, text, EmbedDim)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, EmbedModelID, vec, limit)
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
