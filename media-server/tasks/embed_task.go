package tasks

import (
	"context"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"os/exec"
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

// EmbedModelID is the default (SigLIP 2) embedding model ID. The *active* model
// is resolved at runtime from config via ActiveEmbedModel() — this constant is
// the canonical default identity and is what tests embed under. Vectors are
// stored keyed by model ID, so models coexist non-destructively. See
// embedmodels.go for the registry of supported models.
const EmbedModelID = DefaultEmbedModelID

// EmbedDim is the default (SigLIP 2 base) embedding dimension. Per-model
// dimensions live in the registry (EmbedModel.Dim); this remains for the
// default-model code paths and tests.
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
	// vectorIndexModel records which embedding model the installed index holds.
	// "" is a wildcard (matches any model) — used by tests and the legacy
	// SetVectorIndex entry point. Production installs an index for a specific
	// model so searches for a *different* model fall back to brute-force rather
	// than querying the wrong model's vectors.
	vectorIndexModel string
)

// SetVectorIndex installs a model-agnostic (wildcard) ANN index. Prefer
// SetVectorIndexForModel in production; this is retained for tests and callers
// that operate on a single known model. nil disables it → brute-force.
func SetVectorIndex(idx embedindex.VectorIndex) {
	SetVectorIndexForModel(idx, "")
}

// SetVectorIndexForModel installs the active ANN index and records the model it
// was built for. Searches whose model differs skip the index (brute-force).
func SetVectorIndexForModel(idx embedindex.VectorIndex, model string) {
	vectorIndexMu.Lock()
	vectorIndex = idx
	vectorIndexModel = model
	vectorIndexMu.Unlock()
}

// IndexedModel returns the model ID the installed index was built for ("" when
// none or wildcard).
func IndexedModel() string {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	return vectorIndexModel
}

// indexSearch runs a locked ANN search for model. ok is false when no index is
// installed or the index holds a different model's vectors (caller brute-forces).
func indexSearch(model string, query []float32, k int) ([]embedindex.SearchHit, bool) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return nil, false
	}
	if vectorIndexModel != "" && vectorIndexModel != model {
		return nil, false
	}
	return vectorIndex.Search(query, k), true
}

// indexAdd inserts one vector into the active index under lock, but only when
// the index holds the same model (or is a wildcard). No-op if no index.
func indexAdd(model, path string, vec []float32) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return
	}
	if vectorIndexModel != "" && vectorIndexModel != model {
		return
	}
	vectorIndex.Add(path, vec)
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

// IndexProgress is called during a build with the number of vectors added so
// far and the total, so callers can render progress. nil disables reporting.
type IndexProgress func(done, total int)

// RebuildActiveIndex builds the ANN index for the currently-configured active
// model and installs it (tagged with that model). Used at startup and after the
// active model changes in config. onProgress (may be nil) is invoked as the
// index builds. Returns the installed model ID and vector count, or an error (in
// which case the previous index is left untouched).
func RebuildActiveIndex(db *sql.DB, onProgress IndexProgress) (string, int, error) {
	model := ActiveEmbedModel()
	idx, err := BuildIndexFromDB(db, model.ID, onProgress)
	if err != nil {
		return model.ID, 0, err
	}
	SetVectorIndexForModel(idx, model.ID)
	return model.ID, idx.Len(), nil
}

// BuildIndexFromDB constructs an ANN index from all stored vectors for model,
// reporting progress to onProgress (may be nil) as vectors are inserted.
func BuildIndexFromDB(db *sql.DB, model string, onProgress IndexProgress) (embedindex.VectorIndex, error) {
	all, err := media.LoadAllEmbeddings(db, model)
	if err != nil {
		return nil, err
	}
	idx := embedindex.New()
	total := len(all)
	if onProgress != nil {
		onProgress(0, total) // start the bar (covers the empty-DB case too)
	}
	for i, e := range all {
		idx.Add(e.Path, embedvec.Normalize(e.Vec))
		// Throttle callback frequency; always fire on the last item.
		if onProgress != nil && ((i+1)%512 == 0 || i+1 == total) {
			onProgress(i+1, total)
		}
	}
	return idx, nil
}

// SearchByVector returns the top-limit most similar media to query using the
// installed ANN index (when present) or brute-force cosine over all stored
// embeddings. No self-exclusion is performed — callers that need it (e.g.
// SimilarByPath) must filter afterward.
func SearchByVector(db *sql.DB, model string, query []float32, limit int) ([]SimilarHit, error) {
	if raw, ok := indexSearch(model, query, limit); ok {
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

// SimilarByPath returns the top-limit most similar media to path, INCLUDING the
// query item itself — it has cosine similarity 1.0, so it ranks first, which is
// the desired "find similar" UX (you see the source item, then its neighbours).
// When an ANN index is installed it uses that; otherwise it falls back to
// brute-force cosine over all stored embeddings.
func SimilarByPath(db *sql.DB, model, path string, limit int) ([]SimilarHit, error) {
	query, ok, err := media.GetEmbedding(db, path, model)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, fmt.Errorf("no embedding for %q (model %q)", path, model)
	}
	return SearchByVector(db, model, query, limit)
}

// SimilarByPathOrEmbed is like SimilarByPath but, when the query item has no
// stored embedding for the model yet, it embeds the file on the fly so
// "find similar" works on any media — not only already-indexed items — and
// persists the result so subsequent searches are instant.
func SimilarByPathOrEmbed(ctx context.Context, db *sql.DB, modelID, path string, limit int) ([]SimilarHit, error) {
	vec, ok, err := media.GetEmbedding(db, path, modelID)
	if err != nil {
		return nil, err
	}
	if ok {
		return SearchByVector(db, modelID, vec, limit)
	}
	m, found := EmbedModelByID(modelID)
	if !found {
		m = ActiveEmbedModel()
	}
	fresh, err := embedFileWithModel(ctx, path, m)
	if err != nil {
		return nil, fmt.Errorf("embed query item %q: %w", path, err)
	}
	// Persist + index so future searches over this item are instant.
	if uerr := media.UpsertEmbedding(db, path, m.ID, fresh, 0); uerr == nil {
		indexAdd(m.ID, path, embedvec.Normalize(fresh))
	}
	return SearchByVector(db, m.ID, fresh, limit)
}

// embedSubprocessError wraps a failed embed-subprocess run, surfacing the
// subprocess's stderr (captured into ExitError.Stderr by cmd.Output) so the real
// cause is visible instead of a bare "exit status 1".
func embedSubprocessError(err error) error {
	var ee *exec.ExitError
	if errors.As(err, &ee) && len(ee.Stderr) > 0 {
		return fmt.Errorf("embed subprocess failed: %s: %w", strings.TrimSpace(string(ee.Stderr)), err)
	}
	return fmt.Errorf("embed subprocess failed: %w", err)
}

func shouldSkipEmbed(db *sql.DB, path, model string) bool {
	ok, err := media.HasEmbedding(db, path, model)
	return err == nil && ok
}

// runEmbedSubprocess invokes embed.exe for one image and returns the decoded,
// already-L2-normalized vector. The model profile drives preprocessing
// (mean/std, crop, input/output tensor names, dimension, pooling) so different
// models — e.g. SigLIP 2 (pooled output) vs DINOv2 (CLS of last_hidden_state) —
// share one binary. imageModelPath is the on-disk path to the model's image
// encoder.
func runEmbedSubprocess(ctx context.Context, embedBin, imageModelPath, ortLib, imagePath string, m EmbedModel) ([]float32, error) {
	args := []string{
		"--model=" + imageModelPath,
		"--image=" + imagePath,
		fmt.Sprintf("--dim=%d", m.Dim),
		"--input=" + m.ImgInput,
		"--output=" + m.ImgOutput,
		fmt.Sprintf("--width=%d", m.Width),
		fmt.Sprintf("--height=%d", m.Height),
		fmt.Sprintf("--mean=%g,%g,%g", m.Mean[0], m.Mean[1], m.Mean[2]),
		fmt.Sprintf("--std=%g,%g,%g", m.Std[0], m.Std[1], m.Std[2]),
	}
	if m.CropPct > 0 && m.CropPct < 1 {
		args = append(args, fmt.Sprintf("--crop-pct=%g", m.CropPct), "--crop-mode="+m.CropMode)
	}
	if m.Pooling != "" {
		args = append(args, "--pooling="+m.Pooling)
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, embedSubprocessError(err)
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
func runEmbedTextSubprocess(ctx context.Context, embedBin, textModel, tokenizer, ortLib, text string, m EmbedModel) ([]float32, error) {
	args := []string{
		"--text=" + text,
		"--text-model=" + textModel,
		"--tokenizer=" + tokenizer,
		fmt.Sprintf("--dim=%d", m.Dim),
	}
	if m.TextInput != "" {
		args = append(args, "--text-input="+m.TextInput)
	}
	if m.TextOutput != "" {
		args = append(args, "--text-output="+m.TextOutput)
	}
	if m.SeqLen > 0 {
		args = append(args, fmt.Sprintf("--seq-len=%d", m.SeqLen))
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, embedSubprocessError(err)
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
	// Text->image search needs a text encoder; resolve the multimodal model
	// (the active model if it's multimodal, otherwise SigLIP 2). Vectors are
	// matched against this model's image embeddings.
	m := TextSearchModel()
	if !m.Multimodal || m.TextModelFile == "" {
		return nil, fmt.Errorf("model %q does not support text search", m.ID)
	}
	textModel, err := deps.ModelPath(m.ID, m.TextModelFile)
	if err != nil {
		return nil, fmt.Errorf("text model not installed: %w", err)
	}
	if textModel == "" {
		return nil, fmt.Errorf("text model not installed")
	}
	tokenizer, err := deps.ModelPath(m.ID, m.TokenizerFile)
	if err != nil {
		return nil, fmt.Errorf("tokenizer not installed: %w", err)
	}
	if tokenizer == "" {
		return nil, fmt.Errorf("tokenizer not installed")
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed")
	}
	vec, err := runEmbedTextSubprocess(ctx, embedBin, textModel, tokenizer, ortLib, text, m)
	if err != nil {
		return nil, err
	}
	return SearchByVector(db, m.ID, vec, limit)
}

// embedModelOverrideFromJob returns an explicit `--model=<id>` (or `--model
// <id>`) from the job arguments when present. Lets an embed job target a model
// other than the configured active one (background migration).
func embedModelOverrideFromJob(j *jobqueue.Job) (string, bool) {
	for i := 0; i < len(j.Arguments); i++ {
		arg := j.Arguments[i]
		if strings.HasPrefix(arg, "--model=") {
			if v := strings.TrimSpace(arg[len("--model="):]); v != "" {
				return v, true
			}
		}
		if arg == "--model" && i+1 < len(j.Arguments) {
			if v := strings.TrimSpace(j.Arguments[i+1]); v != "" {
				return v, true
			}
		}
	}
	return "", false
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

	// Resolve the embedding model. An explicit `--model=<id>` in the job
	// overrides the configured active model — this enables zero-downtime
	// migration: embed the whole library under a new model in the background
	// while the active model still serves search, then flip the config.
	model := ActiveEmbedModel()
	if id, ok := embedModelOverrideFromJob(j); ok {
		if m, known := EmbedModelByID(id); known {
			model = m
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown --model %q; using active model %q", id, model.ID))
		}
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Embedding model: %s (dim %d)", model.ID, model.Dim))

	// Resolve model + runtime + binary (deps first, like autotag).
	imageModel, _ := deps.ModelPath(model.ID, model.ImageModelFile)
	if imageModel == "" {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s not installed; install it from Dependencies", model.DisplayName))
		q.ErrorJob(j.ID)
		return fmt.Errorf("model %s not installed", model.ID)
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		q.PushJobStdout(j.ID, "embed binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return fmt.Errorf("embed binary not installed")
	}

	// Embed via a pool of persistent workers (model loaded once per worker, not
	// per image). The pool size + ONNX threads come from the performance config.
	processed, skipped, err := runEmbedPool(ctx, j, q, paths, fromQuery, model, imageModel, embedBin)
	if err != nil {
		if ctx.Err() != nil {
			q.PushJobStdout(j.ID, "Task canceled")
			_ = q.CancelJob(j.ID)
			return err
		}
		q.PushJobStdout(j.ID, "Embedding failed: "+err.Error())
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Completed: %d embedded, %d skipped", processed, skipped))
	q.CompleteJob(j.ID)
	return nil
}
