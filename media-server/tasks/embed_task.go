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
// Package-level vector index — serialised behind a single mutex so concurrent
// embed workers can call indexAdd without data-racing on the index.
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

// IndexSize returns the number of active (non-tombstoned) vectors in the
// installed index, or 0 when none is installed. Exported for the index-status
// API alongside IndexedModel.
func IndexSize() int {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return 0
	}
	return vectorIndex.Len()
}

// indexSearch runs a locked ANN search for model, restricted to allow when
// non-nil. ok is false when no index is installed or the index holds a
// different model's vectors (caller brute-forces).
func indexSearch(model string, query []float32, k int, allow PathSet, s embedindex.Scoring) ([]embedindex.SearchHit, bool) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return nil, false
	}
	if vectorIndexModel != "" && vectorIndexModel != model {
		return nil, false
	}
	return vectorIndex.SearchScored(query, k, allow, s), true
}

// indexSearchShared runs a locked shared-concept search for model, restricted
// to allow when non-nil. ok is false when no index is installed or the index
// holds a different model's vectors (caller brute-forces).
func indexSearchShared(model string, spec embedindex.SharedSpec, k int, allow PathSet, s embedindex.Scoring) ([]embedindex.SearchHit, bool, error) {
	vectorIndexMu.Lock()
	defer vectorIndexMu.Unlock()
	if vectorIndex == nil {
		return nil, false, nil
	}
	if vectorIndexModel != "" && vectorIndexModel != model {
		return nil, false, nil
	}
	hits, err := vectorIndex.SearchSharedScored(spec, k, allow, s)
	return hits, true, err
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

// IndexRenamePath re-keys a path in the live vector index after its database
// rows have been rewritten (see media.MovePath). Without it, similarity search
// keeps returning the OLD path until the next index rebuild — a hit the media
// handler then can't serve. The vector is re-read from the DB under the new
// path rather than carried across, so this is also correct when the row was
// re-embedded in between. No-op when nothing is indexed for the model.
func IndexRenamePath(db *sql.DB, from, to string) {
	IndexDelete(from)
	model := IndexedModel()
	if model == "" {
		// Wildcard/no index installed: fall back to the configured model, the
		// one a production index would have been built for.
		model = ActiveEmbedModel().ID
	}
	vec, ok, err := media.GetEmbedding(db, to, model)
	if err != nil || !ok {
		return
	}
	indexAdd(model, to, vec)
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
	// Centered: media similarity is ranking-only, so it benefits from the
	// hub-suppressing mean subtraction. Face indexes must NOT use this — their
	// calibrated absolute thresholds assume plain cosine (embedindex.New).
	idx := embedindex.NewCentered()
	total := len(all)
	if onProgress != nil {
		onProgress(0, total) // start the bar (covers the empty-DB case too)
	}
	for i, e := range all {
		idx.Add(e.Path, e.Vec) // Add L2-normalizes internally
		// Throttle callback frequency; always fire on the last item.
		if onProgress != nil && ((i+1)%512 == 0 || i+1 == total) {
			onProgress(i+1, total)
		}
	}
	return idx, nil
}

// PathSet restricts a similarity search to a subset of media paths (nil = no
// restriction). Composed queries resolve their SQL predicates into a PathSet
// first, so the vector scan ranks within the filtered set instead of the
// whole library.
type PathSet map[string]struct{}

// SearchByVector returns the top-limit most similar media to query using the
// installed ANN index (when present) or brute-force cosine over all stored
// embeddings. No self-exclusion is performed — callers that need it (e.g.
// SimilarByPath) must filter afterward.
func SearchByVector(db *sql.DB, model string, query []float32, limit int) ([]SimilarHit, error) {
	return searchByVectorWithin(db, model, query, limit, nil)
}

// searchByVectorWithin is SearchByVector restricted to allow (nil = whole
// library), scored the installed index's way. Use it for image queries; text
// and blended queries must use searchByVectorCrossModal.
func searchByVectorWithin(db *sql.DB, model string, query []float32, limit int, allow PathSet) ([]SimilarHit, error) {
	return searchByVectorScored(db, model, query, limit, allow, embedindex.ScoreDefault)
}

// searchByVectorCrossModal is searchByVectorWithin for a query vector carrying
// a TEXT-encoder component. Such a vector is not drawn from the stored image
// distribution, so the library mean must not be subtracted from it — see
// embedindex.ScorePlain.
func searchByVectorCrossModal(db *sql.DB, model string, query []float32, limit int, allow PathSet) ([]SimilarHit, error) {
	return searchByVectorScored(db, model, query, limit, allow, embedindex.ScorePlain)
}

func searchByVectorScored(db *sql.DB, model string, query []float32, limit int, allow PathSet, s embedindex.Scoring) ([]SimilarHit, error) {
	if raw, ok := indexSearch(model, query, limit, allow, s); ok {
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
	// The fallback always scores plain cosine — which is exactly right for
	// ScorePlain, and (unlike searchSharedWithin) does NOT mirror the centered
	// index for ScoreDefault. Image rankings therefore shift slightly while no
	// index is installed; see the note in searchSharedWithin.
	hits := make([]SimilarHit, 0, len(all))
	for _, e := range all {
		if allow != nil {
			if _, ok := allow[e.Path]; !ok {
				continue
			}
		}
		// CosineSim (not raw dot) so legacy rows stored before the embed binary
		// normalized its output still rank correctly.
		hits = append(hits, SimilarHit{Path: e.Path, Score: embedvec.CosineSim(query, e.Vec)})
	}
	sortSimilar(hits)
	if limit > 0 && len(hits) > limit {
		hits = hits[:limit]
	}
	return hits, nil
}

// sortSimilar orders hits the same way embedindex does — score descending,
// path ascending on ties — so the brute-force fallback and the installed index
// return identical result orders, not just identical scores.
func sortSimilar(hits []SimilarHit) {
	sort.Slice(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return hits[i].Path < hits[j].Path
	})
}

// libraryCenter returns the mean of the already-normalized vecs, or nil when
// there are fewer than embedindex.MinCenterCount of them — the same gate the
// centered index applies, below which the mean is a statistic of almost
// nothing.
func libraryCenter(vecs [][]float32) []float32 {
	if len(vecs) < embedindex.MinCenterCount || len(vecs[0]) == 0 {
		return nil
	}
	mu := make([]float32, len(vecs[0]))
	for _, v := range vecs {
		if len(v) != len(mu) {
			continue
		}
		for d, s := range v {
			mu[d] += s / float32(len(vecs))
		}
	}
	return mu
}

// centerBy returns normalize(v − mu), the unit vector v moved into centered
// scoring space. v is returned unchanged when its dimensionality doesn't match.
func centerBy(v, mu []float32) []float32 {
	if len(v) != len(mu) {
		return v
	}
	c := make([]float32, len(v))
	for d := range v {
		c[d] = v[d] - mu[d]
	}
	return embedvec.Normalize(c)
}

// searchSharedWithin runs a shared-concept ("must match all") multi-example
// search via the installed index, or brute-force over all stored embeddings
// when none matches the model. For an image-only query the brute-force path
// mirrors the centered index: above embedindex.MinCenterCount vectors it
// subtracts the library mean before scoring, so rankings don't depend on
// whether the index happened to be installed. (searchByVectorScored's fallback
// does NOT do this — the two disagree on that point.) A cross-modal query
// (ScorePlain) is never centered on either path.
func searchSharedWithin(db *sql.DB, model string, spec embedindex.SharedSpec, limit int, allow PathSet, s embedindex.Scoring) ([]SimilarHit, error) {
	if raw, ok, err := indexSearchShared(model, spec, limit, allow, s); ok {
		if err != nil {
			return nil, err
		}
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
	// Normalize once (legacy rows may predate normalized embed output), then
	// center when the library is big enough for the mean to be meaningful and
	// the query isn't cross-modal.
	vecs := make([][]float32, len(all))
	for i, e := range all {
		vecs[i] = embedvec.Normalize(e.Vec)
	}
	var mu []float32
	if s != embedindex.ScorePlain {
		mu = libraryCenter(vecs)
	}
	center := func(v []float32) []float32 {
		if mu == nil {
			return v
		}
		return centerBy(v, mu)
	}
	prep := func(vs [][]float32) [][]float32 {
		out := make([][]float32, len(vs))
		for i, v := range vs {
			out[i] = center(embedvec.Normalize(v))
		}
		return out
	}
	sq, err := embedvec.NewSharedQuery(prep(spec.Pos), spec.PosW, prep(spec.Neg), spec.NegW,
		embedvec.DefaultSharedBeta, embedvec.DefaultSharedLambda)
	if err != nil {
		return nil, err
	}
	score := sq.Scorer()
	hits := make([]SimilarHit, 0, len(all))
	for i, e := range all {
		if allow != nil {
			if _, ok := allow[e.Path]; !ok {
				continue
			}
		}
		hits = append(hits, SimilarHit{Path: e.Path, Score: score(center(vecs[i]))})
	}
	sortSimilar(hits)
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
// persists the result so subsequent searches are instant. allow restricts the
// results (not the query item) to a path subset; nil = whole library.
func SimilarByPathOrEmbed(ctx context.Context, db *sql.DB, modelID, path string, limit int, allow PathSet) ([]SimilarHit, error) {
	vec, ok, err := media.GetEmbedding(db, path, modelID)
	if err != nil {
		return nil, err
	}
	if ok {
		return searchByVectorWithin(db, modelID, vec, limit, allow)
	}
	m, found := EmbedModelByID(modelID)
	if !found {
		m = ActiveEmbedModel()
	}
	fresh, err := embedAndPersist(ctx, db, m, path)
	if err != nil {
		return nil, err
	}
	return searchByVectorWithin(db, m.ID, fresh, limit, allow)
}

// embedAndPersist embeds path under m, then persists + indexes the vector so
// future searches over this item are instant.
func embedAndPersist(ctx context.Context, db *sql.DB, m EmbedModel, path string) ([]float32, error) {
	return embedAndPersistAt(ctx, db, m, path, path)
}

// embedAndPersistAt embeds the readable file at filePath but persists (and
// indexes) the vector under dbPath — the two differ for s3:// media, where
// the item is downloaded to a temp file first.
func embedAndPersistAt(ctx context.Context, db *sql.DB, m EmbedModel, dbPath, filePath string) ([]float32, error) {
	fresh, err := embedFileWithModel(ctx, filePath, m)
	if err != nil {
		return nil, fmt.Errorf("embed query item %q: %w", dbPath, err)
	}
	if uerr := media.UpsertEmbedding(db, dbPath, m.ID, fresh, 0); uerr == nil {
		indexAdd(m.ID, dbPath, fresh) // index normalizes internally
	}
	return fresh, nil
}

// ImageQueryVectorForPath returns path's image embedding under m: the stored
// vector when present, otherwise embedded on the fly (and persisted). Used as
// the image half of a blended image+text query.
func ImageQueryVectorForPath(ctx context.Context, db *sql.DB, m EmbedModel, path string) ([]float32, error) {
	return ImageQueryVectorForPathAt(ctx, db, m, path, path)
}

// ImageQueryVectorForPathAt is ImageQueryVectorForPath for media whose
// readable file lives somewhere other than its library path (s3:// items
// downloaded to a temp file): the DB lookup/persist key is dbPath, the
// bytes come from filePath.
func ImageQueryVectorForPathAt(ctx context.Context, db *sql.DB, m EmbedModel, dbPath, filePath string) ([]float32, error) {
	vec, ok, err := media.GetEmbedding(db, dbPath, m.ID)
	if err != nil {
		return nil, err
	}
	if ok {
		return vec, nil
	}
	return embedAndPersistAt(ctx, db, m, dbPath, filePath)
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
	// Independent deadline: query-time embeds run on a REQUEST context, and a
	// client that never disconnects must not be able to wedge this handler
	// (and its socket) forever behind a hung ONNX call.
	ctx, cancel := context.WithTimeout(ctx, OnnxFileTimeout())
	defer cancel()
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
	// Same independent deadline as runEmbedSubprocess: a hung text encode must
	// not pin a request handler forever.
	ctx, cancel := context.WithTimeout(ctx, OnnxFileTimeout())
	defer cancel()
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

// TextQueryVector encodes text with the multimodal text-search model's text
// encoder and returns the vector plus the model it belongs to (which is also
// the image-embedding space the vector must be matched/blended against).
// Returns an error (not a panic) when the model, tokenizer, or embed binary
// is absent.
func TextQueryVector(ctx context.Context, text string) ([]float32, EmbedModel, error) {
	// Text->image search needs a text encoder; resolve the multimodal model
	// (the active model if it's multimodal, otherwise SigLIP 2). Vectors are
	// matched against this model's image embeddings.
	m := TextSearchModel()
	if !m.Multimodal || m.TextModelFile == "" {
		return nil, m, fmt.Errorf("model %q does not support text search", m.ID)
	}
	textModel, err := deps.ModelPath(m.ID, m.TextModelFile)
	if err != nil {
		return nil, m, fmt.Errorf("text model not installed: %w", err)
	}
	if textModel == "" {
		return nil, m, fmt.Errorf("text model not installed")
	}
	tokenizer, err := deps.ModelPath(m.ID, m.TokenizerFile)
	if err != nil {
		return nil, m, fmt.Errorf("tokenizer not installed: %w", err)
	}
	if tokenizer == "" {
		return nil, m, fmt.Errorf("tokenizer not installed")
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, m, fmt.Errorf("embed binary not installed")
	}
	vec, err := runEmbedTextSubprocess(ctx, embedBin, textModel, tokenizer, ortLib, text, m)
	if err != nil {
		return nil, m, err
	}
	return vec, m, nil
}

// SearchByText encodes text via the SigLIP 2 text encoder subprocess and
// returns the top-limit most similar media by cosine similarity, restricted
// to allow when non-nil. Returns an error (not a panic) when the model,
// tokenizer, or embed binary is absent.
func SearchByText(ctx context.Context, db *sql.DB, text string, limit int, allow PathSet) ([]SimilarHit, error) {
	vec, m, err := TextQueryVector(ctx, text)
	if err != nil {
		return nil, err
	}
	return searchByVectorCrossModal(db, m.ID, vec, limit, allow)
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
