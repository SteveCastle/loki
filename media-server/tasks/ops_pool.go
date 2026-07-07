package tasks

// ops_pool.go — embed and ONNX autotag as ItemOps. Both are backed by pools
// of persistent serve workers (model loaded once per worker); the pool is
// started in Prepare and workers are checked out per Process call, so these
// ops compose with any other op in a single per-file pass while keeping the
// worker-pool performance of the old standalone tasks.

import (
	"context"
	"fmt"
	"os"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/media"
)

// servePool is a fixed-size checkout pool of persistent serve workers.
// A slot holding nil means its worker died (timeout kill) and is lazily
// restarted on the next acquire. background pools run their workers at
// below-normal OS priority (scheduler-initiated jobs).
type servePool struct {
	bin        string
	args       []string
	background bool
	slots      chan *serveWorker
}

func newServePool(ctx context.Context, n int, bin string, args []string, background bool) (*servePool, error) {
	p := &servePool{bin: bin, args: args, background: background, slots: make(chan *serveWorker, n)}
	for i := 0; i < n; i++ {
		w, err := startServeWorker(ctx, bin, args, background)
		if err != nil {
			p.close()
			return nil, err
		}
		p.slots <- w
	}
	return p, nil
}

// acquire checks out a worker, restarting a dead slot if needed.
func (p *servePool) acquire(ctx context.Context) (*serveWorker, error) {
	select {
	case w := <-p.slots:
		if w != nil {
			return w, nil
		}
		nw, err := startServeWorker(ctx, p.bin, p.args, p.background)
		if err != nil {
			p.slots <- nil // keep the slot; the next acquire retries
			return nil, fmt.Errorf("restart worker: %w", err)
		}
		return nw, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// release returns a healthy worker to the pool.
func (p *servePool) release(w *serveWorker) { p.slots <- w }

// discard kills a stuck worker and leaves a dead slot for lazy restart.
func (p *servePool) discard(w *serveWorker) {
	w.kill()
	p.slots <- nil
}

// close shuts down all checked-in workers. Callers must have returned every
// worker (the runner drains its compute workers before closing processors).
func (p *servePool) close() {
	for {
		select {
		case w := <-p.slots:
			if w != nil {
				w.close()
			}
		default:
			return
		}
	}
}

// -----------------------------------------------------------------------------
// Embed op
// -----------------------------------------------------------------------------

func registerEmbedItemOp() {
	RegisterItemOp(ItemOp{
		ID:   "embed",
		Name: "Visual Embedding (ONNX)",
		Options: []TaskOption{
			{Name: "model", Label: "Embedding Model", Type: "string", Description: "Embedding model ID (default: the configured active model)"},
		},
		Concurrency: func() int {
			workers, _ := ResolveEmbedResources()
			return workers
		},
		Prepare: prepareEmbedOp,
	})
}

func prepareEmbedOp(run *ItemRun) (*ItemProcessor, error) {
	q, j := run.Queue, run.Job
	db := q.Db

	// An explicit model option overrides the configured active model — this
	// enables zero-downtime migration: embed the library under a new model in
	// the background while the active model still serves search.
	model := ActiveEmbedModel()
	if id, _ := run.Opts["model"].(string); id != "" {
		if m, known := EmbedModelByID(id); known {
			model = m
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("Unknown embed model %q; using active model %q", id, model.ID))
		}
	}

	imageModel, _ := deps.ModelPath(model.ID, model.ImageModelFile)
	if imageModel == "" {
		return nil, fmt.Errorf("%s not installed; install it from Dependencies", model.DisplayName)
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed; install it from Dependencies")
	}

	_, threads := ResolveEmbedResources()
	ortLib, provider := resolveONNXRuntime(EmbedProviderFromConfig())
	if EmbedProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Embedding model: %s (dim %d), %d worker(s), %d thread(s) each, provider=%s",
		model.ID, model.Dim, run.Workers, threads, provider))

	pool, err := newServePool(j.Ctx, run.Workers, embedBin, buildServeArgs(imageModel, ortLib, model, provider, threads), run.Background)
	if err != nil {
		return nil, fmt.Errorf("start embed worker: %w", err)
	}

	// The live coverage counter tracks the ACTIVE model only; a --model
	// override run (background migration) must not advance it.
	countsForStats := model.ID == ActiveEmbedModel().ID
	timeout := OnnxFileTimeout()

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return media.HasEmbedding(db, path, model.ID) },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			imagePath, tempFrame, ferr := extractFrameForFile(ctx, path, timeout)
			if ferr != nil {
				return nil, fmt.Errorf("frame extract: %w", ferr)
			}
			defer func() {
				if tempFrame != "" {
					_ = os.Remove(tempFrame)
				}
			}()

			w, aerr := pool.acquire(ctx)
			if aerr != nil {
				return nil, aerr
			}
			vec, err, timedOut := runWithTimeout(ctx, timeout, func() ([]float32, error) { return w.embed(imagePath) })
			if timedOut {
				pool.discard(w)
				return nil, fmt.Errorf("timed out after %s", timeout)
			}
			pool.release(w)
			if err != nil {
				return nil, err
			}
			return &ItemCommit{
				Commit: func() error {
					if err := media.UpsertEmbedding(db, path, model.ID, vec, 0); err != nil {
						return err
					}
					indexAdd(model.ID, path, vec) // index normalizes internally
					if countsForStats {
						notifyProgress(ProgressEmbedding, 1)
					}
					return nil
				},
				Detail: "embedded (" + model.ID + ")",
			}, nil
		},
		Close: pool.close,
	}, nil
}

// -----------------------------------------------------------------------------
// ONNX autotag op
// -----------------------------------------------------------------------------

func registerAutotagItemOp() {
	RegisterItemOp(ItemOp{
		ID:   "autotag",
		Name: "Auto Tag (ONNX)",
		Concurrency: func() int {
			workers, _ := ResolveAutotagResources()
			return workers
		},
		Prepare: prepareAutotagOp,
	})
}

// suggestedCategory is the tag category ONNX auto-tagging writes into.
const suggestedCategory = "Suggested"

func prepareAutotagOp(run *ItemRun) (*ItemProcessor, error) {
	q, j := run.Queue, run.Job
	db := q.Db
	cfg := appconfig.Get()

	if err := EnsureCategoryExists(db, suggestedCategory, 0); err != nil {
		return nil, fmt.Errorf("ensure category: %w", err)
	}

	_, threads := ResolveAutotagResources()
	ortLib, provider := resolveONNXRuntime(AutotagProviderFromConfig())
	if AutotagProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}

	// Resolve the active tagging model + its files (deps first, config fallback).
	tagger := ActiveTaggerModel()
	modelPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.ModelFile), cfg.OnnxTagger.ModelPath)
	if modelPath == "" {
		return nil, fmt.Errorf("%s not installed; install it from Dependencies", tagger.DisplayName)
	}
	labelsPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.LabelsFile), cfg.OnnxTagger.LabelsPath)
	configPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.ConfigFile), cfg.OnnxTagger.ConfigPath)
	onnxtagBin := deps.BundledOrEmpty("onnxtag")
	if onnxtagBin == "" {
		return nil, fmt.Errorf("onnxtag binary not installed; install it from Dependencies")
	}

	args := []string{"--serve", "--model=" + modelPath, "--provider=" + provider}
	if labelsPath != "" {
		args = append(args, "--labels="+labelsPath)
	}
	if configPath != "" {
		args = append(args, "--config="+configPath)
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	if cfg.OnnxTagger.GeneralThreshold > 0 {
		args = append(args, fmt.Sprintf("--general-thresh=%g", cfg.OnnxTagger.GeneralThreshold))
	}
	if cfg.OnnxTagger.CharacterThreshold > 0 {
		args = append(args, fmt.Sprintf("--character-thresh=%g", cfg.OnnxTagger.CharacterThreshold))
	}
	if threads > 0 {
		args = append(args, fmt.Sprintf("--threads=%d", threads))
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Tagging model: %s, %d worker(s), %d thread(s) each, provider=%s",
		tagger.ID, run.Workers, threads, provider))

	pool, err := newServePool(j.Ctx, run.Workers, onnxtagBin, args, run.Background)
	if err != nil {
		return nil, fmt.Errorf("start onnxtag worker: %w", err)
	}

	overwrite := run.Overwrite
	timeout := OnnxFileTimeout()

	return &ItemProcessor{
		SkipExisting: func(path string) (bool, error) { return hasSuggestedTags(db, path) },
		Process: func(ctx context.Context, path string) (*ItemCommit, error) {
			imagePath, tempFrame, ferr := extractFrameForFile(ctx, path, timeout)
			if ferr != nil {
				return nil, fmt.Errorf("frame extract: %w", ferr)
			}
			defer func() {
				if tempFrame != "" {
					_ = os.Remove(tempFrame)
				}
			}()

			w, aerr := pool.acquire(ctx)
			if aerr != nil {
				return nil, aerr
			}
			tagStrings, err, timedOut := runWithTimeout(ctx, timeout, func() ([]string, error) { return classifyViaWorker(w, imagePath) })
			if timedOut {
				pool.discard(w)
				return nil, fmt.Errorf("timed out after %s", timeout)
			}
			pool.release(w)
			if err != nil {
				return nil, err
			}
			tags := tagsToTagInfos(tagStrings)
			if len(tags) == 0 {
				return nil, nil // nothing above threshold — nothing to write
			}
			return &ItemCommit{
				Commit: func() error {
					if overwrite {
						if err := removeSuggestedTagsForFile(db, path); err != nil {
							return fmt.Errorf("remove suggested tags: %w", err)
						}
					}
					return insertTagsForFile(db, path, tags)
				},
				Detail: fmt.Sprintf("%d tag(s) suggested", len(tags)),
			}, nil
		},
		Close: pool.close,
	}, nil
}
