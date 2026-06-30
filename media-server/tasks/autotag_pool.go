package tasks

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

// autotagResult is one tagged-or-skipped image handed to the collector.
type autotagResult struct {
	mediaPath string
	tags      []TagInfo
	ok        bool
}

// classifyViaWorker runs one image through an onnxtag --serve worker using the
// length-framed protocol: the worker replies "OK <n>" followed by n tag lines,
// or "ERR <msg>".
func classifyViaWorker(w *serveWorker, imagePath string) ([]string, error) {
	if err := w.writeLine(imagePath); err != nil {
		return nil, err
	}
	head, ok := w.readLine()
	if !ok {
		return nil, fmt.Errorf("worker died: %s", w.stderrString())
	}
	if msg, found := strings.CutPrefix(head, "ERR "); found {
		return nil, fmt.Errorf("%s", msg)
	}
	countStr, found := strings.CutPrefix(head, "OK ")
	if !found {
		return nil, fmt.Errorf("unexpected response %q", head)
	}
	n, err := strconv.Atoi(strings.TrimSpace(countStr))
	if err != nil {
		return nil, fmt.Errorf("bad tag count %q", countStr)
	}
	tags := make([]string, 0, n)
	for i := 0; i < n; i++ {
		line, ok := w.readLine()
		if !ok {
			return nil, fmt.Errorf("worker died mid-response: %s", w.stderrString())
		}
		tags = append(tags, line)
	}
	return tags, nil
}

// tagsToTagInfos parses the worker's "name:score" lines into TagInfo, dropping
// the score suffix and assigning the "Suggested" category (matching the old
// per-image autotag behavior).
func tagsToTagInfos(tags []string) []TagInfo {
	var out []TagInfo
	for _, t := range tags {
		name := t
		if pos := strings.LastIndex(t, ":"); pos > 0 {
			name = t[:pos]
		}
		name = strings.TrimSpace(name)
		if name == "" {
			continue
		}
		out = append(out, TagInfo{Label: name, Category: "Suggested"})
	}
	return out
}

// runAutotagPool tags all paths using a pool of persistent onnxtag workers (the
// 1.26 GB tagger model loads once per worker, not per image). Mirrors
// runEmbedPool. Returns processed/skipped counts.
func runAutotagPool(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool) (int, int, error) {
	cfg := appconfig.Get()
	workers, threads := ResolveAutotagResources()
	ortLib, provider := resolveONNXRuntime(AutotagProviderFromConfig())
	if AutotagProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}

	// Resolve the active tagging model + its files (deps first, config fallback).
	tagger := ActiveTaggerModel()
	q.PushJobStdout(j.ID, fmt.Sprintf("Tagging model: %s", tagger.ID))
	modelPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.ModelFile), cfg.OnnxTagger.ModelPath)
	if modelPath == "" {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s not installed; install it from Dependencies", tagger.DisplayName))
		q.ErrorJob(j.ID)
		return 0, 0, fmt.Errorf("tagger model %s not installed", tagger.ID)
	}
	labelsPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.LabelsFile), cfg.OnnxTagger.LabelsPath)
	configPath := firstNonEmpty(depModelPathOrEmpty(tagger.ID, tagger.ConfigFile), cfg.OnnxTagger.ConfigPath)

	onnxtagBin := deps.BundledOrEmpty("onnxtag")
	if onnxtagBin == "" {
		q.PushJobStdout(j.ID, "onnxtag binary not installed; install it from Dependencies")
		q.ErrorJob(j.ID)
		return 0, 0, fmt.Errorf("onnxtag binary not installed")
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

	if workers > len(paths) {
		workers = len(paths)
	}
	if workers < 1 {
		workers = 1
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Tagging with %d worker(s), %d thread(s) each, provider=%s", workers, threads, provider))

	// Start workers.
	pool := make([]*serveWorker, 0, workers)
	for i := 0; i < workers; i++ {
		wkr, err := startServeWorker(ctx, onnxtagBin, args)
		if err != nil {
			for _, p := range pool {
				p.close()
			}
			return 0, 0, fmt.Errorf("start onnxtag worker: %w", err)
		}
		pool = append(pool, wkr)
	}

	jobs := make(chan string)
	results := make(chan autotagResult, workers*2)

	var processed, skipped int
	var collectorWG sync.WaitGroup
	collectorWG.Add(1)
	go func() {
		defer collectorWG.Done()
		for r := range results {
			if !r.ok || len(r.tags) == 0 {
				skipped++
				continue
			}
			if err := insertTagsForFile(q.Db, r.mediaPath, r.tags); err != nil {
				q.PushJobStdout(j.ID, "  Failed to insert tags: "+err.Error())
				skipped++
				continue
			}
			processed++
			q.RegisterOutputFile(j.ID, r.mediaPath)
			if (processed+skipped)%50 == 0 {
				q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d tagged, %d skipped (of %d)", processed, skipped, len(paths)))
			}
		}
	}()

	timeout := OnnxFileTimeout()
	var workerWG sync.WaitGroup
	for _, wkr := range pool {
		workerWG.Add(1)
		go func(w *serveWorker) {
			defer workerWG.Done()
			defer func() {
				if w != nil {
					w.close()
				}
			}()
			for mediaPath := range jobs {
				if ctx.Err() != nil {
					return
				}
				if !fromQuery {
					if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
						results <- autotagResult{ok: false}
						continue
					}
				}
				imagePath, tempFrame, ferr := extractFrameForFile(ctx, mediaPath, timeout)
				if ferr != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("  frame extract failed/timed out (%s): %v", filepath.Base(mediaPath), ferr))
					results <- autotagResult{ok: false}
					continue
				}
				if w == nil { // a previous restart failed; drain remaining as skips
					if tempFrame != "" {
						_ = os.Remove(tempFrame)
					}
					results <- autotagResult{ok: false}
					continue
				}
				tagStrings, err, timedOut := runWithTimeout(ctx, timeout, func() ([]string, error) { return classifyViaWorker(w, imagePath) })
				if tempFrame != "" {
					_ = os.Remove(tempFrame)
				}
				if timedOut {
					q.PushJobStdout(j.ID, fmt.Sprintf("  timed out after %s, skipping + restarting worker: %s", timeout, filepath.Base(mediaPath)))
					w.kill()
					if nw, rerr := startServeWorker(ctx, onnxtagBin, args); rerr == nil {
						w = nw
					} else {
						q.PushJobStdout(j.ID, "  worker restart failed; its remaining files will be skipped: "+rerr.Error())
						w = nil
					}
					results <- autotagResult{ok: false}
					continue
				}
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					q.PushJobStdout(j.ID, fmt.Sprintf("  tag failed (%s): %v", filepath.Base(mediaPath), err))
					results <- autotagResult{ok: false}
					continue
				}
				results <- autotagResult{mediaPath: mediaPath, tags: tagsToTagInfos(tagStrings), ok: true}
			}
		}(wkr)
	}

	feedDone := make(chan struct{})
	go func() {
		defer close(feedDone)
		defer close(jobs)
		for _, p := range paths {
			if ctx.Err() != nil {
				return
			}
			jobs <- p
		}
	}()

	<-feedDone
	workerWG.Wait()
	close(results)
	collectorWG.Wait()

	if ctx.Err() != nil {
		return processed, skipped, ctx.Err()
	}
	return processed, skipped, nil
}

// depModelPathOrEmpty returns the installed model file path or "" if missing.
func depModelPathOrEmpty(modelID, rel string) string {
	p, _ := deps.ModelPath(modelID, rel)
	return p
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
