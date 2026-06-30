package tasks

import (
	"bufio"
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
)

// runWithTimeout runs fn in a goroutine and waits up to d for it to finish.
// timedOut is true when d elapses first — fn's goroutine stays blocked until the
// caller terminates whatever it's waiting on (e.g. kills the worker subprocess).
// d <= 0 disables the timeout. Also returns on ctx cancellation.
func runWithTimeout[T any](ctx context.Context, d time.Duration, fn func() (T, error)) (val T, err error, timedOut bool) {
	type res struct {
		v T
		e error
	}
	ch := make(chan res, 1) // buffered so the goroutine never blocks after we leave
	go func() { v, e := fn(); ch <- res{v, e} }()

	if d <= 0 {
		select {
		case <-ctx.Done():
			return val, ctx.Err(), false
		case r := <-ch:
			return r.v, r.e, false
		}
	}
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return val, ctx.Err(), false
	case <-timer.C:
		return val, nil, true
	case r := <-ch:
		return r.v, r.e, false
	}
}

// extractFrameForFile returns the path to feed the model plus a temp-frame path
// to delete afterward (empty for non-video). Video frame extraction is bounded
// by a per-file timeout so a corrupt video can't hang the worker indefinitely.
func extractFrameForFile(ctx context.Context, mediaPath string, timeout time.Duration) (imagePath, tempFrame string, err error) {
	switch strings.ToLower(filepath.Ext(mediaPath)) {
	case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".gif":
		fctx := ctx
		if timeout > 0 {
			var cancel context.CancelFunc
			fctx, cancel = context.WithTimeout(ctx, timeout)
			defer cancel()
		}
		fp, e := extractVideoFrame(fctx, mediaPath, "")
		if e != nil {
			return "", "", e
		}
		return fp, fp, nil
	}
	return mediaPath, "", nil
}

// directMLRuntimeLib returns the path to the optional DirectML onnxruntime.dll
// if it is installed, else "". The GPU runtime (onnxruntime.dll + DirectML.dll)
// is downloaded into <DataDir>/runtimes/onnxruntime-directml/.
func directMLRuntimeLib() string {
	p := filepath.Join(platform.GetDataDir(), "runtimes", "onnxruntime-directml", "onnxruntime"+platform.SharedLibExtension())
	if info, err := os.Stat(p); err == nil && !info.IsDir() {
		return p
	}
	return ""
}

// embedFileWithModel computes an embedding for a single media file using model
// m's image encoder (extracting a video frame first when needed). Used to embed
// a "find similar" query item on the fly when it isn't indexed yet.
func embedFileWithModel(ctx context.Context, path string, m EmbedModel) ([]float32, error) {
	imageModel, err := deps.ModelPath(m.ID, m.ImageModelFile)
	if err != nil || imageModel == "" {
		return nil, fmt.Errorf("%s image model not installed: %w", m.DisplayName, err)
	}
	embedBin := deps.BundledOrEmpty("embed")
	if embedBin == "" {
		return nil, fmt.Errorf("embed binary not installed")
	}
	ortLib := deps.BundledOrEmpty("onnxruntime")

	imagePath, tempFrame, ferr := extractFrameForFile(ctx, path, 0)
	if ferr != nil {
		return nil, fmt.Errorf("extract frame from %q: %w", path, ferr)
	}
	vec, err := runEmbedSubprocess(ctx, embedBin, imageModel, ortLib, imagePath, m)
	if tempFrame != "" {
		_ = os.Remove(tempFrame)
	}
	return vec, err
}

// DirectMLRuntimeInstalled reports whether the optional GPU (DirectML) runtime
// is present in the data dir.
func DirectMLRuntimeInstalled() bool { return directMLRuntimeLib() != "" }

// DirectMLRuntimeDir is the directory the GPU runtime is installed into.
func DirectMLRuntimeDir() string {
	return filepath.Join(platform.GetDataDir(), "runtimes", "onnxruntime-directml")
}

// resolveONNXRuntime picks the onnxruntime shared library and the effective
// provider for any local ONNX task (embed, autotag). DirectML is used only when
// requested AND its runtime is installed; otherwise it falls back to the bundled
// CPU runtime.
func resolveONNXRuntime(requested string) (ortLib, provider string) {
	if requested == "directml" {
		if lib := directMLRuntimeLib(); lib != "" {
			return lib, "directml"
		}
		// Requested GPU but runtime missing → CPU fallback (caller logs).
	}
	return deps.BundledOrEmpty("onnxruntime"), "cpu"
}

// embedResult is one stored-or-failed embedding handed to the collector.
type embedResult struct {
	mediaPath string
	vec       []float32
	ok        bool
}

// serveWorker is one persistent embed.exe --serve subprocess plus its pipes.
type serveWorker struct {
	cmd    *exec.Cmd
	stdin  *bufio.Writer
	stdinC interface{ Close() error }
	stdout *bufio.Scanner
	stderr *strings.Builder
}

// startServeWorker launches one persistent worker and waits for its READY line.
func startServeWorker(ctx context.Context, embedBin string, args []string) (*serveWorker, error) {
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	stdinPipe, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	var stderr strings.Builder
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	sc := bufio.NewScanner(stdoutPipe)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024) // base64 vectors can be large
	// Wait for READY (the worker emits it once the model has loaded).
	if !sc.Scan() {
		_ = cmd.Wait()
		return nil, fmt.Errorf("embed worker exited before ready: %s", strings.TrimSpace(stderr.String()))
	}
	if line := strings.TrimSpace(sc.Text()); line != "READY" {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, fmt.Errorf("embed worker unexpected handshake %q: %s", line, strings.TrimSpace(stderr.String()))
	}
	return &serveWorker{
		cmd:    cmd,
		stdin:  bufio.NewWriter(stdinPipe),
		stdinC: stdinPipe,
		stdout: sc,
		stderr: &stderr,
	}, nil
}

// embed sends one image path and reads back the vector (or an error).
func (w *serveWorker) embed(imagePath string) ([]float32, error) {
	if _, err := w.stdin.WriteString(imagePath + "\n"); err != nil {
		return nil, err
	}
	if err := w.stdin.Flush(); err != nil {
		return nil, err
	}
	if !w.stdout.Scan() {
		return nil, fmt.Errorf("embed worker died: %s", strings.TrimSpace(w.stderr.String()))
	}
	line := strings.TrimSpace(w.stdout.Text())
	if strings.HasPrefix(line, "ERR ") {
		return nil, fmt.Errorf("%s", strings.TrimPrefix(line, "ERR "))
	}
	raw, err := base64.StdEncoding.DecodeString(line)
	if err != nil {
		return nil, fmt.Errorf("decode vector: %w", err)
	}
	return embedvec.Decode(raw)
}

// writeLine sends one request line to the worker's stdin and flushes.
func (w *serveWorker) writeLine(s string) error {
	if _, err := w.stdin.WriteString(s + "\n"); err != nil {
		return err
	}
	return w.stdin.Flush()
}

// readLine reads one response line from the worker's stdout.
func (w *serveWorker) readLine() (string, bool) {
	if !w.stdout.Scan() {
		return "", false
	}
	return strings.TrimSpace(w.stdout.Text()), true
}

// stderrString returns whatever the worker has written to stderr (diagnostics).
func (w *serveWorker) stderrString() string { return strings.TrimSpace(w.stderr.String()) }

// kill force-terminates a stuck/timed-out worker and reaps it. Unlike close()
// (which relies on the process exiting on stdin EOF), kill works when the worker
// is blocked in compute/decode and not reading stdin.
func (w *serveWorker) kill() {
	if w.cmd != nil && w.cmd.Process != nil {
		_ = w.cmd.Process.Kill()
	}
	if w.cmd != nil {
		_ = w.cmd.Wait()
	}
}

// close shuts the worker down (closing stdin → EOF → process exits).
func (w *serveWorker) close() {
	_ = w.stdinC.Close()
	_ = w.cmd.Wait()
}

// runEmbedPool embeds all paths using a pool of persistent worker processes,
// storing each result under model.ID. It returns counts of processed/skipped.
// The model is loaded once per worker (not per image), which is the dominant
// speedup over the old spawn-per-image path.
func runEmbedPool(ctx context.Context, j *jobqueue.Job, q *jobqueue.Queue, paths []string, fromQuery bool, model EmbedModel, imageModel, embedBin string) (int, int, error) {
	workers, threads := ResolveEmbedResources()
	ortLib, provider := resolveONNXRuntime(EmbedProviderFromConfig())
	if EmbedProviderFromConfig() == "directml" && provider != "directml" {
		q.PushJobStdout(j.ID, "DirectML runtime not installed; falling back to CPU. Install it from Dependencies.")
	}
	if workers > len(paths) {
		workers = len(paths)
	}
	if workers < 1 {
		workers = 1
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("Embedding with %d worker(s), %d thread(s) each, provider=%s", workers, threads, provider))

	baseArgs := buildServeArgs(imageModel, ortLib, model, provider, threads)

	// Start the worker pool. If a worker fails to start, abort.
	pool := make([]*serveWorker, 0, workers)
	for i := 0; i < workers; i++ {
		wkr, err := startServeWorker(ctx, embedBin, baseArgs)
		if err != nil {
			for _, p := range pool {
				p.close()
			}
			return 0, 0, fmt.Errorf("start embed worker: %w", err)
		}
		pool = append(pool, wkr)
	}

	jobs := make(chan string)
	results := make(chan embedResult, workers*2)

	// Collector: single goroutine owns all DB writes (avoids sqlite write
	// contention) and ANN index inserts.
	var processed, skipped int
	var collectorWG sync.WaitGroup
	collectorWG.Add(1)
	go func() {
		defer collectorWG.Done()
		for r := range results {
			if !r.ok {
				skipped++
				continue
			}
			if err := media.UpsertEmbedding(q.Db, r.mediaPath, model.ID, r.vec, 0); err != nil {
				q.PushJobStdout(j.ID, "  Failed to store embedding: "+err.Error())
				skipped++
				continue
			}
			indexAdd(model.ID, r.mediaPath, embedvec.Normalize(r.vec))
			q.RegisterOutputFile(j.ID, r.mediaPath)
			processed++
			if (processed+skipped)%50 == 0 {
				q.PushJobStdout(j.ID, fmt.Sprintf("Progress: %d embedded, %d skipped (of %d)", processed, skipped, len(paths)))
			}
		}
	}()

	// Workers: each owns one subprocess and pulls paths off the channel.
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
				if shouldSkipEmbed(q.Db, mediaPath, model.ID) {
					results <- embedResult{ok: false}
					continue
				}
				if !fromQuery {
					if _, err := os.Stat(mediaPath); os.IsNotExist(err) {
						results <- embedResult{ok: false}
						continue
					}
				}
				imagePath, tempFrame, ferr := extractFrameForFile(ctx, mediaPath, timeout)
				if ferr != nil {
					q.PushJobStdout(j.ID, fmt.Sprintf("  frame extract failed/timed out (%s): %v", filepath.Base(mediaPath), ferr))
					results <- embedResult{ok: false}
					continue
				}
				if w == nil { // a previous restart failed; can't process — drain as skips
					if tempFrame != "" {
						_ = os.Remove(tempFrame)
					}
					results <- embedResult{ok: false}
					continue
				}
				vec, err, timedOut := runWithTimeout(ctx, timeout, func() ([]float32, error) { return w.embed(imagePath) })
				if tempFrame != "" {
					_ = os.Remove(tempFrame)
				}
				if timedOut {
					q.PushJobStdout(j.ID, fmt.Sprintf("  timed out after %s, skipping + restarting worker: %s", timeout, filepath.Base(mediaPath)))
					w.kill()
					if nw, rerr := startServeWorker(ctx, embedBin, baseArgs); rerr == nil {
						w = nw
					} else {
						q.PushJobStdout(j.ID, "  worker restart failed; its remaining files will be skipped: "+rerr.Error())
						w = nil
					}
					results <- embedResult{ok: false}
					continue
				}
				if err != nil {
					if ctx.Err() != nil {
						return
					}
					q.PushJobStdout(j.ID, fmt.Sprintf("  embed failed (%s): %v", filepath.Base(mediaPath), err))
					results <- embedResult{ok: false}
					continue
				}
				results <- embedResult{mediaPath: mediaPath, vec: vec, ok: true}
			}
		}(wkr)
	}

	// Feed paths (stops early on cancel).
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

// buildServeArgs assembles the `embed.exe --serve` arguments for a model +
// provider + thread config (mirrors runEmbedSubprocess's one-shot args).
func buildServeArgs(imageModelPath, ortLib string, m EmbedModel, provider string, threads int) []string {
	args := []string{
		"--serve",
		"--model=" + imageModelPath,
		fmt.Sprintf("--dim=%d", m.Dim),
		"--input=" + m.ImgInput,
		"--output=" + m.ImgOutput,
		fmt.Sprintf("--width=%d", m.Width),
		fmt.Sprintf("--height=%d", m.Height),
		fmt.Sprintf("--mean=%g,%g,%g", m.Mean[0], m.Mean[1], m.Mean[2]),
		fmt.Sprintf("--std=%g,%g,%g", m.Std[0], m.Std[1], m.Std[2]),
		"--provider=" + provider,
	}
	if m.CropPct > 0 && m.CropPct < 1 {
		args = append(args, fmt.Sprintf("--crop-pct=%g", m.CropPct), "--crop-mode="+m.CropMode)
	}
	if m.Pooling != "" {
		args = append(args, "--pooling="+m.Pooling)
	}
	if threads > 0 {
		args = append(args, fmt.Sprintf("--threads=%d", threads))
	}
	if ortLib != "" {
		args = append(args, "--ort="+ortLib)
	}
	return args
}
