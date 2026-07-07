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
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/embedvec"
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

// ExtractFrameForMedia returns a decodable image path for any media item: the
// path itself for images, or a freshly-extracted temp frame for videos —
// the SAME deterministic midpoint frame the scan tasks analyze, so relative
// face bboxes recorded at scan time line up with it. tempFrame (when non-"")
// is the caller's to delete. Exported for the face-crop endpoint.
func ExtractFrameForMedia(ctx context.Context, mediaPath string) (imagePath, tempFrame string, err error) {
	return extractFrameForFile(ctx, mediaPath, 30*time.Second)
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

// serveWorker is one persistent embed.exe --serve subprocess plus its pipes.
type serveWorker struct {
	cmd    *exec.Cmd
	stdin  *bufio.Writer
	stdinC interface{ Close() error }
	stdout *bufio.Scanner
	stderr *strings.Builder
}

// startServeWorker launches one persistent worker and waits for its READY
// line. background workers run at below-normal OS priority so the machine's
// foreground work (games, the user's apps) always wins the CPU — used for
// scheduler-initiated jobs.
func startServeWorker(ctx context.Context, embedBin string, args []string, background bool) (*serveWorker, error) {
	cmd := exec.CommandContext(ctx, embedBin, args...)
	platform.HideSubprocessWindow(cmd)
	if background {
		platform.SetBackgroundPriority(cmd)
	}
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
	if background {
		platform.DeprioritizeStarted(cmd.Process.Pid)
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
