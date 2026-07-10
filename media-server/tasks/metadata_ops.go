package tasks

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	_ "golang.org/x/image/bmp"
	xdraw "golang.org/x/image/draw"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/transcribe"
)

func hasExistingMetadata(db *sql.DB, path, metadataType string) (bool, error) {
	query := fmt.Sprintf(`SELECT %s FROM media WHERE path = ?`, metadataType)
	var value sql.NullString
	if err := db.QueryRow(query, path).Scan(&value); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, err
	}
	return value.Valid && value.String != "", nil
}

func hasExistingDimensions(db *sql.DB, path string) (bool, error) {
	var width, height sql.NullInt64
	if err := db.QueryRow(`SELECT width, height FROM media WHERE path = ?`, path).Scan(&width, &height); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, err
	}
	return width.Valid && height.Valid, nil
}

func updateMediaMetadata(db *sql.DB, path, metadataType, value string) error {
	query := fmt.Sprintf(`UPDATE media SET %s = ? WHERE path = ?`, metadataType)
	_, err := db.Exec(query, value, path)
	return err
}

// extractVideoFrame extracts a single representative frame from a video file.
// It seeks to the midpoint of the video (title cards, black intros, and fade-ins
// cluster at the start, so the middle is far more representative than frame 0),
// falling back to the first frame when the duration is unknown or the mid-seek
// yields nothing (e.g. single-frame GIFs).
//
// Parameters:
//   - ctx: context for cancellation
//   - videoPath: absolute path to the video file
//   - outputPath: desired output path (if empty, generates temp file)
//
// Returns:
//   - string: path to the extracted frame (caller is responsible for cleanup)
//   - error: if extraction fails
func extractVideoFrame(ctx context.Context, videoPath string, outputPath string) (string, error) {
	if outputPath == "" {
		outputPath = filepath.Join(os.TempDir(), fmt.Sprintf("video_frame_%s_%d.jpg",
			strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath)),
			time.Now().UnixNano()))
	}

	// Midpoint seek. Duration comes from a header-only probe; 0 means unknown.
	// Sub-second clips just take the first frame.
	seekTime := probeVideoDuration(ctx, videoPath) / 2
	if seekTime < 0.5 {
		seekTime = 0
	}

	err := runFFmpegSingleFrame(ctx, videoPath, outputPath, seekTime)
	if err != nil && seekTime > 0 && ctx.Err() == nil {
		// Mid-seek produced no frame (stream shorter than the container claims,
		// single-frame animations, etc.) — retry from the start.
		err = runFFmpegSingleFrame(ctx, videoPath, outputPath, 0)
	}
	if err != nil {
		return "", err
	}
	return outputPath, nil
}

// runFFmpegSingleFrame extracts one frame at seekTime seconds into outputPath.
// -ss before -i is a fast keyframe-level seek (no decoding up to that point),
// so cost is roughly constant regardless of where in the video we seek.
func runFFmpegSingleFrame(ctx context.Context, videoPath, outputPath string, seekTime float64) error {
	args := []string{
		"-ss", fmt.Sprintf("%.3f", seekTime),
		"-i", videoPath,
		"-frames:v", "1",
		"-q:v", "2",
		"-y",
		outputPath,
	}
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), args...)
	platform.HideSubprocessWindow(cmd)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("ffmpeg frame extraction failed (seek=%.2fs): %w\nOutput: %s",
			seekTime, err, string(output))
	}
	// ffmpeg exits 0 even when the seek lands past the last frame and nothing is
	// written, so verify the output exists.
	if _, statErr := os.Stat(outputPath); statErr != nil {
		return fmt.Errorf("ffmpeg completed but output file not found (seek=%.2fs): %w", seekTime, statErr)
	}
	return nil
}

// probeVideoDuration returns the container duration in seconds from a
// header-only ffprobe (no frame decoding), or 0 when it can't be determined.
func probeVideoDuration(ctx context.Context, videoPath string) float64 {
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffprobe"),
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	if s == "" || s == "N/A" {
		return 0
	}
	d, err := strconv.ParseFloat(s, 64)
	if err != nil || d < 0 {
		return 0
	}
	return d
}

func describeFileWithOllama(ctx context.Context, q *jobqueue.Queue, jobID, mediaPath, model, customPrompt string) (string, error) {
	ext := strings.ToLower(filepath.Ext(mediaPath))
	var tempImagePath string
	var cleanupPaths []string
	source := "image"
	if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp" {
		tempImagePath = mediaPath
	} else {
		screenshotPath, err := extractVideoFrame(ctx, mediaPath, "")
		if err != nil {
			return "", fmt.Errorf("failed to extract video frame: %w", err)
		}
		cleanupPaths = append(cleanupPaths, screenshotPath)
		tempImagePath = screenshotPath
		source = "video-frame:" + filepath.Base(screenshotPath)
	}
	resizedPath, err := resizeImageIfNeeded(tempImagePath)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return "", fmt.Errorf("failed to resize image: %w", err)
	}
	if resizedPath != tempImagePath {
		cleanupPaths = append(cleanupPaths, resizedPath)
	}
	logImageParseToJob(q, jobID, mediaPath, source, tempImagePath, resizedPath)
	description, err := callOllamaVision(ctx, resizedPath, model, customPrompt)
	if err != nil {
		for _, p := range cleanupPaths {
			_ = os.Remove(p)
		}
		return "", fmt.Errorf("ollama call failed: %w", err)
	}
	for _, p := range cleanupPaths {
		_ = os.Remove(p)
	}
	// Guard against a "blind" reply: the backend returned 200 with prose, but
	// the prose says it never got an image. Treat it as a failure so this text
	// is not persisted as the file's description — a silent save here is how the
	// bug hid for so long. No retry (caller decides); the job log shows it.
	if looksLikeNoImageResponse(description) {
		preview := strings.ReplaceAll(description, "\n", " ")
		if len(preview) > 160 {
			preview = preview[:160] + "…"
		}
		return "", fmt.Errorf("model returned a no-image response (image was not ingested by the backend): %q", preview)
	}
	return description, nil
}

// logImageParseToJob pushes one line into the per-job stdout (visible in the
// Jobs UI) confirming the image was located, decoded, and what is about to be
// handed to the vision backend. A healthy run and a "model went in blind" run
// previously produced identical job logs ("description: generated"); this line
// makes the difference inspectable: the true decoded format/dimensions, whether
// the file was resized/re-encoded, and the byte count actually sent. Pair it
// with the [vision:request]/[vision:response] lines in the server log.
func logImageParseToJob(q *jobqueue.Queue, jobID, originalPath, source, framePath, sentPath string) {
	if q == nil {
		return
	}
	var sentBytes int64
	if info, err := os.Stat(sentPath); err == nil {
		sentBytes = info.Size()
	}
	format, cm, w, h := "UNDECODABLE", "", 0, 0
	if f, err := os.Open(sentPath); err == nil {
		if cfg, fm, derr := image.DecodeConfig(f); derr == nil {
			format, cm, w, h = fm, colorModelName(cfg.ColorModel), cfg.Width, cfg.Height
		}
		_ = f.Close()
	}
	normalized := "no"
	if sentPath != framePath {
		normalized = "yes→" + filepath.Base(sentPath)
	}
	q.PushJobStdout(jobID, fmt.Sprintf(
		"  image: %s | source=%s | decoded=%s/%s %dx%d | normalized=%s | bytesSent=%d",
		filepath.Base(originalPath), source, format, cm, w, h, normalized, sentBytes))
}

// resizeImageIfNeeded normalizes an image for a vision backend: it guarantees
// the bytes handed to the model are in a universally-decodable format (PNG) and
// no larger than maxLongSide on the long edge.
//
// The original path is returned unchanged ONLY when the source is already a
// model-safe format (JPEG/PNG) AND within size. Anything else — webp, bmp,
// gif, tiff — is re-encoded to a PNG temp file even when it doesn't need
// resizing, so every backend gets a format it can decode.
//
// The caller is responsible for removing any returned temp file (it differs
// from the input path exactly when re-encoding happened).
func resizeImageIfNeeded(path string) (string, error) {
	const maxLongSide = 1280

	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return "", fmt.Errorf("image decode failed: %w", err)
	}
	b := img.Bounds()
	longSide := b.Dx()
	if b.Dy() > longSide {
		longSide = b.Dy()
	}

	// JPEG and PNG are accepted by every vision backend we target; leave them
	// untouched when they also fit the size budget. Every other decoded format
	// is normalized below.
	safeFormat := format == "jpeg" || format == "png"
	if safeFormat && longSide <= maxLongSide {
		return path, nil
	}

	dst := img
	if longSide > maxLongSide {
		scale := float64(maxLongSide) / float64(longSide)
		w := int(float64(b.Dx()) * scale)
		h := int(float64(b.Dy()) * scale)
		if w < 1 {
			w = 1
		}
		if h < 1 {
			h = 1
		}
		resized := image.NewRGBA(image.Rect(0, 0, w, h))
		xdraw.CatmullRom.Scale(resized, resized.Bounds(), img, b, xdraw.Over, nil)
		dst = resized
	}

	out, err := os.CreateTemp("", "ollama_resize_*.png")
	if err != nil {
		return "", err
	}
	tmpPath := out.Name()
	if err := png.Encode(out, dst); err != nil {
		_ = out.Close()
		_ = os.Remove(tmpPath)
		return "", err
	}
	if err := out.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", err
	}
	return tmpPath, nil
}

// resolveDescribePrompt returns the custom prompt when it has non-whitespace
// content, otherwise falls back to the prompt stored in app config. Extracted
// so the fallback rule can be unit-tested without a live vision backend.
func resolveDescribePrompt(custom string) string {
	if trimmed := strings.TrimSpace(custom); trimmed != "" {
		return trimmed
	}
	return appconfig.Get().DescribePrompt
}

// callOllamaVision routes to either RunPod or the local Ollama HTTP API for
// image description. The 10-minute deadline preserves the upper bound that
// used to live on the per-request http.Client. If customPrompt is non-empty
// (after trimming) it replaces the configured DescribePrompt for this call
// only.
func callOllamaVision(ctx context.Context, imagePath, _ string, customPrompt string) (string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, 600*time.Second)
	defer cancel()
	return callVisionLLM(timeoutCtx, imagePath, resolveDescribePrompt(customPrompt))
}

// generateTranscriptWithFasterWhisper transcribes one file through the
// transcribe facade — the provider (local Faster-Whisper CLI today, possibly
// an HTTP service later) and its model/language/VAD settings come from config.
func generateTranscriptWithFasterWhisper(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string) (string, error) {
	logFn := func(line string) {
		if q != nil && jobID != "" {
			q.PushJobStdout(jobID, "[transcribe] "+line)
		}
	}
	provider, req, err := transcribe.FromConfig(filePath, logFn)
	if err != nil {
		return "", err
	}
	if err := provider.Available(); err != nil {
		return "", err
	}
	res, err := provider.Transcribe(ctx, req)
	if err != nil {
		return "", err
	}
	return res.Text, nil
}

func getImageDimensions(path string) (int, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	cfg, _, err := image.DecodeConfig(f)
	if err != nil {
		return 0, 0, err
	}
	return cfg.Width, cfg.Height, nil
}

func getVideoDimensionsFFProbe(path string) (int, int, error) {
	cmd := exec.Command(deps.MustBundled("ffprobe"), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", path)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	if err != nil {
		return 0, 0, err
	}
	dims := strings.Split(strings.TrimSpace(string(out)), "x")
	if len(dims) != 2 {
		return 0, 0, errors.New("unexpected ffprobe output: " + string(out))
	}
	width, wErr := strconv.Atoi(dims[0])
	height, hErr := strconv.Atoi(dims[1])
	if wErr != nil || hErr != nil {
		return 0, 0, fmt.Errorf("failed to parse width/height from: %s", string(out))
	}
	return width, height, nil
}

func hashFirstNBytes(r io.Reader, n int64) (string, error) {
	if n < 0 {
		return "", errors.New("invalid byte count")
	}
	h := sha256.New()
	if _, err := io.Copy(h, io.LimitReader(r, n)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

func fileExistsInDatabase(db *sql.DB, path string) (bool, error) {
	var exists int
	if err := db.QueryRow(`SELECT 1 FROM media WHERE path = ? LIMIT 1`, path).Scan(&exists); err != nil {
		if err == sql.ErrNoRows {
			return false, nil
		}
		return false, err
	}
	return true, nil
}
