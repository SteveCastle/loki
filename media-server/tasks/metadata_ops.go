package tasks

import (
	"bufio"
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
	"log"
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
)

// generateDescriptions generates descriptions for media files using Ollama
func generateDescriptions(ctx context.Context, q *jobqueue.Queue, jobID string, filePaths []string, overwrite bool, model string) error {
	// Pre-filter to compute exact candidates and total for progress
	var candidates []string
	for _, filePath := range filePaths {
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: file does not exist: %s", filePath))
			continue
		}
		if !overwrite {
			hasDescription, err := hasExistingMetadata(q.Db, filePath, "description")
			if err != nil {
				log.Printf("Error checking existing description for %s: %v", filePath, err)
				continue
			}
			if hasDescription {
				continue
			}
		}
		candidates = append(candidates, filePath)
	}
	if len(candidates) == 0 {
		q.PushJobStdout(jobID, "Description: 0 files to process")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Description: %d files to process", len(candidates)))

	processed := 0
	for i, filePath := range candidates {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		description, err := describeFileWithOllama(ctx, q, jobID, filePath, model, "")
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to describe %s: %v", filePath, err))
			continue
		}
		if err := updateMediaMetadata(q.Db, filePath, "description", description); err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to update description for %s: %v", filePath, err))
			continue
		}
		processed++
		q.PushJobStdout(jobID, fmt.Sprintf("Description %d/%d: %s", i+1, len(candidates), filepath.Base(filePath)))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Generated descriptions for %d files", processed))
	return nil
}

// generateTranscripts generates transcripts for video files using faster-whisper
func generateTranscripts(ctx context.Context, q *jobqueue.Queue, jobID string, filePaths []string, overwrite bool) error {
	// Pre-filter to compute exact candidates and total for progress
	var candidates []string
	for _, filePath := range filePaths {
		ext := strings.ToLower(filepath.Ext(filePath))
		switch ext {
		case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv":
		default:
			continue
		}
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: file does not exist: %s", filePath))
			continue
		}
		if !overwrite {
			hasTranscript, err := hasExistingMetadata(q.Db, filePath, "transcript")
			if err != nil {
				log.Printf("Error checking existing transcript for %s: %v", filePath, err)
				continue
			}
			if hasTranscript {
				continue
			}
		}
		candidates = append(candidates, filePath)
	}
	if len(candidates) == 0 {
		q.PushJobStdout(jobID, "Transcript: 0 video files to process")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Transcript: %d video files to process", len(candidates)))

	processed := 0
	for i, filePath := range candidates {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		transcript, err := generateTranscriptWithFasterWhisper(ctx, q, jobID, filePath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to transcribe %s: %v", filePath, err))
			continue
		}
		if err := updateMediaMetadata(q.Db, filePath, "transcript", transcript); err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to update transcript for %s: %v", filePath, err))
			continue
		}
		processed++
		q.PushJobStdout(jobID, fmt.Sprintf("Transcript %d/%d: %s", i+1, len(candidates), filepath.Base(filePath)))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Generated transcripts for %d video files", processed))
	return nil
}

// generateHashes generates hashes for media files
func generateHashes(ctx context.Context, q *jobqueue.Queue, jobID string, filePaths []string, overwrite bool) error {
	const maxBytes = 3 * 1024 * 1024
	// Pre-filter to compute exact candidates and total for progress
	var candidates []string
	for _, filePath := range filePaths {
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: file does not exist: %s", filePath))
			continue
		}
		if !overwrite {
			hasHash, err := hasExistingMetadata(q.Db, filePath, "hash")
			if err != nil {
				log.Printf("Error checking existing hash for %s: %v", filePath, err)
				continue
			}
			if hasHash {
				continue
			}
		}
		candidates = append(candidates, filePath)
	}
	if len(candidates) == 0 {
		q.PushJobStdout(jobID, "Hash: 0 files to process")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Hash: %d files to process", len(candidates)))

	processed := 0
	for i, filePath := range candidates {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		fi, err := os.Stat(filePath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to stat %s: %v", filePath, err))
			continue
		}
		file, err := os.Open(filePath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to open %s: %v", filePath, err))
			continue
		}
		hashVal, err := hashFirstNBytes(file, maxBytes)
		file.Close()
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to hash %s: %v", filePath, err))
			continue
		}
		stmt := `UPDATE media SET hash = ?, size = ? WHERE path = ?`
		_, err = q.Db.Exec(stmt, hashVal, fi.Size(), filePath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to update hash for %s: %v", filePath, err))
			continue
		}
		processed++
		q.PushJobStdout(jobID, fmt.Sprintf("Hash %d/%d: %s", i+1, len(candidates), filepath.Base(filePath)))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Generated hashes for %d files", processed))
	return nil
}

// generateDimensions generates width/height dimensions for media files
func generateDimensions(ctx context.Context, q *jobqueue.Queue, jobID string, filePaths []string, overwrite bool) error {
	// Pre-filter to compute exact candidates and total for progress
	var candidates []string
	for _, filePath := range filePaths {
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: file does not exist: %s", filePath))
			continue
		}
		if !overwrite {
			hasDimensions, err := hasExistingDimensions(q.Db, filePath)
			if err != nil {
				log.Printf("Error checking existing dimensions for %s: %v", filePath, err)
				continue
			}
			if hasDimensions {
				continue
			}
		}
		ext := strings.ToLower(filepath.Ext(filePath))
		switch ext {
		case ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic", ".mp4", ".mov", ".avi", ".mkv", ".webm":
			candidates = append(candidates, filePath)
		}
	}
	if len(candidates) == 0 {
		q.PushJobStdout(jobID, "Dimensions: 0 files to process")
		return nil
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Dimensions: %d files to process", len(candidates)))

	processed := 0
	for i, filePath := range candidates {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		ext := strings.ToLower(filepath.Ext(filePath))
		var width, height int
		var err error
		switch ext {
		case ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic":
			width, height, err = getImageDimensions(filePath)
		case ".mp4", ".mov", ".avi", ".mkv", ".webm":
			width, height, err = getVideoDimensionsFFProbe(filePath)
		default:
			continue
		}
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to get dimensions for %s: %v", filePath, err))
			continue
		}
		_, err = q.Db.Exec(`UPDATE media SET width = ?, height = ? WHERE path = ?`, width, height, filePath)
		if err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: failed to update dimensions for %s: %v", filePath, err))
			continue
		}
		processed++
		q.PushJobStdout(jobID, fmt.Sprintf("Dimensions %d/%d: %s", i+1, len(candidates), filepath.Base(filePath)))
	}
	q.PushJobStdout(jobID, fmt.Sprintf("Generated dimensions for %d files", processed))
	return nil
}

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

// extractVideoFrame extracts a single frame from a video file using ffmpeg.
// It intelligently seeks to a representative frame (avoiding black intros) and handles edge cases:
// - Short videos: seeks proportionally (10% duration, minimum 0.1s)
// - Single-frame GIFs: extracts the only frame
// - Very short videos: extracts first available frame
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
	// Get video duration and frame count using ffprobe
	duration, frameCount, err := getVideoMetadata(ctx, videoPath)
	if err != nil {
		return "", fmt.Errorf("failed to probe video metadata: %w", err)
	}

	// Generate output path if not provided
	if outputPath == "" {
		outputPath = filepath.Join(os.TempDir(), fmt.Sprintf("video_frame_%s_%d.jpg",
			strings.TrimSuffix(filepath.Base(videoPath), filepath.Ext(videoPath)),
			time.Now().UnixNano()))
	}

	// Determine optimal seek time based on video characteristics
	var seekTime float64
	if frameCount <= 1 {
		// Single-frame video or GIF - extract the only frame
		seekTime = 0
	} else if duration < 1.0 {
		// Very short video (< 1 second) - seek to 10% or 0.1s, whichever is smaller
		seekTime = duration * 0.1
		if seekTime < 0.1 {
			seekTime = 0
		}
	} else if duration < 5.0 {
		// Short video (1-5 seconds) - seek to 1 second or 20% duration
		seekTime = 1.0
		if duration*0.2 > 1.0 {
			seekTime = duration * 0.2
		}
	} else {
		// Normal video - seek to 3 seconds or 10% duration, whichever is larger (avoiding intros)
		seekTime = 3.0
		if duration*0.1 > seekTime {
			seekTime = duration * 0.1
		}
		// Cap at 30 seconds to avoid seeking too far in very long videos
		if seekTime > 30.0 {
			seekTime = 30.0
		}
	}

	// Build ffmpeg command with optimized parameters
	// -ss before -i for fast seeking
	// -frames:v 1 to extract exactly one frame
	// -q:v 2 for high quality JPEG (scale 2-31, lower is better)
	// -y to overwrite without asking
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
		return "", fmt.Errorf("ffmpeg frame extraction failed (seek=%.2fs, duration=%.2fs, frames=%d): %w\nOutput: %s",
			seekTime, duration, frameCount, err, string(output))
	}

	// Verify output file was created
	if _, statErr := os.Stat(outputPath); statErr != nil {
		return "", fmt.Errorf("ffmpeg completed but output file not found: %w", statErr)
	}

	return outputPath, nil
}

// getVideoMetadata retrieves duration and frame count from a video file using ffprobe
func getVideoMetadata(ctx context.Context, videoPath string) (duration float64, frameCount int, err error) {
	// Query duration
	durationCmd := exec.CommandContext(ctx, deps.MustBundled("ffprobe"),
		"-v", "error",
		"-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath)
	platform.HideSubprocessWindow(durationCmd)

	durationOut, err := durationCmd.Output()
	if err != nil {
		return 0, 0, fmt.Errorf("ffprobe duration query failed: %w", err)
	}

	durationStr := strings.TrimSpace(string(durationOut))
	if durationStr != "" && durationStr != "N/A" {
		duration, err = strconv.ParseFloat(durationStr, 64)
		if err != nil {
			// If parsing fails, try to extract from format
			duration = 1.0 // Default fallback
		}
	} else {
		duration = 1.0 // Default for files without duration metadata
	}

	// Get frame count
	frameCmd := exec.CommandContext(ctx, deps.MustBundled("ffprobe"),
		"-v", "error",
		"-select_streams", "v:0",
		"-count_frames",
		"-show_entries", "stream=nb_read_frames",
		"-of", "default=noprint_wrappers=1:nokey=1",
		videoPath)
	platform.HideSubprocessWindow(frameCmd)

	frameOut, err := frameCmd.Output()
	if err == nil {
		frameStr := strings.TrimSpace(string(frameOut))
		if frameStr != "" && frameStr != "N/A" {
			frameCount, _ = strconv.Atoi(frameStr)
		}
	}

	// If frame count is unavailable, estimate from duration and assume 24fps minimum
	if frameCount == 0 && duration > 0 {
		frameCount = int(duration * 24) // Conservative estimate
		if frameCount == 0 {
			frameCount = 1 // At least one frame
		}
	}

	return duration, frameCount, nil
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

func generateTranscriptWithFasterWhisper(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string) (string, error) {
	// Faster-Whisper is not bundled in this build. Caller falls back to user-configured path.
	exePath := deps.BundledOrEmpty("faster-whisper")
	var err error
	if exePath == "" {
		err = fmt.Errorf("faster-whisper not bundled; configure FasterWhisperPath in settings to enable transcription")
	}
	if err != nil {
		// Fall back to config if dependency system doesn't have it
		if q != nil && jobID != "" {
			q.PushJobStdout(jobID, fmt.Sprintf("[whisper] dependency lookup failed: %v; falling back to config FasterWhisperPath", err))
		}
		exePath = appconfig.Get().FasterWhisperPath
		if strings.TrimSpace(exePath) == "" {
			return "", fmt.Errorf("faster-whisper not found: dependency not installed and FasterWhisperPath not configured. Please install faster-whisper from the Dependencies page")
		}
	}

	// --vad_filter trims non-speech, which dramatically reduces hallucinations
	// during silent stretches in long clips. --language=en skips the
	// (often-wrong on silent openings) auto-detect — change if non-English
	// content needs supporting.
	args := []string{
		"--beep_off",
		"--output_format=vtt",
		"--output_dir=source",
		// faster-whisper itself warns that large-v3 can produce worse
		// results than large-v2 on general content (more hallucinations,
		// some accents regressed). Stick with v2.
		"--model", "large-v2",
		"--vad_filter", "true",
		"--language", "en",
		filePath,
	}
	cmd := exec.CommandContext(ctx, exePath, args...)
	platform.HideSubprocessWindow(cmd)

	// Pipe both stdout and stderr line-by-line into the job stream so failures
	// surface in the job's output (rather than disappearing into a dropped
	// process buffer). faster-whisper-xxl writes progress and errors to
	// stderr — without this we'd lose the actual reason for a failed run.
	pushLine := func(line string) {
		if q != nil && jobID != "" {
			q.PushJobStdout(jobID, "[whisper] "+line)
		}
	}
	pushLine(fmt.Sprintf("running: %s %s", exePath, strings.Join(args, " ")))

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return "", fmt.Errorf("faster-whisper-xxl: stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return "", fmt.Errorf("faster-whisper-xxl: stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("faster-whisper-xxl: start: %w", err)
	}

	scanReader := func(r io.Reader) {
		scanner := bufio.NewScanner(r)
		// Whisper progress lines can be long; bump the buffer so we don't drop them.
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			pushLine(scanner.Text())
		}
	}
	go scanReader(stdout)
	go scanReader(stderr)

	waitErr := cmd.Wait()

	vttPath := filePath[:len(filePath)-len(filepath.Ext(filePath))] + ".vtt"

	// Trust the artifact, not the exit code. faster-whisper-xxl is a
	// PyInstaller-bundled binary that on Windows sometimes returns
	// 0xc0000409 (STATUS_STACK_BUFFER_OVERRUN) AFTER all transcription
	// work is complete and the .vtt is written — a known teardown crash
	// in the bundled CRT/CUDA runtime, not a transcription failure. If
	// the expected output is on disk, treat the run as success regardless
	// of the exit code.
	if stat, statErr := os.Stat(vttPath); statErr == nil && stat.Size() > 0 {
		if waitErr != nil {
			var exitErr *exec.ExitError
			if errors.As(waitErr, &exitErr) {
				pushLine(fmt.Sprintf("exited with code %d but VTT is present (%d bytes); treating as success", exitErr.ExitCode(), stat.Size()))
			} else {
				pushLine(fmt.Sprintf("wait error %v but VTT is present (%d bytes); treating as success", waitErr, stat.Size()))
			}
		} else {
			pushLine("transcription complete; reading " + vttPath)
		}
		return readFileAll(vttPath)
	}

	// No VTT produced — this is a real failure.
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			pushLine(fmt.Sprintf("exited with code %d, no VTT produced", exitErr.ExitCode()))
			return "", fmt.Errorf("faster-whisper-xxl exited with code %d (no VTT produced): %w", exitErr.ExitCode(), waitErr)
		}
		return "", fmt.Errorf("faster-whisper-xxl failed (no VTT produced): %w", waitErr)
	}
	return "", fmt.Errorf("faster-whisper-xxl exited cleanly but no VTT was produced at %s", vttPath)
}

func readFileAll(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	var sb strings.Builder
	s := bufio.NewScanner(f)
	for s.Scan() {
		sb.WriteString(s.Text())
		sb.WriteByte('\n')
	}
	if scanErr := s.Err(); scanErr != nil {
		return "", scanErr
	}
	return sb.String(), nil
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

// Single-file processing functions for the per-file metadata task flow
// fromQuery parameter: if true, file came from database query so skip DB existence checks

// processDescriptionForFile generates a description for a single file
func processDescriptionForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, model string, customPrompt string, fromQuery bool) error {
	// If not from query, check if file exists in database first
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil // File not in database, skip
		}
	}

	if !overwrite {
		hasDescription, err := hasExistingMetadata(q.Db, filePath, "description")
		if err != nil {
			return fmt.Errorf("error checking existing description: %w", err)
		}
		if hasDescription {
			return nil // Skip, already has description
		}
	}

	description, err := describeFileWithOllama(ctx, q, jobID, filePath, model, customPrompt)
	if err != nil {
		return fmt.Errorf("failed to describe: %w", err)
	}
	if err := updateMediaMetadata(q.Db, filePath, "description", description); err != nil {
		return fmt.Errorf("failed to update: %w", err)
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  description: generated"))
	return nil
}

// processTranscriptForFile generates a transcript for a single video file
func processTranscriptForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, fromQuery bool) error {
	ext := strings.ToLower(filepath.Ext(filePath))
	switch ext {
	case ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv":
		// Valid video file
	default:
		return nil // Not a video file, skip silently
	}

	// If not from query, check if file exists in database first
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil // File not in database, skip
		}
	}

	if !overwrite {
		hasTranscript, err := hasExistingMetadata(q.Db, filePath, "transcript")
		if err != nil {
			return fmt.Errorf("error checking existing transcript: %w", err)
		}
		if hasTranscript {
			return nil // Skip, already has transcript
		}
	}

	transcript, err := generateTranscriptWithFasterWhisper(ctx, q, jobID, filePath)
	if err != nil {
		return fmt.Errorf("failed to transcribe: %w", err)
	}
	if err := updateMediaMetadata(q.Db, filePath, "transcript", transcript); err != nil {
		return fmt.Errorf("failed to update: %w", err)
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  transcript: generated"))
	return nil
}

// processHashForFile generates a hash for a single file
func processHashForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, fromQuery bool) error {
	const maxBytes = 3 * 1024 * 1024

	// If not from query, check if file exists in database first
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil // File not in database, skip
		}
	}

	if !overwrite {
		hasHash, err := hasExistingMetadata(q.Db, filePath, "hash")
		if err != nil {
			return fmt.Errorf("error checking existing hash: %w", err)
		}
		if hasHash {
			return nil // Skip, already has hash
		}
	}

	fi, err := os.Stat(filePath)
	if err != nil {
		return fmt.Errorf("failed to stat: %w", err)
	}
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open: %w", err)
	}
	hashVal, err := hashFirstNBytes(file, maxBytes)
	file.Close()
	if err != nil {
		return fmt.Errorf("failed to hash: %w", err)
	}
	stmt := `UPDATE media SET hash = ?, size = ? WHERE path = ?`
	_, err = q.Db.Exec(stmt, hashVal, fi.Size(), filePath)
	if err != nil {
		return fmt.Errorf("failed to update: %w", err)
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  hash: generated"))
	return nil
}

// processDimensionsForFile generates dimensions for a single file
func processDimensionsForFile(ctx context.Context, q *jobqueue.Queue, jobID string, filePath string, overwrite bool, fromQuery bool) error {
	ext := strings.ToLower(filepath.Ext(filePath))
	var isImage, isVideo bool
	switch ext {
	case ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff", ".heic":
		isImage = true
	case ".mp4", ".mov", ".avi", ".mkv", ".webm":
		isVideo = true
	default:
		return nil // Not a supported file type, skip silently
	}

	// If not from query, check if file exists in database first
	if !fromQuery {
		exists, err := fileExistsInDatabase(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking database: %w", err)
		}
		if !exists {
			return nil // File not in database, skip
		}
	}

	if !overwrite {
		hasDimensions, err := hasExistingDimensions(q.Db, filePath)
		if err != nil {
			return fmt.Errorf("error checking existing dimensions: %w", err)
		}
		if hasDimensions {
			return nil // Skip, already has dimensions
		}
	}

	var width, height int
	var err error
	if isImage {
		width, height, err = getImageDimensions(filePath)
	} else if isVideo {
		width, height, err = getVideoDimensionsFFProbe(filePath)
	}
	if err != nil {
		return fmt.Errorf("failed to get dimensions: %w", err)
	}
	_, err = q.Db.Exec(`UPDATE media SET width = ?, height = ? WHERE path = ?`, width, height, filePath)
	if err != nil {
		return fmt.Errorf("failed to update: %w", err)
	}
	q.PushJobStdout(jobID, fmt.Sprintf("  dimensions: %dx%d", width, height))
	return nil
}
