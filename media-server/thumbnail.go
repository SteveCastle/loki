package main

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/storage"
)

const (
	// imageThumbTimeout is the maximum time allowed for generating an image thumbnail.
	imageThumbTimeout = 30 * time.Second
	// videoThumbTimeout is the maximum time allowed for generating a video thumbnail
	// (includes ffprobe duration detection + ffmpeg encoding).
	videoThumbTimeout = 60 * time.Second
)

// thumbSem limits concurrent ffmpeg processes to prevent resource starvation.
var thumbSem = make(chan struct{}, 4)

// inflightMu guards the inflight map for thumbnail deduplication.
var inflightMu sync.Mutex

// inflight tracks thumbnails currently being generated.
// Key is "mediaPath|cache|timeStamp", value is a channel that closes when done.
var inflight = map[string]chan struct{}{}

// thumbnailFileValid reports whether an on-disk thumbnail is usable. A
// frameless .mp4 (empty mdat box) is what older builds cached for
// single-frame GIFs — ffmpeg exits 0 after seeking past the only frame — so
// those must read as missing to get regenerated.
func thumbnailFileValid(path string) bool {
	info, err := os.Stat(path)
	if err != nil || info.Size() == 0 {
		return false
	}
	if strings.ToLower(filepath.Ext(path)) != ".mp4" {
		return true
	}
	return mp4HasFrames(path, info.Size())
}

// mp4HasFrames scans top-level MP4 boxes for an mdat with a non-empty payload.
func mp4HasFrames(path string, fileSize int64) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	hdr := make([]byte, 16)
	var offset int64
	for offset+8 <= fileSize {
		if _, err := f.ReadAt(hdr[:8], offset); err != nil {
			return false
		}
		size := int64(binary.BigEndian.Uint32(hdr[:4]))
		boxType := string(hdr[4:8])
		payload := size - 8
		switch size {
		case 0: // box extends to end of file
			size = fileSize - offset
			payload = size - 8
		case 1: // 64-bit largesize follows the type
			if _, err := f.ReadAt(hdr[8:16], offset+8); err != nil {
				return false
			}
			size = int64(binary.BigEndian.Uint64(hdr[8:16]))
			payload = size - 16
		}
		if boxType == "mdat" && payload > 0 {
			return true
		}
		if size < 8 {
			return false
		}
		offset += size
	}
	return false
}

// formatTimeStamp formats a float64 timestamp the same way JavaScript's
// Number.toString() does — no trailing zeros, no exponent for normal values.
// This ensures the hash matches the Electron app's thumbnail cache.
func formatTimeStamp(ts float64) string {
	return strconv.FormatFloat(ts, 'f', -1, 64)
}

// thumbKey builds a dedup key for a thumbnail generation request.
func thumbKey(mediaPath, cache string, timeStamp float64) string {
	return fmt.Sprintf("%s|%s|%s", mediaPath, cache, formatTimeStamp(timeStamp))
}

// generateThumbnailThrottled wraps generateThumbnail with a concurrency semaphore
// and deduplication. If the same thumbnail is already being generated, it waits
// for that to finish instead of spawning a second ffmpeg process.
// Returns the thumbnail path and any error.
func generateThumbnailThrottled(mediaPath, basePath, cache string, timeStamp float64) (string, error) {
	key := thumbKey(mediaPath, cache, timeStamp)

	// Check if this thumbnail is already being generated
	inflightMu.Lock()
	if ch, ok := inflight[key]; ok {
		inflightMu.Unlock()
		// Wait for the in-flight generation to finish
		<-ch
		// The other goroutine already generated it; check the result on disk
		thumbPath := getThumbnailPath(mediaPath, basePath, cache, timeStamp)
		if thumbnailFileValid(thumbPath) {
			return thumbPath, nil
		}
		return "", fmt.Errorf("in-flight thumbnail generation failed for %s", mediaPath)
	}
	// Register ourselves as in-flight
	done := make(chan struct{})
	inflight[key] = done
	inflightMu.Unlock()

	defer func() {
		inflightMu.Lock()
		delete(inflight, key)
		close(done)
		inflightMu.Unlock()
	}()

	// Acquire semaphore slot (blocks if 4 ffmpeg processes already running)
	thumbSem <- struct{}{}
	defer func() { <-thumbSem }()

	return generateThumbnail(mediaPath, basePath, cache, timeStamp)
}

// Thumbnail sizes matching the Electron app
var cacheSizes = map[string]int{
	"thumbnail_path_1200": 1200,
	"thumbnail_path_600":  600,
	"thumbnail_path_100":  100,
}

var imageExtensions = map[string]bool{
	".jpg": true, ".jpeg": true, ".png": true, ".bmp": true,
	".svg": true, ".jfif": true, ".pjpeg": true, ".pjp": true, ".webp": true,
}

var videoExtensions = map[string]bool{
	".mp4": true, ".webm": true, ".ogg": true, ".mkv": true,
	".mov": true, ".m4v": true, ".gif": true,
}

func getFileType(filePath string) string {
	ext := strings.ToLower(filepath.Ext(filePath))
	if imageExtensions[ext] {
		return "image"
	}
	if videoExtensions[ext] {
		return "video"
	}
	return "other"
}

func createHash(input string) string {
	h := sha256.Sum256([]byte(input))
	return fmt.Sprintf("%x", h)
}

// getThumbnailPath computes the expected thumbnail path for a media file.
func getThumbnailPath(mediaPath, basePath, cache string, timeStamp float64) string {
	thumbDir := filepath.Join(basePath, cache)
	hashInput := mediaPath
	if timeStamp > 0 {
		hashInput += formatTimeStamp(timeStamp)
	}
	fileName := createHash(hashInput)
	if getFileType(mediaPath) == "video" {
		fileName += ".mp4"
	}
	return filepath.Join(thumbDir, fileName)
}

func getS3ThumbnailPath(mediaPath string, backend *storage.S3Backend, cache string, timeStamp float64) string {
	hashInput := mediaPath
	if timeStamp > 0 {
		hashInput += formatTimeStamp(timeStamp)
	}
	fileName := createHash(hashInput)
	if getFileType(mediaPath) == "video" {
		fileName += ".mp4"
	} else {
		fileName += ".png"
	}
	return backend.ThumbnailPath(fileName)
}

func generateS3ThumbnailThrottled(ctx context.Context, mediaPath string, backend *storage.S3Backend, cache string, timeStamp float64) (string, error) {
	key := thumbKey(mediaPath, cache, timeStamp)

	inflightMu.Lock()
	if ch, ok := inflight[key]; ok {
		inflightMu.Unlock()
		<-ch
		thumbPath := getS3ThumbnailPath(mediaPath, backend, cache, timeStamp)
		exists, _ := backend.Exists(ctx, thumbPath)
		if exists {
			return thumbPath, nil
		}
		return "", fmt.Errorf("in-flight S3 thumbnail generation failed for %s", mediaPath)
	}
	done := make(chan struct{})
	inflight[key] = done
	inflightMu.Unlock()

	defer func() {
		inflightMu.Lock()
		delete(inflight, key)
		close(done)
		inflightMu.Unlock()
	}()

	thumbSem <- struct{}{}
	defer func() { <-thumbSem }()

	// Download source to temp file
	reader, err := backend.Download(ctx, mediaPath)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", mediaPath, err)
	}
	defer reader.Close()

	ext := filepath.Ext(mediaPath)
	tmpSource, err := os.CreateTemp("", "loki-thumb-src-*"+ext)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	tmpSourcePath := tmpSource.Name()
	defer os.Remove(tmpSourcePath)
	if _, err := io.Copy(tmpSource, reader); err != nil {
		tmpSource.Close()
		return "", fmt.Errorf("failed to write temp source: %w", err)
	}
	tmpSource.Close()

	// Generate thumbnail to temp output using existing ffmpeg functions
	ffmpegPath := depspkg.BundledOrEmpty("ffmpeg")
	if ffmpegPath == "" {
		return "", fmt.Errorf("ffmpeg not found")
	}

	// Determine output extension
	outExt := ".png"
	if getFileType(mediaPath) == "video" {
		outExt = ".mp4"
	}
	tmpOutput, err := os.CreateTemp("", "loki-thumb-out-*"+outExt)
	if err != nil {
		return "", fmt.Errorf("failed to create temp output: %w", err)
	}
	tmpOutputPath := tmpOutput.Name()
	tmpOutput.Close()
	defer os.Remove(tmpOutputPath)

	fileType := getFileType(mediaPath)
	switch fileType {
	case "image":
		if err := generateImageThumbnail(ffmpegPath, tmpSourcePath, tmpOutputPath, cache); err != nil {
			return "", err
		}
	case "video":
		if err := generateVideoThumbnail(ffmpegPath, tmpSourcePath, tmpOutputPath, cache, timeStamp); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("unsupported file type: %s", ext)
	}

	// Upload result to S3
	thumbPath := getS3ThumbnailPath(mediaPath, backend, cache, timeStamp)
	outputFile, err := os.Open(tmpOutputPath)
	if err != nil {
		return "", fmt.Errorf("failed to open generated thumbnail: %w", err)
	}
	defer outputFile.Close()

	contentType := "image/png"
	if fileType == "video" {
		contentType = "video/mp4"
	}
	if err := backend.Upload(ctx, thumbPath, outputFile, contentType); err != nil {
		return "", fmt.Errorf("failed to upload thumbnail: %w", err)
	}

	return thumbPath, nil
}

// generateThumbnail creates a thumbnail for the given media file using ffmpeg.
// Returns the full path to the generated thumbnail.
func generateThumbnail(mediaPath, basePath, cache string, timeStamp float64) (string, error) {
	ffmpegPath := depspkg.BundledOrEmpty("ffmpeg")
	if ffmpegPath == "" {
		return "", fmt.Errorf("ffmpeg not found")
	}

	thumbPath := getThumbnailPath(mediaPath, basePath, cache, timeStamp)

	// Ensure the thumbnail directory exists
	if err := os.MkdirAll(filepath.Dir(thumbPath), 0755); err != nil {
		return "", fmt.Errorf("failed to create thumbnail directory: %w", err)
	}

	fileType := getFileType(mediaPath)
	switch fileType {
	case "image":
		if err := generateImageThumbnail(ffmpegPath, mediaPath, thumbPath, cache); err != nil {
			return "", err
		}
	case "video":
		if err := generateVideoThumbnail(ffmpegPath, mediaPath, thumbPath, cache, timeStamp); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("unsupported file type for thumbnails: %s", filepath.Ext(mediaPath))
	}

	return thumbPath, nil
}

func generateImageThumbnail(ffmpegPath, mediaPath, thumbPath, cache string) error {
	targetSize := 600
	if sz, ok := cacheSizes[cache]; ok {
		targetSize = sz
	}

	scaleExpr := fmt.Sprintf("scale='min(%d,iw)':-2:force_original_aspect_ratio=decrease", targetSize)
	args := []string{
		"-y",
		"-i", mediaPath,
		"-vf", scaleExpr,
		"-f", "image2",
		"-vcodec", "png",
		"-frames:v", "1",
		thumbPath,
	}

	ctx, cancel := context.WithTimeout(context.Background(), imageThumbTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	platform.HideSubprocessWindow(cmd)
	if output, err := cmd.CombinedOutput(); err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			log.Printf("ffmpeg image thumbnail timed out after %v for %s", imageThumbTimeout, mediaPath)
			return fmt.Errorf("ffmpeg timed out after %v", imageThumbTimeout)
		}
		log.Printf("ffmpeg image thumbnail failed for %s: %s", mediaPath, string(output))
		return fmt.Errorf("ffmpeg failed: %w", err)
	}
	return nil
}

func generateVideoThumbnail(ffmpegPath, mediaPath, thumbPath, cache string, timeStamp float64) error {
	ffprobePath := depspkg.BundledOrEmpty("ffprobe")

	// Get video duration using ffprobe (with its own timeout)
	durationSec := 0.0
	if ffprobePath != "" {
		probeCtx, probeCancel := context.WithTimeout(context.Background(), 15*time.Second)
		defer probeCancel()

		probeArgs := []string{
			"-v", "quiet",
			"-print_format", "json",
			"-show_format",
			mediaPath,
		}
		cmd := exec.CommandContext(probeCtx, ffprobePath, probeArgs...)
		platform.HideSubprocessWindow(cmd)
		if output, err := cmd.Output(); err == nil {
			var result struct {
				Format struct {
					Duration string `json:"duration"`
				} `json:"format"`
			}
			if json.Unmarshal(output, &result) == nil {
				fmt.Sscanf(result.Format.Duration, "%f", &durationSec)
			}
		} else if probeCtx.Err() == context.DeadlineExceeded {
			log.Printf("ffprobe timed out for %s, proceeding without duration", mediaPath)
		}
	}

	thumbnailTime := timeStamp
	if thumbnailTime == 0 {
		thumbnailTime = durationSec / 2
	}
	// Seeking near the end of a very short file (e.g. a single-frame GIF,
	// duration ~0.04s) lands after the only frame: ffmpeg exits 0 but
	// encodes nothing. Take the first frame instead.
	if durationSec > 0 && (durationSec < 1 || thumbnailTime >= durationSec) {
		thumbnailTime = 0
	}
	useMiddle := durationSec > 6

	targetSize := 600
	if sz, ok := cacheSizes[cache]; ok {
		targetSize = sz
	}
	scaleExpr := fmt.Sprintf("scale='min(%d,iw)':'min(%d,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2", targetSize, targetSize)

	run := func(seek float64) error {
		timeStr := fmt.Sprintf("%.3f", seek)
		var args []string
		if useMiddle && seek > 0 {
			args = []string{"-y", "-ss", timeStr, "-i", mediaPath, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
		} else {
			args = []string{"-y", "-i", mediaPath, "-ss", timeStr, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
		}

		ctx, cancel := context.WithTimeout(context.Background(), videoThumbTimeout)
		defer cancel()

		cmd := exec.CommandContext(ctx, ffmpegPath, args...)
		platform.HideSubprocessWindow(cmd)
		if output, err := cmd.CombinedOutput(); err != nil {
			if ctx.Err() == context.DeadlineExceeded {
				log.Printf("ffmpeg video thumbnail timed out after %v for %s", videoThumbTimeout, mediaPath)
				return fmt.Errorf("ffmpeg timed out after %v", videoThumbTimeout)
			}
			log.Printf("ffmpeg video thumbnail failed for %s: %s", mediaPath, string(output))
			return fmt.Errorf("ffmpeg failed: %w", err)
		}
		return nil
	}

	if err := run(thumbnailTime); err != nil {
		return err
	}
	if thumbnailFileValid(thumbPath) {
		return nil
	}
	if thumbnailTime > 0 {
		log.Printf("ffmpeg encoded no frames at %.3fs for %s, retrying from first frame", thumbnailTime, mediaPath)
		if err := run(0); err != nil {
			return err
		}
		if thumbnailFileValid(thumbPath) {
			return nil
		}
	}
	os.Remove(thumbPath)
	return fmt.Errorf("ffmpeg produced an empty thumbnail for %s", mediaPath)
}
