package main

import (
	"context"
	"crypto/sha256"
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
		if _, err := os.Stat(thumbPath); err == nil {
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
	ffmpegPath := depspkg.GetFFmpegPath()
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
	ffmpegPath := depspkg.GetFFmpegPath()
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
	ffprobePath := depspkg.GetFFprobePath()

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
	useMiddle := durationSec > 6

	timeStr := fmt.Sprintf("%.3f", thumbnailTime)
	targetSize := 600
	if sz, ok := cacheSizes[cache]; ok {
		targetSize = sz
	}
	scaleExpr := fmt.Sprintf("scale='min(%d,iw)':'min(%d,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2", targetSize, targetSize)

	var args []string
	if useMiddle {
		args = []string{"-y", "-ss", timeStr, "-i", mediaPath, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
	} else {
		args = []string{"-y", "-i", mediaPath, "-ss", timeStr, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
	}

	ctx, cancel := context.WithTimeout(context.Background(), videoThumbTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
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
