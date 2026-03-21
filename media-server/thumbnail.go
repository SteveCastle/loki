package main

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	depspkg "github.com/stevecastle/shrike/deps"
)

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
func getThumbnailPath(mediaPath, basePath, cache string, timeStamp int) string {
	thumbDir := filepath.Join(basePath, cache)
	hashInput := mediaPath
	if timeStamp > 0 {
		hashInput += fmt.Sprintf("%d", timeStamp)
	}
	fileName := createHash(hashInput)
	if getFileType(mediaPath) == "video" {
		fileName += ".mp4"
	}
	return filepath.Join(thumbDir, fileName)
}

// generateThumbnail creates a thumbnail for the given media file using ffmpeg.
// Returns the full path to the generated thumbnail.
func generateThumbnail(mediaPath, basePath, cache string, timeStamp int) (string, error) {
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
		if err := generateVideoThumbnail(ffmpegPath, mediaPath, thumbPath, timeStamp); err != nil {
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

	cmd := exec.Command(ffmpegPath, args...)
	if output, err := cmd.CombinedOutput(); err != nil {
		log.Printf("ffmpeg image thumbnail failed for %s: %s", mediaPath, string(output))
		return fmt.Errorf("ffmpeg failed: %w", err)
	}
	return nil
}

func generateVideoThumbnail(ffmpegPath, mediaPath, thumbPath string, timeStamp int) error {
	ffprobePath := depspkg.GetFFprobePath()

	// Get video duration using ffprobe
	durationSec := 0.0
	if ffprobePath != "" {
		probeArgs := []string{
			"-v", "quiet",
			"-print_format", "json",
			"-show_format",
			mediaPath,
		}
		cmd := exec.Command(ffprobePath, probeArgs...)
		if output, err := cmd.Output(); err == nil {
			var result struct {
				Format struct {
					Duration string `json:"duration"`
				} `json:"format"`
			}
			if json.Unmarshal(output, &result) == nil {
				fmt.Sscanf(result.Format.Duration, "%f", &durationSec)
			}
		}
	}

	thumbnailTime := float64(timeStamp)
	if thumbnailTime == 0 {
		thumbnailTime = durationSec / 2
	}
	useMiddle := durationSec > 6

	timeStr := fmt.Sprintf("%.3f", thumbnailTime)
	scaleExpr := "scale='min(400,iw)':'min(400,ih)':force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2"

	var args []string
	if useMiddle {
		args = []string{"-y", "-ss", timeStr, "-i", mediaPath, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
	} else {
		args = []string{"-y", "-i", mediaPath, "-ss", timeStr, "-vf", scaleExpr, "-t", "2", "-an", thumbPath}
	}

	cmd := exec.Command(ffmpegPath, args...)
	if output, err := cmd.CombinedOutput(); err != nil {
		log.Printf("ffmpeg video thumbnail failed for %s: %s", mediaPath, string(output))
		return fmt.Errorf("ffmpeg failed: %w", err)
	}
	return nil
}
