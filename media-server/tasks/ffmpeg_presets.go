package tasks

import (
	"fmt"
	"path/filepath"
	"sync"

	"github.com/stevecastle/shrike/jobqueue"
)

// --- FFmpeg Scale ---

var ffmpegScaleOptions = []TaskOption{
	{Name: "width", Label: "Width", Type: "number", Default: 1280.0, Description: "Target width in pixels (height is calculated automatically)"},
}

func ffmpegScaleTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegScaleOptions)
	width, _ := opts["width"].(float64)
	if width == 0 {
		width = 1280
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		vf := fmt.Sprintf("scale=%d:-1", int(width))
		output := filepath.Join(dir, name+"_scaled"+ext)
		return []string{"-vf", vf, output}, output
	})
}

// --- FFmpeg Convert ---

var ffmpegConvertOptions = []TaskOption{
	{Name: "format", Label: "Output Format", Type: "enum", Choices: []string{"mp4", "webm", "mkv", "mov", "gif", "mp3", "wav"}, Default: "mp4", Description: "Target container/format"},
}

func ffmpegConvertTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegConvertOptions)
	format, _ := opts["format"].(string)
	if format == "" {
		format = "mp4"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"."+format)
		return []string{output}, output
	})
}

// --- FFmpeg Extract Audio ---

var ffmpegExtractAudioOptions = []TaskOption{
	{Name: "format", Label: "Audio Format", Type: "enum", Choices: []string{"mp3", "wav", "aac", "flac", "ogg"}, Default: "mp3", Description: "Output audio format"},
}

var audioCodecMap = map[string]string{
	"mp3":  "libmp3lame",
	"wav":  "pcm_s16le",
	"aac":  "aac",
	"flac": "flac",
	"ogg":  "libvorbis",
}

func ffmpegExtractAudioTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegExtractAudioOptions)
	format, _ := opts["format"].(string)
	if format == "" {
		format = "mp3"
	}
	codec := audioCodecMap[format]
	if codec == "" {
		codec = "libmp3lame"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"."+format)
		return []string{"-vn", "-acodec", codec, output}, output
	})
}

// --- FFmpeg Screenshot ---

var ffmpegScreenshotOptions = []TaskOption{
	{Name: "timestamp", Label: "Timestamp", Type: "string", Default: "00:00:01", Description: "Timestamp to capture (HH:MM:SS or seconds)"},
	{Name: "format", Label: "Image Format", Type: "enum", Choices: []string{"jpg", "png", "webp"}, Default: "jpg", Description: "Output image format"},
}

func ffmpegScreenshotTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegScreenshotOptions)
	timestamp, _ := opts["timestamp"].(string)
	if timestamp == "" {
		timestamp = "00:00:01"
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "jpg"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"_screenshot."+format)
		return []string{"-ss", timestamp, "-frames:v", "1", output}, output
	})
}

// --- FFmpeg Thumbnail ---

var ffmpegThumbnailOptions = []TaskOption{
	{Name: "timestamp", Label: "Timestamp", Type: "string", Default: "00:00:01", Description: "Timestamp to capture (HH:MM:SS or seconds)"},
	{Name: "width", Label: "Width", Type: "number", Default: 600.0, Description: "Thumbnail width in pixels"},
	{Name: "format", Label: "Image Format", Type: "enum", Choices: []string{"jpg", "png", "webp"}, Default: "jpg", Description: "Output image format"},
}

func ffmpegThumbnailTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegThumbnailOptions)
	timestamp, _ := opts["timestamp"].(string)
	if timestamp == "" {
		timestamp = "00:00:01"
	}
	width, _ := opts["width"].(float64)
	if width == 0 {
		width = 600
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "jpg"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		vf := fmt.Sprintf("scale=%d:-1", int(width))
		output := filepath.Join(dir, name+"_thumb."+format)
		return []string{"-ss", timestamp, "-frames:v", "1", "-vf", vf, output}, output
	})
}
