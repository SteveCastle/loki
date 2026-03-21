package main

import (
	"crypto/sha256"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/platform"
)

var hlsValidPresets = map[string]bool{
	"passthrough": true,
	"480p":        true,
	"720p":        true,
	"1080p":       true,
}

var hlsFilenameRe = regexp.MustCompile(`^(master|stream)\.m3u8$|^segment_\d{3,}\.ts$`)

// hlsInflightMu guards the inflight map for HLS generation deduplication.
var hlsInflightMu sync.Mutex

// hlsInflight tracks HLS generations currently in progress.
var hlsInflight = map[string]*hlsInflightEntry{}

type hlsInflightEntry struct {
	done chan struct{}
	err  error
}

// hlsCacheDir returns the cache directory for a given media file's HLS output.
func hlsCacheDir(basePath, mediaPath string) string {
	h := sha256.Sum256([]byte(mediaPath))
	return filepath.Join(basePath, "hls", fmt.Sprintf("%x", h))
}

// hlsBasePath returns the base path for HLS cache storage.
func hlsBasePath() string {
	return platform.GetDataDir()
}

// isValidHlsFilename checks that a filename matches allowed HLS patterns.
func isValidHlsFilename(name string) bool {
	if strings.Contains(name, "..") || strings.Contains(name, "/") || strings.Contains(name, "\\") {
		return false
	}
	return hlsFilenameRe.MatchString(name)
}

// isValidHlsPreset checks that a preset name is in the allowed set.
func isValidHlsPreset(preset string) bool {
	return hlsValidPresets[preset]
}
