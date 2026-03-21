package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// mediaExtRegex matches supported media file extensions (case-insensitive).
// Shared between /api/fs/list and /api/fs/scan to avoid drift.
var mediaExtRegex = regexp.MustCompile(
	`(?i)\.(jpg|jpeg|jfif|webp|png|webm|mp4|mov|mpeg|gif|mkv|m4v|mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape)$`,
)

// isMediaFile returns true if the filename has a supported media extension.
func isMediaFile(name string) bool {
	return mediaExtRegex.MatchString(name)
}

// validatePathWithinRoots checks that path is under one of the allowed roots.
// If roots is empty, all paths are allowed.
// An empty path is always allowed (used to request root listing).
func validatePathWithinRoots(path string, roots []string) error {
	if len(roots) == 0 || path == "" {
		return nil
	}

	cleaned := filepath.Clean(path)

	// Resolve symlinks to prevent escaping via symlink
	resolved, err := filepath.EvalSymlinks(cleaned)
	if err != nil {
		// If the path doesn't exist yet, use the cleaned version
		resolved = cleaned
	}

	for _, root := range roots {
		cleanRoot := filepath.Clean(root)
		rel, err := filepath.Rel(cleanRoot, resolved)
		if err != nil {
			continue
		}
		// If rel starts with "..", the path is outside the root
		if !strings.HasPrefix(rel, "..") {
			return nil
		}
	}

	return fmt.Errorf("path %q is not within any allowed root", path)
}
