package main

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/tasks"
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

// hlsHandler dispatches GET to hlsServeMaster and DELETE to hlsCleanup.
func hlsHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		switch r.Method {
		case http.MethodGet:
			hlsServeMaster(w, r)
		case http.MethodDelete:
			hlsCleanup(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

// hlsServeMaster handles GET /media/hls?path=... and returns master.m3u8.
// If the cache does not exist, it generates passthrough HLS on-the-fly with
// inflight deduplication so concurrent requests don't start duplicate ffmpeg jobs.
// Once the master.m3u8 exists it redirects to /media/hls/<hash>/master.m3u8 so
// that hls.js can resolve relative playlist URLs correctly.
func hlsServeMaster(w http.ResponseWriter, r *http.Request) {
	mediaPath := r.URL.Query().Get("path")
	if mediaPath == "" {
		http.Error(w, "missing path parameter", http.StatusBadRequest)
		return
	}

	cacheDir := hlsCacheDir(hlsBasePath(), mediaPath)
	masterPath := filepath.Join(cacheDir, "master.m3u8")

	// Fast path: already cached — redirect so relative URLs resolve correctly.
	if _, err := os.Stat(masterPath); err == nil {
		h := sha256.Sum256([]byte(mediaPath))
		hash := fmt.Sprintf("%x", h)
		http.Redirect(w, r, fmt.Sprintf("/media/hls/%s/master.m3u8", hash), http.StatusFound)
		return
	}

	// Inflight deduplication.
	hlsInflightMu.Lock()
	if entry, ok := hlsInflight[cacheDir]; ok {
		hlsInflightMu.Unlock()
		<-entry.done
		if entry.err != nil {
			http.Error(w, "HLS generation failed", http.StatusInternalServerError)
			return
		}
	} else {
		entry := &hlsInflightEntry{done: make(chan struct{})}
		hlsInflight[cacheDir] = entry
		hlsInflightMu.Unlock()

		genErr := generatePassthroughHLS(mediaPath, cacheDir)
		entry.err = genErr

		hlsInflightMu.Lock()
		delete(hlsInflight, cacheDir)
		hlsInflightMu.Unlock()
		close(entry.done)

		if genErr != nil {
			http.Error(w, "HLS generation failed", http.StatusInternalServerError)
			return
		}
	}

	h := sha256.Sum256([]byte(mediaPath))
	hash := fmt.Sprintf("%x", h)
	http.Redirect(w, r, fmt.Sprintf("/media/hls/%s/master.m3u8", hash), http.StatusFound)
}

// hlsCleanup handles DELETE /media/hls?path=... — clears HLS cache for one file
// (if path is given) or for all files (if no path).
func hlsCleanup(w http.ResponseWriter, r *http.Request) {
	mediaPath := r.URL.Query().Get("path")
	if mediaPath != "" {
		cacheDir := hlsCacheDir(hlsBasePath(), mediaPath)
		if err := os.RemoveAll(cacheDir); err != nil {
			http.Error(w, "failed to remove cache: "+err.Error(), http.StatusInternalServerError)
			return
		}
	} else {
		allHlsDir := filepath.Join(hlsBasePath(), "hls")
		if err := os.RemoveAll(allHlsDir); err != nil {
			http.Error(w, "failed to remove all HLS cache: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}
	w.WriteHeader(http.StatusNoContent)
}

// hlsSegmentHandler handles GET /media/hls/<hash>/<preset>/<filename>.
// It validates each path component and serves the file with the correct Content-Type.
func hlsSegmentHandler(d *Dependencies) http.HandlerFunc {
	hexRe := regexp.MustCompile(`^[0-9a-f]+$`)
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")

		path := strings.TrimPrefix(r.URL.Path, "/media/hls/")
		parts := strings.Split(path, "/")

		// Expect either <hash>/master.m3u8 (2 parts) or <hash>/<preset>/<filename> (3 parts).
		if len(parts) == 2 {
			hash := parts[0]
			filename := parts[1]
			if !hexRe.MatchString(hash) {
				http.Error(w, "invalid hash", http.StatusBadRequest)
				return
			}
			if !isValidHlsFilename(filename) {
				http.Error(w, "invalid filename", http.StatusBadRequest)
				return
			}
			filePath := filepath.Join(hlsBasePath(), "hls", hash, filename)
			w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
			http.ServeFile(w, r, filePath)
			return
		}

		if len(parts) == 3 {
			hash := parts[0]
			preset := parts[1]
			filename := parts[2]
			if !hexRe.MatchString(hash) {
				http.Error(w, "invalid hash", http.StatusBadRequest)
				return
			}
			if !isValidHlsPreset(preset) {
				http.Error(w, "invalid preset", http.StatusBadRequest)
				return
			}
			if !isValidHlsFilename(filename) {
				http.Error(w, "invalid filename", http.StatusBadRequest)
				return
			}
			filePath := filepath.Join(hlsBasePath(), "hls", hash, preset, filename)
			contentType := "application/vnd.apple.mpegurl"
			if strings.HasSuffix(filename, ".ts") {
				contentType = "video/MP2T"
			}
			w.Header().Set("Content-Type", contentType)
			http.ServeFile(w, r, filePath)
			return
		}

		http.Error(w, "invalid HLS path", http.StatusBadRequest)
	}
}

// generatePassthroughHLS creates the passthrough HLS cache for a media file.
// It creates the output directory, runs ffmpeg with stream-copy settings, and
// writes a minimal master.m3u8 that references the generated stream playlist.
func generatePassthroughHLS(mediaPath, cacheDir string) error {
	outDir := filepath.Join(cacheDir, "passthrough")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("failed to create HLS output dir: %w", err)
	}

	playlistPath := filepath.Join(outDir, "stream.m3u8")
	segmentPattern := filepath.Join(outDir, "segment_%03d.ts")

	ffmpegPath := depspkg.GetFFmpegPath()
	if ffmpegPath == "" {
		return fmt.Errorf("ffmpeg not found")
	}

	args := tasks.HlsBuildPassthroughArgs(mediaPath, playlistPath, segmentPattern)
	cmd := exec.Command(ffmpegPath, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg failed: %w\n%s", err, string(out))
	}

	masterPath := filepath.Join(cacheDir, "master.m3u8")
	masterContent := "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH=0,NAME=\"passthrough\"\npassthrough/stream.m3u8\n"
	if err := os.WriteFile(masterPath, []byte(masterContent), 0644); err != nil {
		return fmt.Errorf("failed to write master.m3u8: %w", err)
	}

	return nil
}
