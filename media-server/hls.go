package main

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"encoding/json"
	"strconv"
	"strings"
	"sync"
	"time"

	depspkg "github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

var hlsValidPresets = map[string]bool{
	"passthrough": true,
	"480p":        true,
	"720p":        true,
	"1080p":       true,
}

var hlsFilenameRe = regexp.MustCompile(`^(master|stream)\.m3u8$|^segment_\d{3,}\.ts$|^duration\.json$`)

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
// If the cache does not exist, it kicks off passthrough HLS generation in the
// background and writes the master playlist immediately so playback can start
// as soon as the first segment is ready (ffmpeg writes segments progressively).
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

	// Kick off generation (non-blocking). Dedup concurrent requests.
	hlsInflightMu.Lock()
	if _, ok := hlsInflight[cacheDir]; !ok {
		entry := &hlsInflightEntry{done: make(chan struct{})}
		hlsInflight[cacheDir] = entry

		// Write master playlist and output dirs synchronously so redirect works.
		outDir := filepath.Join(cacheDir, "passthrough")
		os.MkdirAll(outDir, 0755)
		masterContent := "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH=0,NAME=\"passthrough\"\npassthrough/stream.m3u8\n"
		os.WriteFile(masterPath, []byte(masterContent), 0644)

		// Probe source duration (fast) and write duration.json so the
		// client can show total length before generation completes.
		if dur := probeDuration(mediaPath); dur > 0 {
			durJSON, _ := json.Marshal(map[string]float64{"duration": dur})
			os.WriteFile(filepath.Join(cacheDir, "duration.json"), durJSON, 0644)
		}

		// Start ffmpeg in the background — segments appear progressively.
		go func() {
			genErr := generatePassthroughHLS(mediaPath, cacheDir)
			entry.err = genErr

			hlsInflightMu.Lock()
			delete(hlsInflight, cacheDir)
			hlsInflightMu.Unlock()
			close(entry.done)
		}()
	}
	hlsInflightMu.Unlock()

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
// If a file doesn't exist yet but generation is in progress, it waits briefly
// for ffmpeg to produce it rather than returning an immediate 404.
func hlsSegmentHandler(d *Dependencies) http.HandlerFunc {
	hexRe := regexp.MustCompile(`^[0-9a-f]+$`)
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")

		path := strings.TrimPrefix(r.URL.Path, "/media/hls/")
		parts := strings.Split(path, "/")

		var filePath string
		var contentType string

		switch len(parts) {
		case 2:
			// <hash>/master.m3u8
			hash := parts[0]
			filename := parts[1]
			if !hexRe.MatchString(hash) || !isValidHlsFilename(filename) {
				http.Error(w, "invalid path", http.StatusBadRequest)
				return
			}
			filePath = filepath.Join(hlsBasePath(), "hls", hash, filename)
			if strings.HasSuffix(filename, ".json") {
				contentType = "application/json"
			} else {
				contentType = "application/vnd.apple.mpegurl"
			}

		case 3:
			// <hash>/<preset>/<filename>
			hash := parts[0]
			preset := parts[1]
			filename := parts[2]
			if !hexRe.MatchString(hash) || !isValidHlsPreset(preset) || !isValidHlsFilename(filename) {
				http.Error(w, "invalid path", http.StatusBadRequest)
				return
			}
			filePath = filepath.Join(hlsBasePath(), "hls", hash, preset, filename)
			contentType = "application/vnd.apple.mpegurl"
			if strings.HasSuffix(filename, ".ts") {
				contentType = "video/MP2T"
			}

		default:
			http.Error(w, "invalid HLS path", http.StatusBadRequest)
			return
		}

		// If the file doesn't exist yet, wait up to 30 seconds for ffmpeg to
		// produce it (generation is async). Poll every 500ms.
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			ctx := r.Context()
			ticker := time.NewTicker(500 * time.Millisecond)
			defer ticker.Stop()
			timeout := time.After(30 * time.Second)
			found := false
			for !found {
				select {
				case <-ctx.Done():
					return
				case <-timeout:
					http.Error(w, "not found", http.StatusNotFound)
					return
				case <-ticker.C:
					if _, err := os.Stat(filePath); err == nil {
						found = true
					}
				}
			}
		}

		w.Header().Set("Content-Type", contentType)
		http.ServeFile(w, r, filePath)
	}
}

// generatePassthroughHLS creates the passthrough HLS cache for a media file.
// It uses "-hls_playlist_type event" so ffmpeg writes the stream.m3u8
// progressively as each segment completes, allowing playback to start
// before the entire file is processed. Once ffmpeg finishes, it appends
// #EXT-X-ENDLIST to convert the playlist to VOD for full seeking.
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

	// Use "event" playlist type so the manifest is written progressively.
	// The standard HlsBuildPassthroughArgs uses "vod" which only writes
	// the playlist after ALL segments are done. We override that here.
	args := []string{
		"-y", "-i", mediaPath,
		"-c", "copy",
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "event",
		"-hls_segment_filename", segmentPattern,
		playlistPath,
	}

	cmd := exec.Command(ffmpegPath, args...)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("ffmpeg failed: %w\n%s", err, string(out))
	}

	// Append #EXT-X-ENDLIST to signal VOD completion so hls.js enables
	// full seeking and stops polling for new segments.
	playlist, err := os.ReadFile(playlistPath)
	if err == nil && !strings.Contains(string(playlist), "#EXT-X-ENDLIST") {
		f, err := os.OpenFile(playlistPath, os.O_APPEND|os.O_WRONLY, 0644)
		if err == nil {
			f.WriteString("#EXT-X-ENDLIST\n")
			f.Close()
		}
	}

	return nil
}

// probeDuration uses ffprobe to get the source media duration in seconds.
func probeDuration(mediaPath string) float64 {
	ffprobePath := depspkg.GetFFprobePath()
	if ffprobePath == "" {
		return 0
	}
	cmd := exec.Command(ffprobePath,
		"-v", "quiet",
		"-print_format", "json",
		"-show_format",
		mediaPath,
	)
	out, err := cmd.Output()
	if err != nil {
		return 0
	}
	var data struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if json.Unmarshal(out, &data) != nil {
		return 0
	}
	dur, _ := strconv.ParseFloat(data.Format.Duration, 64)
	return dur
}
