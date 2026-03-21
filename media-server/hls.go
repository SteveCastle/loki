package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
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

var hlsFilenameRe = regexp.MustCompile(`^(master|stream)\.m3u8$|^segment_\d{3,}\.ts$`)

// hlsInflightMu guards the inflight map for HLS generation deduplication.
var hlsInflightMu sync.Mutex

// hlsInflight tracks HLS generations currently in progress or queued.
// Key is the cache directory path.
var hlsInflight = map[string]*hlsInflightEntry{}

// hlsSem limits concurrent HLS ffmpeg processes to prevent resource starvation.
var hlsSem = make(chan struct{}, 2)

type hlsInflightEntry struct {
	done     chan struct{}
	cancel   context.CancelFunc
	err      error
	progress float64 // 0.0 to 1.0
	queued   bool    // true if waiting for a semaphore slot
}

// hlsStaleTimeout is how long ffmpeg can run without progress before being killed.
const hlsStaleTimeout = 2 * time.Minute

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

// --- HTTP Handlers ---

// hlsHandler dispatches GET and DELETE for /media/hls.
// GET returns JSON status: {status: "ready"|"processing"|"error", ...}
// DELETE clears HLS cache.
func hlsHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		switch r.Method {
		case http.MethodGet:
			hlsStatus(w, r)
		case http.MethodDelete:
			hlsCleanup(w, r)
		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

type hlsStatusResponse struct {
	Status   string  `json:"status"`             // "ready", "processing", "error"
	URL      string  `json:"url,omitempty"`       // manifest URL when ready
	Progress float64 `json:"progress,omitempty"`  // 0.0-1.0 when processing
	Error    string  `json:"error,omitempty"`     // error message
	Duration float64 `json:"duration,omitempty"`  // source duration in seconds
}

// hlsStatus returns the HLS generation status for a media file.
// If already cached: {status: "ready", url: "/media/hls/<hash>/master.m3u8"}
// If generating: {status: "processing", progress: 0.45, duration: 120.5}
// If not started: kicks off generation and returns processing status.
func hlsStatus(w http.ResponseWriter, r *http.Request) {
	mediaPath := r.URL.Query().Get("path")
	if mediaPath == "" {
		http.Error(w, "missing path parameter", http.StatusBadRequest)
		return
	}

	cacheDir := hlsCacheDir(hlsBasePath(), mediaPath)
	masterPath := filepath.Join(cacheDir, "master.m3u8")
	h := sha256.Sum256([]byte(mediaPath))
	hash := fmt.Sprintf("%x", h)

	w.Header().Set("Content-Type", "application/json")

	// Already cached — ready to play.
	if _, err := os.Stat(masterPath); err == nil {
		log.Printf("[hls] ready (cached): %s", filepath.Base(mediaPath))
		json.NewEncoder(w).Encode(hlsStatusResponse{
			Status: "ready",
			URL:    fmt.Sprintf("/media/hls/%s/master.m3u8", hash),
		})
		return
	}

	// Check if generation is in progress or queued.
	hlsInflightMu.Lock()
	entry, inflight := hlsInflight[cacheDir]
	if inflight {
		progress := entry.progress
		queued := entry.queued
		hlsInflightMu.Unlock()

		// Check if it just finished (entry.done closed).
		select {
		case <-entry.done:
			if entry.err != nil {
				json.NewEncoder(w).Encode(hlsStatusResponse{
					Status: "error",
					Error:  entry.err.Error(),
				})
			} else {
				json.NewEncoder(w).Encode(hlsStatusResponse{
					Status: "ready",
					URL:    fmt.Sprintf("/media/hls/%s/master.m3u8", hash),
				})
			}
		default:
			status := "processing"
			if queued {
				status = "queued"
			}
			json.NewEncoder(w).Encode(hlsStatusResponse{
				Status:   status,
				Progress: progress,
			})
		}
		return
	}

	// Not cached and not in progress — clean up any leftover partial cache and start fresh.
	os.RemoveAll(cacheDir)

	ctx, cancel := context.WithCancel(context.Background())
	entry = &hlsInflightEntry{done: make(chan struct{}), cancel: cancel}
	hlsInflight[cacheDir] = entry
	hlsInflightMu.Unlock()

	// Probe duration (fast) for progress calculation and client display.
	duration := probeDuration(mediaPath)
	log.Printf("[hls] starting generation: %s (duration: %.1fs)", filepath.Base(mediaPath), duration)

	entry.queued = true
	go func() {
		// Wait for a semaphore slot (limits concurrent ffmpeg processes).
		log.Printf("[hls] queued: %s", filepath.Base(mediaPath))
		hlsSem <- struct{}{}
		hlsInflightMu.Lock()
		entry.queued = false
		hlsInflightMu.Unlock()
		log.Printf("[hls] encoding started: %s", filepath.Base(mediaPath))

		genErr := generatePassthroughHLS(ctx, mediaPath, cacheDir, duration, entry)
		<-hlsSem // Release slot.
		entry.err = genErr

		if genErr != nil {
			log.Printf("[hls] generation failed: %s — %v", filepath.Base(mediaPath), genErr)
			// Clean up partial output so the next request starts fresh.
			os.RemoveAll(cacheDir)
		} else {
			log.Printf("[hls] generation complete: %s", filepath.Base(mediaPath))
		}

		hlsInflightMu.Lock()
		delete(hlsInflight, cacheDir)
		hlsInflightMu.Unlock()
		cancel()
		close(entry.done)
	}()

	json.NewEncoder(w).Encode(hlsStatusResponse{
		Status:   "processing",
		Progress: 0,
		Duration: duration,
	})
}

func hlsCleanup(w http.ResponseWriter, r *http.Request) {
	mediaPath := r.URL.Query().Get("path")
	if mediaPath != "" {
		cacheDir := hlsCacheDir(hlsBasePath(), mediaPath)
		log.Printf("[hls] clearing cache for: %s", filepath.Base(mediaPath))
		os.RemoveAll(cacheDir)
	} else {
		log.Printf("[hls] clearing all HLS cache")
		os.RemoveAll(filepath.Join(hlsBasePath(), "hls"))
	}
	w.WriteHeader(http.StatusNoContent)
}

// hlsSegmentHandler serves HLS files from the cache directory.
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
			hash, filename := parts[0], parts[1]
			if !hexRe.MatchString(hash) || !isValidHlsFilename(filename) {
				http.Error(w, "invalid path", http.StatusBadRequest)
				return
			}
			filePath = filepath.Join(hlsBasePath(), "hls", hash, filename)
			contentType = "application/vnd.apple.mpegurl"

		case 3:
			hash, preset, filename := parts[0], parts[1], parts[2]
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

		if _, err := os.Stat(filePath); err != nil {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", contentType)
		http.ServeFile(w, r, filePath)
	}
}

// --- Generation ---

// generatePassthroughHLS runs ffmpeg to create the full VOD HLS stream.
// It uses a context for cancellation, parses ffmpeg's -progress output for
// progress tracking, and detects stalls (no progress for hlsStaleTimeout).
func generatePassthroughHLS(ctx context.Context, mediaPath, cacheDir string, duration float64, entry *hlsInflightEntry) error {
	outDir := filepath.Join(cacheDir, "passthrough")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("failed to create HLS output dir: %w", err)
	}

	playlistPath := filepath.Join(outDir, "stream.m3u8")
	segmentPattern := filepath.Join(outDir, "segment_%03d.ts")

	ffmpegPath := depspkg.GetFFmpegPath()
	if ffmpegPath == "" {
		log.Printf("[hls] ERROR: ffmpeg not found")
		return fmt.Errorf("ffmpeg not found")
	}

	args := []string{
		"-y", "-i", mediaPath,
		"-c", "copy",
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", segmentPattern,
		playlistPath,
		"-progress", "pipe:1",
	}

	log.Printf("[hls] ffmpeg args: %v", args)

	cmd := exec.CommandContext(ctx, ffmpegPath, args...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Printf("[hls] ERROR: stdout pipe: %v", err)
		return fmt.Errorf("stdout pipe: %w", err)
	}

	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("[hls] ERROR: stderr pipe: %v", err)
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		log.Printf("[hls] ERROR: ffmpeg start: %v", err)
		return fmt.Errorf("ffmpeg start: %w", err)
	}

	log.Printf("[hls] ffmpeg pid %d started for %s", cmd.Process.Pid, filepath.Base(mediaPath))

	// Drain stderr in background.
	var stderrBuf strings.Builder
	stderrDone := make(chan struct{})
	go func() {
		s := bufio.NewScanner(stderr)
		for s.Scan() {
			stderrBuf.WriteString(s.Text())
			stderrBuf.WriteString("\n")
		}
		close(stderrDone)
	}()

	// Stale detection: kill ffmpeg if no progress for hlsStaleTimeout.
	lastProgressTime := time.Now()
	staleCancel := make(chan struct{})
	go func() {
		ticker := time.NewTicker(15 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-staleCancel:
				return
			case <-ctx.Done():
				return
			case <-ticker.C:
				hlsInflightMu.Lock()
				elapsed := time.Since(lastProgressTime)
				hlsInflightMu.Unlock()
				if elapsed > hlsStaleTimeout {
					log.Printf("[hls] WARNING: ffmpeg stalled for %s (no progress for %v), killing",
						filepath.Base(mediaPath), elapsed.Round(time.Second))
					cmd.Process.Kill()
					return
				}
			}
		}
	}()

	// Parse -progress output to track progress.
	lastLoggedPct := -1
	scanner := bufio.NewScanner(stdout)
	for scanner.Scan() {
		line := scanner.Text()
		if duration > 0 && strings.HasPrefix(line, "out_time_us=") {
			usStr := strings.TrimPrefix(line, "out_time_us=")
			if us, err := strconv.ParseInt(usStr, 10, 64); err == nil && us > 0 {
				progress := float64(us) / (duration * 1_000_000)
				if progress > 1 {
					progress = 1
				}
				hlsInflightMu.Lock()
				entry.progress = progress
				lastProgressTime = time.Now()
				hlsInflightMu.Unlock()

				pct := int(progress * 10) * 10
				if pct > lastLoggedPct {
					lastLoggedPct = pct
					log.Printf("[hls] progress %d%% — %s", pct, filepath.Base(mediaPath))
				}
			}
		}
	}

	close(staleCancel)
	<-stderrDone

	if err := cmd.Wait(); err != nil {
		errMsg := stderrBuf.String()
		lines := strings.Split(strings.TrimSpace(errMsg), "\n")
		tail := lines
		if len(tail) > 10 {
			tail = tail[len(tail)-10:]
		}
		log.Printf("[hls] ERROR: ffmpeg failed for %s: %v\n  stderr (last lines):\n  %s",
			filepath.Base(mediaPath), err, strings.Join(tail, "\n  "))
		return fmt.Errorf("ffmpeg failed: %w\n%s", err, strings.Join(tail, "\n"))
	}

	// Write master playlist now that all segments are ready.
	masterContent := "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-STREAM-INF:BANDWIDTH=0,NAME=\"passthrough\"\npassthrough/stream.m3u8\n"
	masterPath := filepath.Join(cacheDir, "master.m3u8")
	if err := os.WriteFile(masterPath, []byte(masterContent), 0644); err != nil {
		log.Printf("[hls] ERROR: failed to write master.m3u8: %v", err)
		return fmt.Errorf("failed to write master.m3u8: %w", err)
	}

	log.Printf("[hls] master.m3u8 written: %s", masterPath)
	return nil
}

// probeDuration uses ffprobe to get the source media duration in seconds.
func probeDuration(mediaPath string) float64 {
	ffprobePath := depspkg.GetFFprobePath()
	if ffprobePath == "" {
		log.Printf("[hls] WARNING: ffprobe not found, cannot determine duration")
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
		log.Printf("[hls] WARNING: ffprobe failed for %s: %v", filepath.Base(mediaPath), err)
		return 0
	}
	var data struct {
		Format struct {
			Duration string `json:"duration"`
		} `json:"format"`
	}
	if json.Unmarshal(out, &data) != nil {
		log.Printf("[hls] WARNING: ffprobe output parse failed for %s", filepath.Base(mediaPath))
		return 0
	}
	dur, _ := strconv.ParseFloat(data.Format.Duration, 64)
	return dur
}
