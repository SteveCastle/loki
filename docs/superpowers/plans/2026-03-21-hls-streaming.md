# HLS Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HLS streaming support so videos can be served as adaptive-bitrate HLS with full seek support, with a toggle to fall back to direct source playback.

**Architecture:** A new Go task (`hls`) generates HLS segments via ffmpeg into a cache directory. New HTTP endpoints serve the master playlist and segments using path-based URLs for correct relative resolution. The React video player integrates hls.js to play from the manifest, with fallback to direct source on error or when HLS is disabled. HLS is only available in web mode (where the Go server runs); in Electron mode the `gsm://` protocol serves files directly and HLS is not used.

**Tech Stack:** Go (media server), ffmpeg (HLS segmentation), hls.js (browser playback), React/TypeScript (client), XState (state management)

**Spec:** `docs/superpowers/specs/2026-03-21-hls-streaming-design.md`

---

## File Structure

### New Files
- `media-server/tasks/hls.go` — HLS generation task (ffprobe + ffmpeg), preset definitions, master playlist writer
- `media-server/hls.go` — HLS HTTP handlers (master, segment, cleanup), cache utilities, inflight dedup
- `media-server/hls_test.go` — Tests for cache path computation and filename/preset validation

### Modified Files
- `media-server/tasks/registry.go:33` — Add `RegisterTask("hls", ...)` call
- `media-server/main.go:2731` — Add HLS endpoint routes after `/media/file`
- `media-server/main_linux.go:2320` — Add HLS endpoint routes after `/media/file`
- `media-server/main_darwin.go:2320` — Add HLS endpoint routes after `/media/file`
- `src/renderer/platform.ts:236,393` — Add `hlsUrl` export (web mode only)
- `src/renderer/components/media-viewers/video.tsx:1-307` — Add hls.js integration
- `src/renderer/components/detail/detail.tsx:68-81` — Pass `useHLS` prop to Video
- `src/settings.ts:34-54,58-87,509-530` — Add `useHLS` setting
- `package.json` — Add `hls.js` dependency

---

### Task 1: HLS Cache Utilities and Path Computation (Go)

**Files:**
- Create: `media-server/hls.go`
- Create: `media-server/hls_test.go`

- [ ] **Step 1: Write tests for HLS cache path computation and validation**

Create `media-server/hls_test.go`:

```go
package main

import (
	"testing"
)

func TestHlsCacheDir(t *testing.T) {
	dir := hlsCacheDir("/base", "/path/to/video.mp4")
	if dir == "" {
		t.Fatal("expected non-empty cache dir")
	}
	dir2 := hlsCacheDir("/base", "/path/to/video.mp4")
	if dir != dir2 {
		t.Fatalf("expected deterministic hash, got %s vs %s", dir, dir2)
	}
	dir3 := hlsCacheDir("/base", "/path/to/other.mp4")
	if dir == dir3 {
		t.Fatal("expected different hash for different input")
	}
}

func TestValidateHlsFilename(t *testing.T) {
	valid := []string{"master.m3u8", "stream.m3u8", "segment_000.ts", "segment_123.ts"}
	for _, f := range valid {
		if !isValidHlsFilename(f) {
			t.Errorf("expected %q to be valid", f)
		}
	}
	invalid := []string{"../etc/passwd", "foo.exe", "segment_.ts", "stream.ts", "master.ts", "../../secret.m3u8"}
	for _, f := range invalid {
		if isValidHlsFilename(f) {
			t.Errorf("expected %q to be invalid", f)
		}
	}
}

func TestValidateHlsPreset(t *testing.T) {
	valid := []string{"passthrough", "480p", "720p", "1080p"}
	for _, p := range valid {
		if !isValidHlsPreset(p) {
			t.Errorf("expected %q to be valid preset", p)
		}
	}
	invalid := []string{"../hack", "4k", "", "PASSTHROUGH"}
	for _, p := range invalid {
		if isValidHlsPreset(p) {
			t.Errorf("expected %q to be invalid preset", p)
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test -run "TestHlsCacheDir|TestValidateHls" -v`
Expected: Compilation error — functions not defined yet

- [ ] **Step 3: Implement cache utilities**

Create `media-server/hls.go`:

```go
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
	"github.com/stevecastle/shrike/renderer"
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
// Key is the cache directory path, value is a result channel.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test -run "TestHlsCacheDir|TestValidateHls" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/hls.go media-server/hls_test.go
git commit -m "feat(hls): add cache path computation and validation utilities"
```

---

### Task 2: HLS Generation Task (Go)

**Files:**
- Create: `media-server/tasks/hls.go`
- Modify: `media-server/tasks/registry.go:33`

- [ ] **Step 1: Create the HLS task**

Create `media-server/tasks/hls.go`. This file contains all preset definitions (single source of truth), the ffprobe probe function, ffmpeg arg builders, master playlist writer, and the main task function.

```go
package tasks

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

type hlsMeta struct {
	SourceMtime int64    `json:"source_mtime"`
	GeneratedAt int64    `json:"generated_at"`
	Presets     []string `json:"presets"`
}

type probeResult struct {
	Width    int
	Height   int
	HasVideo bool
	HasAudio bool
	VCodec   string
}

// HlsPresetDef describes a transcoding preset. Exported for use by HTTP handlers.
type HlsPresetDef struct {
	Width        int
	Height       int
	VideoBitrate string
	AudioBitrate string
}

// HlsPresetDefs is the single source of truth for HLS quality presets.
var HlsPresetDefs = map[string]HlsPresetDef{
	"480p":  {854, 480, "1000k", "128k"},
	"720p":  {1280, 720, "3000k", "192k"},
	"1080p": {1920, 1080, "8000k", "256k"},
}

// hlsTask generates HLS segments for media files.
func hlsTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	var files []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("hls: using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQuery(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "hls: failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		files = mediaPaths
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "hls: no input paths or query provided")
			q.CompleteJob(j.ID)
			return nil
		}
		files = parseInputPaths(raw)
	}

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "hls: no files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	// Parse arguments
	presetMode := "passthrough"
	var selectedPresets []string
	for i := 0; i < len(j.Arguments); i++ {
		switch j.Arguments[i] {
		case "--preset":
			if i+1 < len(j.Arguments) {
				i++
				presetMode = j.Arguments[i]
			}
		case "--presets":
			if i+1 < len(j.Arguments) {
				i++
				selectedPresets = strings.Split(j.Arguments[i], ",")
			}
		}
	}

	basePath := platform.GetDataDir()

	for _, src := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "hls: task canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		default:
		}

		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}

		q.PushJobStdout(j.ID, "hls: processing "+filepath.Base(abs))

		// Check cache validity
		h := sha256.Sum256([]byte(abs))
		outDir := filepath.Join(basePath, "hls", fmt.Sprintf("%x", h))
		metaPath := filepath.Join(outDir, ".meta")

		srcInfo, err := os.Stat(abs)
		if err != nil {
			q.PushJobStdout(j.ID, "hls: cannot stat source: "+err.Error())
			continue
		}

		if metaBytes, err := os.ReadFile(metaPath); err == nil {
			var meta hlsMeta
			if json.Unmarshal(metaBytes, &meta) == nil {
				if meta.SourceMtime == srcInfo.ModTime().Unix() {
					q.PushJobStdout(j.ID, "hls: cache is current, skipping "+filepath.Base(abs))
					continue
				}
			}
		}

		// Probe source
		probe, err := probeMedia(ctx, abs)
		if err != nil {
			q.PushJobStdout(j.ID, "hls: probe failed: "+err.Error())
			continue
		}

		// Determine which presets to generate
		presetsToGen := []string{"passthrough"}
		if presetMode == "adaptive" {
			candidates := selectedPresets
			if len(candidates) == 0 {
				candidates = []string{"480p", "720p", "1080p"}
			}
			for _, p := range candidates {
				def, ok := HlsPresetDefs[p]
				if !ok {
					continue
				}
				if probe.HasVideo && probe.Height >= def.Height {
					presetsToGen = append(presetsToGen, p)
				}
			}
		}

		// Generate each preset
		for _, preset := range presetsToGen {
			select {
			case <-ctx.Done():
				q.PushJobStdout(j.ID, "hls: task canceled")
				q.ErrorJob(j.ID)
				return ctx.Err()
			default:
			}

			presetDir := filepath.Join(outDir, preset)
			os.MkdirAll(presetDir, 0755)

			q.PushJobStdout(j.ID, fmt.Sprintf("hls: generating %s preset for %s", preset, filepath.Base(abs)))

			var args []string
			if preset == "passthrough" {
				args = hlsBuildPassthroughArgs(abs, presetDir)
			} else {
				def := HlsPresetDefs[preset]
				args = hlsBuildTranscodeArgs(abs, presetDir, def.Width, def.Height, def.VideoBitrate, def.AudioBitrate)
			}

			cmd, err := deps.GetExec(ctx, "ffmpeg", "ffmpeg", args...)
			if err != nil {
				q.PushJobStdout(j.ID, "hls: ffmpeg not available: "+err.Error())
				q.ErrorJob(j.ID)
				return err
			}

			stderr, err := cmd.StderrPipe()
			if err != nil {
				q.PushJobStdout(j.ID, "hls: stderr pipe error: "+err.Error())
				continue
			}

			if err := cmd.Start(); err != nil {
				q.PushJobStdout(j.ID, "hls: ffmpeg start error: "+err.Error())
				continue
			}

			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				line := scanner.Text()
				if strings.Contains(line, "time=") || strings.Contains(line, "error") || strings.Contains(line, "Error") {
					q.PushJobStdout(j.ID, "hls: "+line)
				}
			}

			if err := cmd.Wait(); err != nil {
				// If passthrough failed (incompatible codecs), fall back to transcode
				if preset == "passthrough" && probe.HasVideo {
					q.PushJobStdout(j.ID, "hls: passthrough remux failed, falling back to transcode")
					transcodeArgs := hlsBuildTranscodeArgs(abs, presetDir, probe.Width, probe.Height, "8000k", "256k")
					cmd2, err2 := deps.GetExec(ctx, "ffmpeg", "ffmpeg", transcodeArgs...)
					if err2 != nil {
						q.PushJobStdout(j.ID, "hls: transcode fallback failed: "+err2.Error())
						continue
					}
					stderr2, _ := cmd2.StderrPipe()
					cmd2.Start()
					s2 := bufio.NewScanner(stderr2)
					for s2.Scan() {
						line := s2.Text()
						if strings.Contains(line, "time=") || strings.Contains(line, "error") {
							q.PushJobStdout(j.ID, "hls: "+line)
						}
					}
					if err := cmd2.Wait(); err != nil {
						q.PushJobStdout(j.ID, "hls: transcode fallback also failed: "+err.Error())
						continue
					}
				} else {
					q.PushJobStdout(j.ID, "hls: ffmpeg error: "+err.Error())
					continue
				}
			}

			q.PushJobStdout(j.ID, fmt.Sprintf("hls: completed %s preset", preset))
		}

		// Write master playlist
		hlsWriteMasterPlaylist(outDir, presetsToGen, probe, q, j.ID)

		// Write .meta
		meta := hlsMeta{
			SourceMtime: srcInfo.ModTime().Unix(),
			GeneratedAt: time.Now().Unix(),
			Presets:     presetsToGen,
		}
		metaBytes, _ := json.Marshal(meta)
		os.WriteFile(metaPath, metaBytes, 0644)

		q.PushJobStdout(j.ID, "hls: finished "+filepath.Base(abs))
	}

	q.CompleteJob(j.ID)
	return nil
}

// probeMedia uses ffprobe to get source media info.
func probeMedia(ctx context.Context, path string) (*probeResult, error) {
	ffprobePath := deps.GetFFprobePath()
	if ffprobePath == "" {
		return nil, fmt.Errorf("ffprobe not found")
	}

	cmd := exec.CommandContext(ctx, ffprobePath,
		"-v", "quiet",
		"-print_format", "json",
		"-show_streams",
		path,
	)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w", err)
	}

	var data struct {
		Streams []struct {
			CodecType string `json:"codec_type"`
			CodecName string `json:"codec_name"`
			Width     int    `json:"width"`
			Height    int    `json:"height"`
		} `json:"streams"`
	}
	if err := json.Unmarshal(out, &data); err != nil {
		return nil, fmt.Errorf("ffprobe parse error: %w", err)
	}

	result := &probeResult{}
	for _, s := range data.Streams {
		switch s.CodecType {
		case "video":
			result.HasVideo = true
			result.Width = s.Width
			result.Height = s.Height
			result.VCodec = s.CodecName
		case "audio":
			result.HasAudio = true
		}
	}
	return result, nil
}

func hlsBuildPassthroughArgs(input, outDir string) []string {
	return []string{
		"-y", "-i", input,
		"-c", "copy",
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", filepath.Join(outDir, "segment_%03d.ts"),
		filepath.Join(outDir, "stream.m3u8"),
	}
}

func hlsBuildTranscodeArgs(input, outDir string, width, height int, vBitrate, aBitrate string) []string {
	scaleFilter := fmt.Sprintf("scale=%d:-2:force_original_aspect_ratio=decrease", width)
	return []string{
		"-y", "-i", input,
		"-vf", scaleFilter,
		"-c:v", "libx264", "-b:v", vBitrate, "-preset", "medium",
		"-c:a", "aac", "-b:a", aBitrate,
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", filepath.Join(outDir, "segment_%03d.ts"),
		filepath.Join(outDir, "stream.m3u8"),
	}
}

func hlsWriteMasterPlaylist(outDir string, presets []string, probe *probeResult, q *jobqueue.Queue, jobID string) {
	var lines []string
	lines = append(lines, "#EXTM3U")
	lines = append(lines, "#EXT-X-VERSION:3")

	for _, preset := range presets {
		var bw int
		var res string
		if preset == "passthrough" {
			bw = 10000000
			if probe.HasVideo {
				res = fmt.Sprintf("%dx%d", probe.Width, probe.Height)
			}
		} else if def, ok := HlsPresetDefs[preset]; ok {
			bitrate, _ := strconv.Atoi(strings.TrimSuffix(def.VideoBitrate, "k"))
			bw = bitrate * 1000
			res = fmt.Sprintf("%dx%d", def.Width, def.Height)
		}

		inf := fmt.Sprintf("#EXT-X-STREAM-INF:BANDWIDTH=%d", bw)
		if res != "" {
			inf += ",RESOLUTION=" + res
		}
		lines = append(lines, "")
		lines = append(lines, inf)
		lines = append(lines, preset+"/stream.m3u8")
	}

	content := strings.Join(lines, "\n") + "\n"
	masterPath := filepath.Join(outDir, "master.m3u8")
	if err := os.WriteFile(masterPath, []byte(content), 0644); err != nil {
		q.PushJobStdout(jobID, "hls: failed to write master playlist: "+err.Error())
	}
}

// HlsBuildPassthroughArgs is exported for use by the HTTP handler's on-demand generation.
func HlsBuildPassthroughArgs(input, outDir string) []string {
	return hlsBuildPassthroughArgs(input, outDir)
}
```

- [ ] **Step 2: Register the task**

Modify `media-server/tasks/registry.go:33` — add after the `lora-dataset` line:

```go
	RegisterTask("hls", "HLS Transcode", hlsTask)
```

- [ ] **Step 3: Verify the task compiles**

Run: `cd media-server && go build ./...`
Expected: No compilation errors

- [ ] **Step 4: Commit**

```bash
git add media-server/tasks/hls.go media-server/tasks/registry.go
git commit -m "feat(hls): add HLS generation task with ffmpeg"
```

---

### Task 3: HLS HTTP Handlers (Go)

**Files:**
- Modify: `media-server/hls.go` (append handlers to existing file from Task 1)
- Modify: `media-server/main.go:2731`
- Modify: `media-server/main_linux.go:2320`
- Modify: `media-server/main_darwin.go:2320`

- [ ] **Step 1: Add HTTP handlers to hls.go**

Append to `media-server/hls.go`. The `hlsHandler` is a single handler that dispatches on HTTP method (GET for master playlist, DELETE for cleanup):

```go
// hlsHandler handles the /media/hls endpoint.
// GET: returns master playlist (generates on-demand if not cached)
// DELETE: clears HLS cache (for one file if path given, or all)
func hlsHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			hlsServeMaster(w, r)
		case http.MethodDelete:
			hlsCleanup(w, r)
		default:
			http.Error(w, "Use GET or DELETE", http.StatusMethodNotAllowed)
		}
	}
}

func hlsServeMaster(w http.ResponseWriter, r *http.Request) {
	mediaPath := r.URL.Query().Get("path")
	if mediaPath == "" {
		http.Error(w, "path parameter required", http.StatusBadRequest)
		return
	}

	base := hlsBasePath()
	cacheDir := hlsCacheDir(base, mediaPath)
	masterPath := filepath.Join(cacheDir, "master.m3u8")

	// Check if cached
	if _, err := os.Stat(masterPath); os.IsNotExist(err) {
		// Generate passthrough on-the-fly with deduplication
		hlsInflightMu.Lock()
		if entry, ok := hlsInflight[cacheDir]; ok {
			hlsInflightMu.Unlock()
			<-entry.done
			if entry.err != nil {
				http.Error(w, "HLS generation failed: "+entry.err.Error(), http.StatusInternalServerError)
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
				http.Error(w, "HLS generation failed: "+genErr.Error(), http.StatusInternalServerError)
				return
			}
		}
	}

	if _, err := os.Stat(masterPath); err != nil {
		http.Error(w, "HLS master playlist not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	http.ServeFile(w, r, masterPath)
}

func hlsCleanup(w http.ResponseWriter, r *http.Request) {
	base := hlsBasePath()
	mediaPath := r.URL.Query().Get("path")

	if mediaPath != "" {
		cacheDir := hlsCacheDir(base, mediaPath)
		os.RemoveAll(cacheDir)
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "cleared HLS cache for %s", mediaPath)
	} else {
		hlsDir := filepath.Join(base, "hls")
		os.RemoveAll(hlsDir)
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "cleared all HLS cache")
	}
}

// hlsSegmentHandler serves individual HLS files (playlists and segments).
// Matches paths like /media/hls/<hash>/master.m3u8 or /media/hls/<hash>/<preset>/<filename>
func hlsSegmentHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		path := strings.TrimPrefix(r.URL.Path, "/media/hls/")
		parts := strings.Split(path, "/")

		if len(parts) < 2 {
			http.Error(w, "invalid HLS path", http.StatusBadRequest)
			return
		}

		hash := parts[0]
		// Validate hash is hex
		for _, c := range hash {
			if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')) {
				http.Error(w, "invalid hash", http.StatusBadRequest)
				return
			}
		}

		var filePath string
		if len(parts) == 2 {
			// /media/hls/<hash>/master.m3u8
			if !isValidHlsFilename(parts[1]) {
				http.Error(w, "invalid filename", http.StatusBadRequest)
				return
			}
			filePath = filepath.Join(hlsBasePath(), "hls", hash, parts[1])
		} else if len(parts) == 3 {
			// /media/hls/<hash>/<preset>/<filename>
			preset := parts[1]
			filename := parts[2]
			if !isValidHlsPreset(preset) || !isValidHlsFilename(filename) {
				http.Error(w, "invalid preset or filename", http.StatusBadRequest)
				return
			}
			filePath = filepath.Join(hlsBasePath(), "hls", hash, preset, filename)
		} else {
			http.Error(w, "invalid HLS path", http.StatusBadRequest)
			return
		}

		if _, err := os.Stat(filePath); err != nil {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}

		if strings.HasSuffix(filePath, ".m3u8") {
			w.Header().Set("Content-Type", "application/vnd.apple.mpegurl")
		} else if strings.HasSuffix(filePath, ".ts") {
			w.Header().Set("Content-Type", "video/MP2T")
		}
		w.Header().Set("Access-Control-Allow-Origin", "*")

		http.ServeFile(w, r, filePath)
	}
}

// generatePassthroughHLS generates passthrough HLS for a single media file.
func generatePassthroughHLS(mediaPath, cacheDir string) error {
	presetDir := filepath.Join(cacheDir, "passthrough")
	os.MkdirAll(presetDir, 0755)

	ffmpegPath := depspkg.GetFFmpegPath()
	if ffmpegPath == "" {
		return fmt.Errorf("ffmpeg not found")
	}

	args := []string{
		"-y", "-i", mediaPath,
		"-c", "copy",
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", filepath.Join(presetDir, "segment_%03d.ts"),
		filepath.Join(presetDir, "stream.m3u8"),
	}

	cmd := exec.Command(ffmpegPath, args...)
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg failed: %w", err)
	}

	// Write minimal master playlist
	master := "#EXTM3U\n#EXT-X-VERSION:3\n\n#EXT-X-STREAM-INF:BANDWIDTH=10000000\npassthrough/stream.m3u8\n"
	return os.WriteFile(filepath.Join(cacheDir, "master.m3u8"), []byte(master), 0644)
}
```

- [ ] **Step 2: Register routes in all three platform main files**

In `media-server/main.go`, after line 2731 (the `/media/file` handler line), add:

```go
	mux.HandleFunc("/media/hls", renderer.ApplyMiddlewares(hlsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/hls/", renderer.ApplyMiddlewares(hlsSegmentHandler(deps), renderer.RoleAdmin))
```

In `media-server/main_linux.go`, after line 2320 (the `/media/file` handler line), add the same two lines.

In `media-server/main_darwin.go`, after line 2320 (the `/media/file` handler line), add the same two lines.

- [ ] **Step 3: Verify compilation**

Run: `cd media-server && go build ./...`
Expected: No compilation errors

- [ ] **Step 4: Commit**

```bash
git add media-server/hls.go media-server/main.go media-server/main_linux.go media-server/main_darwin.go
git commit -m "feat(hls): add HTTP handlers for HLS manifest and segment serving"
```

---

### Task 4: Settings — Add useHLS Toggle (TypeScript)

**Files:**
- Modify: `src/settings.ts:34-54,58-87,509-530`
- Modify: `src/renderer/state.tsx:334-415`

- [ ] **Step 1: Add useHLS to SettingKey union**

In `src/settings.ts`, add `'useHLS'` to the `SettingKey` union (after `'layoutMode'` around line 54):

```typescript
  | 'useHLS';
```

- [ ] **Step 2: Add useHLS to Settings type**

In `src/settings.ts`, add to the `Settings` type (after `layoutMode` around line 87):

```typescript
  useHLS: boolean;
```

- [ ] **Step 3: Add useHLS default setting constant and SETTINGS entry**

Add the setting constant near the other setting constants (before the `SETTINGS` object around line 509):

```typescript
const USE_HLS: Setting<boolean> = {
  name: 'useHLS',
  defaultValue: false,
};
```

Add to the `SETTINGS` object (after `layoutMode: LAYOUT_MODE`):

```typescript
  useHLS: USE_HLS,
```

- [ ] **Step 4: Add useHLS to getInitialContext defaults**

In `src/renderer/state.tsx`, inside `getInitialContext` where `store.getMany()` is called (around line 334-415), add alongside the other settings defaults:

```typescript
  useHLS: false,
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `npx webpack --mode development`
Expected: No type errors related to useHLS

- [ ] **Step 6: Commit**

```bash
git add src/settings.ts src/renderer/state.tsx
git commit -m "feat(hls): add useHLS setting toggle"
```

---

### Task 5: Platform Function — hlsUrl (TypeScript)

**Files:**
- Modify: `src/renderer/platform.ts:236,393`

HLS is web-mode only. In Electron mode, the Go server does not run — files are served via the `gsm://` custom protocol which does not support HLS. The `hlsUrl` function returns `null` in Electron mode, and the video player uses this to determine HLS availability.

- [ ] **Step 1: Add hlsUrl declaration**

Near line 236 in `src/renderer/platform.ts`, alongside the `mediaUrl` declaration, add:

```typescript
export let hlsUrl: ((path: string) => string) | null;
```

- [ ] **Step 2: Set Electron mode to null**

In the Electron mode block (after line 320, after the `mediaUrl` assignment), add:

```typescript
  hlsUrl = null; // HLS not available in Electron mode (no Go server)
```

- [ ] **Step 3: Add Web mode implementation**

In the Web mode block (after line 397, after the `mediaUrl` assignment), add:

```typescript
  hlsUrl = (path) =>
    `/media/hls?path=${encodeURIComponent(path)}`;
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `npx webpack --mode development`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/renderer/platform.ts
git commit -m "feat(hls): add hlsUrl platform function (web mode only)"
```

---

### Task 6: Install hls.js Dependency

**Files:**
- Modify: `package.json` (root)

- [ ] **Step 1: Install hls.js**

Run: `npm install hls.js`

- [ ] **Step 2: Verify installation**

Run: `node -e "require('hls.js'); console.log('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "feat(hls): add hls.js dependency"
```

---

### Task 7: Video Player — hls.js Integration (TypeScript/React)

**Files:**
- Modify: `src/renderer/components/media-viewers/video.tsx:1-307`

- [ ] **Step 1: Add hls.js import and useHLS prop**

At the top of `video.tsx`, add the import:

```typescript
import Hls from 'hls.js';
```

Update the platform import to include `hlsUrl`:

```typescript
import { mediaUrl, hlsUrl, fetchMediaPreview as platformFetchMediaPreview } from '../../platform';
```

Add `useHLS` to the Props type (around line 15-33):

```typescript
  useHLS?: boolean;
```

Add `useHLS = false` to the destructured props in the component function signature (around line 74).

- [ ] **Step 2: Add hls.js lifecycle hook**

Inside the `Video` component, after the volume `useEffect` (around line 184) and before the `if (error)` check (around line 186), add:

```typescript
  const hlsRef = useRef<Hls | null>(null);
  const [hlsFailed, setHlsFailed] = useState(false);

  // Reset hlsFailed when path changes
  useEffect(() => {
    setHlsFailed(false);
  }, [path]);

  useEffect(() => {
    if (!useHLS || hlsFailed || !mediaRef?.current || cache || !hlsUrl) return;

    const video = mediaRef.current;

    if (Hls.isSupported()) {
      const hls = new Hls({
        enableWorker: true,
        lowLatencyMode: false,
      });
      hlsRef.current = hls;

      hls.loadSource(hlsUrl(path));
      hls.attachMedia(video);

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(() => {});
      });

      hls.on(Hls.Events.ERROR, (_event, data) => {
        if (data.fatal) {
          console.log('hls.js fatal error:', data.type, data.details);
          hls.destroy();
          hlsRef.current = null;
          setHlsFailed(true);
        }
      });

      return () => {
        hls.destroy();
        hlsRef.current = null;
      };
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Safari native HLS
      video.src = hlsUrl(path);
    }
  }, [useHLS, hlsFailed, path, mediaRef, cache]);
```

- [ ] **Step 3: Modify the video src logic**

In the non-cache render path (around line 254, the `<video>` element), change the `src` attribute.

From:
```typescript
          src={mediaUrl(path)}
```

To:
```typescript
          src={useHLS && !hlsFailed && hlsUrl ? undefined : mediaUrl(path)}
```

When hls.js is active, it manages the source via `attachMedia`, so `src` is `undefined`. When HLS is disabled, unavailable (Electron), or has failed, it falls back to `mediaUrl(path)`.

- [ ] **Step 4: Verify TypeScript compiles**

Run: `npx webpack --mode development`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/media-viewers/video.tsx
git commit -m "feat(hls): integrate hls.js in video player with fallback"
```

---

### Task 8: Wire useHLS Setting to Video Component

**Files:**
- Modify: `src/renderer/components/detail/detail.tsx:68-81`

The detail view renders the `<Video>` component at line 68. The `settings` object is already available in scope (passed as parameter at line 60). The `useHLS` value comes from `settings.useHLS`.

- [ ] **Step 1: Pass useHLS prop to Video**

In `src/renderer/components/detail/detail.tsx`, find the `<Video>` element at line 68-81. Add `useHLS` prop:

```typescript
      <Video
        key={path}
        path={path}
        scaleMode={settings.scaleMode}
        settable
        coverSize={coverSize}
        handleLoad={handleLoad}
        mediaRef={mediaRef as React.RefObject<HTMLVideoElement>}
        playSound={settings.playSound}
        volume={settings.volume}
        showControls={settings.showControls}
        orientation={orientation}
        startTime={startTime}
        useHLS={settings.useHLS}
      />
```

Note: Do NOT add `useHLS` to the list/thumbnail Video usages (in `list-item.tsx`, `duplicates.tsx`, `thumbnails.tsx`). Those use cached thumbnails, not direct playback, so HLS is not applicable there.

- [ ] **Step 2: Verify TypeScript compiles**

Run: `npx webpack --mode development`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/renderer/components/detail/detail.tsx
git commit -m "feat(hls): wire useHLS setting to detail view video component"
```

---

### Task 9: End-to-End Testing and Cleanup

- [ ] **Step 1: Verify Go server builds**

Run: `cd media-server && go build ./...`
Expected: Clean build

- [ ] **Step 2: Run Go tests**

Run: `cd media-server && go test ./...`
Expected: All tests pass

- [ ] **Step 3: Verify TypeScript builds**

Run: `npx webpack --mode development`
Expected: Clean build

- [ ] **Step 4: Manual end-to-end test**

1. Start the Go server and web client
2. Ensure a video file is available in the media library
3. With useHLS=false: video plays directly (existing behavior unchanged)
4. With useHLS=true: video plays via HLS (passthrough generates on first request)
5. Seek to various positions in HLS mode — confirm seeking works
6. Switch between videos — confirm proper hls.js cleanup and reload
7. Test with a VP9/WebM file — confirm passthrough fallback to transcode works
8. Hit `DELETE /media/hls` — confirm cache is cleared
9. In Electron mode: confirm HLS is disabled and direct playback works normally

- [ ] **Step 5: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "feat(hls): end-to-end cleanup and fixes"
```
