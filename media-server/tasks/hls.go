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
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// HlsPreset holds the parameters for a single HLS quality tier.
type HlsPreset struct {
	Width    int
	Height   int
	Bitrate  string
	ABitrate string
}

// HlsPresetDefs is the exported single source of truth for HLS quality tiers.
var HlsPresetDefs = map[string]HlsPreset{
	"480p":  {854, 480, "1000k", "128k"},
	"720p":  {1280, 720, "3000k", "192k"},
	"1080p": {1920, 1080, "8000k", "256k"},
}

// hlsMetaInfo is persisted alongside HLS output to detect stale cache.
type hlsMetaInfo struct {
	SourceMtime int64    `json:"source_mtime"`
	Presets     []string `json:"presets"`
	GeneratedAt int64    `json:"generated_at"`
}

// hlsTask is the main task function for HLS generation.
func hlsTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	ctx := j.Ctx

	// Parse --preset and --presets from arguments.
	presetMode := "passthrough" // default
	var requestedPresets []string
	for i, arg := range j.Arguments {
		lower := strings.ToLower(arg)
		switch lower {
		case "--preset":
			if i+1 < len(j.Arguments) {
				presetMode = strings.TrimSpace(j.Arguments[i+1])
			}
		case "--presets":
			if i+1 < len(j.Arguments) {
				for _, p := range strings.Split(j.Arguments[i+1], ",") {
					p = strings.TrimSpace(p)
					if p != "" {
						requestedPresets = append(requestedPresets, p)
					}
				}
			}
		}
		if strings.HasPrefix(lower, "--preset=") {
			presetMode = strings.TrimSpace(arg[len("--preset="):])
		}
		if strings.HasPrefix(lower, "--presets=") {
			for _, p := range strings.Split(arg[len("--presets="):], ",") {
				p = strings.TrimSpace(p)
				if p != "" {
					requestedPresets = append(requestedPresets, p)
				}
			}
		}
	}

	// If no explicit presets list given, derive from presetMode.
	if len(requestedPresets) == 0 {
		if presetMode == "adaptive" {
			requestedPresets = []string{"480p", "720p", "1080p"}
		}
		// passthrough only generates the passthrough tier; handled below per file.
	}

	// Gather files.
	var files []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("hls: using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
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

	ffprobePath := deps.GetFFprobePath()

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

		base := filepath.Base(abs)
		q.PushJobStdout(j.ID, fmt.Sprintf("hls: processing %s", base))

		// Compute output dir.
		h := sha256.Sum256([]byte(abs))
		outDir := filepath.Join(platform.GetDataDir(), "hls", fmt.Sprintf("%x", h))

		// Cache check: compare source mtime against .meta JSON.
		metaPath := filepath.Join(outDir, ".meta")
		srcStat, err := os.Stat(abs)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("hls: cannot stat source %s: %v", base, err))
			continue
		}
		srcMtime := srcStat.ModTime().Unix()

		if metaData, err := os.ReadFile(metaPath); err == nil {
			var meta hlsMetaInfo
			if json.Unmarshal(metaData, &meta) == nil && meta.SourceMtime == srcMtime {
				q.PushJobStdout(j.ID, fmt.Sprintf("hls: cache is current for %s, skipping", base))
				continue
			}
		}

		// Probe source.
		probeInfo, err := probeMedia(ctx, ffprobePath, abs)
		if err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("hls: probe failed for %s: %v", base, err))
			continue
		}

		// Determine which presets to generate.
		// Always include passthrough; skip transcoded presets for audio-only files.
		presetsToRun := []string{"passthrough"}
		if probeInfo.hasVideo {
			for _, p := range requestedPresets {
				if def, ok := HlsPresetDefs[p]; ok {
					if probeInfo.height >= def.Height {
						presetsToRun = append(presetsToRun, p)
					}
				}
			}
		}

		if err := os.MkdirAll(outDir, 0755); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("hls: failed to create output dir: %v", err))
			continue
		}

		generatedPresets := []string{}

		for _, preset := range presetsToRun {
			select {
			case <-ctx.Done():
				q.PushJobStdout(j.ID, "hls: task canceled")
				q.ErrorJob(j.ID)
				return ctx.Err()
			default:
			}

			presetDir := filepath.Join(outDir, preset)
			if err := os.MkdirAll(presetDir, 0755); err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("hls: failed to create preset dir %s: %v", preset, err))
				continue
			}

			playlistPath := filepath.Join(presetDir, "stream.m3u8")
			segmentPattern := filepath.Join(presetDir, "segment_%03d.ts")

			var ffmpegArgs []string
			if preset == "passthrough" {
				ffmpegArgs = hlsBuildPassthroughArgs(abs, playlistPath, segmentPattern)
			} else {
				def := HlsPresetDefs[preset]
				ffmpegArgs = hlsBuildTranscodeArgs(abs, playlistPath, segmentPattern, def)
			}

			q.PushJobStdout(j.ID, fmt.Sprintf("hls: running preset %s for %s", preset, base))
			runErr := hlsRunFFmpeg(ctx, j.ID, q, ffmpegArgs)

			if runErr != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("hls: preset %s failed for %s: %v", preset, base, runErr))
				if preset == "passthrough" && probeInfo.hasVideo {
					// Fall back to transcoding at source resolution (video files only).
					q.PushJobStdout(j.ID, fmt.Sprintf("hls: falling back to transcode at source resolution for %s", base))
					fallbackPreset := HlsPreset{
						Width:    probeInfo.width,
						Height:   probeInfo.height,
						Bitrate:  "8000k",
						ABitrate: "256k",
					}
					fallbackArgs := hlsBuildTranscodeArgs(abs, playlistPath, segmentPattern, fallbackPreset)
					if fbErr := hlsRunFFmpeg(ctx, j.ID, q, fallbackArgs); fbErr != nil {
						q.PushJobStdout(j.ID, fmt.Sprintf("hls: fallback transcode also failed for %s: %v", base, fbErr))
					} else {
						generatedPresets = append(generatedPresets, preset)
					}
				}
				continue
			}
			generatedPresets = append(generatedPresets, preset)
		}

		if len(generatedPresets) == 0 {
			q.PushJobStdout(j.ID, fmt.Sprintf("hls: no presets generated for %s", base))
			continue
		}

		// Write master playlist.
		if err := hlsWriteMasterPlaylist(outDir, generatedPresets); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("hls: failed to write master playlist for %s: %v", base, err))
			continue
		}

		// Write .meta cache file.
		meta := hlsMetaInfo{
			SourceMtime: srcMtime,
			Presets:     generatedPresets,
			GeneratedAt: time.Now().Unix(),
		}
		if metaBytes, err := json.Marshal(meta); err == nil {
			_ = os.WriteFile(metaPath, metaBytes, 0644)
		}

		masterPath := filepath.Join(outDir, "master.m3u8")
		q.PushJobStdout(j.ID, fmt.Sprintf("hls: completed %s (presets: %s)", base, strings.Join(generatedPresets, ", ")))
		// Output master playlist path for downstream chaining
		q.PushJobStdout(j.ID, masterPath)
	}

	q.CompleteJob(j.ID)
	return nil
}

// hlsProbeInfo holds the result of ffprobe analysis.
type hlsProbeInfo struct {
	width    int
	height   int
	hasVideo bool
	hasAudio bool
	vcodec   string
	acodec   string
}

// probeMedia runs ffprobe to extract stream metadata from a media file.
func probeMedia(ctx context.Context, ffprobePath, src string) (hlsProbeInfo, error) {
	args := []string{
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=width,height,codec_name",
		"-of", "json",
		src,
	}
	cmd := exec.CommandContext(ctx, ffprobePath, args...)
	out, err := cmd.Output()
	if err != nil {
		return hlsProbeInfo{}, fmt.Errorf("ffprobe video stream: %w", err)
	}

	var videoResult struct {
		Streams []struct {
			Width     int    `json:"width"`
			Height    int    `json:"height"`
			CodecName string `json:"codec_name"`
		} `json:"streams"`
	}
	if err := json.Unmarshal(out, &videoResult); err != nil {
		return hlsProbeInfo{}, fmt.Errorf("ffprobe video json: %w", err)
	}

	info := hlsProbeInfo{}
	if len(videoResult.Streams) > 0 {
		info.hasVideo = true
		info.width = videoResult.Streams[0].Width
		info.height = videoResult.Streams[0].Height
		info.vcodec = videoResult.Streams[0].CodecName
	}

	// Probe audio stream.
	aargs := []string{
		"-v", "error",
		"-select_streams", "a:0",
		"-show_entries", "stream=codec_name",
		"-of", "json",
		src,
	}
	acmd := exec.CommandContext(ctx, ffprobePath, aargs...)
	aout, aerr := acmd.Output()
	if aerr == nil {
		var audioResult struct {
			Streams []struct {
				CodecName string `json:"codec_name"`
			} `json:"streams"`
		}
		if json.Unmarshal(aout, &audioResult) == nil && len(audioResult.Streams) > 0 {
			info.hasAudio = true
			info.acodec = audioResult.Streams[0].CodecName
		}
	}

	return info, nil
}

// hlsBuildPassthroughArgs returns ffmpeg arguments for passthrough (stream copy) HLS.
func hlsBuildPassthroughArgs(input, playlistPath, segmentPattern string) []string {
	return []string{
		"-y", "-i", input,
		"-c", "copy",
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", segmentPattern,
		playlistPath,
	}
}

// HlsBuildPassthroughArgs is the exported wrapper for HTTP handler use.
func HlsBuildPassthroughArgs(input, playlistPath, segmentPattern string) []string {
	return hlsBuildPassthroughArgs(input, playlistPath, segmentPattern)
}

// hlsBuildTranscodeArgs returns ffmpeg arguments for transcoded HLS at a given preset.
func hlsBuildTranscodeArgs(input, playlistPath, segmentPattern string, preset HlsPreset) []string {
	vf := fmt.Sprintf("scale=%d:-2:force_original_aspect_ratio=decrease", preset.Width)
	return []string{
		"-y", "-i", input,
		"-vf", vf,
		"-c:v", "libx264",
		"-b:v", preset.Bitrate,
		"-preset", "medium",
		"-c:a", "aac",
		"-b:a", preset.ABitrate,
		"-f", "hls",
		"-hls_time", "6",
		"-hls_segment_type", "mpegts",
		"-hls_playlist_type", "vod",
		"-hls_segment_filename", segmentPattern,
		playlistPath,
	}
}

// hlsRunFFmpeg runs ffmpeg with the given args, streaming stderr to job stdout.
func hlsRunFFmpeg(ctx context.Context, jobID string, q *jobqueue.Queue, args []string) error {
	cmd, err := deps.GetExec(ctx, "ffmpeg", "ffmpeg", args...)
	if err != nil {
		return fmt.Errorf("prepare ffmpeg: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("stderr pipe: %w", err)
	}

	doneErr := make(chan struct{})
	go func() {
		s := bufio.NewScanner(stderr)
		for s.Scan() {
			_ = q.PushJobStdout(jobID, "ffmpeg: "+s.Text())
		}
		close(doneErr)
	}()

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start ffmpeg: %w", err)
	}

	scan := bufio.NewScanner(stdout)
	for scan.Scan() {
		_ = q.PushJobStdout(jobID, scan.Text())
	}
	runErr := cmd.Wait()
	<-doneErr
	return runErr
}

// hlsWriteMasterPlaylist writes a master HLS playlist referencing each generated preset.
func hlsWriteMasterPlaylist(outDir string, presets []string) error {
	var sb strings.Builder
	sb.WriteString("#EXTM3U\n")
	sb.WriteString("#EXT-X-VERSION:3\n")

	for _, preset := range presets {
		if preset == "passthrough" {
			sb.WriteString("#EXT-X-STREAM-INF:BANDWIDTH=0,NAME=\"passthrough\"\n")
			sb.WriteString("passthrough/stream.m3u8\n")
		} else if def, ok := HlsPresetDefs[preset]; ok {
			bw := hlsParseBandwidth(def.Bitrate) + hlsParseBandwidth(def.ABitrate)
			sb.WriteString(fmt.Sprintf(
				"#EXT-X-STREAM-INF:BANDWIDTH=%d,RESOLUTION=%dx%d,NAME=\"%s\"\n",
				bw, def.Width, def.Height, preset,
			))
			sb.WriteString(fmt.Sprintf("%s/stream.m3u8\n", preset))
		}
	}

	masterPath := filepath.Join(outDir, "master.m3u8")
	return os.WriteFile(masterPath, []byte(sb.String()), 0644)
}

// hlsParseBandwidth parses a bitrate string like "3000k" or "8000k" into bits/s.
func hlsParseBandwidth(s string) int {
	s = strings.ToLower(strings.TrimSpace(s))
	if strings.HasSuffix(s, "k") {
		var n int
		fmt.Sscanf(s, "%dk", &n)
		return n * 1000
	}
	if strings.HasSuffix(s, "m") {
		var n int
		fmt.Sscanf(s, "%dm", &n)
		return n * 1000000
	}
	var n int
	fmt.Sscanf(s, "%d", &n)
	return n
}
