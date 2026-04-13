package tasks

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

// isImageExt returns true for image file extensions (case-insensitive).
func isImageExt(ext string) bool {
	switch strings.ToLower(ext) {
	case ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".tif", ".tiff":
		return true
	}
	return false
}

// imageOutputExt returns the output extension for an image file.
// For lossless filter output we use png; passthrough for formats that already work well.
func imageOutputExt(ext string) string {
	switch strings.ToLower(ext) {
	case ".png", ".bmp", ".tif", ".tiff":
		return ".png"
	default:
		return ext // jpg, webp, gif, heic keep their extension
	}
}

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

// --- FFmpeg Extract Audio Clip ---

var ffmpegExtractAudioClipOptions = []TaskOption{
	{Name: "start", Label: "Start Time", Type: "string", Default: "00:00:00", Description: "Start timestamp (HH:MM:SS or seconds)"},
	{Name: "end", Label: "End Time", Type: "string", Default: "00:00:30", Description: "End timestamp (HH:MM:SS or seconds)"},
	{Name: "format", Label: "Audio Format", Type: "enum", Choices: []string{"mp3", "wav", "aac", "flac", "ogg"}, Default: "mp3", Description: "Output audio format"},
}

func ffmpegExtractAudioClipTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegExtractAudioClipOptions)
	start, _ := opts["start"].(string)
	if start == "" {
		start = "00:00:00"
	}
	end, _ := opts["end"].(string)
	if end == "" {
		end = "00:00:30"
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "mp3"
	}
	codec := audioCodecMap[format]
	if codec == "" {
		codec = "libmp3lame"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"_clip."+format)
		return []string{"-ss", start, "-to", end, "-vn", "-acodec", codec, output}, output
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

// --- FFmpeg Reverse ---

func ffmpegReverseTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"_reversed"+ext)
		return []string{"-vf", "reverse", "-af", "areverse", output}, output
	})
}

// --- FFmpeg Speed ---

var ffmpegSpeedOptions = []TaskOption{
	{Name: "factor", Label: "Speed Factor", Type: "number", Default: 2.0, Description: "Playback speed multiplier (e.g. 2 = 2x faster, 0.5 = half speed)"},
}

func ffmpegSpeedTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegSpeedOptions)
	factor, _ := opts["factor"].(float64)
	if factor <= 0 {
		factor = 2.0
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		vf := fmt.Sprintf("setpts=%f*PTS", 1.0/factor)
		af := fmt.Sprintf("atempo=%f", factor)
		output := filepath.Join(dir, name+"_speed"+ext)
		return []string{"-vf", vf, "-af", af, output}, output
	})
}

// --- FFmpeg Grayscale ---

func ffmpegGrayscaleTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		output := filepath.Join(dir, name+"_grayscale"+ext)
		return []string{"-vf", "hue=s=0", output}, output
	})
}

// --- FFmpeg Blur ---

var ffmpegBlurOptions = []TaskOption{
	{Name: "strength", Label: "Blur Strength", Type: "number", Default: 5.0, Description: "Blur radius (higher = more blur)"},
}

func ffmpegBlurTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegBlurOptions)
	strength, _ := opts["strength"].(float64)
	if strength <= 0 {
		strength = 5
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		vf := fmt.Sprintf("boxblur=%d:%d", int(strength), int(strength))
		output := filepath.Join(dir, name+"_blurred"+ext)
		return []string{"-vf", vf, output}, output
	})
}

// --- FFmpeg Resize ---

var ffmpegResizeOptions = []TaskOption{
	{Name: "width", Label: "Width", Type: "number", Default: 1280.0, Description: "Target width in pixels (-1 to keep aspect ratio)"},
	{Name: "height", Label: "Height", Type: "number", Default: -1.0, Description: "Target height in pixels (-1 to keep aspect ratio)"},
}

func ffmpegResizeTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegResizeOptions)
	width, _ := opts["width"].(float64)
	height, _ := opts["height"].(float64)
	if width == 0 && height == 0 {
		width = 1280
		height = -1
	}
	// Ensure at least one dimension is specified
	if width == 0 {
		width = -1
	}
	if height == 0 {
		height = -1
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		// Use scale with divisible-by-2 constraint for video compatibility
		vf := fmt.Sprintf("scale=%d:%d:force_original_aspect_ratio=decrease,pad=ceil(iw/2)*2:ceil(ih/2)*2", int(width), int(height))
		if isImageExt(ext) {
			// Images don't need the pad filter for even dimensions
			vf = fmt.Sprintf("scale=%d:%d", int(width), int(height))
			outExt := imageOutputExt(ext)
			output := filepath.Join(dir, name+"_resized"+outExt)
			return []string{"-vf", vf, output}, output
		}
		output := filepath.Join(dir, name+"_resized"+ext)
		return []string{"-vf", vf, "-c:a", "copy", output}, output
	})
}

// --- FFmpeg Crop ---

var ffmpegCropOptions = []TaskOption{
	{Name: "width", Label: "Crop Width", Type: "number", Default: 0.0, Description: "Width of the crop area (0 = keep original)"},
	{Name: "height", Label: "Crop Height", Type: "number", Default: 0.0, Description: "Height of the crop area (0 = keep original)"},
	{Name: "x", Label: "X Offset", Type: "number", Default: -1.0, Description: "Horizontal offset (-1 = center)"},
	{Name: "y", Label: "Y Offset", Type: "number", Default: -1.0, Description: "Vertical offset (-1 = center)"},
	{Name: "preset", Label: "Preset", Type: "enum", Choices: []string{"custom", "square", "16:9", "9:16", "4:3"}, Default: "custom", Description: "Crop to a preset aspect ratio (overrides width/height)"},
}

func ffmpegCropTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegCropOptions)
	w, _ := opts["width"].(float64)
	h, _ := opts["height"].(float64)
	x, _ := opts["x"].(float64)
	y, _ := opts["y"].(float64)
	preset, _ := opts["preset"].(string)

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		var vf string
		switch preset {
		case "square":
			vf = "crop=min(iw\\,ih):min(iw\\,ih)"
		case "16:9":
			vf = "crop=min(iw\\,ih*16/9):min(ih\\,iw*9/16)"
		case "9:16":
			vf = "crop=min(iw\\,ih*9/16):min(ih\\,iw*16/9)"
		case "4:3":
			vf = "crop=min(iw\\,ih*4/3):min(ih\\,iw*3/4)"
		default:
			cw := "iw"
			ch := "ih"
			if w > 0 {
				cw = fmt.Sprintf("%d", int(w))
			}
			if h > 0 {
				ch = fmt.Sprintf("%d", int(h))
			}
			cx := "(iw-ow)/2"
			cy := "(ih-oh)/2"
			if x >= 0 {
				cx = fmt.Sprintf("%d", int(x))
			}
			if y >= 0 {
				cy = fmt.Sprintf("%d", int(y))
			}
			vf = fmt.Sprintf("crop=%s:%s:%s:%s", cw, ch, cx, cy)
		}

		if isImageExt(ext) {
			outExt := imageOutputExt(ext)
			output := filepath.Join(dir, name+"_cropped"+outExt)
			return []string{"-vf", vf, output}, output
		}
		output := filepath.Join(dir, name+"_cropped"+ext)
		return []string{"-vf", vf, "-c:a", "copy", output}, output
	})
}

// --- FFmpeg Rotate ---

var ffmpegRotateOptions = []TaskOption{
	{Name: "angle", Label: "Rotation", Type: "enum", Choices: []string{"90", "180", "270"}, Default: "90", Description: "Rotation angle in degrees clockwise"},
}

func ffmpegRotateTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegRotateOptions)
	angle, _ := opts["angle"].(string)

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		var vf string
		switch angle {
		case "180":
			vf = "transpose=1,transpose=1"
		case "270":
			vf = "transpose=2"
		default: // "90"
			vf = "transpose=1"
		}

		if isImageExt(ext) {
			outExt := imageOutputExt(ext)
			output := filepath.Join(dir, name+"_rotated"+outExt)
			return []string{"-vf", vf, output}, output
		}
		output := filepath.Join(dir, name+"_rotated"+ext)
		return []string{"-vf", vf, "-c:a", "copy", output}, output
	})
}

// --- FFmpeg Caption ---

var ffmpegCaptionOptions = []TaskOption{
	{Name: "text", Label: "Caption Text", Type: "string", Required: true, Description: "Text to display on the video"},
	{Name: "start", Label: "Start Time", Type: "string", Default: "00:00:00", Description: "When the caption appears (HH:MM:SS or seconds)"},
	{Name: "end", Label: "End Time", Type: "string", Default: "00:00:05", Description: "When the caption disappears (HH:MM:SS or seconds)"},
	{Name: "fontsize", Label: "Font Size", Type: "number", Default: 24.0, Description: "Font size in pixels"},
	{Name: "fontcolor", Label: "Font Color", Type: "string", Default: "white", Description: "Text color (e.g. white, yellow, #RRGGBB)"},
}

func ffmpegCaptionTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegCaptionOptions)
	text, _ := opts["text"].(string)
	if text == "" {
		q.PushJobStdout(j.ID, "ffmpeg-caption: no text provided")
		q.ErrorJob(j.ID)
		return fmt.Errorf("caption text is required")
	}
	start, _ := opts["start"].(string)
	if start == "" {
		start = "00:00:00"
	}
	end, _ := opts["end"].(string)
	if end == "" {
		end = "00:00:05"
	}
	fontsize, _ := opts["fontsize"].(float64)
	if fontsize <= 0 {
		fontsize = 24
	}
	fontcolor, _ := opts["fontcolor"].(string)
	if fontcolor == "" {
		fontcolor = "white"
	}

	return runFFmpegOnFiles(j, q, mu, func(abs, dir, name, ext string) ([]string, string) {
		// Escape special characters for the drawtext filter
		escaped := text
		for _, c := range []string{":", "'", "\\", "%"} {
			escaped = strings.ReplaceAll(escaped, c, `\`+c)
		}

		vf := fmt.Sprintf(
			"drawtext=text='%s':fontsize=%d:fontcolor=%s:borderw=2:bordercolor=black:x=(w-text_w)/2:y=h-text_h-40:enable='between(t,%s,%s)'",
			escaped, int(fontsize), fontcolor, timeToSeconds(start), timeToSeconds(end),
		)

		output := filepath.Join(dir, name+"_captioned"+ext)
		return []string{"-vf", vf, "-c:a", "copy", output}, output
	})
}

// --- FFmpeg Thumbnail Sheet ---

var ffmpegThumbSheetOptions = []TaskOption{
	{Name: "columns", Label: "Columns", Type: "number", Default: 4.0, Description: "Number of columns in the grid"},
	{Name: "rows", Label: "Rows", Type: "number", Default: 4.0, Description: "Number of rows in the grid"},
	{Name: "width", Label: "Tile Width", Type: "number", Default: 320.0, Description: "Width of each thumbnail in pixels"},
	{Name: "format", Label: "Image Format", Type: "enum", Choices: []string{"jpg", "png", "webp"}, Default: "jpg", Description: "Output image format"},
}

func ffmpegThumbSheetTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	opts := ParseOptions(j, ffmpegThumbSheetOptions)
	cols, _ := opts["columns"].(float64)
	if cols <= 0 {
		cols = 4
	}
	rows, _ := opts["rows"].(float64)
	if rows <= 0 {
		rows = 4
	}
	width, _ := opts["width"].(float64)
	if width <= 0 {
		width = 320
	}
	format, _ := opts["format"].(string)
	if format == "" {
		format = "jpg"
	}

	ctx := j.Ctx

	var files []string
	if qstr, ok := extractQueryFromJob(j); ok {
		q.PushJobStdout(j.ID, fmt.Sprintf("thumbsheet: using query to select files: %s", qstr))
		mediaPaths, err := getMediaPathsByQueryFast(q.Db, qstr)
		if err != nil {
			q.PushJobStdout(j.ID, "thumbsheet: failed to load paths from query: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}
		files = mediaPaths
	} else {
		raw := strings.TrimSpace(j.Input)
		if raw == "" {
			q.PushJobStdout(j.ID, "thumbsheet: no input paths or query provided")
			q.CompleteJob(j.ID)
			return nil
		}
		files = parseInputPaths(raw)
	}

	if len(files) == 0 {
		q.PushJobStdout(j.ID, "thumbsheet: no files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	for _, src := range files {
		select {
		case <-ctx.Done():
			q.PushJobStdout(j.ID, "thumbsheet: task canceled")
			q.ErrorJob(j.ID)
			return ctx.Err()
		default:
		}

		abs := src
		if a, err := filepath.Abs(src); err == nil {
			abs = filepath.FromSlash(a)
		}
		ext := filepath.Ext(abs)

		if isImageExt(ext) {
			q.PushJobStdout(j.ID, "thumbsheet: skipping image "+filepath.Base(abs))
			continue
		}

		dir := filepath.Dir(abs)
		base := filepath.Base(abs)
		name := strings.TrimSuffix(base, ext)

		outputDir := dir
		if j.WorkflowID != "" {
			originalDir := stripLokiTemp(abs)
			outputDir = filepath.Join(originalDir, ".loki-temp", j.ID)
			if err := os.MkdirAll(outputDir, 0755); err != nil {
				q.PushJobStdout(j.ID, "thumbsheet: failed to create temp dir: "+err.Error())
				q.ErrorJob(j.ID)
				return err
			}
		}

		output := filepath.Join(outputDir, name+"_thumbsheet."+format)
		vf := fmt.Sprintf("fps=1,scale=%d:-1,tile=%dx%d", int(width), int(cols), int(rows))
		finalArgs := []string{"-i", abs, "-vf", vf, "-frames:v", "1", output}

		q.PushJobStdout(j.ID, "thumbsheet: running on "+base+" -> "+filepath.Base(output))

		cmd, err := deps.GetExec(ctx, "ffmpeg", "ffmpeg", finalArgs...)
		if err != nil {
			q.PushJobStdout(j.ID, "thumbsheet: failed to prepare: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		stderr, err := cmd.StderrPipe()
		if err != nil {
			q.PushJobStdout(j.ID, "thumbsheet: stderr pipe error: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		doneErr := make(chan struct{})
		go func() {
			s := bufio.NewScanner(stderr)
			for s.Scan() {
				_ = q.PushJobStdout(j.ID, "ffmpeg: "+s.Text())
			}
			close(doneErr)
		}()

		if err := cmd.Start(); err != nil {
			q.PushJobStdout(j.ID, "thumbsheet: failed to start: "+err.Error())
			q.ErrorJob(j.ID)
			return err
		}

		_ = cmd.Wait()
		<-doneErr

		q.PushJobStdout(j.ID, "thumbsheet: completed for "+base)
		q.RegisterOutputFile(j.ID, output, abs)
	}

	q.CompleteJob(j.ID)
	return nil
}

// timeToSeconds converts a timestamp string to a seconds string for ffmpeg expressions.
// Accepts "HH:MM:SS", "MM:SS", or raw seconds. Returns the input as-is if already numeric.
func timeToSeconds(ts string) string {
	ts = strings.TrimSpace(ts)
	parts := strings.Split(ts, ":")
	switch len(parts) {
	case 3:
		return fmt.Sprintf("(%s*3600+%s*60+%s)", parts[0], parts[1], parts[2])
	case 2:
		return fmt.Sprintf("(%s*60+%s)", parts[0], parts[1])
	default:
		return ts
	}
}
