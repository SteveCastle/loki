package tasks

import (
	"bufio"
	"context"
	"fmt"
	"image"
	"image/draw"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	xdraw "golang.org/x/image/draw"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/platform"
)

// --- scene-lock: static edges for a Scene editor motion layer ---
//
// A motion layer is a video of one region composited back into a still
// through a feathered mask. The engine's video never matches the still
// exactly: tone drifts, detail is resampled, static parts shimmer. In the
// mask's soft edge that shows as a halo or a crawling seam, so the edge would
// not be static. This pass fixes the video against the still it was made
// from (the crop that was its first frame):
//
//   - wherever the video does not really move, every frame IS the still,
//     pixel for pixel (a motion-adaptive freeze: no shimmer, no flicker);
//   - where it moves, it keeps the motion but takes the still's
//     low-frequency tone (frame 0 vs the still), so it can't drift in
//     colour or exposure against its surroundings;
//   - a smooth ramp between the two (motion energy, blurred).
//
// Together with the mask, the region's boundary is the still itself on every
// frame, whatever the engine did. Runs as its own CPU job chained after the
// reshoot job, so the GPU is free the moment generation ends.

var sceneLockOptions = []TaskOption{
	{Name: "still", Label: "Still Frame", Type: "string", Default: "", Description: "The still the video was generated from (its first frame); static parts of the video are locked to it"},
}

func sceneLockTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	const tag = "scene-lock"
	ctx := j.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}
	still := strings.TrimSpace(optString(ParseOptions(j, sceneLockOptions), "still"))
	if still == "" {
		return fail("needs the still frame (--still)", nil)
	}
	video := ""
	for _, line := range strings.Split(j.Input, "\n") {
		if l := strings.TrimSpace(line); l != "" && mediaext.IsVideo(l) {
			video = l
		}
	}
	if video == "" {
		return fail("no video in the input", nil)
	}
	out := strings.TrimSuffix(still, filepath.Ext(still)) + "_locked.mp4"
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: locking the static parts of %s to %s", tag, filepath.Base(video), filepath.Base(still)))
	_ = q.SetJobProgress(j.ID, 0, 2)
	stats, err := lockMotionEdges(ctx, video, still, out, func(pass int) { _ = q.SetJobProgress(j.ID, pass, 2) })
	if err != nil {
		os.Remove(out)
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		return fail(err.Error(), err)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d frames, %.0f%% of the area locked to the still -> %s", tag, stats.Frames, stats.Locked*100, out))
	q.RegisterOutputFile(j.ID, out, video)
	_ = q.SetJobProgress(j.ID, 2, 2)
	q.CompleteJob(j.ID)
	return nil
}

type lockStats struct {
	Frames int
	Locked float64 // share of pixels fully frozen to the still
}

type videoInfo struct {
	W, H  int
	FPS   string
	Audio bool
}

func probeVideoInfo(ctx context.Context, path string) (videoInfo, error) {
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffprobe"), "-v", "error", "-select_streams", "v:0",
		"-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path)
	platform.HideSubprocessWindow(cmd)
	b, err := cmd.Output()
	if err != nil {
		return videoInfo{}, fmt.Errorf("ffprobe: %w", err)
	}
	parts := strings.Split(strings.TrimSpace(strings.SplitN(string(b), "\n", 2)[0]), ",")
	if len(parts) < 3 {
		return videoInfo{}, fmt.Errorf("ffprobe: unexpected %q", string(b))
	}
	w, _ := strconv.Atoi(parts[0])
	h, _ := strconv.Atoi(parts[1])
	if w <= 0 || h <= 0 {
		return videoInfo{}, fmt.Errorf("ffprobe: no video size")
	}
	return videoInfo{W: w, H: h, FPS: parts[2], Audio: probeHasAudio(path)}, nil
}

// decodeFrames streams the video's frames as packed RGB24 to fn.
func decodeFrames(ctx context.Context, path string, w, h int, fn func(i int, rgb []byte) error) (int, error) {
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", path,
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-")
	platform.HideSubprocessWindow(cmd)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, err
	}
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	r := bufio.NewReaderSize(stdout, 1<<20)
	frame := make([]byte, w*h*3)
	n := 0
	var ferr error
	for {
		if _, err := io.ReadFull(r, frame); err != nil {
			break
		}
		if ferr = fn(n, frame); ferr != nil {
			break
		}
		n++
	}
	if ferr != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return n, ferr
	}
	if err := cmd.Wait(); err != nil {
		return n, fmt.Errorf("decoding %s: %w", filepath.Base(path), err)
	}
	return n, nil
}

// loadStillRGB reads a still and resamples it to w×h, as float32 RGB.
func loadStillRGB(path string, w, h int) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	src, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("still %s: %w", filepath.Base(path), err)
	}
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	if src.Bounds().Dx() == w && src.Bounds().Dy() == h {
		draw.Draw(dst, dst.Bounds(), src, src.Bounds().Min, draw.Src)
	} else {
		xdraw.CatmullRom.Scale(dst, dst.Bounds(), src, src.Bounds(), draw.Src, nil)
	}
	out := make([]float32, w*h*3)
	for i := 0; i < w*h; i++ {
		out[i*3] = float32(dst.Pix[i*4])
		out[i*3+1] = float32(dst.Pix[i*4+1])
		out[i*3+2] = float32(dst.Pix[i*4+2])
	}
	return out, nil
}

// boxBlur blurs one channel (stride channels, offset ch) in place: three
// separable box passes of radius r approximate a gaussian.
func boxBlur(buf []float32, w, h, stride, ch, r int) {
	if r < 1 {
		return
	}
	tmp := make([]float32, max(w, h))
	line := make([]float32, max(w, h))
	pass := func(n int, get func(i int) float32, set func(i int, v float32)) {
		for i := 0; i < n; i++ {
			line[i] = get(i)
		}
		var sum float32
		for k := -r; k <= r; k++ {
			sum += line[clampInt(k, 0, n-1)]
		}
		inv := 1 / float32(2*r+1)
		for i := 0; i < n; i++ {
			tmp[i] = sum * inv
			sum += line[clampInt(i+r+1, 0, n-1)] - line[clampInt(i-r, 0, n-1)]
		}
		for i := 0; i < n; i++ {
			set(i, tmp[i])
		}
	}
	for it := 0; it < 3; it++ {
		for y := 0; y < h; y++ {
			row := y * w
			pass(w, func(i int) float32 { return buf[(row+i)*stride+ch] }, func(i int, v float32) { buf[(row+i)*stride+ch] = v })
		}
		for x := 0; x < w; x++ {
			pass(h, func(i int) float32 { return buf[(i*w+x)*stride+ch] }, func(i int, v float32) { buf[(i*w+x)*stride+ch] = v })
		}
	}
}

// Motion energy (max luma change from frame 0, blurred) below lockStill is
// frozen to the still; above lockMove it is all video.
const (
	lockStill = 4.0
	lockMove  = 18.0
)

func smoothstep(a, b, x float64) float64 {
	t := clampFloat((x-a)/(b-a), 0, 1)
	return t * t * (3 - 2*t)
}

// lockMotionEdges writes out = the video with its static parts locked to the
// still (see the file comment). onPass reports the two decoding passes.
func lockMotionEdges(ctx context.Context, video, still, out string, onPass func(int)) (lockStats, error) {
	info, err := probeVideoInfo(ctx, video)
	if err != nil {
		return lockStats{}, err
	}
	w, h := info.W, info.H
	S, err := loadStillRGB(still, w, h)
	if err != nil {
		return lockStats{}, err
	}

	// Pass 1: frame 0 and the motion energy.
	f0 := make([]float32, w*h*3)
	motion := make([]float32, w*h)
	luma := func(p []byte, i int) float32 {
		return 0.299*float32(p[i*3]) + 0.587*float32(p[i*3+1]) + 0.114*float32(p[i*3+2])
	}
	var l0 []float32
	frames, err := decodeFrames(ctx, video, w, h, func(n int, rgb []byte) error {
		if n == 0 {
			l0 = make([]float32, w*h)
			for i := range f0 {
				f0[i] = float32(rgb[i])
			}
			for i := 0; i < w*h; i++ {
				l0[i] = luma(rgb, i)
			}
			return nil
		}
		for i := 0; i < w*h; i++ {
			if d := float32(math.Abs(float64(luma(rgb, i) - l0[i]))); d > motion[i] {
				motion[i] = d
			}
		}
		return nil
	})
	if err != nil {
		return lockStats{}, err
	}
	if frames == 0 {
		return lockStats{}, fmt.Errorf("%s has no frames", filepath.Base(video))
	}
	onPass(1)

	short := min(w, h)
	// Spread the motion a little so moving things keep a margin of motion
	// around them, then map it to the freeze weight E (1 = the still).
	boxBlur(motion, w, h, 1, 0, max(2, short/96))
	E := make([]float32, w*h)
	locked := 0
	for i, m := range motion {
		e := float32(1 - smoothstep(lockStill, lockMove, float64(m)))
		E[i] = e
		if e > 0.999 {
			locked++
		}
	}
	// Low-frequency tone correction: blur(still - frame 0).
	corr := make([]float32, w*h*3)
	for i := range corr {
		corr[i] = S[i] - f0[i]
	}
	for c := 0; c < 3; c++ {
		boxBlur(corr, w, h, 3, c, max(4, short/20))
	}

	// Pass 2: blend and encode.
	args := []string{"-hide_banner", "-loglevel", "error", "-y",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", w, h), "-r", info.FPS, "-i", "-"}
	if info.Audio {
		args = append(args, "-i", video, "-map", "0:v", "-map", "1:a", "-c:a", "copy")
	}
	args = append(args, "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out)
	enc := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), args...)
	platform.HideSubprocessWindow(enc)
	var encErr strings.Builder
	enc.Stderr = &encErr
	stdin, err := enc.StdinPipe()
	if err != nil {
		return lockStats{}, err
	}
	if err := enc.Start(); err != nil {
		return lockStats{}, err
	}
	bw := bufio.NewWriterSize(stdin, 1<<20)
	buf := make([]byte, w*h*3)
	_, derr := decodeFrames(ctx, video, w, h, func(n int, rgb []byte) error {
		for i := 0; i < w*h; i++ {
			e := E[i]
			for c := 0; c < 3; c++ {
				k := i*3 + c
				v := e*S[k] + (1-e)*(float32(rgb[k])+corr[k])
				switch {
				case v <= 0:
					buf[k] = 0
				case v >= 255:
					buf[k] = 255
				default:
					buf[k] = byte(v + 0.5)
				}
			}
		}
		_, err := bw.Write(buf)
		return err
	})
	ferr := bw.Flush()
	stdin.Close()
	werr := enc.Wait()
	if derr != nil {
		return lockStats{}, derr
	}
	if ferr != nil || werr != nil {
		return lockStats{}, fmt.Errorf("encoding: %v %v: %s", ferr, werr, strings.TrimSpace(encErr.String()))
	}
	onPass(2)
	return lockStats{Frames: frames, Locked: float64(locked) / float64(w*h)}, nil
}
