package tasks

import (
	"bufio"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	xdraw "golang.org/x/image/draw"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/platform"
)

// --- scene-lock: finishing a Scene editor motion layer ---
//
// A motion layer is a video of one region composited back into a still
// through a feathered mask. The engine's video never matches the still
// exactly, and it is small: ~1 MP however big the region is. This pass
// finishes it against the still it was made from (the crop that was its
// first frame):
//
//   - freeze (Edges: Locked): wherever the video does not really move, every
//     frame IS the still, pixel for pixel (a motion-adaptive freeze: no
//     shimmer, no flicker), and where it moves it takes the still's
//     low-frequency tone, so it can't drift against its surroundings. With
//     the mask, the region's boundary is the still itself on every frame.
//   - detail (Detail: Match still): the video comes out at the region's own
//     size (up to sceneFinishMaxLong), Lanczos-upscaled and sharpened, with
//     the still's fine detail only where the video is near-static. Never on
//     content that moves: the still's detail does not move with it, so it
//     reads as a static ghost of the picture over the motion (measured: the
//     first version added it to moving areas at up to full strength).
//
//   - end fade: an engine clip that fades to black (or white) over its last
//     second is cut where the fade starts, sound included (see endFadeKeep).
//   - start flash: a few near-white (or black) frames the engine sometimes
//     opens with are replaced by the first real frame (see startFlashFrames).
//
// It also writes the video's motion map (<out>_motion.png: alpha = how much
// each pixel moves), which the editor shows and can grow a mask to. Runs as
// its own CPU job chained after the reshoot job, so the GPU is free the
// moment generation ends.

// sceneFinishMaxLong caps the finished video's long edge (memory: the pass
// holds a few float copies of one frame).
const sceneFinishMaxLong = 2560

// sceneMotionMapLong is the long edge of a motion map PNG.
const sceneMotionMapLong = 512

var sceneLockOptions = []TaskOption{
	{Name: "still", Label: "Still Frame", Type: "string", Default: "", Description: "The still the video was generated from (its first frame)"},
	{Name: "freeze", Label: "Lock Static Parts", Type: "bool", Default: true, Description: "Parts of the video that don't move become the still exactly; moving parts take its tone"},
	{Name: "detail", Label: "Match Still Detail", Type: "bool", Default: false, Description: "Output at the region's size with the still's fine detail added back into the motion"},
	{Name: "out", Label: "Output", Type: "string", Default: "", Description: "Output file (beside the still); empty = <still>_locked.mp4"},
	{Name: "size", Label: "Region Size", Type: "string", Default: "", Description: "WxH of the region the video fills (with detail: the output size, capped)"},
	{Name: "trimfade", Label: "Trim End Fade", Type: "bool", Default: true, Description: "Cut a fade to black or white at the end of the video, and cover a white or black flash at its start"},
}

type finishOptions struct {
	Freeze bool
	Detail bool
	// TrimFade cuts a fade to black/white off the end (endFadeKeep) and
	// covers a flash at the start (startFlashFrames).
	TrimFade bool
	// RegionW/H: the size the video is shown at (0 = the video's own).
	RegionW, RegionH int
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
	opts := ParseOptions(j, sceneLockOptions)
	still := strings.TrimSpace(optString(opts, "still"))
	if still == "" {
		return fail("needs the still frame (--still)", nil)
	}
	fo := finishOptions{Freeze: optBool(opts, "freeze"), Detail: optBool(opts, "detail"), TrimFade: optBool(opts, "trimfade")}
	if w, h, ok := parseWxH(optString(opts, "size")); ok {
		fo.RegionW, fo.RegionH = w, h
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
	if o := strings.TrimSpace(optString(opts, "out")); o != "" {
		if filepath.Dir(absPath(o)) != filepath.Dir(absPath(still)) || !mediaext.IsVideo(o) {
			return fail("--out must be a video file beside the still", nil)
		}
		out = absPath(o)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: finishing %s against %s (lock %v, detail %v)", tag, filepath.Base(video), filepath.Base(still), fo.Freeze, fo.Detail))
	_ = q.SetJobProgress(j.ID, 0, 2)
	stats, err := finishMotion(ctx, video, still, out, fo, func(pass int) { _ = q.SetJobProgress(j.ID, pass, 2) })
	if err != nil {
		os.Remove(out)
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		return fail(err.Error(), err)
	}
	if stats.Flash > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: the video opened with %d flash frame(s): replaced by its first real frame", tag, stats.Flash))
	}
	if stats.Trimmed > 0 {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: the video faded out over its last %d frames: cut where the fade starts", tag, stats.Trimmed))
	}
	if stats.Misaligned {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: the video's first frame doesn't line up with the still (match %.2f): the engine reframed it, so it was left unlocked", tag, stats.Match))
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d frames at %dx%d, %.0f%% of the area locked to the still (match %.2f) -> %s", tag, stats.Frames, stats.W, stats.H, stats.Locked*100, stats.Match, out))
	q.RegisterOutputFile(j.ID, out, video)
	_ = q.SetJobProgress(j.ID, 2, 2)
	q.CompleteJob(j.ID)
	return nil
}

func parseWxH(s string) (int, int, bool) {
	parts := strings.Split(strings.ToLower(strings.TrimSpace(s)), "x")
	if len(parts) != 2 {
		return 0, 0, false
	}
	w, err1 := strconv.Atoi(parts[0])
	h, err2 := strconv.Atoi(parts[1])
	if err1 != nil || err2 != nil || w <= 0 || h <= 0 {
		return 0, 0, false
	}
	return w, h, true
}

type lockStats struct {
	Frames int
	W, H   int
	Locked float64 // share of pixels fully frozen to the still
	// Match is the share of the still's textured tiles the video's first
	// frame lines up with (1 = the same picture); below lockMinMatch the
	// engine reframed it and nothing was frozen or detailed (Misaligned).
	Match      float64
	Misaligned bool
	// Trimmed: frames of an end fade cut off the video (endFadeKeep).
	Trimmed int
	// Flash: frames at the start replaced by the first real one (startFlashFrames).
	Flash int
}

// lockMinMatch: when fewer than this share of the still's textured tiles line
// up with the video's first frame, it is not the still (it was reframed or
// rescaled). Freezing or detailing it would blend two pictures that don't
// line up: a ghost of the still's subject over the whole region (a 5:6 crop
// the engine pillarboxed matched 9% of tiles, a well-framed clip 100%).
// Per tile, so content that legitimately differs in part of frame 0 (motion
// already under way) doesn't count as reframing.
const lockMinMatch = 0.34

// frameMatch is the share of 8×8 tiles, among those where the still has
// texture, in which the luma of the two RGB frames (w×h, float 0-255,
// lightly blurred) correlates at 0.6 or more: tone and brightness shifts
// don't lower it, a reframed or rescaled picture does. 1 when the still has
// no texture to compare.
func frameMatch(frame, still []float32, w, h int) float64 {
	la, lb := make([]float32, w*h), make([]float32, w*h)
	for i := range la {
		la[i] = 0.299*frame[i*3] + 0.587*frame[i*3+1] + 0.114*frame[i*3+2]
		lb[i] = 0.299*still[i*3] + 0.587*still[i*3+1] + 0.114*still[i*3+2]
	}
	r := max(1, min(w, h)/96)
	boxBlur(la, w, h, 1, 0, r)
	boxBlur(lb, w, h, 1, 0, r)
	const grid = 8
	textured, matched := 0, 0
	for ty := 0; ty < grid; ty++ {
		for tx := 0; tx < grid; tx++ {
			x0, x1 := tx*w/grid, (tx+1)*w/grid
			y0, y1 := ty*h/grid, (ty+1)*h/grid
			var ma, mb, n float64
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					ma += float64(la[y*w+x])
					mb += float64(lb[y*w+x])
					n++
				}
			}
			if n == 0 {
				continue
			}
			ma, mb = ma/n, mb/n
			var sab, saa, sbb float64
			for y := y0; y < y1; y++ {
				for x := x0; x < x1; x++ {
					da, db := float64(la[y*w+x])-ma, float64(lb[y*w+x])-mb
					sab += da * db
					saa += da * da
					sbb += db * db
				}
			}
			// Flat in the still (sky, a wall): nothing to line up.
			if sbb/n < 4 {
				continue
			}
			textured++
			if saa > 0 && sab/math.Sqrt(saa*sbb) >= 0.6 {
				matched++
			}
		}
	}
	if textured == 0 {
		return 1
	}
	return float64(matched) / float64(textured)
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

// errDecodeDone, returned by a decode callback, stops decoding without error.
var errDecodeDone = errors.New("decode done")

// decodeFrames streams the video's frames, scaled to w×h (Lanczos), as packed
// RGB24 to fn.
func decodeFrames(ctx context.Context, path string, w, h int, fn func(i int, rgb []byte) error) (int, error) {
	return decodeFramesVF(ctx, path, w, h, "", fn)
}

// decodeFramesVF is decodeFrames with extra filters after the scale.
func decodeFramesVF(ctx context.Context, path string, w, h int, extra string, fn func(i int, rgb []byte) error) (int, error) {
	vf := fmt.Sprintf("scale=%d:%d:flags=lanczos", w, h)
	if extra != "" {
		vf += "," + extra
	}
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", path,
		"-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
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
		if ferr == errDecodeDone {
			return n, nil
		}
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
// frozen to the still; above lockMove it is all video. The still's detail
// follows the same ramp: none on anything that moves.
const (
	lockStill = 4.0
	lockMove  = 18.0
)

// detailSharpen is the unsharp mask the moving video gets instead (detail
// from its own pixels, so it moves with them).
const detailSharpen = "unsharp=5:5:0.7:5:5:0"

func smoothstep(a, b, x float64) float64 {
	t := clampFloat((x-a)/(b-a), 0, 1)
	return t * t * (3 - 2*t)
}

func luma(p []byte, i int) float32 {
	return 0.299*float32(p[i*3]) + 0.587*float32(p[i*3+1]) + 0.114*float32(p[i*3+2])
}

// measureMotion decodes the video at w×h and returns frame 0 (RGB) and each
// pixel's motion energy: its largest luma change from frame 0, blurred so
// moving things keep a margin of motion around them. limit > 0 reads only
// that many frames; the first skip frames are passed over (frame skip is
// frame 0).
func measureMotion(ctx context.Context, video string, w, h, skip, limit int) (f0, motion []float32, frames int, err error) {
	f0 = make([]float32, w*h*3)
	motion = make([]float32, w*h)
	var l0 []float32
	frames, err = decodeFrames(ctx, video, w, h, func(n int, rgb []byte) error {
		if limit > 0 && n >= limit {
			return errDecodeDone
		}
		if n < skip {
			return nil
		}
		if n == skip {
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
		return nil, nil, 0, err
	}
	if frames == 0 {
		return nil, nil, 0, fmt.Errorf("%s has no frames", filepath.Base(video))
	}
	boxBlur(motion, w, h, 1, 0, max(2, min(w, h)/96))
	return f0, motion, frames, nil
}

// finishSize is the output size: the video's own, or with detail the region's
// (capped at sceneFinishMaxLong, never smaller than the video), even.
func finishSize(info videoInfo, o finishOptions) (int, int) {
	w, h := info.W, info.H
	if o.Detail && o.RegionW > 0 && o.RegionH > 0 {
		rw, rh := float64(o.RegionW), float64(o.RegionH)
		if long := math.Max(rw, rh); long > sceneFinishMaxLong {
			rw, rh = rw*sceneFinishMaxLong/long, rh*sceneFinishMaxLong/long
		}
		if int(rw)*int(rh) > w*h {
			w, h = int(rw), int(rh)
		}
	}
	return w &^ 1, h &^ 1
}

// motionMapPath is where a video's motion map lives.
func motionMapPath(video string) string {
	return strings.TrimSuffix(video, filepath.Ext(video)) + "_motion.png"
}

// writeMotionMap stores a motion energy field as a white PNG whose alpha is
// how much each pixel moves (0 = still, 255 = clearly moving), long edge
// sceneMotionMapLong.
func writeMotionMap(path string, motion []float32, w, h int) error {
	full := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i, m := range motion {
		a := uint8(math.Round(255 * smoothstep(lockStill, lockMove, float64(m))))
		full.Pix[i*4], full.Pix[i*4+1], full.Pix[i*4+2], full.Pix[i*4+3] = 255, 255, 255, a
	}
	var img image.Image = full
	if long := max(w, h); long > sceneMotionMapLong {
		sw, sh := max(1, w*sceneMotionMapLong/long), max(1, h*sceneMotionMapLong/long)
		small := image.NewNRGBA(image.Rect(0, 0, sw, sh))
		xdraw.ApproxBiLinear.Scale(small, small.Bounds(), full, full.Bounds(), draw.Src, nil)
		img = small
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// MotionMapDataURL returns the motion map of a video as a PNG data URL,
// computing (and caching beside the video) it when missing.
func MotionMapDataURL(ctx context.Context, video string) (string, error) {
	p := motionMapPath(video)
	if _, err := os.Stat(p); err != nil {
		info, err := probeVideoInfo(ctx, video)
		if err != nil {
			return "", err
		}
		w, h := info.W, info.H
		if long := max(w, h); long > sceneMotionMapLong {
			w, h = max(2, w*sceneMotionMapLong/long), max(2, h*sceneMotionMapLong/long)
		}
		_, motion, _, err := measureMotion(ctx, video, w, h, 0, 0)
		if err != nil {
			return "", err
		}
		if err := writeMotionMap(p, motion, w, h); err != nil {
			return "", err
		}
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return "", err
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(b), nil
}

// finishMotion writes out = the video finished against the still (see the
// file comment), plus its motion map. onPass reports the two decoding passes.
func finishMotion(ctx context.Context, video, still, out string, o finishOptions, onPass func(int)) (lockStats, error) {
	info, err := probeVideoInfo(ctx, video)
	if err != nil {
		return lockStats{}, err
	}
	w, h := finishSize(info, o)
	S, err := loadStillRGB(still, w, h)
	if err != nil {
		return lockStats{}, err
	}

	// An end fade is cut before anything is measured: fading frames change
	// everywhere, so they would read as motion over the whole picture.
	// A flash at the start is covered the same way: its frames would read
	// as motion everywhere and as a frame 0 that is not the still.
	keep, trimmed, skip := 0, 0, 0
	if o.TrimFade {
		y, err := clipLumas(ctx, video)
		if err != nil {
			return lockStats{}, err
		}
		if k := endFadeKeep(y); k < len(y) {
			keep, trimmed = k, len(y)-k
		}
		skip = startFlashFrames(y)
	}

	// Pass 1: frame 0 and the motion energy.
	f0, motion, frames, err := measureMotion(ctx, video, w, h, skip, keep)
	if err != nil {
		return lockStats{}, err
	}
	if err := writeMotionMap(motionMapPath(out), motion, w, h); err != nil {
		return lockStats{}, err
	}
	onPass(1)

	// An engine that reframed the picture can't be finished against it.
	match := frameMatch(f0, S, w, h)
	misaligned := match < lockMinMatch
	if misaligned {
		o.Freeze, o.Detail = false, false
	}

	short := min(w, h)
	// E: 1 = the still. Without freeze nothing is frozen.
	E := make([]float32, w*h)
	locked := 0
	if o.Freeze {
		for i, m := range motion {
			e := float32(1 - smoothstep(lockStill, lockMove, float64(m)))
			E[i] = e
			if e > 0.999 {
				locked++
			}
		}
	}
	// Low-frequency tone correction (freeze only): blur(still - frame 0).
	var corr []float32
	if o.Freeze {
		corr = make([]float32, w*h*3)
		for i := range corr {
			corr[i] = S[i] - f0[i]
		}
		for c := 0; c < 3; c++ {
			boxBlur(corr, w, h, 3, c, max(4, short/20))
		}
	}
	// The still's fine detail (detail only): still - blur(still), at the
	// scale the upscale lost, weighted down where motion is large.
	var hf, K []float32
	if o.Detail {
		hf = make([]float32, w*h*3)
		copy(hf, S)
		r := max(1, int(math.Round(1.5*float64(w)/float64(info.W))))
		for c := 0; c < 3; c++ {
			boxBlur(hf, w, h, 3, c, r)
		}
		for i := range hf {
			hf[i] = S[i] - hf[i]
		}
		K = make([]float32, w*h)
		for i, m := range motion {
			K[i] = float32(1 - smoothstep(lockStill, lockMove, float64(m)))
		}
	}

	// Pass 2: blend and encode.
	args := []string{"-hide_banner", "-loglevel", "error", "-y",
		"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", fmt.Sprintf("%dx%d", w, h), "-r", info.FPS, "-i", "-"}
	if info.Audio {
		args = append(args, "-i", video, "-map", "0:v", "-map", "1:a", "-c:a", "copy")
		if keep > 0 {
			if fps := parseRate(info.FPS); fps > 0 {
				args = append(args, "-t", formatFloatArg(float64(keep)/fps))
			}
		}
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
	sharpen := ""
	if o.Detail && w > info.W {
		sharpen = detailSharpen
	}
	_, derr := decodeFramesVF(ctx, video, w, h, sharpen, func(n int, rgb []byte) error {
		if keep > 0 && n >= keep {
			return errDecodeDone
		}
		// flash frames: the first real frame stands in for them (written
		// skip+1 times below), so the length and the sound stay in step
		if n < skip {
			return nil
		}
		for i := 0; i < w*h; i++ {
			e := E[i]
			for c := 0; c < 3; c++ {
				k := i*3 + c
				moving := float32(rgb[k])
				if corr != nil {
					moving += corr[k]
				}
				if hf != nil {
					moving += K[i] * hf[k]
				}
				v := e*S[k] + (1-e)*moving
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
		reps := 1
		if n == skip {
			reps += skip
		}
		for ; reps > 0; reps-- {
			if _, err := bw.Write(buf); err != nil {
				return err
			}
		}
		return nil
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
	return lockStats{Frames: frames, W: w, H: h, Locked: float64(locked) / float64(w*h), Match: match, Misaligned: misaligned, Trimmed: trimmed, Flash: skip}, nil
}

// clipLumas reads the video's per-frame mean luma (at a thumbnail size).
func clipLumas(ctx context.Context, video string) ([]float64, error) {
	const side = 64
	var lumas []float64
	_, err := decodeFrames(ctx, video, side, side, func(n int, rgb []byte) error {
		var sum float64
		for i := 0; i < side*side; i++ {
			sum += float64(luma(rgb, i))
		}
		lumas = append(lumas, sum/(side*side))
		return nil
	})
	return lumas, err
}

// startFlashFrames is how many frames at the start of a clip whose per-frame
// mean luma is y are a flash: a short burst far from the clip's usual
// brightness that it then drops straight back to (measured: a finetune's clip
// opened with two frames at luma 229, then held 122). At most half a second;
// a clip that starts dark or bright and stays so is the content.
func startFlashFrames(y []float64) int {
	n := len(y)
	if n < 24 {
		return 0
	}
	mid := append([]float64(nil), y[n/5:n*3/5]...)
	sort.Float64s(mid)
	base := mid[len(mid)/2]
	tol := math.Max(30, 0.3*base)
	k := 0
	for k < 12 && math.Abs(y[k]-base) > tol {
		k++
	}
	if k == 0 || k == 12 || math.Abs(y[k]-base) > tol/2 {
		return 0
	}
	return k
}

// endFadeKeep is how many frames to keep of a clip whose per-frame mean luma
// is y: all of them, unless the clip ends in a fade to black or white, which
// the engine adds on its own (measured: a spin faded from luma 116 to 22 over
// its last second). A fade is a run at the end that drifts steadily away
// from the clip's usual brightness and ends far from it; the cut is at the
// last frame still near it. A clip that darkens for most of its length
// (night falling) is the content, not a fade.
func endFadeKeep(y []float64) int {
	n := len(y)
	if n < 24 {
		return n
	}
	if k := fadeOutKeep(y); k < n {
		return k
	}
	inv := make([]float64, n)
	for i, v := range y {
		inv[i] = 255 - v
	}
	return fadeOutKeep(inv)
}

// fadeOutKeep is endFadeKeep for a fade to black only.
func fadeOutKeep(y []float64) int {
	n := len(y)
	head := append([]float64(nil), y[:n*3/5]...)
	sort.Float64s(head)
	base := head[len(head)/2]
	// Too dark to fade any further: nothing to tell.
	if base < 20 || y[n-1] > base*0.6 {
		return n
	}
	k := n - 1
	for k > 0 && y[k] < base*0.95 {
		k--
	}
	keep := k + 1
	if keep < n/2 {
		return n
	}
	// Steady: nearly every step of the tail goes darker (or holds).
	down := 0
	for j := keep; j < n; j++ {
		if y[j] <= y[j-1]+1 {
			down++
		}
	}
	if float64(down) < 0.85*float64(n-keep) {
		return n
	}
	return keep
}

// lockMotionEdges is finishMotion with only the freeze (the original pass).
func lockMotionEdges(ctx context.Context, video, still, out string, onPass func(int)) (lockStats, error) {
	return finishMotion(ctx, video, still, out, finishOptions{Freeze: true, TrimFade: true}, onPass)
}
