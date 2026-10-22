package tasks

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/platform"
)

// --- scene loops ---
//
// A motion layer plays on a loop, and the engine's clip ends where it ends:
// looping it raw jumps. Each tool here makes a new version of a motion layer
// that loops without a seam:
//
//   - crossfade: the clip's last moments dissolve into its first, so the
//     end flows into the start (right for anything moving one way: water,
//     smoke, fire, clouds);
//   - natural: finds the two frames that look most alike and loops between
//     them, with a short crossfade over any small difference left;
//   - boomerang: forward then backward (only reads right for back-and-forth
//     motion: swaying, breathing, flickering).
//
// loopSpeed is the other half: synced scenes (the default) play every layer
// at the speed that fits a whole number of its loops into the scene length,
// so the exported video itself loops as one.

const (
	LoopCrossfade = "crossfade"
	LoopNatural   = "natural"
	LoopBoomerang = "boomerang"
)

// layerTiming is a motion layer's own timing: Speed (playback rate, 1 = as
// generated), Offset (seconds of scene time its loop is shifted by, so
// layers don't pulse together) and Volume (its soundtrack's gain).
type layerTiming struct {
	Speed, Offset, Volume float64
}

// sceneLayerTiming reads a layer's timing options (speed 0.25-4, offset
// >= 0, volume 0-2; missing = 1, 0, 1).
func sceneLayerTiming(opts map[string]any) layerTiming {
	t := layerTiming{Speed: 1, Volume: 1}
	if _, ok := opts["speed"]; ok {
		if s := optFloat(opts, "speed"); s > 0 {
			t.Speed = clampFloat(s, 0.25, 4)
		}
	}
	t.Offset = math.Max(0, optFloat(opts, "offset"))
	if _, ok := opts["volume"]; ok {
		t.Volume = clampFloat(optFloat(opts, "volume"), 0, 2)
	}
	return t
}

// clipPhase is where (clip seconds) a looped motion clip starts so that it
// is Offset scene-seconds into its loop at scene time 0: the clip plays at
// PTS multiplier Speed, so scene time t shows clip time t/Speed.
func clipPhase(l sceneGraphLayer) float64 {
	if l.Offset <= 0 || l.Duration <= 0 {
		return 0
	}
	m := l.Speed
	if m <= 0 {
		m = 1
	}
	return math.Mod(l.Offset/m, l.Duration)
}

// atempoChain changes audio speed by factor t (> 0); atempo takes 0.5-2 per
// instance, so larger changes chain several.
func atempoChain(t float64) []string {
	var out []string
	for t > 2 {
		out = append(out, "atempo=2")
		t /= 2
	}
	for t < 0.5 {
		out = append(out, "atempo=0.5")
		t /= 0.5
	}
	return append(out, fmt.Sprintf("atempo=%.6f", t))
}

// loopSpeed is the PTS multiplier that fits a d-second loop a whole number
// of times into a scene of length L (1 = as generated; >1 slower).
func loopSpeed(L, d float64) float64 {
	if L <= 0 || d <= 0 {
		return 1
	}
	n := math.Max(1, math.Round(L/d))
	return L / (n * d)
}

func parseRate(r string) float64 {
	if a, b, ok := strings.Cut(r, "/"); ok {
		x, _ := strconv.ParseFloat(a, 64)
		y, _ := strconv.ParseFloat(b, 64)
		if y > 0 {
			return x / y
		}
		return 0
	}
	v, _ := strconv.ParseFloat(r, 64)
	return v
}

// loopEncodeArgs encodes a loop at a constant fps. Its filter graph ends with
// setpts=N/(fps*TB): trimmed and concatenated segments carry timestamps the
// output's frame-rate sync otherwise "corrects" by dropping or duplicating a
// frame, typically the last one, right at the loop seam.
func loopEncodeArgs(fps float64) []string {
	return []string{"-fps_mode", "cfr", "-r", strconv.FormatFloat(fps, 'f', 6, 64),
		"-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"}
}

// countFrames returns how many video frames a clip has (decodes it: clips
// here are seconds long).
func countFrames(ctx context.Context, video string) (int, error) {
	cmd := exec.CommandContext(ctx, deps.MustBundled("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
		"-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", video)
	platform.HideSubprocessWindow(cmd)
	b, err := cmd.Output()
	if err != nil {
		return 0, fmt.Errorf("ffprobe: %w", err)
	}
	n, err := strconv.Atoi(strings.TrimSpace(strings.SplitN(string(b), "\n", 2)[0]))
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("ffprobe: no frame count for %s", filepath.Base(video))
	}
	return n, nil
}

// crossfadeLoop writes frames [start, end) of src as a loop: its last cf
// frames dissolve into its first cf, so out frame k < cf mixes
// src(end-cf+k) into src(start+k), and the result is end-start-cf frames
// whose last frame (src end-cf-1) runs straight into its first (src end-cf).
// Everything is cut by frame number: cutting by timestamps drops or repeats
// a frame at the seam, which is exactly the jump this is meant to remove.
// The dissolve is weighted by T*fps (the frame index from 0), not blend's N:
// ffmpeg 7.1's blend counts N from 1, which starts the dissolve a frame in.
func crossfadeLoop(ctx context.Context, src, out string, start, end, cf int, fps float64, audio bool) error {
	if cf < 1 || end-start <= 2*cf {
		return fmt.Errorf("the clip is too short to crossfade (%d frames)", end-start)
	}
	filter := fmt.Sprintf("[0:v]split=3[a][b][m];"+
		"[a]trim=start_frame=%[1]d:end_frame=%[2]d,setpts=PTS-STARTPTS[tail];"+
		"[b]trim=start_frame=%[3]d:end_frame=%[4]d,setpts=PTS-STARTPTS[head];"+
		"[m]trim=start_frame=%[4]d:end_frame=%[1]d,setpts=PTS-STARTPTS[mid];"+
		"[tail][head]blend=all_expr=A*(1-T*%[6]s/%[5]d)+B*(T*%[6]s/%[5]d)[x];"+
		"[x][mid]concat=n=2:v=1:a=0,setpts=N/(%[6]s*TB)[v]", end-cf, end, start, start+cf, cf, strconv.FormatFloat(fps, 'f', 6, 64))
	maps := []string{"-map", "[v]"}
	if audio {
		t := func(frame int) string { return strconv.FormatFloat(float64(frame)/fps, 'f', 6, 64) }
		c := strconv.FormatFloat(float64(cf)/fps, 'f', 6, 64)
		filter += fmt.Sprintf(";[0:a]asplit=3[aa][ab][am];"+
			"[aa]atrim=start=%[1]s:end=%[2]s,asetpts=PTS-STARTPTS,afade=t=out:d=%[5]s[at];"+
			"[ab]atrim=start=%[3]s:end=%[4]s,asetpts=PTS-STARTPTS,afade=t=in:d=%[5]s[ah];"+
			"[am]atrim=start=%[4]s:end=%[1]s,asetpts=PTS-STARTPTS[amid];"+
			"[at][ah]amix=inputs=2:normalize=0[ax];[ax][amid]concat=n=2:v=0:a=1[a]",
			t(end-cf), t(end), t(start), t(start+cf), c)
		maps = append(maps, "-map", "[a]", "-c:a", "aac", "-b:a", "192k")
	}
	args := []string{"-i", src, "-filter_complex", filter}
	args = append(args, maps...)
	args = append(args, loopEncodeArgs(fps)...)
	return runSceneFFmpeg(ctx, append(args, out))
}

// naturalLoop is a loop found in a clip: play frames [Start, End) and wrap.
type naturalLoop struct {
	Start, End int     // frames
	FPS        float64 // of the clip
	Cost       float64 // seam difference relative to a normal frame step (1 ≈ invisible)
}

// findNaturalLoop compares small grayscale frames of the clip and returns
// the longest-reasonable loop whose wrap (frame End-1 -> Start) looks most
// like an ordinary step from one frame to the next. Loops are at least
// minFrac of the clip and 1.5 s long.
func findNaturalLoop(ctx context.Context, video string, minFrac float64) (naturalLoop, error) {
	info, err := probeVideoInfo(ctx, video)
	if err != nil {
		return naturalLoop{}, err
	}
	fps := parseRate(info.FPS)
	if fps <= 0 {
		fps = 24
	}
	w := 96
	h := max(2, int(math.Round(float64(w)*float64(info.H)/float64(info.W))))
	var frames [][]float32
	if _, err := decodeFrames(ctx, video, w, h, func(_ int, rgb []byte) error {
		g := make([]float32, w*h)
		for i := range g {
			g[i] = luma(rgb, i)
		}
		frames = append(frames, g)
		return nil
	}); err != nil {
		return naturalLoop{}, err
	}
	n := len(frames)
	minLen := max(int(math.Ceil(1.5*fps)), int(math.Ceil(minFrac*float64(n))))
	if n < minLen+1 {
		return naturalLoop{}, fmt.Errorf("the clip is too short to find a loop in (%d frames)", n)
	}
	sad := func(a, b []float32) float64 {
		s := 0.0
		for i := range a {
			s += math.Abs(float64(a[i] - b[i]))
		}
		return s / float64(len(a))
	}
	// A normal step between frames, for scale.
	steps := make([]float64, 0, n-1)
	for i := 1; i < n; i++ {
		steps = append(steps, sad(frames[i-1], frames[i]))
	}
	sort.Float64s(steps)
	base := math.Max(steps[len(steps)/2], 0.5)

	best := naturalLoop{FPS: fps, Cost: math.Inf(1)}
	// Wrapping from frame j-1 back to i must look like the step j-1 -> j:
	// f_i ≈ f_j, and the motion must carry on the same way, so the frames
	// either side of the seam match too (same place, same direction). Every
	// candidate has both neighbours, so none escapes the direction check.
	for i := 1; i+minLen < n-1; i++ {
		for j := i + minLen; j < n-1; j++ {
			c := (sad(frames[j], frames[i]) + sad(frames[j-1], frames[i-1]) + sad(frames[j+1], frames[i+1])) / 3
			c /= base
			// Prefer longer loops among near-equal seams.
			c -= 0.15 * float64(j-i) / float64(n)
			if c < best.Cost {
				best.Start, best.End, best.Cost = i, j, c
			}
		}
	}
	return best, nil
}

// LoopVersion adds a seamless-loop version of motion layer lid's version n.
func LoopVersion(ctx context.Context, id, lid string, n int, kind string) (*Scene, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return nil, err
	}
	var src SceneVersion
	found := false
	for _, l := range s.Layers {
		if l.ID == lid && l.Kind == SceneKindMotion && n >= 0 && n < len(l.Versions) {
			src, found = l.Versions[n], true
		}
	}
	if !found {
		return nil, ErrLayerNotFound
	}
	if src.Duration <= 0 {
		src.Duration = probeVideoDuration(ctx, src.Path)
	}
	stem := strings.TrimSuffix(src.Path, filepath.Ext(src.Path))
	pick := func(suffix string) string {
		out := stem + suffix + ".mp4"
		if _, err := os.Stat(out); err == nil {
			out = resolveConflict(out)
		}
		return out
	}

	var out, note string
	switch kind {
	case LoopBoomerang:
		info, err := probeVideoInfo(ctx, src.Path)
		if err != nil {
			return nil, err
		}
		fps := parseRate(info.FPS)
		out, note = pick("_loop"), fmt.Sprintf("boomerang of v%d", n+1)
		filter := fmt.Sprintf("[0:v]split[f][r0];[r0]reverse[r];[f][r]concat=n=2:v=1:a=0,setpts=N/(%s*TB)[v]", strconv.FormatFloat(fps, 'f', 6, 64))
		maps := []string{"-map", "[v]"}
		if src.Audio {
			filter += ";[0:a]asplit[af][ar0];[ar0]areverse[ar];[af][ar]concat=n=2:v=0:a=1[a]"
			maps = append(maps, "-map", "[a]", "-c:a", "aac", "-b:a", "192k")
		}
		args := append([]string{"-i", src.Path, "-filter_complex", filter}, maps...)
		args = append(args, loopEncodeArgs(fps)...)
		if err := runSceneFFmpeg(ctx, append(args, out)); err != nil {
			return nil, err
		}
	case LoopCrossfade:
		info, err := probeVideoInfo(ctx, src.Path)
		if err != nil {
			return nil, err
		}
		fps := parseRate(info.FPS)
		frames, err := countFrames(ctx, src.Path)
		if err != nil {
			return nil, err
		}
		// Up to a second of dissolve, never more than a quarter of the clip.
		cf := max(1, min(int(math.Round(fps)), frames/4))
		out, note = pick("_xfade"), fmt.Sprintf("crossfade loop of v%d", n+1)
		if err := crossfadeLoop(ctx, src.Path, out, 0, frames, cf, fps, src.Audio); err != nil {
			return nil, err
		}
	case LoopNatural:
		nl, err := findNaturalLoop(ctx, src.Path, 0.4)
		if err != nil {
			return nil, err
		}
		out = pick("_natural")
		note = fmt.Sprintf("natural loop %.1f–%.1fs of v%d", float64(nl.Start)/nl.FPS, float64(nl.End)/nl.FPS, n+1)
		// The found loop is frames [Start, End); a short dissolve (up to
		// 0.3 s) hides what difference its seam has left.
		cf := max(1, min(int(math.Round(0.3*nl.FPS)), (nl.End-nl.Start)/6))
		if err := crossfadeLoop(ctx, src.Path, out, nl.Start, nl.End, cf, nl.FPS, src.Audio); err != nil {
			return nil, err
		}
	default:
		return nil, fmt.Errorf("unknown loop kind %q (crossfade, natural or boomerang)", kind)
	}

	sceneMu.Lock()
	defer sceneMu.Unlock()
	s, err = loadSceneLocked(id)
	if err != nil {
		os.Remove(out)
		return nil, err
	}
	for _, l := range s.Layers {
		if l.ID == lid {
			v := src
			v.Path, v.JobID, v.Created, v.Note = out, "", time.Now().UTC(), note
			v.Duration = probeVideoDuration(ctx, out)
			l.Versions = append(l.Versions, v)
			l.Active = len(l.Versions) - 1
		}
	}
	if err := saveSceneLocked(s); err != nil {
		return nil, err
	}
	return s, nil
}

// BoomerangVersion is LoopVersion(..., LoopBoomerang).
func BoomerangVersion(ctx context.Context, id, lid string, n int) (*Scene, error) {
	return LoopVersion(ctx, id, lid, n, LoopBoomerang)
}
