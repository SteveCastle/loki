package tasks

import (
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	xdraw "golang.org/x/image/draw"

	"github.com/stevecastle/shrike/deps"
)

func TestLoopSpeed(t *testing.T) {
	cases := []struct{ L, d, want float64 }{
		{10, 5, 1},          // two loops exactly
		{10, 4.8, 10 / 9.6}, // two loops, slowed a little
		{5.2, 5.2, 1},
		{12, 5, 1.2},      // round(2.4)=2 loops of 6 s
		{6, 15, 6.0 / 15}, // a clip longer than the scene fits once (sped up)
	}
	for _, c := range cases {
		if got := loopSpeed(c.L, c.d); math.Abs(got-c.want) > 1e-9 {
			t.Errorf("loopSpeed(%v, %v) = %v, want %v", c.L, c.d, got, c.want)
		}
	}
	if loopSpeed(0, 5) != 1 || loopSpeed(5, 0) != 1 {
		t.Error("degenerate inputs play as generated")
	}
}

func TestSceneGraphRetimesSyncedLoops(t *testing.T) {
	layers := []sceneGraphLayer{{Media: "m.mp4", Motion: true, Audio: true, Speed: 1.25, Region: SceneRect{W: 64, H: 64}, Mask: "k.png"}}
	g := strings.Join(sceneGraphArgs("b.png", 100, 100, layers, true, 5, SceneRect{}), " ")
	for _, want := range []string{"[1:v]setpts=PTS*1.250000,scale=64:64", "[1:a]atempo=0.800000[at0]", "[at0]anull[aout]"} {
		if !strings.Contains(g, want) {
			t.Errorf("graph lacks %q:\n%s", want, g)
		}
	}
	// Stills (flatten) never retime.
	if g := strings.Join(sceneGraphArgs("b.png", 100, 100, layers, false, 0, SceneRect{}), " "); strings.Contains(g, "setpts") {
		t.Errorf("a single frame must not retime:\n%s", g)
	}
}

func TestKeepOnlyVersion(t *testing.T) {
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-00000000d001"
	dir, _ := sceneDir(id)
	_ = os.MkdirAll(dir, 0755)
	var vs []SceneVersion
	for i := 0; i < 3; i++ {
		p := filepath.Join(dir, fmt.Sprintf("v%d.mp4", i))
		_ = os.WriteFile(p, []byte("x"), 0644)
		_ = os.WriteFile(motionMapPath(p), []byte("x"), 0644)
		vs = append(vs, SceneVersion{Path: p})
	}
	writeTestScene(t, &Scene{ID: id, Width: 10, Height: 10, Layers: []*SceneLayer{{ID: "m", Kind: SceneKindMotion, Versions: vs, Active: 2}}})
	s, err := KeepOnlyVersion(id, "m", 1)
	if err != nil {
		t.Fatal(err)
	}
	l := s.Layers[0]
	if len(l.Versions) != 1 || l.Active != 0 || l.Versions[0].Path != vs[1].Path {
		t.Fatalf("kept %+v active %d", l.Versions, l.Active)
	}
	for i, v := range vs {
		_, err := os.Stat(v.Path)
		_, merr := os.Stat(motionMapPath(v.Path))
		if (i == 1) != (err == nil) || (i == 1) != (merr == nil) {
			t.Errorf("version %d file kept=%v map kept=%v", i, err == nil, merr == nil)
		}
	}
}

func ffmpegTest(t *testing.T) {
	t.Helper()
	if deps.BundledOrEmpty("ffmpeg") == "" {
		t.Skip("bundled ffmpeg not available")
	}
}

func meanAbs(a, b image.Image, r image.Rectangle) float64 {
	sum, n := 0.0, 0
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			r1, g1, b1, _ := a.At(x, y).RGBA()
			r2, g2, b2, _ := b.At(x, y).RGBA()
			sum += math.Abs(float64(r1>>8)-float64(r2>>8)) + math.Abs(float64(g1>>8)-float64(g2>>8)) + math.Abs(float64(b1>>8)-float64(b2>>8))
			n += 3
		}
	}
	return sum / float64(n)
}

// framesOf decodes every frame of a clip.
func framesOf(t *testing.T, video string) []image.Image {
	t.Helper()
	dir := t.TempDir()
	if err := runSceneFFmpeg(context.Background(), []string{"-i", video, filepath.Join(dir, "f_%04d.png")}); err != nil {
		t.Fatal(err)
	}
	names, _ := filepath.Glob(filepath.Join(dir, "f_*.png"))
	out := make([]image.Image, 0, len(names))
	for _, n := range names {
		out = append(out, readPNG(t, n))
	}
	return out
}

func grab(t *testing.T, video string, at float64) image.Image {
	t.Helper()
	out := fmt.Sprintf("%s.%.3f.png", video, at)
	if err := runSceneFFmpeg(context.Background(), []string{"-ss", formatFloatArg(at), "-i", video, "-frames:v", "1", out}); err != nil {
		t.Fatal(err)
	}
	return readPNG(t, out)
}

// Detail match: the engine's small video comes out at the region's size,
// carrying the still's fine detail (a fine pattern the small video cannot
// hold), and the motion map shows where it moves.
func TestSceneFinishDetailMatch(t *testing.T) {
	ffmpegTest(t)
	dir := t.TempDir()
	still := filepath.Join(dir, "crop.png")
	video := filepath.Join(dir, "crop_reshoot.mp4")
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i",
		"nullsrc=s=384x256,geq=lum=128+60*sin(X*1.4)*sin(Y*1.4):cb=128:cr=128", "-frames:v", "1", still}); err != nil {
		t.Fatal(err)
	}
	// The engine's version: a quarter-size copy with a square moving in the top-left.
	if err := runSceneFFmpeg(context.Background(), []string{"-loop", "1", "-i", still, "-f", "lavfi", "-i", "color=white:s=12x12",
		"-filter_complex", "[0:v]scale=96:64[s];[s][1:v]overlay=x=4+t*20:y=6:shortest=1,format=yuv420p",
		"-t", "1", "-r", "24", "-c:v", "libx264", "-crf", "12", video}); err != nil {
		t.Fatal(err)
	}
	plain := filepath.Join(dir, "plain.mp4")
	sharp := filepath.Join(dir, "sharp.mp4")
	if _, err := finishMotion(context.Background(), video, still, plain, finishOptions{RegionW: 384, RegionH: 256}, func(int) {}); err != nil {
		t.Fatal(err)
	}
	st, err := finishMotion(context.Background(), video, still, sharp, finishOptions{Detail: true, RegionW: 384, RegionH: 256}, func(int) {})
	if err != nil {
		t.Fatal(err)
	}
	if st.W != 384 || st.H != 256 {
		t.Fatalf("detail output %dx%d, want the region's 384x256", st.W, st.H)
	}
	s := readPNG(t, still)
	pf, sf := grab(t, plain, 0.5), grab(t, sharp, 0.5)
	if pf.Bounds().Dx() != 96 {
		t.Fatalf("without detail the video keeps its own size, got %v", pf.Bounds())
	}
	area := image.Rect(200, 140, 380, 250) // static
	up := image.NewRGBA(s.Bounds())
	xdraw.CatmullRom.Scale(up, up.Bounds(), pf, pf.Bounds(), draw.Src, nil)
	plainErr, sharpErr := meanAbs(up, s, area), meanAbs(sf, s, area)
	if sharpErr > plainErr*0.5 {
		t.Errorf("detail match should recover the still's fine detail: error %.1f (upscaled engine video %.1f)", sharpErr, plainErr)
	}
	m := readPNG(t, motionMapPath(sharp))
	b := m.Bounds()
	alpha := func(fx, fy float64) uint32 {
		_, _, _, a := m.At(b.Min.X+int(fx*float64(b.Dx())), b.Min.Y+int(fy*float64(b.Dy()))).RGBA()
		return a >> 8
	}
	if alpha(0.2, 0.18) < 128 || alpha(0.85, 0.8) > 10 {
		t.Errorf("motion map: moving %d, static %d", alpha(0.2, 0.18), alpha(0.85, 0.8))
	}
	// Computed on demand for any version too (cached beside it).
	url, err := MotionMapDataURL(context.Background(), video)
	if err != nil || !strings.HasPrefix(url, "data:image/png;base64,") {
		t.Fatalf("on-demand motion map: %v", err)
	}
}

// A clip whose motion repeats every 2 s: the natural loop is a whole period,
// and every loop version runs from its last frame into its first by exactly
// one ordinary frame step.
func TestSceneLoops(t *testing.T) {
	ffmpegTest(t)
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-00000000d002"
	dir, _ := sceneDir(id)
	_ = os.MkdirAll(dir, 0755)
	clip := filepath.Join(dir, "motion-m-1_locked.mp4")
	// A sine pattern drifting one way with a 2 s period: every frame steps by
	// the same amount, so a dropped or repeated frame at a seam is obvious.
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "nullsrc=s=128x96:r=24,geq=lum=128+90*sin(X/6-PI*T):cb=128:cr=128",
		"-t", "5", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", clip}); err != nil {
		t.Fatal(err)
	}
	nl, err := findNaturalLoop(context.Background(), clip, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	period := float64(nl.End-nl.Start) / nl.FPS
	if math.Abs(period-2) > 0.1 && math.Abs(period-4) > 0.1 {
		t.Errorf("natural loop %d..%d = %.2fs, want a whole 2 s period (cost %.2f)", nl.Start, nl.End, period, nl.Cost)
	}
	writeTestScene(t, &Scene{ID: id, Width: 128, Height: 96, Layers: []*SceneLayer{{ID: "m", Kind: SceneKindMotion,
		Versions: []SceneVersion{{Path: clip, Duration: 5}}, Active: 0}}})

	for _, kind := range []string{LoopCrossfade, LoopNatural, LoopBoomerang} {
		s, err := LoopVersion(context.Background(), id, "m", 0, kind)
		if err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		v := s.Layers[0].Versions[s.Layers[0].Active]
		if v.Duration <= 0 || v.Note == "" {
			t.Errorf("%s version: %+v", kind, v)
		}
		// The wrap (last frame -> first) must be one ordinary frame step: no
		// dropped or repeated frame, no jump.
		frames := framesOf(t, v.Path)
		k := len(frames)
		wrap := meanAbs(frames[k-1], frames[0], frames[0].Bounds())
		step := 0.0
		for i := k - 6; i < k-1; i++ {
			step += meanAbs(frames[i], frames[i+1], frames[0].Bounds()) / 5
		}
		if kind != LoopBoomerang && (wrap > 1.3*step || wrap < 0.7*step) {
			t.Errorf("%s: the loop jumps: last->first differs %.2f, the step before it %.2f", kind, wrap, step)
		}
		if kind == LoopBoomerang && wrap > 0.3*step {
			t.Errorf("boomerang ends where it starts: last->first differs %.2f", wrap)
		}
	}
	if _, err := LoopVersion(context.Background(), id, "m", 0, "spin"); err == nil {
		t.Error("unknown loop kinds are refused")
	}
}

func TestSceneLayerTiming(t *testing.T) {
	if got := sceneLayerTiming(nil); got != (layerTiming{Speed: 1, Volume: 1}) {
		t.Fatalf("defaults %+v", got)
	}
	got := sceneLayerTiming(map[string]any{"speed": 9.0, "offset": -2.0, "volume": 3.0})
	if got != (layerTiming{Speed: 4, Offset: 0, Volume: 2}) {
		t.Fatalf("clamped %+v", got)
	}
	if got := sceneLayerTiming(map[string]any{"volume": 0.0}); got.Volume != 0 {
		t.Fatalf("volume 0 is a mute, not the default: %+v", got)
	}
	// A 4 s clip at PTS multiplier 2 (half speed) offset by 3 scene-seconds
	// starts 1.5 s into the clip; offsets wrap at the clip's length.
	if p := clipPhase(sceneGraphLayer{Duration: 4, Speed: 2, Offset: 3}); math.Abs(p-1.5) > 1e-9 {
		t.Fatalf("phase %v", p)
	}
	if p := clipPhase(sceneGraphLayer{Duration: 4, Speed: 1, Offset: 9}); math.Abs(p-1) > 1e-9 {
		t.Fatalf("wrapped phase %v", p)
	}
	if got := strings.Join(atempoChain(5), ","); got != "atempo=2,atempo=2,atempo=1.250000" {
		t.Fatalf("atempo chain %s", got)
	}
	if got := strings.Join(atempoChain(0.3), ","); got != "atempo=0.5,atempo=0.600000" {
		t.Fatalf("atempo chain %s", got)
	}
}

func TestSceneGraphLayerTimingAndVolume(t *testing.T) {
	layers := []sceneGraphLayer{{Media: "m.mp4", Motion: true, Audio: true, Duration: 4, Speed: 2, Offset: 3, Volume: 0.5, Region: SceneRect{W: 64, H: 64}, Mask: "k.png"}}
	g := strings.Join(sceneGraphArgs("b.png", 100, 100, layers, true, 8, SceneRect{}), " ")
	for _, want := range []string{"-stream_loop -1 -ss 1.5000 -i m.mp4", "setpts=PTS*2.000000", "[1:a]atempo=0.500000,volume=0.5000[at0]"} {
		if !strings.Contains(g, want) {
			t.Errorf("graph lacks %q:\n%s", want, g)
		}
	}
}

func TestSceneTimingInExportAndLength(t *testing.T) {
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-00000000e001"
	dir, _ := sceneDir(id)
	_ = os.MkdirAll(dir, 0755)
	clip := filepath.Join(dir, "c.mp4")
	_ = os.WriteFile(clip, []byte("x"), 0644)
	mask := maskDataURL(t, 10, 10, SceneRect{W: 5, H: 5})
	layer := func(id string, opts map[string]any, d float64) *SceneLayer {
		return &SceneLayer{ID: id, Kind: SceneKindMotion, Mask: mask, Options: opts,
			Versions: []SceneVersion{{Path: clip, Duration: d, Audio: true, Region: SceneRect{W: 5, H: 5}}}, Active: 0}
	}
	s := &Scene{ID: id, Width: 10, Height: 10, Layers: []*SceneLayer{
		layer("a", map[string]any{"speed": 2.0}, 6),                // 3 s at double speed
		layer("b", map[string]any{"speed": 0.5, "sound": true}, 2), // 4 s at half speed: the longest
		layer("c", map[string]any{"sound": true, "volume": 0.0}, 1),
	}}
	if d := sceneDuration(s); d != 4 {
		t.Fatalf("scene length %v, want 4 (b's 2 s clip at half speed)", d)
	}
	gl, err := sceneGraphLayers(dir, s.Layers, false)
	if err != nil {
		t.Fatal(err)
	}
	if gl[0].Speed != 0.5 || gl[1].Speed != 2 || gl[2].Audio {
		t.Fatalf("speeds %v %v, c audio %v (volume 0 mutes)", gl[0].Speed, gl[1].Speed, gl[2].Audio)
	}
}

// An offset layer starts its loop part-way through (and keeps looping past
// the clip's end); a half-speed layer takes twice as long per loop.
func TestSceneExportOffsetAndSpeed(t *testing.T) {
	ffmpegTest(t)
	dir := t.TempDir()
	clip := filepath.Join(dir, "c.mp4")
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "nullsrc=s=64x48:r=24,geq=lum=128+90*sin(X/6-PI*T):cb=128:cr=128",
		"-t", "2", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", clip}); err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(dir, "base.png")
	solidPNG(t, base, 64, 48, color.RGBA{0, 0, 0, 255})
	mask := filepath.Join(dir, "mask.png")
	if err := os.WriteFile(mask, func() []byte {
		b, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(maskDataURL(t, 64, 48, SceneRect{W: 64, H: 48}), "data:image/png;base64,"))
		return b
	}(), 0644); err != nil {
		t.Fatal(err)
	}
	src := framesOf(t, clip)
	nearest := func(f image.Image) int {
		best, bd := -1, math.Inf(1)
		for i, s := range src {
			if d := meanAbs(f, s, s.Bounds()); d < bd {
				best, bd = i, d
			}
		}
		return best
	}
	render := func(g sceneGraphLayer, dur float64) []image.Image {
		g.Media, g.Mask, g.Motion, g.Region, g.Duration = clip, mask, true, SceneRect{W: 64, H: 48}, 2
		out := filepath.Join(dir, fmt.Sprintf("o%v_%v.mp4", g.Offset, g.Speed))
		if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(base, 64, 48, []sceneGraphLayer{g}, true, dur, SceneRect{}), out)); err != nil {
			t.Fatal(err)
		}
		return framesOf(t, out)
	}
	off := render(sceneGraphLayer{Speed: 1, Offset: 0.5}, 3)
	if n := nearest(off[0]); n < 11 || n > 13 {
		t.Errorf("offset 0.5 s: first frame is clip frame %d, want ~12", n)
	}
	// 3 s of a 2 s clip from 0.5 s: frame 48 (2 s in) wrapped to clip 0.5 s.
	if n := nearest(off[48]); n < 11 || n > 13 {
		t.Errorf("offset loop: frame 48 is clip frame %d, want ~12", n)
	}
	slow := render(sceneGraphLayer{Speed: 2}, 4)
	if n := nearest(slow[48]); n < 23 || n > 25 {
		t.Errorf("half speed: 2 s in shows clip frame %d, want ~24", n)
	}
}
