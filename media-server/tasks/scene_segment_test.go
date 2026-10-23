package tasks

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// fakeSegMain is a stand-in for `loki-segment --serve` (LOKI_FAKE_SEG=1): it
// answers open with the image's size and segment with a 20x20 square mask
// round the first click (or the middle for text), logging each open to
// FAKE_SEG_LOG.
func fakeSegMain() int {
	in := bufio.NewScanner(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	w, h := 0, 0
	say := func(v any) {
		b, _ := json.Marshal(v)
		out.Write(append(b, '\n'))
		out.Flush()
	}
	for in.Scan() {
		var req map[string]any
		if err := json.Unmarshal(in.Bytes(), &req); err != nil {
			say(map[string]any{"error": "bad json"})
			continue
		}
		switch req["cmd"] {
		case "open":
			path, _ := req["path"].(string)
			f, err := os.Open(path)
			if err != nil {
				say(map[string]any{"error": err.Error()})
				continue
			}
			cfg, _, err := image.DecodeConfig(f)
			f.Close()
			if err != nil {
				say(map[string]any{"error": err.Error()})
				continue
			}
			w, h = cfg.Width, cfg.Height
			if l := os.Getenv("FAKE_SEG_LOG"); l != "" {
				lf, _ := os.OpenFile(l, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
				fmt.Fprintln(lf, "open", path)
				lf.Close()
			}
			say(map[string]any{"ok": true, "width": w, "height": h, "frames": 1})
		case "segment":
			cx, cy := w/2, h/2
			label := "click"
			if t, ok := req["text"].(string); ok {
				label = t
			} else if pts, ok := req["points"].([]any); ok && len(pts) > 0 {
				p := pts[0].([]any)
				cx, cy = int(p[0].(float64)), int(p[1].(float64))
			}
			dir, _ := req["out"].(string)
			os.MkdirAll(dir, 0o755)
			img := image.NewGray(image.Rect(0, 0, w, h))
			for y := cy - 10; y < cy+10; y++ {
				for x := cx - 10; x < cx+10; x++ {
					img.Set(x, y, color.Gray{255})
				}
			}
			mp := filepath.Join(dir, "1.png")
			f, _ := os.Create(mp)
			png.Encode(f, img)
			f.Close()
			say(map[string]any{"width": w, "height": h, "objects": []any{map[string]any{
				"id": 1, "label": label, "score": 0.9, "bbox": []int{cx - 10, cy - 10, 20, 20}, "mask": mp}}})
		case "close":
			say(map[string]any{"ok": true})
		default:
			say(map[string]any{"error": "unknown cmd"})
		}
	}
	return 0
}

// A pick goes through one session: the picture is flattened and opened once,
// each click comes back as a cropped mask where it was clicked; while a
// generation holds the GPU each pick runs in a session of its own.
func TestSceneSegmentSession(t *testing.T) {
	ffmpegTest(t)
	withSceneRoot(t)
	exe, _ := os.Executable()
	segBinary = func() (string, error) { return exe, nil }
	t.Setenv("LOKI_FAKE_SEG", "1")
	log := filepath.Join(t.TempDir(), "opens.log")
	t.Setenv("FAKE_SEG_LOG", log)
	t.Cleanup(closeSegSessions)

	src := filepath.Join(t.TempDir(), "photo.png")
	solidPNG(t, src, 160, 120, color.RGBA{0, 0, 255, 255})
	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	opens := func() int {
		b, _ := os.ReadFile(log)
		return strings.Count(string(b), "open ")
	}
	for i, pt := range [][3]float64{{40, 30, 1}, {100, 80, 1}} {
		res, err := SegmentScene(context.Background(), s.ID, SegmentRequest{Points: [][3]float64{pt}})
		if err != nil {
			t.Fatal(err)
		}
		if res.Width != 160 || len(res.Objects) != 1 {
			t.Fatalf("pick %d: %+v", i, res)
		}
		o := res.Objects[0]
		// The mask is cropped to what it covers (+2 px for the soft edge).
		if o.Rect.X != int(pt[0])-12 || o.Rect.Y != int(pt[1])-12 || o.Rect.W != 24 || !strings.HasPrefix(o.Mask, "data:image/png;base64,") {
			t.Fatalf("pick %d: rect %+v", i, o.Rect)
		}
	}
	if n := opens(); n != 1 {
		t.Fatalf("the picture was opened %d times, want once", n)
	}
	// A generation starts: the session lets go, and picks don't keep one.
	end := beginGPUHeavy()
	if _, err := SegmentScene(context.Background(), s.ID, SegmentRequest{Text: "person"}); err != nil {
		t.Fatal(err)
	}
	segSessions.Lock()
	kept := len(segSessions.m)
	segSessions.Unlock()
	end()
	if kept != 0 || opens() != 2 {
		t.Fatalf("during a generation: %d sessions kept, %d opens", kept, opens())
	}
}

func TestMergeMattes(t *testing.T) {
	dir := t.TempDir()
	write := func(obj string, n int, x0 int) {
		d := filepath.Join(dir, obj)
		os.MkdirAll(d, 0o755)
		img := image.NewGray(image.Rect(0, 0, 8, 4))
		for y := 0; y < 4; y++ {
			for x := x0; x < x0+4; x++ {
				img.Set(x, y, color.Gray{200})
			}
		}
		f, _ := os.Create(filepath.Join(d, fmt.Sprintf("%05d.png", n)))
		png.Encode(f, img)
		f.Close()
	}
	for n := 0; n < 3; n++ {
		write("1", n, 0)
		write("2", n, 4)
	}
	out := filepath.Join(dir, "m")
	frames, err := mergeMattes([]string{filepath.Join(dir, "1"), filepath.Join(dir, "2")}, out)
	if err != nil || frames != 3 {
		t.Fatalf("frames %d, err %v", frames, err)
	}
	m := readPNG(t, filepath.Join(out, "00002.png"))
	for _, x := range []int{1, 6} {
		if r, _, _, _ := m.At(x, 2).RGBA(); r>>8 != 200 {
			t.Fatalf("x=%d: %d, want both objects in the matte", x, r>>8)
		}
	}
}

// The export composites a motion layer through its mask times its matte: a
// matte keeping only the left half shows the clip there and the picture
// below on the right, though the mask covers both.
func TestSceneExportThroughMatte(t *testing.T) {
	ffmpegTest(t)
	dir := t.TempDir()
	base := filepath.Join(dir, "base.png")
	solidPNG(t, base, 64, 48, color.RGBA{0, 0, 255, 255})
	clip := filepath.Join(dir, "clip.mp4")
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "color=red:s=32x24:r=24", "-t", "1", "-pix_fmt", "yuv420p", clip}); err != nil {
		t.Fatal(err)
	}
	maskPath := filepath.Join(dir, "mask.png")
	mf, _ := os.Create(maskPath)
	full := image.NewNRGBA(image.Rect(0, 0, 64, 48))
	for i := 3; i < len(full.Pix); i += 4 {
		full.Pix[i], full.Pix[i-1], full.Pix[i-2], full.Pix[i-3] = 255, 255, 255, 255
	}
	png.Encode(mf, full)
	mf.Close()
	matte := filepath.Join(dir, "matte")
	os.MkdirAll(matte, 0o755)
	for n := 0; n < 24; n++ {
		img := image.NewGray(image.Rect(0, 0, 32, 24))
		for y := 0; y < 24; y++ {
			for x := 0; x < 16; x++ {
				img.Set(x, y, color.Gray{255})
			}
		}
		f, _ := os.Create(filepath.Join(matte, fmt.Sprintf("%05d.png", n)))
		png.Encode(f, img)
		f.Close()
	}
	layers := []sceneGraphLayer{{Media: clip, Motion: true, Region: SceneRect{X: 16, Y: 12, W: 32, H: 24}, Mask: maskPath, Duration: 1, Speed: 1, Matte: matte, MatteFPS: "24"}}
	out := filepath.Join(dir, "out.mp4")
	// Longer than the clip: the clip and its matte loop (the bug: an image
	// sequence looped with -stream_loop restarts its timestamps, and the mask
	// x matte blend buffered without end; 42 GB before it was killed).
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if err := runSceneFFmpeg(ctx, append(sceneGraphArgs(base, 64, 48, layers, true, 2.5, SceneRect{}), out)); err != nil {
		t.Fatal(err)
	}
	if d := probeVideoDuration(context.Background(), out); d < 2.4 {
		t.Fatalf("export is %.2f s, want 2.5", d)
	}
	f := grab(t, out, 1.5)
	px := func(x, y int) (uint32, uint32) { r, _, b, _ := f.At(x, y).RGBA(); return r >> 8, b >> 8 }
	if r, b := px(22, 24); r < 180 || b > 80 {
		t.Fatalf("left half of the region: r=%d b=%d, want the clip (red)", r, b)
	}
	if r, b := px(42, 24); r > 80 || b < 180 {
		t.Fatalf("right half of the region: r=%d b=%d, want the picture below (blue): the matte cuts it", r, b)
	}
}

// A video export flattens runs of still layers first, keeping their place in
// the stack: a still above a motion layer still covers it.
func TestSceneCollapseStillsKeepsOrder(t *testing.T) {
	ffmpegTest(t)
	dir := t.TempDir()
	base := filepath.Join(dir, "base.png")
	solidPNG(t, base, 64, 48, color.RGBA{0, 0, 255, 255})
	full := func(name string, c color.NRGBA, r image.Rectangle) string {
		img := image.NewNRGBA(image.Rect(0, 0, 64, 48))
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				img.Set(x, y, c)
			}
		}
		p := filepath.Join(dir, name)
		f, _ := os.Create(p)
		png.Encode(f, img)
		f.Close()
		return p
	}
	red := full("red.png", color.NRGBA{255, 0, 0, 255}, image.Rect(0, 0, 64, 48))
	white := full("white.png", color.NRGBA{255, 255, 255, 255}, image.Rect(0, 0, 64, 48))
	allMask := full("all.png", color.NRGBA{255, 255, 255, 255}, image.Rect(0, 0, 64, 48))
	leftMask := full("left.png", color.NRGBA{255, 255, 255, 255}, image.Rect(0, 0, 24, 48))
	clip := filepath.Join(dir, "green.mp4")
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "color=green:s=64x48:r=24", "-t", "1", "-pix_fmt", "yuv420p", clip}); err != nil {
		t.Fatal(err)
	}
	layers := []sceneGraphLayer{
		{Media: red, Mask: allMask, Region: SceneRect{W: 64, H: 48}},                                       // a still below
		{Media: clip, Motion: true, Mask: allMask, Region: SceneRect{W: 64, H: 48}, Duration: 1, Speed: 1}, // the motion
		{Media: white, Mask: leftMask, Region: SceneRect{W: 64, H: 48}},                                    // a still above, left third
	}
	b, out, err := collapseStills(context.Background(), dir, base, 64, 48, layers)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || !out[0].Motion || out[1].Motion {
		t.Fatalf("collapsed to %d layers: %+v", len(out), out)
	}
	mp4 := filepath.Join(dir, "out.mp4")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(b, 64, 48, out, true, 1, SceneRect{}), mp4)); err != nil {
		t.Fatal(err)
	}
	f := grab(t, mp4, 0.5)
	r, g, bl, _ := f.At(8, 24).RGBA()
	if r>>8 < 200 || g>>8 < 200 || bl>>8 < 200 {
		t.Fatalf("left: %d,%d,%d, want the still above (white) over the motion", r>>8, g>>8, bl>>8)
	}
	r, g, _, _ = f.At(48, 24).RGBA()
	if g>>8 < 90 || r>>8 > 80 {
		t.Fatalf("right: r=%d g=%d, want the motion (green) over the still below", r>>8, g>>8)
	}
}

// A clean plate under a roto is not what the motion layer was made from, and
// the matte's first frame comes back as a mask piece in clip pixels.
func TestSceneCleanPlate(t *testing.T) {
	plate := &SceneLayer{ID: "c", Kind: SceneKindStill, Options: map[string]any{"cleanFor": "m"}, Versions: []SceneVersion{{Path: "c.png"}}, Active: 0}
	still := &SceneLayer{ID: "s", Kind: SceneKindStill, Versions: []SceneVersion{{Path: "s.png"}}, Active: 0}
	if got := sceneBelow([]*SceneLayer{still, plate}); len(got) != 1 || got[0] != "s:s.png" {
		t.Fatalf("sceneBelow = %v, want the clean plate left out", got)
	}

	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-00000000c1ea"
	writeTestScene(t, &Scene{ID: id, Width: 64, Height: 48})
	dir, _ := sceneDir(id)
	matte := filepath.Join(dir, "clip_matte")
	os.MkdirAll(matte, 0o755)
	img := image.NewGray(image.Rect(0, 0, 32, 24))
	for y := 4; y < 12; y++ {
		for x := 10; x < 20; x++ {
			img.Set(x, y, color.Gray{255})
		}
	}
	f, _ := os.Create(filepath.Join(matte, "00000.png"))
	png.Encode(f, img)
	f.Close()
	writeTestScene(t, &Scene{ID: id, Width: 64, Height: 48, Layers: []*SceneLayer{{ID: "m", Kind: SceneKindMotion, Active: 0,
		Versions: []SceneVersion{{Path: filepath.Join(dir, "clip.mp4"), Track: &SceneTrack{State: "completed", Dir: matte, Frames: 1}}}}}})
	url, rect, w, h, err := VersionMatteFrame(id, "m", 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if w != 32 || h != 24 || rect.X != 8 || rect.Y != 2 || rect.W != 14 || !strings.HasPrefix(url, "data:image/png") {
		t.Fatalf("matte frame: %dx%d rect %+v", w, h, rect)
	}
}
