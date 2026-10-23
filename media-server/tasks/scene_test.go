package tasks

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

func withSceneRoot(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	sceneRootOverride = dir
	t.Cleanup(func() { sceneRootOverride = "" })
	return dir
}

// writeTestScene stores a scene document without going through ffmpeg.
func writeTestScene(t *testing.T, s *Scene) {
	t.Helper()
	dir, err := sceneDir(s.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	sceneMu.Lock()
	defer sceneMu.Unlock()
	if err := saveSceneLocked(s); err != nil {
		t.Fatal(err)
	}
}

func TestSceneRectClamp(t *testing.T) {
	r := SceneRect{X: -20, Y: 10, W: 100, H: 500}.clamp(64, 200)
	if r != (SceneRect{X: 0, Y: 10, W: 64, H: 190}) {
		t.Fatalf("clamp = %+v", r)
	}
	if r := (SceneRect{X: 300, Y: 300, W: 10, H: 10}).clamp(320, 320); r.X+r.W > 320 || r.W < 16 {
		t.Fatalf("clamp past edge = %+v", r)
	}
}

func TestSceneMotionSize(t *testing.T) {
	for _, r := range []SceneRect{{W: 400, H: 300}, {W: 300, H: 900}, {W: 1920, H: 1080}, {W: 50, H: 50}, {W: 421, H: 507}} {
		w, h := sceneMotionSize(r, 0.5)
		if w%32 != 0 || h%32 != 0 || w < 256 || h < 256 {
			t.Fatalf("%+v -> %dx%d not on the 32 grid / too small", r, w, h)
		}
		got := float64(w) / float64(h)
		trained := false
		for _, c := range sceneCanvases {
			trained = trained || math.Abs(got-c.aspect) < 0.01
		}
		if !trained {
			t.Fatalf("%+v -> %dx%d is not a shape the model is trained on", r, w, h)
		}
	}
	if w, h := sceneMotionSize(SceneRect{W: 421, H: 507}, 0.9); w != 768 || h != 1024 {
		t.Fatalf("a 5:6 region at High = %dx%d, want 3:4 768x1024", w, h)
	}
}

// Motion regions grow to a trained shape around the mask, inside the frame.
func TestSceneMotionRegion(t *testing.T) {
	cases := []struct{ in SceneRect }{
		{SceneRect{X: 509, Y: 261, W: 421, H: 507}}, // the girl: 5:6 -> 3:4, taller
		{SceneRect{X: 0, Y: 0, W: 490, H: 472}},     // near-square at the corner
		{SceneRect{X: 100, Y: 0, W: 300, H: 768}},   // full height: 3:4 would be too tall, so wider
	}
	for _, c := range cases {
		r := sceneMotionRegion(c.in, 1344, 768)
		if r.X < 0 || r.Y < 0 || r.X+r.W > 1344 || r.Y+r.H > 768 {
			t.Fatalf("%+v -> %+v leaves the frame", c.in, r)
		}
		if r.X > c.in.X || r.Y > c.in.Y || r.X+r.W < c.in.X+c.in.W || r.Y+r.H < c.in.Y+c.in.H {
			t.Fatalf("%+v -> %+v doesn't contain the region", c.in, r)
		}
		a := float64(r.W) / float64(r.H)
		trained := false
		for _, cv := range sceneCanvases {
			trained = trained || math.Abs(a-cv.aspect) < 0.01
		}
		if !trained {
			t.Fatalf("%+v -> %+v aspect %.3f is not trained", c.in, r, a)
		}
	}
}

func TestSceneStillLongEdge(t *testing.T) {
	if sceneStillLongEdge(SceneRect{W: 300, H: 200}) != 1024 {
		t.Fatal("small crops should be generated at 1024")
	}
	if sceneStillLongEdge(SceneRect{W: 1500, H: 900}) != 0 {
		t.Fatal("mid-size crops keep their size")
	}
	if sceneStillLongEdge(SceneRect{W: 4000, H: 3000}) != 2048 {
		t.Fatal("huge crops are capped")
	}
}

func TestSceneGraphArgs(t *testing.T) {
	layers := []sceneGraphLayer{
		{Media: "edit.png", Region: SceneRect{X: 10, Y: 20, W: 100, H: 50}, Mask: "m1.png"},
		{Media: "move.mp4", Motion: true, Audio: true, Region: SceneRect{X: 0, Y: 0, W: 64, H: 64}, Mask: "m2.png"},
	}
	still := strings.Join(sceneGraphArgs("base.png", 640, 480, layers, false, 0, SceneRect{X: 1, Y: 2, W: 30, H: 40}), " ")
	for _, want := range []string{
		"-i base.png", "-i edit.png", "-i m1.png", "-i move.mp4",
		"[1:v]scale=100:50", "[2:v]format=rgba,alphaextract,scale=640:480,crop=100:50:10:20",
		"[b0][l0]overlay=10:20", "[b2]crop=30:40:1:2,format=rgb24[vout]", "-frames:v 1",
	} {
		if !strings.Contains(still, want) {
			t.Errorf("still graph lacks %q:\n%s", want, still)
		}
	}
	if strings.Contains(still, "-loop") || strings.Contains(still, "[aout]") {
		t.Errorf("a single frame needs no looping or audio:\n%s", still)
	}

	video := strings.Join(sceneGraphArgs("base.png", 640, 480, layers, true, 7.5, SceneRect{}), " ")
	for _, want := range []string{"-loop 1 -framerate 24 -i base.png", "-stream_loop -1 -i move.mp4", "[3:a]anull[aout]", "-map [aout]", "-t 7.5", "fps=24", "yuv420p"} {
		if !strings.Contains(video, want) {
			t.Errorf("video graph lacks %q:\n%s", want, video)
		}
	}
}

func TestUpdateSceneKeepsServerFields(t *testing.T) {
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-000000000001"
	writeTestScene(t, &Scene{
		ID: id, Width: 800, Height: 600, Layers: []*SceneLayer{
			{ID: "a", Kind: SceneKindStill, Versions: []SceneVersion{{Path: "x.png"}}, Active: 0, Error: "old"},
			{ID: "b", Kind: SceneKindMotion, Versions: []SceneVersion{}, Active: -1},
		},
	})
	in := &Scene{Name: "Renamed", Duration: 99, Layers: []*SceneLayer{
		// Client copy without versions, a new mask and an out-of-frame region.
		{ID: "a", Kind: "weird", Name: "Sky", Region: SceneRect{X: 700, Y: 500, W: 400, H: 400}, Mask: "data:image/png;base64,AAAA", Active: 0, Prompt: "storm"},
		{ID: "c", Kind: SceneKindMotion, Active: 5, Mask: "javascript:alert(1)"},
	}}
	s, err := UpdateScene(nil, id, in)
	if err != nil {
		t.Fatal(err)
	}
	if s.Name != "Renamed" || s.Duration != 60 {
		t.Fatalf("name/duration = %q %v", s.Name, s.Duration)
	}
	if len(s.Layers) != 2 || s.Layers[0].ID != "a" || s.Layers[1].ID != "c" {
		t.Fatalf("layers = %+v", s.Layers)
	}
	a := s.Layers[0]
	if len(a.Versions) != 1 || a.Active != 0 || a.Error != "old" {
		t.Fatalf("server fields of a lost: %+v", a)
	}
	if a.Kind != SceneKindStill || a.Prompt != "storm" || a.Region.X+a.Region.W > 800 || a.Region.Y+a.Region.H > 600 {
		t.Fatalf("editor fields of a: %+v", a)
	}
	c := s.Layers[1]
	if c.Active != -1 || c.Mask != "" || c.Versions == nil {
		t.Fatalf("new layer c: %+v", c)
	}
}

func TestUpdateSceneKeepsMaskOptsAndReorder(t *testing.T) {
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-000000000002"
	writeTestScene(t, &Scene{
		ID: id, Width: 800, Height: 600, Layers: []*SceneLayer{
			{ID: "a", Kind: SceneKindStill, Versions: []SceneVersion{{Path: "a.png", Below: []string{}}}, Active: 0},
			{ID: "b", Kind: SceneKindMotion, Versions: []SceneVersion{{Path: "b.mp4", Below: []string{"a:a.png"}}}, Active: 0},
		},
	})
	// The editor moved b under a and set b's mask options.
	in := &Scene{Layers: []*SceneLayer{
		{ID: "b", Kind: SceneKindMotion, Active: 0, MaskOpts: json.RawMessage(`{"invert":true,"expand":-4}`)},
		{ID: "a", Kind: SceneKindStill, Active: 0},
	}}
	s, err := UpdateScene(nil, id, in)
	if err != nil {
		t.Fatal(err)
	}
	if s.Layers[0].ID != "b" || s.Layers[1].ID != "a" {
		t.Fatalf("order = %s, %s", s.Layers[0].ID, s.Layers[1].ID)
	}
	if string(s.Layers[0].MaskOpts) != `{"invert":true,"expand":-4}` {
		t.Fatalf("maskOpts = %s", s.Layers[0].MaskOpts)
	}
	// What b was made from is kept, so the editor can see a is no longer under it.
	if got := s.Layers[0].Versions[0].Below; len(got) != 1 || got[0] != "a:a.png" {
		t.Fatalf("below = %v", got)
	}
}

func TestSceneBelow(t *testing.T) {
	layers := []*SceneLayer{
		{ID: "s1", Kind: SceneKindStill, Versions: []SceneVersion{{Path: "1.png"}, {Path: "2.png"}}, Active: 1},
		{ID: "hid", Kind: SceneKindStill, Hidden: true, Versions: []SceneVersion{{Path: "h.png"}}, Active: 0},
		{ID: "none", Kind: SceneKindStill, Versions: []SceneVersion{}, Active: -1},
		{ID: "m", Kind: SceneKindMotion, Versions: []SceneVersion{{Path: "m.mp4"}}, Active: 0},
		{ID: "s2", Kind: SceneKindStill, Versions: []SceneVersion{{Path: "3.png"}}, Active: 0},
	}
	got := sceneBelow(layers)
	if len(got) != 2 || got[0] != "s1:2.png" || got[1] != "s2:3.png" {
		t.Fatalf("sceneBelow = %v", got)
	}
}

func TestReconcileRecoversClearedJob(t *testing.T) {
	withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-000000000002"
	dir, _ := sceneDir(id)
	input := filepath.Join(dir, "still-a-1.png")
	writeTestScene(t, &Scene{ID: id, Width: 100, Height: 100, Layers: []*SceneLayer{
		{ID: "a", Kind: SceneKindStill, Versions: []SceneVersion{}, Active: -1,
			Job: &SceneJob{ID: "gone", Region: SceneRect{W: 50, H: 50}, Prompt: "p", Input: input}},
		{ID: "b", Kind: SceneKindStill, Versions: []SceneVersion{}, Active: -1,
			Job: &SceneJob{ID: "gone-too", Input: filepath.Join(dir, "still-b-1.png")}},
	}})
	if err := os.WriteFile(filepath.Join(dir, "still-a-1_edit.png"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	q := jobqueue.NewQueue()
	s, err := GetScene(q, id)
	if err != nil {
		t.Fatal(err)
	}
	a, b := s.Layers[0], s.Layers[1]
	if a.Job != nil || len(a.Versions) != 1 || a.Active != 0 || a.Versions[0].Prompt != "p" {
		t.Fatalf("a not recovered: %+v", a)
	}
	if b.Job != nil || b.Error == "" {
		t.Fatalf("b should report the missing job: %+v", b)
	}
}

func TestInSceneDir(t *testing.T) {
	root := withSceneRoot(t)
	id := "0123abcd-0000-0000-0000-000000000003"
	if !inSceneDir(id, filepath.Join(root, id, "x.png")) {
		t.Fatal("file in the scene folder")
	}
	for _, p := range []string{filepath.Join(root, id), filepath.Join(root, id, "..", "x.png"), filepath.Join(root, "other", "x.png"), ""} {
		if inSceneDir(id, p) {
			t.Fatalf("%q is not in the scene folder", p)
		}
	}
	if _, err := sceneDir("../../etc"); err == nil {
		t.Fatal("ids are validated")
	}
}

// maskDataURL is a full-frame mask covering rect r of a w×h frame.
func maskDataURL(t *testing.T, w, h int, r SceneRect) string {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := r.Y; y < r.Y+r.H; y++ {
		for x := r.X; x < r.X+r.W; x++ {
			img.Set(x, y, color.NRGBA{255, 255, 255, 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func solidPNG(t *testing.T, path string, w, h int, c color.Color) {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, c)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
}

func readPNG(t *testing.T, path string) image.Image {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		t.Fatal(err)
	}
	return img
}

// End to end through the bundled ffmpeg: scene creation normalises the base,
// a still layer composites only inside its mask, and a crop of the flattened
// stills is what the next layer's engine would see.
func TestSceneFlattenWithFFmpeg(t *testing.T) {
	if deps.BundledOrEmpty("ffmpeg") == "" {
		t.Skip("bundled ffmpeg not available")
	}
	withSceneRoot(t)
	src := filepath.Join(t.TempDir(), "photo.png")
	solidPNG(t, src, 160, 120, color.RGBA{0, 0, 255, 255})

	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	if s.Width != 160 || s.Height != 120 {
		t.Fatalf("base is %dx%d", s.Width, s.Height)
	}
	dir, _ := sceneDir(s.ID)
	edit := filepath.Join(dir, "still-a-1_edit.png")
	// The engine's result for a 80×60 region, generated larger (as retouch
	// does for small regions) and scaled back into place.
	solidPNG(t, edit, 160, 120, color.RGBA{255, 0, 0, 255})
	region := SceneRect{X: 40, Y: 30, W: 80, H: 60}
	layer := &SceneLayer{ID: "a", Kind: SceneKindStill, Region: region,
		Mask:     maskDataURL(t, 80, 60, SceneRect{X: 30, Y: 22, W: 20, H: 15}), // half-res mask: 60..100 × 44..74
		Versions: []SceneVersion{{Path: edit, Region: region}}, Active: 0}
	layers, err := sceneGraphLayers(dir, s.Width, s.Height, []*SceneLayer{layer}, true)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "flat.png")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, layers, false, 0, SceneRect{}), out)); err != nil {
		t.Fatal(err)
	}
	img := readPNG(t, out)
	isRed := func(x, y int) bool {
		r, g, b, _ := img.At(x, y).RGBA()
		return r > 0xd000 && g < 0x3000 && b < 0x3000
	}
	if !isRed(80, 60) {
		t.Errorf("inside the mask should be the edit")
	}
	if isRed(45, 35) || isRed(10, 10) {
		t.Errorf("outside the mask (but inside the region) must stay the base")
	}

	crop := filepath.Join(dir, "crop.png")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, layers, false, 0, SceneRect{X: 70, Y: 50, W: 40, H: 30}), crop)); err != nil {
		t.Fatal(err)
	}
	if b := readPNG(t, crop).Bounds(); b.Dx() != 40 || b.Dy() != 30 {
		t.Fatalf("crop is %v", b)
	}

	// A 1 s video export of the scene.
	if testing.Short() {
		return
	}
	mp4 := filepath.Join(dir, "out.mp4")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, layers, true, 1, SceneRect{}), mp4)); err != nil {
		t.Fatal(err)
	}
	if d := probeVideoDuration(context.Background(), mp4); d < 0.9 || d > 1.2 {
		t.Fatalf("export duration %.2f", d)
	}

	// A motion layer with sound: a 1 s clip (as reshoot would deliver for a
	// region) loops for the 2.5 s export and its soundtrack is mixed in.
	clip := filepath.Join(dir, "motion-b-1_reshoot.mp4")
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "testsrc=size=64x48:rate=24:duration=1", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", clip}); err != nil {
		t.Fatal(err)
	}
	mregion := SceneRect{X: 0, Y: 0, W: 64, H: 48}
	motion := &SceneLayer{ID: "b", Kind: SceneKindMotion, Region: mregion, Options: map[string]any{"sound": true},
		Mask:     maskDataURL(t, 160, 120, SceneRect{X: 8, Y: 8, W: 40, H: 30}),
		Versions: []SceneVersion{{Path: clip, Region: mregion, Duration: 1, Audio: true}}, Active: 0}
	s.Layers = []*SceneLayer{layer, motion}
	writeTestScene(t, s)
	all, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers, false)
	if err != nil || len(all) != 2 || !all[1].Audio {
		t.Fatalf("graph layers = %+v, %v", all, err)
	}
	full := filepath.Join(dir, "full.mp4")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, all, true, 2.5, SceneRect{}), full)); err != nil {
		t.Fatal(err)
	}
	if d := probeVideoDuration(context.Background(), full); d < 2.4 || d > 2.7 {
		t.Fatalf("looped export duration %.2f", d)
	}
	if !probeHasAudio(full) {
		t.Fatal("the motion layer's sound should be in the export")
	}

	s2, err := BoomerangVersion(context.Background(), s.ID, "b", 0)
	if err != nil {
		t.Fatal(err)
	}
	b := s2.Layers[1]
	if len(b.Versions) != 2 || b.Active != 1 || math.Abs(b.Versions[1].Duration-2) > 0.1 {
		t.Fatalf("boomerang version: %+v", b.Versions)
	}
	if d := probeVideoDuration(context.Background(), b.Versions[1].Path); d < 1.9 || d > 2.2 {
		t.Fatalf("boomerang duration %.2f", d)
	}
}

func TestSceneMotionPrompt(t *testing.T) {
	p := sceneMotionPrompt(sceneMotionPromptInput{Motion: "the clouds roll.", LockEdges: true})
	for _, want := range []string{"[keyframe completion]", "starts exactly from <Picture 1>", "The motion: the clouds roll.", "locked off", "no people, faces", "Silent.", "near the edges of the frame stays perfectly still"} {
		if !strings.Contains(p, want) {
			t.Errorf("prompt lacks %q:\n%s", want, p)
		}
	}
	if strings.Contains(p, "breathing") || strings.Contains(p, "blinks") {
		t.Errorf("region prompts must not assume a person:\n%s", p)
	}
	p = sceneMotionPrompt(sceneMotionPromptInput{Subject: "the waterfall", Sound: true})
	if !strings.Contains(p, "the waterfall, as shown in <Picture 1>") || !strings.Contains(p, "drift gently") || strings.Contains(p, "Silent.") || strings.Contains(p, "edges of the frame stays") {
		t.Errorf("ambient + subject + sound prompt:\n%s", p)
	}
}

// The lock pass: wherever the engine's video doesn't move, every frame comes
// out exactly as the still, even when the video drifted in tone; the moving
// part keeps moving.
func TestSceneLockStaticEdges(t *testing.T) {
	if deps.BundledOrEmpty("ffmpeg") == "" {
		t.Skip("bundled ffmpeg not available")
	}
	dir := t.TempDir()
	still := filepath.Join(dir, "crop.png")
	video := filepath.Join(dir, "crop_reshoot.mp4")
	out := filepath.Join(dir, "crop_locked.mp4")
	// The still: a gradient. The "engine's" video: the same picture 6% brighter
	// (tone drift) with a white square moving through the top-left quarter.
	if err := runSceneFFmpeg(context.Background(), []string{"-f", "lavfi", "-i", "gradients=s=192x128:c0=0x203040:c1=0xc08040:x0=0:y0=0:x1=192:y1=128", "-frames:v", "1", still}); err != nil {
		t.Fatal(err)
	}
	if err := runSceneFFmpeg(context.Background(), []string{"-loop", "1", "-i", still, "-f", "lavfi", "-i", "color=white:s=16x16",
		"-filter_complex", "[0:v]eq=brightness=0.06[b];[b][1:v]overlay=x='5+t*40':y=10:shortest=1,format=yuv420p",
		"-t", "1", "-r", "24", "-c:v", "libx264", "-crf", "12", video}); err != nil {
		t.Fatal(err)
	}
	stats, err := lockMotionEdges(context.Background(), video, still, out, func(int) {})
	if err != nil {
		t.Fatal(err)
	}
	if stats.Frames < 20 || stats.Locked < 0.6 {
		t.Fatalf("stats %+v", stats)
	}
	frameAt := func(at string) image.Image {
		p := filepath.Join(dir, "f"+at+".png")
		if err := runSceneFFmpeg(context.Background(), []string{"-ss", at, "-i", out, "-frames:v", "1", p}); err != nil {
			t.Fatal(err)
		}
		return readPNG(t, p)
	}
	s := readPNG(t, still)
	diff := func(a image.Image, x, y int) int {
		r1, g1, b1, _ := a.At(x, y).RGBA()
		r2, g2, b2, _ := s.At(x, y).RGBA()
		d := func(p, q uint32) int {
			v := int(p>>8) - int(q>>8)
			if v < 0 {
				return -v
			}
			return v
		}
		return max(d(r1, r2), d(g1, g2), d(b1, b2))
	}
	f0, f1 := frameAt("0.2"), frameAt("0.8")
	// Static far corner: the still (only the mp4's own yuv rounding left),
	// though the source video was 6% brighter there.
	for _, f := range []image.Image{f0, f1} {
		if d := diff(f, 170, 110); d > 6 {
			t.Errorf("static pixel is %d levels off the still", d)
		}
	}
	// The moving square is still there and moves.
	if diff(f1, 5+32+8, 18) < 40 || diff(f0, 5+32+8, 18) > 40 {
		t.Errorf("the motion should survive the lock")
	}
}

// chainScene stores a scene whose motion layer has a reshoot -> scene-lock
// generation queued on q, the way GenerateLayer leaves it.
func chainScene(t *testing.T, q *jobqueue.Queue, id string) (dir string, gen, lock string) {
	t.Helper()
	withSceneRoot(t)
	dir, _ = sceneDir(id)
	gen, lock = "gen-"+id[len(id)-4:], "lock-"+id[len(id)-4:]
	input := filepath.Join(dir, "motion-m-1.png")
	if _, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{
		{ID: gen, Command: "reshoot", Input: input},
		{ID: lock, Command: "scene-lock", Arguments: []string{"--still=" + input}, Dependencies: []string{gen}},
	}}); err != nil {
		t.Fatal(err)
	}
	writeTestScene(t, &Scene{ID: id, Width: 100, Height: 100, Layers: []*SceneLayer{{
		ID: "m", Name: "Motion 1", Kind: SceneKindMotion, Versions: []SceneVersion{}, Active: -1,
		Job: &SceneJob{ID: lock, Chain: []string{gen, lock}, State: "pending", Region: SceneRect{W: 50, H: 50}, Prompt: "p", Input: input},
	}}})
	return dir, gen, lock
}

func claim(t *testing.T, q *jobqueue.Queue, want string) {
	t.Helper()
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != want {
		t.Fatalf("claimed %+v (%v), want %s", j, err, want)
	}
}

func finish(t *testing.T, q *jobqueue.Queue, id, out string) {
	t.Helper()
	if err := os.WriteFile(out, []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}
	_ = q.RegisterOutputFile(id, out)
	_ = q.CompleteJob(id)
}

func state(q *jobqueue.Queue, id string) jobqueue.JobState {
	j, _, _ := q.JobSnapshot(id, 0)
	return j.State
}

// The bug report: the lock step got cancelled before it ran while the engine
// was still generating. The generation must not be dropped; the engine keeps
// going and the lock is queued again.
func TestSceneChainLockCancelledBeforeItRan(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c001"
	dir, gen, lock := chainScene(t, q, id)
	claim(t, q, gen)
	_ = q.SetJobProgress(gen, 3, 10)
	_ = q.CancelJob(lock)

	s, err := GetScene(q, id)
	if err != nil {
		t.Fatal(err)
	}
	l := s.Layers[0]
	if l.Job == nil || l.Error != "" || l.Job.State != "in_progress" || l.Job.Done != 3 || l.Job.Step != "reshoot" {
		t.Fatalf("the running generation should still be followed: job %+v err %q", l.Job, l.Error)
	}
	if state(q, gen) != jobqueue.StateInProgress {
		t.Fatal("the engine step must keep running")
	}
	// The engine finishes: the lock is queued again on its output.
	raw := filepath.Join(dir, ".loki-temp", "raw.mp4")
	_ = os.MkdirAll(filepath.Dir(raw), 0755)
	finish(t, q, gen, raw)
	s, _ = GetScene(q, id)
	l = s.Layers[0]
	if l.Job == nil || l.Job.Requeued != 1 || l.Job.ID == lock || len(l.Job.Chain) != 2 || l.Job.Chain[0] != gen || l.Job.Step != "scene-lock" {
		t.Fatalf("lock should be queued again: %+v", l.Job)
	}
	relock := l.Job.ID
	claim(t, q, relock)
	if j, _, _ := q.JobSnapshot(relock, 0); j.Input != raw {
		t.Fatalf("re-queued lock input = %q, want the engine output", j.Input)
	}
	finish(t, q, relock, filepath.Join(dir, "motion-m-1_locked.mp4"))
	s, _ = GetScene(q, id)
	l = s.Layers[0]
	if l.Job != nil || len(l.Versions) != 1 || !strings.HasSuffix(l.Versions[0].Path, "_locked.mp4") || l.Versions[0].Note != "" {
		t.Fatalf("locked version expected: job %+v versions %+v", l.Job, l.Versions)
	}
}

func TestSceneChainEngineCancelledCancelsTheRest(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c002"
	_, gen, lock := chainScene(t, q, id)
	claim(t, q, gen)
	_ = q.CancelJob(gen)
	s, _ := GetScene(q, id)
	l := s.Layers[0]
	if l.Job != nil || !strings.Contains(l.Error, "cancelled") {
		t.Fatalf("layer: job %+v err %q", l.Job, l.Error)
	}
	if state(q, lock) != jobqueue.StateCancelled {
		t.Fatal("the lock step must not be left pending")
	}
}

func TestSceneChainLockFailedKeepsEngineResult(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c003"
	dir, gen, lock := chainScene(t, q, id)
	claim(t, q, gen)
	finish(t, q, gen, filepath.Join(dir, "raw.mp4"))
	claim(t, q, lock)
	_ = q.ErrorJob(lock)
	s, _ := GetScene(q, id)
	l := s.Layers[0]
	if l.Job != nil || len(l.Versions) != 1 || !strings.Contains(l.Versions[0].Note, "edges not locked") {
		t.Fatalf("engine result should be kept: job %+v versions %+v err %q", l.Job, l.Versions, l.Error)
	}
}

func TestSceneCancelLayerCancelsWholeChain(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c004"
	_, gen, lock := chainScene(t, q, id)
	claim(t, q, gen)
	s, err := CancelLayer(q, id, "m", "")
	if err != nil || s.Layers[0].Job != nil {
		t.Fatalf("cancel: %v %+v", err, s.Layers[0].Job)
	}
	if state(q, gen) != jobqueue.StateCancelled || state(q, lock) != jobqueue.StateCancelled {
		t.Fatalf("both steps cancelled? gen %v lock %v", state(q, gen), state(q, lock))
	}
	_, lines, _ := q.JobSnapshot(gen, 5)
	if len(lines) == 0 || !strings.Contains(lines[len(lines)-1], "Cancel pressed on layer \"Motion 1\"") {
		t.Fatalf("the job log should say why: %v", lines)
	}
}

func TestSceneDeletingLayerCancelsWholeChain(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c005"
	_, gen, lock := chainScene(t, q, id)
	claim(t, q, gen)
	if _, err := UpdateScene(q, id, &Scene{Layers: []*SceneLayer{}}); err != nil {
		t.Fatal(err)
	}
	if state(q, gen) != jobqueue.StateCancelled || state(q, lock) != jobqueue.StateCancelled {
		t.Fatalf("both steps cancelled? gen %v lock %v", state(q, gen), state(q, lock))
	}
}

// queuedScene is chainScene plus a second motion generation queued behind it.
func queuedScene(t *testing.T, q *jobqueue.Queue, id string) (dir, gen, lock, gen2, lock2 string) {
	t.Helper()
	dir, gen, lock = chainScene(t, q, id)
	gen2, lock2 = gen+"-2", lock+"-2"
	input := filepath.Join(dir, "motion-m-1.2.png")
	if _, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{
		{ID: gen2, Command: "reshoot", Input: input},
		{ID: lock2, Command: "scene-lock", Arguments: []string{"--still=" + input}, Dependencies: []string{gen2}},
	}}); err != nil {
		t.Fatal(err)
	}
	s, _ := GetScene(q, id)
	s.Layers[0].Queued = []*SceneJob{{ID: lock2, Chain: []string{gen2, lock2}, State: "pending", Region: SceneRect{W: 50, H: 50}, Prompt: "p2", Input: input}}
	writeTestScene(t, s)
	return dir, gen, lock, gen2, lock2
}

// A generation started while another runs waits its turn, then lands as the
// next version.
func TestSceneQueuedGenerationRunsAfterTheFirst(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c006"
	dir, gen, lock, gen2, lock2 := queuedScene(t, q, id)
	claim(t, q, gen)
	finish(t, q, gen, filepath.Join(dir, "raw.mp4"))
	claim(t, q, lock)
	finish(t, q, lock, filepath.Join(dir, "motion-m-1_locked.mp4"))
	s, _ := GetScene(q, id)
	l := s.Layers[0]
	if len(l.Versions) != 1 || l.Job == nil || l.Job.ID != lock2 || len(l.Queued) != 0 {
		t.Fatalf("the queued generation should be running: versions %d job %+v queued %d", len(l.Versions), l.Job, len(l.Queued))
	}
	claim(t, q, gen2)
	finish(t, q, gen2, filepath.Join(dir, "raw2.mp4"))
	claim(t, q, lock2)
	finish(t, q, lock2, filepath.Join(dir, "motion-m-1.2_locked.mp4"))
	s, _ = GetScene(q, id)
	l = s.Layers[0]
	if l.Job != nil || len(l.Versions) != 2 || l.Active != 1 || l.Versions[1].Prompt != "p2" {
		t.Fatalf("second version expected: job %+v versions %+v active %d", l.Job, l.Versions, l.Active)
	}
}

// Both generations ended while nobody looked: one read folds both in.
func TestSceneQueuedGenerationsFoldInOneRead(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c007"
	dir, gen, lock, gen2, lock2 := queuedScene(t, q, id)
	for _, st := range []struct{ id, out string }{{gen, "raw.mp4"}, {lock, "motion-m-1_locked.mp4"}, {gen2, "raw2.mp4"}, {lock2, "motion-m-1.2_locked.mp4"}} {
		claim(t, q, st.id)
		finish(t, q, st.id, filepath.Join(dir, st.out))
	}
	s, _ := GetScene(q, id)
	if l := s.Layers[0]; l.Job != nil || len(l.Queued) != 0 || len(l.Versions) != 2 {
		t.Fatalf("both versions expected: job %+v queued %d versions %d", l.Job, len(l.Queued), len(l.Versions))
	}
}

// Cancelling the running generation starts the queued one; cancelling a
// queued one leaves the running one alone; no id cancels them all.
func TestSceneCancelOneGeneration(t *testing.T) {
	q := jobqueue.NewQueue()
	id := "0123abcd-0000-0000-0000-00000000c008"
	_, gen, lock, gen2, lock2 := queuedScene(t, q, id)
	claim(t, q, gen)
	s, err := CancelLayer(q, id, "m", lock2)
	if err != nil || s.Layers[0].Job == nil || s.Layers[0].Job.ID != lock || len(s.Layers[0].Queued) != 0 {
		t.Fatalf("cancel queued: %v job %+v queued %d", err, s.Layers[0].Job, len(s.Layers[0].Queued))
	}
	if state(q, gen) != jobqueue.StateInProgress || state(q, gen2) != jobqueue.StateCancelled || state(q, lock2) != jobqueue.StateCancelled {
		t.Fatalf("only the queued chain cancelled? gen %v gen2 %v lock2 %v", state(q, gen), state(q, gen2), state(q, lock2))
	}

	q = jobqueue.NewQueue()
	id = "0123abcd-0000-0000-0000-00000000c009"
	_, gen, lock, gen2, lock2 = queuedScene(t, q, id)
	claim(t, q, gen)
	s, err = CancelLayer(q, id, "m", lock)
	if err != nil || s.Layers[0].Job == nil || s.Layers[0].Job.ID != lock2 || s.Layers[0].Error != "" {
		t.Fatalf("cancel running: %v job %+v err %q", err, s.Layers[0].Job, s.Layers[0].Error)
	}
	if state(q, gen) != jobqueue.StateCancelled || state(q, gen2) != jobqueue.StatePending {
		t.Fatalf("gen %v gen2 %v", state(q, gen), state(q, gen2))
	}

	q = jobqueue.NewQueue()
	id = "0123abcd-0000-0000-0000-00000000c00a"
	_, _, _, gen2, _ = queuedScene(t, q, id)
	s, err = CancelLayer(q, id, "m", "")
	if err != nil || s.Layers[0].Job != nil || len(s.Layers[0].Queued) != 0 || state(q, gen2) != jobqueue.StateCancelled {
		t.Fatalf("cancel all: %v job %+v queued %d gen2 %v", err, s.Layers[0].Job, len(s.Layers[0].Queued), state(q, gen2))
	}
}

// colourDataURL is a w×h RGBA PNG data URL painted c inside r (transparent
// elsewhere): a paint layer's raster.
func colourDataURL(t *testing.T, w, h int, r SceneRect, c color.NRGBA) string {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := r.Y; y < r.Y+r.H; y++ {
		for x := r.X; x < r.X+r.W; x++ {
			img.Set(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// A paint layer in the picture is part of what the layers above are made
// from; a guide one is not, but a reference to it renders the paint over that
// picture, aligned to the referencing layer's crop.
func TestScenePaintLayerWithFFmpeg(t *testing.T) {
	if deps.BundledOrEmpty("ffmpeg") == "" {
		t.Skip("bundled ffmpeg not available")
	}
	withSceneRoot(t)
	src := filepath.Join(t.TempDir(), "lineart.png")
	solidPNG(t, src, 160, 120, color.RGBA{0, 0, 255, 255})
	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	dir, _ := sceneDir(s.ID)
	// Painted red over frame 40..80 × 30..60 (the raster is half size).
	paint := &SceneLayer{ID: "p", Kind: SceneKindPaint, Mask: colourDataURL(t, 80, 60, SceneRect{X: 20, Y: 15, W: 20, H: 15}, color.NRGBA{255, 0, 0, 255}),
		Versions: []SceneVersion{}, Active: -1}
	region := SceneRect{X: 20, Y: 20, W: 80, H: 60}
	edit := &SceneLayer{ID: "e", Kind: SceneKindStill, Region: region, Mask: maskDataURL(t, 160, 120, region),
		Options: map[string]any{"refs": []any{map[string]any{"layer": "p", "note": "colour guide"}}}, Versions: []SceneVersion{}, Active: -1}
	s.Layers = []*SceneLayer{paint, edit}

	flatten := func(name string) image.Image {
		layers, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers[:1], true)
		if err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(dir, name)
		if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, layers, false, 0, SceneRect{}), out)); err != nil {
			t.Fatal(err)
		}
		return readPNG(t, out)
	}
	red := func(img image.Image, x, y int) bool {
		r, g, b, _ := img.At(x, y).RGBA()
		return r>>8 > 200 && g>>8 < 60 && b>>8 < 60
	}
	in := flatten("in.png")
	if !red(in, 60, 45) || red(in, 20, 10) {
		t.Fatalf("paint in the picture: (60,45)=%v (20,10)=%v", in.At(60, 45), in.At(20, 10))
	}
	if got := sceneBelow(s.Layers[:1]); len(got) != 1 || got[0] != "p:paint" {
		t.Fatalf("sceneBelow = %v", got)
	}

	paint.Options = map[string]any{"guide": true}
	if guide := flatten("guide.png"); red(guide, 60, 45) {
		t.Fatalf("a guide paint layer is in the picture: %v", guide.At(60, 45))
	}
	refs, err := resolveLayerRefs(context.Background(), s, 1, dir, region, sceneRefs(edit.Options), filepath.Join(dir, "still-e-1.png"))
	if err != nil {
		t.Fatal(err)
	}
	if len(refs) != 1 || refs[0].Path == "" || refs[0].Kind != "image" || refs[0].Note != "colour guide" {
		t.Fatalf("refs = %+v", refs)
	}
	ref := readPNG(t, refs[0].Path)
	if b := ref.Bounds(); b.Dx() != region.W || b.Dy() != region.H {
		t.Fatalf("reference is %v, want the region's %dx%d", b, region.W, region.H)
	}
	// Frame (60,45) is (40,25) in the crop; frame (25,25) = (5,5) is the picture below.
	if !red(ref, 40, 25) || red(ref, 5, 5) {
		t.Fatalf("reference: (40,25)=%v (5,5)=%v", ref.At(40, 25), ref.At(5, 5))
	}
}

func TestSceneFinishPrompt(t *testing.T) {
	refs := []sceneRef{{Path: "a.png", Kind: "image"}, {Kind: "image", Layer: "p1"}}
	got := sceneFinishPrompt(map[string]any{"finish": FinishColour, "paint": "p1"}, "warm autumn tones.", refs)
	if !strings.Contains(got, "painted on <image3> as a guide: warm autumn tones. The paint") {
		t.Fatalf("colour: %s", got)
	}
	got = sceneFinishPrompt(map[string]any{"finish": FinishSketch}, "", nil)
	if !strings.Contains(got, "stands in for the real thing: replace it with the real thing as") || strings.Contains(got, "{") {
		t.Fatalf("sketch, no words: %s", got)
	}
	got = sceneFinishPrompt(map[string]any{"finish": FinishSketch, "grounding": "Paint {what} in <image1>."}, "a sun", nil)
	if got != "Paint a sun in <image1>." {
		t.Fatalf("custom grounding: %s", got)
	}
	if got := sceneFinishPrompt(map[string]any{}, "make it night", nil); got != "make it night" {
		t.Fatalf("plain layer: %s", got)
	}
}
