package tasks

import (
	"context"
	"encoding/json"
	"image/color"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/features"
	"github.com/stevecastle/shrike/jobqueue"
)

func TestSceneDreamSize(t *testing.T) {
	for _, r := range []SceneRect{{W: 200, H: 120}, {W: 2000, H: 3000}, {W: 1344, H: 768}} {
		w, h := sceneDreamSize(r)
		if w%16 != 0 || h%16 != 0 {
			t.Fatalf("%+v -> %dx%d off the 16 grid", r, w, h)
		}
		if mp := float64(w*h) / 1e6; mp < 0.85 || mp > 1.15 {
			t.Fatalf("%+v -> %dx%d = %.2f MP, want ~1", r, w, h, mp)
		}
		if a, b := float64(r.W)/float64(r.H), float64(w)/float64(h); a/b > 1.05 || b/a > 1.05 {
			t.Fatalf("%+v -> %dx%d changes the shape", r, w, h)
		}
	}
}

// An Edit layer on Kromoss: a dream job that inpaints the crop through the
// layer's mask (cropped to the region, white = redraw) at ~1 MP.
func TestSceneGenerateWithDream(t *testing.T) {
	if !features.Dream {
		t.Skip("built without -tags dream: Kromoss layers run as Retouch")
	}
	ffmpegTest(t)
	withSceneRoot(t)
	q := jobqueue.NewQueue()
	src := filepath.Join(t.TempDir(), "photo.png")
	solidPNG(t, src, 320, 240, color.RGBA{40, 80, 120, 255})
	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	in := *s
	region := SceneRect{X: 60, Y: 40, W: 200, H: 160}
	in.Layers = []*SceneLayer{{ID: "e", Kind: SceneKindStill, Mask: maskDataURL(t, 320, 240, SceneRect{X: 100, Y: 80, W: 100, H: 80}), Region: region,
		Prompt: "a stone well", Options: map[string]any{"engine": "dream", "steps": 12.0}}}
	if _, err := UpdateScene(q, s.ID, &in); err != nil {
		t.Fatal(err)
	}
	s2, err := GenerateLayer(context.Background(), q, s.ID, "e")
	if err != nil {
		t.Fatal(err)
	}
	job, _, _ := q.JobSnapshot(s2.Layers[0].Job.Chain[0], 0)
	if job.Command != "dream" {
		t.Fatalf("command %q", job.Command)
	}
	arg := func(k string) string {
		for _, a := range job.Arguments {
			if strings.HasPrefix(a, "--"+k+"=") {
				return strings.TrimPrefix(a, "--"+k+"=")
			}
		}
		return ""
	}
	if arg("prompt") != "a stone well" || arg("strength") != "1" || arg("turbo") != "true" || arg("nolibrary") != "true" {
		t.Fatalf("args %v", job.Arguments)
	}
	w, h := sceneDreamSize(region)
	if arg("size") != strconv.Itoa(w)+"x"+strconv.Itoa(h) {
		t.Fatalf("size %q", arg("size"))
	}
	// The mask: the region's crop of the layer's mask, white where it redraws.
	m := readPNG(t, arg("mask"))
	if b := m.Bounds(); b.Dx() != region.W || b.Dy() != region.H {
		t.Fatalf("mask %v, want the region's %dx%d", b, region.W, region.H)
	}
	in1, _, _, _ := m.At(90, 80).RGBA()  // frame (150,120): inside the mask
	out1, _, _, _ := m.At(10, 10).RGBA() // frame (70,50): outside
	if in1>>8 < 200 || out1>>8 > 50 {
		t.Fatalf("mask values in %d, out %d", in1>>8, out1>>8)
	}
}

func itoa(n int) string {
	return strings.TrimSpace(strings.Repeat(" ", 0) + formatFloatArg(float64(n)))
}

// Masks live with versions: a generation is made with the mask the editor
// sends (the layer's minus a fit to the shown result), the version keeps it,
// and the editor's later changes to a version's mask are kept by its path.
func TestSceneVersionMasks(t *testing.T) {
	ffmpegTest(t)
	withSceneRoot(t)
	q := jobqueue.NewQueue()
	src := filepath.Join(t.TempDir(), "photo.png")
	solidPNG(t, src, 320, 240, color.RGBA{40, 80, 120, 255})
	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	in := *s
	layerShapes := json.RawMessage(`[{"t":"rect","x":40,"y":40,"w":240,"h":160},{"t":"image","fit":"v1.png"}]`)
	in.Layers = []*SceneLayer{{ID: "e", Kind: SceneKindStill, Mask: maskDataURL(t, 320, 240, SceneRect{X: 40, Y: 40, W: 240, H: 160}),
		Region: SceneRect{X: 20, Y: 20, W: 280, H: 200}, Shapes: layerShapes, Feather: 8, Prompt: "a stone well", Options: map[string]any{"engine": "dream"}}}
	if _, err := UpdateScene(q, s.ID, &in); err != nil {
		t.Fatal(err)
	}
	genShapes := json.RawMessage(`[{"t":"rect","x":100,"y":80,"w":100,"h":80}]`)
	gm := &SceneGenMask{SceneMask: SceneMask{Shapes: genShapes, Feather: 8}, Mask: maskDataURL(t, 320, 240, SceneRect{X: 100, Y: 80, W: 100, H: 80}),
		Region: SceneRect{X: 80, Y: 60, W: 140, H: 120}}
	s2, err := GenerateLayerMask(context.Background(), q, s.ID, "e", gm)
	if err != nil {
		t.Fatal(err)
	}
	j := s2.Layers[0].Job
	if j.Region != gm.Region || j.Mask == nil || !jsonSame(j.Mask.Shapes, genShapes) || j.Mask.Feather != 8 {
		t.Fatalf("job made with region %+v mask %+v, want the sent mask", j.Region, j.Mask)
	}
	job, _, _ := q.JobSnapshot(j.Chain[0], 0)
	for _, a := range job.Arguments {
		if strings.HasPrefix(a, "--mask=") {
			if b := readPNG(t, strings.TrimPrefix(a, "--mask=")).Bounds(); b.Dx() != gm.Region.W || b.Dy() != gm.Region.H {
				t.Fatalf("engine mask %v, want the sent region's", b)
			}
		}
	}

	// It finishes: the version keeps the mask it was made with.
	out := strings.TrimSuffix(j.Input, filepath.Ext(j.Input)) + "_dream.png"
	solidPNG(t, out, gm.Region.W, gm.Region.H, color.RGBA{200, 0, 0, 255})
	claim(t, q, j.ID)
	_ = q.RegisterOutputFile(j.ID, out)
	_ = q.CompleteJob(j.ID)
	s3, err := GetScene(q, s.ID)
	if err != nil {
		t.Fatal(err)
	}
	v := s3.Layers[0].Versions
	if len(v) != 1 || v[0].Mask == nil || !jsonSame(v[0].Mask.Shapes, genShapes) {
		t.Fatalf("version %+v, want the generation's mask", v)
	}

	// The editor reshapes the shown version: kept by path; an unknown path is ignored.
	edited := json.RawMessage(`[{"t":"rect","x":90,"y":70,"w":120,"h":100}]`)
	in2 := *s3
	l := *s3.Layers[0]
	l.Versions = []SceneVersion{{Path: v[0].Path, Mask: &SceneMask{Shapes: edited, Feather: 4}}, {Path: "nope.png", Mask: &SceneMask{}}}
	in2.Layers = []*SceneLayer{&l}
	s4, err := UpdateScene(q, s.ID, &in2)
	if err != nil {
		t.Fatal(err)
	}
	if v := s4.Layers[0].Versions; len(v) != 1 || !jsonSame(v[0].Mask.Shapes, edited) || v[0].Mask.Feather != 4 || v[0].Region != gm.Region {
		t.Fatalf("after the edit %+v", v)
	}

	// Without a sent mask, a generation snapshots the layer's own.
	s5, err := GenerateLayer(context.Background(), q, s.ID, "e")
	if err != nil {
		t.Fatal(err)
	}
	if m := s5.Layers[0].Job.Mask; m == nil || !jsonSame(m.Shapes, layerShapes) {
		t.Fatalf("plain generation snapshot %+v", m)
	}
}

// jsonSame: the same JSON value (a scene file round trip re-indents raw JSON).
func jsonSame(a, b json.RawMessage) bool {
	var x, y any
	if json.Unmarshal(a, &x) != nil || json.Unmarshal(b, &y) != nil {
		return false
	}
	xa, _ := json.Marshal(x)
	ya, _ := json.Marshal(y)
	return string(xa) == string(ya)
}

// A sketch finish names its target: the prompt says where the sketch is and
// that only it changes, and with a target image (Retouch) points at it; with
// none (Kromoss) that sentence goes.
func TestSceneSketchTargetPrompt(t *testing.T) {
	refs := []sceneRef{{Kind: "image", Path: "t.png", Target: true}}
	got := sceneFinishPrompt(map[string]any{"finish": FinishSketch}, "a scared man running", refs, " at the bottom left", "")
	for _, want := range []string{"the rough painted sketch at the bottom left stands in for a scared man running", "<image2> shows the sketch alone on black",
		"even one that looks like a scared man running"} {
		if !strings.Contains(got, want) {
			t.Errorf("lacks %q:\n%s", want, got)
		}
	}
	got = sceneFinishPrompt(map[string]any{"finish": FinishSketch}, "a sun", nil, "", "")
	if strings.Contains(got, "{") || strings.Contains(got, "alone on black") || !strings.Contains(got, "the rough painted sketch stands in for a sun") {
		t.Errorf("no target:\n%s", got)
	}
	region := SceneRect{X: 100, Y: 100, W: 900, H: 600}
	for _, c := range []struct {
		paint SceneRect
		want  string
	}{
		{SceneRect{X: 120, Y: 550, W: 100, H: 100}, " at the bottom left"},
		{SceneRect{X: 500, Y: 350, W: 100, H: 100}, " in the middle"},
		{SceneRect{X: 850, Y: 360, W: 100, H: 80}, " on the right"},
		{SceneRect{X: 500, Y: 110, W: 100, H: 60}, " at the top"},
		{SceneRect{X: 100, Y: 100, W: 800, H: 500}, ""}, // fills most of it
		{SceneRect{X: 1500, Y: 900, W: 50, H: 50}, ""},  // outside
	} {
		if got := sketchWhere(c.paint, region); got != c.want {
			t.Errorf("sketchWhere(%+v) = %q, want %q", c.paint, got, c.want)
		}
	}
}

// End to end: a Retouch sketch finish is given the paint alone on black,
// lined up with its crop, as its last reference.
func TestSceneSketchTargetReference(t *testing.T) {
	ffmpegTest(t)
	withSceneRoot(t)
	q := jobqueue.NewQueue()
	src := filepath.Join(t.TempDir(), "photo.png")
	solidPNG(t, src, 320, 240, color.RGBA{40, 80, 120, 255})
	s, err := CreateScene(context.Background(), src, 0)
	if err != nil {
		t.Fatal(err)
	}
	in := *s
	sketch := SceneRect{X: 40, Y: 150, W: 40, H: 60}
	region := SceneRect{X: 0, Y: 0, W: 320, H: 240}
	in.Layers = []*SceneLayer{
		{ID: "p", Kind: SceneKindPaint, Mask: maskDataURL(t, 320, 240, sketch), Shapes: json.RawMessage(`[{"t":"stroke","pts":[60,180],"r":20}]`)},
		{ID: "f", Kind: SceneKindStill, Mask: maskDataURL(t, 320, 240, sketch), Region: region, Prompt: "a man",
			Options: map[string]any{"finish": FinishSketch, "paint": "p"}},
	}
	if _, err := UpdateScene(q, s.ID, &in); err != nil {
		t.Fatal(err)
	}
	s2, err := GenerateLayer(context.Background(), q, s.ID, "f")
	if err != nil {
		t.Fatal(err)
	}
	job, _, _ := q.JobSnapshot(s2.Layers[1].Job.Chain[0], 0)
	var prompt, target string
	for _, a := range job.Arguments {
		if v, ok := strings.CutPrefix(a, "--prompt="); ok {
			prompt = v
		}
		if v, ok := strings.CutPrefix(a, "--refs="); ok {
			target = strings.Split(v, "\n")[len(strings.Split(v, "\n"))-1]
		}
	}
	if !strings.HasSuffix(target, "-target.png") || !strings.Contains(prompt, "sketch at the bottom left") || !strings.Contains(prompt, "<image2> shows the sketch alone") {
		t.Fatalf("target %q, prompt:\n%s", target, prompt)
	}
	img := readPNG(t, target)
	if b := img.Bounds(); b.Dx() != region.W || b.Dy() != region.H {
		t.Fatalf("target %v, want the crop's size", b)
	}
	r, g, b, _ := img.At(60, 180).RGBA()    // on the sketch
	r2, g2, b2, _ := img.At(250, 50).RGBA() // elsewhere: black
	if (r+g+b)>>8 < 200 || (r2+g2+b2)>>8 > 10 {
		t.Fatalf("target pixels: sketch %d, elsewhere %d", (r+g+b)>>8, (r2+g2+b2)>>8)
	}
}

// The editor shows the grounding it expects the server to send: its copy in
// scene-api.ts (FINISH_GROUNDING) must match sceneFinishGrounding word for word.
func TestSceneFinishGroundingMatchesEditor(t *testing.T) {
	b, err := os.ReadFile(filepath.Join("..", "..", "src", "renderer", "components", "scene", "scene-api.ts"))
	if err != nil {
		t.Skip("editor source not here:", err)
	}
	src := string(b)
	start := strings.Index(src, "export const FINISH_GROUNDING")
	if start < 0 {
		t.Fatal("FINISH_GROUNDING not found in scene-api.ts")
	}
	block := src[start : start+strings.Index(src[start:], "\n};")]
	key := regexp.MustCompile(`(?m)^  (\w+):\s*$`)
	lit := regexp.MustCompile(`'([^']*)'|"([^"]*)"`)
	locs := key.FindAllStringSubmatchIndex(block, -1)
	got := map[string]string{}
	for i, l := range locs {
		end := len(block)
		if i+1 < len(locs) {
			end = locs[i+1][0]
		}
		var sb strings.Builder
		for _, m := range lit.FindAllStringSubmatch(block[l[1]:end], -1) {
			sb.WriteString(m[1] + m[2])
		}
		got[block[l[2]:l[3]]] = sb.String()
	}
	if len(got) != len(sceneFinishGrounding) {
		t.Fatalf("editor has modes %v, server %d", got, len(sceneFinishGrounding))
	}
	for mode, want := range sceneFinishGrounding {
		if got[mode] != want {
			t.Errorf("%s differs:\neditor: %s\nserver: %s", mode, got[mode], want)
		}
	}
}

// A paint layer of only mannequin poses goes to the engine as the skeleton on
// black; anything else painted on it keeps the paint-over-picture reference.
func TestPaintIsPose(t *testing.T) {
	for _, c := range []struct {
		shapes string
		want   bool
	}{
		{`[{"t":"pose","kp":[],"w":4}]`, true},
		{`[{"t":"pose","kp":[],"w":4},{"t":"stroke","pts":[1,2],"r":3,"off":true}]`, true},
		{`[{"t":"pose","kp":[],"w":4},{"t":"stroke","pts":[1,2],"r":3}]`, false},
		{`[]`, false},
		{``, false},
	} {
		if got := paintIsPose(&SceneLayer{Shapes: json.RawMessage(c.shapes)}); got != c.want {
			t.Errorf("%s: %v, want %v", c.shapes, got, c.want)
		}
	}
}

// Repose speaks plainly ("make the person do the exact same pose as shown in
// <image2>"): naming the guide a skeleton or keypoint map, or asking to
// "repose", made Qwen Image 2.1 paint the skeleton onto the person instead of
// moving them (measured on the same crop, guide and seed).
func TestSceneReposeGroundingAvoidsDrawingWords(t *testing.T) {
	g := strings.ToLower(sceneFinishGrounding[FinishRepose])
	for _, w := range []string{"skeleton", "keypoint", "openpose", "repose", "lines", "colour"} {
		if strings.Contains(g, w) {
			t.Errorf("the repose grounding says %q:\n%s", w, g)
		}
	}
	if !strings.Contains(g, "do the exact same pose as shown in {guide}") {
		t.Errorf("repose grounding:\n%s", g)
	}
}
