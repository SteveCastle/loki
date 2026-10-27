package tasks

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/jobqueue"
)

// writeMaskPNG writes a gray mask, white inside r.
func writeMaskPNG(t *testing.T, path string, w, h int, r image.Rectangle) {
	t.Helper()
	m := image.NewGray(image.Rect(0, 0, w, h))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			m.SetGray(x, y, color.Gray{255})
		}
	}
	if err := writePNG(path, m); err != nil {
		t.Fatal(err)
	}
}

// Rough paint covering most of one object (with gaps) and spilling a little
// onto its neighbour colours the whole object and leaves the neighbour alone.
func TestRecolourBaseWholeObject(t *testing.T) {
	dir := t.TempDir()
	const w, h = 200, 100
	a, b := image.Rect(0, 0, 100, 100), image.Rect(100, 0, 200, 100)
	pic := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := uint8(140 + (x+y)%20) // a little texture
			if x >= 100 {
				v = 70
			}
			pic.SetNRGBA(x, y, color.NRGBA{v, v, v, 255})
		}
	}
	crop := filepath.Join(dir, "crop.png")
	if err := writePNG(crop, pic); err != nil {
		t.Fatal(err)
	}
	// Red dabs over A's middle (its corners left bare), one spilling 6 px into B.
	paint := image.NewNRGBA(image.Rect(0, 0, w, h))
	red := color.NRGBA{200, 30, 30, 255}
	for _, r := range []image.Rectangle{image.Rect(15, 15, 60, 85), image.Rect(55, 25, 106, 75)} {
		for y := r.Min.Y; y < r.Max.Y; y++ {
			for x := r.Min.X; x < r.Max.X; x++ {
				paint.SetNRGBA(x, y, red)
			}
		}
	}
	paintPath := filepath.Join(dir, "paint.png")
	if err := writePNG(paintPath, paint); err != nil {
		t.Fatal(err)
	}
	// SAM, faked: clicks read as A (part), A+B (whole); everything finds A and B.
	seg := func(req map[string]any) (map[string]any, error) {
		out := req["out"].(string)
		_ = os.MkdirAll(out, 0o755)
		obj := func(name string, r image.Rectangle) map[string]any {
			p := filepath.Join(out, name+".png")
			writeMaskPNG(t, p, w, h, r)
			return map[string]any{"mask": p}
		}
		if req["everything"] == true {
			return map[string]any{"objects": []any{obj("a", a), obj("b", b)}}, nil
		}
		o := obj("part", a)
		o["candidates"] = []any{obj("whole", image.Rect(0, 0, w, h))}
		return map[string]any{"objects": []any{o}}, nil
	}
	out := filepath.Join(dir, "crop_edit.png")
	st, err := recolourBase(crop, paintPath, out, seg, func(string) {})
	if err != nil {
		t.Fatal(err)
	}
	if st.Colours != 1 || st.Patches != 1 {
		t.Fatalf("stats %+v", st)
	}
	res, err := loadPNG(out)
	if err != nil {
		t.Fatal(err)
	}
	reddish := func(x, y int) bool {
		r, g, bb, _ := res.At(x, y).RGBA()
		return r>>8 > g>>8+40 && r>>8 > bb>>8+40
	}
	// A's bare corner took the colour; B (spilled on, unpainted far side) kept its grey.
	if !reddish(4, 4) || !reddish(95, 95) || !reddish(30, 50) {
		t.Fatalf("A not recoloured edge to edge: (4,4)=%v (95,95)=%v", res.At(4, 4), res.At(95, 95))
	}
	for _, p := range [][2]int{{103, 50}, {150, 50}, {195, 5}} {
		if c := color.NRGBAModel.Convert(res.At(p[0], p[1])).(color.NRGBA); c.R != 70 || c.G != 70 || c.B != 70 {
			t.Fatalf("B changed at %v: %v", p, c)
		}
	}
	// The shading stays: A's texture still varies under the new colour.
	r1, _, _, _ := res.At(10, 10).RGBA()
	r2, _, _, _ := res.At(19, 10).RGBA()
	if r1 == r2 {
		t.Fatal("the base coat flattened the texture")
	}
}

func paintDataURL(t *testing.T, w, h int, r image.Rectangle, c color.NRGBA) string {
	t.Helper()
	m := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			m.SetNRGBA(x, y, c)
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, m); err != nil {
		t.Fatal(err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

// A Recolour layer queues its base coat alone, or with words the coat and then
// Retouch finishing it (shown the crop, never the paint).
func TestSceneRecolourChain(t *testing.T) {
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
	in.Layers = []*SceneLayer{
		{ID: "p", Kind: SceneKindPaint, Mask: paintDataURL(t, 320, 240, image.Rect(100, 80, 200, 160), color.NRGBA{200, 30, 30, 255}),
			Region: SceneRect{X: 100, Y: 80, W: 100, H: 80}, Options: map[string]any{"guide": true}},
		{ID: "c", Kind: SceneKindStill, Mask: maskDataURL(t, 320, 240, SceneRect{X: 90, Y: 70, W: 120, H: 100}), Region: region,
			Options: map[string]any{"finish": FinishColour, "paint": "p", "refs": []any{map[string]any{"id": "r1", "layer": "p", "kind": "image"}}}},
	}
	if _, err := UpdateScene(q, s.ID, &in); err != nil {
		t.Fatal(err)
	}
	chain := func() []jobqueue.Job {
		s2, err := GenerateLayer(context.Background(), q, s.ID, "c")
		if err != nil {
			t.Fatal(err)
		}
		l := s2.Layers[1]
		j := l.Job
		if len(l.Queued) > 0 {
			j = l.Queued[len(l.Queued)-1]
		}
		var out []jobqueue.Job
		for _, id := range j.Chain {
			job, _, _ := q.JobSnapshot(id, 0)
			out = append(out, job)
		}
		return out
	}
	coat := chain()
	if len(coat) != 1 || coat[0].Command != "scene-recolour" {
		t.Fatalf("no words: %d steps, first %q", len(coat), coat[0].Command)
	}
	if paint := argOf(coat[0].Arguments, "paint"); paint == "" || !fileExists(paint) {
		t.Fatalf("paint crop %q missing", paint)
	}

	in2, err := GetScene(q, s.ID)
	if err != nil {
		t.Fatal(err)
	}
	in2.Layers[1].Prompt = "warm evening light"
	if _, err := UpdateScene(q, s.ID, in2); err != nil {
		t.Fatal(err)
	}
	fin := chain()
	if len(fin) != 2 || fin[0].Command != "scene-recolour" || fin[1].Command != "retouch" {
		t.Fatalf("with words: %v", fin)
	}
	base := argOf(fin[0].Arguments, "out")
	if base == "" || fin[1].Input != base {
		t.Fatalf("retouch input %q, coat out %q", fin[1].Input, base)
	}
	refs := argOf(fin[1].Arguments, "refs")
	if strings.Contains(refs, "-ref-") || refs != fin[0].Input {
		t.Fatalf("retouch refs %q: want only the crop %q, never the paint", refs, fin[0].Input)
	}
	if p := argOf(fin[1].Arguments, "prompt"); !strings.Contains(p, "already laid in: warm evening light.") || !strings.Contains(p, "<image2> is the same picture before") {
		t.Fatalf("prompt %q", p)
	}
}

func argOf(args []string, k string) string {
	for _, a := range args {
		if strings.HasPrefix(a, "--"+k+"=") {
			return strings.TrimPrefix(a, "--"+k+"=")
		}
	}
	return ""
}

func TestColourName(t *testing.T) {
	for hex, want := range map[string]string{
		"#2e6b3f": "green", "#3a7bd5": "blue", "#c46a1d": "orange", "#1c1512": "black", "#f4f3ee": "white",
		"#2b3a6e": "navy blue", "#c09a6b": "tan", "#e9c2a6": "peach", "#7a1f2b": "maroon", "#5a6270": "blue-grey",
	} {
		rgb, _ := parseHexColour(hex)
		if got := colourName(rgb); got != want+" ("+hex+")" {
			t.Errorf("%s: %q, want %q", hex, got, want)
		}
	}
}

func parseHexColour(s string) ([3]float64, bool) {
	var r, g, b int
	if _, err := fmt.Sscanf(s, "#%02x%02x%02x", &r, &g, &b); err != nil {
		return [3]float64{}, false
	}
	return [3]float64{float64(r) / 255, float64(g) / 255, float64(b) / 255}, true
}
