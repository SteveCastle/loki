package tasks

import (
	"context"
	"image/color"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

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
