package tasks

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"sync"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/jobqueue"
)

// stitchTestScene is a 96×64 scene: a striped base and one still layer whose
// generation is the base drifted 3 px and brightened, through a soft disc.
func stitchTestScene(t *testing.T) *Scene {
	t.Helper()
	root := t.TempDir()
	sceneRootOverride = root
	t.Cleanup(func() { sceneRootOverride = "" })
	id := "aaaaaaaa-0000-0000-0000-000000000001"
	dir := filepath.Join(root, id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	w, h := 96, 64
	stripes := func(dx int, lift uint8) *image.NRGBA {
		img := image.NewNRGBA(image.Rect(0, 0, w, h))
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				v := uint8(60)
				if ((x+dx)/5+y/7)%2 == 0 {
					v = 190
				}
				img.SetNRGBA(x, y, color.NRGBA{v + lift/2, v, v + lift, 255})
			}
		}
		return img
	}
	base := filepath.Join(dir, "base.png")
	gen := filepath.Join(dir, "still-l1-1_edit.png")
	mustPNG(t, base, stripes(0, 0))
	mustPNG(t, gen, stripes(3, 30))
	mask := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			dx, dy := float64(x-48), float64(y-32)
			d := dx*dx + dy*dy
			a := 0.0
			if d < 18*18 {
				a = 1
			} else if d < 26*26 {
				a = 1 - (math.Sqrt(d)-18)/8
			}
			mask.SetNRGBA(x, y, color.NRGBA{255, 255, 255, uint8(a * 255)})
		}
	}
	return &Scene{
		ID: id, Base: base, Width: w, Height: h,
		Layers: []*SceneLayer{{
			ID: "l1", Name: "Edit 1", Kind: SceneKindStill, Feather: 8, Mask: dataURLPNG(t, mask),
			Region:   SceneRect{X: 0, Y: 0, W: w, H: h},
			Versions: []SceneVersion{{Path: gen, Region: SceneRect{X: 0, Y: 0, W: w, H: h}}}, Active: 0,
		}},
	}
}

func mustPNG(t *testing.T, path string, img image.Image) {
	t.Helper()
	if err := writeStitchPNG(path, img); err != nil {
		t.Fatal(err)
	}
}

func dataURLPNG(t *testing.T, img image.Image) string {
	t.Helper()
	var b strings.Builder
	enc := base64.NewEncoder(base64.StdEncoding, &b)
	if err := png.Encode(enc, img); err != nil {
		t.Fatal(err)
	}
	enc.Close()
	return "data:image/png;base64," + b.String()
}

func TestSceneStitchKeysChainBottomUp(t *testing.T) {
	s := stitchTestScene(t)
	k1 := sceneStitchKeys(s.Layers)["l1"]
	if k1 == "" {
		t.Fatal("the still layer should be stitchable")
	}
	// A layer above with the same inputs but a different stack below gets a different key.
	top := *s.Layers[0]
	top.ID = "l2"
	keys := sceneStitchKeys([]*SceneLayer{s.Layers[0], &top})
	alone := sceneStitchKeys([]*SceneLayer{&top})
	if keys["l2"] == alone["l2"] {
		t.Error("a layer's key must depend on the layers below it")
	}
	// Opting out drops the key; hiding does too.
	s.Layers[0].Options = map[string]any{"stitch": false}
	if _, ok := sceneStitchKeys(s.Layers)["l1"]; ok {
		t.Error("options.stitch=false should opt out")
	}
	s.Layers[0].Options = nil
	s.Layers[0].Feather = 9
	if sceneStitchKeys(s.Layers)["l1"] == k1 {
		t.Error("a mask change (feather) must change the key")
	}
}

func TestEnsureStitchesFeedsTheGraph(t *testing.T) {
	if deps.BundledOrEmpty("ffmpeg") == "" {
		t.Skip("bundled ffmpeg not available")
	}
	s := stitchTestScene(t)
	dir, _ := sceneDir(s.ID)
	if err := ensureStitches(context.Background(), s, len(s.Layers)); err != nil {
		t.Fatal(err)
	}
	key := sceneStitchKeys(s.Layers)["l1"]
	st := readStitch(dir, "l1", key)
	if st == nil {
		t.Fatal("no stitch written")
	}
	if st.Report.Skipped != "" {
		t.Fatalf("skipped: %s", st.Report.Skipped)
	}
	gl, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(gl) != 1 || !gl[0].OwnAlpha || gl[0].Media != st.Path {
		t.Fatalf("graph layer %+v, want the stitch with its own alpha", gl)
	}
	// The graph composites it (one input for the layer, no mask).
	args := strings.Join(sceneGraphArgs(s.Base, s.Width, s.Height, gl, false, 0, SceneRect{}), " ")
	if strings.Contains(args, "alphamerge") || strings.Count(args, "-i ") != 2 {
		t.Fatalf("own-alpha layer graph: %s", args)
	}
	out := filepath.Join(dir, "flat.png")
	if err := runSceneFFmpeg(context.Background(), append(sceneGraphArgs(s.Base, s.Width, s.Height, gl, false, 0, SceneRect{}), out)); err != nil {
		t.Fatal(err)
	}
	// Reading the scene decorates the version and reports nothing pending.
	decorateStitches(s)
	if s.Stitching || s.Layers[0].active().Stitch == nil {
		t.Fatalf("decorate: stitching=%v stitch=%v", s.Stitching, s.Layers[0].active().Stitch)
	}
}

// TestStitchRealScene stitches a real scene and writes before/after pictures:
// LOKI_STITCH_SCENE=<scene folder> LOKI_STITCH_OUT=<folder> go test ./tasks -run StitchRealScene -v
func TestStitchRealScene(t *testing.T) {
	src := os.Getenv("LOKI_STITCH_SCENE")
	outDir := os.Getenv("LOKI_STITCH_OUT")
	if src == "" || outDir == "" {
		t.Skip("set LOKI_STITCH_SCENE and LOKI_STITCH_OUT")
	}
	root := t.TempDir()
	sceneRootOverride = root
	t.Cleanup(func() { sceneRootOverride = "" })
	id := filepath.Base(src)
	dir := filepath.Join(root, id)
	if err := copyTree(src, dir); err != nil {
		t.Fatal(err)
	}
	// scene.json names files in the original folder: point them here.
	b, err := os.ReadFile(filepath.Join(dir, "scene.json"))
	if err != nil {
		t.Fatal(err)
	}
	esc := func(p string) string { j, _ := json.Marshal(p); return strings.Trim(string(j), `"`) }
	b = []byte(strings.ReplaceAll(string(b), esc(src), esc(dir)))
	if err := os.WriteFile(filepath.Join(dir, "scene.json"), b, 0644); err != nil {
		t.Fatal(err)
	}
	s, err := loadSceneLocked(id)
	if err != nil {
		t.Fatal(err)
	}
	// Stitch every still, whatever the scene says.
	for _, l := range s.Layers {
		delete(l.Options, "stitch")
	}
	os.MkdirAll(outDir, 0755)
	ctx := context.Background()
	flat := func(name string) {
		gl, err := sceneGraphLayers(dir, s.Width, s.Height, s.Layers, true)
		if err != nil {
			t.Fatal(err)
		}
		if err := runSceneFFmpeg(ctx, append(sceneGraphArgs(s.Base, s.Width, s.Height, gl, false, 0, SceneRect{}), filepath.Join(outDir, name))); err != nil {
			t.Fatal(err)
		}
	}
	flat("before.png")
	t0 := time.Now()
	if err := ensureStitches(ctx, s, len(s.Layers)); err != nil {
		t.Fatal(err)
	}
	t.Logf("stitched in %v", time.Since(t0))
	flat("after.png")
	keys := sceneStitchKeys(s.Layers)
	for _, l := range s.Layers {
		key, ok := keys[l.ID]
		if !ok {
			continue
		}
		st := readStitch(dir, l.ID, key)
		if st == nil {
			t.Errorf("%s: no stitch", l.Name)
			continue
		}
		r, _ := json.Marshal(st.Report)
		t.Logf("%-8s region %v: %s", l.Name, l.active().Region, r)
		for _, f := range []string{st.Path, st.Seam, st.Heal} {
			if f != "" {
				copyFile(f, filepath.Join(outDir, filepath.Base(f)))
			}
		}
	}
}

func copyTree(src, dst string) error {
	return filepath.Walk(src, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, 0755)
		}
		if strings.HasPrefix(info.Name(), "stitch-") {
			return nil
		}
		return copyFile(p, target)
	})
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func TestHealMergeChangesOnlyTheBand(t *testing.T) {
	dir := t.TempDir()
	w, h := 40, 30
	init := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range init.Pix {
		init.Pix[i] = 100
	}
	for i := 3; i < len(init.Pix); i += 4 {
		init.Pix[i] = 255
	}
	// The engine's output: a different size, and different everywhere.
	healed := image.NewNRGBA(image.Rect(0, 0, 80, 60))
	for i := range healed.Pix {
		healed.Pix[i] = 200
	}
	mask := image.NewGray(image.Rect(0, 0, w, h))
	for x := 10; x < 20; x++ {
		mask.Pix[15*w+x] = 255
	}
	ip, mp, hp := filepath.Join(dir, "c.png"), filepath.Join(dir, "c-mask.png"), filepath.Join(dir, "c_dream.png")
	mustPNG(t, ip, init)
	mustPNG(t, mp, mask)
	mustPNG(t, hp, healed)
	q := jobqueue.NewQueue()
	id, err := q.AddJob("", "scene-heal-merge", []string{"--init=" + ip, "--mask=" + mp, "--out=" + filepath.Join(dir, "c_healed.png")}, hp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := sceneHealMergeTask(q.GetJob(id), q, &sync.Mutex{}); err != nil {
		t.Fatal(err)
	}
	out, err := loadNRGBA(filepath.Join(dir, "c_healed.png"))
	if err != nil {
		t.Fatal(err)
	}
	if out.Bounds().Dx() != w || out.Bounds().Dy() != h {
		t.Fatalf("size %v, want the init's", out.Bounds())
	}
	if got := out.NRGBAAt(15, 15).R; got < 190 {
		t.Errorf("in the band: %d, want the healed pixel", got)
	}
	if got := out.NRGBAAt(15, 5).R; got != 100 {
		t.Errorf("outside the band: %d, want the picture's own 100", got)
	}
	if fileExists(filepath.Join(dir, "c_healed.png")) && sceneJobOutput(SceneKindStill, ip) != filepath.Join(dir, "c_healed.png") {
		t.Errorf("the scene should take the merged picture, not the engine's raw band: %s", sceneJobOutput(SceneKindStill, ip))
	}
}
