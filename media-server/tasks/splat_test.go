package tasks

import (
	"encoding/binary"
	"image"
	"image/color"
	"image/jpeg"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSplatParamsDefaultsAndClamps(t *testing.T) {
	p := splatParamsFromOptions(map[string]any{})
	if p.Quality != "standard" || p.Frames != 200 || p.MaxRes != 1600 || p.SHDegree != 3 || p.Steps != 0 {
		t.Errorf("defaults = %+v", p)
	}
	p = splatParamsFromOptions(map[string]any{
		"quality": "HIGH", "frames": 5.0, "maxres": 99999.0, "shdegree": 7.0, "steps": 10.0,
	})
	if p.Quality != "high" || p.Frames != 30 || p.MaxRes != 4096 || p.SHDegree != 3 || p.Steps != 100 {
		t.Errorf("clamped = %+v", p)
	}
	if p := splatParamsFromOptions(map[string]any{"quality": "bogus", "shdegree": 0.0}); p.Quality != "standard" || p.SHDegree != 0 {
		t.Errorf("bogus quality / sh 0 = %+v", p)
	}
}

func TestClassifySplatInputs(t *testing.T) {
	imgs, vids, skipped := classifySplatInputs([]string{
		`C:\c\a.jpg`, `C:\c\b.PNG`, `C:\c\walk.mp4`, `C:\c\notes.txt`, `C:\c\a.jpg`,
	})
	if len(imgs) != 2 || len(vids) != 1 || strings.Join(skipped, "|") != `C:\c\notes.txt` {
		t.Errorf("imgs=%q vids=%q skipped=%q", imgs, vids, skipped)
	}
}

func writeJPEG(t *testing.T, path string, sharp bool) {
	t.Helper()
	img := image.NewGray(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			v := uint8(128)
			if sharp && (x/4+y/4)%2 == 0 {
				v = 255
			} else if sharp {
				v = 0
			}
			img.SetGray(x, y, color.Gray{Y: v})
		}
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if err := jpeg.Encode(f, img, &jpeg.Options{Quality: 95}); err != nil {
		t.Fatal(err)
	}
}

func TestSharpnessPrefersDetail(t *testing.T) {
	dir := t.TempDir()
	flat, checker := filepath.Join(dir, "flat.jpg"), filepath.Join(dir, "checker.jpg")
	writeJPEG(t, flat, false)
	writeJPEG(t, checker, true)
	if a, b := sharpness(flat), sharpness(checker); !(b > a) {
		t.Errorf("sharpness flat=%v checker=%v; want checker > flat", a, b)
	}
}

func TestBestSparseModelPicksMostImages(t *testing.T) {
	sparse := t.TempDir()
	for name, n := range map[string]uint64{"0": 12, "1": 80, "2": 3} {
		d := filepath.Join(sparse, name)
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
		f, _ := os.Create(filepath.Join(d, "images.bin"))
		_ = binary.Write(f, binary.LittleEndian, n)
		f.Close()
	}
	best, n, pieces := bestSparseModel(sparse)
	if filepath.Base(best) != "1" || n != 80 || pieces != 3 {
		t.Errorf("best = %s (%d of %d pieces)", best, n, pieces)
	}
	if _, n, _ := bestSparseModel(filepath.Join(sparse, "missing")); n != 0 {
		t.Errorf("missing dir = %d", n)
	}
}

func TestScaleFilterNeverEnlarges(t *testing.T) {
	if f := scaleFilter(1600); !strings.Contains(f, "min(1600,iw)") || !strings.Contains(f, "min(1600,ih)") {
		t.Errorf("scaleFilter = %s", f)
	}
}
