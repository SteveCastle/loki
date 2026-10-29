package stitch

import (
	"image"
	"math"
	"math/rand"
	"testing"
)

func TestSignedDistanceMatchesBruteForce(t *testing.T) {
	w, h := 23, 17
	rng := rand.New(rand.NewSource(1))
	inside := make([]bool, w*h)
	for i := range inside {
		inside[i] = rng.Float64() < 0.2
	}
	d := signedDistance(inside, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			best := math.Inf(1)
			for yy := 0; yy < h; yy++ {
				for xx := 0; xx < w; xx++ {
					if inside[yy*w+xx] != inside[y*w+x] {
						best = math.Min(best, math.Hypot(float64(x-xx), float64(y-yy)))
					}
				}
			}
			want := best - 0.5
			if inside[y*w+x] {
				want = -want
			}
			if got := float64(d.p[y*w+x]); math.Abs(got-want) > 1e-4 {
				t.Fatalf("(%d,%d): %v, want %v", x, y, got, want)
			}
		}
	}
}

func TestMaxflowMinCut(t *testing.T) {
	// s=0, t=3: s→1 (3), s→2 (2), 1→2 (1), 1→3 (2), 2→3 (3): max flow 5.
	g := newMaxflow(4)
	g.edge(0, 1, 3, 0)
	g.edge(0, 2, 2, 0)
	g.edge(1, 2, 1, 0)
	g.edge(1, 3, 2, 0)
	g.edge(2, 3, 3, 0)
	if f := g.run(0, 3); f != 5 {
		t.Fatalf("flow %d, want 5", f)
	}
	side := g.sourceSide(0)
	if !side[0] || side[3] {
		t.Fatalf("cut sides wrong: %v", side)
	}
}

// texture is a deterministic picture with structure at several scales and
// straight edges (what drift shows on).
func texture(w, h int, seed int64, dx, dy float64) rgb {
	rng := rand.New(rand.NewSource(seed))
	type wave struct{ fx, fy, ph, amp float64 }
	var ws [3][]wave
	for k := 0; k < 3; k++ {
		for i := 0; i < 6; i++ {
			ws[k] = append(ws[k], wave{rng.Float64()*0.3 - 0.15, rng.Float64()*0.3 - 0.15, rng.Float64() * 6, 0.05 + rng.Float64()*0.08})
		}
	}
	c := newRGB(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			X, Y := float64(x)+dx, float64(y)+dy
			for k := 0; k < 3; k++ {
				v := 0.45
				for _, wv := range ws[k] {
					v += wv.amp * math.Sin(wv.fx*X+wv.fy*Y+wv.ph)
				}
				// a few hard stripes
				if int(math.Floor((X+2*Y)/23))%3 == 0 {
					v += 0.15
				}
				c[k].p[y*w+x] = float32(math.Max(0.02, math.Min(0.98, v)))
			}
		}
	}
	return c
}

func circleMask(w, h int, cx, cy, r, feather float64) []float32 {
	m := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			d := math.Hypot(float64(x)-cx, float64(y)-cy) - r
			a := 0.5 - d/feather
			m[y*w+x] = float32(math.Max(0, math.Min(1, a)))
		}
	}
	return m
}

// over composites the stitched image over O through its alpha.
func over(res *Result, O rgb) rgb {
	w, h := O[0].w, O[0].h
	out := newRGB(w, h)
	for i := 0; i < w*h; i++ {
		a := float32(res.Image.Pix[i*4+3]) / 255
		for k := 0; k < 3; k++ {
			out[k].p[i] = float32(res.Image.Pix[i*4+k])/255*a + O[k].p[i]*(1-a)
		}
	}
	return out
}

func TestIdenticalGenerationChangesNothing(t *testing.T) {
	w, h := 160, 128
	O := texture(w, h, 3, 0, 0)
	res := stitch(O, O.clone(), circleMask(w, h, 80, 64, 35, 15), Options{Feather: 15, Seed: 1})
	got := over(res, O)
	for k := 0; k < 3; k++ {
		for i := range got[k].p {
			if d := math.Abs(float64(got[k].p[i] - O[k].p[i])); d > 2.5/255 {
				t.Fatalf("pixel %d channel %d moved by %.1f/255", i, k, d*255)
			}
		}
	}
}

func TestDriftAndToneAreCorrectedAtTheSeam(t *testing.T) {
	w, h := 200, 160
	O := texture(w, h, 7, 0, 0)
	// The generation: the same scene drifted (2.6, -1.4) px and brightened,
	// with something new inside the mask.
	G := texture(w, h, 7, 2.6, -1.4)
	inner := texture(w, h, 99, 0, 0)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			for k := 0; k < 3; k++ {
				G[k].p[i] += 0.06
				if math.Hypot(float64(x)-100, float64(y)-80) < 28 {
					G[k].p[i] = inner[k].p[i]
				}
			}
		}
	}
	mask := circleMask(w, h, 100, 80, 40, 15)
	res := stitch(O, G, mask, Options{Feather: 15, Seed: 1})
	r := res.Report
	t.Logf("report %+v", r)
	if r.Skipped != "" {
		t.Fatalf("skipped: %s", r.Skipped)
	}
	if r.Drift < 1.2 {
		t.Errorf("drift corrected %.2f px, want the ~3 px it drifted to show", r.Drift)
	}
	if r.Tone < 0.5*0.06*255*math.Sqrt(3) {
		t.Errorf("tone corrected %.1f, want about %.1f", r.Tone, 0.06*255*math.Sqrt(3))
	}
	// Along the seam the blend adds little gradient neither picture has.
	if r.Residual > 1.5 {
		t.Errorf("seam residual %.2f/255", r.Residual)
	}
	// Composited, the new content survives in the middle and O survives
	// well outside.
	got := over(res, O)
	ci := 80*w + 100
	if d := math.Abs(float64(got[0].p[ci] - inner[0].p[ci])); d > 0.08 {
		t.Errorf("centre lost the generation: off by %.3f", d)
	}
	oi := 5*w + 5
	if d := math.Abs(float64(got[0].p[oi] - O[0].p[oi])); d > 2.5/255 {
		t.Errorf("far outside changed by %.1f/255", d*255)
	}
}

func TestStitchChecksSizes(t *testing.T) {
	a := image.NewNRGBA(image.Rect(0, 0, 10, 10))
	b := image.NewNRGBA(image.Rect(0, 0, 11, 10))
	if _, err := Stitch(a, b, make([]float32, 100), Options{}); err == nil {
		t.Fatal("want a size error")
	}
}

func TestOpacityLayerIsLeftPlain(t *testing.T) {
	w, h := 64, 64
	O := texture(w, h, 1, 0, 0)
	m := circleMask(w, h, 32, 32, 20, 4)
	for i := range m {
		m[i] *= 0.5
	}
	res := stitch(O, texture(w, h, 2, 0, 0), m, Options{Feather: 4})
	if res.Report.Skipped == "" {
		t.Fatal("a half-opacity layer should be left as the plain composite")
	}
}

// A detail edit (fine texture added, the colours the same) differs from the
// picture by little per pixel: the seam must not hand it back.
func TestSubtleDetailEditIsKept(t *testing.T) {
	w, h := 150, 170
	O := texture(w, h, 5, 0, 0)
	G := O.clone()
	rng := rand.New(rand.NewSource(9))
	for i := range G[0].p {
		n := float32(rng.NormFloat64()) * 0.03
		for k := 0; k < 3; k++ {
			G[k].p[i] += n
		}
	}
	mask := circleMask(w, h, 75, 85, 45, 15)
	res := stitch(O, G, mask, Options{Feather: 15, Seed: 1})
	got := over(res, O)
	// Inside the drawn mask (well past the few pixels the seam may move in) it is the edit.
	var dG, dO float64
	var n int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if math.Hypot(float64(x)-75, float64(y)-85) > 33 {
				continue
			}
			i := y*w + x
			dG += math.Abs(float64(got[0].p[i] - G[0].p[i]))
			dO += math.Abs(float64(got[0].p[i] - O[0].p[i]))
			n++
		}
	}
	if dG/float64(n) > 0.004 || dO < 3*dG {
		t.Fatalf("inside the mask: off the edit by %.4f, off the picture by %.4f", dG/float64(n), dO/float64(n))
	}
}

// With the generation at a higher resolution the result is made at that
// size and keeps its fine detail.
func TestDetailResolutionIsKept(t *testing.T) {
	w, h := 80, 60
	O := texture(w, h, 5, 0, 0)
	big := texture(4*w, 4*h, 5, 0, 0)
	// fine checker only the big one has
	for y := 0; y < 4*h; y++ {
		for x := 0; x < 4*w; x++ {
			if (x+y)%2 == 0 {
				for k := 0; k < 3; k++ {
					big[k].p[y*4*w+x] += 0.1
				}
			}
		}
	}
	img := image.NewNRGBA(image.Rect(0, 0, 4*w, 4*h))
	for i := 0; i < 16*w*h; i++ {
		for k := 0; k < 3; k++ {
			img.Pix[i*4+k] = to8(big[k].p[i])
		}
		img.Pix[i*4+3] = 255
	}
	small := texture(w, h, 5, 0, 0)
	res := stitch(O, small, circleMask(w, h, 40, 30, 20, 6), Options{Feather: 6, Seed: 1, Detail: img})
	b := res.Image.Bounds()
	if b.Dx() != 4*w || b.Dy() != 4*h {
		t.Fatalf("result %v, want the detail's size", b)
	}
	// The checker survives in the middle: neighbours differ by about 0.1.
	c := (4*30*4*w + 4*40) * 4
	d := math.Abs(float64(res.Image.Pix[c]) - float64(res.Image.Pix[c+4]))
	if d < 15 {
		t.Fatalf("fine detail lost: neighbours differ by %.0f/255", d)
	}
}
