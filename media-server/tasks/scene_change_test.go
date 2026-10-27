package tasks

import (
	"bytes"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"math"
	"testing"
)

// The fitted mask takes what the engine really drew: an object bigger than
// its sketch and the glow spreading from it, never re-rendered texture far
// from the mask.
func TestChangeMapFollowsTheResult(t *testing.T) {
	const w, h = 256, 160
	before := image.NewRGBA(image.Rect(0, 0, w, h))
	after := image.NewRGBA(image.Rect(0, 0, w, h))
	mask := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sky := color.RGBA{90, 150, 220, 255}
			before.Set(x, y, sky)
			c := sky
			// The engine's sun: a disc of radius 18 (the sketch was 12) and a glow fading out to radius 60.
			d := math.Hypot(float64(x-60), float64(y-60))
			if d < 18 {
				c = color.RGBA{255, 250, 230, 255}
			} else if d < 60 {
				k := 1 - (d-18)/42
				c = color.RGBA{uint8(90 + 165*k*k), uint8(150 + 100*k*k), uint8(220 + 10*k*k), 255}
			}
			// Far away, fine texture the engine re-rendered (a checker of small differences).
			if x > 200 && (x/2+y/2)%2 == 0 {
				c = color.RGBA{c.R + 40, c.G + 40, c.B, 255}
			}
			after.Set(x, y, c)
			if math.Hypot(float64(x-60), float64(y-60)) < 12 {
				mask.Set(x, y, color.NRGBA{255, 255, 255, 255})
			}
		}
	}
	url, err := changeMapDataURL(before, after, mask, SceneRect{W: w, H: h}, w, h)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := base64.StdEncoding.DecodeString(url[len("data:image/png;base64,"):])
	img, err := png.Decode(bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	alpha := func(x, y int) uint32 { _, _, _, a := img.At(x, y).RGBA(); return a >> 8 }
	if a := alpha(60+16, 60); a < 200 {
		t.Errorf("the disc past the sketch: alpha %d, want it covered", a)
	}
	if a := alpha(60+35, 60); a < 100 {
		t.Errorf("the glow 35 px out: alpha %d, want it kept", a)
	}
	if a := alpha(230, 120); a > 10 {
		t.Errorf("texture noise far from the mask: alpha %d, want none", a)
	}
}

// The lock's alignment check: a tone shift still matches, a reframed picture doesn't.
func TestFrameMatch(t *testing.T) {
	const w, h = 96, 64
	still := make([]float32, w*h*3)
	tinted := make([]float32, w*h*3)
	squeezed := make([]float32, w*h*3)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			v := float32(40 + 150*((x/12+y/12)%2))
			i := (y*w + x) * 3
			for c := 0; c < 3; c++ {
				still[i+c] = v
				tinted[i+c] = v*0.8 + 30
			}
			// Pillarboxed: the picture squeezed into the middle half, black either side.
			if x >= w/4 && x < 3*w/4 {
				sx := (x - w/4) * 2
				sv := float32(40 + 150*((sx/12+y/12)%2))
				squeezed[i], squeezed[i+1], squeezed[i+2] = sv, sv, sv
			}
		}
	}
	if m := frameMatch(still, tinted, w, h); m < 0.95 {
		t.Errorf("a tone shift: match %.2f, want ~1", m)
	}
	if m := frameMatch(still, squeezed, w, h); m >= lockMinMatch {
		t.Errorf("a pillarboxed frame: match %.2f, want below %.2f", m, lockMinMatch)
	}
}
