package stitch

import (
	"image"
	"math"
	"runtime"
	"sync"
)

// plane is one float channel, row-major, w×h. Pictures are three planes in
// sRGB 0-1 (the compositor blends in sRGB, so the stitch does too).
type plane struct {
	w, h int
	p    []float32
}

func newPlane(w, h int) *plane { return &plane{w: w, h: h, p: make([]float32, w*h)} }

func (a *plane) clone() *plane {
	b := newPlane(a.w, a.h)
	copy(b.p, a.p)
	return b
}

func (a *plane) at(x, y int) float32 {
	if x < 0 {
		x = 0
	} else if x >= a.w {
		x = a.w - 1
	}
	if y < 0 {
		y = 0
	} else if y >= a.h {
		y = a.h - 1
	}
	return a.p[y*a.w+x]
}

// bilinear samples at (x, y) with the edges clamped.
func (a *plane) bilinear(x, y float32) float32 {
	x0, y0 := int(math.Floor(float64(x))), int(math.Floor(float64(y)))
	fx, fy := x-float32(x0), y-float32(y0)
	v00, v10 := a.at(x0, y0), a.at(x0+1, y0)
	v01, v11 := a.at(x0, y0+1), a.at(x0+1, y0+1)
	return (v00*(1-fx)+v10*fx)*(1-fy) + (v01*(1-fx)+v11*fx)*fy
}

type rgb [3]*plane

func newRGB(w, h int) rgb { return rgb{newPlane(w, h), newPlane(w, h), newPlane(w, h)} }

func (c rgb) clone() rgb { return rgb{c[0].clone(), c[1].clone(), c[2].clone()} }

// luma is Rec.601 luma (what the flow and the grain are measured on).
func (c rgb) luma() *plane {
	y := newPlane(c[0].w, c[0].h)
	for i := range y.p {
		y.p[i] = 0.299*c[0].p[i] + 0.587*c[1].p[i] + 0.114*c[2].p[i]
	}
	return y
}

func fromImage(img image.Image) rgb {
	b := img.Bounds()
	c := newRGB(b.Dx(), b.Dy())
	if n, ok := img.(*image.NRGBA); ok {
		for y := 0; y < b.Dy(); y++ {
			row := n.Pix[(y)*n.Stride:]
			for x := 0; x < b.Dx(); x++ {
				i := y*b.Dx() + x
				c[0].p[i] = float32(row[x*4]) / 255
				c[1].p[i] = float32(row[x*4+1]) / 255
				c[2].p[i] = float32(row[x*4+2]) / 255
			}
		}
		return c
	}
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := y*b.Dx() + x
			c[0].p[i], c[1].p[i], c[2].p[i] = float32(r)/65535, float32(g)/65535, float32(bl)/65535
		}
	}
	return c
}

func to8(v float32) uint8 {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return uint8(v*255 + 0.5)
}

// parallelRows runs fn over [0, h) in row bands on every core.
func parallelRows(h int, fn func(y0, y1 int)) {
	n := runtime.GOMAXPROCS(0)
	if n > h {
		n = h
	}
	if n <= 1 || h < 64 {
		fn(0, h)
		return
	}
	var wg sync.WaitGroup
	step := (h + n - 1) / n
	for y0 := 0; y0 < h; y0 += step {
		y1 := min(h, y0+step)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(y0, y1)
		}()
	}
	wg.Wait()
}

func gaussKernel(sigma float32) []float32 {
	r := int(math.Ceil(float64(sigma) * 3))
	if r < 1 {
		r = 1
	}
	k := make([]float32, 2*r+1)
	var sum float32
	for i := -r; i <= r; i++ {
		v := float32(math.Exp(-float64(i*i) / (2 * float64(sigma) * float64(sigma))))
		k[i+r] = v
		sum += v
	}
	for i := range k {
		k[i] /= sum
	}
	return k
}

// blur is a separable gaussian with clamped edges.
func blur(a *plane, sigma float32) *plane {
	if sigma <= 0 {
		return a.clone()
	}
	k := gaussKernel(sigma)
	r := len(k) / 2
	w, h := a.w, a.h
	tmp := newPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			row := a.p[y*w : (y+1)*w]
			out := tmp.p[y*w : (y+1)*w]
			for x := 0; x < w; x++ {
				var s float32
				for i := -r; i <= r; i++ {
					xx := x + i
					if xx < 0 {
						xx = 0
					} else if xx >= w {
						xx = w - 1
					}
					s += k[i+r] * row[xx]
				}
				out[x] = s
			}
		}
	})
	out := newPlane(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < w; x++ {
				var s float32
				for i := -r; i <= r; i++ {
					yy := y + i
					if yy < 0 {
						yy = 0
					} else if yy >= h {
						yy = h - 1
					}
					s += k[i+r] * tmp.p[yy*w+x]
				}
				out.p[y*w+x] = s
			}
		}
	})
	return out
}

// blurWeighted is normalized convolution: the weighted mean of v (weights wt)
// around each pixel, and the blurred weight (how much evidence there was).
func blurWeighted(v, wt *plane, sigma float32) (*plane, *plane) {
	vw := newPlane(v.w, v.h)
	for i := range vw.p {
		vw.p[i] = v.p[i] * wt.p[i]
	}
	num, den := blur(vw, sigma), blur(wt, sigma)
	for i := range num.p {
		if den.p[i] > 1e-6 {
			num.p[i] /= den.p[i]
		} else {
			num.p[i] = 0
		}
	}
	return num, den
}

// down halves a plane (blur, then every other pixel).
func down(a *plane) *plane {
	b := blur(a, 1)
	w, h := (a.w+1)/2, (a.h+1)/2
	out := newPlane(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.p[y*w+x] = b.p[min(2*y, a.h-1)*a.w+min(2*x, a.w-1)]
		}
	}
	return out
}

// up resamples a plane to w×h (bilinear, pixel centres aligned).
func up(a *plane, w, h int) *plane {
	out := newPlane(w, h)
	sx, sy := float32(a.w)/float32(w), float32(a.h)/float32(h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			fy := (float32(y)+0.5)*sy - 0.5
			for x := 0; x < w; x++ {
				out.p[y*w+x] = a.bilinear((float32(x)+0.5)*sx-0.5, fy)
			}
		}
	})
	return out
}

// gradMag is the central-difference gradient magnitude.
func gradMag(a *plane) *plane {
	g := newPlane(a.w, a.h)
	for y := 0; y < a.h; y++ {
		for x := 0; x < a.w; x++ {
			dx := (a.at(x+1, y) - a.at(x-1, y)) / 2
			dy := (a.at(x, y+1) - a.at(x, y-1)) / 2
			g.p[y*a.w+x] = float32(math.Sqrt(float64(dx*dx + dy*dy)))
		}
	}
	return g
}

// signedDistance is the euclidean distance of every pixel to the edge of
// inside (true pixels): negative inside, positive outside, about ±0.5 next to
// the edge. Felzenszwalb & Huttenlocher's exact transform, both ways.
func signedDistance(inside []bool, w, h int) *plane {
	dOut := edt(inside, w, h, true)  // outside pixels: distance to the nearest inside one
	dIn := edt(inside, w, h, false) // inside pixels: distance to the nearest outside one
	d := newPlane(w, h)
	for i := range d.p {
		if inside[i] {
			d.p[i] = -(dIn[i] - 0.5)
		} else {
			d.p[i] = dOut[i] - 0.5
		}
	}
	return d
}

// edt: distance of each pixel to the nearest pixel where inside == target
// (0 there); +Inf-ish when there is none.
func edt(inside []bool, w, h int, target bool) []float32 {
	const inf = 1e12
	f := make([]float64, w*h)
	for i, v := range inside {
		if v == target {
			f[i] = 0
		} else {
			f[i] = inf
		}
	}
	tmp := make([]float64, max(w, h))
	out := make([]float64, max(w, h))
	v := make([]int, max(w, h))
	z := make([]float64, max(w, h)+1)
	for x := 0; x < w; x++ {
		for y := 0; y < h; y++ {
			tmp[y] = f[y*w+x]
		}
		dt1(tmp[:h], out[:h], v, z)
		for y := 0; y < h; y++ {
			f[y*w+x] = out[y]
		}
	}
	for y := 0; y < h; y++ {
		copy(tmp[:w], f[y*w:(y+1)*w])
		dt1(tmp[:w], out[:w], v, z)
		copy(f[y*w:(y+1)*w], out[:w])
	}
	d := make([]float32, w*h)
	for i := range f {
		d[i] = float32(math.Sqrt(f[i]))
	}
	return d
}

// dt1 is the 1-D squared distance transform (lower envelope of parabolas).
func dt1(f, d []float64, v []int, z []float64) {
	n := len(f)
	k := 0
	v[0] = 0
	z[0] = math.Inf(-1)
	z[1] = math.Inf(1)
	for q := 1; q < n; q++ {
		var s float64
		for {
			p := v[k]
			s = ((f[q] + float64(q*q)) - (f[p] + float64(p*p))) / float64(2*q-2*p)
			if s <= z[k] && k > 0 {
				k--
				continue
			}
			break
		}
		if s <= z[k] {
			// k == 0 and the new parabola wins everywhere
			v[0] = q
			z[0] = math.Inf(-1)
			z[1] = math.Inf(1)
			continue
		}
		k++
		v[k] = q
		z[k] = s
		z[k+1] = math.Inf(1)
	}
	k = 0
	for q := 0; q < n; q++ {
		for z[k+1] < float64(q) {
			k++
		}
		p := v[k]
		d[q] = float64((q-p)*(q-p)) + f[p]
	}
}
