package stitch

import "math"

// flowField is a displacement per pixel: G sampled at (x+u, y+v) lines up
// with O at (x, y).
type flowField struct {
	u, v *plane
}

func (f flowField) mag(i int) float32 {
	return float32(math.Hypot(float64(f.u.p[i]), float64(f.v.p[i])))
}

// warp samples every plane of c at (x+u, y+v).
func warp(c rgb, f flowField) rgb {
	w, h := c[0].w, c[0].h
	out := newRGB(w, h)
	parallelRows(h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < w; x++ {
				i := y*w + x
				sx, sy := float32(x)+f.u.p[i], float32(y)+f.v.p[i]
				for k := 0; k < 3; k++ {
					out[k].p[i] = c[k].bilinear(sx, sy)
				}
			}
		}
	})
	return out
}

func warpPlane(a *plane, u, v *plane) *plane {
	out := newPlane(a.w, a.h)
	parallelRows(a.h, func(y0, y1 int) {
		for y := y0; y < y1; y++ {
			for x := 0; x < a.w; x++ {
				i := y*a.w + x
				out.p[i] = a.bilinear(float32(x)+u.p[i], float32(y)+v.p[i])
			}
		}
	})
	return out
}

// highpass takes away the local mean, so the flow is measured on structure
// and a tone shift between the two pictures can't pull it.
func highpass(a *plane, sigma float32) *plane {
	lo := blur(a, sigma)
	out := newPlane(a.w, a.h)
	for i := range out.p {
		out.p[i] = a.p[i] - lo.p[i]
	}
	return out
}

// estimateFlow measures where G's structure sits relative to O's, from the
// pixels with weight > 0 (where the two should show the same thing): weighted
// Lucas–Kanade, coarse to fine, each level's flow smoothed by normalized
// convolution so that flat patches borrow from textured ones. Returns the
// flow and its confidence (0-1: how well structured and how much it helped).
func estimateFlow(o, g *plane, weight *plane, maxShift float32) (flowField, *plane) {
	oh, gh := highpass(o, 3), highpass(g, 3)
	type level struct{ o, g, w *plane }
	levels := []level{{oh, gh, weight}}
	for len(levels) < 4 {
		l := levels[len(levels)-1]
		if min(l.o.w, l.o.h) < 48 || float32(int(1)<<len(levels)) > maxShift {
			break
		}
		levels = append(levels, level{down(l.o), down(l.g), down(l.w)})
	}
	var u, v *plane
	for li := len(levels) - 1; li >= 0; li-- {
		l := levels[li]
		w, h := l.o.w, l.o.h
		if u == nil {
			u, v = newPlane(w, h), newPlane(w, h)
		} else {
			u, v = up(u, w, h), up(v, w, h)
			for i := range u.p {
				u.p[i] *= 2
				v.p[i] *= 2
			}
		}
		lim := maxShift / float32(int(1)<<li)
		for it := 0; it < 6; it++ {
			gw := warpPlane(l.g, u, v)
			ix, iy, it2 := newPlane(w, h), newPlane(w, h), newPlane(w, h)
			for y := 0; y < h; y++ {
				for x := 0; x < w; x++ {
					i := y*w + x
					ix.p[i] = 0.25 * (gw.at(x+1, y) - gw.at(x-1, y) + l.o.at(x+1, y) - l.o.at(x-1, y))
					iy.p[i] = 0.25 * (gw.at(x, y+1) - gw.at(x, y-1) + l.o.at(x, y+1) - l.o.at(x, y-1))
					it2.p[i] = gw.p[i] - l.o.p[i]
				}
			}
			prod := func(a, b *plane) *plane {
				p := newPlane(w, h)
				for i := range p.p {
					p.p[i] = a.p[i] * b.p[i] * l.w.p[i]
				}
				return blur(p, 2.5)
			}
			a11, a12, a22 := prod(ix, ix), prod(ix, iy), prod(iy, iy)
			b1, b2 := prod(ix, it2), prod(iy, it2)
			var trace, n float64
			for i := range a11.p {
				if l.w.p[i] > 0 {
					trace += float64(a11.p[i] + a22.p[i])
					n++
				}
			}
			if n == 0 {
				return flowField{newPlane(o.w, o.h), newPlane(o.w, o.h)}, newPlane(o.w, o.h)
			}
			lambda := float32(0.05*trace/n) + 1e-9
			du, dv, conf := newPlane(w, h), newPlane(w, h), newPlane(w, h)
			for i := range du.p {
				p, q, r := a11.p[i]+lambda, a12.p[i], a22.p[i]+lambda
				det := p*r - q*q
				if det <= 0 {
					continue
				}
				du.p[i] = -(r*b1.p[i] - q*b2.p[i]) / det
				dv.p[i] = -(-q*b1.p[i] + p*b2.p[i]) / det
				// the smaller eigenvalue: how well both directions are pinned down
				tr, dt := a11.p[i]+a22.p[i], a11.p[i]*a22.p[i]-q*q
				disc := float32(math.Sqrt(math.Max(0, float64(tr*tr/4-dt))))
				conf.p[i] = tr/2 - disc
			}
			// Spread the update from where it is well measured.
			sdu, _ := blurWeighted(du, conf, 1.5)
			sdv, _ := blurWeighted(dv, conf, 1.5)
			for i := range u.p {
				d := float32(math.Hypot(float64(sdu.p[i]), float64(sdv.p[i])))
				if d > 1 {
					sdu.p[i] /= d
					sdv.p[i] /= d
				}
				u.p[i] = clampf(u.p[i]+sdu.p[i], -lim, lim)
				v.p[i] = clampf(v.p[i]+sdv.p[i], -lim, lim)
			}
		}
	}
	// Confidence at full resolution: structure (smaller eigenvalue, relative)
	// times how much the flow actually reduced the difference.
	w, h := o.w, o.h
	gw := warpPlane(gh, u, v)
	before, after := newPlane(w, h), newPlane(w, h)
	ix, iy := newPlane(w, h), newPlane(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			d0 := gh.p[i] - oh.p[i]
			d1 := gw.p[i] - oh.p[i]
			before.p[i] = d0 * d0
			after.p[i] = d1 * d1
			ix.p[i] = 0.5 * (oh.at(x+1, y) - oh.at(x-1, y))
			iy.p[i] = 0.5 * (oh.at(x, y+1) - oh.at(x, y-1))
		}
	}
	before, after = blur(before, 3), blur(after, 3)
	sxx, syy, sxy := newPlane(w, h), newPlane(w, h), newPlane(w, h)
	for i := range sxx.p {
		sxx.p[i] = ix.p[i] * ix.p[i]
		syy.p[i] = iy.p[i] * iy.p[i]
		sxy.p[i] = ix.p[i] * iy.p[i]
	}
	sxx, syy, sxy = blur(sxx, 2.5), blur(syy, 2.5), blur(sxy, 2.5)
	conf := newPlane(w, h)
	var top float32
	for i := range conf.p {
		tr, dt := sxx.p[i]+syy.p[i], sxx.p[i]*syy.p[i]-sxy.p[i]*sxy.p[i]
		disc := float32(math.Sqrt(math.Max(0, float64(tr*tr/4-dt))))
		conf.p[i] = tr/2 - disc
		if weight.p[i] > 0 && conf.p[i] > top {
			top = conf.p[i]
		}
	}
	for i := range conf.p {
		c := float32(0)
		if top > 0 {
			c = float32(math.Min(1, float64(conf.p[i]/(0.25*top))))
		}
		// Helped: the warped difference is clearly smaller than before.
		gain := float32(0)
		if before.p[i] > 1e-9 {
			gain = clampf(1-after.p[i]/before.p[i], 0, 1)
		}
		conf.p[i] = c * gain * weight.p[i]
	}
	return flowField{u, v}, conf
}

func clampf(v, lo, hi float32) float32 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
