package stitch

// membrane solves for the smooth field x that follows data f where it is
// trusted (weight c) and is otherwise as flat as possible:
//
//	minimise  Σ c_i (x_i − f_i)²  +  Σ_neighbours (x_i − x_j)²
//
// with x pinned to f where fixed is true. The length a value spreads over is
// about 1/√c pixels, so a weak data weight gives a broad, gentle field (tone)
// and a strong one a field that hugs the data (flow). Red-black SOR, started
// from the solution at half resolution (so broad fields converge fast).
func membrane(f, c *plane, fixed []bool, iters int) *plane {
	w, h := f.w, f.h
	var x *plane
	if min(w, h) >= 24 {
		// Half resolution: data and weights averaged (weights add up over the
		// four pixels a coarse one stands for, while the neighbour coupling
		// per coarse pixel stays one: 4× the data weight per coarse pixel).
		cw, ch := (w+1)/2, (h+1)/2
		fc, cc := newPlane(cw, ch), newPlane(cw, ch)
		fixc := make([]bool, cw*ch)
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				i, j := y*w+x, (y/2)*cw+x/2
				fc.p[j] += c.p[i] * f.p[i]
				cc.p[j] += c.p[i]
				if fixed != nil && fixed[i] {
					fixc[j] = true
				}
			}
		}
		for j := range fc.p {
			if cc.p[j] > 1e-9 {
				fc.p[j] /= cc.p[j]
			}
		}
		// A coarse pixel is pinned only when all of it is (else its value would be wrong at the edge).
		if fixed != nil {
			for y := 0; y < ch; y++ {
				for x := 0; x < cw; x++ {
					all := true
					for dy := 0; dy < 2 && all; dy++ {
						for dx := 0; dx < 2; dx++ {
							xx, yy := 2*x+dx, 2*y+dy
							if xx < w && yy < h && !fixed[yy*w+xx] {
								all = false
								break
							}
						}
					}
					fixc[y*cw+x] = all
					if all {
						var s float32
						var n float32
						for dy := 0; dy < 2; dy++ {
							for dx := 0; dx < 2; dx++ {
								xx, yy := 2*x+dx, 2*y+dy
								if xx < w && yy < h {
									s += f.p[yy*w+xx]
									n++
								}
							}
						}
						fc.p[y*cw+x] = s / n
					}
				}
			}
		}
		xc := membrane(fc, cc, fixc, iters)
		x = up(xc, w, h)
	} else {
		x = newPlane(w, h)
		// Start from the trusted mean.
		var s, n float32
		for i := range f.p {
			s += c.p[i] * f.p[i]
			n += c.p[i]
		}
		if n > 0 {
			for i := range x.p {
				x.p[i] = s / n
			}
		}
	}
	if fixed != nil {
		for i, fx := range fixed {
			if fx {
				x.p[i] = f.p[i]
			}
		}
	}
	const omega = 1.7
	for it := 0; it < iters; it++ {
		for colour := 0; colour < 2; colour++ {
			parallelRows(h, func(y0, y1 int) {
				for y := y0; y < y1; y++ {
					for xx := (y + colour) & 1; xx < w; xx += 2 {
						i := y*w + xx
						if fixed != nil && fixed[i] {
							continue
						}
						num := c.p[i] * f.p[i]
						den := c.p[i]
						if xx > 0 {
							num += x.p[i-1]
							den++
						}
						if xx < w-1 {
							num += x.p[i+1]
							den++
						}
						if y > 0 {
							num += x.p[i-w]
							den++
						}
						if y < h-1 {
							num += x.p[i+w]
							den++
						}
						if den > 0 {
							x.p[i] += omega * (num/den - x.p[i])
						}
					}
				}
			})
		}
	}
	return x
}
