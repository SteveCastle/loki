// Package stitch blends a generated patch into the picture it was made from
// so that its boundary disappears. The Scene editor's engines regenerate a
// whole crop (the mask plus context around it), so around every mask edge
// there are two versions of the same pixels: O, the picture under the layer,
// and G, the generation. Where the two should agree (outside the mask) their
// disagreement is measured and corrected, and the seam is moved to where
// they agree best:
//
//  1. Align: G's structure is registered to O's outside the mask (weighted
//     Lucas–Kanade) and the displacement is spread into a smooth field that
//     fades to nothing deep inside the mask, so lines that drifted a few
//     pixels bend back to meet the originals at the edge.
//  2. Tone: the colour difference that remains outside the mask is spread
//     inward as a gentle field that settles on the patch's overall shift, so
//     the edge matches exactly without tinting the middle.
//  3. Grain: noise the picture has and the generation lacks is added back.
//  4. Seam: a minimum cut through the band round the mask edge, through the
//     pixels where the two agree (and along edges, where a seam hides).
//  5. Blend: a Laplacian pyramid blend across that seam, wide for colour,
//     narrow for detail.
//  6. Residual: what still shows (gradient the seam adds that neither
//     picture has) is mapped, for an optional generative healing pass.
//
// The result is one RGBA image the size of the crop whose alpha carries the
// blend: composited over O through its own alpha it reproduces the blend.
package stitch

import (
	"errors"
	"image"
	"math"
	"math/rand"
	"sort"
)

// Steps switch the stages on and off (all on by default; off is for
// comparing).
type Steps struct {
	NoAlign, NoTone, NoGrain, NoSeam, NoMultiband bool
}

type Options struct {
	// Feather is the layer's feather (px); the band the seam may move in is
	// derived from it unless Band is set.
	Feather float64
	Band    int
	// Seed makes the grain reproducible.
	Seed  int64
	Steps Steps
	// Detail is the generation at its own (higher) resolution: the result
	// is then made at that size, every decision (seam, tone, alignment)
	// taken at the crop's size and applied to it, with its detail kept.
	Detail image.Image
}

// Report says what the stitch found and did.
type Report struct {
	Band         int     `json:"band"`
	Drift        float64 `json:"drift"`     // mean displacement corrected near the edge, px
	DriftMax     float64 `json:"driftMax"`  // its 95th percentile, px
	Tone         float64 `json:"tone"`      // the patch's overall colour shift corrected (0-255 RGB distance)
	GrainAdded   float64 `json:"grainAdded"` // noise added (0-255 sd)
	SeamLength   int     `json:"seamLength"` // px
	Residual     float64 `json:"residual"`   // what the seam still adds (0-255 excess gradient, mean along it)
	HealPixels   int     `json:"healPixels"` // where the seam still shows
	Skipped      string  `json:"skipped,omitempty"`
}

type Result struct {
	// Image is the blended patch, straight alpha: composite it over O.
	Image *image.NRGBA
	// Seam draws the seam (and where it still shows) for the editor.
	Seam *image.NRGBA
	// Heal is where the seam still shows (alpha), nil when nowhere.
	Heal   *image.Alpha
	Report Report
}

// Stitch blends g into o through mask (coverage 0-1 per pixel, row-major),
// all the size of the crop.
func Stitch(o, g image.Image, mask []float32, opt Options) (*Result, error) {
	ob, gb := o.Bounds(), g.Bounds()
	w, h := ob.Dx(), ob.Dy()
	if gb.Dx() != w || gb.Dy() != h || len(mask) != w*h {
		return nil, errors.New("stitch: picture, generation and mask differ in size")
	}
	O, G := fromImage(o), fromImage(g)
	return stitch(O, G, mask, opt), nil
}

// inwardLimit is how far inside the drawn mask the seam (and the alignment)
// may reach: the edit inside the mask is the point of the layer, and a
// detail edit differs little per pixel, so a seam free to wander in would
// hand it back to the picture below.
func inwardLimit(feather float64) float32 {
	return float32(math.Max(3, math.Min(10, feather/2)))
}

// core marks the pixels at least depth inside the mask, or, for a mask too
// thin to have any, its deepest part.
func core(d *plane, depth float32) []bool {
	minD := float32(0)
	for _, v := range d.p {
		minD = min(minD, v)
	}
	if minD > -depth {
		depth = -minD * 0.6
	}
	out := make([]bool, len(d.p))
	for i, v := range d.p {
		out[i] = v <= -depth
	}
	return out
}

func stitch(O, G rgb, mask []float32, opt Options) *Result {
	w, h := O[0].w, O[0].h
	n := w * h
	rep := Report{}
	band := opt.Band
	if band <= 0 {
		band = int(math.Round(opt.Feather + 10))
	}
	band = max(8, min(band, 64))
	rep.Band = band
	R := float32(band)
	inward := inwardLimit(opt.Feather)

	m := &plane{w: w, h: h, p: mask}
	var top float32
	inside := make([]bool, n)
	for i, a := range mask {
		inside[i] = a >= 0.5
		if a > top {
			top = a
		}
	}
	// A layer faded by its opacity has no inside to stitch.
	if top < 0.95 {
		rep.Skipped = "the mask is never fully on (layer opacity)"
		return plainResult(O, G, m, rep)
	}
	d := signedDistance(inside, w, h)

	// Where O and G should show the same thing: outside the mask, near it.
	near := newPlane(w, h)
	for i := range near.p {
		if mask[i] < 0.02 && d.p[i] > 1 && d.p[i] < 3*R {
			near.p[i] = 1
		}
	}
	nearCount := 0
	for _, v := range near.p {
		if v > 0 {
			nearCount++
		}
	}
	if nearCount < 64 {
		rep.Skipped = "no context round the mask to match against"
		return plainResult(O, G, m, rep)
	}

	// 1. Align.
	Gw := G
	var field *flowField
	if !opt.Steps.NoAlign {
		flow, conf := estimateFlow(O.luma(), G.luma(), near, max(4, R/2))
		// The field: the measured flow where it is trusted, nothing inside
		// the mask past the edge band (the edit itself is left as made),
		// smooth in between.
		fixed := core(d, 2*inward)
		cw := newPlane(w, h)
		for i := range cw.p {
			cw.p[i] = 0.2 * conf.p[i]
		}
		fu, fv := flow.u.clone(), flow.v.clone()
		for i, fx := range fixed {
			if fx {
				fu.p[i], fv.p[i] = 0, 0
			}
		}
		u := membrane(fu, cw, fixed, 60)
		v := membrane(fv, cw, fixed, 60)
		field = &flowField{u, v}
		Gw = warp(G, *field)
		var mags []float64
		for i := range d.p {
			if d.p[i] > -R/2 && d.p[i] < R/2 {
				mags = append(mags, float64(field.mag(i)))
			}
		}
		rep.Drift, rep.DriftMax = meanP95(mags)
	}

	// 2. Tone.
	Gt := Gw
	var tone rgb
	if !opt.Steps.NoTone {
		Gt, tone = toneMatch(O, Gw, mask, near, d, R, &rep)
	}

	// 3. Grain (at full resolution it is added there instead, per pixel).
	Gn := Gt
	var grain float32
	if !opt.Steps.NoGrain {
		grain = grainLevel(O, Gt, d, R, &rep)
		if grain > 0 && opt.Detail == nil {
			Gn = addGrain(Gt, d, R, grain, opt.Seed)
		}
	}

	// 4. Seam.
	var L *plane
	if opt.Steps.NoSeam {
		L = newPlane(w, h)
		for i := range L.p {
			if inside[i] {
				L.p[i] = 1
			}
		}
	} else {
		L = seamCut(O, Gt, mask, d, R, inward)
	}

	// 5. Blend.
	var C rgb
	var A *plane
	if opt.Steps.NoMultiband {
		soft := blur(L, float32(math.Max(1, opt.Feather/2)))
		C = newRGB(w, h)
		for k := 0; k < 3; k++ {
			for i := range soft.p {
				C[k].p[i] = Gn[k].p[i]*soft.p[i] + O[k].p[i]*(1-soft.p[i])
			}
		}
		A = soft
	} else {
		C, A = multiband(O, Gn, L, band)
	}

	// 6. Residual.
	seamPx := seamPixels(L)
	resid := excessGradient(C, O, Gn)
	rep.Residual = 255 * meanAt(resid, seamPx, 2)
	for _, s := range seamPx {
		if s {
			rep.SeamLength++
		}
	}
	heal := healMap(resid, seamPx, &rep)

	img := bake(C, O, A)
	if opt.Detail != nil {
		img = detailed(opt.Detail, O, C, A, L, field, tone, grain, opt.Seed)
	}
	return &Result{Image: img, Seam: seamOverlay(seamPx, heal, w, h), Heal: heal, Report: rep}
}

// detailed makes the result at the full-resolution generation's size: the
// blend C upsampled, plus the generation's own fine detail (aligned and
// tone-corrected the same way) wherever the seam gave the generation, grain
// added per pixel there, alpha the blend's.
func detailed(detail image.Image, O, C rgb, A, L *plane, field *flowField, tone rgb, grain float32, seed int64) *image.NRGBA {
	w, h := O[0].w, O[0].h
	D := fromImage(detail)
	W, H := D[0].w, D[0].h
	sx, sy := float32(W)/float32(w), float32(H)/float32(h)
	if field != nil {
		u, v := up(field.u, W, H), up(field.v, W, H)
		for i := range u.p {
			u.p[i] *= sx
			v.p[i] *= sy
		}
		D = warp(D, flowField{u, v})
	}
	if tone[0] != nil {
		for k := 0; k < 3; k++ {
			t := up(tone[k], W, H)
			for i := range t.p {
				D[k].p[i] += t.p[i]
			}
		}
	}
	gL := up(blur(L, 0.7), W, H)
	a := up(A, W, H)
	// Averaged over the sx*sy pixels a crop pixel stands for, this is the grain.
	var noise []float32
	gs := grain * float32(math.Sqrt(float64(sx*sy)))
	if grain > 0 {
		rng := rand.New(rand.NewSource(seed))
		noise = make([]float32, W*H)
		for i := range noise {
			noise[i] = float32(rng.NormFloat64())
		}
	}
	var hiC rgb
	for k := 0; k < 3; k++ {
		// detail: what the full-resolution picture has beyond the crop's resolution
		lo := up(downTo(D[k], w, h), W, H)
		c := up(C[k], W, H)
		for i := range c.p {
			c.p[i] += (D[k].p[i] - lo.p[i]) * gL.p[i]
			if noise != nil {
				c.p[i] += gs * gL.p[i] * noise[i]
			}
		}
		hiC[k] = c
	}
	o := rgb{up(O[0], W, H), up(O[1], W, H), up(O[2], W, H)}
	out := image.NewNRGBA(image.Rect(0, 0, W, H))
	for i := 0; i < W*H; i++ {
		a8 := to8(clampf(a.p[i], 0, 1))
		out.Pix[i*4+3] = a8
		if a8 == 0 {
			continue
		}
		aq := float32(a8) / 255
		for k := 0; k < 3; k++ {
			out.Pix[i*4+k] = to8((hiC[k].p[i] - (1-aq)*o[k].p[i]) / aq)
		}
	}
	return out
}

// downTo resamples a plane to w x h, low-passed first so it doesn't alias.
func downTo(a *plane, w, h int) *plane {
	sx, sy := float32(a.w)/float32(w), float32(a.h)/float32(h)
	b := blur(a, max(0, 0.5*max(sx, sy)))
	out := newPlane(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.p[y*w+x] = b.bilinear((float32(x)+0.5)*sx-0.5, (float32(y)+0.5)*sy-0.5)
		}
	}
	return out
}

// plainResult is the feathered composite, as the compositor would make it.
func plainResult(O, G rgb, m *plane, rep Report) *Result {
	w, h := O[0].w, O[0].h
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range m.p {
		img.Pix[i*4] = to8(G[0].p[i])
		img.Pix[i*4+1] = to8(G[1].p[i])
		img.Pix[i*4+2] = to8(G[2].p[i])
		img.Pix[i*4+3] = to8(m.p[i])
	}
	return &Result{Image: img, Report: rep}
}

// toneMatch corrects G's colour toward O's: measured outside the mask
// (robustly: pixels whose content differs don't count), spread as a smooth
// field that settles on the overall shift inside.
func toneMatch(O, G rgb, mask []float32, near, d *plane, R float32, rep *Report) (rgb, rgb) {
	w, h := O[0].w, O[0].h
	oy, gy := O.luma(), G.luma()
	ly := blur(oy, 2)
	lg := blur(gy, 2)
	wt := newPlane(w, h)
	for i := range wt.p {
		if near.p[i] == 0 {
			continue
		}
		// Content that differs (not a tone shift) shows as a local
		// difference in structure; it gets little say.
		s := (oy.p[i] - ly.p[i]) - (gy.p[i] - lg.p[i])
		wt.p[i] = 1 / (1 + (s/0.03)*(s/0.03))
	}
	out := newRGB(w, h)
	var fields rgb
	// Length scales (a membrane's data weight 1/ℓ² spreads a value about ℓ
	// px): the measurement is smoothed over a third of the band, and the
	// boundary's own correction gives way to the overall shift within about
	// one and a half bands inside.
	aw := float32(1 / math.Pow(float64(R)/3, 2))
	mu := float32(1 / math.Pow(1.5*float64(R), 2))
	var shift [3]float64
	for k := 0; k < 3; k++ {
		diff := newPlane(w, h)
		for i := range diff.p {
			diff.p[i] = O[k].p[i] - G[k].p[i]
		}
		// low frequency only: what's measured is a colour, not a texture
		ld := blur(diff, 2)
		g := weightedMedian(ld.p, wt.p)
		shift[k] = float64(g)
		f, c := newPlane(w, h), newPlane(w, h)
		for i := range f.p {
			a := wt.p[i] * aw
			c.p[i] = a + mu
			f.p[i] = (a*ld.p[i] + mu*g) / (a + mu)
		}
		t := membrane(f, c, nil, 80)
		for i := range t.p {
			out[k].p[i] = G[k].p[i] + t.p[i]
		}
		fields[k] = t
	}
	rep.Tone = 255 * math.Sqrt(shift[0]*shift[0]+shift[1]*shift[1]+shift[2]*shift[2])
	return out, fields
}

// grainLevel is the noise O has and G lacks (sd, 0-1), 0 when none:
// measured like for like, both pictures over the same context outside the
// mask (the generation re-rendered those pixels). Inside the mask O is what
// the edit replaced; its texture says nothing about grain.
func grainLevel(O, G rgb, d *plane, R float32, rep *Report) float32 {
	w, h := O[0].w, O[0].h
	ctx := make([]bool, w*h)
	for i := range ctx {
		ctx[i] = d.p[i] > 2 && d.p[i] < 3*R
	}
	so, sg := grainSigma(O.luma(), ctx), grainSigma(G.luma(), ctx)
	if so < 1.3*sg || so*so-sg*sg < (0.8/255)*(0.8/255) {
		return 0
	}
	s := float32(math.Sqrt(so*so - sg*sg))
	rep.GrainAdded = 255 * float64(s)
	return s
}

// addGrain adds luma-correlated noise of sd s (film or sensor grain) inside
// and near the mask.
func addGrain(G rgb, d *plane, R, s float32, seed int64) rgb {
	w, h := G[0].w, G[0].h
	rng := rand.New(rand.NewSource(seed))
	out := G.clone()
	for i := 0; i < w*h; i++ {
		if d.p[i] >= R {
			continue
		}
		l := float32(rng.NormFloat64()) * s
		for k := 0; k < 3; k++ {
			out[k].p[i] += l + 0.25*s*float32(rng.NormFloat64())
		}
	}
	return out
}

// grainSigma is grain, not texture: noise measured per 32 px tile and the
// quiet end taken (the 25th percentile). Grain is everywhere, so even the
// quietest tiles have it; a picture that is clean but sparkles in places
// (generated art, bokeh) has quiet tiles that are clean.
func grainSigma(a *plane, sel []bool) float64 {
	const tile = 32
	w, h := a.w, a.h
	var sigmas []float64
	for ty := 0; ty < h; ty += tile {
		for tx := 0; tx < w; tx += tile {
			sub := make([]bool, w*h)
			n := 0
			for y := ty; y < min(h, ty+tile); y++ {
				for x := tx; x < min(w, tx+tile); x++ {
					if sel[y*w+x] {
						sub[y*w+x] = true
						n++
					}
				}
			}
			if n < tile*tile/3 {
				continue
			}
			sigmas = append(sigmas, noiseSigma(a, sub))
		}
	}
	if len(sigmas) < 3 {
		return 0
	}
	sort.Float64s(sigmas)
	return sigmas[len(sigmas)/4]
}

func noiseSigma(a *plane, sel []bool) float64 {
	gm := gradMag(a)
	var gs []float64
	for i, ok := range sel {
		if ok {
			gs = append(gs, float64(gm.p[i]))
		}
	}
	if len(gs) < 64 {
		return 0
	}
	sort.Float64s(gs)
	flat := float32(gs[len(gs)/2]) // only the flatter half: edges aren't noise
	var sum float64
	var cnt int
	w := a.w
	for y := 1; y < a.h-1; y++ {
		for x := 1; x < w-1; x++ {
			i := y*w + x
			if !sel[i] || gm.p[i] > flat {
				continue
			}
			v := a.p[i-w-1] - 2*a.p[i-w] + a.p[i-w+1] - 2*a.p[i-1] + 4*a.p[i] - 2*a.p[i+1] + a.p[i+w-1] - 2*a.p[i+w] + a.p[i+w+1]
			sum += math.Abs(float64(v))
			cnt++
		}
	}
	if cnt == 0 {
		return 0
	}
	return math.Sqrt(math.Pi/2) / 6 * sum / float64(cnt)
}

// seamCut labels each pixel G (1) or O (0) with a minimum cut through the
// band round the mask edge. Deep inside is G; where the mask hardly reaches,
// and the crop's own border outside the mask, is O. A cut through pixels
// where O and G differ costs that difference; along an edge it costs less
// (a seam hides there); straying from the drawn edge costs a little; leaving
// out what changed inside the mask costs a lot.
func seamCut(O, G rgb, mask []float32, d *plane, R, inward float32) *plane {
	w, h := O[0].w, O[0].h
	n := w * h
	diff := newPlane(w, h)
	for i := range diff.p {
		var s float32
		for k := 0; k < 3; k++ {
			e := O[k].p[i] - G[k].p[i]
			s += e * e
		}
		diff.p[i] = float32(math.Sqrt(float64(s)))
	}
	diff = blur(diff, 0.7)
	gO, gG := gradMag(O.luma()), gradMag(G.luma())

	const (
		free = iota
		toG
		toO
	)
	state := make([]uint8, n)
	keep := core(d, inward)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			switch {
			case keep[i]:
				state[i] = toG
			case mask[i] < 0.1 || d.p[i] >= R:
				state[i] = toO
			case (x == 0 || y == 0 || x == w-1 || y == h-1) && d.p[i] > 0:
				state[i] = toO
			}
		}
	}
	idx := make([]int32, n)
	nodes := 0
	for i := range idx {
		idx[i] = -1
		if state[i] == free {
			idx[i] = int32(nodes)
			nodes++
		}
	}
	L := newPlane(w, h)
	for i := range L.p {
		if state[i] == toG {
			L.p[i] = 1
		}
	}
	if nodes == 0 {
		return L
	}
	s, t := nodes, nodes+1
	g := newMaxflow(nodes + 2)
	const scale = 10000
	cost := func(i, j int) int64 {
		e := (diff.p[i] + diff.p[j]) / 2
		e /= 1 + 3*(gO.p[i]+gO.p[j]+gG.p[i]+gG.p[j])
		prior := 0.004 * (abs32(d.p[i]) + abs32(d.p[j])) / (2 * R)
		return int64((e+0.002+prior)*scale) + 1
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if state[i] == free {
				p := int(idx[i])
				// Data: keeping out what changed inside, or letting in what changed outside.
				in := 2 * max(0, mask[i]-0.5)
				out := 2 * max(0, 0.5-mask[i])
				cO := int64(2.0 * in * diff.p[i] * scale)
				cG := int64(0.5 * out * diff.p[i] * scale)
				if cO > 0 || cG > 0 {
					g.edge(s, p, cO, 0)
					g.edge(p, t, cG, 0)
				}
			}
			for _, nb := range [2][2]int{{1, 0}, {0, 1}} {
				xx, yy := x+nb[0], y+nb[1]
				if xx >= w || yy >= h {
					continue
				}
				j := yy*w + xx
				a, b := state[i], state[j]
				if a != free && b != free {
					continue
				}
				c := cost(i, j)
				switch {
				case a == free && b == free:
					g.edge(int(idx[i]), int(idx[j]), c, c)
				case a == free:
					term(g, int(idx[i]), b == toG, s, t, c)
				default:
					term(g, int(idx[j]), a == toG, s, t, c)
				}
			}
		}
	}
	g.run(s, t)
	side := g.sourceSide(s)
	for i := range idx {
		if idx[i] >= 0 && side[idx[i]] {
			L.p[i] = 1
		}
	}
	return L
}

// term ties a free node to a fixed neighbour's terminal.
func term(g *maxflow, p int, toG bool, s, t int, c int64) {
	if toG {
		g.edge(s, p, c, 0)
	} else {
		g.edge(p, t, c, 0)
	}
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// multiband blends G into O across L's edge with a Laplacian pyramid, as many
// levels as the band allows. A is the alpha that reproduces the blend over O.
func multiband(O, G rgb, L *plane, band int) (rgb, *plane) {
	w, h := O[0].w, O[0].h
	levels := 1
	for (1<<levels) <= band/2 && levels < 5 && min(w, h)>>levels >= 8 {
		levels++
	}
	// mask pyramid
	ms := []*plane{L}
	for k := 1; k < levels; k++ {
		ms = append(ms, down(ms[k-1]))
	}
	// the mask levels, each smoothed a little (a hard step aliases)
	for k := range ms {
		ms[k] = blur(ms[k], 1)
	}
	C := newRGB(w, h)
	for c := 0; c < 3; c++ {
		gO, gG := []*plane{O[c]}, []*plane{G[c]}
		for k := 1; k < levels; k++ {
			gO = append(gO, down(gO[k-1]))
			gG = append(gG, down(gG[k-1]))
		}
		// coarsest: gaussian levels blended; finer: the laplacian bands
		acc := newPlane(gO[levels-1].w, gO[levels-1].h)
		for i := range acc.p {
			a := ms[levels-1].p[i]
			acc.p[i] = gG[levels-1].p[i]*a + gO[levels-1].p[i]*(1-a)
		}
		for k := levels - 2; k >= 0; k-- {
			lw, lh := gO[k].w, gO[k].h
			upO, upG := up(gO[k+1], lw, lh), up(gG[k+1], lw, lh)
			acc = up(acc, lw, lh)
			for i := range acc.p {
				a := ms[k].p[i]
				lapO := gO[k].p[i] - upO.p[i]
				lapG := gG[k].p[i] - upG.p[i]
				acc.p[i] += lapG*a + lapO*(1-a)
			}
		}
		C[c] = acc
	}
	// Where the blend differs from O at all: the widest mask level reaches
	// furthest, the finest is sharpest.
	A := ms[0].clone()
	for k := 1; k < levels; k++ {
		u := up(ms[k], w, h)
		for i := range A.p {
			A.p[i] = max(A.p[i], u.p[i])
		}
	}
	return C, A
}

// bake turns the blend C into an image that reproduces it when composited
// over O through its own alpha A: X = (C − (1−A)·O) / A.
func bake(C, O rgb, A *plane) *image.NRGBA {
	w, h := O[0].w, O[0].h
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		a := clampf(A.p[i], 0, 1)
		a8 := to8(a)
		img.Pix[i*4+3] = a8
		if a8 == 0 {
			continue
		}
		aq := float32(a8) / 255
		for k := 0; k < 3; k++ {
			x := (C[k].p[i] - (1-aq)*O[k].p[i]) / aq
			img.Pix[i*4+k] = to8(x)
		}
	}
	return img
}

// excessGradient is how much stronger C's gradient is than both pictures'
// own: what a seam adds.
func excessGradient(C, O, G rgb) *plane {
	gc, go_, gg := gradMag(C.luma()), gradMag(O.luma()), gradMag(G.luma())
	out := newPlane(gc.w, gc.h)
	for i := range out.p {
		out.p[i] = max(0, gc.p[i]-max(go_.p[i], gg.p[i]))
	}
	return out
}

// seamPixels are the pixels on either side of a change in L.
func seamPixels(L *plane) []bool {
	w, h := L.w, L.h
	s := make([]bool, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			a := L.p[i] >= 0.5
			if x+1 < w && (L.p[i+1] >= 0.5) != a {
				s[i], s[i+1] = true, true
			}
			if y+1 < h && (L.p[i+w] >= 0.5) != a {
				s[i], s[i+w] = true, true
			}
		}
	}
	return s
}

// meanAt is the mean of v over the pixels within r of a selected one.
func meanAt(v *plane, sel []bool, r int) float64 {
	w, h := v.w, v.h
	var sum float64
	var n int
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if !sel[y*w+x] {
				continue
			}
			best := float32(0)
			for dy := -r; dy <= r; dy++ {
				for dx := -r; dx <= r; dx++ {
					best = max(best, v.at(x+dx, y+dy))
				}
			}
			sum += float64(best)
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// healMap marks the stretches of seam that still show: excess gradient well
// above what a clean seam leaves, grown into a soft band to regenerate.
func healMap(resid *plane, seam []bool, rep *Report) *image.Alpha {
	w, h := resid.w, resid.h
	along := blur(resid, 1.5)
	const threshold = 6.0 / 255
	hot := newPlane(w, h)
	any := false
	for i, s := range seam {
		if s && along.p[i] > threshold {
			hot.p[i] = 1
			any = true
		}
	}
	if !any {
		return nil
	}
	// grow ~6 px, soften
	inside := make([]bool, w*h)
	for i := range inside {
		inside[i] = hot.p[i] > 0
	}
	dist := signedDistance(inside, w, h)
	img := image.NewAlpha(image.Rect(0, 0, w, h))
	soft := newPlane(w, h)
	for i := range soft.p {
		if dist.p[i] < 6 {
			soft.p[i] = 1
		}
	}
	soft = blur(soft, 2)
	for i := range soft.p {
		a := to8(soft.p[i])
		img.Pix[i] = a
		if a > 127 {
			rep.HealPixels++
		}
	}
	return img
}

// seamOverlay draws the seam (cyan) and where it still shows (orange).
func seamOverlay(seam []bool, heal *image.Alpha, w, h int) *image.NRGBA {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < w*h; i++ {
		if heal != nil && heal.Pix[i] > 0 {
			a := heal.Pix[i] / 2
			img.Pix[i*4], img.Pix[i*4+1], img.Pix[i*4+2], img.Pix[i*4+3] = 255, 140, 0, a
		}
		if seam[i] {
			img.Pix[i*4], img.Pix[i*4+1], img.Pix[i*4+2], img.Pix[i*4+3] = 0, 230, 255, 230
		}
	}
	return img
}

func weightedMedian(v, w []float32) float32 {
	type pair struct{ v, w float32 }
	var ps []pair
	var tot float32
	for i := range v {
		if w[i] > 0 {
			ps = append(ps, pair{v[i], w[i]})
			tot += w[i]
		}
	}
	if len(ps) == 0 {
		return 0
	}
	sort.Slice(ps, func(a, b int) bool { return ps[a].v < ps[b].v })
	var acc float32
	for _, p := range ps {
		acc += p.w
		if acc >= tot/2 {
			return p.v
		}
	}
	return ps[len(ps)-1].v
}

func meanP95(v []float64) (float64, float64) {
	if len(v) == 0 {
		return 0, 0
	}
	var s float64
	for _, x := range v {
		s += x
	}
	sort.Float64s(v)
	return s / float64(len(v)), v[len(v)*95/100]
}
