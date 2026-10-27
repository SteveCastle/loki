package tasks

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	xdraw "golang.org/x/image/draw"

	"github.com/stevecastle/shrike/jobqueue"
)

// --- scene-recolour: a Recolour finish's base coat ---
//
// The paint of a Recolour finish is a rough colour sketch: loose patches that
// overshoot some edges, stop short of others and leave gaps. Retouch can't use
// it: shown the paint over a picture that already has colour, it copies the
// patches onto the picture (tools/recolour lab, 2026-10-25: every prompt
// tried). So the base coat is made here, without diffusion:
//
//   - each paint colour's patches are clicked into loki-segment (a few points
//     well inside each patch), and of SAM's readings (part / object / whole,
//     plus its automatic masks) the one that best matches the patch wins
//     (F-beta 0.5: precision first, so a figure painted only on its robe gets
//     the robe, not the whole figure). That is "what the paint is on";
//   - the object takes the colour edge to edge, gaps included; paint that
//     only spills onto a neighbour is too little of it to win it;
//   - each object is tinted in OKLab: the paint's hue and chroma, and the
//     picture's own lightness shifted to the paint's, so shading, highlights
//     and texture stay (a base colour coat, exact to the picked shade).
//
// With the layer's words (light, shading) a Retouch job then finishes the
// coat (GenerateLayerMask chains it); without, the coat is the result.

// sceneRecolourWork is the long edge the analysis runs at.
const sceneRecolourWork = 1024

var sceneRecolourOptions = []TaskOption{
	{Name: "scene", Label: "Scene", Type: "string", Default: "", Description: "The scene id (its selection session)"},
	{Name: "paint", Label: "Paint", Type: "string", Default: "", Description: "The colour paint over the crop: an RGBA PNG the crop's size"},
	{Name: "out", Label: "Output", Type: "string", Default: "", Description: "Output PNG beside the crop; empty = <crop>_edit.png"},
}

func sceneRecolourTask(j *jobqueue.Job, q *jobqueue.Queue, _ *sync.Mutex) error {
	const tag = "scene-recolour"
	ctx := j.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}
	opts := ParseOptions(j, sceneRecolourOptions)
	crop := strings.TrimSpace(strings.Split(j.Input, "\n")[0])
	paint := strings.TrimSpace(optString(opts, "paint"))
	if crop == "" || paint == "" {
		return fail("needs the crop (input) and its paint (--paint)", nil)
	}
	out := strings.TrimSuffix(crop, filepath.Ext(crop)) + "_edit.png"
	if o := strings.TrimSpace(optString(opts, "out")); o != "" {
		if filepath.Dir(absPath(o)) != filepath.Dir(absPath(crop)) {
			return fail("--out must be beside the crop", nil)
		}
		out = absPath(o)
	}
	seg := func(req map[string]any) (map[string]any, error) {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		return segRequest("recolour:"+optString(opts, "scene"), crop, req)
	}
	_ = q.SetJobProgress(j.ID, 0, 1)
	st, err := recolourBase(crop, paint, out, seg, func(msg string) { q.PushJobStdout(j.ID, tag+": "+msg) })
	if err != nil {
		os.Remove(out)
		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		return fail(err.Error(), err)
	}
	q.PushJobStdout(j.ID, fmt.Sprintf("%s: %d colour(s), %d patch(es), %.0f%% of the crop recoloured -> %s", tag, st.Colours, st.Patches, st.Covered*100, out))
	q.RegisterOutputFile(j.ID, out, crop)
	_ = q.SetJobProgress(j.ID, 1, 1)
	q.CompleteJob(j.ID)
	return nil
}

type recolourStats struct {
	Colours, Patches int
	Covered          float64
}

// renderPaintCrop writes paint's colour painting over region (frame pixels
// of a fw×fh scene) as an RGBA PNG the region's size: what scene-recolour
// reads as the rough colour of the crop.
func renderPaintCrop(paint *SceneLayer, fw, fh int, region SceneRect, out string) error {
	img, err := decodeMaskDataURL(paint.Mask)
	if err != nil {
		return fmt.Errorf("the paint layer's painting: %w", err)
	}
	b := img.Bounds()
	sx, sy := float64(b.Dx())/float64(fw), float64(b.Dy())/float64(fh)
	src := image.Rect(b.Min.X+int(float64(region.X)*sx), b.Min.Y+int(float64(region.Y)*sy),
		b.Min.X+int(math.Ceil(float64(region.X+region.W)*sx)), b.Min.Y+int(math.Ceil(float64(region.Y+region.H)*sy)))
	dst := image.NewNRGBA(image.Rect(0, 0, region.W, region.H))
	xdraw.NearestNeighbor.Scale(dst, dst.Bounds(), img, src, draw.Src, nil)
	return writePNG(out, dst)
}

// segFunc runs one loki-segment request on the crop (segRequest's answer).
type segFunc func(req map[string]any) (map[string]any, error)

// recolourBase writes the base coat of crop (a PNG) from paint (RGBA PNG, any
// size: scaled to the crop) to out.
func recolourBase(crop, paint, out string, seg segFunc, logf func(string)) (recolourStats, error) {
	var st recolourStats
	pic, err := loadPNG(crop)
	if err != nil {
		return st, err
	}
	pb := pic.Bounds()
	W, H := pb.Dx(), pb.Dy()
	pim, err := loadPNG(paint)
	if err != nil {
		return st, err
	}
	// The analysis runs at a working size; masks come back at the crop's.
	k := math.Min(1, float64(sceneRecolourWork)/float64(max(W, H)))
	w, h := max(1, int(math.Round(float64(W)*k))), max(1, int(math.Round(float64(H)*k)))
	pw := resizeNRGBA(pim, w, h)

	cols, which := paintColours(pw, w, h)
	if len(cols) == 0 {
		return st, errors.New("nothing is painted over this area")
	}
	st.Colours = len(cols)

	tmp, err := os.MkdirTemp(filepath.Dir(crop), ".recolour-")
	if err != nil {
		return st, err
	}
	defer os.RemoveAll(tmp)

	// SAM's automatic masks: more readings for every patch to choose from.
	var every []bitmap
	if resp, err := seg(map[string]any{"everything": true, "out": filepath.Join(tmp, "every")}); err == nil {
		every = segMasks(resp, w, h, false)
	} else {
		logf("automatic masks failed (" + err.Error() + "): clicks only")
	}

	type pick struct {
		col  int
		mask bitmap
		prec float64
	}
	var picks []pick
	closeR := max(2, int(0.01*float64(max(w, h))))
	for c := range cols {
		sel := newBitmap(w, h)
		for i, v := range which {
			sel.on[i] = v == c
		}
		comp, n := sel.close(closeR).label()
		for id := 1; id <= n; id++ {
			patch := newBitmap(w, h)
			area := 0
			for i, v := range comp {
				if v == id && sel.on[i] {
					patch.on[i] = true
					area++
				}
			}
			if float64(area) < 0.0005*float64(w*h) {
				continue
			}
			st.Patches++
			region := newBitmap(w, h)
			for i, v := range comp {
				region.on[i] = v == id
			}
			var pts [][3]float64
			for _, p := range region.innerPoints(patch, 5) {
				pts = append(pts, [3]float64{float64(p[0]) / k, float64(p[1]) / k, 1})
			}
			cands := append([]bitmap{}, every...)
			if resp, err := seg(map[string]any{"points": pts, "multimask": true, "out": filepath.Join(tmp, fmt.Sprintf("c%d-%d", c, id))}); err == nil {
				cands = append(segMasks(resp, w, h, true), cands...)
			} else {
				logf("a patch's selection failed: " + err.Error())
			}
			best, bf, bp := bitmap{}, 0.0, 0.0
			for _, m := range cands {
				if f, p := fBeta(m, patch, 0.5); f > bf {
					best, bf, bp = m, f, p
				}
			}
			// Nothing SAM found fits: the rough patch itself, closed, is the area.
			if bf == 0 || bp < 0.35 {
				best, bp = region, 1
			}
			// A textured area (a floor, grass) comes back speckled: close it and fill its holes.
			picks = append(picks, pick{col: c, mask: best.close(max(2, int(0.006*float64(max(w, h))))).fillHoles(), prec: bp})
		}
	}
	// Big first, so a smaller object inside a bigger one keeps its own colour.
	sort.SliceStable(picks, func(a, b int) bool { return picks[a].mask.count() > picks[b].mask.count() })
	labels := make([]int, w*h)
	for i := range labels {
		labels[i] = -1
	}
	for _, p := range picks {
		for i, on := range p.mask.on {
			if on {
				labels[i] = p.col
			}
		}
	}
	fillIslands(labels, w, h, 0.02)

	covered := 0
	for _, v := range labels {
		if v >= 0 {
			covered++
		}
	}
	st.Covered = float64(covered) / float64(w*h)
	res := tintLabels(pic, labels, w, h, cols)
	return st, writePNG(out, res)
}

// paintColours finds the distinct colours of the painting (soft brush edges
// and opacity blend them, so near ones merge) and which one each painted
// pixel is (-1: unpainted).
func paintColours(p *image.NRGBA, w, h int) ([][3]float64, []int) {
	type bin struct {
		n   int
		sum [3]float64
	}
	bins := map[int]*bin{}
	painted := 0
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := p.PixOffset(x, y)
			if p.Pix[i+3] < 128 {
				continue
			}
			painted++
			key := int(p.Pix[i]>>3)<<10 | int(p.Pix[i+1]>>3)<<5 | int(p.Pix[i+2]>>3)
			b := bins[key]
			if b == nil {
				b = &bin{}
				bins[key] = b
			}
			b.n++
			for c := 0; c < 3; c++ {
				b.sum[c] += float64(p.Pix[i+c]) / 255
			}
		}
	}
	keys := make([]int, 0, len(bins))
	for k := range bins {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool { return bins[keys[a]].n > bins[keys[b]].n })
	var cols [][3]float64 // OKLab
	var weights []float64
	for _, k := range keys {
		b := bins[k]
		c := toOKLab([3]float64{b.sum[0] / float64(b.n), b.sum[1] / float64(b.n), b.sum[2] / float64(b.n)})
		near := -1
		for i, cc := range cols {
			// Only a soft edge's blend: skin and white socks are 0.05 apart and must stay two colours.
			if okDist(c, cc) < 0.025 {
				near = i
				break
			}
		}
		switch {
		case near >= 0:
			t := weights[near] + float64(b.n)
			for ch := 0; ch < 3; ch++ {
				cols[near][ch] = (cols[near][ch]*weights[near] + c[ch]*float64(b.n)) / t
			}
			weights[near] = t
		case float64(b.n) >= 0.01*float64(painted):
			cols = append(cols, c)
			weights = append(weights, float64(b.n))
		}
	}
	which := make([]int, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := p.PixOffset(x, y)
			which[y*w+x] = -1
			if p.Pix[i+3] < 128 || len(cols) == 0 {
				continue
			}
			c := toOKLab([3]float64{float64(p.Pix[i]) / 255, float64(p.Pix[i+1]) / 255, float64(p.Pix[i+2]) / 255})
			best, bd := -1, 0.12
			for j, cc := range cols {
				if d := okDist(c, cc); d < bd {
					best, bd = j, d
				}
			}
			which[y*w+x] = best
		}
	}
	return cols, which
}

// paintColourWords is the colours painted in paint (an RGBA PNG) in words,
// most painted first: "dark green (#2e6b3f), blue (#3a7bd5)". The engine pass
// that finishes a Recolour holds them (a colour it isn't told about drifts
// back toward the picture's old one); "" when it can't be read.
func paintColourWords(paint string) string {
	img, err := loadPNG(paint)
	if err != nil {
		return ""
	}
	b := img.Bounds()
	k := math.Min(1, 512/float64(max(b.Dx(), b.Dy())))
	w, h := max(1, int(float64(b.Dx())*k)), max(1, int(float64(b.Dy())*k))
	cols, _ := paintColours(resizeNRGBA(img, w, h), w, h)
	words := make([]string, 0, len(cols))
	for _, c := range cols {
		words = append(words, colourName(fromOKLab(c)))
	}
	return strings.Join(words, ", ")
}

// colourName names an sRGB colour (0-1) for a prompt, with its hex code:
// "dark green (#2e6b3f)", "pale peach (#f2c9b0)".
func colourName(c [3]float64) string {
	hex := fmt.Sprintf("#%02x%02x%02x", int(math.Round(c[0]*255)), int(math.Round(c[1]*255)), int(math.Round(c[2]*255)))
	mx, mn := math.Max(c[0], math.Max(c[1], c[2])), math.Min(c[0], math.Min(c[1], c[2]))
	l := (mx + mn) / 2
	s := 0.0
	if d := mx - mn; d > 1e-9 {
		s = d / (1 - math.Abs(2*l-1))
	}
	hsv := rgbHue(c)
	if l < 0.1 {
		return "black (" + hex + ")"
	}
	if s < 0.12 || (l > 0.9 && s < 0.3) {
		// A grey with a clear tint is named for it: a slate jacket is blue-grey, not grey.
		if s >= 0.06 && l >= 0.2 && l <= 0.85 {
			for _, b := range []struct {
				lim  float64
				name string
			}{{20, "red-grey"}, {50, "brown-grey"}, {70, "yellow-grey"}, {170, "green-grey"}, {260, "blue-grey"}, {330, "purple-grey"}, {361, "red-grey"}} {
				if hsv < b.lim {
					return b.name + " (" + hex + ")"
				}
			}
		}
		switch {
		case l > 0.9:
			return "white (" + hex + ")"
		case l > 0.65:
			return "light grey (" + hex + ")"
		case l < 0.35:
			return "dark grey (" + hex + ")"
		}
		return "grey (" + hex + ")"
	}
	base := "red"
	for _, b := range []struct {
		lim  float64
		name string
	}{{15, "red"}, {40, "orange"}, {65, "yellow"}, {80, "yellow-green"}, {160, "green"}, {190, "teal"}, {250, "blue"}, {280, "violet"}, {330, "magenta"}, {361, "red"}} {
		if hsv < b.lim {
			base = b.name
			break
		}
	}
	switch {
	case (base == "orange" || base == "yellow") && (l < 0.3 || (l < 0.45 && s < 0.6)):
		base = "brown"
	case base == "orange" && l > 0.5 && s < 0.55:
		base = "tan"
	case base == "orange" && l > 0.75:
		base = "peach"
	case base == "red" && l < 0.33:
		base = "maroon"
	case base == "blue" && l < 0.33:
		base = "navy blue"
	case (base == "red" || base == "magenta") && l > 0.75:
		base = "pink"
	}
	mod := ""
	switch {
	case l < 0.3 && base != "navy blue" && base != "maroon" && base != "brown":
		mod = "dark "
	case l > 0.8:
		mod = "pale "
	case s < 0.3:
		mod = "muted "
	}
	return mod + base + " (" + hex + ")"
}

// rgbHue is an sRGB colour's hue in degrees.
func rgbHue(c [3]float64) float64 {
	mx, mn := math.Max(c[0], math.Max(c[1], c[2])), math.Min(c[0], math.Min(c[1], c[2]))
	d := mx - mn
	if d < 1e-9 {
		return 0
	}
	var h float64
	switch mx {
	case c[0]:
		h = math.Mod((c[1]-c[2])/d, 6)
	case c[1]:
		h = (c[2]-c[0])/d + 2
	default:
		h = (c[0]-c[1])/d + 4
	}
	h *= 60
	if h < 0 {
		h += 360
	}
	return h
}

// tintLabels gives each labelled pixel of pic its colour: the paint's hue and
// chroma, and pic's lightness shifted so its mean over the colour's area is
// the paint's. Labels (w×h) are scaled to pic with a soft edge.
func tintLabels(pic image.Image, labels []int, w, h int, cols [][3]float64) *image.NRGBA {
	b := pic.Bounds()
	W, H := b.Dx(), b.Dy()
	src := image.NewNRGBA(image.Rect(0, 0, W, H))
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			src.Set(x, y, pic.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	// Per colour: its coverage at the crop's size, softened over a couple of pixels.
	alpha := make([][]float64, len(cols))
	soft := max(1, int(math.Round(float64(W)/float64(w))))
	for c := range cols {
		a := make([]float64, W*H)
		for y := 0; y < H; y++ {
			ly := min(h-1, y*h/H)
			for x := 0; x < W; x++ {
				if labels[ly*w+min(w-1, x*w/W)] == c {
					a[y*W+x] = 1
				}
			}
		}
		alpha[c] = fieldBlur(fieldBlur(a, W, H, soft), W, H, soft)
	}
	lab := make([][3]float64, W*H)
	for i := range lab {
		p := src.Pix[i*4 : i*4+3]
		lab[i] = toOKLab([3]float64{float64(p[0]) / 255, float64(p[1]) / 255, float64(p[2]) / 255})
	}
	shift := make([]float64, len(cols))
	for c := range cols {
		var s, n float64
		for i, a := range alpha[c] {
			s += a * lab[i][0]
			n += a
		}
		if n > 0 {
			shift[c] = cols[c][0] - s/n
		}
	}
	out := image.NewNRGBA(image.Rect(0, 0, W, H))
	copy(out.Pix, src.Pix)
	for i := range lab {
		var total float64
		var mix [3]float64
		for c := range cols {
			a := alpha[c][i]
			if a <= 0 {
				continue
			}
			t := [3]float64{math.Max(0, math.Min(1, lab[i][0]+shift[c])), cols[c][1], cols[c][2]}
			for ch := 0; ch < 3; ch++ {
				mix[ch] += a * t[ch]
			}
			total += a
		}
		if total <= 0 {
			continue
		}
		keep := math.Max(0, 1-total)
		if total > 1 {
			for ch := 0; ch < 3; ch++ {
				mix[ch] /= total
			}
			total = 1
		}
		var m [3]float64
		for ch := 0; ch < 3; ch++ {
			m[ch] = mix[ch] + keep*lab[i][ch]
		}
		rgb := fromOKLab(m)
		for ch := 0; ch < 3; ch++ {
			out.Pix[i*4+ch] = uint8(math.Round(math.Max(0, math.Min(1, rgb[ch])) * 255))
		}
	}
	return out
}

// segMasks loads every mask of a loki-segment answer (with candidates when
// asked) at w×h.
func segMasks(resp map[string]any, w, h int, candidates bool) []bitmap {
	b, _ := json.Marshal(resp)
	var raw struct {
		Objects []segRawObj `json:"objects"`
	}
	if json.Unmarshal(b, &raw) != nil {
		return nil
	}
	var paths []string
	for _, o := range raw.Objects {
		paths = append(paths, o.Mask)
		if candidates {
			for _, c := range o.Candidates {
				paths = append(paths, c.Mask)
			}
		}
	}
	var out []bitmap
	for _, p := range paths {
		if p == "" {
			continue
		}
		img, err := loadPNG(p)
		if err != nil {
			continue
		}
		g := resizeNRGBA(img, w, h)
		m := newBitmap(w, h)
		for i := range m.on {
			m.on[i] = g.Pix[i*4] >= 128
		}
		if m.count() > 0 {
			out = append(out, m)
		}
	}
	return out
}

// fBeta scores how well cand matches patch: F-beta of its precision (how much
// of cand is painted) and recall (how much of the patch it covers).
func fBeta(cand, patch bitmap, beta float64) (float64, float64) {
	inter, n, pn := 0, 0, 0
	for i, on := range cand.on {
		if on {
			n++
			if patch.on[i] {
				inter++
			}
		}
		if patch.on[i] {
			pn++
		}
	}
	if inter == 0 {
		return 0, 0
	}
	p, r := float64(inter)/float64(n), float64(inter)/float64(pn)
	return (1 + beta*beta) * p * r / (beta*beta*p + r), p
}

// fillIslands gives every unlabelled island smaller than maxFrac of the area,
// enclosed by one colour, that colour (the gaps between SAM's pieces).
func fillIslands(labels []int, w, h int, maxFrac float64) {
	free := newBitmap(w, h)
	for i, v := range labels {
		free.on[i] = v < 0
	}
	comp, n := free.label()
	members := make([][]int, n+1)
	for i, id := range comp {
		if id > 0 {
			members[id] = append(members[id], i)
		}
	}
	for id := 1; id <= n; id++ {
		if float64(len(members[id])) > maxFrac*float64(w*h) {
			continue
		}
		col, ok := -2, true
		for _, i := range members[id] {
			x, y := i%w, i/w
			for _, d := range [4][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} {
				xx, yy := x+d[0], y+d[1]
				if xx < 0 || yy < 0 || xx >= w || yy >= h {
					ok = false
					continue
				}
				v := labels[yy*w+xx]
				if v < 0 {
					continue
				}
				if col == -2 {
					col = v
				} else if v != col {
					ok = false
				}
			}
			if !ok {
				break
			}
		}
		if ok && col >= 0 {
			for _, i := range members[id] {
				labels[i] = col
			}
		}
	}
}

// bitmap is a w×h binary mask.
type bitmap struct {
	w, h int
	on   []bool
}

func newBitmap(w, h int) bitmap { return bitmap{w: w, h: h, on: make([]bool, w*h)} }

func (m bitmap) count() int {
	n := 0
	for _, v := range m.on {
		if v {
			n++
		}
	}
	return n
}

// grow dilates (r > 0) or erodes (r < 0) by a square of radius |r|.
func (m bitmap) grow(r int) bitmap {
	want := r > 0
	if r < 0 {
		r = -r
	}
	pass := func(src []bool, horiz bool) []bool {
		out := make([]bool, len(src))
		n, lines := m.w, m.h
		if !horiz {
			n, lines = m.h, m.w
		}
		at := func(line, i int) int {
			if horiz {
				return line*m.w + i
			}
			return i*m.w + line
		}
		for line := 0; line < lines; line++ {
			// prefix count of `want` pixels along the line
			pre := make([]int, n+1)
			for i := 0; i < n; i++ {
				pre[i+1] = pre[i]
				if src[at(line, i)] == want {
					pre[i+1]++
				}
			}
			for i := 0; i < n; i++ {
				lo, hi := max(0, i-r), min(n, i+r+1)
				hit := pre[hi]-pre[lo] > 0
				if want {
					out[at(line, i)] = hit
				} else {
					out[at(line, i)] = !hit
				}
			}
		}
		return out
	}
	return bitmap{w: m.w, h: m.h, on: pass(pass(m.on, true), false)}
}

func (m bitmap) close(r int) bitmap { return m.grow(r).grow(-r) }

// fillHoles turns on every off pixel not connected to the border.
func (m bitmap) fillHoles() bitmap {
	outside := make([]bool, len(m.on))
	var stack []int
	push := func(i int) {
		if !m.on[i] && !outside[i] {
			outside[i] = true
			stack = append(stack, i)
		}
	}
	for x := 0; x < m.w; x++ {
		push(x)
		push((m.h-1)*m.w + x)
	}
	for y := 0; y < m.h; y++ {
		push(y * m.w)
		push(y*m.w + m.w - 1)
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		x, y := i%m.w, i/m.w
		if x > 0 {
			push(i - 1)
		}
		if x < m.w-1 {
			push(i + 1)
		}
		if y > 0 {
			push(i - m.w)
		}
		if y < m.h-1 {
			push(i + m.w)
		}
	}
	out := newBitmap(m.w, m.h)
	for i := range out.on {
		out.on[i] = !outside[i]
	}
	return out
}

// label numbers the 4-connected components of the on pixels (1..n; 0 = off).
func (m bitmap) label() ([]int, int) {
	lab := make([]int, len(m.on))
	n := 0
	var stack []int
	for s, on := range m.on {
		if !on || lab[s] != 0 {
			continue
		}
		n++
		lab[s] = n
		stack = append(stack[:0], s)
		for len(stack) > 0 {
			i := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			x, y := i%m.w, i/m.w
			for _, j := range [4]int{i - 1, i + 1, i - m.w, i + m.w} {
				if j < 0 || j >= len(m.on) || (j == i-1 && x == 0) || (j == i+1 && x == m.w-1) || (j < i-1 && y == 0) {
					continue
				}
				if m.on[j] && lab[j] == 0 {
					lab[j] = n
					stack = append(stack, j)
				}
			}
		}
	}
	return lab, n
}

// innerPoints picks up to n points of patch deep inside region (chamfer
// distance to its edge), spread at least a quarter of the patch's size apart.
func (m bitmap) innerPoints(patch bitmap, n int) [][2]int {
	const inf = 1 << 30
	d := make([]int, len(m.on))
	for i, on := range m.on {
		if on {
			d[i] = inf
		}
	}
	w, h := m.w, m.h
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if d[i] == 0 {
				continue
			}
			if x > 0 {
				d[i] = min(d[i], d[i-1]+3)
			}
			if y > 0 {
				d[i] = min(d[i], d[i-w]+3)
				if x > 0 {
					d[i] = min(d[i], d[i-w-1]+4)
				}
				if x < w-1 {
					d[i] = min(d[i], d[i-w+1]+4)
				}
			}
			if x == 0 || y == 0 || x == w-1 || y == h-1 {
				d[i] = min(d[i], 3)
			}
		}
	}
	for y := h - 1; y >= 0; y-- {
		for x := w - 1; x >= 0; x-- {
			i := y*w + x
			if d[i] == 0 {
				continue
			}
			if x < w-1 {
				d[i] = min(d[i], d[i+1]+3)
			}
			if y < h-1 {
				d[i] = min(d[i], d[i+w]+3)
				if x < w-1 {
					d[i] = min(d[i], d[i+w+1]+4)
				}
				if x > 0 {
					d[i] = min(d[i], d[i+w-1]+4)
				}
			}
		}
	}
	var idx []int
	x0, y0, x1, y1 := w, h, -1, -1
	for i, on := range patch.on {
		if on {
			idx = append(idx, i)
			x, y := i%w, i/w
			x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x), max(y1, y)
		}
	}
	sort.Slice(idx, func(a, b int) bool { return d[idx[a]] > d[idx[b]] })
	gap := 0.25 * float64(max(x1-x0, y1-y0))
	var pts [][2]int
	for _, i := range idx {
		x, y := i%w, i/w
		ok := true
		for _, p := range pts {
			if math.Hypot(float64(x-p[0]), float64(y-p[1])) <= gap {
				ok = false
				break
			}
		}
		if ok {
			pts = append(pts, [2]int{x, y})
			if len(pts) >= n {
				break
			}
		}
	}
	return pts
}

// fieldBlur blurs a w×h field with a (2r+1) box, horizontally then vertically.
func fieldBlur(a []float64, w, h, r int) []float64 {
	tmp := make([]float64, len(a))
	out := make([]float64, len(a))
	for y := 0; y < h; y++ {
		s := 0.0
		for x := -r; x <= r; x++ {
			s += a[y*w+min(w-1, max(0, x))]
		}
		for x := 0; x < w; x++ {
			tmp[y*w+x] = s / float64(2*r+1)
			s += a[y*w+min(w-1, x+r+1)] - a[y*w+max(0, x-r)]
		}
	}
	for x := 0; x < w; x++ {
		s := 0.0
		for y := -r; y <= r; y++ {
			s += tmp[min(h-1, max(0, y))*w+x]
		}
		for y := 0; y < h; y++ {
			out[y*w+x] = s / float64(2*r+1)
			s += tmp[min(h-1, y+r+1)*w+x] - tmp[max(0, y-r)*w+x]
		}
	}
	return out
}

func loadPNG(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func writePNG(path string, img image.Image) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

func resizeNRGBA(img image.Image, w, h int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	b := img.Bounds()
	if b.Dx() == w && b.Dy() == h {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				out.Set(x, y, color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)))
			}
		}
		return out
	}
	r := resizeRGBA(img, w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.Set(x, y, color.NRGBAModel.Convert(r.At(x, y)))
		}
	}
	return out
}

// OKLab (Björn Ottosson) from and to sRGB in 0-1.
func toOKLab(c [3]float64) [3]float64 {
	lin := func(v float64) float64 {
		if v <= 0.04045 {
			return v / 12.92
		}
		return math.Pow((v+0.055)/1.055, 2.4)
	}
	r, g, b := lin(c[0]), lin(c[1]), lin(c[2])
	l := math.Cbrt(0.4122214708*r + 0.5363325363*g + 0.0514459929*b)
	m := math.Cbrt(0.2119034982*r + 0.6806995451*g + 0.1073969566*b)
	s := math.Cbrt(0.0883024619*r + 0.2817188376*g + 0.6299787005*b)
	return [3]float64{
		0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
		1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
		0.0259040371*l + 0.7827717662*m - 0.8086757660*s,
	}
}

func fromOKLab(c [3]float64) [3]float64 {
	l := c[0] + 0.3963377774*c[1] + 0.2158037573*c[2]
	m := c[0] - 0.1055613458*c[1] - 0.0638541728*c[2]
	s := c[0] - 0.0894841775*c[1] - 1.2914855480*c[2]
	l, m, s = l*l*l, m*m*m, s*s*s
	r := 4.0767416621*l - 3.3077115913*m + 0.2309699292*s
	g := -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
	b := -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
	enc := func(v float64) float64 {
		v = math.Max(0, math.Min(1, v))
		if v <= 0.0031308 {
			return v * 12.92
		}
		return 1.055*math.Pow(v, 1/2.4) - 0.055
	}
	return [3]float64{enc(r), enc(g), enc(b)}
}

func okDist(a, b [3]float64) float64 {
	return math.Sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]))
}
