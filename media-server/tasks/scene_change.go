package tasks

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/draw"
	"image/png"
	"math"
	"os"
	"strings"

	_ "image/jpeg"

	xdraw "golang.org/x/image/draw"
)

// Where a still version changed the picture it was made from. A layer's mask
// is drawn before the engine runs, and the engine never follows it exactly: a
// house sketched as a box comes back a little wider, with a shadow on the
// grass; a sun sketched as a disc comes back as a glare that lights half the
// sky. Composited through the sketch's mask all of that is cut off. The change
// map is what the editor grows the mask with ("Fit mask to result").
//
// Engines re-render the whole crop, so fine texture (grass, foliage) differs
// everywhere a little: that is noise, not change. Two measures together:
//   - fine: per-pixel difference after a light blur, crisp object edges, only
//     trusted close to the mask;
//   - coarse: difference of strongly blurred copies (re-rendered texture
//     averages out, a glow or a shadow does not), kept where it is connected
//     to the mask through changed pixels, so a glow comes along however far it
//     spreads, up to a cap.

const (
	// Fine difference (0-1, max over channels) below which a pixel is
	// unchanged and above which it clearly changed, and how far from the mask
	// it counts, as a share of the mask's size (square root of its area).
	changeQuiet = 0.07
	changeClear = 0.18
	changeReach = 0.35
	// Coarse difference thresholds, the blur (share of the long edge) it is
	// measured after, and the farthest a connected change may reach (share
	// of the mask's size).
	coarseQuiet = 0.03
	coarseClear = 0.09
	coarseBlur  = 0.008
	coarseReach = 4.0
)

// VersionChangeMap returns where still version n of layer lid differs from
// the picture it was made from, near the layer's mask: a PNG data URL (white,
// alpha = change) covering the version's region, and that region.
func VersionChangeMap(id, lid string, n int) (string, SceneRect, error) {
	sceneMu.Lock()
	s, err := loadSceneLocked(id)
	sceneMu.Unlock()
	if err != nil {
		return "", SceneRect{}, err
	}
	for _, l := range s.Layers {
		if l.ID != lid || l.Kind != SceneKindStill || n < 0 || n >= len(l.Versions) {
			continue
		}
		v := l.Versions[n]
		crop := sceneVersionCrop(v)
		if crop == "" {
			return "", SceneRect{}, fmt.Errorf("the picture v%d was made from is no longer on disk", n+1)
		}
		before, err := decodeImageFile(crop)
		if err != nil {
			return "", SceneRect{}, err
		}
		after, err := decodeImageFile(v.Path)
		if err != nil {
			return "", SceneRect{}, err
		}
		mask, err := decodeMaskDataURL(l.Mask)
		if err != nil {
			return "", SceneRect{}, err
		}
		url, err := changeMapDataURL(before, after, mask, v.Region, s.Width, s.Height)
		return url, v.Region, err
	}
	return "", SceneRect{}, ErrLayerNotFound
}

func decodeImageFile(p string) (image.Image, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func decodeMaskDataURL(u string) (image.Image, error) {
	const prefix = "data:image/png;base64,"
	if !strings.HasPrefix(u, prefix) {
		return nil, fmt.Errorf("the layer has no mask")
	}
	b, err := base64.StdEncoding.DecodeString(u[len(prefix):])
	if err != nil {
		return nil, err
	}
	return png.Decode(bytes.NewReader(b))
}

// changeMapDataURL compares before and after (both covering region of a
// fw×fh frame, any size) at sceneMotionMapLong, restricted to near what mask
// (full frame, any size) covers.
func changeMapDataURL(before, after, mask image.Image, region SceneRect, fw, fh int) (string, error) {
	w, h := region.W, region.H
	if long := max(w, h); long > sceneMotionMapLong {
		w, h = max(2, w*sceneMotionMapLong/long), max(2, h*sceneMotionMapLong/long)
	}
	a := resizeRGBA(before, w, h)
	b := resizeRGBA(after, w, h)
	// The mask over the region, at the same size.
	mb := mask.Bounds()
	sx, sy := float64(mb.Dx())/float64(fw), float64(mb.Dy())/float64(fh)
	src := image.Rect(mb.Min.X+int(float64(region.X)*sx), mb.Min.Y+int(float64(region.Y)*sy),
		mb.Min.X+int(math.Ceil(float64(region.X+region.W)*sx)), mb.Min.Y+int(math.Ceil(float64(region.Y+region.H)*sy)))
	m := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.ApproxBiLinear.Scale(m, m.Bounds(), mask, src, draw.Src, nil)

	inside := make([]bool, w*h)
	area := 0
	for i := range inside {
		if m.Pix[i*4+3] >= 128 {
			inside[i] = true
			area++
		}
	}
	size := math.Sqrt(float64(area))
	reach := max(4.0, changeReach*size)
	near := distanceTo(inside, w, h)

	fine := blurredDiff(a, b, w, h)
	r := max(1, int(math.Round(coarseBlur*float64(max(w, h)))))
	coarse := blurredDiff(blurRGBA(blurRGBA(a, r), r), blurRGBA(blurRGBA(b, r), r), w, h)
	limit := max(8, coarseReach*size)
	connected := growFrom(inside, coarse, near, coarseQuiet, limit, w, h)

	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	for i := range fine {
		// Fine: full strength within reach of the mask, fading out over the next quarter of it.
		c := smoothstep(changeQuiet, changeClear, fine[i]) * (1 - smoothstep(reach, reach*1.25, near[i]))
		if connected[i] {
			// Fading out over the last third of the cap, so it never shows as an edge.
			c = math.Max(c, smoothstep(coarseQuiet, coarseClear, coarse[i])*(1-smoothstep(limit*0.66, limit, near[i])))
		}
		out.Pix[i*4], out.Pix[i*4+1], out.Pix[i*4+2], out.Pix[i*4+3] = 255, 255, 255, uint8(math.Round(255*c))
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, out); err != nil {
		return "", err
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

// growFrom is every pixel reachable from seed through pixels whose change is
// at least quiet, no farther than limit from the seed (dist: each pixel's
// distance to it).
func growFrom(seed []bool, change, dist []float64, quiet, limit float64, w, h int) []bool {
	in := make([]bool, w*h)
	var queue []int
	for i, s := range seed {
		if s {
			in[i] = true
			queue = append(queue, i)
		}
	}
	for len(queue) > 0 {
		i := queue[len(queue)-1]
		queue = queue[:len(queue)-1]
		x, y := i%w, i/w
		for _, n := range [4][2]int{{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}} {
			if n[0] < 0 || n[1] < 0 || n[0] >= w || n[1] >= h {
				continue
			}
			j := n[1]*w + n[0]
			if !in[j] && change[j] >= quiet && dist[j] <= limit {
				in[j] = true
				queue = append(queue, j)
			}
		}
	}
	return in
}

// blurRGBA is img blurred by a (2r+1)² box (twice ≈ a gaussian).
func blurRGBA(img *image.RGBA, r int) *image.RGBA {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	tmp := image.NewRGBA(img.Rect)
	out := image.NewRGBA(img.Rect)
	pass := func(src, dst *image.RGBA, horizontal bool) {
		for y := 0; y < h; y++ {
			for x := 0; x < w; x++ {
				var sum [4]int
				n := 0
				for k := -r; k <= r; k++ {
					xx, yy := x, y
					if horizontal {
						xx += k
					} else {
						yy += k
					}
					if xx < 0 || yy < 0 || xx >= w || yy >= h {
						continue
					}
					p := (yy*w + xx) * 4
					for c := 0; c < 4; c++ {
						sum[c] += int(src.Pix[p+c])
					}
					n++
				}
				p := (y*w + x) * 4
				for c := 0; c < 4; c++ {
					dst.Pix[p+c] = uint8(sum[c] / n)
				}
			}
		}
	}
	pass(img, tmp, true)
	pass(tmp, out, false)
	return out
}

func resizeRGBA(img image.Image, w, h int) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	xdraw.ApproxBiLinear.Scale(out, out.Bounds(), img, img.Bounds(), draw.Src, nil)
	return out
}

// blurredDiff is the per-pixel colour difference of a and b (0-1, the largest
// channel difference) after a 3×3 box blur of each, so fine texture the
// engine re-rendered doesn't read as change.
func blurredDiff(a, b *image.RGBA, w, h int) []float64 {
	out := make([]float64, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			var sa, sb [3]float64
			n := 0.0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= w || yy >= h {
						continue
					}
					k := (yy*w + xx) * 4
					for c := 0; c < 3; c++ {
						sa[c] += float64(a.Pix[k+c])
						sb[c] += float64(b.Pix[k+c])
					}
					n++
				}
			}
			d := 0.0
			for c := 0; c < 3; c++ {
				d = math.Max(d, math.Abs(sa[c]-sb[c])/n/255)
			}
			out[y*w+x] = d
		}
	}
	return out
}

// distanceTo is each pixel's distance (pixels, chamfer 1/√2) to the nearest
// set pixel; very large when none is set.
func distanceTo(set []bool, w, h int) []float64 {
	d := make([]float64, w*h)
	for i, s := range set {
		if !s {
			d[i] = 1e9
		}
	}
	r2 := math.Sqrt2
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			if d[i] == 0 {
				continue
			}
			v := d[i]
			if x > 0 {
				v = math.Min(v, d[i-1]+1)
			}
			if y > 0 {
				v = math.Min(v, d[i-w]+1)
				if x > 0 {
					v = math.Min(v, d[i-w-1]+r2)
				}
				if x < w-1 {
					v = math.Min(v, d[i-w+1]+r2)
				}
			}
			d[i] = v
		}
	}
	for y := h - 1; y >= 0; y-- {
		for x := w - 1; x >= 0; x-- {
			i := y*w + x
			if d[i] == 0 {
				continue
			}
			v := d[i]
			if x < w-1 {
				v = math.Min(v, d[i+1]+1)
			}
			if y < h-1 {
				v = math.Min(v, d[i+w]+1)
				if x < w-1 {
					v = math.Min(v, d[i+w+1]+r2)
				}
				if x > 0 {
					v = math.Min(v, d[i+w-1]+r2)
				}
			}
			d[i] = v
		}
	}
	return d
}
