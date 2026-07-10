package onnxface

import (
	"errors"
	"image"
	"image/color"
	"image/draw"
)

// similarity is a 2D similarity transform (uniform scale + rotation +
// translation, no reflection):
//
//	x' = a*x - b*y + tx
//	y' = b*x + a*y + ty
type similarity struct {
	a, b, tx, ty float64
}

// apply maps a source point through the transform.
func (m similarity) apply(x, y float64) (float64, float64) {
	return m.a*x - m.b*y + m.tx, m.b*x + m.a*y + m.ty
}

// invert returns the inverse transform. The rotation+scale block [[a,-b],[b,a]]
// has inverse [[a,b],[-b,a]]/(a²+b²).
func (m similarity) invert() (similarity, error) {
	s2 := m.a*m.a + m.b*m.b
	if s2 == 0 {
		return similarity{}, errors.New("onnxface: degenerate similarity transform")
	}
	ai := m.a / s2
	bi := -m.b / s2
	// t_inv = -R_inv * t
	return similarity{
		a:  ai,
		b:  bi,
		tx: -(ai*m.tx - bi*m.ty),
		ty: -(bi*m.tx + ai*m.ty),
	}, nil
}

// estimateSimilarity computes the least-squares similarity transform mapping
// src points onto dst points (the closed-form Procrustes solution, equivalent
// to Umeyama's method when the optimal alignment involves no reflection —
// which is always the case for face landmarks mapped to the upright template).
func estimateSimilarity(src, dst [][2]float32) (similarity, error) {
	n := len(src)
	if n < 2 || len(dst) != n {
		return similarity{}, errors.New("onnxface: need >= 2 matching points")
	}
	var sxm, sym, dxm, dym float64
	for i := 0; i < n; i++ {
		sxm += float64(src[i][0])
		sym += float64(src[i][1])
		dxm += float64(dst[i][0])
		dym += float64(dst[i][1])
	}
	fn := float64(n)
	sxm /= fn
	sym /= fn
	dxm /= fn
	dym /= fn

	// With demeaned coordinates: a = Σ(xs·xd + ys·yd)/Σ(xs²+ys²),
	// b = Σ(xs·yd − ys·xd)/Σ(xs²+ys²).
	var num1, num2, den float64
	for i := 0; i < n; i++ {
		xs := float64(src[i][0]) - sxm
		ys := float64(src[i][1]) - sym
		xd := float64(dst[i][0]) - dxm
		yd := float64(dst[i][1]) - dym
		num1 += xs*xd + ys*yd
		num2 += xs*yd - ys*xd
		den += xs*xs + ys*ys
	}
	if den == 0 {
		return similarity{}, errors.New("onnxface: source points are coincident")
	}
	a := num1 / den
	b := num2 / den
	return similarity{
		a:  a,
		b:  b,
		tx: dxm - (a*sxm - b*sym),
		ty: dym - (b*sxm + a*sym),
	}, nil
}

// alignFace warps src so the five detected landmarks land on the canonical
// 112×112 template, returning the aligned crop. It estimates the
// landmark→template similarity transform, then inverse-maps each destination
// pixel back into src with bilinear sampling (out-of-bounds samples are black,
// matching OpenCV warpAffine's default border).
func alignFace(src image.Image, landmarks [5][2]float32) (*image.NRGBA, error) {
	fwd, err := estimateSimilarity(landmarks[:], alignTemplate[:])
	if err != nil {
		return nil, err
	}
	inv, err := fwd.invert()
	if err != nil {
		return nil, err
	}

	// Flatten to NRGBA once so sampling is O(1) per pixel regardless of the
	// source image type (image.Decode can return YCbCr, paletted, etc.).
	b := src.Bounds()
	flat, ok := src.(*image.NRGBA)
	if !ok || b.Min != (image.Point{}) {
		flat = image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(flat, flat.Bounds(), src, b.Min, draw.Src)
	}
	w := flat.Rect.Dx()
	h := flat.Rect.Dy()

	dst := image.NewNRGBA(image.Rect(0, 0, AlignSize, AlignSize))
	for dy := 0; dy < AlignSize; dy++ {
		for dx := 0; dx < AlignSize; dx++ {
			sx, sy := inv.apply(float64(dx), float64(dy))
			dst.SetNRGBA(dx, dy, sampleBilinear(flat, w, h, sx, sy))
		}
	}
	return dst, nil
}

// cropExpanded cuts an expanded square region around a detection's bbox and
// scales it to size×size — the "alignment" for detectors without landmarks
// (anime heads): expand ~1.5× turns a head box into a head+hair/bust crop,
// which carries the identity signal for drawn characters. Out-of-image area
// is black, mirroring warpAffine's border behavior.
func cropExpanded(src image.Image, d Detection, expand float32, size int) *image.NRGBA {
	if expand <= 0 {
		expand = 1
	}
	b := src.Bounds()
	cx := float64(d.X) + float64(d.W)/2
	cy := float64(d.Y) + float64(d.H)/2
	side := float64(d.W)
	if float64(d.H) > side {
		side = float64(d.H)
	}
	side *= float64(expand)
	if side < 1 {
		side = 1
	}
	x0 := cx - side/2
	y0 := cy - side/2
	scale := float64(size) / side

	// Flatten once for O(1) sampling (same as alignFace).
	flat, ok := src.(*image.NRGBA)
	if !ok || b.Min != (image.Point{}) {
		flat = image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(flat, flat.Bounds(), src, b.Min, draw.Src)
	}
	w := flat.Rect.Dx()
	h := flat.Rect.Dy()

	dst := image.NewNRGBA(image.Rect(0, 0, size, size))
	for dy := 0; dy < size; dy++ {
		for dx := 0; dx < size; dx++ {
			sx := x0 + (float64(dx)+0.5)/scale
			sy := y0 + (float64(dy)+0.5)/scale
			dst.SetNRGBA(dx, dy, sampleBilinear(flat, w, h, sx, sy))
		}
	}
	return dst
}

// sampleBilinear reads the source at fractional coordinates with bilinear
// interpolation; samples fully outside the image return black.
func sampleBilinear(img *image.NRGBA, w, h int, x, y float64) color.NRGBA {
	x0 := int(floorf(x))
	y0 := int(floorf(y))
	fx := x - float64(x0)
	fy := y - float64(y0)

	c00 := pixelOrBlack(img, w, h, x0, y0)
	c10 := pixelOrBlack(img, w, h, x0+1, y0)
	c01 := pixelOrBlack(img, w, h, x0, y0+1)
	c11 := pixelOrBlack(img, w, h, x0+1, y0+1)

	lerp2 := func(a, b, c, d uint8) uint8 {
		top := float64(a) + (float64(b)-float64(a))*fx
		bot := float64(c) + (float64(d)-float64(c))*fx
		v := top + (bot-top)*fy
		if v < 0 {
			return 0
		}
		if v > 255 {
			return 255
		}
		return uint8(v + 0.5)
	}
	return color.NRGBA{
		R: lerp2(c00.R, c10.R, c01.R, c11.R),
		G: lerp2(c00.G, c10.G, c01.G, c11.G),
		B: lerp2(c00.B, c10.B, c01.B, c11.B),
		A: 255,
	}
}

func pixelOrBlack(img *image.NRGBA, w, h, x, y int) color.NRGBA {
	if x < 0 || y < 0 || x >= w || y >= h {
		return color.NRGBA{A: 255}
	}
	return img.NRGBAAt(x, y)
}

func floorf(v float64) float64 {
	f := float64(int(v))
	if v < f {
		f--
	}
	return f
}
