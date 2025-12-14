// sbs_occl.go
package main

import (
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"

	_ "golang.org/x/image/webp"

	xdraw "golang.org/x/image/draw"
)

func main() {
	inPath := flag.String("in", "", "input color image path (PNG/JPEG/WEBP)")
	depthPath := flag.String("depth", "", "input depth map path")
	outPath := flag.String("out", "stereo.png", "output side-by-side PNG path")

	strength := flag.Float64("strength", 0.015, "max parallax as fraction of image width (ignored if --max-shift-px>0)")
	maxShiftPX := flag.Int("max-shift-px", 0, "explicit max parallax in pixels (overrides --strength if >0)")
	invertDepth := flag.Bool("invert-depth", true, "invert depth (treat black as near)")
	gamma := flag.Float64("gamma", 1.0, "gamma for depth shaping (>=0.1)")
	depthMode := flag.String("depth-mode", "inverse", "disparity mapping: inverse|linear")
	nearPct := flag.Float64("near-pct", 2, "percentile for near clip (0..100)")
	farPct := flag.Float64("far-pct", 98, "percentile for far clip (0..100)")
	convergeU := flag.Float64("converge-u", 0.75, "zero-parallax depth in normalized 0..1 after depth shaping")

	// Simple SBS path (reference-based)
	simple := flag.Bool("simple", true, "use simple SBS generation (based on reference)")
	mode := flag.String("mode", "Parallel", "stereo mode: Parallel|Cross")
	depthScale := flag.Float64("depth-scale", 30.0, "depth scale factor; shift ≈ depth*depth_scale/width (px)")
	blurRadius := flag.Int("blur-radius", 3, "pre-blur radius (odd, >=1) for simple SBS")
	focalU := flag.Float64("focal-u", 0, "focus plane in 0..1 (0 near, 1 far) for simple blur weighting")

	edgeFeather := flag.Float64("edge-feather", 1.5, "feather width in px around occlusion edges (0..2)")
	fillPasses := flag.Int("fill-passes", 5, "push-pull hole fill passes (0..5)")
	doJointBilateral := flag.Bool("joint-bilateral", true, "apply joint-bilateral smoothing to depth")
	jbRadius := flag.Int("jb-radius", 3, "joint-bilateral radius (px)")
	jbSigmaC := flag.Float64("jb-sigma-c", 12.0, "joint-bilateral color sigma (luma 0..255)")

	ratioTol := flag.Float64("ratio-tol", 0.002, "max allowed aspect ratio mismatch before error (e.g., 0.002 = 0.2%)")
	threads := flag.Int("threads", runtime.GOMAXPROCS(0), "worker goroutines")

	// Fisheye / VR pre-distortion
	fisheye := flag.Bool("fisheye", false, "apply VR-style barrel pre-distortion to each eye")
	k1 := flag.Float64("k1", -0.28, "radial distortion coefficient k1 (negative = barrel)")
	k2 := flag.Float64("k2", 0.06, "radial distortion coefficient k2")
	fisheyeScale := flag.Float64("fisheye-scale", 0.95, "uniform pre-scale before distortion (0.85..1.1)")
	lensCenterX := flag.Float64("lens-center-x", 0.5, "lens center X fraction (0..1)")
	lensCenterY := flag.Float64("lens-center-y", 0.5, "lens center Y fraction (0..1)")
	vrPreset := flag.String("vr-preset", "quest3", "VR lens preset: quest3|quest2|index|psvr2|none")

	flag.Parse()
	if *inPath == "" || *depthPath == "" {
		fmt.Fprintln(os.Stderr, "usage: --in <image> --depth <depth> [--out stereo.png] ...")
		os.Exit(2)
	}
	if *gamma < 0.1 {
		*gamma = 0.1
	}
	clamp01Ptr(lensCenterX)
	clamp01Ptr(lensCenterY)

	// Apply VR preset (overrides k1, k2, fisheye-scale) unless set to none
	if *vrPreset != "none" {
		applyVRPreset(*vrPreset, k1, k2, fisheyeScale)
	}

	// Load
	srcImg, err := loadImage(*inPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load input image: %v\n", err)
		os.Exit(1)
	}
	w, h := srcImg.Bounds().Dx(), srcImg.Bounds().Dy()

	depthImgRaw, err := loadImage(*depthPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to load depth image: %v\n", err)
		os.Exit(1)
	}
	dw, dh := depthImgRaw.Bounds().Dx(), depthImgRaw.Bounds().Dy()

	// Aspect ratio check (before scaling)
	arA := float64(w) / float64(h)
	arB := float64(dw) / float64(dh)
	if math.Abs(arA-arB)/arA > *ratioTol {
		fmt.Fprintf(os.Stderr, "aspect ratios differ too much: image %.6f vs depth %.6f (tol=%.6f)\n", arA, arB, *ratioTol)
		os.Exit(1)
	}

	// Prep color RGBA and Gray depth aligned to (w,h)
	rgba := toRGBA(srcImg)
	depthGray := image.NewGray(image.Rect(0, 0, w, h))
	xdraw.ApproxBiLinear.Scale(depthGray, depthGray.Bounds(), depthImgRaw, depthImgRaw.Bounds(), draw.Src, nil)

	// SIMPLE SBS SHORT-CIRCUIT
	if *simple {
		base := rgba
		if *blurRadius > 0 {
			base = depthWeightedBlurRGBA(base, depthGray, *blurRadius, *focalU, *invertDepth, *threads)
		}
		out := simpleSBS(base, depthGray, *mode, *depthScale, *invertDepth)
		if *fisheye {
			left := image.NewRGBA(image.Rect(0, 0, w, h))
			right := image.NewRGBA(image.Rect(0, 0, w, h))
			draw.Draw(left, left.Bounds(), out, image.Point{}, draw.Src)
			draw.Draw(right, right.Bounds(), out, image.Point{X: w, Y: 0}, draw.Src)
			left = fisheyeWarp(left, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
			right = fisheyeWarp(right, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
			final := image.NewRGBA(image.Rect(0, 0, w*2, h))
			draw.Draw(final, image.Rect(0, 0, w, h), left, image.Point{}, draw.Src)
			draw.Draw(final, image.Rect(w, 0, w*2, h), right, image.Point{}, draw.Src)
			if err := savePNG(*outPath, final); err != nil {
				fmt.Fprintf(os.Stderr, "Failed to save PNG: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Wrote %s (%dx%d -> %dx%d SBS simple) fisheye=%v\n", *outPath, w, h, w*2, h, *fisheye)
			return
		}
		if err := savePNG(*outPath, out); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to save PNG: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Wrote %s (%dx%d -> %dx%d SBS simple) fisheye=%v\n", *outPath, w, h, w*2, h, *fisheye)
		return
	}

	// Disabled complex SBS path
	if false {
		// Optional joint-bilateral smoothing on depth using color luma guidance
		if *doJointBilateral {
			depthGray = jointBilateralGray(depthGray, rgba, *jbRadius, *jbSigmaC, *threads)
		}
		// Build depth buffer (float32 z in 0..1 after clipping+gamma+invert)
		zbuf := make([]float32, w*h)
		fillDepthBuffer(zbuf, depthGray, *invertDepth, *gamma, *nearPct, *farPct)
		// Determine max shift in pixels
		maxShift := float64(*maxShiftPX)
		if maxShift <= 0 {
			maxShift = math.Max(0.0, *strength) * float64(w)
		}
		if maxShift > float64(w)/2 {
			maxShift = float64(w) / 2
		}
		// Map depth to disparity (pixels)
		disp := make([]float32, w*h)
		mapDepthToDisparity(disp, zbuf, float32(maxShift), *depthMode, float32(*convergeU))
		// Per-eye synthesis by forward splatting
		left := image.NewRGBA(image.Rect(0, 0, w, h))
		right := image.NewRGBA(image.Rect(0, 0, w, h))
		leftTouched := make([]uint8, w*h)
		rightTouched := make([]uint8, w*h)
		forwardSplatEye(rgba, disp, zbuf, +0.5, left, leftTouched, *edgeFeather, *threads)
		forwardSplatEye(rgba, disp, zbuf, -0.5, right, rightTouched, *edgeFeather, *threads)
		if *fillPasses > 0 {
			pushPullFill(left, leftTouched, *fillPasses)
			pushPullFill(right, rightTouched, *fillPasses)
		}
		if *fisheye {
			left = fisheyeWarp(left, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
			right = fisheyeWarp(right, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
		}
		out := image.NewRGBA(image.Rect(0, 0, w*2, h))
		draw.Draw(out, image.Rect(0, 0, w, h), left, image.Point{}, draw.Src)
		draw.Draw(out, image.Rect(w, 0, w*2, h), right, image.Point{}, draw.Src)
		_ = out
	}

	// Optional joint-bilateral smoothing on depth using color luma guidance
	if *doJointBilateral {
		depthGray = jointBilateralGray(depthGray, rgba, *jbRadius, *jbSigmaC, *threads)
	}

	// Build depth buffer (float32 z in 0..1 after clipping+gamma+invert)
	zbuf := make([]float32, w*h)
	fillDepthBuffer(zbuf, depthGray, *invertDepth, *gamma, *nearPct, *farPct)

	// Determine max shift in pixels
	maxShift := float64(*maxShiftPX)
	if maxShift <= 0 {
		maxShift = math.Max(0.0, *strength) * float64(w)
	}
	if maxShift > float64(w)/2 {
		maxShift = float64(w) / 2
	}

	// Map depth to disparity (pixels). inverse-depth is more optical: d ∝ 1/z
	disp := make([]float32, w*h)
	mapDepthToDisparity(disp, zbuf, float32(maxShift), *depthMode, float32(*convergeU))

	// Per-eye synthesis by forward splatting with Z-buffer
	left := image.NewRGBA(image.Rect(0, 0, w, h))
	right := image.NewRGBA(image.Rect(0, 0, w, h))
	leftTouched := make([]uint8, w*h) // mask of filled pixels
	rightTouched := make([]uint8, w*h)

	forwardSplatEye(rgba, disp, zbuf, +0.5, left, leftTouched, *edgeFeather, *threads) // +½ disparity
	forwardSplatEye(rgba, disp, zbuf, -0.5, right, rightTouched, *edgeFeather, *threads)

	// Push–pull inpainting for holes
	if *fillPasses > 0 {
		pushPullFill(left, leftTouched, *fillPasses)
		pushPullFill(right, rightTouched, *fillPasses)
	}

	// Optional fisheye prewarp
	if *fisheye {
		left = fisheyeWarp(left, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
		right = fisheyeWarp(right, *k1, *k2, *fisheyeScale, *lensCenterX, *lensCenterY, *threads)
	}

	// Compose SBS
	out := image.NewRGBA(image.Rect(0, 0, w*2, h))
	draw.Draw(out, image.Rect(0, 0, w, h), left, image.Point{}, draw.Src)
	draw.Draw(out, image.Rect(w, 0, w*2, h), right, image.Point{}, draw.Src)
	if err := savePNG(*outPath, out); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to save PNG: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Wrote %s (%dx%d -> %dx%d SBS) fisheye=%v\n", *outPath, w, h, w*2, h, *fisheye)
}

/*** Depth prep ***/

func fillDepthBuffer(zbuf []float32, gray *image.Gray, invert bool, gamma float64, nearPct, farPct float64) {
	w, h := gray.Bounds().Dx(), gray.Bounds().Dy()
	N := w * h

	// Collect histogram/values for percentile clipping
	arr := make([]uint8, 0, N)
	for y := 0; y < h; y++ {
		row := gray.Pix[y*gray.Stride : y*gray.Stride+w]
		arr = append(arr, row...)
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i] < arr[j] })

	pClamp := func(p float64) uint8 {
		if p <= 0 {
			return arr[0]
		}
		if p >= 100 {
			return arr[len(arr)-1]
		}
		idx := int((p / 100.0) * float64(len(arr)-1))
		if idx < 0 {
			idx = 0
		}
		if idx >= len(arr) {
			idx = len(arr) - 1
		}
		return arr[idx]
	}
	vNear := float32(pClamp(nearPct))
	vFar := float32(pClamp(farPct))
	if vNear == vFar {
		// Fallback
		vNear, vFar = 255, 0
	}
	// Map to 0..1 with clipping then gamma/invert
	den := float32(vFar - vNear)
	if den == 0 {
		den = 1
	}
	g := float32(gamma)

	for y := 0; y < h; y++ {
		row := gray.Pix[y*gray.Stride : y*gray.Stride+w]
		for x := 0; x < w; x++ {
			v := float32(row[x])
			if v < vNear {
				v = vNear
			}
			if v > vFar {
				v = vFar
			}
			zn := (v - vNear) / den // 0..1
			if invert {
				zn = 1 - zn
			}
			if g != 1.0 {
				zn = float32(math.Pow(float64(zn), float64(g)))
			}
			zbuf[y*w+x] = zn
		}
	}
}

// map depth to disparity (px). inverse: d = Dmax * (1/(eps+z) - 1), re-normalized to 0..1 → centered later by +/-0.5 factor per eye.
// linear: d = Dmax * z (centered later).
func mapDepthToDisparity(disp, z []float32, dmax float32, mode string, convergeU float32) {
	const eps = 1e-4
	n := len(z)
	switch mode {
	case "inverse":
		// Normalize inverse depth into 0..1
		imin, imax := float32(+1e9), float32(-1e9)
		tmp := make([]float32, n)
		for i := 0; i < n; i++ {
			v := 1.0 / (eps + z[i])
			tmp[i] = v
			if v < imin {
				imin = v
			}
			if v > imax {
				imax = v
			}
		}
		den := imax - imin
		if den < 1e-6 {
			den = 1
		}
		for i := 0; i < n; i++ {
			u := (tmp[i] - imin) / den // 0..1
			d := dmax * (2 * (u - convergeU))
			if d < -dmax {
				d = -dmax
			} else if d > dmax {
				d = dmax
			}
			disp[i] = d
		}
	default: // "linear"
		for i := 0; i < n; i++ {
			u := z[i] // 0..1
			d := dmax * (2 * (u - convergeU))
			if d < -dmax {
				d = -dmax
			} else if d > dmax {
				d = dmax
			}
			disp[i] = d
		}
	}
}

/*** Forward splatting with Z-buffer and feathering ***/

// eyeSign: +0.5 for left, -0.5 for right (each gets half the disparity)
func forwardSplatEye(src *image.RGBA, disp, z []float32, eyeSign float32, dst *image.RGBA, touched []uint8, feather float64, workers int) {
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	const big = float32(1e9)

	// Per-eye Z-buffer (smaller z => nearer if we use z directly after shaping; we want foreground to win)
	zbuf := make([]float32, w*h)
	for i := range zbuf {
		zbuf[i] = big
	}
	// Accumulation buffers to avoid dimming and fill gaps
	colSum := make([]float32, w*h*4)
	wSum := make([]float32, w*h)
	for i := range touched {
		touched[i] = 0
	}

	rows := splitRows(h, workers)
	var wg sync.WaitGroup
	for _, r := range rows {
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			srcPix := src.Pix
			sStride := src.Stride

			for y := y0; y < y1; y++ {
				for x := 0; x < w; x++ {
					i := y*w + x
					shift := eyeSign * disp[i] // signed px
					sx := float32(x) + shift

					// Distribute into floor/ceil of sx with linear weights
					cx := int(math.Floor(float64(sx)))
					frac := float32(sx) - float32(cx)
					if cx < 0 || cx >= w {
						continue
					}

					srcI := y*sStride + x*4
					col := [4]uint8{
						srcPix[srcI+0],
						srcPix[srcI+1],
						srcPix[srcI+2],
						srcPix[srcI+3],
					}

					zv := z[i]

					// write helper accumulates weighted color and normalizes later
					write := func(tx int, wgt float32) {
						if tx < 0 || tx >= w || wgt <= 0 {
							return
						}
						pi := y*w + tx
						// Depth-aware accumulation: if significantly nearer, reset accumulators
						const zEps = 1e-4
						if zv < zbuf[pi]-zEps {
							colSum[pi*4+0] = 0
							colSum[pi*4+1] = 0
							colSum[pi*4+2] = 0
							colSum[pi*4+3] = 0
							wSum[pi] = 0
							zbuf[pi] = zv
						}
						// Accept if within epsilon of current front layer
						if zv <= zbuf[pi]+zEps {
							// Optional seam feather via alpha-scaled weight
							fw := wgt
							if feather > 0 {
								fw *= (1 - float32(math.Min(1.0, float64(frac)*feather)))
							}
							colSum[pi*4+0] += float32(col[0]) * fw
							colSum[pi*4+1] += float32(col[1]) * fw
							colSum[pi*4+2] += float32(col[2]) * fw
							colSum[pi*4+3] += float32(col[3]) * fw
							wSum[pi] += fw
							if touched[pi] == 0 {
								touched[pi] = 1
							}
						}
					}
					// 3-tap Gaussian weights around sx to reduce tearing
					sigma := float32(0.5)
					s2 := 2 * sigma * sigma
					ws := [3]float32{}
					txs := [3]int{cx - 1, cx, cx + 1}
					var sumW float32
					for k := 0; k < 3; k++ {
						// distance from subpixel position to tap center (in pixels)
						d := float32(txs[k]-cx) - frac
						wgt := float32(math.Exp(-float64((d * d) / s2)))
						ws[k] = wgt
						sumW += wgt
					}
					if sumW <= 1e-6 {
						continue
					}
					inv := 1.0 / sumW
					for k := 0; k < 3; k++ {
						wgt := ws[k] * float32(inv)
						write(txs[k], wgt)
					}
				}
			}
		}(r[0], r[1])
	}
	wg.Wait()

	// Normalize accumulated colors into dst image
	dstPix := dst.Pix
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pi := y*w + x
			ws := wSum[pi]
			di := y*dst.Stride + x*4
			if ws > 1e-6 {
				dstPix[di+0] = uint8(colSum[pi*4+0]/ws + 0.5)
				dstPix[di+1] = uint8(colSum[pi*4+1]/ws + 0.5)
				dstPix[di+2] = uint8(colSum[pi*4+2]/ws + 0.5)
				dstPix[di+3] = 255
			} else {
				// leave as zero; hole to be inpainted later
				dstPix[di+0] = 0
				dstPix[di+1] = 0
				dstPix[di+2] = 0
				dstPix[di+3] = 0
			}
		}
	}
}

func clearRGBA(img *image.RGBA) {
	for i := range img.Pix {
		img.Pix[i] = 0
	}
}

/*** Push–pull fill (fast hole inpainting) ***/
type level struct {
	w, h int
	pix  []float32 // RGBA packed as 4 channels
	msk  []float32 // 0..1 coverage
}

func pushPullFill(img *image.RGBA, touched []uint8, passes int) {
	if passes <= 0 {
		return
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	L := []level{}
	// Build finest level
	fin := level{w: w, h: h, pix: make([]float32, w*h*4), msk: make([]float32, w*h)}
	for i := 0; i < w*h; i++ {
		fin.pix[i*4+0] = float32(img.Pix[i*4+0])
		fin.pix[i*4+1] = float32(img.Pix[i*4+1])
		fin.pix[i*4+2] = float32(img.Pix[i*4+2])
		fin.pix[i*4+3] = float32(img.Pix[i*4+3])
		if touched[i] != 0 {
			fin.msk[i] = 1
		} else {
			fin.msk[i] = 0
		}
	}
	L = append(L, fin)
	// Downsample
	for p := 0; p < passes; p++ {
		pw := (L[p].w + 1) / 2
		ph := (L[p].h + 1) / 2
		nl := level{w: pw, h: ph, pix: make([]float32, pw*ph*4), msk: make([]float32, pw*ph)}
		downsampleLevel(&L[p], &nl)
		L = append(L, nl)
	}
	// Up-sample & fill
	for p := len(L) - 2; p >= 0; p-- {
		upsampleAccumulate(&L[p+1], &L[p])
	}
	// Write back
	for i := 0; i < w*h; i++ {
		img.Pix[i*4+0] = uint8(clampF(L[0].pix[i*4+0], 0, 255))
		img.Pix[i*4+1] = uint8(clampF(L[0].pix[i*4+1], 0, 255))
		img.Pix[i*4+2] = uint8(clampF(L[0].pix[i*4+2], 0, 255))
		img.Pix[i*4+3] = 255
	}
}

func downsampleLevel(src, dst *level) {
	for y := 0; y < dst.h; y++ {
		for x := 0; x < dst.w; x++ {
			sumC := [4]float32{}
			sumM := float32(0)
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					sx := x*2 + dx
					sy := y*2 + dy
					if sx >= src.w || sy >= src.h {
						continue
					}
					i := sy*src.w + sx
					m := src.msk[i]
					sumM += m
					for c := 0; c < 4; c++ {
						sumC[c] += src.pix[i*4+c] * m
					}
				}
			}
			j := y*dst.w + x
			if sumM > 0 {
				for c := 0; c < 4; c++ {
					dst.pix[j*4+c] = sumC[c] / sumM
				}
				dst.msk[j] = 1
			} else {
				dst.pix[j*4+0] = 0
				dst.pix[j*4+1] = 0
				dst.pix[j*4+2] = 0
				dst.pix[j*4+3] = 0
				dst.msk[j] = 0
			}
		}
	}
}

func upsampleAccumulate(lo, hi *level) {
	for y := 0; y < hi.h; y++ {
		for x := 0; x < hi.w; x++ {
			i := y*hi.w + x
			if hi.msk[i] >= 0.5 {
				// keep existing
				continue
			}
			// bilinear from low
			fx := (float32(x)+0.5)/float32(hi.w)*float32(lo.w) - 0.5
			fy := (float32(y)+0.5)/float32(hi.h)*float32(lo.h) - 0.5
			x0 := int(math.Floor(float64(fx)))
			y0 := int(math.Floor(float64(fy)))
			tx := fx - float32(x0)
			ty := fy - float32(y0)
			sample := func(xx, yy int) (c [4]float32, m float32) {
				if xx < 0 {
					xx = 0
				}
				if yy < 0 {
					yy = 0
				}
				if xx >= lo.w {
					xx = lo.w - 1
				}
				if yy >= lo.h {
					yy = lo.h - 1
				}
				j := yy*lo.w + xx
				for k := 0; k < 4; k++ {
					c[k] = lo.pix[j*4+k]
				}
				return c, lo.msk[j]
			}
			c00, m00 := sample(x0, y0)
			c10, m10 := sample(x0+1, y0)
			c01, m01 := sample(x0, y0+1)
			c11, m11 := sample(x0+1, y0+1)

			w00 := (1 - tx) * (1 - ty)
			w10 := tx * (1 - ty)
			w01 := (1 - tx) * ty
			w11 := tx * ty

			sumM := w00*m00 + w10*m10 + w01*m01 + w11*m11
			if sumM > 1e-5 {
				var out [4]float32
				for k := 0; k < 4; k++ {
					out[k] = (w00*c00[k]*m00 + w10*c10[k]*m10 + w01*c01[k]*m01 + w11*c11[k]*m11) / sumM
				}
				for k := 0; k < 4; k++ {
					hi.pix[i*4+k] = out[k]
				}
				hi.msk[i] = 1
			}
		}
	}
}

/*** Joint-bilateral (approx) on depth guided by color luma ***/
func jointBilateralGray(depth *image.Gray, rgba *image.RGBA, radius int, sigmaC float64, workers int) *image.Gray {
	if radius <= 0 {
		return depth
	}
	w, h := depth.Bounds().Dx(), depth.Bounds().Dy()
	out := image.NewGray(image.Rect(0, 0, w, h))
	luma := make([]uint8, w*h)
	// precompute luma
	for y := 0; y < h; y++ {
		r := rgba.Pix[y*rgba.Stride : y*rgba.Stride+w*4]
		for x := 0; x < w; x++ {
			i := x * 4
			Y := 0.299*float64(r[i+0]) + 0.587*float64(r[i+1]) + 0.114*float64(r[i+2])
			luma[y*w+x] = uint8(Y + 0.5)
		}
	}

	sigC2 := 2 * sigmaC * sigmaC
	rows := splitRows(h, workers)
	var wg sync.WaitGroup
	for _, rr := range rows {
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				for x := 0; x < w; x++ {
					Yc := float64(luma[y*w+x])
					sumW, sumV := 0.0, 0.0
					for dy := -radius; dy <= radius; dy++ {
						yy := y + dy
						if yy < 0 || yy >= h {
							continue
						}
						for dx := -radius; dx <= radius; dx++ {
							xx := x + dx
							if xx < 0 || xx >= w {
								continue
							}
							// spatial weight (box → cheap). For more quality, add Gaussian on dx,dy.
							Ys := float64(luma[yy*w+xx])
							dc := Ys - Yc
							wc := math.Exp(-(dc * dc) / sigC2)
							sumW += wc
							sumV += wc * float64(depth.GrayAt(xx, yy).Y)
						}
					}
					if sumW > 0 {
						out.SetGray(x, y, color.Gray{Y: uint8(sumV/sumW + 0.5)})
					} else {
						out.SetGray(x, y, depth.GrayAt(x, y))
					}
				}
			}
		}(rr[0], rr[1])
	}
	wg.Wait()
	return out
}

/*** VR fisheye pre-distortion (inverse radial polynomial) ***/

func fisheyeWarp(src *image.RGBA, k1, k2, preScale, cxFrac, cyFrac float64, workers int) *image.RGBA {
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	cx := cxFrac * float64(w-1)
	cy := cyFrac * float64(h-1)
	// Use axis-aligned max for normalization to control magnification
	rx := math.Max(cx, float64(w-1)-cx)
	ry := math.Max(cy, float64(h-1)-cy)
	norm := math.Max(rx, ry)
	if norm == 0 {
		norm = 1
	}
	rows := splitRows(h, workers)
	var wg sync.WaitGroup
	for _, r := range rows {
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				for x := 0; x < w; x++ {
					dx := (float64(x) - cx) / norm
					dy := (float64(y) - cy) / norm
					dx *= preScale
					dy *= preScale
					rd := math.Hypot(dx, dy)
					var sx, sy float64
					if rd > 0 {
						rs := invertRadialPolynomial(rd, k1, k2)
						scale := rs / rd
						sx = cx + dx*scale*norm
						sy = cy + dy*scale*norm
					} else {
						sx = cx
						sy = cy
					}
					var rgba [4]uint8
					bilinearSampleRGBA(src.Pix, src.Stride, w, h, sx, sy, rgba[:])
					i := y*dst.Stride + x*4
					dst.Pix[i+0] = rgba[0]
					dst.Pix[i+1] = rgba[1]
					dst.Pix[i+2] = rgba[2]
					dst.Pix[i+3] = rgba[3]
				}
			}
		}(r[0], r[1])
	}
	wg.Wait()
	return dst
}

// invertRadialPolynomial numerically inverts rd = rs * (1 + k1*rs^2 + k2*rs^4)
// to find source radius rs for a given destination radius rd.
func invertRadialPolynomial(rd, k1, k2 float64) float64 {
	if rd <= 0 {
		return 0
	}
	rs := rd
	for i := 0; i < 8; i++ {
		r2 := rs * rs
		r4 := r2 * r2
		f := rs*(1+k1*r2+k2*r4) - rd
		fp := 1 + 3*k1*r2 + 5*k2*r4
		if fp == 0 {
			break
		}
		rs -= f / fp
		if rs < 0 {
			rs = 0
		}
		if math.Abs(f) < 1e-7 {
			break
		}
	}
	return rs
}

// applyVRPreset sets typical k1, k2, and scale values for known headsets.
// It lowercases the preset name and mutates the provided pointers.
func applyVRPreset(name string, k1, k2, scale *float64) {
	switch strings.ToLower(name) {
	case "quest3":
		// Mild barrel, slightly reduced scale to preserve FOV
		*k1 = -0.28
		*k2 = 0.06
		*scale = 0.95
	case "quest2":
		*k1 = -0.34
		*k2 = 0.09
		*scale = 0.93
	case "index":
		*k1 = -0.30
		*k2 = 0.08
		*scale = 0.95
	case "psvr2":
		*k1 = -0.32
		*k2 = 0.07
		*scale = 0.94
	default:
		// no-op
	}
}

/*** Core utils ***/

func loadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func savePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := png.Encoder{CompressionLevel: png.BestSpeed}
	return enc.Encode(f, img)
}

func toRGBA(img image.Image) *image.RGBA {
	if rgba, ok := img.(*image.RGBA); ok {
		return rgba
	}
	b := img.Bounds()
	dst := image.NewRGBA(b)
	draw.Draw(dst, b, img, b.Min, draw.Src)
	return dst
}

// simpleSBS implements a minimal SBS using depth-proportional horizontal shifts,
// inspired by the referenced Python snippet.
func simpleSBS(color *image.RGBA, depth *image.Gray, mode string, depthScale float64, invert bool) *image.RGBA {
	w, h := color.Bounds().Dx(), color.Bounds().Dy()
	out := image.NewRGBA(image.Rect(0, 0, w*2, h))

	// Fill both halves with the base image
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			si := y*color.Stride + x*4
			// left
			dl := y*out.Stride + x*4
			out.Pix[dl+0] = color.Pix[si+0]
			out.Pix[dl+1] = color.Pix[si+1]
			out.Pix[dl+2] = color.Pix[si+2]
			out.Pix[dl+3] = 255
			// right
			dr := y*out.Stride + (w+x)*4
			out.Pix[dr+0] = color.Pix[si+0]
			out.Pix[dr+1] = color.Pix[si+1]
			out.Pix[dr+2] = color.Pix[si+2]
			out.Pix[dr+3] = 255
		}
	}

	flip := 0
	if strings.ToLower(mode) != "parallel" {
		flip = w
	}
	scaling := depthScale / float64(w)

	// Generate shifted image: for each pixel, shift horizontally on the selected eye
	for y := 0; y < h; y++ {
		row := depth.Pix[y*depth.Stride : y*depth.Stride+w]
		for x := 0; x < w; x++ {
			// Read depth (use first channel)
			dv := float64(row[x])
			if invert {
				dv = 255.0 - dv
			}
			shift := int(dv * scaling)
			newX := x + shift
			if newX >= w {
				newX = w - 1
			}
			if newX < 0 {
				newX = 0
			}
			// Streak fill forward for simple occlusion handling
			for i := 0; i <= shift+10; i++ {
				tx := newX + i
				if tx >= w || tx < 0 {
					break
				}
				si := y*color.Stride + x*4
				dr := y*out.Stride + (flip+tx)*4
				out.Pix[dr+0] = color.Pix[si+0]
				out.Pix[dr+1] = color.Pix[si+1]
				out.Pix[dr+2] = color.Pix[si+2]
				out.Pix[dr+3] = 255
			}
		}
	}
	return out
}

// blurRGBA applies a simple separable box blur with odd radius.
func blurRGBA(src *image.RGBA, radius int, workers int) *image.RGBA {
	if radius < 1 {
		return src
	}
	if radius%2 == 0 {
		radius++
	}
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	tmp := image.NewRGBA(image.Rect(0, 0, w, h))
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	rows := splitRows(h, workers)
	var wg sync.WaitGroup
	// Horizontal pass
	for _, r := range rows {
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				// running sum
				var sum [4]int
				for x := -radius; x <= radius; x++ {
					xx := x
					if xx < 0 {
						xx = 0
					}
					if xx >= w {
						xx = w - 1
					}
					si := y*src.Stride + xx*4
					sum[0] += int(src.Pix[si+0])
					sum[1] += int(src.Pix[si+1])
					sum[2] += int(src.Pix[si+2])
					sum[3] += int(src.Pix[si+3])
				}
				win := 2*radius + 1
				for x := 0; x < w; x++ {
					di := y*tmp.Stride + x*4
					tmp.Pix[di+0] = uint8(sum[0] / win)
					tmp.Pix[di+1] = uint8(sum[1] / win)
					tmp.Pix[di+2] = uint8(sum[2] / win)
					tmp.Pix[di+3] = uint8(sum[3] / win)
					// slide window
					xl := x - radius
					xr := x + radius + 1
					if xl < 0 {
						xl = 0
					}
					if xr >= w {
						xr = w - 1
					}
					siL := y*src.Stride + xl*4
					siR := y*src.Stride + xr*4
					sum[0] += int(src.Pix[siR+0]) - int(src.Pix[siL+0])
					sum[1] += int(src.Pix[siR+1]) - int(src.Pix[siL+1])
					sum[2] += int(src.Pix[siR+2]) - int(src.Pix[siL+2])
					sum[3] += int(src.Pix[siR+3]) - int(src.Pix[siL+3])
				}
			}
		}(r[0], r[1])
	}
	wg.Wait()
	// Vertical pass
	rows = splitRows(w, workers)
	wg = sync.WaitGroup{}
	for _, r := range rows {
		wg.Add(1)
		go func(x0, x1 int) {
			defer wg.Done()
			for x := x0; x < x1; x++ {
				var sum [4]int
				for y := -radius; y <= radius; y++ {
					yy := y
					if yy < 0 {
						yy = 0
					}
					if yy >= h {
						yy = h - 1
					}
					si := yy*tmp.Stride + x*4
					sum[0] += int(tmp.Pix[si+0])
					sum[1] += int(tmp.Pix[si+1])
					sum[2] += int(tmp.Pix[si+2])
					sum[3] += int(tmp.Pix[si+3])
				}
				win := 2*radius + 1
				for y := 0; y < h; y++ {
					di := y*dst.Stride + x*4
					dst.Pix[di+0] = uint8(sum[0] / win)
					dst.Pix[di+1] = uint8(sum[1] / win)
					dst.Pix[di+2] = uint8(sum[2] / win)
					dst.Pix[di+3] = uint8(sum[3] / win)
					// slide window
					yt := y - radius
					yb := y + radius + 1
					if yt < 0 {
						yt = 0
					}
					if yb >= h {
						yb = h - 1
					}
					siT := yt*tmp.Stride + x*4
					siB := yb*tmp.Stride + x*4
					sum[0] += int(tmp.Pix[siB+0]) - int(tmp.Pix[siT+0])
					sum[1] += int(tmp.Pix[siB+1]) - int(tmp.Pix[siT+1])
					sum[2] += int(tmp.Pix[siB+2]) - int(tmp.Pix[siT+2])
					sum[3] += int(tmp.Pix[siB+3]) - int(tmp.Pix[siT+3])
				}
			}
		}(r[0], r[1])
	}
	wg.Wait()
	return dst
}

func bilinearSampleRGBA(pix []uint8, stride, w, h int, fx, fy float64, out []uint8) {
	x := int(math.Floor(fx))
	y := int(math.Floor(fy))
	tx := fx - float64(x)
	ty := fy - float64(y)
	if x < 0 || x >= w || y < 0 || y >= h {
		out[0], out[1], out[2], out[3] = 0, 0, 0, 255
		return
	}
	x1 := x + 1
	y1 := y + 1
	if x1 >= w {
		x1 = x
		tx = 0
	}
	if y1 >= h {
		y1 = y
		ty = 0
	}
	i00 := y*stride + x*4
	i10 := y*stride + x1*4
	i01 := y1*stride + x*4
	i11 := y1*stride + x1*4

	w00 := (1 - tx) * (1 - ty)
	w10 := tx * (1 - ty)
	w01 := (1 - tx) * ty
	w11 := tx * ty

	for c := 0; c < 4; c++ {
		val := w00*float64(pix[i00+c]) + w10*float64(pix[i10+c]) + w01*float64(pix[i01+c]) + w11*float64(pix[i11+c])
		if val < 0 {
			val = 0
		} else if val > 255 {
			val = 255
		}
		out[c] = uint8(val + 0.5)
	}
}

func splitRows(h, workers int) [][2]int {
	if workers < 1 {
		workers = 1
	}
	if workers > h {
		workers = h
	}
	rows := make([][2]int, 0, workers)
	step := h / workers
	start := 0
	for i := 0; i < workers; i++ {
		end := start + step
		if i == workers-1 {
			end = h
		}
		rows = append(rows, [2]int{start, end})
		start = end
	}
	return rows
}

func clamp01Ptr(p *float64) {
	if *p < 0 {
		*p = 0
	} else if *p > 1 {
		*p = 1
	}
}
func clampF(v, a, b float32) float32 {
	if v < a {
		return a
	}
	if v > b {
		return b
	}
	return v
}

/*** Optional: JPEG writer ***/
func saveJPEG(path string, img image.Image, quality int) error {
	if quality < 1 || quality > 100 {
		return errors.New("jpeg quality 1..100")
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return jpeg.Encode(f, img, &jpeg.Options{Quality: quality})
}

/*** (End) ***/

// depthWeightedBlurRGBA blurs the image and blends based on depth distance to focalU.
func depthWeightedBlurRGBA(src *image.RGBA, depth *image.Gray, radius int, focalU float64, invert bool, workers int) *image.RGBA {
	if radius < 1 {
		return src
	}
	// Create a blurred version
	blurred := blurRGBA(src, radius, workers)
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	out := image.NewRGBA(image.Rect(0, 0, w, h))
	// Precompute normalized depth
	sigma := 0.5 // wider DOF falloff; can expose as a flag later
	if sigma < 1e-6 {
		sigma = 1e-6
	}
	rows := splitRows(h, workers)
	var wg sync.WaitGroup
	for _, r := range rows {
		wg.Add(1)
		go func(y0, y1 int) {
			defer wg.Done()
			for y := y0; y < y1; y++ {
				depRow := depth.Pix[y*depth.Stride : y*depth.Stride+w]
				for x := 0; x < w; x++ {
					du := float64(depRow[x]) / 255.0
					if invert {
						du = 1.0 - du
					}
					// Weight towards blur the farther from focus
					d := du - focalU
					ad := math.Abs(d)
					wBlur := ad / sigma
					if wBlur < 0 {
						wBlur = 0
					}
					if wBlur > 1 {
						wBlur = 1
					}
					siS := y*src.Stride + x*4
					siB := y*blurred.Stride + x*4
					di := y*out.Stride + x*4
					for c := 0; c < 3; c++ {
						val := (1-wBlur)*float64(src.Pix[siS+c]) + wBlur*float64(blurred.Pix[siB+c])
						if val < 0 {
							val = 0
						} else if val > 255 {
							val = 255
						}
						out.Pix[di+c] = uint8(val + 0.5)
					}
					out.Pix[di+3] = 255
				}
			}
		}(r[0], r[1])
	}
	wg.Wait()
	return out
}
