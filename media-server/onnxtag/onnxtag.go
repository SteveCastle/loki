//go:build cgo
// +build cgo

package onnxtag

import (
	"bufio"
	"errors"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
	"os"
	"sort"
	"strings"

	resize "github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
	"golang.org/x/image/draw"
)

// Options configures how the classifier runs.
type Options struct {
	// Path to the onnxruntime shared library (.dll/.so/.dylib). If empty, the
	// environment variable ONNXRUNTIME_SHARED_LIBRARY_PATH will be respected.
	ORTSharedLibraryPath string

	// Input and output tensor names in the model graph.
	InputName  string
	OutputName string

	// Image preprocessing settings
	InputWidth         int
	InputHeight        int
	NormalizeMeanRGB   [3]float32 // default {0,0,0}
	NormalizeStddevRGB [3]float32 // default {1,1,1}

	// Labels list (index -> class name). If nil/empty, classes will be named
	// as "class_<index>".
	Labels []string

	// Number of top classes to return. If <= 0, all classes are returned.
	TopK int

	// NumClasses must be set if Labels is empty, to size the output tensor.
	// If Labels is provided, NumClasses is ignored.
	NumClasses int

	// Interpolation filter name: "bicubic", "bilinear", "nearest", or "catmullrom".
	Interpolation string
	// Crop percent (0,1]; if < 1, center-crop a square of this percent of the
	// shorter image side prior to resizing. If 1, no crop.
	CropPct float32
	// Crop mode, currently only "center" is supported.
	CropMode string

	// If provided, only consider these output indices and map to given names.
	// Key: class index (0-based), Value: display name.
	SelectedClassNames map[int]string

	// InputLayout determines the tensor data and shape ordering.
	// Supported values: "NCHW" (default) or "NHWC".
	InputLayout string

	// ColorOrder determines channel order: "RGB" or "BGR".
	ColorOrder string

	// PixelRange selects scale before normalization: "0_1" or "0_255".
	// If "0_255", mean/std normalization is skipped to mirror some pipelines
	// (e.g., wd-tagger uses BGR uint8 -> float32 without /255).
	PixelRange string

	// If true, pad to a square on a white background before resize (letterbox),
	// instead of center cropping.
	PadToSquare bool

	// Optional wd-tagger-like postprocessing
	RatingIndices      []int
	GeneralIndices     []int
	CharacterIndices   []int
	GeneralThreshold   float32
	CharacterThreshold float32
}

// DefaultOptions returns a reasonable default configuration commonly used by
// image classification models. Default input tensor layout is NCHW with RGB.
func DefaultOptions() Options {
	return Options{
		InputName:          "input",
		OutputName:         "output",
		InputWidth:         224,
		InputHeight:        224,
		NormalizeMeanRGB:   [3]float32{0, 0, 0},
		NormalizeStddevRGB: [3]float32{1, 1, 1},
		TopK:               5,
		Interpolation:      "catmullrom",
		CropPct:            1.0,
		CropMode:           "center",
		InputLayout:        "NCHW",
		ColorOrder:         "RGB",
		PixelRange:         "0_1",
		PadToSquare:        false,
		GeneralThreshold:   0.35,
		CharacterThreshold: 0.85,
	}
}

// FromLabelsFile loads labels from a newline-delimited file.
func FromLabelsFile(labelsPath string) ([]string, error) {
	f, err := os.Open(labelsPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var labels []string
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		labels = append(labels, line)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(labels) == 0 {
		return nil, errors.New("labels file is empty")
	}
	return labels, nil
}

// ClassifyImage runs the ONNX model located at modelPath on the given image file
// and returns the top tag strings. The returned slice length will be min(TopK, numClasses)
// if TopK > 0, otherwise numClasses.
//
// Assumptions:
//   - Model expects float32 input with shape either [1, 3, H, W] (NCHW) or
//     [1, H, W, 3] (NHWC) according to Options.InputLayout
//   - Output is a single float32 tensor with shape [1, numClasses]
//   - If Options.Labels is provided, numClasses is inferred from its length; otherwise,
//     the output tensor will still be read and classes will be auto-named if labels are missing.
func ClassifyImage(modelPath, imagePath string, opts Options) ([]string, error) {
	if opts.InputWidth <= 0 || opts.InputHeight <= 0 {
		return nil, fmt.Errorf("invalid input size %dx%d", opts.InputWidth, opts.InputHeight)
	}
	if opts.InputName == "" || opts.OutputName == "" {
		return nil, errors.New("input and output names must be provided")
	}

	if opts.ORTSharedLibraryPath != "" {
		ort.SetSharedLibraryPath(opts.ORTSharedLibraryPath)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}

	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}
	defer ort.DestroyEnvironment()

	imgTensor, err := loadImageAsTensor(imagePath, opts)
	if err != nil {
		return nil, err
	}
	defer imgTensor.Destroy()

	// Determine number of classes. If labels provided, use that; otherwise, require NumClasses.
	numClasses := 0
	if len(opts.Labels) > 0 {
		numClasses = len(opts.Labels)
	} else if opts.NumClasses > 0 {
		numClasses = opts.NumClasses
	} else {
		return nil, errors.New("must provide either Labels or NumClasses in Options to size output tensor")
	}

	// helper to build session and output tensor for given classes
	build := func(classes int) (session *ort.AdvancedSession, out *ort.Tensor[float32], err error) {
		outShape := ort.NewShape(1, int64(classes))
		out, err = ort.NewEmptyTensor[float32](outShape)
		if err != nil {
			return
		}
		session, err = ort.NewAdvancedSession(
			modelPath,
			[]string{opts.InputName},
			[]string{opts.OutputName},
			[]ort.Value{imgTensor},
			[]ort.Value{out},
			nil,
		)
		return
	}

	session, outputTensor, err := build(numClasses)
	if err != nil {
		return nil, err
	}
	// Ensure destruction later
	defer func() {
		outputTensor.Destroy()
		session.Destroy()
	}()

	if err := session.Run(); err != nil {
		// Try to parse a dimension mismatch to discover expected class count
		if _, expected, ok := parseOutputClassesMismatch(err); ok && expected > 0 && expected != numClasses {
			// Recreate session with expected output size from the model
			outputTensor.Destroy()
			session.Destroy()
			session, outputTensor, err = build(expected)
			if err != nil {
				return nil, err
			}
			if err := session.Run(); err != nil {
				return nil, err
			}
		} else {
			return nil, err
		}
	}

	scores := outputTensor.GetData()

	// wd-style postprocessing if category indices are provided
	if (len(opts.GeneralIndices) > 0 || len(opts.CharacterIndices) > 0) && len(opts.Labels) > 0 {
		// Characters first
		characters := make([]scoredIndex, 0, len(opts.CharacterIndices))
		for _, idx := range opts.CharacterIndices {
			if idx >= 0 && idx < len(scores) {
				if scores[idx] >= opts.CharacterThreshold {
					characters = append(characters, scoredIndex{Index: idx, Score: scores[idx]})
				}
			}
		}
		sort.Slice(characters, func(i, j int) bool { return characters[i].Score > characters[j].Score })

		// General next
		general := make([]scoredIndex, 0, len(opts.GeneralIndices))
		for _, idx := range opts.GeneralIndices {
			if idx >= 0 && idx < len(scores) {
				if scores[idx] >= opts.GeneralThreshold {
					general = append(general, scoredIndex{Index: idx, Score: scores[idx]})
				}
			}
		}
		sort.Slice(general, func(i, j int) bool { return general[i].Score > general[j].Score })

		// Concatenate characters then general, include scores
		out := make([]string, 0, len(characters)+len(general))
		for _, ci := range characters {
			name := opts.Labels[ci.Index]
			out = append(out, fmt.Sprintf("%s:%.5f", name, ci.Score))
		}
		for _, gi := range general {
			name := opts.Labels[gi.Index]
			out = append(out, fmt.Sprintf("%s:%.5f", name, gi.Score))
		}
		return out, nil
	}

	// Fallback: top-K across all classes
	if len(opts.SelectedClassNames) > 0 {
		tags := topKSelected(scores, opts.SelectedClassNames, opts.Labels, opts.TopK)
		return tags, nil
	}
	tags := topK(scores, opts.Labels, opts.TopK)
	return tags, nil
}

func loadImageAsTensor(path string, opts Options) (ort.Value, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}

	// Flatten transparency over white background
	b := img.Bounds()
	whiteBG := image.NewRGBA(b)
	// fill white
	draw.Draw(whiteBG, b, &image.Uniform{C: color.White}, image.Point{}, draw.Src)
	draw.Draw(whiteBG, b, img, b.Min, draw.Over)

	// Either pad to square (letterbox) or center-crop
	var src image.Image
	if opts.PadToSquare {
		srcW := b.Dx()
		srcH := b.Dy()
		side := maxInt(srcW, srcH)
		square := image.NewRGBA(image.Rect(0, 0, side, side))
		// white background
		draw.Draw(square, square.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
		x0 := (side - srcW) / 2
		y0 := (side - srcH) / 2
		draw.Draw(square, image.Rect(x0, y0, x0+srcW, y0+srcH), whiteBG, b.Min, draw.Over)
		src = square
	} else {
		// Optional center crop
		src = whiteBG
		if opts.CropPct > 0 && opts.CropPct < 1.0 && strings.ToLower(opts.CropMode) == "center" {
			b2 := whiteBG.Bounds()
			srcW := b2.Dx()
			srcH := b2.Dy()
			side := int(float32(minInt(srcW, srcH)) * opts.CropPct)
			if side > 0 {
				x0 := b2.Min.X + (srcW-side)/2
				y0 := b2.Min.Y + (srcH-side)/2
				src = cropRect(whiteBG, image.Rect(x0, y0, x0+side, y0+side))
			}
		}
	}

	// Resize to target size with chosen resampling; use true Bicubic to match PIL
	var dst image.Image
	if strings.EqualFold(strings.TrimSpace(opts.Interpolation), "bicubic") {
		dst = resize.Resize(uint(opts.InputWidth), uint(opts.InputHeight), src, resize.Bicubic)
	} else {
		rgba := image.NewRGBA(image.Rect(0, 0, opts.InputWidth, opts.InputHeight))
		scaler := chooseScaler(opts.Interpolation)
		scaler.Scale(rgba, rgba.Bounds(), src, src.Bounds(), draw.Over, nil)
		dst = rgba
	}

	// Convert to float32 normalized RGB in desired layout
	numPixels := opts.InputWidth * opts.InputHeight
	data := make([]float32, 3*numPixels)

	// Avoid division by zero
	stdR := opts.NormalizeStddevRGB[0]
	stdG := opts.NormalizeStddevRGB[1]
	stdB := opts.NormalizeStddevRGB[2]
	if stdR == 0 {
		stdR = 1
	}
	if stdG == 0 {
		stdG = 1
	}
	if stdB == 0 {
		stdB = 1
	}

	layout := strings.ToUpper(strings.TrimSpace(opts.InputLayout))
	// Channel order
	colorOrder := strings.ToUpper(strings.TrimSpace(opts.ColorOrder))
	// Pixel range
	scaleToUnit := strings.EqualFold(strings.TrimSpace(opts.PixelRange), "0_1")

	if layout == "NHWC" {
		i := 0
		for y := 0; y < opts.InputHeight; y++ {
			for x := 0; x < opts.InputWidth; x++ {
				c := color.RGBAModel.Convert(dst.At(x, y)).(color.RGBA)
				r := float32(c.R)
				g := float32(c.G)
				b := float32(c.B)
				if scaleToUnit {
					r /= 255.0
					g /= 255.0
					b /= 255.0
					r = (r - opts.NormalizeMeanRGB[0]) / stdR
					g = (g - opts.NormalizeMeanRGB[1]) / stdG
					b = (b - opts.NormalizeMeanRGB[2]) / stdB
				}
				if colorOrder == "BGR" {
					data[i+0] = b
					data[i+1] = g
					data[i+2] = r
				} else { // RGB
					data[i+0] = r
					data[i+1] = g
					data[i+2] = b
				}
				i += 3
			}
		}
	} else { // NCHW
		rOff := 0
		gOff := numPixels
		bOff := 2 * numPixels
		idx := 0
		for y := 0; y < opts.InputHeight; y++ {
			for x := 0; x < opts.InputWidth; x++ {
				c := color.RGBAModel.Convert(dst.At(x, y)).(color.RGBA)
				r := float32(c.R)
				g := float32(c.G)
				b := float32(c.B)
				if scaleToUnit {
					r /= 255.0
					g /= 255.0
					b /= 255.0
					r = (r - opts.NormalizeMeanRGB[0]) / stdR
					g = (g - opts.NormalizeMeanRGB[1]) / stdG
					b = (b - opts.NormalizeMeanRGB[2]) / stdB
				}
				if colorOrder == "BGR" {
					data[bOff+idx] = r
					data[gOff+idx] = g
					data[rOff+idx] = b
				} else { // RGB
					data[rOff+idx] = r
					data[gOff+idx] = g
					data[bOff+idx] = b
				}
				idx++
			}
		}
	}

	var shape ort.Shape
	if layout == "NHWC" {
		shape = ort.NewShape(1, int64(opts.InputHeight), int64(opts.InputWidth), 3)
	} else {
		shape = ort.NewShape(1, 3, int64(opts.InputHeight), int64(opts.InputWidth))
	}
	tensor, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, err
	}
	return tensor, nil
}

func chooseScaler(name string) draw.Scaler {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "bicubic":
		return draw.CatmullRom
	case "bilinear":
		return draw.BiLinear
	case "nearest":
		return draw.NearestNeighbor
	case "catmullrom":
		fallthrough
	default:
		return draw.CatmullRom
	}
}

func cropRect(img image.Image, r image.Rectangle) image.Image {
	// Create a new RGBA image and draw the crop region into it
	dst := image.NewRGBA(image.Rect(0, 0, r.Dx(), r.Dy()))
	draw.NearestNeighbor.Scale(dst, dst.Bounds(), img, r, draw.Src, nil)
	return dst
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

type scoredIndex struct {
	Index int
	Score float32
}

func topK(scores []float32, labels []string, k int) []string {
	n := len(scores)
	if n == 0 {
		return []string{}
	}
	// Build scored indices
	arr := make([]scoredIndex, n)
	for i := 0; i < n; i++ {
		arr[i] = scoredIndex{Index: i, Score: scores[i]}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].Score > arr[j].Score })

	if k <= 0 || k > n {
		k = n
	}

	result := make([]string, 0, k)
	for i := 0; i < k; i++ {
		idx := arr[i].Index
		name := ""
		if len(labels) > 0 && idx < len(labels) {
			name = labels[idx]
		} else {
			name = fmt.Sprintf("class_%d", idx)
		}
		// Include score as suffix if probabilities are available and finite.
		score := arr[i].Score
		if !math.IsNaN(float64(score)) && !math.IsInf(float64(score), 0) {
			name = fmt.Sprintf("%s:%.5f", name, score)
		}
		result = append(result, name)
	}
	return result
}

func topKSelected(scores []float32, selected map[int]string, labels []string, k int) []string {
	if len(selected) == 0 {
		return topK(scores, labels, k)
	}
	// Collect selected indices with scores
	arr := make([]scoredIndex, 0, len(selected))
	for idx, name := range selected {
		_ = name // name used later
		if idx >= 0 && idx < len(scores) {
			arr = append(arr, scoredIndex{Index: idx, Score: scores[idx]})
		}
	}
	sort.Slice(arr, func(i, j int) bool { return arr[i].Score > arr[j].Score })
	if k <= 0 || k > len(arr) {
		k = len(arr)
	}
	result := make([]string, 0, k)
	for i := 0; i < k; i++ {
		idx := arr[i].Index
		name, ok := selected[idx]
		if !ok || name == "" {
			if len(labels) > 0 && idx < len(labels) {
				name = labels[idx]
			} else {
				name = fmt.Sprintf("class_%d", idx)
			}
		}
		score := arr[i].Score
		if !math.IsNaN(float64(score)) && !math.IsInf(float64(score), 0) {
			name = fmt.Sprintf("%s:%.5f", name, score)
		}
		result = append(result, name)
	}
	return result
}

// parseOutputClassesFromError attempts to extract the "Got: N" from an ORT error string
// like: "Got invalid dimensions for output: output for the following indices index: 1 Got: 10862 Expected: 10861"
func parseOutputClassesMismatch(err error) (got int, expected int, ok bool) {
	if err == nil {
		return 0, 0, false
	}
	s := err.Error()
	got = extractFirstIntAfter(s, "Got: ")
	expected = extractFirstIntAfter(s, "Expected: ")
	if got > 0 && expected > 0 {
		return got, expected, true
	}
	return 0, 0, false
}

func extractFirstIntAfter(s, key string) int {
	idx := strings.Index(s, key)
	if idx < 0 {
		return 0
	}
	rest := s[idx+len(key):]
	end := len(rest)
	for i := 0; i < len(rest); i++ {
		if rest[i] < '0' || rest[i] > '9' {
			end = i
			break
		}
	}
	if end == 0 {
		return 0
	}
	num := 0
	for i := 0; i < end; i++ {
		if rest[i] < '0' || rest[i] > '9' {
			break
		}
		num = num*10 + int(rest[i]-'0')
	}
	return num
}
