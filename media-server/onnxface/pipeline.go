//go:build cgo

package onnxface

import (
	"errors"
	"fmt"
	"image"
	"os"
	"strings"

	"golang.org/x/image/draw"

	ort "github.com/yalue/onnxruntime_go"
)

// DetectorConfig configures the YuNet detection session.
type DetectorConfig struct {
	ModelPath string
	// ScoreThreshold drops detections below this confidence (default 0.7).
	ScoreThreshold float32
	// NMSThreshold is the IoU above which overlapping boxes are suppressed
	// (default 0.3, matching OpenCV's FaceDetectorYN).
	NMSThreshold float32
	// MinSize drops faces whose bbox is smaller than this many pixels on
	// either edge, measured in ORIGINAL image coordinates (default 40).
	MinSize int

	ORTLib   string
	Provider string // "cpu" or "directml"
	Device   int
	Threads  int
}

const (
	defaultScoreThreshold = 0.7
	defaultNMSThreshold   = 0.3
	defaultMinSize        = 40
	yunetInputName        = "input"
	// yunetInputSize: the YuNet 2023mar ONNX export declares a FIXED
	// [1,3,640,640] input (verified empirically — ONNX Runtime rejects other
	// sizes; OpenCV's own DNN engine reshapes the graph, ORT can't). Every
	// image is therefore letterboxed into a 640×640 canvas. Consequence: very
	// small faces in very large photos may fall below detectability once
	// scaled — revisit with tiling or a dynamic-axes re-export if it bites.
	yunetInputSize = 640
)

// Detector is a persistent YuNet ONNX session. Not safe for concurrent use —
// one Detector per worker.
type Detector struct {
	session *ort.DynamicAdvancedSession
	cfg     DetectorConfig
}

// initORT points the binding at the shared library and initializes the
// (process-global, refcounted by us via IsInitialized) ONNX environment.
func initORT(ortLib string) error {
	if ortLib != "" {
		ort.SetSharedLibraryPath(ortLib)
	} else if p := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); p != "" {
		ort.SetSharedLibraryPath(p)
	}
	if ort.IsInitialized() {
		return nil
	}
	return ort.InitializeEnvironment()
}

// sessionOptions mirrors onnxtag's provider setup: DirectML needs sequential
// execution and no memory pattern; CPU takes thread counts.
func sessionOptions(provider string, threads, device int) (*ort.SessionOptions, error) {
	so, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	switch strings.ToLower(strings.TrimSpace(provider)) {
	case "directml":
		_ = so.SetExecutionMode(ort.ExecutionModeSequential)
		_ = so.SetMemPattern(false)
		if err := so.AppendExecutionProviderDirectML(device); err != nil {
			so.Destroy()
			return nil, fmt.Errorf("directml provider unavailable: %w", err)
		}
	default:
		if threads > 0 {
			_ = so.SetIntraOpNumThreads(threads)
			_ = so.SetInterOpNumThreads(threads)
		}
	}
	return so, nil
}

// NewDetector loads the YuNet model into a reusable session.
func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("onnxface: detector ModelPath is required")
	}
	if cfg.ScoreThreshold <= 0 {
		cfg.ScoreThreshold = defaultScoreThreshold
	}
	if cfg.NMSThreshold <= 0 {
		cfg.NMSThreshold = defaultNMSThreshold
	}
	if cfg.MinSize <= 0 {
		cfg.MinSize = defaultMinSize
	}
	if err := initORT(cfg.ORTLib); err != nil {
		return nil, err
	}
	so, err := sessionOptions(cfg.Provider, cfg.Threads, cfg.Device)
	if err != nil {
		return nil, err
	}
	defer so.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{yunetInputName},
		yunetOutputNames(),
		so,
	)
	if err != nil {
		return nil, fmt.Errorf("onnxface: load detector: %w", err)
	}
	return &Detector{session: session, cfg: cfg}, nil
}

// Detect runs YuNet on img and returns detections in original-image pixel
// coordinates, score-descending, with bboxes clamped to the image and faces
// smaller than MinSize dropped.
func (d *Detector) Detect(img image.Image) ([]Detection, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW == 0 || origH == 0 {
		return nil, errors.New("onnxface: empty image")
	}

	// Letterbox into the model's fixed 640×640 input: scale down to fit
	// (never up), place top-left, pad the rest with black.
	scale := fitWithin(origW, origH, yunetInputSize)
	scaledW := int(float64(origW)*scale + 0.5)
	scaledH := int(float64(origH)*scale + 0.5)
	padW, padH := yunetInputSize, yunetInputSize

	canvas := image.NewRGBA(image.Rect(0, 0, padW, padH))
	if scale == 1 {
		draw.Draw(canvas, image.Rect(0, 0, origW, origH), img, b.Min, draw.Src)
	} else {
		draw.CatmullRom.Scale(canvas, image.Rect(0, 0, scaledW, scaledH), img, b, draw.Src, nil)
	}

	// YuNet input: raw BGR 0..255, NCHW float32 (OpenCV blobFromImage defaults).
	data := make([]float32, 3*padW*padH)
	n := padW * padH
	i := 0
	for y := 0; y < padH; y++ {
		for x := 0; x < padW; x++ {
			c := canvas.RGBAAt(x, y)
			data[0*n+i] = float32(c.B)
			data[1*n+i] = float32(c.G)
			data[2*n+i] = float32(c.R)
			i++
		}
	}
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(padH), int64(padW)), data)
	if err != nil {
		return nil, err
	}
	defer tensor.Destroy()

	outputs := make([]ort.Value, 12)
	if err := d.session.Run([]ort.Value{tensor}, outputs); err != nil {
		return nil, err
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	raws := make([]yunetRaw, 0, len(yunetStrides))
	for si, stride := range yunetStrides {
		var maps [4][]float32
		for oi := 0; oi < 4; oi++ {
			out := outputs[si*4+oi]
			t, ok := out.(*ort.Tensor[float32])
			if !ok {
				return nil, fmt.Errorf("onnxface: detector output %d is %T, want float32 tensor", si*4+oi, out)
			}
			maps[oi] = t.GetData()
		}
		raws = append(raws, yunetRaw{stride: stride, cls: maps[0], obj: maps[1], bbox: maps[2], kps: maps[3]})
	}

	dets := nms(decodeYuNet(padW, padH, raws, d.cfg.ScoreThreshold), d.cfg.NMSThreshold)

	// Map back to original coordinates, clamp, and apply the min-size gate.
	inv := float32(1 / scale)
	minSize := float32(d.cfg.MinSize)
	kept := dets[:0]
	for _, det := range dets {
		det.X *= inv
		det.Y *= inv
		det.W *= inv
		det.H *= inv
		for k := range det.Landmarks {
			det.Landmarks[k][0] *= inv
			det.Landmarks[k][1] *= inv
		}
		det = clampToImage(det, float32(origW), float32(origH))
		if det.W < minSize || det.H < minSize {
			continue
		}
		kept = append(kept, det)
	}
	return kept, nil
}

// clampToImage clips a detection's bbox to [0,w]×[0,h] (landmarks may sit
// outside the box and are left as-is; the aligner tolerates them).
func clampToImage(d Detection, w, h float32) Detection {
	x2 := minf(d.X+d.W, w)
	y2 := minf(d.Y+d.H, h)
	d.X = maxf(d.X, 0)
	d.Y = maxf(d.Y, 0)
	d.W = maxf(x2-d.X, 0)
	d.H = maxf(y2-d.Y, 0)
	return d
}

// Close releases the detector session.
func (d *Detector) Close() error {
	if d.session != nil {
		err := d.session.Destroy()
		d.session = nil
		return err
	}
	return nil
}

// RecognizerConfig configures the face-identity embedding session. The
// defaults suit SFace (OpenCV Zoo); BYO research models (ArcFace/AdaFace
// exports) override tensor names, dimension, and preprocessing.
type RecognizerConfig struct {
	ModelPath  string
	Dim        int    // embedding dimension (SFace 128, ArcFace/AdaFace 512)
	InputName  string // default "data"
	OutputName string // default "fc1"
	// Mean/Std are applied per channel on the 0..255 pixel scale:
	// v = (pixel - Mean) / Std. SFace wants raw pixels (Mean 0, Std 1);
	// ArcFace-family models want Mean 127.5, Std 127.5.
	Mean, Std [3]float32
	// ColorOrder is "BGR" (SFace/OpenCV convention, the default) or "RGB"
	// (ArcFace-family exports).
	ColorOrder string

	ORTLib   string
	Provider string
	Device   int
	Threads  int
}

// Recognizer is a persistent face-embedding ONNX session. Not safe for
// concurrent use.
type Recognizer struct {
	session *ort.DynamicAdvancedSession
	cfg     RecognizerConfig
}

// NewRecognizer loads the recognizer model into a reusable session.
func NewRecognizer(cfg RecognizerConfig) (*Recognizer, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("onnxface: recognizer ModelPath is required")
	}
	if cfg.Dim <= 0 {
		return nil, errors.New("onnxface: recognizer Dim must be > 0")
	}
	if cfg.InputName == "" {
		cfg.InputName = "data"
	}
	if cfg.OutputName == "" {
		cfg.OutputName = "fc1"
	}
	if cfg.Std == ([3]float32{}) {
		cfg.Std = [3]float32{1, 1, 1}
	}
	if cfg.ColorOrder == "" {
		cfg.ColorOrder = "BGR"
	}
	if err := initORT(cfg.ORTLib); err != nil {
		return nil, err
	}
	so, err := sessionOptions(cfg.Provider, cfg.Threads, cfg.Device)
	if err != nil {
		return nil, err
	}
	defer so.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		cfg.ModelPath,
		[]string{cfg.InputName},
		[]string{cfg.OutputName},
		so,
	)
	if err != nil {
		return nil, fmt.Errorf("onnxface: load recognizer: %w", err)
	}
	return &Recognizer{session: session, cfg: cfg}, nil
}

// Embed computes the identity embedding of an aligned 112×112 crop. The
// returned vector is raw — callers L2-normalize before storing/comparing.
func (r *Recognizer) Embed(aligned *image.NRGBA) ([]float32, error) {
	bb := aligned.Bounds()
	if bb.Dx() != AlignSize || bb.Dy() != AlignSize {
		return nil, fmt.Errorf("onnxface: aligned crop is %dx%d, want %dx%d", bb.Dx(), bb.Dy(), AlignSize, AlignSize)
	}
	n := AlignSize * AlignSize
	data := make([]float32, 3*n)
	bgr := strings.EqualFold(r.cfg.ColorOrder, "BGR")
	std := r.cfg.Std
	for c := 0; c < 3; c++ {
		if std[c] == 0 {
			std[c] = 1
		}
	}
	i := 0
	for y := 0; y < AlignSize; y++ {
		for x := 0; x < AlignSize; x++ {
			px := aligned.NRGBAAt(x, y)
			// Mean/Std are declared in RGB order regardless of tensor layout.
			rv := (float32(px.R) - r.cfg.Mean[0]) / std[0]
			gv := (float32(px.G) - r.cfg.Mean[1]) / std[1]
			bv := (float32(px.B) - r.cfg.Mean[2]) / std[2]
			if bgr {
				data[0*n+i] = bv
				data[1*n+i] = gv
				data[2*n+i] = rv
			} else {
				data[0*n+i] = rv
				data[1*n+i] = gv
				data[2*n+i] = bv
			}
			i++
		}
	}
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, AlignSize, AlignSize), data)
	if err != nil {
		return nil, err
	}
	defer tensor.Destroy()

	outputs := []ort.Value{nil}
	if err := r.session.Run([]ort.Value{tensor}, outputs); err != nil {
		return nil, err
	}
	out := outputs[0]
	if out == nil {
		return nil, errors.New("onnxface: recognizer produced no output")
	}
	defer out.Destroy()

	t, ok := out.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("onnxface: unexpected recognizer output type %T", out)
	}
	got := t.GetData()
	if len(got) < r.cfg.Dim {
		return nil, fmt.Errorf("onnxface: recognizer output has %d floats, need %d", len(got), r.cfg.Dim)
	}
	vec := make([]float32, r.cfg.Dim)
	copy(vec, got[:r.cfg.Dim])
	return vec, nil
}

// Close releases the recognizer session.
func (r *Recognizer) Close() error {
	if r.session != nil {
		err := r.session.Destroy()
		r.session = nil
		return err
	}
	return nil
}

// Pipeline combines a Detector and a Recognizer: decode → detect → align →
// embed. Not safe for concurrent use — one Pipeline per worker.
type Pipeline struct {
	Det *Detector
	Rec *Recognizer
}

// NewPipeline builds both sessions; on error neither is leaked.
func NewPipeline(dc DetectorConfig, rc RecognizerConfig) (*Pipeline, error) {
	det, err := NewDetector(dc)
	if err != nil {
		return nil, err
	}
	rec, err := NewRecognizer(rc)
	if err != nil {
		det.Close()
		return nil, err
	}
	return &Pipeline{Det: det, Rec: rec}, nil
}

// Process runs the full pipeline on one image file. It returns the faces
// (embedding vectors raw/un-normalized) plus the image dimensions so callers
// can store relative bbox coordinates. A face that fails to align/embed is
// skipped rather than failing the whole image.
func (p *Pipeline) Process(imagePath string) (faces []Face, imgW, imgH int, err error) {
	img, err := decodeImageFile(imagePath)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("decode %s: %w", imagePath, err)
	}
	b := img.Bounds()
	imgW, imgH = b.Dx(), b.Dy()

	dets, err := p.Det.Detect(img)
	if err != nil {
		return nil, imgW, imgH, err
	}
	for _, det := range dets {
		aligned, aerr := alignFace(img, det.Landmarks)
		if aerr != nil {
			continue
		}
		vec, eerr := p.Rec.Embed(aligned)
		if eerr != nil {
			continue
		}
		faces = append(faces, Face{Detection: det, Vec: vec})
	}
	return faces, imgW, imgH, nil
}

// Close releases both sessions.
func (p *Pipeline) Close() error {
	var err error
	if p.Det != nil {
		err = p.Det.Close()
	}
	if p.Rec != nil {
		if e := p.Rec.Close(); err == nil {
			err = e
		}
	}
	return err
}
