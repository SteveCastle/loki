//go:build cgo

package onnxface

import (
	"errors"
	"fmt"
	"image"
	"math"
	"os"
	"strings"

	"golang.org/x/image/draw"

	ort "github.com/yalue/onnxruntime_go"
)

// DetectorConfig configures a detection session.
type DetectorConfig struct {
	ModelPath string
	// Kind selects the model family: "yunet" (photo faces, 5 landmarks,
	// fixed 640×640) or "yolo" (anime heads, no landmarks, dynamic input
	// letterboxed to InputSize). Default "yunet".
	Kind string
	// InputSize is the square letterbox size for "yolo" detectors (default
	// 640). YuNet's export is hard-fixed at 640 regardless.
	InputSize int
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

// NewDetector loads a detection model into a reusable session.
func NewDetector(cfg DetectorConfig) (*Detector, error) {
	if cfg.ModelPath == "" {
		return nil, errors.New("onnxface: detector ModelPath is required")
	}
	if cfg.Kind == "" {
		cfg.Kind = "yunet"
	}
	if cfg.Kind != "yunet" && cfg.Kind != "yolo" {
		return nil, fmt.Errorf("onnxface: unknown detector kind %q", cfg.Kind)
	}
	if cfg.InputSize <= 0 || cfg.Kind == "yunet" {
		cfg.InputSize = yunetInputSize // 640 for both by default; forced for YuNet
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

	inputs := []string{yunetInputName}
	outputs := yunetOutputNames()
	if cfg.Kind == "yolo" {
		inputs = []string{yoloInputName}
		outputs = []string{yoloOutputName}
	}
	session, err := ort.NewDynamicAdvancedSession(cfg.ModelPath, inputs, outputs, so)
	if err != nil {
		return nil, fmt.Errorf("onnxface: load detector: %w", err)
	}
	return &Detector{session: session, cfg: cfg}, nil
}

// letterbox scales img down to fit size×size (never up), places it top-left
// on a black canvas, and returns the canvas plus the applied scale.
func letterbox(img image.Image, size int) (*image.RGBA, float64) {
	b := img.Bounds()
	scale := fitWithin(b.Dx(), b.Dy(), size)
	scaledW := int(float64(b.Dx())*scale + 0.5)
	scaledH := int(float64(b.Dy())*scale + 0.5)
	canvas := image.NewRGBA(image.Rect(0, 0, size, size))
	if scale == 1 {
		draw.Draw(canvas, image.Rect(0, 0, b.Dx(), b.Dy()), img, b.Min, draw.Src)
	} else {
		draw.CatmullRom.Scale(canvas, image.Rect(0, 0, scaledW, scaledH), img, b, draw.Src, nil)
	}
	return canvas, scale
}

// Detect runs the detector on img and returns detections in original-image
// pixel coordinates, score-descending, with bboxes clamped to the image and
// faces smaller than MinSize dropped.
func (d *Detector) Detect(img image.Image) ([]Detection, error) {
	b := img.Bounds()
	origW, origH := b.Dx(), b.Dy()
	if origW == 0 || origH == 0 {
		return nil, errors.New("onnxface: empty image")
	}
	canvas, scale := letterbox(img, d.cfg.InputSize)

	var dets []Detection
	var err error
	if d.cfg.Kind == "yolo" {
		dets, err = d.detectYOLO(canvas)
	} else {
		dets, err = d.detectYuNet(canvas)
	}
	if err != nil {
		return nil, err
	}
	dets = nms(dets, d.cfg.NMSThreshold)

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

// detectYuNet feeds the letterboxed canvas as raw BGR 0..255 NCHW (OpenCV
// blobFromImage defaults) and decodes the 12 per-stride output maps.
func (d *Detector) detectYuNet(canvas *image.RGBA) ([]Detection, error) {
	size := d.cfg.InputSize
	data := make([]float32, 3*size*size)
	n := size * size
	i := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := canvas.RGBAAt(x, y)
			data[0*n+i] = float32(c.B)
			data[1*n+i] = float32(c.G)
			data[2*n+i] = float32(c.R)
			i++
		}
	}
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(size), int64(size)), data)
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
	return decodeYuNet(size, size, raws, d.cfg.ScoreThreshold), nil
}

// detectYOLO feeds the letterboxed canvas as RGB 0..1 NCHW (ultralytics
// convention) and decodes the [1,5,N] raw head.
func (d *Detector) detectYOLO(canvas *image.RGBA) ([]Detection, error) {
	size := d.cfg.InputSize
	data := make([]float32, 3*size*size)
	n := size * size
	i := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			c := canvas.RGBAAt(x, y)
			data[0*n+i] = float32(c.R) / 255
			data[1*n+i] = float32(c.G) / 255
			data[2*n+i] = float32(c.B) / 255
			i++
		}
	}
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(size), int64(size)), data)
	if err != nil {
		return nil, err
	}
	defer tensor.Destroy()

	outputs := []ort.Value{nil}
	if err := d.session.Run([]ort.Value{tensor}, outputs); err != nil {
		return nil, err
	}
	out := outputs[0]
	if out == nil {
		return nil, errors.New("onnxface: detector produced no output")
	}
	defer out.Destroy()
	t, ok := out.(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("onnxface: unexpected detector output type %T", out)
	}
	return decodeYOLO(t.GetData(), d.cfg.ScoreThreshold), nil
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
	// InputSize is the square crop edge the model expects (default 112 —
	// the landmark-aligned template; generic encoders on head crops use 224).
	InputSize int
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
	if cfg.InputSize <= 0 {
		cfg.InputSize = AlignSize
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

// Embed computes the identity embedding of an aligned InputSize² crop. The
// returned vector is raw — callers L2-normalize before storing/comparing.
func (r *Recognizer) Embed(aligned *image.NRGBA) ([]float32, error) {
	size := r.cfg.InputSize
	bb := aligned.Bounds()
	if bb.Dx() != size || bb.Dy() != size {
		return nil, fmt.Errorf("onnxface: aligned crop is %dx%d, want %dx%d", bb.Dx(), bb.Dy(), size, size)
	}
	n := size * size
	data := make([]float32, 3*n)
	bgr := strings.EqualFold(r.cfg.ColorOrder, "BGR")
	std := r.cfg.Std
	for c := 0; c < 3; c++ {
		if std[c] == 0 {
			std[c] = 1
		}
	}
	i := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
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
	tensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(size), int64(size)), data)
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

// PipelineSpec assembles a full detect→align→embed pipeline.
type PipelineSpec struct {
	Detector   DetectorConfig
	Recognizer RecognizerConfig
	// Secondary enables embedding fusion: both recognizers embed the same
	// crop and the L2-normalized vectors are concatenated with per-model
	// weights (the cosine of the concat equals the weighted average of the
	// per-model cosines). Used for the anime pipeline (DINOv2 + SigLIP).
	Secondary *RecognizerConfig
	// Weight/SecondaryWeight are the effective cosine-fusion weights
	// (default 1 / 0). Internally the parts are scaled by sqrt(weight) so
	// the dot product weights them linearly.
	Weight          float32
	SecondaryWeight float32
	// Align selects the crop strategy: "landmarks" (default — the 112×112
	// five-point warp; requires a landmark detector) or "bbox-expand" (an
	// expanded square head crop; the only option for landmark-less
	// detectors like the anime one).
	Align string
	// CropExpand is the bbox expansion factor for "bbox-expand" (default 1.5).
	CropExpand float32
}

// Pipeline combines a Detector and one or two Recognizers: decode → detect →
// align → embed(→fuse). Not safe for concurrent use — one Pipeline per worker.
type Pipeline struct {
	Det  *Detector
	Rec  *Recognizer
	Rec2 *Recognizer
	spec PipelineSpec
}

// NewPipeline builds all sessions; on error none are leaked.
func NewPipeline(spec PipelineSpec) (*Pipeline, error) {
	if spec.Align == "" {
		spec.Align = "landmarks"
	}
	if spec.Align != "landmarks" && spec.Align != "bbox-expand" {
		return nil, fmt.Errorf("onnxface: unknown align mode %q", spec.Align)
	}
	if spec.Align == "landmarks" && spec.Detector.Kind == "yolo" {
		return nil, errors.New("onnxface: yolo detectors emit no landmarks; use bbox-expand alignment")
	}
	if spec.CropExpand <= 0 {
		spec.CropExpand = 1.5
	}
	if spec.Weight <= 0 {
		spec.Weight = 1
	}
	det, err := NewDetector(spec.Detector)
	if err != nil {
		return nil, err
	}
	rec, err := NewRecognizer(spec.Recognizer)
	if err != nil {
		det.Close()
		return nil, err
	}
	p := &Pipeline{Det: det, Rec: rec, spec: spec}
	if spec.Secondary != nil {
		rec2, err := NewRecognizer(*spec.Secondary)
		if err != nil {
			det.Close()
			rec.Close()
			return nil, err
		}
		p.Rec2 = rec2
		if spec.SecondaryWeight <= 0 {
			p.spec.SecondaryWeight = 1
		}
	}
	return p, nil
}

// l2norm normalizes v in place (no-op on the zero vector) and returns it.
func l2norm(v []float32) []float32 {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return v
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
	return v
}

// embedCrop runs one recognizer on the detection using the pipeline's crop
// strategy, sized for that recognizer.
func (p *Pipeline) embedCrop(rec *Recognizer, img image.Image, det Detection) ([]float32, error) {
	var crop *image.NRGBA
	if p.spec.Align == "landmarks" {
		aligned, err := alignFace(img, det.Landmarks)
		if err != nil {
			return nil, err
		}
		crop = aligned
	} else {
		crop = cropExpanded(img, det, p.spec.CropExpand, rec.cfg.InputSize)
	}
	return rec.Embed(crop)
}

// Process runs the full pipeline on one image file. It returns the faces
// (embedding vectors raw for single-recognizer pipelines, or the weighted
// normalized concat for fused ones — callers L2-normalize either way) plus
// the image dimensions so callers can store relative bbox coordinates. A face
// that fails to align/embed is skipped rather than failing the whole image.
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
		vec, eerr := p.embedCrop(p.Rec, img, det)
		if eerr != nil {
			continue
		}
		if p.Rec2 != nil {
			vec2, e2 := p.embedCrop(p.Rec2, img, det)
			if e2 != nil {
				continue
			}
			// Weighted concat of unit vectors: scale each part by
			// sqrt(weight) so cos(concat) = Σ wᵢ·cosᵢ / Σ wᵢ.
			l2norm(vec)
			l2norm(vec2)
			w1 := float32(math.Sqrt(float64(p.spec.Weight)))
			w2 := float32(math.Sqrt(float64(p.spec.SecondaryWeight)))
			fused := make([]float32, 0, len(vec)+len(vec2))
			for _, x := range vec {
				fused = append(fused, x*w1)
			}
			for _, x := range vec2 {
				fused = append(fused, x*w2)
			}
			vec = fused
		}
		faces = append(faces, Face{Detection: det, Vec: vec})
	}
	return faces, imgW, imgH, nil
}

// Close releases all sessions.
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
	if p.Rec2 != nil {
		if e := p.Rec2.Close(); err == nil {
			err = e
		}
	}
	return err
}
