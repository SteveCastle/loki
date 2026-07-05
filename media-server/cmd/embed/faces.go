package main

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/onnxface"
)

// faceJSON is one detected face in the --faces output. Coordinates are
// RELATIVE to the image dimensions ([0,1]) so consumers can store them
// resolution-independently; vec is the base64 float32 embedding, L2-normalized.
type faceJSON struct {
	X         float32       `json:"x"`
	Y         float32       `json:"y"`
	W         float32       `json:"w"`
	H         float32       `json:"h"`
	Score     float32       `json:"score"`
	Landmarks [5][2]float32 `json:"landmarks"`
	Vec       string        `json:"vec"`
}

// facesResultJSON is the per-image --faces output line.
type facesResultJSON struct {
	ImageW int        `json:"image_w"`
	ImageH int        `json:"image_h"`
	Faces  []faceJSON `json:"faces"`
}

// facesFlags carries the faces-mode CLI configuration from main.
type facesFlags struct {
	detectModel string
	detectKind  string // "yunet" | "yolo"
	align       string // "landmarks" | "bbox-expand"
	cropExpand  float64
	recModel    string // recognizer model path (--model)
	dim         int
	faceInput   string
	faceOutput  string
	faceSize    int    // recognizer input size (112 landmarks, 224 crops)
	faceMean    string // RGB, 0..255 scale
	faceStd     string // RGB, 0..255 scale
	faceColor   string // "BGR" (SFace) or "RGB" (ArcFace-family)
	faceWeight  float64
	// optional secondary recognizer (embedding fusion, e.g. DINOv2+SigLIP)
	rec2Model  string
	face2Input string
	face2Out   string
	face2Size  int
	face2Mean  string
	face2Std   string
	face2Color string
	face2Dim   int
	face2Wt    float64
	minScore   float64
	minSize    int
	ortLib     string
	provider   string
	device     int
	threads    int
	serve      bool
	imagePath  string
}

// runFaces executes faces mode: one-shot (--image) or --serve (one image path
// per stdin line → one JSON result line, "ERR <msg>" on failure).
func runFaces(f facesFlags) error {
	if f.detectModel == "" || f.recModel == "" {
		return fmt.Errorf("--detect-model and --model are required in --faces mode")
	}
	if f.dim <= 0 {
		return fmt.Errorf("--dim is required in --faces mode")
	}
	mean, err := parseRGB(f.faceMean)
	if err != nil {
		return fmt.Errorf("invalid --face-mean: %w", err)
	}
	std, err := parseRGB(f.faceStd)
	if err != nil {
		return fmt.Errorf("invalid --face-std: %w", err)
	}

	spec := onnxface.PipelineSpec{
		Detector: onnxface.DetectorConfig{
			ModelPath:      f.detectModel,
			Kind:           f.detectKind,
			ScoreThreshold: float32(f.minScore),
			MinSize:        f.minSize,
			ORTLib:         f.ortLib,
			Provider:       f.provider,
			Device:         f.device,
			Threads:        f.threads,
		},
		Recognizer: onnxface.RecognizerConfig{
			ModelPath:  f.recModel,
			Dim:        f.dim,
			InputName:  f.faceInput,
			OutputName: f.faceOutput,
			InputSize:  f.faceSize,
			Mean:       mean,
			Std:        std,
			ColorOrder: f.faceColor,
			ORTLib:     f.ortLib,
			Provider:   f.provider,
			Device:     f.device,
			Threads:    f.threads,
		},
		Align:      f.align,
		CropExpand: float32(f.cropExpand),
		Weight:     float32(f.faceWeight),
	}
	if f.rec2Model != "" {
		mean2, err := parseRGB(f.face2Mean)
		if err != nil {
			return fmt.Errorf("invalid --face2-mean: %w", err)
		}
		std2, err := parseRGB(f.face2Std)
		if err != nil {
			return fmt.Errorf("invalid --face2-std: %w", err)
		}
		if f.face2Dim <= 0 {
			return fmt.Errorf("--face2-dim is required with --face2-model")
		}
		spec.Secondary = &onnxface.RecognizerConfig{
			ModelPath:  f.rec2Model,
			Dim:        f.face2Dim,
			InputName:  f.face2Input,
			OutputName: f.face2Out,
			InputSize:  f.face2Size,
			Mean:       mean2,
			Std:        std2,
			ColorOrder: f.face2Color,
			ORTLib:     f.ortLib,
			Provider:   f.provider,
			Device:     f.device,
			Threads:    f.threads,
		}
		spec.SecondaryWeight = float32(f.face2Wt)
	}
	pipe, err := onnxface.NewPipeline(spec)
	if err != nil {
		return err
	}
	defer pipe.Close()

	if !f.serve {
		if f.imagePath == "" {
			return fmt.Errorf("--image is required in one-shot --faces mode")
		}
		line, err := processFacesLine(pipe, f.imagePath)
		if err != nil {
			return err
		}
		fmt.Println(line)
		return nil
	}

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 0, 64*1024), 4*1024*1024) // tolerate long paths
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	fmt.Fprintln(out, "READY")
	if err := out.Flush(); err != nil {
		return err
	}
	for in.Scan() {
		path := strings.TrimSpace(in.Text())
		if path == "" {
			fmt.Fprintln(out, "ERR empty path")
			out.Flush()
			continue
		}
		line, err := processFacesLine(pipe, path)
		if err != nil {
			fmt.Fprintln(out, "ERR "+strings.ReplaceAll(err.Error(), "\n", " "))
			out.Flush()
			continue
		}
		fmt.Fprintln(out, line)
		out.Flush()
	}
	return in.Err()
}

// processFacesLine runs the pipeline on one image and renders the JSON line.
func processFacesLine(pipe *onnxface.Pipeline, imagePath string) (string, error) {
	faces, w, h, err := pipe.Process(imagePath)
	if err != nil {
		return "", err
	}
	res := facesResultJSON{ImageW: w, ImageH: h, Faces: make([]faceJSON, 0, len(faces))}
	fw, fh := float32(w), float32(h)
	for _, face := range faces {
		fj := faceJSON{
			X:     face.X / fw,
			Y:     face.Y / fh,
			W:     face.W / fw,
			H:     face.H / fh,
			Score: face.Score,
			Vec:   base64.StdEncoding.EncodeToString(embedvec.Encode(embedvec.Normalize(face.Vec))),
		}
		for k := range face.Landmarks {
			fj.Landmarks[k][0] = face.Landmarks[k][0] / fw
			fj.Landmarks[k][1] = face.Landmarks[k][1] / fh
		}
		res.Faces = append(res.Faces, fj)
	}
	b, err := json.Marshal(res)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// parseRGB parses "r,g,b" into a [3]float32.
func parseRGB(s string) ([3]float32, error) {
	parts := strings.Split(s, ",")
	if len(parts) != 3 {
		return [3]float32{}, fmt.Errorf("expected 3 values, got %d", len(parts))
	}
	var out [3]float32
	for i := range parts {
		var v float64
		if _, err := fmt.Sscanf(strings.TrimSpace(parts[i]), "%f", &v); err != nil {
			return [3]float32{}, err
		}
		out[i] = float32(v)
	}
	return out, nil
}
