package main

import (
	"bufio"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/stevecastle/shrike/embedvec"
	"github.com/stevecastle/shrike/onnxtag"
)

func main() {
	var (
		modelPath, imagePath, ortLibPath string
		inputName, outputName            string
		width, height, dim               int
		meanStr, stdStr                  string
		pooling, cropMode                string
		cropPct                          float64
		showVersion                      bool
		serve                            bool
		provider                         string
		device, threads                  int
		// text mode flags
		textStr, textModel, tokenizerPath string
		textInput, textOutput             string
		seqLen                            int
		// faces mode flags
		facesMode                    bool
		detectModel, detectKind      string
		alignMode                    string
		cropExpand                   float64
		faceInput, faceOutput        string
		faceSize                     int
		faceMean, faceStd, faceColor string
		faceWeight                   float64
		face2Model, face2Input       string
		face2Output                  string
		face2Size, face2Dim          int
		face2Mean, face2Std          string
		face2Color                   string
		face2Weight                  float64
		faceMinScore                 float64
		faceMinSize                  int
	)
	flag.StringVar(&modelPath, "model", "", "Path to ONNX embedding model")
	flag.StringVar(&imagePath, "image", "", "Path to input image")
	flag.StringVar(&ortLibPath, "ort", "", "Path to onnxruntime shared library")
	flag.StringVar(&inputName, "input", "pixel_values", "Model input tensor name")
	flag.StringVar(&outputName, "output", "pooler_output", "Model output tensor name (SigLIP 2 vision/text encoders both emit pooler_output)")
	flag.IntVar(&width, "width", 224, "Model input width")
	flag.IntVar(&height, "height", 224, "Model input height")
	flag.IntVar(&dim, "dim", 0, "Output embedding dimension (required)")
	flag.StringVar(&meanStr, "mean", "0.5,0.5,0.5", "Normalization mean RGB")
	flag.StringVar(&stdStr, "std", "0.5,0.5,0.5", "Normalization stddev RGB")
	flag.StringVar(&pooling, "pooling", "none", "Output pooling: \"none\" (output is already [1,dim]) or \"cls\" (take token 0 of a [1,N,dim] sequence output, e.g. DINOv2)")
	flag.Float64Var(&cropPct, "crop-pct", 1.0, "Center-crop fraction before resize (1.0 disables; e.g. 0.875 for DINOv2)")
	flag.StringVar(&cropMode, "crop-mode", "", "Crop mode: \"center\" enables center crop when --crop-pct < 1")
	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
	// serve mode (persistent worker): load the model once, then read one image
	// path per line on stdin and write one base64 vector (or "ERR <msg>") per
	// line on stdout. Exits on EOF.
	flag.BoolVar(&serve, "serve", false, "Persistent worker: stream image paths on stdin, vectors on stdout")
	flag.StringVar(&provider, "provider", "cpu", "Execution provider: \"cpu\" or \"directml\" (GPU)")
	flag.IntVar(&device, "device", 0, "GPU device id for --provider=directml")
	flag.IntVar(&threads, "threads", 0, "CPU intra-op threads (0 = ONNX Runtime default)")
	// text mode
	flag.StringVar(&textStr, "text", "", "Text to encode (enables text mode)")
	flag.StringVar(&textModel, "text-model", "", "Path to ONNX text encoder model (required in text mode)")
	flag.StringVar(&tokenizerPath, "tokenizer", "", "Path to SentencePiece tokenizer.model (required in text mode)")
	flag.StringVar(&textInput, "text-input", "input_ids", "Text encoder input tensor name")
	flag.StringVar(&textOutput, "text-output", "pooler_output", "Text encoder output tensor name")
	flag.IntVar(&seqLen, "seq-len", 64, "Sequence length for text tokenization")
	// faces mode: detect faces (YuNet), align to the 112x112 five-landmark
	// template, and embed each with a face-identity model (SFace by default;
	// BYO ArcFace-family models via the --face-* preprocessing flags). Output
	// is one JSON line per image; --model is the RECOGNIZER model path here.
	flag.BoolVar(&facesMode, "faces", false, "Face mode: detect+align+embed faces; --model is the recognizer, --detect-model the detector")
	flag.StringVar(&detectModel, "detect-model", "", "Path to the face/head detection ONNX model (faces mode)")
	flag.StringVar(&detectKind, "detect-kind", "yunet", "Detector family: yunet (photo faces, landmarks) or yolo (anime heads)")
	flag.StringVar(&alignMode, "align", "landmarks", "Crop strategy: landmarks (112 template warp) or bbox-expand (expanded head crop)")
	flag.Float64Var(&cropExpand, "crop-expand", 1.5, "Bbox expansion factor for --align=bbox-expand")
	flag.StringVar(&faceInput, "face-input", "data", "Recognizer input tensor name (faces mode)")
	flag.StringVar(&faceOutput, "face-output", "fc1", "Recognizer output tensor name (faces mode)")
	flag.IntVar(&faceSize, "face-size", 112, "Recognizer input crop size (112 landmark models, 224 generic encoders)")
	flag.StringVar(&faceMean, "face-mean", "0,0,0", "Recognizer per-channel RGB mean on the 0..255 scale (127.5 for ArcFace-family)")
	flag.StringVar(&faceStd, "face-std", "1,1,1", "Recognizer per-channel RGB stddev on the 0..255 scale (127.5 for ArcFace-family)")
	flag.StringVar(&faceColor, "face-color", "BGR", "Recognizer channel order: BGR (SFace) or RGB (ArcFace-family)")
	flag.Float64Var(&faceWeight, "face-weight", 1, "Primary recognizer's cosine-fusion weight")
	// Secondary recognizer: embedding fusion (e.g. DINOv2 + SigLIP on anime
	// head crops). Vectors are L2-normalized and weight-concatenated.
	flag.StringVar(&face2Model, "face2-model", "", "Optional secondary recognizer ONNX model (embedding fusion)")
	flag.StringVar(&face2Input, "face2-input", "data", "Secondary recognizer input tensor name")
	flag.StringVar(&face2Output, "face2-output", "fc1", "Secondary recognizer output tensor name")
	flag.IntVar(&face2Size, "face2-size", 112, "Secondary recognizer input crop size")
	flag.IntVar(&face2Dim, "face2-dim", 0, "Secondary recognizer embedding dimension (required with --face2-model)")
	flag.StringVar(&face2Mean, "face2-mean", "0,0,0", "Secondary recognizer RGB mean (0..255 scale)")
	flag.StringVar(&face2Std, "face2-std", "1,1,1", "Secondary recognizer RGB stddev (0..255 scale)")
	flag.StringVar(&face2Color, "face2-color", "RGB", "Secondary recognizer channel order")
	flag.Float64Var(&face2Weight, "face2-weight", 1, "Secondary recognizer's cosine-fusion weight")
	flag.Float64Var(&faceMinScore, "min-score", 0.7, "Minimum detection confidence (faces mode)")
	flag.IntVar(&faceMinSize, "min-size", 40, "Minimum face bbox edge in original-image pixels (faces mode)")
	flag.Parse()

	if showVersion {
		fmt.Println("embed local-build")
		os.Exit(0)
	}

	if dim <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --dim is required")
		flag.Usage()
		os.Exit(2)
	}

	// Faces mode: detect + align + embed. Handled before image/text modes.
	if facesMode {
		err := runFaces(facesFlags{
			detectModel: detectModel,
			detectKind:  detectKind,
			align:       alignMode,
			cropExpand:  cropExpand,
			recModel:    modelPath,
			dim:         dim,
			faceInput:   faceInput,
			faceOutput:  faceOutput,
			faceSize:    faceSize,
			faceMean:    faceMean,
			faceStd:     faceStd,
			faceColor:   faceColor,
			faceWeight:  faceWeight,
			rec2Model:   face2Model,
			face2Input:  face2Input,
			face2Out:    face2Output,
			face2Size:   face2Size,
			face2Mean:   face2Mean,
			face2Std:    face2Std,
			face2Color:  face2Color,
			face2Dim:    face2Dim,
			face2Wt:     face2Weight,
			minScore:    faceMinScore,
			minSize:     faceMinSize,
			ortLib:      ortLibPath,
			provider:    provider,
			device:      device,
			threads:     threads,
			serve:       serve,
			imagePath:   imagePath,
		})
		if err != nil {
			log.Fatalf("faces: %v", err)
		}
		return
	}

	// Text mode: --text is non-empty
	if textStr != "" {
		if textModel == "" || tokenizerPath == "" {
			fmt.Fprintln(os.Stderr, "Error: --text-model and --tokenizer are required in text mode")
			flag.Usage()
			os.Exit(2)
		}
		opts := onnxtag.DefaultOptions()
		opts.InputName = textInput
		opts.OutputName = textOutput
		opts.ORTSharedLibraryPath = ortLibPath

		vec, err := onnxtag.EmbedText(textModel, tokenizerPath, textStr, opts, dim, seqLen)
		if err != nil {
			log.Fatalf("embed text failed: %v", err)
		}
		norm := embedvec.Normalize(vec)
		fmt.Println(base64.StdEncoding.EncodeToString(embedvec.Encode(norm)))
		return
	}

	// Image preprocessing options, shared by serve and one-shot modes.
	opts, err := buildImageOpts(inputName, outputName, width, height, ortLibPath, meanStr, stdStr, cropPct, cropMode)
	if err != nil {
		log.Fatalf("%v", err)
	}

	// Serve mode: load the model once and stream images. Image-only.
	if serve {
		if modelPath == "" {
			fmt.Fprintln(os.Stderr, "Error: --model is required in --serve mode")
			os.Exit(2)
		}
		if err := runServe(modelPath, opts, dim, pooling, provider, device, threads); err != nil {
			log.Fatalf("serve: %v", err)
		}
		return
	}

	// Image mode (existing one-shot behavior)
	if modelPath == "" || imagePath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model, --image and --dim are required")
		flag.Usage()
		os.Exit(2)
	}

	var vec []float32
	switch strings.ToLower(strings.TrimSpace(pooling)) {
	case "cls":
		vec, err = onnxtag.EmbedImageCLS(modelPath, imagePath, opts, dim)
	default: // "none"/"" — output is already a pooled [1,dim] vector
		vec, err = onnxtag.EmbedImage(modelPath, imagePath, opts, dim)
	}
	if err != nil {
		log.Fatalf("embed failed: %v", err)
	}
	norm := embedvec.Normalize(vec)
	fmt.Println(base64.StdEncoding.EncodeToString(embedvec.Encode(norm)))
}

// buildImageOpts assembles the image preprocessing Options from CLI flags. Used
// by both --serve and one-shot image modes.
func buildImageOpts(inputName, outputName string, width, height int, ortLibPath, meanStr, stdStr string, cropPct float64, cropMode string) (onnxtag.Options, error) {
	parse3 := func(s string) ([3]float32, error) {
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

	opts := onnxtag.DefaultOptions()
	opts.InputName = inputName
	opts.OutputName = outputName
	opts.InputWidth = width
	opts.InputHeight = height
	opts.ORTSharedLibraryPath = ortLibPath
	opts.InputLayout = "NCHW"
	opts.ColorOrder = "RGB"
	opts.PixelRange = "0_1"
	mean, err := parse3(meanStr)
	if err != nil {
		return opts, fmt.Errorf("invalid --mean: %w", err)
	}
	std, err := parse3(stdStr)
	if err != nil {
		return opts, fmt.Errorf("invalid --std: %w", err)
	}
	opts.NormalizeMeanRGB = mean
	opts.NormalizeStddevRGB = std
	opts.CropPct = float32(cropPct)
	opts.CropMode = cropMode
	return opts, nil
}

// runServe loads the model once and streams: one image path per line on stdin →
// one base64 vector (or "ERR <msg>") per line on stdout. The first output line
// is "READY", emitted after the model loads, so the parent can confirm startup.
func runServe(modelPath string, opts onnxtag.Options, dim int, pooling, provider string, device, threads int) error {
	emb, err := onnxtag.NewEmbedder(onnxtag.EmbedderConfig{
		ModelPath: modelPath,
		Opts:      opts,
		Dim:       dim,
		Pooling:   strings.ToLower(strings.TrimSpace(pooling)),
		Provider:  onnxtag.EmbedProvider(strings.ToLower(strings.TrimSpace(provider))),
		Threads:   threads,
		Device:    device,
		ORTLib:    opts.ORTSharedLibraryPath,
	})
	if err != nil {
		return err
	}
	defer emb.Close()

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
		vec, err := emb.Embed(path)
		if err != nil {
			fmt.Fprintln(out, "ERR "+strings.ReplaceAll(err.Error(), "\n", " "))
			out.Flush()
			continue
		}
		norm := embedvec.Normalize(vec)
		fmt.Fprintln(out, base64.StdEncoding.EncodeToString(embedvec.Encode(norm)))
		out.Flush()
	}
	return in.Err()
}
