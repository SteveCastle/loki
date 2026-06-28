package main

import (
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
		showVersion                      bool
	)
	flag.StringVar(&modelPath, "model", "", "Path to ONNX embedding model")
	flag.StringVar(&imagePath, "image", "", "Path to input image")
	flag.StringVar(&ortLibPath, "ort", "", "Path to onnxruntime shared library")
	flag.StringVar(&inputName, "input", "pixel_values", "Model input tensor name")
	flag.StringVar(&outputName, "output", "image_embeds", "Model output tensor name")
	flag.IntVar(&width, "width", 224, "Model input width")
	flag.IntVar(&height, "height", 224, "Model input height")
	flag.IntVar(&dim, "dim", 0, "Output embedding dimension (required)")
	flag.StringVar(&meanStr, "mean", "0.5,0.5,0.5", "Normalization mean RGB")
	flag.StringVar(&stdStr, "std", "0.5,0.5,0.5", "Normalization stddev RGB")
	flag.BoolVar(&showVersion, "version", false, "Print version and exit")
	flag.Parse()

	if showVersion {
		fmt.Println("embed local-build")
		os.Exit(0)
	}
	if modelPath == "" || imagePath == "" || dim <= 0 {
		fmt.Fprintln(os.Stderr, "Error: --model, --image and --dim are required")
		flag.Usage()
		os.Exit(2)
	}

	parse3 := func(s string) ([3]float32, error) {
		parts := strings.Split(s, ",")
		if len(parts) != 3 {
			return [3]float32{}, fmt.Errorf("expected 3 values, got %d", len(parts))
		}
		var out [3]float32
		for i := 0; i < 3; i++ {
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
		log.Fatalf("invalid --mean: %v", err)
	}
	std, err := parse3(stdStr)
	if err != nil {
		log.Fatalf("invalid --std: %v", err)
	}
	opts.NormalizeMeanRGB = mean
	opts.NormalizeStddevRGB = std

	vec, err := onnxtag.EmbedImage(modelPath, imagePath, opts, dim)
	if err != nil {
		log.Fatalf("embed failed: %v", err)
	}
	norm := embedvec.Normalize(vec)
	fmt.Println(base64.StdEncoding.EncodeToString(embedvec.Encode(norm)))
}
