package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/stevecastle/shrike/onnxtag"
)

func main() {
	var (
		modelPath     string
		imagePath     string
		labelsPath    string
		configPath    string
		selectedCSV   string
		inputName     string
		outputName    string
		width         int
		height        int
		topK          int
		numClasses    int
		ortLibPath    string
		meanStr       string
		stdStr        string
		layout        string
		color         string
		pixelRange    string
		padSquare     bool
		generalThresh float64
		charThresh    float64
	)

	flag.StringVar(&modelPath, "model", "", "Path to ONNX model file")
	flag.StringVar(&imagePath, "image", "", "Path to input image file")
	flag.StringVar(&labelsPath, "labels", "", "Optional path to labels file (one per line)")
	flag.StringVar(&configPath, "config", "", "Optional path to model config JSON")
	flag.StringVar(&selectedCSV, "selected-tags", "", "Optional CSV of selected tags: tag_id,name,category,count")
	flag.StringVar(&inputName, "input", "input", "Model input tensor name")
	flag.StringVar(&outputName, "output", "output", "Model output tensor name")
	flag.IntVar(&width, "width", 224, "Model input width")
	flag.IntVar(&height, "height", 224, "Model input height")
	flag.IntVar(&topK, "topk", 5, "Top-K tags to return (<=0 for all)")
	flag.IntVar(&numClasses, "classes", 0, "Number of classes (required if --labels not provided)")
	flag.StringVar(&ortLibPath, "ort", "", "Path to onnxruntime shared library (optional)")
	flag.StringVar(&meanStr, "mean", "0,0,0", "Normalization mean RGB as comma-separated floats in [0,1]")
	flag.StringVar(&stdStr, "std", "1,1,1", "Normalization stddev RGB as comma-separated floats")
	flag.StringVar(&layout, "layout", "NCHW", "Input layout: NCHW or NHWC")
	flag.StringVar(&color, "color", "RGB", "Channel order: RGB or BGR")
	flag.StringVar(&pixelRange, "pixel-range", "0_1", "Pixel range: 0_1 (scale/normalize) or 0_255 (no /255)")
	flag.BoolVar(&padSquare, "pad-square", false, "Pad to square (white) before resize instead of center-crop")
	flag.Float64Var(&generalThresh, "general-thresh", 0.35, "General tags threshold (wd style)")
	flag.Float64Var(&charThresh, "character-thresh", 0.85, "Character tags threshold (wd style)")
	flag.Parse()

	if modelPath == "" || imagePath == "" {
		fmt.Fprintln(os.Stderr, "Error: --model and --image are required")
		flag.Usage()
		os.Exit(2)
	}

	opts := onnxtag.DefaultOptions()
	opts.InputName = inputName
	opts.OutputName = outputName
	opts.InputWidth = width
	opts.InputHeight = height
	opts.TopK = topK
	opts.ORTSharedLibraryPath = ortLibPath
	opts.NumClasses = numClasses
	opts.InputLayout = layout
	opts.ColorOrder = color
	opts.PixelRange = pixelRange
	opts.PadToSquare = padSquare
	opts.GeneralThreshold = float32(generalThresh)
	opts.CharacterThreshold = float32(charThresh)

	// Parse mean/std
	parse3 := func(s string) ([3]float32, error) {
		parts := strings.Split(s, ",")
		if len(parts) != 3 {
			return [3]float32{}, fmt.Errorf("expected 3 comma-separated values, got %d", len(parts))
		}
		var out [3]float32
		for i := 0; i < 3; i++ {
			var v float64
			_, err := fmt.Sscanf(strings.TrimSpace(parts[i]), "%f", &v)
			if err != nil {
				return [3]float32{}, err
			}
			out[i] = float32(v)
		}
		return out, nil
	}

	if mean, err := parse3(meanStr); err == nil {
		opts.NormalizeMeanRGB = mean
	} else {
		log.Fatalf("invalid --mean: %v", err)
	}
	if std, err := parse3(stdStr); err == nil {
		opts.NormalizeStddevRGB = std
	} else {
		log.Fatalf("invalid --std: %v", err)
	}

	// Apply config if provided
	if configPath != "" {
		cfg, err := onnxtag.LoadModelConfig(configPath)
		if err != nil {
			log.Fatalf("failed to load config: %v", err)
		}
		cfg.ApplyToOptions(&opts)
	}

	if labelsPath != "" {
		// For wd selected_tags.csv, load labels in row order to align with outputs
		labels, err := onnxtag.LoadWdLabelsFromCSV(labelsPath)
		if err != nil {
			log.Fatalf("failed to load labels: %v", err)
		}
		opts.Labels = labels
		// Also try to load wd category indices from the same CSV when --selected-tags not given
		if selectedCSV == "" {
			if rIdx, gIdx, cIdx, err := onnxtag.LoadWdCategoryIndicesWithOrder(labelsPath); err == nil {
				opts.RatingIndices = rIdx
				opts.GeneralIndices = gIdx
				opts.CharacterIndices = cIdx
			}
		}
	}

	if selectedCSV != "" {
		m, err := onnxtag.LoadSelectedTagsCSV(selectedCSV)
		if err != nil {
			log.Fatalf("failed to load selected tags: %v", err)
		}
		opts.SelectedClassNames = m

		// Also load wd category indices using row order for better parity
		ratingIdx, generalIdx, characterIdx, err := onnxtag.LoadWdCategoryIndicesWithOrder(selectedCSV)
		if err == nil {
			opts.RatingIndices = ratingIdx
			opts.GeneralIndices = generalIdx
			opts.CharacterIndices = characterIdx
		}
	}

	tags, err := onnxtag.ClassifyImage(modelPath, imagePath, opts)
	if err != nil {
		log.Fatalf("classification failed: %v", err)
	}

	// Print one per line to stdout
	for _, t := range tags {
		fmt.Println(t)
	}
}
