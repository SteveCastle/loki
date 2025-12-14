package onnxtag

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// ModelConfig maps the provided JSON structure to the subset we need.
type ModelConfig struct {
	Architecture string `json:"architecture"`
	NumClasses   int    `json:"num_classes"`
	NumFeatures  int    `json:"num_features"`
	GlobalPool   string `json:"global_pool"`
	ModelArgs    struct {
		ImgSize      int   `json:"img_size"`
		RefFeatShape []int `json:"ref_feat_shape"`
	} `json:"model_args"`
	PretrainedCfg struct {
		CustomLoad     bool      `json:"custom_load"`
		InputSize      []int     `json:"input_size"`
		FixedInputSize bool      `json:"fixed_input_size"`
		Interpolation  string    `json:"interpolation"`
		CropPct        float32   `json:"crop_pct"`
		CropMode       string    `json:"crop_mode"`
		Mean           []float32 `json:"mean"`
		Std            []float32 `json:"std"`
		NumClasses     int       `json:"num_classes"`
		// wd-tagger specifics are not explicitly listed in config; we set via Options
	} `json:"pretrained_cfg"`
}

// LoadModelConfig reads and parses a JSON config file.
func LoadModelConfig(path string) (*ModelConfig, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	var cfg ModelConfig
	if err := dec.Decode(&cfg); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// ApplyToOptions maps ModelConfig preprocessing settings into Options.
func (mc *ModelConfig) ApplyToOptions(opts *Options) {
	if mc == nil || opts == nil {
		return
	}
	// Determine input size
	if len(mc.PretrainedCfg.InputSize) == 3 {
		// [C,H,W]
		opts.InputWidth = mc.PretrainedCfg.InputSize[2]
		opts.InputHeight = mc.PretrainedCfg.InputSize[1]
	} else if mc.ModelArgs.ImgSize > 0 {
		opts.InputWidth = mc.ModelArgs.ImgSize
		opts.InputHeight = mc.ModelArgs.ImgSize
	}

	// Interpolation
	if mc.PretrainedCfg.Interpolation != "" {
		opts.Interpolation = mc.PretrainedCfg.Interpolation
	}
	// Crop
	if mc.PretrainedCfg.CropPct > 0 {
		opts.CropPct = mc.PretrainedCfg.CropPct
	}
	if mc.PretrainedCfg.CropMode != "" {
		opts.CropMode = mc.PretrainedCfg.CropMode
	}
	// Mean/Std
	if len(mc.PretrainedCfg.Mean) == 3 {
		opts.NormalizeMeanRGB = [3]float32{mc.PretrainedCfg.Mean[0], mc.PretrainedCfg.Mean[1], mc.PretrainedCfg.Mean[2]}
	}
	if len(mc.PretrainedCfg.Std) == 3 {
		opts.NormalizeStddevRGB = [3]float32{mc.PretrainedCfg.Std[0], mc.PretrainedCfg.Std[1], mc.PretrainedCfg.Std[2]}
	}
	// Classes
	if mc.NumClasses > 0 {
		opts.NumClasses = mc.NumClasses
	} else if mc.PretrainedCfg.NumClasses > 0 {
		opts.NumClasses = mc.PretrainedCfg.NumClasses
	}
	// wd-tagger uses NHWC with BGR, and pixel range 0..255 (no /255 normalization in numpy array)
	// See: app.py prepare_image and predict
	opts.InputLayout = "NHWC"
	opts.ColorOrder = "BGR"
	opts.PixelRange = "0_255"
	// wd-tagger pads to square on white then resizes bicubic
	opts.PadToSquare = true
	if mc.PretrainedCfg.Interpolation == "" {
		opts.Interpolation = "bicubic"
	}
}

// LoadSelectedTagsCSV reads a CSV with header: tag_id,name,category,count
// and returns a mapping of class index (tag_id) -> name. Non-integer tag_id rows
// are skipped.
func LoadSelectedTagsCSV(path string) (map[int]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	headerRead := false
	result := make(map[int]string)
	for {
		rec, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if !headerRead {
			headerRead = true
			// Optional: validate header has at least 2 fields
			continue
		}
		if len(rec) < 2 {
			continue
		}
		idStr := strings.TrimSpace(rec[0])
		name := strings.TrimSpace(rec[1])
		id64, err := strconv.ParseInt(idStr, 10, 64)
		if err != nil {
			// skip non-integer tag_id rows
			continue
		}
		idx := int(id64)
		if name == "" {
			name = fmt.Sprintf("class_%d", idx)
		}
		result[idx] = name
	}
	return result, nil
}

// LoadWdCategoryIndices reads selected_tags.csv and extracts category index lists
// for wd-tagger style post-processing: rating (category==9), general (0), character (4).
// LoadWdCategoryIndicesWithOrder builds index lists using CSV row order for indices.
func LoadWdCategoryIndicesWithOrder(path string) (ratingIdx, generalIdx, characterIdx []int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	headerRead := false
	rowIndex := -1 // will increment at first data row to 0
	for {
		rec, err2 := r.Read()
		if err2 == io.EOF {
			break
		}
		if err2 != nil {
			return nil, nil, nil, err2
		}
		if !headerRead {
			headerRead = true
			continue
		}
		if len(rec) < 4 {
			continue
		}
		rowIndex++
		catStr := strings.TrimSpace(rec[2])
		cat64, err4 := strconv.ParseInt(catStr, 10, 64)
		if err4 != nil {
			continue
		}
		idx := rowIndex // model output index is row order
		cat := int(cat64)
		switch cat {
		case 9:
			ratingIdx = append(ratingIdx, idx)
		case 0:
			generalIdx = append(generalIdx, idx)
		case 4:
			characterIdx = append(characterIdx, idx)
		}
	}
	return ratingIdx, generalIdx, characterIdx, nil
}

// LoadWdLabelsFromCSV returns labels in row order to align with model outputs.
func LoadWdLabelsFromCSV(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = -1
	headerRead := false
	var labels []string
	for {
		rec, err2 := r.Read()
		if err2 == io.EOF {
			break
		}
		if err2 != nil {
			return nil, err2
		}
		if !headerRead {
			headerRead = true
			continue
		}
		if len(rec) < 2 {
			continue
		}
		name := strings.TrimSpace(rec[1])
		if !isKaomoji(name) {
			name = strings.ReplaceAll(name, "_", " ")
		}
		labels = append(labels, name)
	}
	return labels, nil
}

// subset of kaomojis used in wd-tagger to avoid underscore replacement
var kaomojis = map[string]struct{}{
	"0_0": {}, "(o)_(o)": {}, "+_+": {}, "+_-": {}, "._.": {}, "_": {}, "<|>_<|>": {},
	"=_=": {}, ">_<": {}, "3_3": {}, "6_9": {}, ">_o": {}, "@_@": {}, "^_^": {},
	"o_o": {}, "u_u": {}, "x_x": {}, "|_|": {}, "||_||": {},
}

func isKaomoji(s string) bool {
	_, ok := kaomojis[s]
	return ok
}
