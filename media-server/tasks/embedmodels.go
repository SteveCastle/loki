package tasks

import "github.com/stevecastle/shrike/appconfig"

// EmbedModel describes one visual-embedding model: its identity, output
// dimensionality, image preprocessing, output pooling, and (for multimodal
// models) its text encoder. It is the single source of truth that replaces the
// former package-level EmbedModelID/EmbedDim constants, so a second model
// (DINOv2) slots in by adding one entry here plus a manifest download entry.
//
// Vectors are stored keyed by ID in media_embedding, so models coexist
// non-destructively. Switching the active model (appconfig.EmbeddingModel)
// changes which model the embed task writes and which the image->image search
// reads; text->image search always resolves a multimodal model (see
// TextSearchModel).
type EmbedModel struct {
	ID          string
	DisplayName string
	Dim         int
	// Multimodal is true when the model ships a text encoder (text->image
	// search). Image-only models (DINOv2) cannot serve text queries.
	Multimodal bool

	// Image preprocessing + ONNX I/O. These are forwarded to the embed
	// subprocess as flags; defaults match SigLIP 2's pipeline.
	ImageModelFile string // rel path under the model dir, e.g. "image_model.onnx"
	ImgInput       string // input tensor name, e.g. "pixel_values"
	ImgOutput      string // output tensor name, e.g. "pooler_output" / "last_hidden_state"
	Width, Height  int
	Mean, Std      [3]float32
	CropPct        float32 // <1 with CropMode "center" enables center crop; 1.0 disables
	CropMode       string  // "center" or ""
	// Pooling selects how the embedding is read from the model output:
	//   "" / "none" -> the output is already a pooled [1,Dim] vector (SigLIP 2)
	//   "cls"       -> the output is a sequence [1,N,Dim]; take token 0 (DINOv2)
	Pooling string

	// Text encoder (Multimodal only). Empty for image-only models.
	TextModelFile string
	TokenizerFile string
	TextInput     string
	TextOutput    string
	SeqLen        int
}

// DefaultEmbedModelID is used when the configured model is empty or unknown. It
// is a multimodal model so text search always has a fallback.
const DefaultEmbedModelID = appconfig.DefaultEmbeddingModel

// embedModelRegistry is keyed by model ID. Add new models here.
var embedModelRegistry = map[string]EmbedModel{
	"siglip2-base-patch16-224": {
		ID:             "siglip2-base-patch16-224",
		DisplayName:    "SigLIP 2 base (multimodal — image + text)",
		Dim:            768,
		Multimodal:     true,
		ImageModelFile: "image_model.onnx",
		ImgInput:       "pixel_values",
		ImgOutput:      "pooler_output",
		Width:          224,
		Height:         224,
		Mean:           [3]float32{0.5, 0.5, 0.5},
		Std:            [3]float32{0.5, 0.5, 0.5},
		CropPct:        1.0,
		CropMode:       "",
		Pooling:        "none",
		TextModelFile:  "text_model.onnx",
		TokenizerFile:  "tokenizer.model",
		TextInput:      "input_ids",
		TextOutput:     "pooler_output",
		SeqLen:         64,
	},
	"dinov2-base": {
		ID:             "dinov2-base",
		DisplayName:    "DINOv2 base (image-only — self-supervised)",
		Dim:            768,
		Multimodal:     false,
		ImageModelFile: "image_model.onnx",
		ImgInput:       "pixel_values",
		// The export's only output is the token sequence; the usable embedding is
		// the CLS token. (Its pooler_output is the untrained HF pooler.)
		ImgOutput: "last_hidden_state",
		Width:     224,
		Height:    224,
		Mean:      [3]float32{0.485, 0.456, 0.406},
		Std:       [3]float32{0.229, 0.224, 0.225},
		CropPct:   0.875, // ~ shortest-edge 256 then center-crop 224
		CropMode:  "center",
		Pooling:   "cls",
	},
}

// EmbedModelByID returns the registry entry and whether it exists.
func EmbedModelByID(id string) (EmbedModel, bool) {
	m, ok := embedModelRegistry[id]
	return m, ok
}

// EmbedModelList returns all registered models in a stable, display order
// (default/multimodal first) for populating the config UI.
func EmbedModelList() []EmbedModel {
	order := []string{"siglip2-base-patch16-224", "dinov2-base"}
	out := make([]EmbedModel, 0, len(embedModelRegistry))
	seen := map[string]bool{}
	for _, id := range order {
		if m, ok := embedModelRegistry[id]; ok {
			out = append(out, m)
			seen[id] = true
		}
	}
	for id, m := range embedModelRegistry {
		if !seen[id] {
			out = append(out, m)
		}
	}
	return out
}

// ActiveEmbedModel returns the configured active model, falling back to the
// default when the config is empty or names an unknown model.
func ActiveEmbedModel() EmbedModel {
	id := appconfig.Get().EmbeddingModel
	if m, ok := embedModelRegistry[id]; ok {
		return m
	}
	return embedModelRegistry[DefaultEmbedModelID]
}

// TextSearchModel returns the model used for text->image search. Text search
// requires a text encoder, so it uses the active model when that model is
// multimodal, otherwise the default multimodal model (SigLIP 2). The returned
// model's vectors are what text queries are matched against.
func TextSearchModel() EmbedModel {
	if m := ActiveEmbedModel(); m.Multimodal {
		return m
	}
	return embedModelRegistry[DefaultEmbedModelID]
}
