package tasks

import "github.com/stevecastle/shrike/appconfig"

// TaggerModel describes one auto-tagging model: its identity and the files it
// needs (all resolved under the model's dir via deps.ModelPath). This mirrors
// the embedding model registry (embedmodels.go) so the tagger is switchable —
// a new model slots in by adding one entry here plus a manifest download entry.
type TaggerModel struct {
	ID          string
	DisplayName string
	ModelFile   string // ONNX model, e.g. "model.onnx"
	LabelsFile  string // label/category CSV, e.g. "selected_tags.csv"
	ConfigFile  string // preprocessing config JSON, e.g. "config.json"
}

// DefaultTaggerModelID is used when the configured model is empty or unknown.
const DefaultTaggerModelID = appconfig.DefaultAutotagModel

// taggerModelRegistry is keyed by model ID. Add new tagging models here.
var taggerModelRegistry = map[string]TaggerModel{
	"wd-eva02-large-tagger-v3": {
		ID:          "wd-eva02-large-tagger-v3",
		DisplayName: "WD EVA02 Large v3 (anime / illustration)",
		ModelFile:   "model.onnx",
		LabelsFile:  "selected_tags.csv",
		ConfigFile:  "config.json",
	},
}

// TaggerModelByID returns the registry entry and whether it exists.
func TaggerModelByID(id string) (TaggerModel, bool) {
	m, ok := taggerModelRegistry[id]
	return m, ok
}

// TaggerModelList returns all registered tagger models in a stable display
// order (default first) for populating the config UI.
func TaggerModelList() []TaggerModel {
	order := []string{"wd-eva02-large-tagger-v3"}
	out := make([]TaggerModel, 0, len(taggerModelRegistry))
	seen := map[string]bool{}
	for _, id := range order {
		if m, ok := taggerModelRegistry[id]; ok {
			out = append(out, m)
			seen[id] = true
		}
	}
	for id, m := range taggerModelRegistry {
		if !seen[id] {
			out = append(out, m)
		}
	}
	return out
}

// ActiveTaggerModel returns the configured active tagging model, falling back to
// the default when the config is empty or names an unknown model.
func ActiveTaggerModel() TaggerModel {
	id := appconfig.Get().AutotagModel
	if m, ok := taggerModelRegistry[id]; ok {
		return m
	}
	return taggerModelRegistry[DefaultTaggerModelID]
}
