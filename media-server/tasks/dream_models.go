package tasks

// The diffusion models loki-dream can run, for the pickers of the Transform
// flow's Generate / Reimagine and the Scene editor's "new scene from a
// prompt". The engine is the authority: `loki-dream --list-models --json`
// scans its model folders (LOKI_MODELS included), detects each checkpoint's
// architecture (MOSS-Image, Krea 2, FLUX.2 klein, Anima, SDXL) and says what it
// runs with and what a first run would download. The answer is cached for a
// while; a refresh asks again.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/platform"
)

// DreamModelNeed is a file a model needs beside its checkpoint (its text
// encoder or VAE): here, or downloaded on first use.
type DreamModelNeed struct {
	File       string  `json:"file"`
	Present    bool    `json:"present"`
	DownloadGB float64 `json:"download_gb,omitempty"`
	From       string  `json:"from,omitempty"`
}

// DreamModel is one diffusion model loki-dream can run.
type DreamModel struct {
	// File is the bare file name a job names in `model`.
	File  string `json:"file"`
	Label string `json:"label"`
	// Architecture: moss | krea2 | flux2 | anima | sdxl; ArchitectureName for people.
	Architecture     string `json:"architecture"`
	ArchitectureName string `json:"architecture_name"`
	// Distilled checkpoints run few steps without guidance (no negative prompt).
	Distilled bool    `json:"distilled"`
	Steps     int     `json:"steps"`
	CFG       float64 `json:"cfg"`
	// Turbo: --turbo is a distillation LoRA for this model (MOSS-Image).
	Turbo bool `json:"turbo"`
	LoRAs bool `json:"loras"`
	// Local: the checkpoint is here; otherwise a run downloads DownloadGB.
	Local       bool            `json:"local"`
	Format      string          `json:"format,omitempty"`
	SizeGB      float64         `json:"size_gb,omitempty"`
	DownloadGB  float64         `json:"download_gb,omitempty"`
	From        string          `json:"from,omitempty"`
	BaseModel   string          `json:"base_model,omitempty"`
	TextEncoder *DreamModelNeed `json:"text_encoder,omitempty"`
	VAE         *DreamModelNeed `json:"vae,omitempty"`
}

// DreamModelList is the catalog as the API serves it.
type DreamModelList struct {
	Models []DreamModel `json:"models"`
	// Default is the model a job without `model` runs: the server's
	// dreamModel setting, else the engine's own default.
	Default string `json:"default"`
	// Where the engine that answered runs ("this machine" or a worker).
	Where string `json:"where,omitempty"`
	// Note explains an empty or partial list (no engine yet, an old engine).
	Note    string    `json:"note,omitempty"`
	Fetched time.Time `json:"fetched"`
}

const dreamModelsTTL = 2 * time.Minute

var (
	dreamModelsMu    sync.Mutex
	dreamModelsCache *DreamModelList
)

// DreamModels returns the catalog, from the cache unless it is stale or
// refresh is set.
func DreamModels(ctx context.Context, refresh bool) DreamModelList {
	dreamModelsMu.Lock()
	defer dreamModelsMu.Unlock()
	if !refresh && dreamModelsCache != nil && time.Since(dreamModelsCache.Fetched) < dreamModelsTTL {
		return *dreamModelsCache
	}
	l := fetchDreamModels(ctx)
	// a failed answer is not kept: the next request asks again
	if l.Note == "" || len(l.Models) > 0 {
		dreamModelsCache = &l
	}
	return l
}

// dreamModelArch is the architecture of a listed model file (case-insensitive
// bare name), from the cached catalog; "" when unknown.
func dreamModelArch(file string) string {
	dreamModelsMu.Lock()
	defer dreamModelsMu.Unlock()
	if dreamModelsCache == nil {
		return ""
	}
	for _, m := range dreamModelsCache.Models {
		if strings.EqualFold(m.File, file) || strings.EqualFold(strings.TrimSuffix(m.File, ".safetensors"), file) {
			return m.Architecture
		}
	}
	return ""
}

// listDreamModels runs the engine's --list-models (a variable so tests can
// stand in an answer).
var listDreamModels = func(ctx context.Context) ([]byte, string, error) {
	tool, ok := locateAITool(dreamTool)
	if !ok {
		return nil, "", errDreamNotInstalled
	}
	if !aiToolHelpHas(ctx, tool, "--list-models") {
		return nil, tool.Where(), errDreamTooOld
	}
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	args := []string{"--list-models", "--json"}
	if tool.Remote() {
		out, err := runRemote(ctx, tool, args, true, nil, nil)
		if err != nil {
			return nil, tool.Where(), fmt.Errorf("loki-dream --list-models on %s: %v", tool.Where(), err)
		}
		return []byte(strings.Join(out, "\n")), tool.Where(), nil
	}
	cmd := exec.CommandContext(ctx, tool.Bin, args...)
	if len(tool.Env) > 0 {
		cmd.Env = append(os.Environ(), tool.Env...)
	}
	platform.HideSubprocessWindow(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, tool.Where(), fmt.Errorf("%s --list-models: %s", filepath.Base(tool.Bin), msg)
	}
	return stdout.Bytes(), tool.Where(), nil
}

var (
	errDreamNotInstalled = fmt.Errorf("loki-dream is not installed yet: the first Generate job installs it with its default model")
	errDreamTooOld       = fmt.Errorf("this loki-dream is too old to list its models; update it to choose one")
)

func fetchDreamModels(ctx context.Context) DreamModelList {
	l := DreamModelList{Models: []DreamModel{}, Fetched: time.Now()}
	body, where, err := listDreamModels(ctx)
	l.Where = where
	if err != nil {
		l.Note = err.Error()
	} else {
		var raw struct {
			Models  []DreamModel `json:"models"`
			Default string       `json:"default"`
		}
		if jerr := json.Unmarshal(bytes.TrimSpace(body), &raw); jerr != nil {
			l.Note = "could not read loki-dream's model list: " + jerr.Error()
		} else {
			for _, m := range raw.Models {
				if m.File != "" {
					l.Models = append(l.Models, m)
				}
			}
			l.Default = raw.Default
		}
	}
	if c := strings.TrimSpace(appconfig.Get().DreamModel); c != "" {
		l.Default = c
	}
	return l
}

// validDreamModelName reports whether a job's `model` is a bare file name:
// the engine resolves it inside its own model folders, so a job can never
// point it at an arbitrary path.
func validDreamModelName(name string) bool {
	return name != "" && !strings.ContainsAny(name, `/\:`) && name != "." && name != ".." && !strings.HasPrefix(name, ".")
}

// dreamSpecFor is the engine spec for a run of `model`: the Kromoss weights
// are a dependency only when the model is MOSS-Image (or unknown); the other
// architectures' engine fetches its own text encoder and VAE.
func dreamSpecFor(model string) aiToolSpec {
	if model == "" {
		return dreamTool
	}
	if a := dreamModelArch(model); a != "" && a != "moss" {
		return aiToolSpec{Bin: dreamTool.Bin}
	}
	return dreamTool
}
