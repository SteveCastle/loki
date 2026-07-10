package appconfig

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/platform"
)

// StorageRoot represents a single storage root, either a local filesystem path or an S3-compatible bucket.
type StorageRoot struct {
	Type            string `json:"type"`              // "local" or "s3"
	Path            string `json:"path,omitempty"`    // local filesystem path
	Label           string `json:"label"`             // display name in UI
	Default         bool   `json:"default,omitempty"` // true = destination for uploads/downloads
	Endpoint        string `json:"endpoint,omitempty"`
	Region          string `json:"region,omitempty"`
	Bucket          string `json:"bucket,omitempty"`
	Prefix          string `json:"prefix,omitempty"`
	AccessKey       string `json:"accessKey,omitempty"`
	SecretKey       string `json:"secretKey,omitempty"`
	ThumbnailPrefix string `json:"thumbnailPrefix,omitempty"`
}

// ByoFaceModel declares a bring-your-own face recognizer: a user-supplied
// ONNX model (typically a research-licensed ArcFace/AdaFace export that can't
// be shipped) usable as the active FaceModel. Detected faces are aligned to
// the standard 112×112 five-landmark template and fed to this model.
type ByoFaceModel struct {
	ID   string `json:"id"`             // unique; face vectors are stored keyed by this
	Name string `json:"name,omitempty"` // display name (defaults to ID)
	// ModelPath is the absolute path to the recognizer ONNX file on disk.
	ModelPath string `json:"modelPath"`
	Dim       int    `json:"dim"` // embedding dimension (512 for ArcFace-family)
	// Tensor names; default "data"/"fc1" when empty.
	InputName  string `json:"inputName,omitempty"`
	OutputName string `json:"outputName,omitempty"`
	// Mean/Std are per-channel RGB on the 0..255 pixel scale; ArcFace-family
	// models want 127.5/127.5. Empty means raw pixels (mean 0, std 1).
	Mean []float64 `json:"mean,omitempty"`
	Std  []float64 `json:"std,omitempty"`
	// ColorOrder is "RGB" (ArcFace-family) or "BGR"; default "BGR".
	ColorOrder string `json:"colorOrder,omitempty"`
	// MatchThreshold is the cosine similarity at/above which two faces are
	// considered the same person by clustering. 0 uses a conservative default,
	// but recognizers differ (SFace and ArcFace have different score
	// distributions) so BYO entries should set it explicitly.
	MatchThreshold float64 `json:"matchThreshold,omitempty"`
}

// DefaultEmbeddingModel is the visual-embedding model used when none is
// configured. Must match an ID in the tasks package's embed-model registry.
// Kept as a literal here (not imported from tasks) so appconfig stays a leaf
// package with no dependency cycle.
const DefaultEmbeddingModel = "siglip2-base-patch16-224"

// DefaultFaceModel is the face-identity recognizer used when none is
// configured: SFace (OpenCV Zoo, Apache-2.0). Must match an ID in the tasks
// package's face-model registry or a ByoFaceModels entry.
const DefaultFaceModel = "sface"

// DefaultTranscriptionProvider / DefaultTranscriptionModel are used when the
// transcription section is empty. The provider id must match a registration
// in the transcribe package. An empty model means "use the provider's
// default", which is platform-aware (e.g. large-v3-turbo where the XXL
// build is available, large-v2 on macOS) — so appconfig stays a leaf
// package and the transcribe provider owns the choice.
const (
	DefaultTranscriptionProvider = "whisper-cli"
	DefaultTranscriptionModel    = ""
)

// DefaultAutotagModel is the auto-tagging model used when none is configured.
// Must match an ID in the tasks package's tagger-model registry. Literal here
// (not imported from tasks) to keep appconfig a leaf package.
const DefaultAutotagModel = "wd-eva02-large-tagger-v3"

// DefaultPort is the HTTP listen port used when none is configured.
// 10111 is "L0K1" in leet: L→1, O→0, K→11 (11th letter), I→1.
const DefaultPort = 10111

// Config holds application configuration including database path, LLM prompts, and AI model paths.
type Config struct {
	DBPath string `json:"dbPath"`

	// HTTP listen port. Changing it requires a server restart. Overridable
	// via the LOWKEY_PORT env var (Docker-friendly); 0/invalid falls back to
	// DefaultPort at load time.
	Port int `json:"port"`

	// Download path for media files
	DownloadPath string `json:"downloadPath"`

	// Active vision-inference backend. One of: "off", "ollama", "runpod".
	// Drives the routing in tasks.callVisionLLM — provider-specific fields
	// below (OllamaBaseURL/Model, RunPodEndpoint/APIKey) are only consulted
	// for the matching provider. New engines slot in by adding a constant
	// + a switch case + a config sub-section; the field stays a string so
	// the on-disk format doesn't break when the set of providers grows.
	InferenceProvider string `json:"inferenceProvider"`

	// Ollama / LLM settings
	OllamaBaseURL  string `json:"ollamaBaseUrl"`
	OllamaModel    string `json:"ollamaModel"`
	DescribePrompt string `json:"describePrompt"`

	// RunPod serverless vision settings. Active when InferenceProvider ==
	// "runpod". The worker is expected to expose an OpenAI-compatible
	// chat-completions interface (e.g. SvenBrnn/runpod-worker-ollama).
	// The endpoint may point at either `/run` (async, polled) or `/runsync`
	// (inline response).
	RunPodEndpoint string `json:"runpodEndpoint"`
	RunPodAPIKey   string `json:"runpodApiKey"`

	// LM Studio vision settings. Active when InferenceProvider == "lmstudio".
	// LM Studio exposes an OpenAI-compatible /v1/chat/completions endpoint
	// (default base http://localhost:1234). API key is rarely needed for
	// local installs but supported for proxied / remote setups.
	LMStudioBaseURL string `json:"lmstudioBaseUrl"`
	LMStudioModel   string `json:"lmstudioModel"`
	LMStudioAPIKey  string `json:"lmstudioApiKey"`

	// llama.cpp server vision settings. Active when InferenceProvider ==
	// "llamacpp". The official llama.cpp `server` binary exposes an
	// OpenAI-compatible /v1/chat/completions endpoint (default base
	// http://localhost:8080). API key is optional and only honored if set.
	LlamaCppBaseURL string `json:"llamacppBaseUrl"`
	LlamaCppModel   string `json:"llamacppModel"`
	LlamaCppAPIKey  string `json:"llamacppApiKey"`

	// Per-provider concurrency caps. Drives the jobqueue host-bucket limit
	// for the corresponding inference bucket — a single GPU local install
	// typically wants 1 at a time, while RunPod serverless can absorb many
	// concurrent requests because it scales out per-call. Add one field
	// per new engine; values <= 0 leave the queue at its default of 1.
	InferenceConcurrency struct {
		Ollama   int `json:"ollama"`
		RunPod   int `json:"runpod"`
		LMStudio int `json:"lmstudio"`
		LlamaCpp int `json:"llamacpp"`
	} `json:"inferenceConcurrency"`

	// AutoProcessOps is the comma-separated per-item op list the scheduled
	// combined job runs. Empty = every op (hash, dimensions, describe,
	// transcribe, autotag, embed, faces).
	//
	// NOTE: the scheduler's on/off mode is deliberately NOT config — it is
	// runtime-only state that resets to "stopped" on every server start, so
	// background compute is always an affirmative per-session choice. (A
	// legacy "autoProcessMode" key may linger in old config files; it is
	// ignored.)
	AutoProcessOps string `json:"autoProcessOps"`

	// LocalComputeConcurrency caps how many resource-intensive LOCAL jobs
	// (ONNX embed/autotag/faces, transcription, and local-LLM inference) may
	// run at the same time, machine-wide, regardless of which per-task bucket
	// each one lives in. Every such job holds a shared "local-compute" slot in
	// addition to its own bucket. 1 (the default) serializes all heavy local
	// work — each job still parallelizes internally via its worker pool, which
	// is sized to own the machine. Raise it only if the hardware has headroom
	// for genuinely concurrent model workloads. Remote inference (RunPod) does
	// not consume a slot. Values <= 0 fall back to the default.
	LocalComputeConcurrency int `json:"localComputeConcurrency"`

	// ONNX tagger settings
	OnnxTagger struct {
		ModelPath            string  `json:"modelPath"`
		LabelsPath           string  `json:"labelsPath"`
		ConfigPath           string  `json:"configPath"`
		ORTSharedLibraryPath string  `json:"ortSharedLibraryPath"`
		GeneralThreshold     float64 `json:"generalThreshold"`
		CharacterThreshold   float64 `json:"characterThreshold"`
	} `json:"onnxTagger"`

	// Active visual-embedding model ID. Governs both indexing (the `embed`
	// task) and image->image similarity search. Vectors are stored keyed by
	// model, so switching is non-destructive — previously-embedded vectors are
	// retained and become searchable again on switch-back (only the in-memory
	// ANN index rebuilds). Must be a known ID from tasks' embed-model registry
	// (e.g. "siglip2-base-patch16-224", "dinov2-base"); empty falls back to the
	// default. Text->image search always uses a multimodal model (SigLIP 2)
	// regardless of this setting.
	EmbeddingModel string `json:"embeddingModel"`

	// Embedding execution provider: "cpu" or "directml" (GPU, Windows DX12).
	// "directml" requires the optional GPU runtime to be installed; absent it,
	// the embed task falls back to CPU.
	EmbeddingProvider string `json:"embeddingProvider"`

	// Embedding performance preset governing parallelism: "low" (~25% of cores,
	// keeps the system responsive), "balanced" (~50%), "max" (all but one core),
	// or "custom" (use EmbeddingWorkers / EmbeddingThreadsPerWorker).
	EmbeddingPerformance string `json:"embeddingPerformance"`

	// Advanced overrides used when EmbeddingPerformance == "custom" (0 = derive
	// from the preset). Workers is the number of parallel embed worker processes;
	// ThreadsPerWorker is the ONNX Runtime intra-op thread count per worker.
	EmbeddingWorkers          int `json:"embeddingWorkers"`
	EmbeddingThreadsPerWorker int `json:"embeddingThreadsPerWorker"`

	// Active auto-tagging model ID. Must be a known ID from the tasks package's
	// tagger-model registry (e.g. "wd-eva02-large-tagger-v3"); empty falls back
	// to the default. Switchable like EmbeddingModel.
	AutotagModel string `json:"autotagModel"`

	// Auto-tagging (ONNX) execution provider + performance, mirroring the
	// embedding settings so tagging can be tuned independently. Provider is
	// "cpu" or "directml" (shares the same GPU runtime as embedding).
	AutotagProvider         string `json:"autotagProvider"`
	AutotagPerformance      string `json:"autotagPerformance"`
	AutotagWorkers          int    `json:"autotagWorkers"`
	AutotagThreadsPerWorker int    `json:"autotagThreadsPerWorker"`

	// Active face-identity recognizer ID: "sface" (built-in, Apache-2.0,
	// downloadable from Dependencies) or the ID of a ByoFaceModels entry.
	// Face vectors are stored keyed by model, so switching is non-destructive.
	// Detection is always YuNet regardless of this setting.
	FaceModel string `json:"faceModel"`

	// Face task execution provider + performance, mirroring the embedding
	// settings ("cpu" or "directml"; presets low/balanced/max/custom).
	FaceProvider         string `json:"faceProvider"`
	FacePerformance      string `json:"facePerformance"`
	FaceWorkers          int    `json:"faceWorkers"`
	FaceThreadsPerWorker int    `json:"faceThreadsPerWorker"`

	// FaceRouting: "auto" (default) classifies each media item as photo vs
	// anime via a SigLIP text probe and scans/searches it under the matching
	// recognizer, so one faces job handles both domains; "single" pins
	// everything to FaceModel (pre-routing behavior).
	FaceRouting string `json:"faceRouting"`

	// Saved grouping tuner (the People panel's Tune sliders). These apply to
	// EVERY clustering pass — the Group new faces / Rebuild buttons, tuned
	// regroups, and the incremental in-scan passes — with explicit
	// faces-cluster job flags overriding them per run. Zero values mean
	// "use the built-in default" (offset 0 IS the default; 0 min-cluster /
	// min-quality read as unset).
	FaceClusterThresholdOffset float64 `json:"faceClusterThresholdOffset"` // added to each recognizer's default threshold (−0.2…0.3)
	FaceClusterMinCluster      int     `json:"faceClusterMinCluster"`      // members needed to mint a new group (default 3)
	FaceClusterMinQuality      float64 `json:"faceClusterMinQuality"`      // detection-confidence floor for founding groups (default 0.75)

	// Bring-your-own face recognizers (research-licensed models like ArcFace
	// or AdaFace exports that can't be shipped). The user supplies the ONNX
	// file; entries here make it selectable as FaceModel.
	ByoFaceModels []ByoFaceModel `json:"byoFaceModels,omitempty"`

	// Per-file processing timeout (seconds) for the local ONNX tasks (embed,
	// autotag). A single file that exceeds this — e.g. a corrupt image stuck in
	// decode or a bad video stuck in frame extraction — is skipped and the job
	// continues (the stuck worker is killed and replaced). <= 0 disables the
	// timeout.
	OnnxFileTimeoutSeconds int `json:"onnxFileTimeoutSeconds"`

	// Transcription settings. Provider names an implementation in the
	// transcribe package's registry ("whisper-cli" today; HTTP or other
	// engines can register later). Model is provider-specific ("" = provider
	// default), Language is an ISO hint ("" = auto-detect), VADFilter trims
	// non-speech before transcribing.
	TranscriptionProvider  string `json:"transcriptionProvider"`
	TranscriptionModel     string `json:"transcriptionModel"`
	TranscriptionLanguage  string `json:"transcriptionLanguage"`
	TranscriptionVADFilter bool   `json:"transcriptionVadFilter"`

	// Optional path to a user-supplied faster-whisper executable. Overrides
	// the binary installed via the Dependencies downloader.
	FasterWhisperPath string `json:"fasterWhisperPath"`

	// Discord authentication token for media export
	DiscordToken string `json:"discordToken"`

	// JWT Secret for authentication
	JWTSecret string `json:"jwtSecret"`

	// SetupComplete is true once the first-run setup wizard has finished (or
	// been inferred complete for installs that predate the wizard). While
	// false, unauthenticated page requests are funneled to /setup and the
	// setup APIs are open; once true, the wizard locks behind admin auth.
	SetupComplete bool `json:"setupComplete"`

	// Storage roots for web filesystem browsing
	Roots []StorageRoot `json:"roots"`

	// Deprecated: kept for migration only — use Roots instead
	RootPaths []string `json:"rootPaths,omitempty"`
}

var (
	cfgMu sync.RWMutex
	cfg   Config
)

// Env-supplied storage roots (LOWKEY_ROOTS / LOWKEY_ROOT_<N>) override the
// config file at runtime but must never be baked into it by Save. Track the
// injected set so Save can recognize and skip it.
var (
	envRootsMu     sync.Mutex
	envRootsActive bool
	envRootsVal    []StorageRoot
)

// EnvRootsActive reports whether storage roots are currently supplied by
// environment variables, meaning the config file's roots are overridden at
// runtime and edits to them only take effect once the env vars are removed.
func EnvRootsActive() bool {
	envRootsMu.Lock()
	defer envRootsMu.Unlock()
	return envRootsActive
}

// downloadPathRoot builds the default local root migrated from DownloadPath
// for configs that predate the storage-roots system.
func downloadPathRoot(downloadPath string) []StorageRoot {
	return []StorageRoot{{
		Type:    "local",
		Path:    downloadPath,
		Label:   "Downloads",
		Default: true,
	}}
}

// defaultDownloadPath returns the default download path (~/media).
func defaultDownloadPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "media"
	}
	return filepath.Join(home, "media")
}

// DefaultDBPath returns the default database path.
// Uses the platform-specific data directory.
func DefaultDBPath() string {
	return filepath.Join(platform.GetDataDir(), "media.db")
}

// DefaultConfigDir returns the default config directory path.
// Uses the platform-specific data directory.
func DefaultConfigDir() string {
	return platform.GetDataDir()
}

// defaultConfig returns a Config populated with sensible defaults.
func defaultConfig() Config {
	return Config{
		DBPath:                 DefaultDBPath(),
		Port:                   DefaultPort,
		DownloadPath:           defaultDownloadPath(),
		InferenceProvider:      "ollama",
		EmbeddingModel:         DefaultEmbeddingModel,
		EmbeddingProvider:      "cpu",
		EmbeddingPerformance:   "balanced",
		AutotagModel:           DefaultAutotagModel,
		AutotagProvider:        "cpu",
		AutotagPerformance:     "balanced",
		FaceModel:              DefaultFaceModel,
		FaceProvider:           "cpu",
		FacePerformance:        "balanced",
		FaceRouting:            "auto",
		OnnxFileTimeoutSeconds: 120,
		TranscriptionProvider:  DefaultTranscriptionProvider,
		TranscriptionModel:     DefaultTranscriptionModel,
		TranscriptionLanguage:  "en",
		TranscriptionVADFilter: true,
		OllamaBaseURL:          "http://localhost:11434",
		OllamaModel:            "llama3.2-vision",
		DescribePrompt:         "Please describe this image, paying special attention to the people, the color of hair, clothing, items, text and captions, and actions being performed.",
		LMStudioBaseURL:        "http://localhost:1234",
		LlamaCppBaseURL:        "http://localhost:8080",
		InferenceConcurrency: struct {
			Ollama   int `json:"ollama"`
			RunPod   int `json:"runpod"`
			LMStudio int `json:"lmstudio"`
			LlamaCpp int `json:"llamacpp"`
		}{
			Ollama:   1, // local single-GPU Ollama: one at a time
			RunPod:   4, // serverless scales out per request
			LMStudio: 1, // local single-GPU LM Studio: one at a time
			LlamaCpp: 1, // local single-GPU llama.cpp: one at a time
		},
		LocalComputeConcurrency: 1, // one heavy local model workload at a time
		OnnxTagger: struct {
			ModelPath            string  `json:"modelPath"`
			LabelsPath           string  `json:"labelsPath"`
			ConfigPath           string  `json:"configPath"`
			ORTSharedLibraryPath string  `json:"ortSharedLibraryPath"`
			GeneralThreshold     float64 `json:"generalThreshold"`
			CharacterThreshold   float64 `json:"characterThreshold"`
		}{
			GeneralThreshold:   0.35,
			CharacterThreshold: 0.85,
		},
		JWTSecret: uuid.New().String(),
		Roots:     []StorageRoot{},
	}
}

// Get returns a copy of the current in-memory config.
func Get() Config {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return cfg
}

// ListenAddr returns the bind address (":<port>") for the HTTP server.
func (c Config) ListenAddr() string {
	return fmt.Sprintf(":%d", c.Port)
}

// LocalBaseURL returns the loopback base URL ("http://localhost:<port>")
// used for log messages and opening the web UI in a browser.
func (c Config) LocalBaseURL() string {
	return fmt.Sprintf("http://localhost:%d", c.Port)
}

// Set replaces the in-memory config.
func Set(c Config) {
	cfgMu.Lock()
	cfg = c
	cfgMu.Unlock()
}

func isJSONObject(raw []byte) bool {
	raw = bytes.TrimSpace(raw)
	return len(raw) > 0 && raw[0] == '{'
}

func deepMergeJSON(dst, src map[string]json.RawMessage) {
	for k, v := range src {
		if existing, ok := dst[k]; ok && isJSONObject(existing) && isJSONObject(v) {
			var dstObj map[string]json.RawMessage
			var srcObj map[string]json.RawMessage
			if err := json.Unmarshal(existing, &dstObj); err != nil {
				dst[k] = v
				continue
			}
			if err := json.Unmarshal(v, &srcObj); err != nil {
				dst[k] = v
				continue
			}
			deepMergeJSON(dstObj, srcObj)
			merged, err := json.Marshal(dstObj)
			if err != nil {
				dst[k] = v
				continue
			}
			dst[k] = merged
			continue
		}
		dst[k] = v
	}
}

// getConfigPath returns the full path to the config.json file.
// It is a variable so tests can override it to use temp directories.
var getConfigPath = func() (string, error) {
	// Explicit override first — deployments (Docker, tests, side-by-side
	// instances) can pin the config file location.
	if v := strings.TrimSpace(os.Getenv("LOWKEY_CONFIG_PATH")); v != "" {
		return v, nil
	}
	// Safety net: under `go test`, NEVER touch the real config file. A unit
	// test that reaches Save() through production code (as the scheduler's
	// SetMode once did) must not overwrite the developer's live
	// %APPDATA% config — that incident reset a real dbPath and jwtSecret.
	// Tests that care about the path set LOWKEY_CONFIG_PATH explicitly (or
	// override this var from inside the package).
	if underGoTest() {
		return filepath.Join(os.TempDir(), fmt.Sprintf("lowkey-test-config-%d.json", os.Getpid())), nil
	}
	configDir := DefaultConfigDir()
	return filepath.Join(configDir, "config.json"), nil
}

// underGoTest reports whether we're running inside a `go test` binary (the
// testing framework registers the test.v flag at init).
func underGoTest() bool {
	return flag.Lookup("test.v") != nil
}

// Load reads the config from disk and updates the in-memory config. It returns the config and path.
// If the config file doesn't exist, it creates one with default values.
// This function safely handles missing directories and creates them as needed.
func Load() (Config, string, error) {
	path, err := getConfigPath()
	if err != nil {
		return Config{}, "", err
	}

	// Ensure config directory exists
	configDir := filepath.Dir(path)
	if err := os.MkdirAll(configDir, 0755); err != nil {
		return Config{}, "", fmt.Errorf("failed to create config directory %s: %v", configDir, err)
	}

	// Check if config file exists
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Config file doesn't exist - create it with defaults
			def := defaultConfig()

			// Seed the default storage root from the download path BEFORE
			// saving, so the file carries a real, user-editable root instead
			// of regenerating a phantom one in memory on every boot.
			if def.DownloadPath != "" {
				def.Roots = downloadPathRoot(def.DownloadPath)
			}

			// Save the default config (without env overrides — env vars stay
			// in the environment; the file only records persistent choices)
			savedPath, saveErr := Save(def)
			if saveErr != nil {
				return Config{}, path, fmt.Errorf("failed to create default config file: %v", saveErr)
			}

			// Apply env overrides on first run too, or a fresh container
			// would boot once on pure defaults (wrong DB path, no storage
			// roots) and then silently switch on the next restart.
			applyEnvOverrides(&def)

			// Ensure the database directory exists (after overrides so an
			// env-provided LOWKEY_DB_PATH gets its directory created)
			dbDir := filepath.Dir(def.DBPath)
			if err := os.MkdirAll(dbDir, 0755); err != nil {
				return Config{}, "", fmt.Errorf("failed to create database directory %s: %v", dbDir, err)
			}

			// Env roots replaced the seeded root entirely; otherwise follow an
			// env-overridden download path in memory (the file keeps the
			// persistent path — env values are never baked in).
			if !EnvRootsActive() && len(def.Roots) == 1 {
				def.Roots[0].Path = def.DownloadPath
			}

			Set(def)
			return def, savedPath, nil
		}
		return Config{}, path, fmt.Errorf("failed to read config file at %s: %v", path, err)
	}

	var c Config
	if err := json.Unmarshal(data, &c); err != nil {
		return Config{}, path, fmt.Errorf("failed to parse config JSON: %v", err)
	}

	// Whether the file has an explicit roots key ("roots": null counts as
	// absent). Distinguishes "config predates storage roots" (migrate the
	// download path below) from "user deliberately cleared all roots"
	// (respect the empty list — do not resurrect a Downloads root).
	rootsKeyPresent := false
	{
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(data, &raw); err == nil {
			if v, ok := raw["roots"]; ok && !bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
				rootsKeyPresent = true
			}
		}
	}

	// Merge defaults for any missing fields
	def := defaultConfig()
	needsSave := false
	// needsFullRewrite is set when we must overwrite the file completely (not
	// merge) to remove deprecated keys from disk (e.g. after rootPaths migration).
	needsFullRewrite := false

	// Migrate legacy rootPaths to typed Roots
	if len(c.RootPaths) > 0 && len(c.Roots) == 0 {
		for _, p := range c.RootPaths {
			c.Roots = append(c.Roots, StorageRoot{
				Type:  "local",
				Path:  p,
				Label: p,
			})
		}
		c.RootPaths = nil
		needsSave = true
		needsFullRewrite = true
	}

	if c.DBPath == "" {
		c.DBPath = def.DBPath
		needsSave = true
	}
	if c.Port <= 0 || c.Port > 65535 {
		// 0 = configs predating the field; out-of-range = hand-edited junk.
		c.Port = def.Port
		needsSave = true
	}
	if c.DownloadPath == "" {
		c.DownloadPath = def.DownloadPath
	}
	if c.OllamaBaseURL == "" {
		c.OllamaBaseURL = def.OllamaBaseURL
	}
	if c.OllamaModel == "" {
		c.OllamaModel = def.OllamaModel
	}
	// InferenceProvider migration: configs predating this field need a
	// sensible value. If the user already set RunPod credentials assume
	// they wanted RunPod (matches the old "auto-detect from fields"
	// behavior); otherwise default to ollama so vision tasks keep working.
	if c.InferenceProvider == "" {
		if strings.TrimSpace(c.RunPodEndpoint) != "" && strings.TrimSpace(c.RunPodAPIKey) != "" {
			c.InferenceProvider = "runpod"
		} else {
			c.InferenceProvider = def.InferenceProvider
		}
		needsSave = true
	}
	if c.DescribePrompt == "" {
		c.DescribePrompt = def.DescribePrompt
	}
	if c.EmbeddingModel == "" {
		c.EmbeddingModel = def.EmbeddingModel
		needsSave = true
	}
	if c.EmbeddingProvider == "" {
		c.EmbeddingProvider = def.EmbeddingProvider
		needsSave = true
	}
	if c.EmbeddingPerformance == "" {
		c.EmbeddingPerformance = def.EmbeddingPerformance
		needsSave = true
	}
	if c.AutotagModel == "" {
		c.AutotagModel = def.AutotagModel
		needsSave = true
	}
	if c.AutotagProvider == "" {
		c.AutotagProvider = def.AutotagProvider
		needsSave = true
	}
	if c.AutotagPerformance == "" {
		c.AutotagPerformance = def.AutotagPerformance
		needsSave = true
	}
	if c.FaceModel == "" {
		c.FaceModel = def.FaceModel
		needsSave = true
	}
	if c.FaceProvider == "" {
		c.FaceProvider = def.FaceProvider
		needsSave = true
	}
	if c.FacePerformance == "" {
		c.FacePerformance = def.FacePerformance
		needsSave = true
	}
	if c.FaceRouting == "" {
		c.FaceRouting = def.FaceRouting
		needsSave = true
	}
	if c.OnnxFileTimeoutSeconds == 0 {
		// 0 = unset (pre-existing config). Fill the default so the timeout is on.
		// A negative value is preserved and means "disabled" at use time.
		c.OnnxFileTimeoutSeconds = def.OnnxFileTimeoutSeconds
		needsSave = true
	}
	// Transcription migration: configs predating the section get the full
	// default block (including VADFilter=true). An empty provider is the
	// "section unset" signal, so an explicitly saved VADFilter=false with a
	// provider present is preserved.
	if c.TranscriptionProvider == "" {
		c.TranscriptionProvider = def.TranscriptionProvider
		c.TranscriptionModel = def.TranscriptionModel
		c.TranscriptionLanguage = def.TranscriptionLanguage
		c.TranscriptionVADFilter = def.TranscriptionVADFilter
		needsSave = true
	}
	if c.OnnxTagger.GeneralThreshold == 0 {
		c.OnnxTagger.GeneralThreshold = def.OnnxTagger.GeneralThreshold
	}
	if c.OnnxTagger.CharacterThreshold == 0 {
		c.OnnxTagger.CharacterThreshold = def.OnnxTagger.CharacterThreshold
	}
	// Fill in inference-concurrency defaults for configs predating these
	// fields; treat 0 as "use default" so users can deliberately raise the
	// cap to a large number but won't accidentally land at zero (which
	// would stall the bucket entirely).
	if c.InferenceConcurrency.Ollama <= 0 {
		c.InferenceConcurrency.Ollama = def.InferenceConcurrency.Ollama
	}
	if c.InferenceConcurrency.RunPod <= 0 {
		c.InferenceConcurrency.RunPod = def.InferenceConcurrency.RunPod
	}
	if c.InferenceConcurrency.LMStudio <= 0 {
		c.InferenceConcurrency.LMStudio = def.InferenceConcurrency.LMStudio
	}
	if c.InferenceConcurrency.LlamaCpp <= 0 {
		c.InferenceConcurrency.LlamaCpp = def.InferenceConcurrency.LlamaCpp
	}
	if c.LocalComputeConcurrency <= 0 {
		c.LocalComputeConcurrency = def.LocalComputeConcurrency
	}
	if c.LMStudioBaseURL == "" {
		c.LMStudioBaseURL = def.LMStudioBaseURL
	}
	if c.LlamaCppBaseURL == "" {
		c.LlamaCppBaseURL = def.LlamaCppBaseURL
	}
	if c.JWTSecret == "" {
		c.JWTSecret = uuid.New().String()
		needsSave = true
	}

	// Migrate DownloadPath into a default root for configs that predate the
	// storage-roots system. Persisted so the root becomes a real entry the
	// user can rename or delete; a file that already has an explicit roots
	// key (even an empty list) is left alone.
	migratedDownloadRoot := false
	if !rootsKeyPresent && len(c.Roots) == 0 && c.DownloadPath != "" {
		c.Roots = downloadPathRoot(c.DownloadPath)
		migratedDownloadRoot = true
		needsSave = true
	}

	// Ensure the database directory exists
	dbDir := filepath.Dir(c.DBPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		return Config{}, path, fmt.Errorf("failed to create database directory %s: %v", dbDir, err)
	}

	// Save config if we had to fill in critical missing fields.
	// Use a full rewrite (not a merge) when deprecated keys must be removed.
	if needsSave {
		if needsFullRewrite {
			if saveErr := writeConfigDirect(path, c); saveErr != nil {
				fmt.Printf("Warning: failed to save migrated config: %v\n", saveErr)
			}
		} else {
			if _, saveErr := Save(c); saveErr != nil {
				// Log but don't fail - we can continue with the in-memory config
				fmt.Printf("Warning: failed to save updated config: %v\n", saveErr)
			}
		}
	}

	// Apply environment variable overrides (useful for Docker / container deployments)
	applyEnvOverrides(&c)

	// Env roots replace the migrated root entirely; otherwise follow an
	// env-overridden download path in memory (the file keeps the persistent
	// path — env values are never baked in).
	if migratedDownloadRoot && !EnvRootsActive() && len(c.Roots) == 1 {
		c.Roots[0].Path = c.DownloadPath
	}

	Set(c)
	return c, path, nil
}

// DefaultRoot returns the storage root designated as default.
// Resolution: first root with Default==true, else the first root, else nil.
func DefaultRoot(roots []StorageRoot) *StorageRoot {
	for i := range roots {
		if roots[i].Default {
			return &roots[i]
		}
	}
	if len(roots) > 0 {
		return &roots[0]
	}
	return nil
}

// applyEnvOverrides overrides config fields with environment variables when set.
// Environment variables take highest priority, overriding both defaults and config file values.
func applyEnvOverrides(c *Config) {
	if v := os.Getenv("LOWKEY_DB_PATH"); v != "" {
		c.DBPath = v
	}
	if v := os.Getenv("LOWKEY_PORT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 65535 {
			c.Port = n
		} else {
			log.Printf("Warning: LOWKEY_PORT=%q is not a valid port (1-65535); ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_DOWNLOAD_PATH"); v != "" {
		c.DownloadPath = v
	}
	if v := os.Getenv("LOWKEY_OLLAMA_BASE_URL"); v != "" {
		c.OllamaBaseURL = v
	}
	if v := os.Getenv("LOWKEY_OLLAMA_MODEL"); v != "" {
		c.OllamaModel = v
	}
	if v := os.Getenv("LOWKEY_INFERENCE_PROVIDER"); v != "" {
		c.InferenceProvider = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_RUNPOD_ENDPOINT"); v != "" {
		c.RunPodEndpoint = v
	}
	if v := os.Getenv("LOWKEY_RUNPOD_API_KEY"); v != "" {
		c.RunPodAPIKey = v
	}
	// Per-provider concurrency caps. Parse as int; warn and skip on bad
	// values rather than silently zeroing the bucket.
	if v := os.Getenv("LOWKEY_INFERENCE_OLLAMA_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.InferenceConcurrency.Ollama = n
		} else {
			log.Printf("Warning: LOWKEY_INFERENCE_OLLAMA_CONCURRENCY=%q is not a positive integer; ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_INFERENCE_RUNPOD_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.InferenceConcurrency.RunPod = n
		} else {
			log.Printf("Warning: LOWKEY_INFERENCE_RUNPOD_CONCURRENCY=%q is not a positive integer; ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_LMSTUDIO_BASE_URL"); v != "" {
		c.LMStudioBaseURL = v
	}
	if v := os.Getenv("LOWKEY_LMSTUDIO_MODEL"); v != "" {
		c.LMStudioModel = v
	}
	if v := os.Getenv("LOWKEY_LMSTUDIO_API_KEY"); v != "" {
		c.LMStudioAPIKey = v
	}
	if v := os.Getenv("LOWKEY_LLAMACPP_BASE_URL"); v != "" {
		c.LlamaCppBaseURL = v
	}
	if v := os.Getenv("LOWKEY_LLAMACPP_MODEL"); v != "" {
		c.LlamaCppModel = v
	}
	if v := os.Getenv("LOWKEY_LLAMACPP_API_KEY"); v != "" {
		c.LlamaCppAPIKey = v
	}
	if v := os.Getenv("LOWKEY_INFERENCE_LMSTUDIO_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.InferenceConcurrency.LMStudio = n
		} else {
			log.Printf("Warning: LOWKEY_INFERENCE_LMSTUDIO_CONCURRENCY=%q is not a positive integer; ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_INFERENCE_LLAMACPP_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.InferenceConcurrency.LlamaCpp = n
		} else {
			log.Printf("Warning: LOWKEY_INFERENCE_LLAMACPP_CONCURRENCY=%q is not a positive integer; ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_TRANSCRIPTION_PROVIDER"); v != "" {
		c.TranscriptionProvider = v
	}
	if v := os.Getenv("LOWKEY_TRANSCRIPTION_MODEL"); v != "" {
		c.TranscriptionModel = v
	}
	if v := os.Getenv("LOWKEY_TRANSCRIPTION_LANGUAGE"); v != "" {
		c.TranscriptionLanguage = v
	}
	if v := os.Getenv("LOWKEY_TRANSCRIPTION_VAD"); v != "" {
		switch strings.ToLower(v) {
		case "true", "1", "yes", "on":
			c.TranscriptionVADFilter = true
		case "false", "0", "no", "off":
			c.TranscriptionVADFilter = false
		default:
			log.Printf("Warning: LOWKEY_TRANSCRIPTION_VAD=%q is not a boolean; ignored", v)
		}
	}
	if v := os.Getenv("LOWKEY_JWT_SECRET"); v != "" {
		c.JWTSecret = v
	}
	if v := os.Getenv("LOWKEY_DISCORD_TOKEN"); v != "" {
		c.DiscordToken = v
	}
	if v := os.Getenv("LOWKEY_FASTER_WHISPER_PATH"); v != "" {
		c.FasterWhisperPath = v
	}
	if v := os.Getenv("LOWKEY_EMBEDDING_MODEL"); v != "" {
		c.EmbeddingModel = strings.TrimSpace(v)
	}
	if v := os.Getenv("LOWKEY_EMBEDDING_PROVIDER"); v != "" {
		c.EmbeddingProvider = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_EMBEDDING_PERFORMANCE"); v != "" {
		c.EmbeddingPerformance = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_EMBEDDING_WORKERS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.EmbeddingWorkers = n
		}
	}
	if v := os.Getenv("LOWKEY_EMBEDDING_THREADS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.EmbeddingThreadsPerWorker = n
		}
	}
	if v := os.Getenv("LOWKEY_AUTOTAG_MODEL"); v != "" {
		c.AutotagModel = strings.TrimSpace(v)
	}
	if v := os.Getenv("LOWKEY_AUTOTAG_PROVIDER"); v != "" {
		c.AutotagProvider = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_AUTOTAG_PERFORMANCE"); v != "" {
		c.AutotagPerformance = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_AUTOTAG_WORKERS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.AutotagWorkers = n
		}
	}
	if v := os.Getenv("LOWKEY_AUTOTAG_THREADS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.AutotagThreadsPerWorker = n
		}
	}
	if v := os.Getenv("LOWKEY_FACE_MODEL"); v != "" {
		c.FaceModel = strings.TrimSpace(v)
	}
	if v := os.Getenv("LOWKEY_FACE_PROVIDER"); v != "" {
		c.FaceProvider = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_FACE_PERFORMANCE"); v != "" {
		c.FacePerformance = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_FACE_ROUTING"); v != "" {
		c.FaceRouting = strings.ToLower(strings.TrimSpace(v))
	}
	if v := os.Getenv("LOWKEY_FACE_WORKERS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.FaceWorkers = n
		}
	}
	if v := os.Getenv("LOWKEY_FACE_THREADS"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			c.FaceThreadsPerWorker = n
		}
	}
	if v := os.Getenv("LOWKEY_ONNX_FILE_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			c.OnnxFileTimeoutSeconds = n
		}
	}

	// Storage roots from environment variables.
	// LOWKEY_ROOTS (JSON array) and LOWKEY_ROOT_<N> (simple local paths) are
	// mutually exclusive — if LOWKEY_ROOTS is set it wins; otherwise numbered
	// LOWKEY_ROOT_* vars are collected. Either way, env roots replace any
	// roots from the config file.
	envRootsApplied := false
	if roots, ok := parseEnvRoots(); ok {
		c.Roots = roots
		envRootsApplied = true
	}

	// LOWKEY_DEFAULT_ROOT sets which root is the default destination for
	// uploads and downloads. Value is a 1-based index or a label string.
	// Only applies when using LOWKEY_ROOT_<N> vars (LOWKEY_ROOTS JSON
	// should set "default":true directly on the entry).
	if v := os.Getenv("LOWKEY_DEFAULT_ROOT"); v != "" {
		applyDefaultRoot(c, v)
	}

	// Snapshot the injected set (post-default-flag) so Save can recognize it
	// and keep it out of the config file.
	envRootsMu.Lock()
	envRootsActive = envRootsApplied
	envRootsVal = nil
	if envRootsApplied {
		envRootsVal = append([]StorageRoot(nil), c.Roots...)
	}
	envRootsMu.Unlock()
}

// applyDefaultRoot marks one root as default based on a 1-based index or label.
func applyDefaultRoot(c *Config, v string) {
	// Clear any existing defaults first
	for i := range c.Roots {
		c.Roots[i].Default = false
	}
	// Try as 1-based index
	if idx, err := strconv.Atoi(v); err == nil && idx >= 1 && idx <= len(c.Roots) {
		c.Roots[idx-1].Default = true
		return
	}
	// Try as label match
	for i := range c.Roots {
		if strings.EqualFold(c.Roots[i].Label, v) {
			c.Roots[i].Default = true
			return
		}
	}
	log.Printf("Warning: LOWKEY_DEFAULT_ROOT=%q does not match any root index or label", v)
}

// parseEnvRoots reads storage roots from environment variables.
// Returns the roots and true if any env-based roots were found.
//
// Two formats are supported:
//
//  1. LOWKEY_ROOTS — a JSON array of StorageRoot objects. Supports all fields
//     including S3 configuration. Example:
//     LOWKEY_ROOTS='[{"type":"s3","label":"My Bucket","bucket":"media","endpoint":"https://s3.example.com","region":"us-east-1","accessKey":"AK","secretKey":"SK"}]'
//
//  2. LOWKEY_ROOT_1, LOWKEY_ROOT_2, ... — numbered local path shortcuts.
//     Format: "path" or "path:label". Examples:
//     LOWKEY_ROOT_1=/mnt/photos
//     LOWKEY_ROOT_2=/mnt/videos:Videos
func parseEnvRoots() ([]StorageRoot, bool) {
	// Full JSON takes priority
	if v := os.Getenv("LOWKEY_ROOTS"); v != "" {
		var roots []StorageRoot
		if err := json.Unmarshal([]byte(v), &roots); err != nil {
			log.Printf("Warning: failed to parse LOWKEY_ROOTS JSON: %v", err)
			return nil, false
		}
		return roots, true
	}

	// Collect numbered LOWKEY_ROOT_* vars
	var keys []string
	for _, env := range os.Environ() {
		if k, _, ok := strings.Cut(env, "="); ok && strings.HasPrefix(k, "LOWKEY_ROOT_") {
			keys = append(keys, k)
		}
	}
	if len(keys) == 0 {
		return nil, false
	}

	sort.Strings(keys) // deterministic order: LOWKEY_ROOT_1, LOWKEY_ROOT_2, ...
	var roots []StorageRoot
	for _, k := range keys {
		v := os.Getenv(k)
		if v == "" {
			continue
		}
		path, label, _ := strings.Cut(v, ":")
		if label == "" {
			label = path
		}
		roots = append(roots, StorageRoot{
			Type:  "local",
			Path:  path,
			Label: label,
		})
	}
	return roots, len(roots) > 0
}

// writeConfigDirect writes the config to disk as a clean JSON file, without
// merging with any existing file contents. Used when deprecated keys must be
// removed (e.g. after a migration).
func writeConfigDirect(path string, c Config) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %v", err)
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}
	return os.WriteFile(path, data, 0644)
}

// Save writes the config to disk, creating the directory as needed. Returns the path.
func Save(c Config) (string, error) {
	path, err := getConfigPath()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return path, fmt.Errorf("failed to create config directory: %v", err)
	}
	base := map[string]json.RawMessage{}
	if existing, readErr := os.ReadFile(path); readErr == nil {
		var tmp map[string]json.RawMessage
		if err := json.Unmarshal(existing, &tmp); err == nil {
			base = tmp
		}
	}

	marshaled, err := json.Marshal(c)
	if err != nil {
		return path, fmt.Errorf("failed to marshal config: %v", err)
	}
	incoming := map[string]json.RawMessage{}
	if err := json.Unmarshal(marshaled, &incoming); err != nil {
		return path, fmt.Errorf("failed to map config JSON: %v", err)
	}

	// Env-supplied storage roots must not be baked into the file: when the
	// roots being saved are exactly the env-injected set, keep the file's own
	// roots. Roots that differ were deliberately edited and are persisted
	// (they still lose to the env vars at runtime until those are removed).
	envRootsMu.Lock()
	if envRootsActive && slices.Equal(c.Roots, envRootsVal) {
		delete(incoming, "roots")
	}
	envRootsMu.Unlock()

	deepMergeJSON(base, incoming)

	mergedData, err := json.MarshalIndent(base, "", "  ")
	if err != nil {
		return path, fmt.Errorf("failed to marshal merged config: %v", err)
	}
	if err := os.WriteFile(path, mergedData, 0644); err != nil {
		return path, fmt.Errorf("failed to write config file: %v", err)
	}
	Set(c)
	return path, nil
}
