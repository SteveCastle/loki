package appconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
	"github.com/stevecastle/shrike/platform"
)

// Config holds application configuration including database path, LLM prompts, and AI model paths.
type Config struct {
	DBPath string `json:"dbPath"`

	// Download path for media files
	DownloadPath string `json:"downloadPath"`

	// Ollama / LLM settings
	OllamaBaseURL  string `json:"ollamaBaseUrl"`
	OllamaModel    string `json:"ollamaModel"`
	DescribePrompt string `json:"describePrompt"`
	AutotagPrompt  string `json:"autotagPrompt"`

	// ONNX tagger settings
	OnnxTagger struct {
		ModelPath            string  `json:"modelPath"`
		LabelsPath           string  `json:"labelsPath"`
		ConfigPath           string  `json:"configPath"`
		ORTSharedLibraryPath string  `json:"ortSharedLibraryPath"`
		GeneralThreshold     float64 `json:"generalThreshold"`
		CharacterThreshold   float64 `json:"characterThreshold"`
	} `json:"onnxTagger"`

	// Optional path to faster-whisper executable
	FasterWhisperPath string `json:"fasterWhisperPath"`

	// Discord authentication token for media export
	DiscordToken string `json:"discordToken"`

	// JWT Secret for authentication
	JWTSecret string `json:"jwtSecret"`
}

var (
	cfgMu sync.RWMutex
	cfg   Config
)

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
		DBPath:         DefaultDBPath(),
		DownloadPath:   defaultDownloadPath(),
		OllamaBaseURL:  "http://localhost:11434",
		OllamaModel:    "llama3.2-vision",
		DescribePrompt: "Please describe this image, paying special attention to the people, the color of hair, clothing, items, text and captions, and actions being performed.",
		AutotagPrompt:  "Please analyze this image and select the most appropriate tags from the following list. Return your response as a JSON array containing objects with \"label\" and \"category\" fields.\n\n%s\n\nLook at the image carefully and select only the tags that accurately describe what you see. Focus on:\n- Objects and subjects visible in the image\n- Colors and visual characteristics\n- Composition and style elements\n- Setting or environment\n- Actions or activities if present\n\nReturn your response in this exact JSON format:\n[{\"label\": \"tag_name\", \"category\": \"category_name\"}]\n\nOnly select tags that clearly apply to this image. If no tags from the list match what you see, return an empty array [].",
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
	}
}

// Get returns a copy of the current in-memory config.
func Get() Config {
	cfgMu.RLock()
	defer cfgMu.RUnlock()
	return cfg
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
func getConfigPath() (string, error) {
	configDir := DefaultConfigDir()
	return filepath.Join(configDir, "config.json"), nil
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

			// Ensure the database directory exists
			dbDir := filepath.Dir(def.DBPath)
			if err := os.MkdirAll(dbDir, 0755); err != nil {
				return Config{}, "", fmt.Errorf("failed to create database directory %s: %v", dbDir, err)
			}

			// Save the default config
			savedPath, saveErr := Save(def)
			if saveErr != nil {
				return Config{}, path, fmt.Errorf("failed to create default config file: %v", saveErr)
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

	// Merge defaults for any missing fields
	def := defaultConfig()
	needsSave := false

	if c.DBPath == "" {
		c.DBPath = def.DBPath
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
	if c.DescribePrompt == "" {
		c.DescribePrompt = def.DescribePrompt
	}
	if c.AutotagPrompt == "" {
		c.AutotagPrompt = def.AutotagPrompt
	}
	if c.OnnxTagger.GeneralThreshold == 0 {
		c.OnnxTagger.GeneralThreshold = def.OnnxTagger.GeneralThreshold
	}
	if c.OnnxTagger.CharacterThreshold == 0 {
		c.OnnxTagger.CharacterThreshold = def.OnnxTagger.CharacterThreshold
	}
	if c.JWTSecret == "" {
		c.JWTSecret = uuid.New().String()
		needsSave = true
	}

	// Ensure the database directory exists
	dbDir := filepath.Dir(c.DBPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		return Config{}, path, fmt.Errorf("failed to create database directory %s: %v", dbDir, err)
	}

	// Save config if we had to fill in critical missing fields
	if needsSave {
		if _, saveErr := Save(c); saveErr != nil {
			// Log but don't fail - we can continue with the in-memory config
			fmt.Printf("Warning: failed to save updated config: %v\n", saveErr)
		}
	}

	Set(c)
	return c, path, nil
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
