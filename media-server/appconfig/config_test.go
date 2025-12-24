package appconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// TestDefaultConfig verifies default configuration values
func TestDefaultConfig(t *testing.T) {
	cfg := defaultConfig()

	if cfg.OllamaBaseURL != "http://localhost:11434" {
		t.Errorf("Default OllamaBaseURL = %q; want %q", cfg.OllamaBaseURL, "http://localhost:11434")
	}

	if cfg.OllamaModel != "llama3.2-vision" {
		t.Errorf("Default OllamaModel = %q; want %q", cfg.OllamaModel, "llama3.2-vision")
	}

	if cfg.DescribePrompt == "" {
		t.Error("Default DescribePrompt should not be empty")
	}

	if cfg.AutotagPrompt == "" {
		t.Error("Default AutotagPrompt should not be empty")
	}

	if cfg.OnnxTagger.GeneralThreshold != 0.35 {
		t.Errorf("Default GeneralThreshold = %f; want 0.35", cfg.OnnxTagger.GeneralThreshold)
	}

	if cfg.OnnxTagger.CharacterThreshold != 0.85 {
		t.Errorf("Default CharacterThreshold = %f; want 0.85", cfg.OnnxTagger.CharacterThreshold)
	}
}

// TestDefaultDownloadPath verifies the download path generation
func TestDefaultDownloadPath(t *testing.T) {
	path := defaultDownloadPath()

	// Should end with "media"
	if filepath.Base(path) != "media" {
		t.Errorf("Default download path should end with 'media'; got %q", path)
	}

	// Should be within user's home directory or be a fallback
	home, err := os.UserHomeDir()
	if err == nil {
		expectedPath := filepath.Join(home, "media")
		if path != expectedPath {
			t.Errorf("Default download path = %q; want %q", path, expectedPath)
		}
	}
}

// TestGetSet verifies Get/Set functions for in-memory config
func TestGetSet(t *testing.T) {
	// Save original and restore after test
	original := Get()
	defer Set(original)

	testConfig := Config{
		DBPath:        "/test/path/db.sqlite",
		DownloadPath:  "/test/downloads",
		OllamaBaseURL: "http://test:11434",
		OllamaModel:   "test-model",
	}

	Set(testConfig)

	retrieved := Get()

	if retrieved.DBPath != testConfig.DBPath {
		t.Errorf("Get().DBPath = %q; want %q", retrieved.DBPath, testConfig.DBPath)
	}
	if retrieved.DownloadPath != testConfig.DownloadPath {
		t.Errorf("Get().DownloadPath = %q; want %q", retrieved.DownloadPath, testConfig.DownloadPath)
	}
	if retrieved.OllamaBaseURL != testConfig.OllamaBaseURL {
		t.Errorf("Get().OllamaBaseURL = %q; want %q", retrieved.OllamaBaseURL, testConfig.OllamaBaseURL)
	}
	if retrieved.OllamaModel != testConfig.OllamaModel {
		t.Errorf("Get().OllamaModel = %q; want %q", retrieved.OllamaModel, testConfig.OllamaModel)
	}
}

// TestIsJSONObject tests the JSON object detection helper
func TestIsJSONObject(t *testing.T) {
	tests := []struct {
		input    string
		expected bool
	}{
		{`{}`, true},
		{`{"key": "value"}`, true},
		{`  {  }  `, true},
		{`[]`, false},
		{`"string"`, false},
		{`123`, false},
		{`null`, false},
		{``, false},
	}

	for _, tt := range tests {
		result := isJSONObject([]byte(tt.input))
		if result != tt.expected {
			t.Errorf("isJSONObject(%q) = %v; want %v", tt.input, result, tt.expected)
		}
	}
}

// TestDeepMergeJSON tests the JSON merge functionality
func TestDeepMergeJSON(t *testing.T) {
	tests := []struct {
		name     string
		dst      string
		src      string
		expected string
	}{
		{
			name:     "Simple merge",
			dst:      `{"a": "1"}`,
			src:      `{"b": "2"}`,
			expected: `{"a":"1","b":"2"}`,
		},
		{
			name:     "Override value",
			dst:      `{"a": "1"}`,
			src:      `{"a": "2"}`,
			expected: `{"a":"2"}`,
		},
		{
			name:     "Nested merge",
			dst:      `{"nested": {"a": "1"}}`,
			src:      `{"nested": {"b": "2"}}`,
			expected: `{"nested":{"a":"1","b":"2"}}`,
		},
		{
			name:     "Add new nested",
			dst:      `{"a": "1"}`,
			src:      `{"nested": {"b": "2"}}`,
			expected: `{"a":"1","nested":{"b":"2"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var dst map[string]json.RawMessage
			var src map[string]json.RawMessage

			json.Unmarshal([]byte(tt.dst), &dst)
			json.Unmarshal([]byte(tt.src), &src)

			deepMergeJSON(dst, src)

			result, _ := json.Marshal(dst)

			// Parse both for comparison (order-independent)
			var resultMap, expectedMap map[string]interface{}
			json.Unmarshal(result, &resultMap)
			json.Unmarshal([]byte(tt.expected), &expectedMap)

			if !mapsEqual(resultMap, expectedMap) {
				t.Errorf("deepMergeJSON result = %s; want %s", result, tt.expected)
			}
		})
	}
}

// mapsEqual compares two maps recursively
func mapsEqual(a, b map[string]interface{}) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		bv, ok := b[k]
		if !ok {
			return false
		}
		if !valuesEqual(v, bv) {
			return false
		}
	}
	return true
}

func valuesEqual(a, b interface{}) bool {
	switch av := a.(type) {
	case map[string]interface{}:
		bv, ok := b.(map[string]interface{})
		if !ok {
			return false
		}
		return mapsEqual(av, bv)
	default:
		return a == b
	}
}

// TestConfigStructFields verifies Config struct has expected fields
func TestConfigStructFields(t *testing.T) {
	cfg := Config{
		DBPath:            "/path/to/db",
		DownloadPath:      "/path/to/downloads",
		OllamaBaseURL:     "http://localhost:11434",
		OllamaModel:       "model",
		DescribePrompt:    "describe prompt",
		AutotagPrompt:     "autotag prompt",
		FasterWhisperPath: "/path/to/whisper",
		DiscordToken:      "token",
	}
	cfg.OnnxTagger.ModelPath = "/model"
	cfg.OnnxTagger.LabelsPath = "/labels"
	cfg.OnnxTagger.ConfigPath = "/config"
	cfg.OnnxTagger.ORTSharedLibraryPath = "/ort"
	cfg.OnnxTagger.GeneralThreshold = 0.5
	cfg.OnnxTagger.CharacterThreshold = 0.9

	// Verify all fields are set
	if cfg.DBPath != "/path/to/db" {
		t.Error("DBPath not set correctly")
	}
	if cfg.OnnxTagger.ModelPath != "/model" {
		t.Error("OnnxTagger.ModelPath not set correctly")
	}
	if cfg.OnnxTagger.GeneralThreshold != 0.5 {
		t.Error("OnnxTagger.GeneralThreshold not set correctly")
	}
}

// TestConfigJSONMarshal verifies Config can be marshaled to JSON
func TestConfigJSONMarshal(t *testing.T) {
	cfg := defaultConfig()
	cfg.DBPath = "/test/db.sqlite"

	data, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("json.Marshal error = %v", err)
	}

	// Verify it's valid JSON
	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("Result is not valid JSON: %v", err)
	}

	// Check expected keys exist
	expectedKeys := []string{"dbPath", "downloadPath", "ollamaBaseUrl", "ollamaModel", "describePrompt", "autotagPrompt", "onnxTagger"}
	for _, key := range expectedKeys {
		if _, ok := parsed[key]; !ok {
			t.Errorf("Expected key %q not found in JSON output", key)
		}
	}
}

// TestConfigJSONUnmarshal verifies Config can be unmarshaled from JSON
func TestConfigJSONUnmarshal(t *testing.T) {
	jsonData := `{
		"dbPath": "/test/db.sqlite",
		"downloadPath": "/test/downloads",
		"ollamaBaseUrl": "http://test:11434",
		"ollamaModel": "test-model",
		"describePrompt": "describe",
		"autotagPrompt": "autotag",
		"onnxTagger": {
			"modelPath": "/model.onnx",
			"labelsPath": "/labels.json",
			"generalThreshold": 0.5,
			"characterThreshold": 0.9
		}
	}`

	var cfg Config
	if err := json.Unmarshal([]byte(jsonData), &cfg); err != nil {
		t.Fatalf("json.Unmarshal error = %v", err)
	}

	if cfg.DBPath != "/test/db.sqlite" {
		t.Errorf("DBPath = %q; want %q", cfg.DBPath, "/test/db.sqlite")
	}
	if cfg.OllamaBaseURL != "http://test:11434" {
		t.Errorf("OllamaBaseURL = %q; want %q", cfg.OllamaBaseURL, "http://test:11434")
	}
	if cfg.OnnxTagger.ModelPath != "/model.onnx" {
		t.Errorf("OnnxTagger.ModelPath = %q; want %q", cfg.OnnxTagger.ModelPath, "/model.onnx")
	}
	if cfg.OnnxTagger.GeneralThreshold != 0.5 {
		t.Errorf("OnnxTagger.GeneralThreshold = %f; want 0.5", cfg.OnnxTagger.GeneralThreshold)
	}
}

// TestConfigConcurrency tests concurrent access to Get/Set
func TestConfigConcurrency(t *testing.T) {
	// Save original and restore after test
	original := Get()
	defer Set(original)

	done := make(chan bool)

	// Writer goroutine
	go func() {
		for i := 0; i < 100; i++ {
			Set(Config{DBPath: "/path"})
		}
		done <- true
	}()

	// Reader goroutine
	go func() {
		for i := 0; i < 100; i++ {
			_ = Get()
		}
		done <- true
	}()

	// Wait for both to complete
	<-done
	<-done
}
