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

func TestConfigRootPaths(t *testing.T) {
	// RootPaths is the deprecated field; verify it still round-trips via JSON
	// (needed for migration testing).
	c := Config{
		DBPath:    "/tmp/test.db",
		JWTSecret: "test-secret",
		RootPaths: []string{"/mnt/media", "/home/user/photos"},
	}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var c2 Config
	if err := json.Unmarshal(data, &c2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(c2.RootPaths) != 2 || c2.RootPaths[0] != "/mnt/media" || c2.RootPaths[1] != "/home/user/photos" {
		t.Fatalf("unexpected RootPaths: %v", c2.RootPaths)
	}
}

func TestConfigRootsDefaultEmpty(t *testing.T) {
	c := defaultConfig()
	if c.Roots == nil {
		t.Fatal("Roots should not be nil, should be empty slice")
	}
	if len(c.Roots) != 0 {
		t.Fatalf("expected empty Roots, got: %v", c.Roots)
	}
}

// TestMigrateRootPathsToRoots verifies that loading a config with the legacy
// rootPaths field migrates it into typed Roots entries and saves back to disk.
func TestMigrateRootPathsToRoots(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	// Write a legacy config that only has rootPaths.
	legacy := map[string]interface{}{
		"dbPath":    filepath.Join(dir, "media.db"),
		"jwtSecret": "test-secret",
		"rootPaths": []string{"/mnt/media", "/home/user/photos"},
	}
	legacyData, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy config: %v", err)
	}
	if err := os.WriteFile(cfgFile, legacyData, 0644); err != nil {
		t.Fatalf("write legacy config: %v", err)
	}

	// Override getConfigPath to point at the temp file.
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if len(c.Roots) != 2 {
		t.Fatalf("expected 2 Roots after migration, got %d", len(c.Roots))
	}
	if c.Roots[0].Type != "local" || c.Roots[0].Path != "/mnt/media" || c.Roots[0].Label != "/mnt/media" {
		t.Errorf("unexpected Roots[0]: %+v", c.Roots[0])
	}
	if c.Roots[1].Type != "local" || c.Roots[1].Path != "/home/user/photos" || c.Roots[1].Label != "/home/user/photos" {
		t.Errorf("unexpected Roots[1]: %+v", c.Roots[1])
	}
	if len(c.RootPaths) != 0 {
		t.Errorf("expected RootPaths cleared after migration, got %v", c.RootPaths)
	}

	// Verify the migrated config was persisted to disk.
	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatalf("read back config: %v", err)
	}
	var saved Config
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatalf("unmarshal saved config: %v", err)
	}
	if len(saved.Roots) != 2 {
		t.Errorf("expected 2 Roots persisted, got %d", len(saved.Roots))
	}
	if len(saved.RootPaths) != 0 {
		t.Errorf("expected rootPaths omitted after migration, got %v", saved.RootPaths)
	}
}

// TestS3ConfigParsing verifies that an S3-typed StorageRoot round-trips through JSON.
func TestS3ConfigParsing(t *testing.T) {
	c := Config{
		DBPath:    "/tmp/test.db",
		JWTSecret: "secret",
		Roots: []StorageRoot{
			{
				Type:            "s3",
				Label:           "My Bucket",
				Endpoint:        "https://s3.example.com",
				Region:          "us-east-1",
				Bucket:          "my-bucket",
				Prefix:          "media/",
				AccessKey:       "AKID",
				SecretKey:       "SECRET",
				ThumbnailPrefix: "thumbs/",
			},
		},
	}

	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var c2 Config
	if err := json.Unmarshal(data, &c2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(c2.Roots) != 1 {
		t.Fatalf("expected 1 root, got %d", len(c2.Roots))
	}
	r := c2.Roots[0]
	if r.Type != "s3" {
		t.Errorf("Type = %q; want %q", r.Type, "s3")
	}
	if r.Bucket != "my-bucket" {
		t.Errorf("Bucket = %q; want %q", r.Bucket, "my-bucket")
	}
	if r.AccessKey != "AKID" {
		t.Errorf("AccessKey = %q; want %q", r.AccessKey, "AKID")
	}
	if r.ThumbnailPrefix != "thumbs/" {
		t.Errorf("ThumbnailPrefix = %q; want %q", r.ThumbnailPrefix, "thumbs/")
	}
	// Path should be empty for S3 roots
	if r.Path != "" {
		t.Errorf("Path = %q; want empty for S3 root", r.Path)
	}
}

// TestApplyEnvOverrides verifies environment variables override config values.
func TestApplyEnvOverrides(t *testing.T) {
	c := defaultConfig()

	// Set env vars
	envs := map[string]string{
		"LOWKEY_DB_PATH":             "/env/db.sqlite",
		"LOWKEY_DOWNLOAD_PATH":       "/env/downloads",
		"LOWKEY_OLLAMA_BASE_URL":     "http://env-ollama:11434",
		"LOWKEY_OLLAMA_MODEL":        "env-model",
		"LOWKEY_JWT_SECRET":          "env-secret",
		"LOWKEY_DISCORD_TOKEN":       "env-discord",
		"LOWKEY_FASTER_WHISPER_PATH": "/env/whisper",
	}
	for k, v := range envs {
		t.Setenv(k, v)
	}

	applyEnvOverrides(&c)

	if c.DBPath != "/env/db.sqlite" {
		t.Errorf("DBPath = %q; want %q", c.DBPath, "/env/db.sqlite")
	}
	if c.DownloadPath != "/env/downloads" {
		t.Errorf("DownloadPath = %q; want %q", c.DownloadPath, "/env/downloads")
	}
	if c.OllamaBaseURL != "http://env-ollama:11434" {
		t.Errorf("OllamaBaseURL = %q; want %q", c.OllamaBaseURL, "http://env-ollama:11434")
	}
	if c.OllamaModel != "env-model" {
		t.Errorf("OllamaModel = %q; want %q", c.OllamaModel, "env-model")
	}
	if c.JWTSecret != "env-secret" {
		t.Errorf("JWTSecret = %q; want %q", c.JWTSecret, "env-secret")
	}
	if c.DiscordToken != "env-discord" {
		t.Errorf("DiscordToken = %q; want %q", c.DiscordToken, "env-discord")
	}
	if c.FasterWhisperPath != "/env/whisper" {
		t.Errorf("FasterWhisperPath = %q; want %q", c.FasterWhisperPath, "/env/whisper")
	}
}

// TestApplyEnvOverridesUnset verifies unset env vars don't change config.
func TestApplyEnvOverridesUnset(t *testing.T) {
	c := defaultConfig()
	original := c.OllamaBaseURL

	applyEnvOverrides(&c)

	if c.OllamaBaseURL != original {
		t.Errorf("OllamaBaseURL changed to %q; should remain %q when env unset", c.OllamaBaseURL, original)
	}
}

// TestParseEnvRootsNumbered verifies LOWKEY_ROOT_N env vars produce local roots.
func TestParseEnvRootsNumbered(t *testing.T) {
	t.Setenv("LOWKEY_ROOT_1", "/mnt/photos")
	t.Setenv("LOWKEY_ROOT_2", "/mnt/videos:Videos")

	roots, ok := parseEnvRoots()
	if !ok {
		t.Fatal("parseEnvRoots returned false; want true")
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots; want 2", len(roots))
	}

	// LOWKEY_ROOT_1: path only — label defaults to path
	if roots[0].Type != "local" || roots[0].Path != "/mnt/photos" || roots[0].Label != "/mnt/photos" {
		t.Errorf("roots[0] = %+v; want local /mnt/photos", roots[0])
	}

	// LOWKEY_ROOT_2: path:label
	if roots[1].Type != "local" || roots[1].Path != "/mnt/videos" || roots[1].Label != "Videos" {
		t.Errorf("roots[1] = %+v; want local /mnt/videos:Videos", roots[1])
	}
}

// TestParseEnvRootsJSON verifies LOWKEY_ROOTS JSON produces full roots.
func TestParseEnvRootsJSON(t *testing.T) {
	jsonRoots := `[
		{"type":"local","path":"/mnt/photos","label":"Photos"},
		{"type":"s3","label":"My Bucket","bucket":"media-bucket","endpoint":"https://s3.example.com","region":"us-east-1","accessKey":"AK","secretKey":"SK","thumbnailPrefix":"thumbs/"}
	]`
	t.Setenv("LOWKEY_ROOTS", jsonRoots)

	roots, ok := parseEnvRoots()
	if !ok {
		t.Fatal("parseEnvRoots returned false; want true")
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots; want 2", len(roots))
	}

	if roots[0].Type != "local" || roots[0].Path != "/mnt/photos" {
		t.Errorf("roots[0] = %+v; want local /mnt/photos", roots[0])
	}

	s3 := roots[1]
	if s3.Type != "s3" {
		t.Errorf("Type = %q; want s3", s3.Type)
	}
	if s3.Bucket != "media-bucket" {
		t.Errorf("Bucket = %q; want media-bucket", s3.Bucket)
	}
	if s3.Endpoint != "https://s3.example.com" {
		t.Errorf("Endpoint = %q; want https://s3.example.com", s3.Endpoint)
	}
	if s3.AccessKey != "AK" || s3.SecretKey != "SK" {
		t.Errorf("credentials wrong: ak=%q sk=%q", s3.AccessKey, s3.SecretKey)
	}
	if s3.ThumbnailPrefix != "thumbs/" {
		t.Errorf("ThumbnailPrefix = %q; want thumbs/", s3.ThumbnailPrefix)
	}
}

// TestParseEnvRootsJSONOverridesNumbered verifies LOWKEY_ROOTS wins over LOWKEY_ROOT_N.
func TestParseEnvRootsJSONOverridesNumbered(t *testing.T) {
	t.Setenv("LOWKEY_ROOTS", `[{"type":"local","path":"/json-path","label":"JSON"}]`)
	t.Setenv("LOWKEY_ROOT_1", "/numbered-path")

	roots, ok := parseEnvRoots()
	if !ok {
		t.Fatal("parseEnvRoots returned false; want true")
	}
	if len(roots) != 1 || roots[0].Path != "/json-path" {
		t.Errorf("LOWKEY_ROOTS should take priority; got %+v", roots)
	}
}

// TestParseEnvRootsNone verifies no env vars returns false.
func TestParseEnvRootsNone(t *testing.T) {
	_, ok := parseEnvRoots()
	if ok {
		t.Error("parseEnvRoots returned true with no env vars set")
	}
}

// TestParseEnvRootsBadJSON verifies malformed JSON is ignored gracefully.
func TestParseEnvRootsBadJSON(t *testing.T) {
	t.Setenv("LOWKEY_ROOTS", "not valid json")

	_, ok := parseEnvRoots()
	if ok {
		t.Error("parseEnvRoots returned true for invalid JSON")
	}
}

// TestApplyEnvOverridesRootsReplacesConfig verifies env roots replace config file roots.
func TestApplyEnvOverridesRootsReplacesConfig(t *testing.T) {
	c := defaultConfig()
	c.Roots = []StorageRoot{
		{Type: "local", Path: "/original", Label: "Original"},
	}

	t.Setenv("LOWKEY_ROOT_1", "/env-path:EnvRoot")

	applyEnvOverrides(&c)

	if len(c.Roots) != 1 || c.Roots[0].Path != "/env-path" || c.Roots[0].Label != "EnvRoot" {
		t.Errorf("env roots should replace config roots; got %+v", c.Roots)
	}
}

// TestLoadNewConfig verifies Load() creates a default config when none exists,
// using an overridden getConfigPath pointing at a temp directory.
func TestLoadNewConfig(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, path, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if path != cfgFile {
		t.Errorf("path = %q; want %q", path, cfgFile)
	}
	if c.OllamaBaseURL != "http://localhost:11434" {
		t.Errorf("OllamaBaseURL = %q; want default", c.OllamaBaseURL)
	}
	if _, err := os.Stat(cfgFile); err != nil {
		t.Errorf("config file not created: %v", err)
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

// TestDefaultRootExplicit verifies DefaultRoot returns the root marked default.
func TestDefaultRootExplicit(t *testing.T) {
	roots := []StorageRoot{
		{Type: "local", Path: "/a", Label: "A"},
		{Type: "local", Path: "/b", Label: "B", Default: true},
		{Type: "local", Path: "/c", Label: "C"},
	}
	r := DefaultRoot(roots)
	if r == nil || r.Path != "/b" {
		t.Errorf("DefaultRoot = %+v; want root B", r)
	}
}

// TestDefaultRootFallsBackToFirst verifies DefaultRoot returns first when none marked.
func TestDefaultRootFallsBackToFirst(t *testing.T) {
	roots := []StorageRoot{
		{Type: "local", Path: "/a", Label: "A"},
		{Type: "local", Path: "/b", Label: "B"},
	}
	r := DefaultRoot(roots)
	if r == nil || r.Path != "/a" {
		t.Errorf("DefaultRoot = %+v; want root A (first)", r)
	}
}

// TestDefaultRootEmpty verifies DefaultRoot returns nil with no roots.
func TestDefaultRootEmpty(t *testing.T) {
	r := DefaultRoot(nil)
	if r != nil {
		t.Errorf("DefaultRoot = %+v; want nil", r)
	}
}

// TestDefaultRootJSONRoundTrip verifies the default field survives JSON encoding.
func TestDefaultRootJSONRoundTrip(t *testing.T) {
	root := StorageRoot{Type: "s3", Label: "Bucket", Bucket: "media", Default: true}
	data, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	var decoded StorageRoot
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if !decoded.Default {
		t.Error("Default field lost after JSON round-trip")
	}
}

// TestLOWKEY_DEFAULT_ROOT_Index verifies LOWKEY_DEFAULT_ROOT with a 1-based index.
func TestLOWKEY_DEFAULT_ROOT_Index(t *testing.T) {
	t.Setenv("LOWKEY_ROOT_1", "/mnt/a:A")
	t.Setenv("LOWKEY_ROOT_2", "/mnt/b:B")
	t.Setenv("LOWKEY_DEFAULT_ROOT", "2")

	c := defaultConfig()
	applyEnvOverrides(&c)

	if len(c.Roots) != 2 {
		t.Fatalf("got %d roots; want 2", len(c.Roots))
	}
	if c.Roots[0].Default {
		t.Error("root 0 should not be default")
	}
	if !c.Roots[1].Default {
		t.Error("root 1 should be default (index 2)")
	}
}

// TestLOWKEY_DEFAULT_ROOT_Label verifies LOWKEY_DEFAULT_ROOT with a label string.
func TestLOWKEY_DEFAULT_ROOT_Label(t *testing.T) {
	t.Setenv("LOWKEY_ROOT_1", "/mnt/a:Alpha")
	t.Setenv("LOWKEY_ROOT_2", "/mnt/b:Beta")
	t.Setenv("LOWKEY_DEFAULT_ROOT", "Beta")

	c := defaultConfig()
	applyEnvOverrides(&c)

	if !c.Roots[1].Default {
		t.Error("root with label Beta should be default")
	}
}

// TestDownloadPathMigration verifies DownloadPath becomes a default root when no roots exist.
func TestDownloadPathMigration(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	// Write a config with only downloadPath, no roots
	legacy := map[string]interface{}{
		"dbPath":       filepath.Join(dir, "media.db"),
		"jwtSecret":    "test-secret",
		"downloadPath": "/my/downloads",
	}
	data, _ := json.Marshal(legacy)
	os.WriteFile(cfgFile, data, 0644)

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if len(c.Roots) != 1 {
		t.Fatalf("expected 1 root from migration, got %d", len(c.Roots))
	}
	if c.Roots[0].Path != "/my/downloads" {
		t.Errorf("migrated root path = %q; want /my/downloads", c.Roots[0].Path)
	}
	if !c.Roots[0].Default {
		t.Error("migrated root should be marked default")
	}
}
