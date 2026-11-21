package appconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// TestDefaultConfig verifies default configuration values
func TestDefaultConfig(t *testing.T) {
	cfg := defaultConfig()

	// Text generation runs loki-scribe: no model configured means the engine
	// default; the batching knobs have defaults.
	if cfg.TextModel != "" {
		t.Errorf("Default TextModel = %q; want empty (engine default)", cfg.TextModel)
	}
	if cfg.TextParallel != DefaultTextParallel || cfg.TextBatch != DefaultTextBatch || cfg.TextMaxImageTokens != DefaultTextMaxImageTokens {
		t.Errorf("Default text knobs = %d/%d/%d", cfg.TextParallel, cfg.TextBatch, cfg.TextMaxImageTokens)
	}

	if cfg.DescribePrompt == "" {
		t.Error("Default DescribePrompt should not be empty")
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
		DBPath:       "/test/path/db.sqlite",
		DownloadPath: "/test/downloads",
		TextModel:    "test-model",
	}

	Set(testConfig)

	retrieved := Get()

	if retrieved.DBPath != testConfig.DBPath {
		t.Errorf("Get().DBPath = %q; want %q", retrieved.DBPath, testConfig.DBPath)
	}
	if retrieved.DownloadPath != testConfig.DownloadPath {
		t.Errorf("Get().DownloadPath = %q; want %q", retrieved.DownloadPath, testConfig.DownloadPath)
	}
	if retrieved.TextModel != testConfig.TextModel {
		t.Errorf("Get().TextModel = %q; want %q", retrieved.TextModel, testConfig.TextModel)
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
		TextModel:         "model",
		DescribePrompt:    "describe prompt",
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
	expectedKeys := []string{"dbPath", "downloadPath", "textModel", "describePrompt", "onnxTagger"}
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
		"textModel": "org/repo/model.gguf",
		"describePrompt": "describe",
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
	if cfg.TextModel != "org/repo/model.gguf" {
		t.Errorf("TextModel = %q; want %q", cfg.TextModel, "org/repo/model.gguf")
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
		"LOWKEY_DB_PATH":               "/env/db.sqlite",
		"LOWKEY_PORT":                  "9999",
		"LOWKEY_DOWNLOAD_PATH":         "/env/downloads",
		"LOWKEY_TEXT_MODEL":            "org/repo/env.gguf",
		"LOWKEY_TEXT_MMPROJ":           "/env/mmproj.gguf",
		"LOWKEY_TEXT_PARALLEL":         "4",
		"LOWKEY_TEXT_BATCH":            "32",
		"LOWKEY_TEXT_MAX_IMAGE_TOKENS": "1024",
		"LOWKEY_JWT_SECRET":            "env-secret",
		"LOWKEY_DISCORD_TOKEN":         "env-discord",
		"LOWKEY_FASTER_WHISPER_PATH":   "/env/whisper",
	}
	for k, v := range envs {
		t.Setenv(k, v)
	}

	applyEnvOverrides(&c)

	if c.DBPath != "/env/db.sqlite" {
		t.Errorf("DBPath = %q; want %q", c.DBPath, "/env/db.sqlite")
	}
	if c.Port != 9999 {
		t.Errorf("Port = %d; want 9999", c.Port)
	}
	if c.DownloadPath != "/env/downloads" {
		t.Errorf("DownloadPath = %q; want %q", c.DownloadPath, "/env/downloads")
	}
	if c.TextModel != "org/repo/env.gguf" || c.TextMMProj != "/env/mmproj.gguf" {
		t.Errorf("TextModel/TextMMProj = %q/%q; want the env values", c.TextModel, c.TextMMProj)
	}
	if c.TextParallel != 4 || c.TextBatch != 32 || c.TextMaxImageTokens != 1024 {
		t.Errorf("text knobs = %d/%d/%d; want 4/32/1024", c.TextParallel, c.TextBatch, c.TextMaxImageTokens)
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

// TestApplyEnvOverridesInvalidPort verifies a malformed LOWKEY_PORT is ignored.
func TestApplyEnvOverridesInvalidPort(t *testing.T) {
	for _, bad := range []string{"loki", "0", "-1", "70000"} {
		c := defaultConfig()
		t.Setenv("LOWKEY_PORT", bad)
		applyEnvOverrides(&c)
		if c.Port != DefaultPort {
			t.Errorf("LOWKEY_PORT=%q: Port = %d; want default %d", bad, c.Port, DefaultPort)
		}
	}
}

// TestDefaultPort verifies the default config carries the leet port and the
// derived address helpers agree with it.
func TestDefaultPort(t *testing.T) {
	c := defaultConfig()
	if c.Port != 10111 {
		t.Errorf("Port = %d; want 10111", c.Port)
	}
	if got := c.ListenAddr(); got != ":10111" {
		t.Errorf("ListenAddr() = %q; want %q", got, ":10111")
	}
	if got := c.LocalBaseURL(); got != "http://localhost:10111" {
		t.Errorf("LocalBaseURL() = %q; want %q", got, "http://localhost:10111")
	}
}

// TestApplyEnvOverridesUnset verifies unset env vars don't change config.
func TestApplyEnvOverridesUnset(t *testing.T) {
	c := defaultConfig()
	original := c.TextModel

	applyEnvOverrides(&c)

	if c.TextModel != original {
		t.Errorf("TextModel changed to %q; should remain %q when env unset", c.TextModel, original)
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
	if c.TextBatch != DefaultTextBatch {
		t.Errorf("TextBatch = %d; want default", c.TextBatch)
	}
	if _, err := os.Stat(cfgFile); err != nil {
		t.Errorf("config file not created: %v", err)
	}
}

// TestLoadNewConfigAppliesEnvOverrides verifies env vars are honored on the
// very first Load (no config file yet) — a fresh Docker container must come
// up with LOWKEY_DB_PATH / LOWKEY_ROOTS applied, not pure defaults.
func TestLoadNewConfigAppliesEnvOverrides(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	envDB := filepath.Join(dir, "db", "media.db")
	t.Setenv("LOWKEY_DB_PATH", envDB)
	t.Setenv("LOWKEY_ROOTS", `[{"type":"local","path":"/mnt/photos","label":"Photos","default":true}]`)

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.DBPath != envDB {
		t.Errorf("DBPath = %q; want env override %q", c.DBPath, envDB)
	}
	if len(c.Roots) != 1 || c.Roots[0].Label != "Photos" {
		t.Errorf("Roots = %+v; want the single LOWKEY_ROOTS entry", c.Roots)
	}
	if _, err := os.Stat(filepath.Dir(envDB)); err != nil {
		t.Errorf("db directory for env override not created: %v", err)
	}

	// The file on disk should hold defaults, not env values — env stays env.
	data, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatalf("read config file: %v", err)
	}
	var onDisk Config
	if err := json.Unmarshal(data, &onDisk); err != nil {
		t.Fatalf("parse config file: %v", err)
	}
	if onDisk.DBPath == envDB {
		t.Errorf("env override leaked into persisted config: %q", onDisk.DBPath)
	}
}

// TestTranscriptionDefaultsMigration verifies configs predating the
// transcription section get the full default block, while an explicitly
// saved VADFilter=false (with provider present) is preserved.
func TestTranscriptionDefaultsMigration(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	// Pre-existing config with no transcription section.
	if err := os.WriteFile(cfgFile, []byte(`{"dbPath":"`+filepath.ToSlash(filepath.Join(dir, "m.db"))+`"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.TranscriptionProvider != DefaultTranscriptionProvider ||
		c.TranscriptionModel != DefaultTranscriptionModel ||
		c.TranscriptionLanguage != "en" || !c.TranscriptionVADFilter {
		t.Errorf("migration defaults wrong: %+v", c)
	}

	// Explicit VADFilter=false with provider set must survive a reload.
	c.TranscriptionVADFilter = false
	if _, err := Save(c); err != nil {
		t.Fatalf("Save: %v", err)
	}
	c2, _, err := Load()
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if c2.TranscriptionVADFilter {
		t.Error("explicit VADFilter=false was clobbered by defaults on reload")
	}
}

func TestTranscriptionEnvOverrides(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	t.Setenv("LOWKEY_TRANSCRIPTION_PROVIDER", "custom")
	t.Setenv("LOWKEY_TRANSCRIPTION_MODEL", "tiny")
	t.Setenv("LOWKEY_TRANSCRIPTION_LANGUAGE", "de")
	t.Setenv("LOWKEY_TRANSCRIPTION_VAD", "false")
	t.Setenv("LOWKEY_TRANSCRIPTION_INITIAL_PROMPT", "Names: Lowkey.")
	t.Setenv("LOWKEY_TRANSCRIPTION_HOTWORDS", "SigLIP")
	t.Setenv("LOWKEY_TRANSCRIPTION_VOCAL_EXTRACT", "true")

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.TranscriptionProvider != "custom" || c.TranscriptionModel != "tiny" ||
		c.TranscriptionLanguage != "de" || c.TranscriptionVADFilter {
		t.Errorf("env overrides not applied: %+v", c)
	}
	if c.TranscriptionInitialPrompt != "Names: Lowkey." || c.TranscriptionHotwords != "SigLIP" ||
		!c.TranscriptionVocalExtract {
		t.Errorf("vocabulary env overrides not applied: %+v", c)
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

// TestLegacyTextModelMigration: a config of the HTTP-provider era keeps its
// local GGUF server's model as the text model (LM Studio, llama.cpp, Strata),
// gets none from Ollama / RunPod, and loses the retired keys on save.
func TestLegacyTextModelMigration(t *testing.T) {
	cases := []struct {
		name   string
		legacy map[string]interface{}
		want   string
	}{
		{"lmstudio", map[string]interface{}{"inferenceProvider": "lmstudio", "lmstudioModel": "qwen3.8-27b-uncensored", "ollamaModel": "llava"}, "qwen3.8-27b-uncensored"},
		{"llamacpp", map[string]interface{}{"inferenceProvider": "llamacpp", "llamacppModel": "org/repo/m.gguf"}, "org/repo/m.gguf"},
		{"ollama", map[string]interface{}{"inferenceProvider": "ollama", "ollamaModel": "llava", "lmstudioModel": "unused"}, ""},
		{"runpod", map[string]interface{}{"inferenceProvider": "runpod", "runpodEndpoint": "https://api.runpod.ai/v2/x/run", "runpodApiKey": "k"}, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			cfgFile := filepath.Join(dir, "config.json")
			tc.legacy["dbPath"] = filepath.Join(dir, "media.db")
			tc.legacy["jwtSecret"] = "test-secret"
			data, _ := json.Marshal(tc.legacy)
			os.WriteFile(cfgFile, data, 0644)

			orig := getConfigPath
			defer func() { getConfigPath = orig }()
			getConfigPath = func() (string, error) { return cfgFile, nil }

			c, _, err := Load()
			if err != nil {
				t.Fatalf("Load: %v", err)
			}
			if c.TextModel != tc.want {
				t.Errorf("TextModel = %q; want %q", c.TextModel, tc.want)
			}
			if _, err := Save(c); err != nil {
				t.Fatalf("Save: %v", err)
			}
			var onDisk map[string]json.RawMessage
			raw, _ := os.ReadFile(cfgFile)
			_ = json.Unmarshal(raw, &onDisk)
			for _, k := range retiredKeys {
				if _, ok := onDisk[k]; ok {
					t.Errorf("retired key %q still in the saved file", k)
				}
			}
			if tc.want != "" && string(onDisk["textModel"]) != strconv.Quote(tc.want) {
				t.Errorf("saved textModel = %s; want %q", onDisk["textModel"], tc.want)
			}
		})
	}
}

// TestDownloadPathMigrationPersisted verifies that a config predating storage
// roots (no "roots" key) gets a Downloads root migrated from downloadPath AND
// that the root is written to disk, so it behaves like a normal editable root.
func TestDownloadPathMigrationPersisted(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")
	legacy := map[string]interface{}{
		"dbPath":       filepath.Join(dir, "media.db"),
		"jwtSecret":    "test-secret",
		"downloadPath": filepath.Join(dir, "dl"),
	}
	data, _ := json.Marshal(legacy)
	if err := os.WriteFile(cfgFile, data, 0644); err != nil {
		t.Fatal(err)
	}
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(c.Roots) != 1 || c.Roots[0].Label != "Downloads" || !c.Roots[0].Default {
		t.Fatalf("expected migrated Downloads root, got %+v", c.Roots)
	}

	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatal(err)
	}
	var saved Config
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Roots) != 1 || saved.Roots[0].Label != "Downloads" {
		t.Errorf("migrated root not persisted; file roots = %+v", saved.Roots)
	}
}

// TestClearedRootsNotResurrected verifies that a deliberately emptied roots
// list ("roots": []) stays empty — Load must not regenerate a Downloads root.
func TestClearedRootsNotResurrected(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")
	cleared := map[string]interface{}{
		"dbPath":       filepath.Join(dir, "media.db"),
		"jwtSecret":    "test-secret",
		"downloadPath": filepath.Join(dir, "dl"),
		"roots":        []interface{}{},
	}
	data, _ := json.Marshal(cleared)
	if err := os.WriteFile(cfgFile, data, 0644); err != nil {
		t.Fatal(err)
	}
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(c.Roots) != 0 {
		t.Errorf("cleared roots resurrected: %+v", c.Roots)
	}
}

// TestFirstRunSeedsDownloadsRootInFile verifies a brand-new config file
// carries the Downloads root on disk (not just in memory).
func TestFirstRunSeedsDownloadsRootInFile(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(c.Roots) != 1 || c.Roots[0].Label != "Downloads" || !c.Roots[0].Default {
		t.Fatalf("expected seeded Downloads root, got %+v", c.Roots)
	}

	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatal(err)
	}
	var saved Config
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Roots) != 1 || saved.Roots[0].Label != "Downloads" {
		t.Errorf("seeded root not persisted; file roots = %+v", saved.Roots)
	}
}

// TestEnvRootsNotBakedOnSave verifies that Save keeps env-injected roots out
// of the config file: the file's own roots survive a save round-trip while
// LOWKEY_ROOTS is active.
func TestEnvRootsNotBakedOnSave(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")
	onDisk := map[string]interface{}{
		"dbPath":    filepath.Join(dir, "media.db"),
		"jwtSecret": "test-secret",
		"roots": []map[string]interface{}{
			{"type": "local", "path": "/persistent", "label": "Persistent", "default": true},
		},
	}
	data, _ := json.Marshal(onDisk)
	if err := os.WriteFile(cfgFile, data, 0644); err != nil {
		t.Fatal(err)
	}
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }
	t.Setenv("LOWKEY_ROOTS", `[{"type":"local","path":"/env-root","label":"Env"}]`)

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !EnvRootsActive() {
		t.Fatal("EnvRootsActive() = false, want true")
	}
	if len(c.Roots) != 1 || c.Roots[0].Label != "Env" {
		t.Fatalf("runtime roots should be the env set, got %+v", c.Roots)
	}

	// Simulate the config POST: save the loaded (env-overridden) config back.
	if _, err := Save(c); err != nil {
		t.Fatalf("Save: %v", err)
	}
	raw, err := os.ReadFile(cfgFile)
	if err != nil {
		t.Fatal(err)
	}
	var saved Config
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Roots) != 1 || saved.Roots[0].Label != "Persistent" {
		t.Errorf("env roots were baked into the file; file roots = %+v", saved.Roots)
	}

	// Deliberately edited roots (different from the env set) DO persist.
	c.Roots = []StorageRoot{{Type: "local", Path: "/edited", Label: "Edited"}}
	if _, err := Save(c); err != nil {
		t.Fatalf("Save edited: %v", err)
	}
	raw, _ = os.ReadFile(cfgFile)
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Roots) != 1 || saved.Roots[0].Label != "Edited" {
		t.Errorf("deliberate root edit not persisted; file roots = %+v", saved.Roots)
	}
}

// TestMigratedRootFollowsEnvDownloadPath verifies the in-memory migrated root
// tracks an env-overridden download path while the file keeps the persistent
// (non-env) path.
func TestMigratedRootFollowsEnvDownloadPath(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")
	fileDL := filepath.Join(dir, "file-dl")
	envDL := filepath.Join(dir, "env-dl")
	legacy := map[string]interface{}{
		"dbPath":       filepath.Join(dir, "media.db"),
		"jwtSecret":    "test-secret",
		"downloadPath": fileDL,
	}
	data, _ := json.Marshal(legacy)
	if err := os.WriteFile(cfgFile, data, 0644); err != nil {
		t.Fatal(err)
	}
	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }
	t.Setenv("LOWKEY_DOWNLOAD_PATH", envDL)

	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(c.Roots) != 1 || c.Roots[0].Path != envDL {
		t.Errorf("in-memory migrated root should follow env download path %q, got %+v", envDL, c.Roots)
	}

	raw, _ := os.ReadFile(cfgFile)
	var saved Config
	if err := json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	if len(saved.Roots) != 1 || saved.Roots[0].Path != fileDL {
		t.Errorf("file root should keep the persistent path %q, got %+v", fileDL, saved.Roots)
	}
}

func TestAllowPublicAccessRoundTripAndEnvOverride(t *testing.T) {
	dir := t.TempDir()
	cfgFile := filepath.Join(dir, "config.json")

	orig := getConfigPath
	defer func() { getConfigPath = orig }()
	getConfigPath = func() (string, error) { return cfgFile, nil }

	// Default is off.
	c, _, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if c.AllowPublicAccess {
		t.Error("AllowPublicAccess should default to false")
	}

	// Save with the flag on, reload, expect it back.
	c.AllowPublicAccess = true
	if _, err := Save(c); err != nil {
		t.Fatalf("Save: %v", err)
	}
	c2, _, err := Load()
	if err != nil {
		t.Fatalf("reload: %v", err)
	}
	if !c2.AllowPublicAccess {
		t.Error("AllowPublicAccess did not survive Save/Load round-trip")
	}

	// Env override forces it off despite the saved true.
	t.Setenv("LOWKEY_ALLOW_PUBLIC_ACCESS", "off")
	c3, _, err := Load()
	if err != nil {
		t.Fatalf("reload with env: %v", err)
	}
	if c3.AllowPublicAccess {
		t.Error("LOWKEY_ALLOW_PUBLIC_ACCESS=off should override the saved true")
	}

	// Garbage env value is ignored (saved value wins).
	t.Setenv("LOWKEY_ALLOW_PUBLIC_ACCESS", "banana")
	c4, _, err := Load()
	if err != nil {
		t.Fatalf("reload with garbage env: %v", err)
	}
	if !c4.AllowPublicAccess {
		t.Error("garbage env value should be ignored, keeping the saved true")
	}
}
