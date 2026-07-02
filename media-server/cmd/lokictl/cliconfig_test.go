package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolvePrecedence(t *testing.T) {
	t.Setenv("LOKICTL_TEST_KEY", "")
	if got := resolve("flagv", "LOKICTL_TEST_KEY", "filev", "def"); got != "flagv" {
		t.Errorf("flag should win, got %q", got)
	}
	t.Setenv("LOKICTL_TEST_KEY", "envv")
	if got := resolve("", "LOKICTL_TEST_KEY", "filev", "def"); got != "envv" {
		t.Errorf("env should win over file, got %q", got)
	}
	t.Setenv("LOKICTL_TEST_KEY", "")
	if got := resolve("", "LOKICTL_TEST_KEY", "filev", "def"); got != "filev" {
		t.Errorf("file should win over default, got %q", got)
	}
	if got := resolve("", "LOKICTL_TEST_KEY", "", "def"); got != "def" {
		t.Errorf("default should apply, got %q", got)
	}
}

func TestCLIConfigRoundtrip(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LOKICTL_CONFIG_DIR", dir)

	if cfg := loadCLIConfig(); cfg.Server != "" || cfg.Token != "" {
		t.Fatalf("expected zero config from empty dir, got %+v", cfg)
	}

	path, err := saveCLIConfig(CLIConfig{Server: "http://x:1", Token: "tok"})
	if err != nil {
		t.Fatalf("saveCLIConfig: %v", err)
	}
	if path != filepath.Join(dir, "config.json") {
		t.Errorf("path = %q", path)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("config file missing: %v", err)
	}

	cfg := loadCLIConfig()
	if cfg.Server != "http://x:1" || cfg.Token != "tok" {
		t.Errorf("roundtrip mismatch: %+v", cfg)
	}
}
