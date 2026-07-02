package main

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// CLIConfig is the persisted client configuration written by "lokictl login".
type CLIConfig struct {
	Server string `json:"server,omitempty"`
	Token  string `json:"token,omitempty"`
}

// cliConfigDir honors LOKICTL_CONFIG_DIR (tests, portable installs), falling
// back to the platform user-config directory.
func cliConfigDir() string {
	if d := os.Getenv("LOKICTL_CONFIG_DIR"); d != "" {
		return d
	}
	base, err := os.UserConfigDir()
	if err != nil {
		home, herr := os.UserHomeDir()
		if herr != nil {
			return "."
		}
		base = home
	}
	return filepath.Join(base, "lokictl")
}

func cliConfigPath() string { return filepath.Join(cliConfigDir(), "config.json") }

// loadCLIConfig returns the zero config when the file is absent or unreadable
// — every field has an env/flag/default fallback.
func loadCLIConfig() CLIConfig {
	var cfg CLIConfig
	b, err := os.ReadFile(cliConfigPath())
	if err != nil {
		return cfg
	}
	_ = json.Unmarshal(b, &cfg)
	return cfg
}

// saveCLIConfig writes the config with owner-only permissions (the token is a
// credential) and returns the path written.
func saveCLIConfig(cfg CLIConfig) (string, error) {
	dir := cliConfigDir()
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return "", err
	}
	path := cliConfigPath()
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		return "", err
	}
	return path, nil
}

// resolve implements the precedence chain flag > env > config file > default.
func resolve(flagV, envKey, fileV, def string) string {
	if flagV != "" {
		return flagV
	}
	if v := os.Getenv(envKey); v != "" {
		return v
	}
	if fileV != "" {
		return fileV
	}
	return def
}
