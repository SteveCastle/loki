//go:build linux
// +build linux

package platform

import (
	"os"
	"os/exec"
	"path/filepath"
)

func getDataDir() string {
	// Follow XDG Base Directory Specification
	xdgDataHome := os.Getenv("XDG_DATA_HOME")
	if xdgDataHome != "" {
		return filepath.Join(xdgDataHome, AppName)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".local", "share", AppName)
}

func getTempDir() string {
	// Try XDG_RUNTIME_DIR first (typically /run/user/UID)
	xdgRuntimeDir := os.Getenv("XDG_RUNTIME_DIR")
	if xdgRuntimeDir != "" {
		return filepath.Join(xdgRuntimeDir, ServerName)
	}

	// Fall back to /tmp
	return filepath.Join("/tmp", ServerName)
}

func getCacheDir() string {
	// Follow XDG Base Directory Specification
	xdgCacheHome := os.Getenv("XDG_CACHE_HOME")
	if xdgCacheHome != "" {
		return filepath.Join(xdgCacheHome, AppName)
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, ".cache", AppName)
}

func binaryExtension() string {
	return ""
}

func sharedLibExtension() string {
	return ".so"
}

func openFile(path string) error {
	// Use xdg-open on Linux to open with default application
	cmd := exec.Command("xdg-open", path)
	return cmd.Start()
}

func ensureExecutable(path string) error {
	// Set executable permissions (owner, group, others can execute)
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	// Add executable bit for owner
	return os.Chmod(path, info.Mode()|0111)
}
