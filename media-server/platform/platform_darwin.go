//go:build darwin
// +build darwin

package platform

import (
	"os"
	"os/exec"
	"path/filepath"
)

func getDataDir() string {
	// On macOS, use ~/Library/Application Support/AppName
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, "Library", "Application Support", AppDisplayName)
}

func getTempDir() string {
	// Use TMPDIR if available, otherwise /tmp
	tmpDir := os.Getenv("TMPDIR")
	if tmpDir != "" {
		return filepath.Join(tmpDir, ServerName)
	}
	return filepath.Join("/tmp", ServerName)
}

func getCacheDir() string {
	// On macOS, use ~/Library/Caches/AppName
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return filepath.Join(home, "Library", "Caches", AppName)
}

func binaryExtension() string {
	return ""
}

func sharedLibExtension() string {
	return ".dylib"
}

func openFile(path string) error {
	// Use macOS 'open' command to open with default application
	cmd := exec.Command("open", path)
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
