//go:build windows
// +build windows

package platform

import (
	"os"
	"os/exec"
	"path/filepath"
)

func getDataDir() string {
	appDataDir := os.Getenv("APPDATA")
	if appDataDir == "" {
		// Fallback for missing APPDATA
		home, err := os.UserHomeDir()
		if err != nil {
			return "."
		}
		return filepath.Join(home, "."+AppName)
	}
	return filepath.Join(appDataDir, AppDisplayName)
}

func getTempDir() string {
	programData := os.Getenv("ProgramData")
	if programData == "" {
		// Fallback to system temp
		return filepath.Join(os.TempDir(), ServerName)
	}
	return filepath.Join(programData, ServerDisplayName, "tmp")
}

func getCacheDir() string {
	// On Windows, cache and data are typically in the same location
	return getDataDir()
}

func binaryExtension() string {
	return ".exe"
}

func sharedLibExtension() string {
	return ".dll"
}

func openFile(path string) error {
	// Use Windows 'start' command to open in default application
	// The empty string after /c start is for the title parameter
	cmd := exec.Command("cmd", "/c", "start", "", path)
	return cmd.Start()
}

func ensureExecutable(path string) error {
	// On Windows, executability is determined by file extension, not permissions
	return nil
}
