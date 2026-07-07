//go:build darwin
// +build darwin

package platform

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
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

// HideSubprocessWindow is a no-op on macOS (no console window concept).
func HideSubprocessWindow(_ *exec.Cmd) {}

// SetBackgroundPriority is a pre-start no-op on this platform; priority is
// applied after start via DeprioritizeStarted (setpriority needs a PID).
func SetBackgroundPriority(_ *exec.Cmd) {}

// DeprioritizeStarted renices an already-started child to a background
// priority so it yields CPU to the user's foreground work. Best-effort.
func DeprioritizeStarted(pid int) {
	_ = syscall.Setpriority(syscall.PRIO_PROCESS, pid, 15)
}
