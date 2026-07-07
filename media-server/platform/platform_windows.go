//go:build windows
// +build windows

package platform

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
)

// HideSubprocessWindow prevents subprocesses from flashing a console window.
// Windows-only effect; no-op on other platforms.
func HideSubprocessWindow(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.HideWindow = true
}

// belowNormalPriorityClass makes the child yield CPU to normal-priority work
// (games, the user's foreground apps) via the OS scheduler itself.
const belowNormalPriorityClass = 0x00004000

// SetBackgroundPriority marks a not-yet-started subprocess to run at
// below-normal CPU priority. Used for scheduler-initiated background jobs so
// their worker processes give the machine back the instant anything else
// wants it. Call before cmd.Start().
func SetBackgroundPriority(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= belowNormalPriorityClass
}

// DeprioritizeStarted lowers the priority of an already-running process.
// No-op on Windows (priority is set at creation via SetBackgroundPriority).
func DeprioritizeStarted(pid int) {}

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
