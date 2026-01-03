//go:build windows
// +build windows

package deps

import (
	"os/exec"
	"syscall"
)

// configureSysProcAttr sets Windows-specific process attributes.
// Hides the console window when running executables.
func configureSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}
