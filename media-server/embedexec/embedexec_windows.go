//go:build windows
// +build windows

package embedexec

import (
	"embed"
	"os/exec"
	"syscall"
)

// Embed Windows binaries from bin_windows directory
//
//go:embed bin_windows/**
var exeFS embed.FS

// binDir is the embedded directory containing platform binaries
const binDir = "bin_windows"

// configureSysProcAttr sets Windows-specific process attributes
func configureSysProcAttr(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
}
