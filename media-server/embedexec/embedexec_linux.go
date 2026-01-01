//go:build linux
// +build linux

package embedexec

import (
	"embed"
	"os/exec"
)

// Embed Linux binaries from bin_linux directory
//
//go:embed bin_linux/**
var exeFS embed.FS

// binDir is the embedded directory containing platform binaries
const binDir = "bin_linux"

// configureSysProcAttr sets Linux-specific process attributes
// On Linux, we don't need to hide windows, so this is a no-op for most cases.
// Could be extended to set process groups or other attributes if needed.
func configureSysProcAttr(cmd *exec.Cmd) {
	// No special configuration needed on Linux
	// Could add: cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// if we need to manage process groups for signal handling
}
