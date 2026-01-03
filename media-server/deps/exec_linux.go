//go:build linux
// +build linux

package deps

import (
	"os/exec"
)

// configureSysProcAttr sets Linux-specific process attributes.
// On Linux, no special configuration is needed.
func configureSysProcAttr(cmd *exec.Cmd) {
	// No special configuration needed on Linux
}
