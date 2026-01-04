//go:build darwin
// +build darwin

package deps

import (
	"os/exec"
)

// configureSysProcAttr sets macOS-specific process attributes.
// On macOS, no special configuration is needed.
func configureSysProcAttr(cmd *exec.Cmd) {
	// No special configuration needed on macOS
}
