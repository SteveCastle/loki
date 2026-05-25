//go:build darwin

package bundled

import (
	"os/exec"
	"time"
)

func removeQuarantine(path string) {
	cmd := exec.Command("xattr", "-d", "com.apple.quarantine", path)
	done := make(chan struct{})
	_ = cmd.Start()
	go func() { _ = cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		_ = cmd.Process.Kill()
	}
}
