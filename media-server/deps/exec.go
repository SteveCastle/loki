package deps

import (
	"context"
	"fmt"
	"os/exec"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/platform"
)

// GetExec builds an exec.Cmd for a dependency executable. Looks up by
// bundled id first (depID then exeName), then PATH. Optional tools
// (yt-dlp, gallery-dl, dce) aren't in the bundled manifest, so the
// PATH fallback is the intended route for them.
func GetExec(ctx context.Context, depID string, exeName string, args ...string) (*exec.Cmd, error) {
	for _, id := range []string{depID, exeName} {
		if id == "" {
			continue
		}
		if exePath, err := bundled.Resolve(id); err == nil {
			cmd := exec.CommandContext(ctx, exePath, args...)
			configureSysProcAttr(cmd)
			return cmd, nil
		}
	}
	systemPath, lookupErr := exec.LookPath(exeName)
	if lookupErr != nil {
		return nil, fmt.Errorf("executable %q not found in bundled registry or system PATH: %v", exeName, lookupErr)
	}
	cmd := exec.CommandContext(ctx, systemPath, args...)
	configureSysProcAttr(cmd)
	return cmd, nil
}

// GetExecutableName returns the platform-specific executable name.
func GetExecutableName(baseName string) string {
	return baseName + platform.BinaryExtension()
}
