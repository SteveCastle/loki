package deps

import (
	"context"
	"fmt"
	"os/exec"

	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/platform"
)

// GetExec builds an exec.Cmd for a dependency executable.
// It looks up the executable path via the bundled binary registry first,
// then falls back to system PATH.
//
// Parameters:
//   - ctx: context for the command
//   - depID: the dependency ID (e.g., "ffmpeg", "yt-dlp")
//   - exeName: the base name of the executable (without extension)
//   - args: command arguments
func GetExec(ctx context.Context, depID string, exeName string, args ...string) (*exec.Cmd, error) {
	fullName := exeName + platform.BinaryExtension()
	exePath, err := bundled.Resolve(fullName)
	if err != nil {
		// Fall back to system PATH
		systemPath, lookupErr := exec.LookPath(exeName)
		if lookupErr != nil {
			return nil, fmt.Errorf("executable %q not found in bundled registry or system PATH: %v", exeName, lookupErr)
		}
		cmd := exec.CommandContext(ctx, systemPath, args...)
		configureSysProcAttr(cmd)
		return cmd, nil
	}

	cmd := exec.CommandContext(ctx, exePath, args...)
	configureSysProcAttr(cmd)
	return cmd, nil
}

// GetExecutableName returns the platform-specific executable name.
func GetExecutableName(baseName string) string {
	return baseName + platform.BinaryExtension()
}
