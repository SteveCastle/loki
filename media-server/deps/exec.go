package deps

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"

	"github.com/stevecastle/shrike/platform"
)

// GetExec builds an exec.Cmd for a dependency executable.
// It looks up the executable path from the dependency installation directory.
// Falls back to system PATH if the dependency is not installed.
//
// Parameters:
//   - ctx: context for the command
//   - depID: the dependency ID (e.g., "ffmpeg", "yt-dlp")
//   - exeName: the base name of the executable (without extension)
//   - args: command arguments
//
// Returns the command and any error encountered.
func GetExec(ctx context.Context, depID string, exeName string, args ...string) (*exec.Cmd, error) {
	// Get the executable path from the dependency
	exePath, err := GetExecutablePath(depID, exeName)
	if err != nil {
		// Fall back to system PATH
		systemPath, lookupErr := exec.LookPath(exeName)
		if lookupErr != nil {
			return nil, fmt.Errorf("executable %q not found in dependency %q or system PATH: %v", exeName, depID, lookupErr)
		}
		cmd := exec.CommandContext(ctx, systemPath, args...)
		configureSysProcAttr(cmd)
		return cmd, nil
	}

	// Check if the executable exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		// Fall back to system PATH
		systemPath, lookupErr := exec.LookPath(exeName)
		if lookupErr != nil {
			return nil, fmt.Errorf("executable %q not installed (expected at %s) and not in system PATH", exeName, exePath)
		}
		cmd := exec.CommandContext(ctx, systemPath, args...)
		configureSysProcAttr(cmd)
		return cmd, nil
	}

	cmd := exec.CommandContext(ctx, exePath, args...)
	configureSysProcAttr(cmd)
	return cmd, nil
}

// GetExecutablePath returns the full path to an executable within a dependency.
func GetExecutablePath(depID string, exeName string) (string, error) {
	dep, ok := Get(depID)
	if !ok {
		return "", fmt.Errorf("unknown dependency: %s", depID)
	}

	// Add platform-specific extension
	fullName := exeName + platform.BinaryExtension()

	// Check metadata store first for tracked file path
	metadata := GetMetadataStore()
	meta, ok := metadata.Get(depID)
	if ok && meta.Files != nil {
		if fileInfo, exists := meta.Files[fullName]; exists && fileInfo.Path != "" {
			return fileInfo.Path, nil
		}
	}

	// Fall back to constructing path from dependency target directory
	return filepath.Join(dep.TargetDir, fullName), nil
}

// GetExecutableName returns the platform-specific executable name.
func GetExecutableName(baseName string) string {
	return baseName + platform.BinaryExtension()
}

// GetFFmpegPath returns the path to the ffmpeg executable.
func GetFFmpegPath() string {
	path, _ := GetExecutablePath("ffmpeg", "ffmpeg")
	return path
}

// GetFFprobePath returns the path to the ffprobe executable.
func GetFFprobePath() string {
	path, _ := GetExecutablePath("ffmpeg", "ffprobe")
	return path
}

// GetYtDlpPath returns the path to the yt-dlp executable.
func GetYtDlpPath() string {
	path, _ := GetExecutablePath("yt-dlp", "yt-dlp")
	return path
}

// GetGalleryDlPath returns the path to the gallery-dl executable.
func GetGalleryDlPath() string {
	path, _ := GetExecutablePath("gallery-dl", "gallery-dl")
	return path
}

// GetDownloadURL returns the platform-specific download URL for common dependencies.
func GetFFmpegDownloadURL() string {
	if runtime.GOOS == "windows" {
		return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
	}
	return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
}

func GetYtDlpDownloadURL() string {
	if runtime.GOOS == "windows" {
		return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
	}
	return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"
}

func GetGalleryDlDownloadURL() string {
	if runtime.GOOS == "windows" {
		return "https://github.com/mikf/gallery-dl/releases/latest/download/gallery-dl.exe"
	}
	return "https://github.com/mikf/gallery-dl/releases/latest/download/gallery-dl.bin"
}
