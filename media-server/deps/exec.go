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
	switch runtime.GOOS {
	case "windows":
		return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
	case "darwin":
		// evermeet.cx provides static FFmpeg builds for macOS (universal binary)
		return "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip"
	default: // linux
		return "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"
	}
}

// GetFFprobeDownloadURL returns the platform-specific download URL for FFprobe.
// On macOS, FFprobe is downloaded separately from evermeet.cx
func GetFFprobeDownloadURL() string {
	if runtime.GOOS == "darwin" {
		return "https://evermeet.cx/ffmpeg/getrelease/ffprobe/zip"
	}
	// On Windows and Linux, ffprobe is included in the FFmpeg archive
	return ""
}

func GetYtDlpDownloadURL() string {
	switch runtime.GOOS {
	case "windows":
		return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp.exe"
	case "darwin":
		// Universal macOS binary (works on both arm64 and x64)
		return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp_macos"
	default: // linux
		return "https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp"
	}
}

func GetGalleryDlDownloadURL() string {
	switch runtime.GOOS {
	case "windows":
		return "https://github.com/mikf/gallery-dl/releases/latest/download/gallery-dl.exe"
	case "darwin":
		// No official macOS binary available - users should install via pip or homebrew
		// Return empty string to indicate not available for direct download
		return ""
	default: // linux
		return "https://github.com/mikf/gallery-dl/releases/latest/download/gallery-dl.bin"
	}
}
