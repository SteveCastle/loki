package deps

import (
	"path/filepath"
	"runtime"

	"github.com/stevecastle/shrike/platform"
)

// GetDepsDir returns the dependencies installation directory for a specific dependency.
// e.g., GetDepsDir("whisper") returns ~/.local/share/lowkey-media-viewer/whisper on Linux
// or %APPDATA%\Lowkey Media Viewer\whisper on Windows.
func GetDepsDir(subdir string) string {
	return filepath.Join(platform.GetDataDir(), subdir)
}

// GetWhisperExecutableName returns the platform-specific Whisper executable name.
func GetWhisperExecutableName() string {
	if runtime.GOOS == "windows" {
		return "faster-whisper-xxl.exe"
	}
	return "faster-whisper-xxl"
}

// GetWhisperDownloadURL returns the platform-specific download URL for Faster Whisper.
func GetWhisperDownloadURL() string {
	if runtime.GOOS == "windows" {
		return "https://github.com/Purfview/whisper-standalone-win/releases/download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.1_windows.7z"
	}
	// Linux version - using the Linux release
	return "https://github.com/Purfview/whisper-standalone-win/releases/download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.1_linux.7z"
}

// GetOnnxRuntimeLibName returns the platform-specific ONNX Runtime library name.
func GetOnnxRuntimeLibName() string {
	return "onnxruntime" + platform.SharedLibExtension()
}

// GetOnnxRuntimeDownloadURL returns the platform-specific download URL for ONNX Runtime.
func GetOnnxRuntimeDownloadURL(version, arch string) string {
	if runtime.GOOS == "windows" {
		if arch == "arm64" {
			return "https://github.com/microsoft/onnxruntime/releases/download/v" + version + "/onnxruntime-win-arm64-" + version + ".zip"
		}
		return "https://github.com/microsoft/onnxruntime/releases/download/v" + version + "/onnxruntime-win-x64-" + version + ".zip"
	}
	// Linux
	if arch == "arm64" {
		return "https://github.com/microsoft/onnxruntime/releases/download/v" + version + "/onnxruntime-linux-aarch64-" + version + ".tgz"
	}
	return "https://github.com/microsoft/onnxruntime/releases/download/v" + version + "/onnxruntime-linux-x64-" + version + ".tgz"
}

// IsOnnxRuntimeArchiveZip returns true if the ONNX Runtime archive is a ZIP file.
func IsOnnxRuntimeArchiveZip() bool {
	return runtime.GOOS == "windows"
}
