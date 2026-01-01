// Package platform provides cross-platform utilities for directory paths,
// binary extensions, and OS-specific operations.
package platform

import (
	"os"
	"path/filepath"
)

// AppName is the application name used for directory naming
const AppName = "lowkey-media-viewer"

// AppDisplayName is the display name used on Windows
const AppDisplayName = "Lowkey Media Viewer"

// ServerName is the server name used for temp directories
const ServerName = "lowkey-media-server"

// ServerDisplayName is the display name for the server on Windows
const ServerDisplayName = "Lowkey Media Server"

// GetDataDir returns the application data directory.
// Windows: %APPDATA%\Lowkey Media Viewer
// Linux: ~/.local/share/lowkey-media-viewer
// Falls back to ~/.lowkey-media-viewer if XDG is not available.
func GetDataDir() string {
	return getDataDir()
}

// GetTempDir returns the temp directory for extracted binaries.
// Windows: %ProgramData%\Lowkey Media Server\tmp
// Linux: /tmp/lowkey-media-server or XDG_RUNTIME_DIR/lowkey-media-server
func GetTempDir() string {
	return getTempDir()
}

// GetCacheDir returns the cache directory for downloaded dependencies.
// Windows: %APPDATA%\Lowkey Media Viewer
// Linux: ~/.cache/lowkey-media-viewer
func GetCacheDir() string {
	return getCacheDir()
}

// BinaryExtension returns the executable file extension for the current platform.
// Windows: ".exe"
// Linux: ""
func BinaryExtension() string {
	return binaryExtension()
}

// SharedLibExtension returns the shared library extension for the current platform.
// Windows: ".dll"
// Linux: ".so"
func SharedLibExtension() string {
	return sharedLibExtension()
}

// OpenFile opens a file or directory with the default application.
// Windows: uses "cmd /c start"
// Linux: uses "xdg-open"
func OpenFile(path string) error {
	return openFile(path)
}

// EnsureExecutable ensures a file has executable permissions.
// On Windows, this is a no-op.
// On Linux, this sets the executable bit.
func EnsureExecutable(path string) error {
	return ensureExecutable(path)
}

// UserHomeDir returns the user's home directory with proper fallbacks.
func UserHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return home
}

// JoinPath is a convenience wrapper around filepath.Join
func JoinPath(elem ...string) string {
	return filepath.Join(elem...)
}
