package storage

import (
	"context"
	"io"
	"regexp"
)

// Entry represents a file or directory returned by List.
type Entry struct {
	Name    string  `json:"name"`
	Path    string  `json:"path"`
	IsDir   bool    `json:"isDir"`
	MtimeMs float64 `json:"mtimeMs"`
	Type    string  `json:"type,omitempty"` // "local" or "s3"
}

// FileInfo is a lightweight record returned by Scan.
type FileInfo struct {
	Path    string  `json:"path"`
	MtimeMs float64 `json:"mtimeMs"`
}

// Backend is the storage abstraction used by the media server.
// Implementations may target the local filesystem, S3, or any other store.
type Backend interface {
	// List returns the immediate children of path (dirs + media files).
	List(ctx context.Context, path string) ([]Entry, error)

	// Scan returns all media files under path.
	// When recursive is true it descends into subdirectories.
	Scan(ctx context.Context, path string, recursive bool) ([]FileInfo, error)

	// Download opens path for reading. The caller must close the returned reader.
	Download(ctx context.Context, path string) (io.ReadCloser, error)

	// Upload writes r to path, creating intermediate directories as needed.
	Upload(ctx context.Context, path string, r io.Reader, contentType string) error

	// MediaURL returns a URL suitable for serving path to the browser.
	MediaURL(path string) (string, error)

	// Exists reports whether path exists in this backend.
	Exists(ctx context.Context, path string) (bool, error)

	// Contains reports whether path is rooted inside this backend.
	Contains(path string) bool

	// Root returns the Entry that represents the top-level directory of this backend.
	Root() Entry
}

// MediaExtRegex matches supported media file extensions (case-insensitive).
var MediaExtRegex = regexp.MustCompile(
	`(?i)\.(jpg|jpeg|jfif|webp|png|webm|mp4|mov|mpeg|gif|mkv|m4v|mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape)$`,
)

// IsMediaFile returns true if name has a supported media extension.
func IsMediaFile(name string) bool {
	return MediaExtRegex.MatchString(name)
}
