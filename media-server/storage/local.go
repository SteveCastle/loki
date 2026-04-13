package storage

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// LocalBackend serves files from a single root directory on the local filesystem.
type LocalBackend struct {
	rootPath string
	label    string
}

// NewLocalBackend creates a LocalBackend rooted at rootPath.
// label is a human-readable name shown in the UI (e.g. the root directory name).
func NewLocalBackend(rootPath, label string) *LocalBackend {
	return &LocalBackend{
		rootPath: filepath.Clean(rootPath),
		label:    label,
	}
}

// List returns subdirectories and media files directly inside path.
// Results are sorted: directories first, then alphabetically by name.
func (b *LocalBackend) List(_ context.Context, path string) ([]Entry, error) {
	dirEntries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	var entries []Entry
	for _, de := range dirEntries {
		info, err := de.Info()
		if err != nil {
			continue
		}
		if de.IsDir() {
			entries = append(entries, Entry{
				Name:    de.Name(),
				Path:    filepath.Join(path, de.Name()),
				IsDir:   true,
				MtimeMs: float64(info.ModTime().UnixMilli()),
				Type:    "local",
			})
		} else if IsMediaFile(de.Name()) {
			entries = append(entries, Entry{
				Name:    de.Name(),
				Path:    filepath.Join(path, de.Name()),
				IsDir:   false,
				MtimeMs: float64(info.ModTime().UnixMilli()),
				Type:    "local",
			})
		}
	}

	// Sort: directories first, then case-insensitive alphabetical.
	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir != entries[j].IsDir {
			return entries[i].IsDir
		}
		return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name)
	})

	return entries, nil
}

// Scan returns all media files under path.
// When recursive is true it descends into subdirectories via WalkDir,
// skipping symlinked directories to prevent loops.
// When recursive is false it reads only the immediate children.
func (b *LocalBackend) Scan(_ context.Context, path string, recursive bool) ([]FileInfo, error) {
	var files []FileInfo

	if recursive {
		err := filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // skip unreadable entries
			}
			// Skip symlinks that point to directories to prevent infinite loops.
			if d.Type()&fs.ModeSymlink != 0 {
				if info, err := os.Stat(p); err == nil && info.IsDir() {
					return filepath.SkipDir
				}
			}
			if !d.IsDir() && IsMediaFile(d.Name()) {
				info, err := d.Info()
				if err != nil {
					return nil
				}
				files = append(files, FileInfo{
					Path:    p,
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			}
			return nil
		})
		if err != nil {
			return nil, err
		}
	} else {
		dirEntries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, de := range dirEntries {
			if !de.IsDir() && IsMediaFile(de.Name()) {
				info, err := de.Info()
				if err != nil {
					continue
				}
				files = append(files, FileInfo{
					Path:    filepath.Join(path, de.Name()),
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			}
		}
	}

	return files, nil
}

// Download opens path for reading. The caller is responsible for closing the reader.
// Relative paths are resolved against the backend's root directory.
func (b *LocalBackend) Download(_ context.Context, path string) (io.ReadCloser, error) {
	return os.Open(b.resolve(path))
}

// Upload writes r to path, creating any missing parent directories.
// Relative paths are resolved against the backend's root directory.
func (b *LocalBackend) Upload(_ context.Context, path string, r io.Reader, _ string) error {
	path = b.resolve(path)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("storage: mkdir %q: %w", filepath.Dir(path), err)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("storage: create %q: %w", path, err)
	}
	defer f.Close()
	if _, err := io.Copy(f, r); err != nil {
		return fmt.Errorf("storage: write %q: %w", path, err)
	}
	return nil
}

// resolve joins relative paths with the backend's root directory.
// Absolute paths are returned as-is.
func (b *LocalBackend) resolve(path string) string {
	if filepath.IsAbs(path) {
		return path
	}
	return filepath.Join(b.rootPath, path)
}

// MediaURL returns the URL used to serve path to the browser via the media endpoint.
func (b *LocalBackend) MediaURL(path string) (string, error) {
	return "/media/file?path=" + url.QueryEscape(path), nil
}

// Exists reports whether path exists on the local filesystem.
// Relative paths are resolved against the backend's root directory.
func (b *LocalBackend) Exists(_ context.Context, path string) (bool, error) {
	_, err := os.Stat(b.resolve(path))
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// Contains reports whether path is located inside the backend's root directory.
func (b *LocalBackend) Contains(path string) bool {
	cleaned := filepath.Clean(path)
	rel, err := filepath.Rel(b.rootPath, cleaned)
	if err != nil {
		return false
	}
	return !strings.HasPrefix(rel, "..")
}

// Root returns the Entry representing the backend's root directory.
func (b *LocalBackend) Root() Entry {
	info, err := os.Stat(b.rootPath)
	var mtimeMs float64
	if err == nil {
		mtimeMs = float64(info.ModTime().UnixMilli())
	}
	name := b.label
	if name == "" {
		name = filepath.Base(b.rootPath)
	}
	return Entry{
		Name:    name,
		Path:    b.rootPath,
		IsDir:   true,
		MtimeMs: mtimeMs,
		Type:    "local",
	}
}
