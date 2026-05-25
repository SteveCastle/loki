package models

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"

	"github.com/stevecastle/shrike/platform"
)

var (
	errNotInstalled = errors.New("models: not installed")

	dataDirOverride string
	dataDirMu       sync.RWMutex
)

// SetDataDirForTest overrides the user data dir. Pass "" to clear.
func SetDataDirForTest(dir string) {
	dataDirMu.Lock()
	dataDirOverride = dir
	dataDirMu.Unlock()
}

func dataDir() string {
	dataDirMu.RLock()
	defer dataDirMu.RUnlock()
	if dataDirOverride != "" {
		return dataDirOverride
	}
	return platform.GetDataDir()
}

// IsNotInstalled reports whether err indicates a missing model file.
func IsNotInstalled(err error) bool { return errors.Is(err, errNotInstalled) }

// ModelDir is the directory that holds one model's files.
func ModelDir(id string) string { return filepath.Join(dataDir(), "models", id) }

// Path returns the absolute path to relPath inside the named model. Returns
// IsNotInstalled error if the file is not present.
func Path(id, relPath string) (string, error) {
	full := filepath.Join(ModelDir(id), relPath)
	info, err := os.Stat(full)
	if err != nil || info.IsDir() {
		return "", fmt.Errorf("%w: %s/%s", errNotInstalled, id, relPath)
	}
	return full, nil
}

// AtomicWriter streams bytes to <final>.partial, then renames to <final> on
// Commit, or deletes the partial on Abort.
type AtomicWriter struct {
	final   string
	partial string
	f       *os.File
}

// NewAtomicWriter opens (or truncates and re-creates) <final>.partial for write.
func NewAtomicWriter(final string) (*AtomicWriter, error) {
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return nil, err
	}
	partial := final + ".partial"
	// Truncate on open: this is the "write from scratch" path. Resume callers
	// should not use NewAtomicWriter; see OpenAtomicResume below.
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return nil, err
	}
	return &AtomicWriter{final: final, partial: partial, f: f}, nil
}

// OpenAtomicResume opens <final>.partial for append, returning the current size
// so the caller can issue a Range request. The returned AtomicWriter behaves
// like NewAtomicWriter on Commit/Abort.
func OpenAtomicResume(final string) (*AtomicWriter, int64, error) {
	if err := os.MkdirAll(filepath.Dir(final), 0o755); err != nil {
		return nil, 0, err
	}
	partial := final + ".partial"
	f, err := os.OpenFile(partial, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil, 0, err
	}
	info, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, 0, err
	}
	return &AtomicWriter{final: final, partial: partial, f: f}, info.Size(), nil
}

func (w *AtomicWriter) Write(p []byte) (int, error) { return w.f.Write(p) }

// Commit closes the temp file and renames it to its final path.
func (w *AtomicWriter) Commit() error {
	if err := w.f.Sync(); err != nil {
		_ = w.f.Close()
		_ = os.Remove(w.partial)
		return err
	}
	if err := w.f.Close(); err != nil {
		_ = os.Remove(w.partial)
		return err
	}
	return os.Rename(w.partial, w.final)
}

// Abort closes and removes the partial file.
func (w *AtomicWriter) Abort() error {
	_ = w.f.Close()
	if err := os.Remove(w.partial); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// Ensure io.Writer interface.
var _ io.Writer = (*AtomicWriter)(nil)
