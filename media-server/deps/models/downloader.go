package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

// ProgressFn receives per-file byte counts. Safe to be nil.
type ProgressFn func(file string, bytesDone, bytesTotal int64)

var ErrChecksumMismatch = errors.New("models: sha256 checksum mismatch")

// Install lock per model id so two concurrent callers for the same model
// don't tread on each other; different models may install in parallel.
var (
	installLocks   = map[string]*sync.Mutex{}
	installLocksMu sync.Mutex
)

func lockForModel(id string) *sync.Mutex {
	installLocksMu.Lock()
	defer installLocksMu.Unlock()
	if l, ok := installLocks[id]; ok {
		return l
	}
	l := &sync.Mutex{}
	installLocks[id] = l
	return l
}

// InstallModel downloads every file of the named model, verifies its
// SHA-256, atomically installs it, and writes <ModelDir>/.meta.json on
// success.
func InstallModel(ctx context.Context, id string, progress ProgressFn) error {
	m, ok := Lookup(id)
	if !ok {
		return fmt.Errorf("models: unknown id %q", id)
	}
	l := lockForModel(id)
	l.Lock()
	defer l.Unlock()

	for _, f := range m.Files {
		dst := filepath.Join(ModelDir(id), f.RelPath)
		if _, err := downloadFileWithRetry(ctx, f.URL, dst, f.SHA256, progress); err != nil {
			return fmt.Errorf("file %s: %w", f.RelPath, err)
		}
	}
	return writeMeta(id, m)
}

// downloadFileWithRetry wraps downloadFile with exponential backoff:
// 1s, 4s, 16s. Network errors retry; HTTP 4xx (except 416) and checksum
// mismatch don't.
func downloadFileWithRetry(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
	delays := []time.Duration{0, time.Second, 4 * time.Second, 16 * time.Second}
	var lastErr error
	for i, d := range delays {
		if d > 0 {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return 0, ctx.Err()
			}
		}
		n, err := downloadFile(ctx, url, dst, want, progress)
		if err == nil {
			return n, nil
		}
		lastErr = err
		if errors.Is(err, ErrChecksumMismatch) || isPermanentHTTPError(err) || errors.Is(err, context.Canceled) {
			return 0, err
		}
		_ = i
	}
	return 0, lastErr
}

type httpStatusError struct {
	Status int
	URL    string
}

func (e *httpStatusError) Error() string {
	return fmt.Sprintf("HTTP %d from %s", e.Status, e.URL)
}

func isPermanentHTTPError(err error) bool {
	var hs *httpStatusError
	if errors.As(err, &hs) {
		return hs.Status >= 400 && hs.Status < 500 && hs.Status != http.StatusRequestedRangeNotSatisfiable
	}
	return false
}

// downloadFile streams url -> dst with optional resume from <dst>.partial,
// verifying the running SHA-256 against want before renaming.
func downloadFile(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
	// Try to resume.
	w, existing, err := OpenAtomicResume(dst)
	if err != nil {
		return 0, err
	}
	// If existing > 0, hash the existing bytes first so the running hash matches the full stream.
	h := sha256.New()
	if existing > 0 {
		if err := hashExisting(w.partial, h); err != nil {
			_ = w.Abort()
			return 0, err
		}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		_ = w.Abort()
		return 0, err
	}
	if existing > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(existing, 10)+"-")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		_ = w.Abort()
		return 0, err
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// Server ignored our Range — truncate and restart.
		if existing > 0 {
			_ = w.Abort()
			w, _, err = OpenAtomicResume(dst) // reopen empty
			if err != nil {
				return 0, err
			}
			_ = os.Truncate(w.partial, 0)
			h = sha256.New()
			existing = 0
		}
	case http.StatusPartialContent:
		// Good — append.
	case http.StatusRequestedRangeNotSatisfiable:
		// Existing file size matches or exceeds; verify checksum below.
	default:
		_ = w.Abort()
		return 0, &httpStatusError{Status: resp.StatusCode, URL: url}
	}

	total := existing
	if resp.ContentLength > 0 {
		total += resp.ContentLength
	}
	mw := io.MultiWriter(w, h)
	written, err := copyWithProgress(ctx, mw, resp.Body, existing, total, progress, filepath.Base(dst))
	if err != nil {
		_ = w.Abort()
		return 0, err
	}

	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		_ = w.Abort()
		return 0, fmt.Errorf("%w: got %s want %s", ErrChecksumMismatch, got, want)
	}
	if err := w.Commit(); err != nil {
		return 0, err
	}
	return existing + written, nil
}

func hashExisting(path string, h io.Writer) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(h, f)
	return err
}

// copyWithProgress copies src→dst in 64KB chunks, calling progress every chunk.
func copyWithProgress(ctx context.Context, dst io.Writer, src io.Reader, start, total int64, p ProgressFn, name string) (int64, error) {
	buf := make([]byte, 64*1024)
	var written int64
	for {
		if err := ctx.Err(); err != nil {
			return written, err
		}
		nr, er := src.Read(buf)
		if nr > 0 {
			nw, ew := dst.Write(buf[:nr])
			written += int64(nw)
			if p != nil {
				p(name, start+written, total)
			}
			if ew != nil {
				return written, ew
			}
		}
		if er == io.EOF {
			return written, nil
		}
		if er != nil {
			return written, er
		}
	}
}

func writeMeta(id string, m Model) error {
	type meta struct {
		Version        string `json:"version"`
		InstalledAt    string `json:"installed_at"`
		SHA256Verified bool   `json:"sha256_verified"`
	}
	doc := meta{Version: m.Version, InstalledAt: time.Now().UTC().Format(time.RFC3339), SHA256Verified: true}
	b, err := json.Marshal(doc)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(ModelDir(id), ".meta.json"), b, 0o644)
}
