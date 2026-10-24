package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"sync"
	"time"
)

// ProgressFn receives per-file byte counts. Safe to be nil.
type ProgressFn func(file string, bytesDone, bytesTotal int64)

var ErrChecksumMismatch = errors.New("models: sha256 checksum mismatch")

// ErrUnavailable marks a model with no files for this OS (a CUDA engine on
// macOS, say): there is nothing to download.
var ErrUnavailable = errors.New("models: not available on this platform")

// Install lock per model id so two concurrent callers for the same model
// don't tread on each other; different models may install in parallel. A
// one-slot channel rather than a mutex so a waiter can give up when its
// context ends (a job cancelled while another install of its model runs).
var (
	installLocks   = map[string]chan struct{}{}
	installLocksMu sync.Mutex
)

func lockForModel(id string) chan struct{} {
	installLocksMu.Lock()
	defer installLocksMu.Unlock()
	if l, ok := installLocks[id]; ok {
		return l
	}
	l := make(chan struct{}, 1)
	installLocks[id] = l
	return l
}

// acquireInstall takes id's install lock, or fails when ctx ends first.
func acquireInstall(ctx context.Context, id string) (release func(), err error) {
	l := lockForModel(id)
	select {
	case l <- struct{}{}:
		return func() { <-l }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Available reports whether the model has anything to install on this OS.
func Available(id string) bool {
	m, ok := Lookup(id)
	return ok && len(m.EffectiveFiles()) > 0
}

// IsInstalled reports whether the model is completely installed, straight
// from disk (no state cache, no state.json write): cheap enough for a
// request path.
func IsInstalled(id string) bool {
	m, ok := Lookup(id)
	return ok && statusFor(m) == StatusInstalled
}

// InstallModel downloads every file of the named model, verifies its
// SHA-256, atomically installs it, and writes <ModelDir>/.meta.json on
// success.
//
// It is safe to call from anywhere at any time: callers for one id are
// serialized, and a model that became installed while the caller waited
// (another job, the Dependencies page) returns at once. Files an earlier,
// interrupted attempt finished are kept (see fileDone), and an interrupted
// download resumes from its .partial.
func InstallModel(ctx context.Context, id string, progress ProgressFn) error {
	m, ok := Lookup(id)
	if !ok {
		return fmt.Errorf("models: unknown id %q", id)
	}
	if len(m.EffectiveFiles()) == 0 {
		return fmt.Errorf("%w: %s has no download for %s", ErrUnavailable, m.Name, runtime.GOOS)
	}
	release, err := acquireInstall(ctx, id)
	if err != nil {
		return err
	}
	defer release()
	if statusFor(m) == StatusInstalled {
		return nil
	}

	marks := readFileMarks(id)
	for _, f := range m.EffectiveFiles() {
		dst := filepath.Join(ModelDir(id), f.RelPath)
		sum, err := resolveChecksum(ctx, f)
		if err != nil {
			return fmt.Errorf("file %s: %w", f.RelPath, err)
		}
		f.SHA256 = sum
		if fileDone(f, dst, marks) {
			if progress != nil {
				progress(filepath.Base(dst), f.SizeBytes, f.SizeBytes)
			}
			continue
		}
		if f.Archive != "" {
			if err := installArchiveFile(ctx, f, dst, progress); err != nil {
				return fmt.Errorf("file %s: %w", f.RelPath, err)
			}
		} else {
			if _, err := downloadFileWithRetry(ctx, f.URL, dst, f.SHA256, progress); err != nil {
				return fmt.Errorf("file %s: %w", f.RelPath, err)
			}
			if f.Exec {
				if err := markExecutable(dst); err != nil {
					return fmt.Errorf("file %s: %w", f.RelPath, err)
				}
			}
		}
		marks[f.RelPath] = f.SHA256
		writeFileMarks(id, marks)
	}
	return writeMeta(id, m)
}

// filesMarker records, per model, the files a previous attempt finished:
// rel_path -> the SHA-256 it was verified against (for an archive, the
// archive's). A model only counts as installed once .meta.json is written,
// so without this a failure on file 5 of 6 would fetch files 1-4 again.
const filesMarker = ".files.json"

func readFileMarks(id string) map[string]string {
	out := map[string]string{}
	if b, err := os.ReadFile(filepath.Join(ModelDir(id), filesMarker)); err == nil {
		_ = json.Unmarshal(b, &out)
	}
	return out
}

// writeFileMarks persists the marks; a failure only costs a re-download.
func writeFileMarks(id string, marks map[string]string) {
	b, err := json.Marshal(marks)
	if err != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(ModelDir(id), filesMarker), b, 0o644)
}

// fileDone reports whether f is already installed at dst: marked complete
// against the same checksum, or (a plain file from before the marks existed)
// present with the expected size and a matching SHA-256, which is then
// marked. A moving "latest" asset whose checksum changed is never done.
func fileDone(f File, dst string, marks map[string]string) bool {
	info, err := os.Stat(dst)
	if err != nil || f.SHA256 == "" {
		return false
	}
	if marks[f.RelPath] == f.SHA256 {
		return true
	}
	if f.Archive != "" || info.IsDir() || (f.SizeBytes > 0 && info.Size() != f.SizeBytes) {
		return false
	}
	if VerifySHA256(dst, f.SHA256) != nil {
		return false
	}
	marks[f.RelPath] = f.SHA256
	return true
}

// downloadFileWithRetry wraps downloadFile with exponential backoff:
// 1s, 4s, 16s. Network errors retry; HTTP 4xx (except 416), cancellation and
// checksum mismatch don't.
//
// The .partial survives network errors and cancellation, so each retry (and
// the next install attempt, possibly days later) resumes where the last one
// stopped. A checksum mismatch discards it; when that attempt had resumed an
// older partial (stale or corrupt bytes), one fresh download is tried before
// giving up.
func downloadFileWithRetry(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
	delays := []time.Duration{0, time.Second, 4 * time.Second, 16 * time.Second}
	var lastErr error
	freshRetried := false
	for _, d := range delays {
		if d > 0 {
			select {
			case <-time.After(d):
			case <-ctx.Done():
				return 0, ctx.Err()
			}
		}
		resumed := false
		if info, err := os.Stat(dst + ".partial"); err == nil && info.Size() > 0 {
			resumed = true
		}
		n, err := downloadFile(ctx, url, dst, want, progress)
		if err == nil {
			return n, nil
		}
		lastErr = err
		if errors.Is(err, ErrChecksumMismatch) && resumed && !freshRetried {
			freshRetried = true // the partial is gone; start over once
			continue
		}
		if errors.Is(err, ErrChecksumMismatch) || isPermanentHTTPError(err) || ctx.Err() != nil {
			return 0, err
		}
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

// downloadFile streams url -> dst with resume from <dst>.partial, verifying
// the running SHA-256 against want before renaming. The partial is kept on
// network errors and cancellation (the next call resumes it with a Range
// request) and discarded only on a checksum mismatch.
func downloadFile(ctx context.Context, url, dst, want string, progress ProgressFn) (int64, error) {
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
		_ = w.Close()
		return 0, err
	}
	if existing > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(existing, 10)+"-")
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		_ = w.Close() // keep the partial: the next attempt resumes it
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
			h = sha256.New()
			existing = 0
		}
	case http.StatusPartialContent:
		// Good — append.
	case http.StatusRequestedRangeNotSatisfiable:
		// The partial already holds the whole file (or more): there is
		// nothing to append, and the body is an error page, not file bytes.
		if existing == 0 {
			_ = w.Close()
			return 0, &httpStatusError{Status: resp.StatusCode, URL: url}
		}
		return finishDownload(w, h, want, existing, 0)
	default:
		_ = w.Close()
		return 0, &httpStatusError{Status: resp.StatusCode, URL: url}
	}

	total := existing
	if resp.ContentLength > 0 {
		total += resp.ContentLength
	}
	mw := io.MultiWriter(w, h)
	written, err := copyWithProgress(ctx, mw, resp.Body, existing, total, progress, filepath.Base(dst))
	if err != nil {
		_ = w.Close() // keep the partial: the next attempt resumes it
		return 0, err
	}
	return finishDownload(w, h, want, existing, written)
}

// finishDownload checks the running hash and commits, or discards the
// partial on a mismatch.
func finishDownload(w *AtomicWriter, h hash.Hash, want string, existing, written int64) (int64, error) {
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

// VerifySHA256 hashes the file at path and compares against want. Returns
// os.ErrNotExist if the file is missing.
func VerifySHA256(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	got := hex.EncodeToString(h.Sum(nil))
	if got != want {
		return fmt.Errorf("%w: got %s want %s", ErrChecksumMismatch, got, want)
	}
	return nil
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
