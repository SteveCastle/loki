package models

import (
	"archive/tar"
	"archive/zip"
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/bodgit/sevenzip"
	"github.com/ulikunitz/xz"
)

// installArchiveFile downloads f's archive (with resume + SHA-256 over the
// archive bytes), extracts f.ArchiveMember to dst atomically, then removes
// the archive. A member ending in "/" means "extract that whole subtree into
// dst as a directory" (7z and zip) — used for multi-file tool bundles like
// Faster-Whisper-XXL and COLMAP. tar.xz archives (Brush's Linux release)
// support a single member only. If dst already exists it is left alone only when the
// archive step is re-run after a partial failure — the extract simply
// overwrites.
func installArchiveFile(ctx context.Context, f File, dst string, progress ProgressFn) error {
	if f.Archive != "zip" && f.Archive != "7z" && f.Archive != "tar.xz" {
		return fmt.Errorf("models: unsupported archive type %q", f.Archive)
	}
	archivePath := dst + ".archive." + f.Archive
	if _, err := downloadFileWithRetry(ctx, f.URL, archivePath, f.SHA256, progress); err != nil {
		return err
	}
	isDir := strings.HasSuffix(f.ArchiveMember, "/")
	var err error
	switch {
	case isDir && f.Archive == "7z":
		err = extractSevenZipDir(ctx, archivePath, f.ArchiveMember, dst, f.Exec, progress)
	case isDir && f.Archive == "zip":
		err = extractZipDir(ctx, archivePath, f.ArchiveMember, dst, f.Exec, progress)
	case isDir:
		err = fmt.Errorf("models: directory extraction is not supported for %q archives", f.Archive)
	case f.Archive == "7z":
		err = fmt.Errorf("models: 7z archives require a directory member (ending in \"/\"), got %q", f.ArchiveMember)
	case f.Archive == "tar.xz":
		err = extractTarXzMember(archivePath, f.ArchiveMember, dst)
	default:
		err = extractZipMember(archivePath, f.ArchiveMember, dst)
	}
	if err != nil {
		return err
	}
	if f.Exec && !isDir {
		if err := markExecutable(dst); err != nil {
			return err
		}
	}
	// The archive is only an intermediate; keep the model dir lean.
	_ = os.Remove(archivePath)
	return nil
}

// extractZipMember extracts one member of a zip archive to dst atomically
// (dst.partial → rename). Member matching is case-insensitive on the
// forward-slash form to survive archives built on other platforms.
func extractZipMember(archivePath, member, dst string) error {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("models: open archive: %w", err)
	}
	defer zr.Close()

	want := strings.ToLower(filepath.ToSlash(member))
	for _, entry := range zr.File {
		if strings.ToLower(filepath.ToSlash(entry.Name)) != want {
			continue
		}
		rc, err := entry.Open()
		if err != nil {
			return fmt.Errorf("models: open archive member: %w", err)
		}
		defer rc.Close()

		w, err := NewAtomicWriter(dst)
		if err != nil {
			return err
		}
		if _, err := io.Copy(w, rc); err != nil {
			_ = w.Abort()
			return fmt.Errorf("models: extract %s: %w", member, err)
		}
		return w.Commit()
	}
	return fmt.Errorf("models: member %q not found in %s", member, filepath.Base(archivePath))
}

// extractSevenZipDir extracts every member under memberPrefix (a directory
// inside the archive, trailing slash) into the directory dst, atomically
// (dst.partial/ → rename). Prefix matching is ASCII-case-insensitive on the
// forward-slash form. Entries are extracted in archive order — 7z solid
// blocks decompress sequentially, so out-of-order access would restart the
// stream per file.
//
// The Purfview bundles are packed on Windows and carry no unix permission
// bits, so when exec is set the root-level regular files (the launcher
// binaries) are marked executable on non-Windows platforms; stored unix
// modes, when present, are preserved too.
func extractSevenZipDir(ctx context.Context, archivePath, memberPrefix string, dst string, exec bool, progress ProgressFn) error {
	zr, err := sevenzip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("models: open archive: %w", err)
	}
	defer zr.Close()
	entries := make([]dirEntry, 0, len(zr.File))
	for _, e := range zr.File {
		e := e
		entries = append(entries, dirEntry{name: e.Name, info: e.FileInfo(), size: int64(e.UncompressedSize),
			open: func() (io.ReadCloser, error) { return e.Open() }})
	}
	return extractEntriesDir(ctx, entries, filepath.Base(archivePath), memberPrefix, dst, exec, progress)
}

// extractZipDir is extractSevenZipDir for zip archives (COLMAP's Windows
// release: bin/ holds the executable and every DLL it loads).
func extractZipDir(ctx context.Context, archivePath, memberPrefix string, dst string, exec bool, progress ProgressFn) error {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("models: open archive: %w", err)
	}
	defer zr.Close()
	entries := make([]dirEntry, 0, len(zr.File))
	for _, e := range zr.File {
		e := e
		entries = append(entries, dirEntry{name: e.Name, info: e.FileInfo(), size: int64(e.UncompressedSize64),
			open: func() (io.ReadCloser, error) { return e.Open() }})
	}
	return extractEntriesDir(ctx, entries, filepath.Base(archivePath), memberPrefix, dst, exec, progress)
}

// dirEntry is one archive member, whatever the archive format.
type dirEntry struct {
	name string
	info os.FileInfo
	size int64
	open func() (io.ReadCloser, error)
}

// extractEntriesDir does the work for the directory extractors: every
// entry under memberPrefix lands in dst, atomically (dst.partial/ → rename).
func extractEntriesDir(ctx context.Context, entries []dirEntry, archiveName, memberPrefix, dst string, exec bool, progress ProgressFn) error {
	prefix := strings.TrimSuffix(filepath.ToSlash(memberPrefix), "/") + "/"
	match := func(name string) (rel string, ok bool) {
		slash := filepath.ToSlash(name)
		if len(slash) < len(prefix) || !strings.EqualFold(slash[:len(prefix)], prefix) {
			return "", false
		}
		return slash[len(prefix):], true
	}

	partial := dst + ".partial"
	if err := os.RemoveAll(partial); err != nil {
		return fmt.Errorf("models: clear stale partial dir: %w", err)
	}
	if err := os.MkdirAll(partial, 0o755); err != nil {
		return err
	}
	fail := func(err error) error {
		_ = os.RemoveAll(partial)
		return err
	}

	// Total uncompressed size under the prefix, for extraction progress.
	var total, done int64
	for _, entry := range entries {
		if _, ok := match(entry.name); ok && !entry.info.IsDir() {
			total += entry.size
		}
	}
	progressName := filepath.Base(dst) + " (extracting)"

	found := false
	for _, entry := range entries {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		rel, ok := match(entry.name)
		if !ok || rel == "" {
			continue
		}
		found = true
		target := filepath.Join(partial, filepath.FromSlash(rel))
		// Reject entries that would escape the extraction root ("../", absolute).
		if cleaned := filepath.Clean(target); cleaned != partial && !strings.HasPrefix(cleaned, partial+string(filepath.Separator)) {
			return fail(fmt.Errorf("models: archive member %q escapes extraction dir", entry.name))
		}
		if entry.info.IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return fail(err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return fail(err)
		}
		rc, err := entry.open()
		if err != nil {
			return fail(fmt.Errorf("models: open archive member %q: %w", entry.name, err))
		}
		out, err := os.OpenFile(target, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
		if err != nil {
			rc.Close()
			return fail(err)
		}
		n, err := copyWithProgress(ctx, out, rc, done, total, progress, progressName)
		rc.Close()
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return fail(fmt.Errorf("models: extract %s: %w", entry.name, err))
		}
		done += n
		// Preserve stored unix exec bits when the archive has them.
		if runtime.GOOS != "windows" {
			if mode := entry.info.Mode().Perm(); mode&0o111 != 0 {
				_ = os.Chmod(target, mode|0o644)
			}
		}
	}
	if !found {
		return fail(fmt.Errorf("models: member %q not found in %s", memberPrefix, archiveName))
	}

	// Windows-packed bundles store no unix modes; make the root-level
	// launchers runnable.
	if exec && runtime.GOOS != "windows" {
		dirEntries, err := os.ReadDir(partial)
		if err != nil {
			return fail(err)
		}
		for _, e := range dirEntries {
			if e.Type().IsRegular() {
				if err := markExecutable(filepath.Join(partial, e.Name())); err != nil {
					return fail(err)
				}
			}
		}
	}

	// Swap into place: drop any previous install, then rename the finished tree.
	if err := os.RemoveAll(dst); err != nil {
		return fail(fmt.Errorf("models: remove previous install: %w", err))
	}
	return os.Rename(partial, dst)
}

// extractTarXzMember streams a .tar.xz to its one wanted member and writes
// it to dst atomically (tar has no index, so this reads up to the member).
func extractTarXzMember(archivePath, member, dst string) error {
	f, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("models: open archive: %w", err)
	}
	defer f.Close()
	xr, err := xz.NewReader(bufio.NewReader(f))
	if err != nil {
		return fmt.Errorf("models: open xz stream: %w", err)
	}
	tr := tar.NewReader(xr)
	want := strings.ToLower(filepath.ToSlash(member))
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("models: read archive: %w", err)
		}
		if h.Typeflag != tar.TypeReg || strings.ToLower(filepath.ToSlash(h.Name)) != want {
			continue
		}
		w, err := NewAtomicWriter(dst)
		if err != nil {
			return err
		}
		if _, err := io.Copy(w, tr); err != nil {
			_ = w.Abort()
			return fmt.Errorf("models: extract %s: %w", member, err)
		}
		return w.Commit()
	}
	return fmt.Errorf("models: member %q not found in %s", member, filepath.Base(archivePath))
}

// markExecutable sets the executable bits on non-Windows platforms.
func markExecutable(path string) error {
	if runtime.GOOS == "windows" {
		return nil
	}
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	return os.Chmod(path, info.Mode()|0o111)
}
