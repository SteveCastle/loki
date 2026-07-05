package models

import (
	"archive/zip"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/bodgit/sevenzip"
)

// installArchiveFile downloads f's archive (with resume + SHA-256 over the
// archive bytes), extracts f.ArchiveMember to dst atomically, then removes
// the archive. A member ending in "/" means "extract that whole subtree into
// dst as a directory" (7z only) — used for multi-file tool bundles like
// Faster-Whisper-XXL. If dst already exists it is left alone only when the
// archive step is re-run after a partial failure — the extract simply
// overwrites.
func installArchiveFile(ctx context.Context, f File, dst string, progress ProgressFn) error {
	if f.Archive != "zip" && f.Archive != "7z" {
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
	case isDir:
		err = fmt.Errorf("models: directory extraction is only supported for 7z archives, not %q", f.Archive)
	case f.Archive == "7z":
		err = fmt.Errorf("models: 7z archives require a directory member (ending in \"/\"), got %q", f.ArchiveMember)
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
	for _, entry := range zr.File {
		if _, ok := match(entry.Name); ok && !entry.FileInfo().IsDir() {
			total += int64(entry.UncompressedSize)
		}
	}
	progressName := filepath.Base(dst) + " (extracting)"

	found := false
	for _, entry := range zr.File {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		rel, ok := match(entry.Name)
		if !ok || rel == "" {
			continue
		}
		found = true
		target := filepath.Join(partial, filepath.FromSlash(rel))
		// Reject entries that would escape the extraction root ("../", absolute).
		if cleaned := filepath.Clean(target); cleaned != partial && !strings.HasPrefix(cleaned, partial+string(filepath.Separator)) {
			return fail(fmt.Errorf("models: archive member %q escapes extraction dir", entry.Name))
		}
		if entry.FileInfo().IsDir() {
			if err := os.MkdirAll(target, 0o755); err != nil {
				return fail(err)
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return fail(err)
		}
		rc, err := entry.Open()
		if err != nil {
			return fail(fmt.Errorf("models: open archive member %q: %w", entry.Name, err))
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
			return fail(fmt.Errorf("models: extract %s: %w", entry.Name, err))
		}
		done += n
		// Preserve stored unix exec bits when the archive has them.
		if runtime.GOOS != "windows" {
			if mode := entry.FileInfo().Mode().Perm(); mode&0o111 != 0 {
				_ = os.Chmod(target, mode|0o644)
			}
		}
	}
	if !found {
		return fail(fmt.Errorf("models: member %q not found in %s", memberPrefix, filepath.Base(archivePath)))
	}

	// Windows-packed bundles store no unix modes; make the root-level
	// launchers runnable.
	if exec && runtime.GOOS != "windows" {
		entries, err := os.ReadDir(partial)
		if err != nil {
			return fail(err)
		}
		for _, e := range entries {
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
