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
)

// installArchiveFile downloads f's archive (with resume + SHA-256 over the
// archive bytes), extracts f.ArchiveMember to dst atomically, then removes
// the archive. If dst already exists it is left alone only when the archive
// step is re-run after a partial failure — the extract simply overwrites.
func installArchiveFile(ctx context.Context, f File, dst string, progress ProgressFn) error {
	if f.Archive != "zip" {
		return fmt.Errorf("models: unsupported archive type %q", f.Archive)
	}
	archivePath := dst + ".archive.zip"
	if _, err := downloadFileWithRetry(ctx, f.URL, archivePath, f.SHA256, progress); err != nil {
		return err
	}
	if err := extractZipMember(archivePath, f.ArchiveMember, dst); err != nil {
		return err
	}
	if f.Exec {
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
