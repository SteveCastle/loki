package embedexec

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
)

// ────────────────────────────────────────────────────────────────────────
// Embed *everything* under bin so we can copy directories, not just .exe.
// ────────────────────────────────────────────────────────────────────────
//
//go:embed bin/**
var exeFS embed.FS

// List returns the embedded exe filenames located directly in bin/,
// e.g. ["helper1.exe", "helper2.exe"].
func List() ([]string, error) {
	entries, err := fs.ReadDir(exeFS, "bin")
	if err != nil {
		return nil, err
	}

	var out []string
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(e.Name(), ".exe") {
			out = append(out, e.Name())
		}
	}
	return out, nil
}

// Extract behaves as before for single-file executables.
// If name contains a path separator, the entire sub-directory
// under bin/ is copied to a fresh temp dir and the path to the exe
// inside that temp dir is returned.
func Extract(name string) (string, func(), error) {
	// Does it look like "dir/foo.exe" ?
	if strings.ContainsAny(name, `/\`) {
		return extractTree(name)
	}
	return extractSingle(name)
}

// ────────────── helpers ────────────────────────────────────────────────

// Single-file branch (unchanged except for factoring out)
func extractSingle(name string) (string, func(), error) {
	data, err := exeFS.ReadFile(path.Join("bin", name))
	if err != nil {
		return "", nil, fmt.Errorf("read embedded exe: %w", err)
	}

	tmpDir := filepath.Join(os.Getenv("ProgramData"), "Shrike", "tmp")
	if err := os.MkdirAll(tmpDir, 0o700); err != nil {
		return "", nil, fmt.Errorf("prep tmp dir: %w", err)
	}

	pat := strings.TrimSuffix(name, ".exe") + "-*.exe"
	tmpFile, err := os.CreateTemp(tmpDir, pat)
	if err != nil {
		return "", nil, fmt.Errorf("create temp file: %w", err)
	}
	cleanup := func() { _ = os.Remove(tmpFile.Name()) }

	if _, err := tmpFile.Write(data); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("write temp exe: %w", err)
	}
	tmpFile.Close() // required on Windows

	return tmpFile.Name(), cleanup, nil
}

// Directory-copy branch.
func extractTree(name string) (string, func(), error) {
	// split dir/and/file.exe (path uses `/` inside embed.FS)
	dirPart, filePart := path.Split(name)
	if dirPart == "" || filePart == "" {
		return "", nil, fmt.Errorf("invalid embedded path %q", name)
	}

	// tmp base e.g. C:\ProgramData\Shrike\tmp\ffmpeg-*   (unique)
	tmpBase := filepath.Join(os.Getenv("ProgramData"), "Shrike", "tmp")
	if err := os.MkdirAll(tmpBase, 0o700); err != nil {
		return "", nil, fmt.Errorf("prep tmp dir: %w", err)
	}
	tmpDir, err := os.MkdirTemp(tmpBase, strings.TrimSuffix(filePart, ".exe")+"-*")
	if err != nil {
		return "", nil, fmt.Errorf("create temp dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	// Walk the embedded sub-tree and recreate it on disk.
	prefix := path.Join("bin", dirPart) // e.g. bin/ffmpeg/
	err = fs.WalkDir(exeFS, prefix, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel := strings.TrimPrefix(p, "bin/")                   // strip embed root
		dest := filepath.Join(tmpDir, filepath.FromSlash(rel)) // OS path

		if d.IsDir() {
			return os.MkdirAll(dest, 0o700)
		}

		data, err := exeFS.ReadFile(p)
		if err != nil {
			return err
		}
		return os.WriteFile(dest, data, 0o700)
	})
	if err != nil {
		cleanup()
		return "", nil, fmt.Errorf("copy dir: %w", err)
	}

	// Full path to the extracted exe inside the copied tree
	exePath := filepath.Join(tmpDir, filepath.FromSlash(name))
	return exePath, cleanup, nil
}

// GetExec extracts (single file or whole dir) then builds an *exec.Cmd.
// If the embedded executable is not found, it falls back to using the system PATH.
func GetExec(ctx context.Context, base string, args ...string) (*exec.Cmd, func(), error) {
	// First try to extract the embedded executable
	exePath, cleanup, err := Extract(base + ".exe")
	if err != nil {
		// If extraction fails, try to find the executable in the system PATH
		// Remove the .exe suffix for the system lookup since exec.LookPath handles it
		systemPath, lookupErr := exec.LookPath(base)
		if lookupErr != nil {
			// If both embedded and system lookup fail, return the original error
			return nil, nil, fmt.Errorf("executable %q not found in embedded files or system PATH: embedded error: %w, system error: %v", base, err, lookupErr)
		}

		// Use the system executable
		cmd := exec.CommandContext(ctx, systemPath, args...)
		cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true} // hide console
		return cmd, nil, nil                                     // no cleanup needed for system executables
	}

	// Use the extracted embedded executable
	cmd := exec.CommandContext(ctx, exePath, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true} // hide console
	return cmd, cleanup, nil
}
