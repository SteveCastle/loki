// Package embedexec provides functionality for extracting and executing
// embedded binaries in a cross-platform manner.
package embedexec

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/platform"
)

// List returns the embedded executable filenames located directly in the bin directory.
// On Windows: ["helper1.exe", "helper2.exe"]
// On Linux: ["helper1", "helper2"]
func List() ([]string, error) {
	entries, err := fs.ReadDir(exeFS, binDir)
	if err != nil {
		return nil, err
	}

	ext := platform.BinaryExtension()
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		// On Windows, filter for .exe files
		// On Linux, include files without extension (excluding hidden files)
		if ext != "" {
			if strings.HasSuffix(e.Name(), ext) {
				out = append(out, e.Name())
			}
		} else {
			// On Linux, include files that don't have common non-executable extensions
			name := e.Name()
			if !strings.HasPrefix(name, ".") && !strings.HasSuffix(name, ".txt") && !strings.HasSuffix(name, ".md") {
				out = append(out, name)
			}
		}
	}
	return out, nil
}

// Extract extracts an embedded executable to a temporary location.
// If name contains a path separator, the entire sub-directory
// under bin/ is copied to a fresh temp dir and the path to the executable
// inside that temp dir is returned.
//
// Returns the path to the extracted executable, a cleanup function, and any error.
func Extract(name string) (string, func(), error) {
	// Does it look like "dir/foo.exe" or "dir/foo"?
	if strings.ContainsAny(name, `/\`) {
		return extractTree(name)
	}
	return extractSingle(name)
}

// extractSingle extracts a single executable file.
func extractSingle(name string) (string, func(), error) {
	data, err := exeFS.ReadFile(path.Join(binDir, name))
	if err != nil {
		return "", nil, fmt.Errorf("read embedded executable: %w", err)
	}

	tmpDir := platform.GetTempDir()
	if err := os.MkdirAll(tmpDir, 0o700); err != nil {
		return "", nil, fmt.Errorf("prep tmp dir: %w", err)
	}

	// Create pattern for temp file
	ext := platform.BinaryExtension()
	baseName := strings.TrimSuffix(name, ext)
	pat := baseName + "-*" + ext

	tmpFile, err := os.CreateTemp(tmpDir, pat)
	if err != nil {
		return "", nil, fmt.Errorf("create temp file: %w", err)
	}
	cleanup := func() { _ = os.Remove(tmpFile.Name()) }

	if _, err := tmpFile.Write(data); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("write temp executable: %w", err)
	}
	tmpFile.Close() // required on Windows before execution

	// Ensure the file is executable (no-op on Windows)
	if err := platform.EnsureExecutable(tmpFile.Name()); err != nil {
		cleanup()
		return "", nil, fmt.Errorf("set executable permissions: %w", err)
	}

	return tmpFile.Name(), cleanup, nil
}

// extractTree extracts an entire directory tree of embedded files.
func extractTree(name string) (string, func(), error) {
	// Split dir/and/file (path uses `/` inside embed.FS)
	dirPart, filePart := path.Split(name)
	if dirPart == "" || filePart == "" {
		return "", nil, fmt.Errorf("invalid embedded path %q", name)
	}

	tmpBase := platform.GetTempDir()
	if err := os.MkdirAll(tmpBase, 0o700); err != nil {
		return "", nil, fmt.Errorf("prep tmp dir: %w", err)
	}

	ext := platform.BinaryExtension()
	baseName := strings.TrimSuffix(filePart, ext)
	tmpDir, err := os.MkdirTemp(tmpBase, baseName+"-*")
	if err != nil {
		return "", nil, fmt.Errorf("create temp dir: %w", err)
	}
	cleanup := func() { _ = os.RemoveAll(tmpDir) }

	// Walk the embedded sub-tree and recreate it on disk.
	prefix := path.Join(binDir, dirPart)
	err = fs.WalkDir(exeFS, prefix, func(p string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		rel := strings.TrimPrefix(p, binDir+"/")               // strip embed root
		dest := filepath.Join(tmpDir, filepath.FromSlash(rel)) // OS path

		if d.IsDir() {
			return os.MkdirAll(dest, 0o700)
		}

		data, err := exeFS.ReadFile(p)
		if err != nil {
			return err
		}
		if err := os.WriteFile(dest, data, 0o700); err != nil {
			return err
		}
		// Ensure executables are marked as such on Linux
		return platform.EnsureExecutable(dest)
	})
	if err != nil {
		cleanup()
		return "", nil, fmt.Errorf("copy dir: %w", err)
	}

	// Full path to the extracted executable inside the copied tree
	exePath := filepath.Join(tmpDir, filepath.FromSlash(name))
	return exePath, cleanup, nil
}

// GetExec extracts (single file or whole dir) then builds an *exec.Cmd.
// If the embedded executable is not found, it falls back to using the system PATH.
func GetExec(ctx context.Context, base string, args ...string) (*exec.Cmd, func(), error) {
	ext := platform.BinaryExtension()

	// First try to extract the embedded executable
	exePath, cleanup, err := Extract(base + ext)
	if err != nil {
		// If extraction fails, try to find the executable in the system PATH
		systemPath, lookupErr := exec.LookPath(base)
		if lookupErr != nil {
			// If both embedded and system lookup fail, return the original error
			return nil, nil, fmt.Errorf("executable %q not found in embedded files or system PATH: embedded error: %w, system error: %v", base, err, lookupErr)
		}

		// Use the system executable
		cmd := exec.CommandContext(ctx, systemPath, args...)
		configureSysProcAttr(cmd)
		return cmd, nil, nil // no cleanup needed for system executables
	}

	// Use the extracted embedded executable
	cmd := exec.CommandContext(ctx, exePath, args...)
	configureSysProcAttr(cmd)
	return cmd, cleanup, nil
}
