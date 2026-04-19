package downloads

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/bodgit/sevenzip"
	"github.com/stevecastle/shrike/platform"
)

// ExtractZip extracts a ZIP archive to the destination directory.
// If stripPrefix is provided, it removes that prefix from extracted file paths.
func ExtractZip(archivePath, destDir string, stripPrefix string, progressCb ProgressCallback) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	for i, file := range reader.File {
		if progressCb != nil && i%10 == 0 {
			progressCb(Progress{
				Status:  StatusExtracting,
				Message: fmt.Sprintf("Extracting %d/%d files...", i+1, len(reader.File)),
			})
		}

		// Apply strip prefix
		name := file.Name
		if stripPrefix != "" && strings.HasPrefix(name, stripPrefix) {
			name = strings.TrimPrefix(name, stripPrefix)
		}

		if name == "" || file.FileInfo().IsDir() {
			continue
		}

		destPath := filepath.Join(destDir, name)

		// Create parent directories
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}

		// Extract file
		if err := extractZipFile(file, destPath); err != nil {
			return err
		}
	}

	return nil
}

func extractZipFile(file *zip.File, destPath string) error {
	rc, err := file.Open()
	if err != nil {
		return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
	}
	defer rc.Close()

	outFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", destPath, err)
	}
	defer outFile.Close()

	if _, err := io.Copy(outFile, rc); err != nil {
		return fmt.Errorf("failed to extract %s: %w", file.Name, err)
	}

	// Set executable permission on Unix-like systems
	if file.Mode()&0111 != 0 {
		if err := platform.EnsureExecutable(destPath); err != nil {
			// Non-fatal error
		}
	}

	return nil
}

// Extract7z extracts a 7z archive to the destination directory.
// If stripPrefix is provided, it removes that prefix from extracted file paths.
func Extract7z(archivePath, destDir string, stripPrefix string, progressCb ProgressCallback) error {
	reader, err := sevenzip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open 7z archive: %w", err)
	}
	defer reader.Close()

	for i, file := range reader.File {
		if progressCb != nil && i%10 == 0 {
			progressCb(Progress{
				Status:  StatusExtracting,
				Message: fmt.Sprintf("Extracting %d/%d files...", i+1, len(reader.File)),
			})
		}

		// Apply strip prefix
		name := file.Name
		if stripPrefix != "" && strings.HasPrefix(name, stripPrefix) {
			name = strings.TrimPrefix(name, stripPrefix)
		}

		if name == "" {
			continue
		}

		destPath := filepath.Join(destDir, name)

		// Create parent directories
		if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}

		info := file.FileInfo()
		if info.IsDir() {
			if err := os.MkdirAll(destPath, 0755); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
			continue
		}

		// Extract file
		if err := extract7zFile(file, destPath); err != nil {
			return err
		}
	}

	return nil
}

func extract7zFile(file *sevenzip.File, destPath string) error {
	rc, err := file.Open()
	if err != nil {
		return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
	}
	defer rc.Close()

	outFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create %s: %w", destPath, err)
	}
	defer outFile.Close()

	if _, err := io.Copy(outFile, rc); err != nil {
		return fmt.Errorf("failed to extract %s: %w", file.Name, err)
	}

	return nil
}

// ExtractTarXz extracts a tar.xz archive using the system tar command.
// On Linux, this requires the xz utilities to be installed.
func ExtractTarXz(archivePath, destDir string, stripComponents int, wildcards string, progressCb ProgressCallback) error {
	if progressCb != nil {
		progressCb(Progress{
			Status:  StatusExtracting,
			Message: "Extracting tar.xz archive...",
		})
	}

	args := []string{"-xf", archivePath, "-C", destDir}
	if stripComponents > 0 {
		args = append(args, fmt.Sprintf("--strip-components=%d", stripComponents))
	}
	if wildcards != "" {
		args = append(args, "--wildcards", wildcards)
	}

	cmd := exec.Command("tar", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("tar extraction failed: %w (output: %s)", err, string(output))
	}

	return nil
}

// ExtractTarGz extracts a tar.gz archive.
func ExtractTarGz(archivePath, destDir string, progressCb ProgressCallback) error {
	if progressCb != nil {
		progressCb(Progress{
			Status:  StatusExtracting,
			Message: "Extracting tar.gz archive...",
		})
	}

	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	tarReader := tar.NewReader(gzReader)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		destPath := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(destPath, 0755); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
				return fmt.Errorf("failed to create directory: %w", err)
			}

			outFile, err := os.Create(destPath)
			if err != nil {
				return fmt.Errorf("failed to create file: %w", err)
			}

			if _, err := io.Copy(outFile, tarReader); err != nil {
				outFile.Close()
				return fmt.Errorf("failed to extract file: %w", err)
			}
			outFile.Close()

			// Set executable permission if needed
			if header.Mode&0111 != 0 {
				if err := platform.EnsureExecutable(destPath); err != nil {
					// Non-fatal error
				}
			}
		}
	}

	return nil
}

// ExtractFileFromTarGz extracts a specific file from a tar.gz archive that matches a pattern.
func ExtractFileFromTarGz(archivePath, destPath string, matchFunc func(name string) bool, progressCb ProgressCallback) error {
	if progressCb != nil {
		progressCb(Progress{
			Status:  StatusExtracting,
			Message: "Searching archive...",
		})
	}

	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	tarReader := tar.NewReader(gzReader)

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		if header.Typeflag != tar.TypeReg {
			continue
		}

		if matchFunc(header.Name) {
			if progressCb != nil {
				progressCb(Progress{
					Status:  StatusExtracting,
					Message: fmt.Sprintf("Extracting %s...", filepath.Base(header.Name)),
				})
			}

			outFile, err := os.Create(destPath)
			if err != nil {
				return fmt.Errorf("failed to create file: %w", err)
			}

			if _, err := io.Copy(outFile, tarReader); err != nil {
				outFile.Close()
				return fmt.Errorf("failed to extract file: %w", err)
			}
			outFile.Close()

			// Set executable permission
			if err := platform.EnsureExecutable(destPath); err != nil {
				// Non-fatal error
			}

			return nil
		}
	}

	return fmt.Errorf("no matching file found in archive")
}

// ExtractFileFromZip extracts a specific file from a ZIP archive that matches a pattern.
func ExtractFileFromZip(archivePath, destPath string, matchFunc func(name string) bool, progressCb ProgressCallback) error {
	if progressCb != nil {
		progressCb(Progress{
			Status:  StatusExtracting,
			Message: "Searching archive...",
		})
	}

	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}

		if matchFunc(file.Name) {
			if progressCb != nil {
				progressCb(Progress{
					Status:  StatusExtracting,
					Message: fmt.Sprintf("Extracting %s...", filepath.Base(file.Name)),
				})
			}

			rc, err := file.Open()
			if err != nil {
				return fmt.Errorf("failed to open file in archive: %w", err)
			}
			defer rc.Close()

			outFile, err := os.Create(destPath)
			if err != nil {
				return fmt.Errorf("failed to create file: %w", err)
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, rc); err != nil {
				return fmt.Errorf("failed to extract file: %w", err)
			}

			return nil
		}
	}

	return fmt.Errorf("no matching file found in archive")
}
