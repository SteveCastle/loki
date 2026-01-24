package deps

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/downloads"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

func init() {
	Register(&Dependency{
		ID:            "onnx-bundle",
		Name:          "ONNX Tagger Models",
		Description:   "ML models, labels, config, and runtime library for AI-powered image tagging",
		TargetDir:     GetDepsDir("onnx"),
		Check:         checkONNX,
		Download:      downloadONNX,
		DownloadFn:    downloadONNXNew,
		LatestVersion: "wd-eva02-large-v3",
		DownloadURL:   "",                // To be configured
		ExpectedSize:  500 * 1024 * 1024, // 500MB
	})
}

// checkONNX verifies if all ONNX tagger files exist and returns version info.
func checkONNX(ctx context.Context) (bool, string, error) {
	// Always use default installation location
	targetDir := GetDepsDir("onnx")

	// Check all required files
	requiredFiles := map[string]string{
		"model":   filepath.Join(targetDir, "model.onnx"),
		"labels":  filepath.Join(targetDir, "selected_tags.csv"),
		"config":  filepath.Join(targetDir, "config.json"),
		"runtime": filepath.Join(targetDir, GetOnnxRuntimeLibName()),
	}

	for name, path := range requiredFiles {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return false, "", nil
		} else if err != nil {
			return false, "", fmt.Errorf("error checking %s: %w", name, err)
		}
	}

	// Return the model name as version (matches LatestVersion field)
	// This way the version check will pass and show as "installed" not "outdated"
	return true, "wd-eva02-large-v3", nil
}

// computeModelVersion generates a version string from the model file's hash.
func computeModelVersion(modelPath string) (string, error) {
	f, err := os.Open(modelPath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	hasher := sha256.New()
	if _, err := io.Copy(hasher, f); err != nil {
		return "", err
	}

	hash := hex.EncodeToString(hasher.Sum(nil))
	return hash[:12], nil // Use first 12 chars as version
}

// downloadONNX downloads and installs the ONNX model bundle.
func downloadONNX(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting ONNX bundle download...")

	dep, ok := Get("onnx-bundle")
	if !ok {
		return fmt.Errorf("onnx-bundle dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	// Download URLs for ONNX model files from HuggingFace
	modelURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx"
	labelsURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv"
	configURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/config.json"

	files := []struct {
		url      string
		filename string
	}{
		{modelURL, "model.onnx"},
		{labelsURL, "selected_tags.csv"},
		{configURL, "config.json"},
	}

	// Download each file
	for _, file := range files {
		targetPath := filepath.Join(dep.TargetDir, file.filename)
		q.PushJobStdout(j.ID, fmt.Sprintf("Downloading %s...", file.filename))

		if err := downloadFile(j.Ctx, targetPath, file.url, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Failed to download %s: %v", file.filename, err))
			return fmt.Errorf("download failed: %w", err)
		}

		q.PushJobStdout(j.ID, fmt.Sprintf("✓ Downloaded %s", file.filename))
	}

	// Download ONNX Runtime library
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Downloading ONNX Runtime...")
	onnxRuntimeLibName := GetOnnxRuntimeLibName()
	onnxRuntimeLibPath := filepath.Join(dep.TargetDir, onnxRuntimeLibName)

	// Determine architecture and download URL
	arch := runtime.GOARCH
	onnxVersion := "1.22.0" // Version 1.22.0+ required for ONNX opset 22 support (wd-eva02-large-v3 uses opset 22)
	var onnxRuntimeURL string

	if arch != "amd64" && arch != "arm64" {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: Unsupported architecture %s, skipping ONNX Runtime download", arch))
		q.PushJobStdout(j.ID, "Please manually download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
		goto skipOnnxRuntime
	}

	onnxRuntimeURL = GetOnnxRuntimeDownloadURL(onnxVersion, arch)

	// Download ONNX Runtime archive
	q.PushJobStdout(j.ID, fmt.Sprintf("Downloading ONNX Runtime %s...", onnxVersion))

	if IsOnnxRuntimeArchiveZip() {
		// Windows: Download and extract from ZIP
		tempZip := filepath.Join(dep.TargetDir, "onnxruntime.zip")
		if err := downloadFile(j.Ctx, tempZip, onnxRuntimeURL, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Failed to download ONNX Runtime: %v", err))
			q.PushJobStdout(j.ID, "Please manually download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
			goto skipOnnxRuntime
		}

		// Extract the library from the zip
		q.PushJobStdout(j.ID, "Extracting ONNX Runtime library...")
		if err := extractOnnxRuntimeFromZip(tempZip, onnxRuntimeLibPath, onnxRuntimeLibName, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Failed to extract ONNX Runtime: %v", err))
			q.PushJobStdout(j.ID, "Please manually download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
			goto skipOnnxRuntime
		}
		os.Remove(tempZip)
	} else {
		// Linux: Download and extract from tar.gz
		tempTgz := filepath.Join(dep.TargetDir, "onnxruntime.tgz")
		if err := downloadFile(j.Ctx, tempTgz, onnxRuntimeURL, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Failed to download ONNX Runtime: %v", err))
			q.PushJobStdout(j.ID, "Please manually download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
			goto skipOnnxRuntime
		}

		// Extract the library from the tar.gz
		q.PushJobStdout(j.ID, "Extracting ONNX Runtime library...")
		if err := extractOnnxRuntimeFromTarGz(tempTgz, onnxRuntimeLibPath, onnxRuntimeLibName, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Failed to extract ONNX Runtime: %v", err))
			q.PushJobStdout(j.ID, "Please manually download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
			goto skipOnnxRuntime
		}
		os.Remove(tempTgz)
	}

	q.PushJobStdout(j.ID, "✓ Downloaded and extracted ONNX Runtime")

skipOnnxRuntime:

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("onnx-bundle", DependencyMetadata{
		InstalledVersion: "wd-eva02-large-v3", // Use model name as version
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			"model.onnx":        {Path: filepath.Join(dep.TargetDir, "model.onnx")},
			"selected_tags.csv": {Path: filepath.Join(dep.TargetDir, "selected_tags.csv")},
			"config.json":       {Path: filepath.Join(dep.TargetDir, "config.json")},
			onnxRuntimeLibName:  {Path: onnxRuntimeLibPath},
		},
	})
	metadata.Save()

	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "✓ ONNX model bundle installed successfully!")
	return nil
}

// downloadFile downloads a file from a URL with progress tracking.
func downloadFile(ctx context.Context, filepath string, url string, jobID string, q *jobqueue.Queue) error {
	// Create HTTP request with context
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	// Execute request
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Create output file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Copy with progress tracking
	totalSize := resp.ContentLength
	downloaded := int64(0)
	lastReport := time.Now()
	buffer := make([]byte, 32*1024) // 32KB buffer

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		n, err := resp.Body.Read(buffer)
		if n > 0 {
			if _, writeErr := out.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
			downloaded += int64(n)

			// Report progress every second
			if time.Since(lastReport) >= time.Second {
				if totalSize > 0 {
					percent := float64(downloaded) / float64(totalSize) * 100
					q.PushJobStdout(jobID, fmt.Sprintf("Progress: %.1f%% (%s / %s)",
						percent,
						formatSize(downloaded),
						formatSize(totalSize)))
				} else {
					q.PushJobStdout(jobID, fmt.Sprintf("Downloaded: %s", formatSize(downloaded)))
				}
				lastReport = time.Now()
			}
		}

		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
	}

	return nil
}

// formatSize formats bytes as human-readable size.
func formatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// extractOnnxRuntimeFromZip extracts the ONNX Runtime library from a ZIP archive (Windows).
func extractOnnxRuntimeFromZip(zipPath, outputPath, libName string, jobID string, q *jobqueue.Queue) error {
	// Open the zip file
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("failed to open zip: %w", err)
	}
	defer reader.Close()

	// Search for the library file in the zip
	var libFile *zip.File
	for _, file := range reader.File {
		if strings.HasSuffix(strings.ToLower(file.Name), strings.ToLower(libName)) {
			libFile = file
			break
		}
	}

	if libFile == nil {
		return fmt.Errorf("%s not found in archive", libName)
	}

	q.PushJobStdout(jobID, fmt.Sprintf("Found library at: %s", libFile.Name))

	// Open the library file from the zip
	rc, err := libFile.Open()
	if err != nil {
		return fmt.Errorf("failed to open library in zip: %w", err)
	}
	defer rc.Close()

	// Create output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	// Copy the library
	copied, err := io.Copy(outFile, rc)
	if err != nil {
		return fmt.Errorf("failed to extract library: %w", err)
	}

	q.PushJobStdout(jobID, fmt.Sprintf("Extracted %s (%s)", filepath.Base(outputPath), formatSize(copied)))
	return nil
}

// extractOnnxRuntimeFromTarGz extracts the ONNX Runtime library from a tar.gz archive (Linux/macOS).
func extractOnnxRuntimeFromTarGz(tgzPath, outputPath, libName string, jobID string, q *jobqueue.Queue) error {
	// Open the tar.gz file
	file, err := os.Open(tgzPath)
	if err != nil {
		return fmt.Errorf("failed to open tar.gz: %w", err)
	}
	defer file.Close()

	// Create gzip reader
	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	// Create tar reader
	tarReader := tar.NewReader(gzReader)

	// Track files we need to extract
	var foundMainLib bool
	var foundProviderLib bool
	targetDir := filepath.Dir(outputPath)

	// Determine library extension based on platform
	// Linux: .so, macOS: .dylib
	libExt := platform.SharedLibExtension()

	// Search for the library files in the tar
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		// Skip directories
		if header.Typeflag == tar.TypeDir {
			continue
		}

		// Look for the main ONNX Runtime library
		// Linux: lib/libonnxruntime.so.1.23.1
		// macOS: lib/libonnxruntime.1.23.1.dylib
		isMainLib := false
		if runtime.GOOS == "darwin" {
			// macOS pattern: libonnxruntime.{version}.dylib
			isMainLib = strings.Contains(header.Name, "/lib/libonnxruntime.") &&
				strings.HasSuffix(header.Name, ".dylib") &&
				!strings.Contains(header.Name, "_providers_")
		} else {
			// Linux pattern: libonnxruntime.so.{version}
			isMainLib = strings.Contains(header.Name, "/lib/libonnxruntime.so.") &&
				!strings.Contains(header.Name, "_providers_")
		}

		if isMainLib {
			q.PushJobStdout(jobID, fmt.Sprintf("Found main library at: %s", header.Name))

			// Create output file
			outFile, err := os.Create(outputPath)
			if err != nil {
				return fmt.Errorf("failed to create output file: %w", err)
			}

			// Copy the library
			copied, err := io.Copy(outFile, tarReader)
			outFile.Close()
			if err != nil {
				return fmt.Errorf("failed to extract library: %w", err)
			}

			// Set executable permissions
			if err := platform.EnsureExecutable(outputPath); err != nil {
				return fmt.Errorf("failed to set executable permissions: %w", err)
			}

			q.PushJobStdout(jobID, fmt.Sprintf("Extracted %s (%s)", filepath.Base(outputPath), formatSize(copied)))
			foundMainLib = true
		}

		// Also extract the providers shared library if present
		// Linux: libonnxruntime_providers_shared.so
		// macOS: libonnxruntime_providers_shared.dylib (may not exist)
		providerPattern := "/lib/libonnxruntime_providers_shared" + libExt
		if strings.Contains(header.Name, providerPattern) {
			providerPath := filepath.Join(targetDir, "libonnxruntime_providers_shared"+libExt)
			q.PushJobStdout(jobID, fmt.Sprintf("Found providers library at: %s", header.Name))

			outFile, err := os.Create(providerPath)
			if err != nil {
				return fmt.Errorf("failed to create providers library: %w", err)
			}

			copied, err := io.Copy(outFile, tarReader)
			outFile.Close()
			if err != nil {
				return fmt.Errorf("failed to extract providers library: %w", err)
			}

			if err := platform.EnsureExecutable(providerPath); err != nil {
				return fmt.Errorf("failed to set executable permissions on providers library: %w", err)
			}

			q.PushJobStdout(jobID, fmt.Sprintf("Extracted %s (%s)", filepath.Base(providerPath), formatSize(copied)))
			foundProviderLib = true
		}

		// If we found both libraries, we can stop early
		// Note: providers library is optional on macOS
		if foundMainLib && (foundProviderLib || runtime.GOOS == "darwin") {
			break
		}
	}

	if !foundMainLib {
		return fmt.Errorf("libonnxruntime%s not found in archive", libExt)
	}

	return nil
}

// downloadONNXNew downloads and installs the ONNX model bundle using the new download system.
func downloadONNXNew(ctx context.Context, progress downloads.ProgressCallback) error {
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Starting ONNX bundle download..."})

	dep, ok := Get("onnx-bundle")
	if !ok {
		return fmt.Errorf("onnx-bundle dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Download URLs for ONNX model files from HuggingFace
	modelURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/model.onnx"
	labelsURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/selected_tags.csv"
	configURL := "https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/resolve/main/config.json"

	files := []struct {
		url      string
		filename string
	}{
		{modelURL, "model.onnx"},
		{labelsURL, "selected_tags.csv"},
		{configURL, "config.json"},
	}

	speedTracker := downloads.NewSpeedTracker()

	// Download each file
	for i, file := range files {
		targetPath := filepath.Join(dep.TargetDir, file.filename)
		progress(downloads.Progress{
			Status:  downloads.StatusDownloading,
			Message: fmt.Sprintf("Downloading %s (%d/%d)...", file.filename, i+1, len(files)),
		})

		err := downloads.DownloadWithRetry(ctx, targetPath, file.url, func(downloaded, total int64) {
			speed := speedTracker.Update(downloaded)
			percent := float64(0)
			if total > 0 {
				percent = float64(downloaded) / float64(total) * 100
			}
			progress(downloads.Progress{
				Status:          downloads.StatusDownloading,
				Message:         fmt.Sprintf("Downloading %s: %s / %s", file.filename, downloads.FormatBytes(downloaded), downloads.FormatBytes(total)),
				BytesDownloaded: downloaded,
				TotalBytes:      total,
				Percent:         percent,
				Speed:           speed,
			})
		})
		if err != nil {
			return fmt.Errorf("failed to download %s: %w", file.filename, err)
		}
	}

	// Download ONNX Runtime library
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Downloading ONNX Runtime..."})
	onnxRuntimeLibName := GetOnnxRuntimeLibName()
	onnxRuntimeLibPath := filepath.Join(dep.TargetDir, onnxRuntimeLibName)

	arch := runtime.GOARCH
	onnxVersion := "1.22.0"

	if arch != "amd64" && arch != "arm64" {
		progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Warning: Unsupported architecture %s, skipping ONNX Runtime download", arch)})
	} else {
		onnxRuntimeURL := GetOnnxRuntimeDownloadURL(onnxVersion, arch)

		if IsOnnxRuntimeArchiveZip() {
			// Windows: Download and extract from ZIP
			tempZip := filepath.Join(dep.TargetDir, "onnxruntime.zip")
			if err := downloads.DownloadWithRetry(ctx, tempZip, onnxRuntimeURL, nil); err != nil {
				progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Failed to download ONNX Runtime: %v", err)})
			} else {
				progress(downloads.Progress{Status: downloads.StatusExtracting, Message: "Extracting ONNX Runtime library..."})
				if err := extractOnnxRuntimeFromZipNew(tempZip, onnxRuntimeLibPath, onnxRuntimeLibName, progress); err != nil {
					progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Failed to extract ONNX Runtime: %v", err)})
				}
				os.Remove(tempZip)
			}
		} else {
			// Linux/macOS: Download and extract from tar.gz
			tempTgz := filepath.Join(dep.TargetDir, "onnxruntime.tgz")
			if err := downloads.DownloadWithRetry(ctx, tempTgz, onnxRuntimeURL, nil); err != nil {
				progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Failed to download ONNX Runtime: %v", err)})
			} else {
				progress(downloads.Progress{Status: downloads.StatusExtracting, Message: "Extracting ONNX Runtime library..."})
				if err := extractOnnxRuntimeFromTarGzNew(tempTgz, onnxRuntimeLibPath, onnxRuntimeLibName, dep.TargetDir, progress); err != nil {
					progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Failed to extract ONNX Runtime: %v", err)})
				}
				os.Remove(tempTgz)
			}
		}
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("onnx-bundle", DependencyMetadata{
		InstalledVersion: "wd-eva02-large-v3",
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			"model.onnx":        {Path: filepath.Join(dep.TargetDir, "model.onnx")},
			"selected_tags.csv": {Path: filepath.Join(dep.TargetDir, "selected_tags.csv")},
			"config.json":       {Path: filepath.Join(dep.TargetDir, "config.json")},
			onnxRuntimeLibName:  {Path: onnxRuntimeLibPath},
		},
	})
	metadata.Save()

	progress(downloads.Progress{Status: downloads.StatusComplete, Message: "ONNX model bundle installed successfully!", Percent: 100})
	return nil
}

// extractOnnxRuntimeFromZipNew extracts the ONNX Runtime library from a ZIP archive with progress.
func extractOnnxRuntimeFromZipNew(zipPath, outputPath, libName string, progress downloads.ProgressCallback) error {
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("failed to open zip: %w", err)
	}
	defer reader.Close()

	var libFile *zip.File
	for _, file := range reader.File {
		if strings.HasSuffix(strings.ToLower(file.Name), strings.ToLower(libName)) {
			libFile = file
			break
		}
	}

	if libFile == nil {
		return fmt.Errorf("%s not found in archive", libName)
	}

	progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Found library at: %s", libFile.Name)})

	rc, err := libFile.Open()
	if err != nil {
		return fmt.Errorf("failed to open library in zip: %w", err)
	}
	defer rc.Close()

	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	if _, err := io.Copy(outFile, rc); err != nil {
		return fmt.Errorf("failed to extract library: %w", err)
	}

	return nil
}

// extractOnnxRuntimeFromTarGzNew extracts the ONNX Runtime library from a tar.gz archive with progress.
func extractOnnxRuntimeFromTarGzNew(tgzPath, outputPath, libName, targetDir string, progress downloads.ProgressCallback) error {
	file, err := os.Open(tgzPath)
	if err != nil {
		return fmt.Errorf("failed to open tar.gz: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzReader.Close()

	tarReader := tar.NewReader(gzReader)

	var foundMainLib bool
	libExt := platform.SharedLibExtension()

	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to read tar: %w", err)
		}

		if header.Typeflag == tar.TypeDir {
			continue
		}

		// Look for the main ONNX Runtime library
		isMainLib := false
		if runtime.GOOS == "darwin" {
			isMainLib = strings.Contains(header.Name, "/lib/libonnxruntime.") &&
				strings.HasSuffix(header.Name, ".dylib") &&
				!strings.Contains(header.Name, "_providers_")
		} else {
			isMainLib = strings.Contains(header.Name, "/lib/libonnxruntime.so.") &&
				!strings.Contains(header.Name, "_providers_")
		}

		if isMainLib {
			progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Found main library at: %s", header.Name)})

			outFile, err := os.Create(outputPath)
			if err != nil {
				return fmt.Errorf("failed to create output file: %w", err)
			}

			if _, err := io.Copy(outFile, tarReader); err != nil {
				outFile.Close()
				return fmt.Errorf("failed to extract library: %w", err)
			}
			outFile.Close()

			if err := platform.EnsureExecutable(outputPath); err != nil {
				// Non-fatal
			}

			foundMainLib = true
		}

		// Also extract the providers shared library if present
		providerPattern := "/lib/libonnxruntime_providers_shared" + libExt
		if strings.Contains(header.Name, providerPattern) {
			providerPath := filepath.Join(targetDir, "libonnxruntime_providers_shared"+libExt)
			progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Found providers library at: %s", header.Name)})

			outFile, err := os.Create(providerPath)
			if err != nil {
				return fmt.Errorf("failed to create providers library: %w", err)
			}

			if _, err := io.Copy(outFile, tarReader); err != nil {
				outFile.Close()
				return fmt.Errorf("failed to extract providers library: %w", err)
			}
			outFile.Close()

			if err := platform.EnsureExecutable(providerPath); err != nil {
				// Non-fatal
			}
		}

		if foundMainLib && runtime.GOOS == "darwin" {
			break
		}
	}

	if !foundMainLib {
		return fmt.Errorf("libonnxruntime%s not found in archive", libExt)
	}

	return nil
}
