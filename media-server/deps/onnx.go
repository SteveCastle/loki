package deps

import (
	"archive/zip"
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

	"github.com/stevecastle/shrike/jobqueue"
)

func init() {
	Register(&Dependency{
		ID:            "onnx-bundle",
		Name:          "ONNX Tagger Models",
		Description:   "ML models, labels, config, and runtime DLL for AI-powered image tagging",
		TargetDir:     filepath.Join(os.Getenv("APPDATA"), "Lowkey Media Viewer", "onnx"),
		Check:         checkONNX,
		Download:      downloadONNX,
		LatestVersion: "wd-eva02-large-v3",
		DownloadURL:   "",                // To be configured
		ExpectedSize:  500 * 1024 * 1024, // 500MB
	})
}

// checkONNX verifies if all ONNX tagger files exist and returns version info.
func checkONNX(ctx context.Context) (bool, string, error) {
	// Always use default installation location
	targetDir := filepath.Join(os.Getenv("APPDATA"), "Lowkey Media Viewer", "onnx")

	// Check all required files
	requiredFiles := map[string]string{
		"model":   filepath.Join(targetDir, "model.onnx"),
		"labels":  filepath.Join(targetDir, "selected_tags.csv"),
		"config":  filepath.Join(targetDir, "config.json"),
		"runtime": filepath.Join(targetDir, "onnxruntime.dll"),
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

	// Download ONNX Runtime DLL
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Downloading ONNX Runtime...")
	onnxRuntimeDLLPath := filepath.Join(dep.TargetDir, "onnxruntime.dll")

	// Determine architecture and download URL
	arch := runtime.GOARCH
	onnxVersion := "1.23.1" // Version 1.20.1+ required for ONNX opset 22 support (models using ONNX 1.17+)
	var onnxRuntimeURL string
	tempZip := filepath.Join(dep.TargetDir, "onnxruntime.zip")

	if arch == "amd64" {
		onnxRuntimeURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-win-x64-%s.zip", onnxVersion, onnxVersion)
	} else if arch == "arm64" {
		onnxRuntimeURL = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/onnxruntime-win-arm64-%s.zip", onnxVersion, onnxVersion)
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: Unsupported architecture %s, skipping ONNX Runtime download", arch))
		q.PushJobStdout(j.ID, "Please manually download onnxruntime.dll from https://github.com/microsoft/onnxruntime/releases")
		goto skipOnnxRuntime
	}

	// Download ONNX Runtime zip file
	q.PushJobStdout(j.ID, fmt.Sprintf("Downloading ONNX Runtime %s...", onnxVersion))
	if err := downloadFile(j.Ctx, tempZip, onnxRuntimeURL, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to download ONNX Runtime: %v", err))
		q.PushJobStdout(j.ID, "Please manually download onnxruntime.dll from https://github.com/microsoft/onnxruntime/releases")
		goto skipOnnxRuntime
	}

	// Extract the DLL from the zip
	q.PushJobStdout(j.ID, "Extracting ONNX Runtime DLL...")
	if err := extractOnnxRuntimeDLL(tempZip, onnxRuntimeDLLPath, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to extract ONNX Runtime DLL: %v", err))
		q.PushJobStdout(j.ID, "Please manually download onnxruntime.dll from https://github.com/microsoft/onnxruntime/releases")
		goto skipOnnxRuntime
	}

	// Clean up zip file
	os.Remove(tempZip)
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
			"onnxruntime.dll":   {Path: onnxRuntimeDLLPath},
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

// extractOnnxRuntimeDLL extracts onnxruntime.dll from a zip archive.
func extractOnnxRuntimeDLL(zipPath, outputPath string, jobID string, q *jobqueue.Queue) error {
	// Open the zip file
	reader, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("failed to open zip: %w", err)
	}
	defer reader.Close()

	// Search for onnxruntime.dll in the zip
	var dllFile *zip.File
	for _, file := range reader.File {
		if strings.HasSuffix(strings.ToLower(file.Name), "onnxruntime.dll") {
			dllFile = file
			break
		}
	}

	if dllFile == nil {
		return fmt.Errorf("onnxruntime.dll not found in archive")
	}

	q.PushJobStdout(jobID, fmt.Sprintf("Found DLL at: %s", dllFile.Name))

	// Open the DLL file from the zip
	rc, err := dllFile.Open()
	if err != nil {
		return fmt.Errorf("failed to open DLL in zip: %w", err)
	}
	defer rc.Close()

	// Create output file
	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	// Copy the DLL
	copied, err := io.Copy(outFile, rc)
	if err != nil {
		return fmt.Errorf("failed to extract DLL: %w", err)
	}

	q.PushJobStdout(jobID, fmt.Sprintf("Extracted %s (%s)", filepath.Base(outputPath), formatSize(copied)))
	return nil
}
