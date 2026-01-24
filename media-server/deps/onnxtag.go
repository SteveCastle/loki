package deps

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/stevecastle/shrike/downloads"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

var LatestOnnxtagVersion = "2.7.3"

func init() {
	Register(&Dependency{
		ID:            "onnxtag",
		Name:          "ONNX Tagger",
		Description:   "AI-powered image tagging using ONNX models",
		TargetDir:     GetDepsDir("onnxtag"),
		Check:         checkOnnxtag,
		Download:      downloadOnnxtag,
		DownloadFn:    downloadOnnxtagNew,
		LatestVersion: LatestOnnxtagVersion,
		DownloadURL:   getOnnxtagDownloadURL(),
		ExpectedSize:  10 * 1024 * 1024, // ~10MB
	})
}

// getOnnxtagDownloadURL returns the platform-specific download URL.
func getOnnxtagDownloadURL() string {
	baseURL := "https://github.com/SteveCastle/loki/releases/download/v" + LatestOnnxtagVersion
	if runtime.GOOS == "windows" {
		return baseURL + "/onnxtag.exe"
	}
	return baseURL + "/onnxtag"
}

// checkOnnxtag verifies if onnxtag executable exists.
func checkOnnxtag(ctx context.Context) (bool, string, error) {
	targetDir := GetDepsDir("onnxtag")
	exePath := filepath.Join(targetDir, GetExecutableName("onnxtag"))

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking onnxtag executable: %w", err)
	}

	return true, LatestOnnxtagVersion, nil
}

// downloadOnnxtag downloads and installs onnxtag.
func downloadOnnxtag(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting onnxtag download...")

	dep, ok := Get("onnxtag")
	if !ok {
		return fmt.Errorf("onnxtag dependency not found in registry")
	}

	// Check if download URL is configured
	if dep.DownloadURL == "" || dep.DownloadURL[:4] == "TODO" {
		q.PushJobStdout(j.ID, "Error: onnxtag download URL not configured")
		q.PushJobStdout(j.ID, "Please update the download URL in deps/onnxtag.go")
		return fmt.Errorf("onnxtag download URL not configured")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// Download the executable
	exeName := GetExecutableName("onnxtag")
	exePath := filepath.Join(dep.TargetDir, exeName)

	q.PushJobStdout(j.ID, "Downloading onnxtag...")
	if err := downloadFile(j.Ctx, exePath, downloadURL, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download failed: %v", err))
		return fmt.Errorf("download failed: %w", err)
	}
	q.PushJobStdout(j.ID, "✓ Download complete")

	// Make executable on Linux
	if err := platform.EnsureExecutable(exePath); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not set executable permissions: %v", err))
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("onnxtag", DependencyMetadata{
		InstalledVersion: LatestOnnxtagVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			exeName: {Path: exePath},
		},
	})
	metadata.Save()

	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "✓ onnxtag installed successfully!")

	return nil
}

// downloadOnnxtagNew downloads and installs onnxtag using the new download system.
func downloadOnnxtagNew(ctx context.Context, progress downloads.ProgressCallback) error {
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Starting onnxtag download..."})

	dep, ok := Get("onnxtag")
	if !ok {
		return fmt.Errorf("onnxtag dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	downloadURL := dep.DownloadURL
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Downloading from %s", downloadURL)})

	// Download the executable
	exeName := GetExecutableName("onnxtag")
	exePath := filepath.Join(dep.TargetDir, exeName)

	speedTracker := downloads.NewSpeedTracker()

	err := downloads.DownloadWithRetry(ctx, exePath, downloadURL, func(downloaded, total int64) {
		speed := speedTracker.Update(downloaded)
		percent := float64(0)
		if total > 0 {
			percent = float64(downloaded) / float64(total) * 100
		}
		progress(downloads.Progress{
			Status:          downloads.StatusDownloading,
			Message:         fmt.Sprintf("Downloading: %s / %s", downloads.FormatBytes(downloaded), downloads.FormatBytes(total)),
			BytesDownloaded: downloaded,
			TotalBytes:      total,
			Percent:         percent,
			Speed:           speed,
		})
	})
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}

	// Make executable on Linux/macOS
	if err := platform.EnsureExecutable(exePath); err != nil {
		// Non-fatal warning
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("onnxtag", DependencyMetadata{
		InstalledVersion: LatestOnnxtagVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			exeName: {Path: exePath},
		},
	})
	metadata.Save()

	progress(downloads.Progress{Status: downloads.StatusComplete, Message: "onnxtag installed successfully!", Percent: 100})
	return nil
}
