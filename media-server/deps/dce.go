package deps

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

var LatestDceVersion = "1.0.0"

func init() {
	Register(&Dependency{
		ID:            "dce",
		Name:          "Discord Chat Exporter",
		Description:   "Discord content exporter for downloading media from Discord",
		TargetDir:     GetDepsDir("dce"),
		Check:         checkDce,
		Download:      downloadDce,
		LatestVersion: LatestDceVersion,
		DownloadURL:   getDceDownloadURL(),
		ExpectedSize:  10 * 1024 * 1024, // ~10MB
	})
}

// getDceDownloadURL returns the platform-specific download URL.
func getDceDownloadURL() string {
	// TODO: Update with actual release URL
	if runtime.GOOS == "windows" {
		return "TODO: Add Windows release URL for dce"
	}
	return "TODO: Add Linux release URL for dce"
}

// checkDce verifies if dce executable exists.
func checkDce(ctx context.Context) (bool, string, error) {
	targetDir := GetDepsDir("dce")
	exePath := filepath.Join(targetDir, GetExecutableName("dce"))

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking dce executable: %w", err)
	}

	return true, LatestDceVersion, nil
}

// downloadDce downloads and installs dce.
func downloadDce(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting dce download...")

	dep, ok := Get("dce")
	if !ok {
		return fmt.Errorf("dce dependency not found in registry")
	}

	// Check if download URL is configured
	if dep.DownloadURL == "" || dep.DownloadURL[:4] == "TODO" {
		q.PushJobStdout(j.ID, "Error: dce download URL not configured")
		q.PushJobStdout(j.ID, "Please update the download URL in deps/dce.go")
		return fmt.Errorf("dce download URL not configured")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// Download the executable
	exeName := GetExecutableName("dce")
	exePath := filepath.Join(dep.TargetDir, exeName)

	q.PushJobStdout(j.ID, "Downloading dce...")
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
	metadata.Update("dce", DependencyMetadata{
		InstalledVersion: LatestDceVersion,
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
	q.PushJobStdout(j.ID, "✓ dce installed successfully!")

	return nil
}
