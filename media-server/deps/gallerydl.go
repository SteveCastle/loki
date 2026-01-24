package deps

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sync"
	"time"

	"github.com/stevecastle/shrike/downloads"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

var LatestGalleryDlVersion = "latest"

func init() {
	Register(&Dependency{
		ID:            "gallery-dl",
		Name:          "gallery-dl",
		Description:   "Image gallery downloader for various websites",
		TargetDir:     GetDepsDir("gallery-dl"),
		Check:         checkGalleryDl,
		Download:      downloadGalleryDl,
		DownloadFn:    downloadGalleryDlNew,
		LatestVersion: LatestGalleryDlVersion,
		DownloadURL:   GetGalleryDlDownloadURL(),
		ExpectedSize:  30 * 1024 * 1024, // ~30MB
	})
}

// checkGalleryDl verifies if gallery-dl executable exists and can run.
func checkGalleryDl(ctx context.Context) (bool, string, error) {
	targetDir := GetDepsDir("gallery-dl")
	exePath := filepath.Join(targetDir, GetExecutableName("gallery-dl"))

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking gallery-dl executable: %w", err)
	}

	// Try to execute with --version flag
	versionCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(versionCtx, exePath, "--version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// File exists but version check failed
		return true, LatestGalleryDlVersion, nil
	}

	version := parseGalleryDlVersion(string(output))
	if version == "unknown" {
		version = LatestGalleryDlVersion
	}
	return true, version, nil
}

// parseGalleryDlVersion extracts version number from gallery-dl's version output.
func parseGalleryDlVersion(output string) string {
	// gallery-dl outputs version like "gallery-dl 1.26.0"
	re := regexp.MustCompile(`gallery-dl (\d+\.\d+\.\d+)`)
	matches := re.FindStringSubmatch(output)
	if len(matches) > 1 {
		return matches[1]
	}
	return "unknown"
}

// downloadGalleryDl downloads and installs gallery-dl.
func downloadGalleryDl(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting gallery-dl download...")

	dep, ok := Get("gallery-dl")
	if !ok {
		return fmt.Errorf("gallery-dl dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// gallery-dl is a single executable, download directly
	exeName := GetExecutableName("gallery-dl")
	exePath := filepath.Join(dep.TargetDir, exeName)

	q.PushJobStdout(j.ID, "Downloading gallery-dl...")
	if err := downloadFile(j.Ctx, exePath, downloadURL, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download failed: %v", err))
		return fmt.Errorf("download failed: %w", err)
	}
	q.PushJobStdout(j.ID, "✓ Download complete")

	// Make executable on Linux
	if err := platform.EnsureExecutable(exePath); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: could not set executable permissions: %v", err))
	}

	// Verify the executable
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		q.PushJobStdout(j.ID, fmt.Sprintf("Warning: %s not found at %s", exeName, exePath))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("✓ Found executable at: %s", exePath))
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("gallery-dl", DependencyMetadata{
		InstalledVersion: LatestGalleryDlVersion,
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
	q.PushJobStdout(j.ID, "✓ gallery-dl installed successfully!")

	return nil
}

// downloadGalleryDlNew downloads and installs gallery-dl using the new download system.
func downloadGalleryDlNew(ctx context.Context, progress downloads.ProgressCallback) error {
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Starting gallery-dl download..."})

	dep, ok := Get("gallery-dl")
	if !ok {
		return fmt.Errorf("gallery-dl dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	downloadURL := dep.DownloadURL
	if downloadURL == "" {
		return fmt.Errorf("gallery-dl is not available for direct download on this platform")
	}

	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Downloading from %s", downloadURL)})

	// gallery-dl is a single executable, download directly
	exeName := GetExecutableName("gallery-dl")
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

	// Verify the executable
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return fmt.Errorf("%s not found at %s", exeName, exePath)
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("gallery-dl", DependencyMetadata{
		InstalledVersion: LatestGalleryDlVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			exeName: {Path: exePath},
		},
	})
	metadata.Save()

	progress(downloads.Progress{Status: downloads.StatusComplete, Message: "gallery-dl installed successfully!", Percent: 100})
	return nil
}
