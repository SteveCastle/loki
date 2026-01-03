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

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

var LatestYtDlpVersion = "2025.12.08"

func init() {
	Register(&Dependency{
		ID:            "yt-dlp",
		Name:          "yt-dlp",
		Description:   "Video downloader for YouTube and other video platforms",
		TargetDir:     GetDepsDir("yt-dlp"),
		Check:         checkYtDlp,
		Download:      downloadYtDlp,
		LatestVersion: LatestYtDlpVersion,
		DownloadURL:   GetYtDlpDownloadURL(),
		ExpectedSize:  20 * 1024 * 1024, // ~20MB
	})
}

// checkYtDlp verifies if yt-dlp executable exists and can run.
func checkYtDlp(ctx context.Context) (bool, string, error) {
	targetDir := GetDepsDir("yt-dlp")
	exePath := filepath.Join(targetDir, GetExecutableName("yt-dlp"))

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking yt-dlp executable: %w", err)
	}

	// Try to execute with --version flag
	versionCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	cmd := exec.CommandContext(versionCtx, exePath, "--version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// File exists but version check failed
		return true, LatestYtDlpVersion, nil
	}

	version := parseYtDlpVersion(string(output))
	if version == "unknown" {
		version = LatestYtDlpVersion
	}
	return true, version, nil
}

// parseYtDlpVersion extracts version number from yt-dlp's version output.
func parseYtDlpVersion(output string) string {
	// yt-dlp outputs version like "2024.01.01" or "2024.01.01.123456"
	re := regexp.MustCompile(`(\d{4}\.\d{2}\.\d{2}(?:\.\d+)?)`)
	matches := re.FindStringSubmatch(output)
	if len(matches) > 1 {
		return matches[1]
	}
	return "unknown"
}

// downloadYtDlp downloads and installs yt-dlp.
func downloadYtDlp(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting yt-dlp download...")

	dep, ok := Get("yt-dlp")
	if !ok {
		return fmt.Errorf("yt-dlp dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// yt-dlp is a single executable, download directly
	exeName := GetExecutableName("yt-dlp")
	exePath := filepath.Join(dep.TargetDir, exeName)

	q.PushJobStdout(j.ID, "Downloading yt-dlp...")
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
	metadata.Update("yt-dlp", DependencyMetadata{
		InstalledVersion: LatestYtDlpVersion,
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
	q.PushJobStdout(j.ID, "✓ yt-dlp installed successfully!")

	return nil
}
