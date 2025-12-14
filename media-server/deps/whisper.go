package deps

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sync"
	"time"

	"github.com/bodgit/sevenzip"
	"github.com/stevecastle/shrike/jobqueue"
)

var LatestFasterWhisperVersion = "1.1.1"

func init() {
	Register(&Dependency{
		ID:            "faster-whisper",
		Name:          "Faster Whisper",
		Description:   "High-performance audio transcription engine for generating video subtitles",
		TargetDir:     filepath.Join(os.Getenv("APPDATA"), "Lowkey Media Viewer", "whisper"),
		Check:         checkWhisper,
		Download:      downloadWhisper,
		LatestVersion: LatestFasterWhisperVersion,
		DownloadURL:   "https://github.com/Purfview/whisper-standalone-win/releases/download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.1_windows.7z",
		ExpectedSize:  200 * 1024 * 1024, // 200MB (approximate)
	})
}

// checkWhisper verifies if Faster Whisper executable exists and can run.
func checkWhisper(ctx context.Context) (bool, string, error) {
	// Always use default installation location
	targetDir := filepath.Join(os.Getenv("APPDATA"), "Lowkey Media Viewer", "whisper")
	exePath := filepath.Join(targetDir, "faster-whisper-xxl.exe")

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking whisper executable: %w", err)
	}

	// Try to execute with --version flag (with timeout to avoid hanging)
	versionCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(versionCtx, exePath, "--version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// File exists but version check failed - return installed with latest version
		return true, LatestFasterWhisperVersion, nil
	}

	// Parse version from output
	version := parseWhisperVersion(string(output))
	if version == "unknown" {
		version = LatestFasterWhisperVersion
	}
	return true, version, nil
}

// parseWhisperVersion extracts version number from Whisper's version output.
func parseWhisperVersion(output string) string {
	// Look for version patterns like "0.10.0" or "v0.10.0"
	re := regexp.MustCompile(`v?(\d+\.\d+\.\d+)`)
	matches := re.FindStringSubmatch(output)
	if len(matches) > 1 {
		return matches[1]
	}
	return "unknown"
}

// downloadWhisper downloads and installs Faster Whisper executable.
func downloadWhisper(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting Faster Whisper download...")

	dep, ok := Get("faster-whisper")
	if !ok {
		return fmt.Errorf("faster-whisper dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	// Determine architecture
	arch := runtime.GOARCH
	q.PushJobStdout(j.ID, fmt.Sprintf("Detected architecture: %s", arch))

	if arch != "amd64" {
		q.PushJobStdout(j.ID, fmt.Sprintf("Unsupported architecture: %s", arch))
		q.PushJobStdout(j.ID, "Faster Whisper is only available for Windows x64")
		return fmt.Errorf("unsupported architecture: %s", arch)
	}

	// Use the download URL from dependency config
	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// Download the 7z file
	temp7z := filepath.Join(dep.TargetDir, "faster-whisper.7z")
	q.PushJobStdout(j.ID, "Downloading Faster Whisper package...")

	if err := downloadFile(j.Ctx, temp7z, downloadURL, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download failed: %v", err))
		return fmt.Errorf("download failed: %w", err)
	}

	q.PushJobStdout(j.ID, "✓ Download complete")

	// Extract the 7z file
	q.PushJobStdout(j.ID, "Extracting files...")
	if err := extractWhisper7z(temp7z, dep.TargetDir, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Extraction failed: %v", err))
		q.PushJobStdout(j.ID, "")
		q.PushJobStdout(j.ID, "The archive was downloaded successfully but extraction failed.")
		q.PushJobStdout(j.ID, fmt.Sprintf("Please manually extract: %s", temp7z))
		q.PushJobStdout(j.ID, fmt.Sprintf("Extract to: %s", dep.TargetDir))
		return fmt.Errorf("extraction failed: %w", err)
	}

	// Clean up 7z file
	os.Remove(temp7z)
	q.PushJobStdout(j.ID, "✓ Extraction complete")

	// Verify the executable
	exePath := filepath.Join(dep.TargetDir, "faster-whisper-xxl.exe")
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		q.PushJobStdout(j.ID, "Warning: faster-whisper-xxl.exe not found in expected location")
		q.PushJobStdout(j.ID, fmt.Sprintf("Expected: %s", exePath))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("✓ Found executable at: %s", exePath))
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("faster-whisper", DependencyMetadata{
		InstalledVersion: LatestFasterWhisperVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files: map[string]FileInfo{
			"faster-whisper-xxl.exe": {Path: exePath},
		},
	})
	metadata.Save()

	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "✓ Faster Whisper installed successfully!")
	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "Note: This tool requires CUDA/GPU support for optimal performance.")
	q.PushJobStdout(j.ID, "If you don't have a compatible GPU, transcription may be slower.")

	return nil
}

// extractWhisper7z extracts all files from the Whisper 7z archive using pure Go.
func extractWhisper7z(archivePath, destDir string, jobID string, q *jobqueue.Queue) error {
	q.PushJobStdout(jobID, "Opening 7z archive...")

	// Open the 7z archive
	reader, err := sevenzip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open 7z archive: %w", err)
	}
	defer reader.Close()

	q.PushJobStdout(jobID, fmt.Sprintf("Found %d files in archive", len(reader.File)))
	q.PushJobStdout(jobID, "Extracting files...")

	// Extract each file
	for i, file := range reader.File {
		if err := extractFile(file, destDir, jobID, q); err != nil {
			return fmt.Errorf("failed to extract %s: %w", file.Name, err)
		}

		// Report progress every 10 files or for the last file
		if (i+1)%10 == 0 || i == len(reader.File)-1 {
			q.PushJobStdout(jobID, fmt.Sprintf("Extracted %d/%d files...", i+1, len(reader.File)))
		}
	}

	q.PushJobStdout(jobID, "✓ Archive extracted successfully")
	return nil
}

// extractFile extracts a single file from the 7z archive.
func extractFile(file *sevenzip.File, destDir string, jobID string, q *jobqueue.Queue) error {
	// Get the file info
	info := file.FileInfo()

	// Strip "Faster-Whisper-XXL/" prefix from path to move files to root
	fileName := file.Name
	if len(fileName) > 18 && fileName[:18] == "Faster-Whisper-XXL/" {
		fileName = fileName[19:] // Skip "Faster-Whisper-XXL/"
	} else if len(fileName) > 18 && fileName[:18] == "Faster-Whisper-XXL\\" {
		fileName = fileName[19:] // Skip "Faster-Whisper-XXL\"
	}

	// Skip if the filename is empty (was just the directory itself)
	if fileName == "" {
		return nil
	}

	// Construct destination path
	destPath := filepath.Join(destDir, fileName)

	// Create parent directories if needed
	if err := os.MkdirAll(filepath.Dir(destPath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// If it's a directory, just create it and return
	if info.IsDir() {
		return os.MkdirAll(destPath, 0755)
	}

	// Open the file in the archive
	rc, err := file.Open()
	if err != nil {
		return fmt.Errorf("failed to open file in archive: %w", err)
	}
	defer rc.Close()

	// Create the destination file
	outFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %w", err)
	}
	defer outFile.Close()

	// Copy the contents
	if _, err := io.Copy(outFile, rc); err != nil {
		return fmt.Errorf("failed to copy file contents: %w", err)
	}

	return nil
}

// formatBytesWhisper formats bytes as human-readable size.
func formatBytesWhisper(bytes int64) string {
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
