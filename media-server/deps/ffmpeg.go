package deps

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/downloads"
	"github.com/stevecastle/shrike/jobqueue"
)

var LatestFFmpegVersion = "N-122535-gff63d4865b-20260124"

func init() {
	Register(&Dependency{
		ID:            "ffmpeg",
		Name:          "FFmpeg",
		Description:   "Media processing toolkit for video/audio conversion, encoding, and analysis",
		TargetDir:     GetDepsDir("ffmpeg"),
		Check:         checkFFmpeg,
		Download:      downloadFFmpeg,
		DownloadFn:    downloadFFmpegNew,
		LatestVersion: LatestFFmpegVersion,
		DownloadURL:   GetFFmpegDownloadURL(),
		ExpectedSize:  150 * 1024 * 1024, // ~150MB compressed
	})
}

// checkFFmpeg verifies if FFmpeg executable exists and can run.
func checkFFmpeg(ctx context.Context) (bool, string, error) {
	targetDir := GetDepsDir("ffmpeg")
	exePath := filepath.Join(targetDir, GetExecutableName("ffmpeg"))

	// Check if file exists
	if _, err := os.Stat(exePath); os.IsNotExist(err) {
		return false, "", nil
	} else if err != nil {
		return false, "", fmt.Errorf("error checking ffmpeg executable: %w", err)
	}

	// Try to execute with -version flag
	versionCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	cmd := exec.CommandContext(versionCtx, exePath, "-version")
	output, err := cmd.CombinedOutput()
	if err != nil {
		// File exists but version check failed
		return true, LatestFFmpegVersion, nil
	}

	version := parseFFmpegVersion(string(output))
	if version == "unknown" {
		version = LatestFFmpegVersion
	}
	return true, version, nil
}

// parseFFmpegVersion extracts version number from FFmpeg's version output.
func parseFFmpegVersion(output string) string {
	// Look for version patterns like "ffmpeg version N-xxxxx-g..." or "ffmpeg version 6.0"
	re := regexp.MustCompile(`ffmpeg version (\S+)`)
	matches := re.FindStringSubmatch(output)
	if len(matches) > 1 {
		return matches[1]
	}
	return "unknown"
}

// downloadFFmpeg downloads and installs FFmpeg.
func downloadFFmpeg(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	q.PushJobStdout(j.ID, "Starting FFmpeg download...")

	dep, ok := Get("ffmpeg")
	if !ok {
		return fmt.Errorf("ffmpeg dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Failed to create directory: %v", err))
		return err
	}

	downloadURL := dep.DownloadURL
	q.PushJobStdout(j.ID, fmt.Sprintf("Download URL: %s", downloadURL))

	// Determine archive type based on platform
	var archivePath string
	switch runtime.GOOS {
	case "windows", "darwin":
		archivePath = filepath.Join(dep.TargetDir, "ffmpeg.zip")
	default: // linux
		archivePath = filepath.Join(dep.TargetDir, "ffmpeg.tar.xz")
	}

	q.PushJobStdout(j.ID, "Downloading FFmpeg...")
	if err := downloadFile(j.Ctx, archivePath, downloadURL, j.ID, q); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Download failed: %v", err))
		return fmt.Errorf("download failed: %w", err)
	}
	q.PushJobStdout(j.ID, "✓ Download complete")

	// Extract the archive
	q.PushJobStdout(j.ID, "Extracting files...")
	switch runtime.GOOS {
	case "windows":
		if err := extractFFmpegZip(archivePath, dep.TargetDir, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Extraction failed: %v", err))
			return fmt.Errorf("extraction failed: %w", err)
		}
	case "darwin":
		if err := extractFFmpegZipDarwin(archivePath, dep.TargetDir, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Extraction failed: %v", err))
			return fmt.Errorf("extraction failed: %w", err)
		}
		// On macOS, FFprobe is downloaded separately
		q.PushJobStdout(j.ID, "Downloading FFprobe...")
		ffprobeURL := GetFFprobeDownloadURL()
		if ffprobeURL != "" {
			ffprobeArchive := filepath.Join(dep.TargetDir, "ffprobe.zip")
			if err := downloadFile(j.Ctx, ffprobeArchive, ffprobeURL, j.ID, q); err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("FFprobe download failed: %v", err))
				return fmt.Errorf("ffprobe download failed: %w", err)
			}
			if err := extractFFmpegZipDarwin(ffprobeArchive, dep.TargetDir, j.ID, q); err != nil {
				q.PushJobStdout(j.ID, fmt.Sprintf("FFprobe extraction failed: %v", err))
				return fmt.Errorf("ffprobe extraction failed: %w", err)
			}
			os.Remove(ffprobeArchive)
			q.PushJobStdout(j.ID, "✓ FFprobe download complete")
		}
	default: // linux
		if err := extractFFmpegTarXz(archivePath, dep.TargetDir, j.ID, q); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Extraction failed: %v", err))
			return fmt.Errorf("extraction failed: %w", err)
		}
	}

	// Clean up archive
	os.Remove(archivePath)
	q.PushJobStdout(j.ID, "✓ Extraction complete")

	// Verify executables
	executables := []string{"ffmpeg", "ffprobe"}
	files := make(map[string]FileInfo)
	for _, exe := range executables {
		exePath := filepath.Join(dep.TargetDir, GetExecutableName(exe))
		if _, err := os.Stat(exePath); os.IsNotExist(err) {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: %s not found at %s", exe, exePath))
		} else {
			q.PushJobStdout(j.ID, fmt.Sprintf("✓ Found %s at: %s", exe, exePath))
			files[GetExecutableName(exe)] = FileInfo{Path: exePath}
		}
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("ffmpeg", DependencyMetadata{
		InstalledVersion: LatestFFmpegVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files:            files,
	})
	metadata.Save()

	q.PushJobStdout(j.ID, "")
	q.PushJobStdout(j.ID, "✓ FFmpeg installed successfully!")

	return nil
}

// extractFFmpegZip extracts FFmpeg from a ZIP archive (Windows).
func extractFFmpegZip(archivePath, destDir string, jobID string, q *jobqueue.Queue) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	q.PushJobStdout(jobID, fmt.Sprintf("Found %d files in archive", len(reader.File)))

	// Find the bin directory prefix (e.g., "ffmpeg-master-latest-win64-gpl/bin/")
	var binPrefix string
	for _, file := range reader.File {
		if strings.Contains(file.Name, "/bin/ffmpeg") {
			parts := strings.Split(file.Name, "/bin/")
			if len(parts) > 0 {
				binPrefix = parts[0] + "/bin/"
				break
			}
		}
	}

	if binPrefix == "" {
		return fmt.Errorf("could not find bin directory in archive")
	}

	q.PushJobStdout(jobID, fmt.Sprintf("Extracting from: %s", binPrefix))

	// Extract only the bin directory contents
	for _, file := range reader.File {
		if !strings.HasPrefix(file.Name, binPrefix) {
			continue
		}

		// Get the filename without the prefix
		relPath := strings.TrimPrefix(file.Name, binPrefix)
		if relPath == "" || file.FileInfo().IsDir() {
			continue
		}

		destPath := filepath.Join(destDir, relPath)

		// Create the file
		outFile, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", destPath, err)
		}

		rc, err := file.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
		}

		_, err = io.Copy(outFile, rc)
		rc.Close()
		outFile.Close()
		if err != nil {
			return fmt.Errorf("failed to extract %s: %w", file.Name, err)
		}

		q.PushJobStdout(jobID, fmt.Sprintf("Extracted: %s", relPath))
	}

	return nil
}

// extractFFmpegZipDarwin extracts FFmpeg/FFprobe from a ZIP archive (macOS).
// The evermeet.cx ZIP contains the executable directly at the root level.
func extractFFmpegZipDarwin(archivePath, destDir string, jobID string, q *jobqueue.Queue) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	q.PushJobStdout(jobID, fmt.Sprintf("Found %d files in archive", len(reader.File)))

	for _, file := range reader.File {
		// Skip directories
		if file.FileInfo().IsDir() {
			continue
		}

		// Get the filename (basename only)
		fileName := filepath.Base(file.Name)

		// Only extract ffmpeg or ffprobe executables
		if fileName != "ffmpeg" && fileName != "ffprobe" && fileName != "ffplay" {
			continue
		}

		destPath := filepath.Join(destDir, fileName)

		// Create the file
		outFile, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", destPath, err)
		}

		rc, err := file.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
		}

		_, err = io.Copy(outFile, rc)
		rc.Close()
		outFile.Close()
		if err != nil {
			return fmt.Errorf("failed to extract %s: %w", file.Name, err)
		}

		// Make executable
		if err := os.Chmod(destPath, 0755); err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: could not set permissions on %s: %v", fileName, err))
		}

		q.PushJobStdout(jobID, fmt.Sprintf("Extracted: %s", fileName))
	}

	return nil
}

// extractFFmpegTarXz extracts FFmpeg from a tar.xz archive (Linux).
func extractFFmpegTarXz(archivePath, destDir string, jobID string, q *jobqueue.Queue) error {
	// Open the file
	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open archive: %w", err)
	}
	defer file.Close()

	// For .tar.xz, we need to use xz decompression
	// Since Go doesn't have built-in xz support, we'll use the system xz command
	// or fall back to a pure Go implementation

	// Try using system tar command which handles xz on most Linux systems
	cmd := exec.Command("tar", "-xf", archivePath, "-C", destDir, "--strip-components=2", "--wildcards", "*/bin/*")
	output, err := cmd.CombinedOutput()
	if err != nil {
		q.PushJobStdout(jobID, fmt.Sprintf("tar extraction output: %s", string(output)))
		return fmt.Errorf("tar extraction failed: %w", err)
	}

	// Make executables executable
	executables := []string{"ffmpeg", "ffprobe", "ffplay"}
	for _, exe := range executables {
		exePath := filepath.Join(destDir, exe)
		if err := os.Chmod(exePath, 0755); err != nil {
			q.PushJobStdout(jobID, fmt.Sprintf("Warning: could not set permissions on %s: %v", exe, err))
		}
	}

	return nil
}

// extractTarGz extracts a tar.gz archive (helper for other dependencies).
func extractTarGz(archivePath, destDir string) error {
	file, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		target := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, 0755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(target), 0755); err != nil {
				return err
			}
			outFile, err := os.Create(target)
			if err != nil {
				return err
			}
			if _, err := io.Copy(outFile, tr); err != nil {
				outFile.Close()
				return err
			}
			outFile.Close()
		}
	}

	return nil
}

// downloadFFmpegNew downloads and installs FFmpeg using the new download system.
func downloadFFmpegNew(ctx context.Context, progress downloads.ProgressCallback) error {
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Starting FFmpeg download..."})

	dep, ok := Get("ffmpeg")
	if !ok {
		return fmt.Errorf("ffmpeg dependency not found in registry")
	}

	// Ensure target directory exists
	if err := os.MkdirAll(dep.TargetDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	downloadURL := dep.DownloadURL
	progress(downloads.Progress{Status: downloads.StatusDownloading, Message: fmt.Sprintf("Downloading from %s", downloadURL)})

	// Determine archive type based on platform
	var archivePath string
	switch runtime.GOOS {
	case "windows", "darwin":
		archivePath = filepath.Join(dep.TargetDir, "ffmpeg.zip")
	default: // linux
		archivePath = filepath.Join(dep.TargetDir, "ffmpeg.tar.xz")
	}

	// Create speed tracker for progress reporting
	speedTracker := downloads.NewSpeedTracker()

	// Download the archive
	err := downloads.DownloadWithRetry(ctx, archivePath, downloadURL, func(downloaded, total int64) {
		speed := speedTracker.Update(downloaded)
		percent := float64(0)
		if total > 0 {
			percent = float64(downloaded) / float64(total) * 100
		}
		progress(downloads.Progress{
			Status:          downloads.StatusDownloading,
			Message:         fmt.Sprintf("Downloading FFmpeg: %s / %s", downloads.FormatBytes(downloaded), downloads.FormatBytes(total)),
			BytesDownloaded: downloaded,
			TotalBytes:      total,
			Percent:         percent,
			Speed:           speed,
		})
	})
	if err != nil {
		return fmt.Errorf("download failed: %w", err)
	}

	progress(downloads.Progress{Status: downloads.StatusExtracting, Message: "Extracting files..."})

	// Extract the archive
	switch runtime.GOOS {
	case "windows":
		if err := extractFFmpegZipNew(archivePath, dep.TargetDir, progress); err != nil {
			return fmt.Errorf("extraction failed: %w", err)
		}
	case "darwin":
		if err := extractFFmpegZipDarwinNew(archivePath, dep.TargetDir, progress); err != nil {
			return fmt.Errorf("extraction failed: %w", err)
		}
		// On macOS, FFprobe is downloaded separately
		progress(downloads.Progress{Status: downloads.StatusDownloading, Message: "Downloading FFprobe..."})
		ffprobeURL := GetFFprobeDownloadURL()
		if ffprobeURL != "" {
			ffprobeArchive := filepath.Join(dep.TargetDir, "ffprobe.zip")
			if err := downloads.DownloadWithRetry(ctx, ffprobeArchive, ffprobeURL, nil); err != nil {
				return fmt.Errorf("ffprobe download failed: %w", err)
			}
			if err := extractFFmpegZipDarwinNew(ffprobeArchive, dep.TargetDir, progress); err != nil {
				return fmt.Errorf("ffprobe extraction failed: %w", err)
			}
			os.Remove(ffprobeArchive)
		}
	default: // linux
		if err := extractFFmpegTarXzNew(archivePath, dep.TargetDir, progress); err != nil {
			return fmt.Errorf("extraction failed: %w", err)
		}
	}

	// Clean up archive
	os.Remove(archivePath)

	// Verify executables
	executables := []string{"ffmpeg", "ffprobe"}
	files := make(map[string]FileInfo)
	for _, exe := range executables {
		exePath := filepath.Join(dep.TargetDir, GetExecutableName(exe))
		if _, err := os.Stat(exePath); err == nil {
			files[GetExecutableName(exe)] = FileInfo{Path: exePath}
		}
	}

	// Update metadata
	metadata := GetMetadataStore()
	metadata.Update("ffmpeg", DependencyMetadata{
		InstalledVersion: LatestFFmpegVersion,
		Status:           StatusInstalled,
		InstallPath:      dep.TargetDir,
		LastChecked:      time.Now(),
		LastUpdated:      time.Now(),
		Files:            files,
	})
	metadata.Save()

	progress(downloads.Progress{Status: downloads.StatusComplete, Message: "FFmpeg installed successfully!", Percent: 100})
	return nil
}

// extractFFmpegZipNew extracts FFmpeg from a ZIP archive (Windows) with progress.
func extractFFmpegZipNew(archivePath, destDir string, progress downloads.ProgressCallback) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	// Find the bin directory prefix
	var binPrefix string
	for _, file := range reader.File {
		if strings.Contains(file.Name, "/bin/ffmpeg") {
			parts := strings.Split(file.Name, "/bin/")
			if len(parts) > 0 {
				binPrefix = parts[0] + "/bin/"
				break
			}
		}
	}

	if binPrefix == "" {
		return fmt.Errorf("could not find bin directory in archive")
	}

	progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Extracting from: %s", binPrefix)})

	// Extract only the bin directory contents
	for _, file := range reader.File {
		if !strings.HasPrefix(file.Name, binPrefix) {
			continue
		}

		relPath := strings.TrimPrefix(file.Name, binPrefix)
		if relPath == "" || file.FileInfo().IsDir() {
			continue
		}

		destPath := filepath.Join(destDir, relPath)

		outFile, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", destPath, err)
		}

		rc, err := file.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
		}

		_, err = io.Copy(outFile, rc)
		rc.Close()
		outFile.Close()
		if err != nil {
			return fmt.Errorf("failed to extract %s: %w", file.Name, err)
		}

		progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Extracted: %s", relPath)})
	}

	return nil
}

// extractFFmpegZipDarwinNew extracts FFmpeg/FFprobe from a ZIP archive (macOS) with progress.
func extractFFmpegZipDarwinNew(archivePath, destDir string, progress downloads.ProgressCallback) error {
	reader, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("failed to open zip archive: %w", err)
	}
	defer reader.Close()

	for _, file := range reader.File {
		if file.FileInfo().IsDir() {
			continue
		}

		fileName := filepath.Base(file.Name)
		if fileName != "ffmpeg" && fileName != "ffprobe" && fileName != "ffplay" {
			continue
		}

		destPath := filepath.Join(destDir, fileName)

		outFile, err := os.Create(destPath)
		if err != nil {
			return fmt.Errorf("failed to create %s: %w", destPath, err)
		}

		rc, err := file.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("failed to open %s in archive: %w", file.Name, err)
		}

		_, err = io.Copy(outFile, rc)
		rc.Close()
		outFile.Close()
		if err != nil {
			return fmt.Errorf("failed to extract %s: %w", file.Name, err)
		}

		// Make executable
		os.Chmod(destPath, 0755)

		progress(downloads.Progress{Status: downloads.StatusExtracting, Message: fmt.Sprintf("Extracted: %s", fileName)})
	}

	return nil
}

// extractFFmpegTarXzNew extracts FFmpeg from a tar.xz archive (Linux) with progress.
func extractFFmpegTarXzNew(archivePath, destDir string, progress downloads.ProgressCallback) error {
	progress(downloads.Progress{Status: downloads.StatusExtracting, Message: "Extracting tar.xz archive..."})

	cmd := exec.Command("tar", "-xf", archivePath, "-C", destDir, "--strip-components=2", "--wildcards", "*/bin/*")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("tar extraction failed: %w (output: %s)", err, string(output))
	}

	// Make executables executable
	executables := []string{"ffmpeg", "ffprobe", "ffplay"}
	for _, exe := range executables {
		exePath := filepath.Join(destDir, exe)
		os.Chmod(exePath, 0755)
	}

	return nil
}
