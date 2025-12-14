package tasks

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/embedexec"
	"github.com/stevecastle/shrike/jobqueue"
)

// ingestGalleryTask downloads media using gallery-dl and adds to database
// Each line of stdout from gallery-dl is the path of a written file
// This is the legacy entry point; prefer ingestGalleryTaskWithOptions for new code.
func ingestGalleryTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	return ingestGalleryTaskWithOptions(j, q, mu, IngestOptions{})
}

// ingestGalleryTaskWithOptions downloads media using gallery-dl and adds to database
// It supports optional follow-up tasks based on the provided IngestOptions.
func ingestGalleryTaskWithOptions(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex, opts IngestOptions) error {
	ctx := j.Ctx
	url := strings.TrimSpace(j.Input)

	if err := ensureMediaTableSchema(q.Db); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error setting up database schema: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Get download path from config
	cfg := appconfig.Get()
	downloadPath := cfg.DownloadPath
	if downloadPath == "" {
		downloadPath = filepath.Join(os.Getenv("USERPROFILE"), "media")
	}

	// Ensure download directory exists
	if err := os.MkdirAll(downloadPath, 0755); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error creating download directory: %v", err))
		q.ErrorJob(j.ID)
		return err
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Starting gallery-dl download: %s", url))
	q.PushJobStdout(j.ID, fmt.Sprintf("Output directory: %s", downloadPath))

	// Build gallery-dl arguments
	// -d sets the base download directory
	// --write-log outputs the downloaded file paths
	args := []string{
		"-d", downloadPath,
		url,
	}

	// Add any extra arguments from job
	for _, arg := range j.Arguments {
		args = append(args, arg)
	}

	cmd, cleanup, err := embedexec.GetExec(ctx, "gallery-dl", args...)
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error starting gallery-dl: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("start gallery-dl: %w", err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	// Handle cancellation
	go func() {
		<-ctx.Done()
		if cmd.Process != nil {
			if runtime.GOOS == "windows" {
				_ = exec.Command("taskkill", "/F", "/T", "/PID", fmt.Sprintf("%d", cmd.Process.Pid)).Run()
			} else {
				_ = cmd.Process.Kill()
			}
		}
	}()

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stdout pipe: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("stdout pipe: %w", err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error getting stderr pipe: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("Error starting command: %s", err))
		q.ErrorJob(j.ID)
		return fmt.Errorf("start: %w", err)
	}

	var downloadedFiles []string
	doneReading := make(chan struct{})
	totalReaders := 2
	doneCount := 0

	// Read stdout - gallery-dl outputs file paths line by line
	go func() {
		s := bufio.NewScanner(stdoutPipe)
		for s.Scan() {
			line := strings.TrimSpace(s.Text())
			q.PushJobStdout(j.ID, line)

			// gallery-dl outputs the file path for each downloaded file
			// Check if the line looks like a valid file path
			if line != "" && isDownloadedFilePath(line, downloadPath) {
				mu.Lock()
				downloadedFiles = append(downloadedFiles, line)
				mu.Unlock()
			}
		}
		mu.Lock()
		doneCount++
		if doneCount == totalReaders {
			close(doneReading)
		}
		mu.Unlock()
	}()

	// Read stderr
	go func() {
		s := bufio.NewScanner(stderrPipe)
		for s.Scan() {
			q.PushJobStdout(j.ID, s.Text())
		}
		mu.Lock()
		doneCount++
		if doneCount == totalReaders {
			close(doneReading)
		}
		mu.Unlock()
	}()

	err = cmd.Wait()
	<-doneReading

	select {
	case <-ctx.Done():
		q.PushJobStdout(j.ID, "Task was canceled")
		_ = q.CancelJob(j.ID)
		return ctx.Err()
	default:
	}

	if err != nil {
		q.PushJobStdout(j.ID, fmt.Sprintf("gallery-dl error: %s", err))
		q.ErrorJob(j.ID)
		return err
	}

	// Add downloaded files to database
	var insertedFiles []string
	for _, filePath := range downloadedFiles {
		// Verify the file exists and get its size
		fi, statErr := os.Stat(filePath)
		if statErr != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: file not found %s: %v", filePath, statErr))
			continue
		}
		size := fi.Size()

		if err := insertMediaRecord(q.Db, filePath, size); err != nil {
			q.PushJobStdout(j.ID, fmt.Sprintf("Warning: failed to insert %s: %v", filePath, err))
			continue
		}
		insertedFiles = append(insertedFiles, filePath)
		q.PushJobStdout(j.ID, fmt.Sprintf("Added to database: %s", filePath))
	}

	q.PushJobStdout(j.ID, fmt.Sprintf("Download completed: %d files added to database", len(insertedFiles)))

	// Queue follow-up tasks for each inserted file
	queueFollowUpTasks(q, j.ID, insertedFiles, opts)

	q.CompleteJob(j.ID)
	return nil
}

// isDownloadedFilePath checks if a line looks like a downloaded file path
// gallery-dl outputs the path of each written file to stdout
func isDownloadedFilePath(line string, downloadPath string) bool {
	// Check if it's an absolute path or starts with download path
	if filepath.IsAbs(line) {
		// Verify it's a media file
		if isMediaFile(line) {
			return true
		}
		// Also accept other file types that gallery-dl might download
		ext := strings.ToLower(filepath.Ext(line))
		switch ext {
		case ".json", ".txt", ".html":
			// Skip metadata files
			return false
		}
		// Accept if it exists in download path
		if strings.HasPrefix(filepath.Clean(line), filepath.Clean(downloadPath)) {
			return true
		}
	}

	// Check if line contains the download path
	if strings.Contains(line, downloadPath) {
		// Extract the path portion if it's a log line
		// gallery-dl format might be like: "# path/to/file.jpg" or just "path/to/file.jpg"
		cleanLine := strings.TrimPrefix(line, "# ")
		if isMediaFile(cleanLine) {
			return true
		}
	}

	return false
}
